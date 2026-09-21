"""Splits a story into scenes using a local LLM through Ollama.

The model never reproduces the story: it returns sentence-number ranges and
Python slices the original text. Context size, batching and retries adapt to
whatever model and hardware are available, so no constant needs editing.
"""

import functools
import re
from collections.abc import Iterator

import ollama
from pydantic import BaseModel

MODEL = "gemma4:e4b"

TOKENS_PER_WORD = 1.6    # rough estimate, includes the numbering overhead
CTX_FLOOR = 4096         # never go below this, even on tiny models
CTX_CAP = 32768          # past this, batching beats a huge context window

DEFAULT_SYSTEM = (
    "You split horror stories into scenes for a narrated video. "
    "You receive the story as numbered sentences. "
    "Return scenes as ranges of sentence numbers, covering every sentence "
    "in order with no gaps and no overlaps. "
    "A scene is one continuous visual moment, usually 2 to 5 sentences. "
    "Start a new scene when the place, the time or the subject changes. "
    "Write image_prompt and mood in English, whatever the story language."
)


class Boundary(BaseModel):
    start: int          # first sentence number of the scene, 1-based
    end: int            # last sentence number, inclusive
    image_prompt: str   # visual description for the image model
    mood: str           # atmosphere, drives the sound design


class Boundaries(BaseModel):
    scenes: list[Boundary]


class Scene(BaseModel):
    text: str
    image_prompt: str
    mood: str
    image_path: str | None = None   # filled in by the image step
    audio_path: str | None = None


@functools.cache
def model_limits(model: str = MODEL) -> tuple[int, int]:
    """Reads the model's real context length and reserves room for the answer.

    Cached: this hits Ollama once per process, not once per call.
    """
    try:
        info = ollama.show(model).modelinfo or {}
        length = next(
            (v for k, v in info.items() if k.endswith(".context_length")),
            CTX_FLOOR,
        )
    except Exception:
        length = CTX_FLOOR  # Ollama unreachable or older API, stay conservative

    ceiling = max(CTX_FLOOR, min(int(length), CTX_CAP))
    reserved = max(1024, ceiling // 4)  # output grows with the scene count
    return ceiling, reserved


def split_sentences(text: str) -> list[str]:
    """Splits on sentence endings and line breaks, keeping punctuation."""
    text = re.sub(r"[ \t]+", " ", text)
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def estimate_tokens(sentences: list[str]) -> int:
    words = sum(len(s.split()) for s in sentences)
    return int(words * TOKENS_PER_WORD)


def batch_sentences(sentences: list[str], budget: int) -> list[list[str]]:
    """Groups sentences into batches that fit the token budget.

    Splitting only ever happens between sentences, so no sentence is ever cut.
    A single sentence longer than the budget still gets its own batch.
    """
    batches, current, count = [], [], 0

    for sentence in sentences:
        tokens = int(len(sentence.split()) * TOKENS_PER_WORD)
        if count + tokens > budget and current:
            batches.append(current)
            current, count = [], 0
        current.append(sentence)
        count += tokens

    if current:
        batches.append(current)
    return batches


def repair(raw: list[Boundary], total: int) -> list[Boundary]:
    """Forces the model's ranges into a clean partition of the sentences.

    Models drop, repeat or overshoot indices. Clamping is cheaper and more
    predictable than retrying the call.
    """
    fixed, cursor = [], 1

    for scene in sorted(raw, key=lambda s: s.start):
        if cursor > total:
            break
        start = cursor                       # always continue where we left off
        end = min(max(scene.end, start), total)
        fixed.append(
            Boundary(
                start=start,
                end=end,
                image_prompt=scene.image_prompt,
                mood=scene.mood,
            )
        )
        cursor = end + 1

    if cursor <= total and fixed:            # absorb anything the model forgot
        last = fixed[-1]
        fixed[-1] = Boundary(
            start=last.start,
            end=total,
            image_prompt=last.image_prompt,
            mood=last.mood,
        )
    return fixed


def _scene_from(sentences: list[str], b: Boundary) -> Scene:
    return Scene(
        text=" ".join(sentences[b.start - 1 : b.end]),
        image_prompt=b.image_prompt,
        mood=b.mood,
    )


def _call(sentences: list[str], system_prompt: str, ctx: int) -> list[Scene]:
    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, start=1))

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": numbered},
        ],
        format=Boundaries.model_json_schema(),
        options={"temperature": 0.2, "num_ctx": ctx, "num_predict": 4096},
    )

    raw = Boundaries.model_validate_json(response.message.content).scenes

    if not raw:  # fall back to one scene rather than losing the text
        raw = [Boundary(start=1, end=len(sentences), image_prompt="", mood="")]

    return [_scene_from(sentences, b) for b in repair(raw, len(sentences))]


def _is_capacity_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        word in text
        for word in ("memory", "context", "oom", "allocate", "too large")
    )


def stream_story(
    story: str, system_prompt: str = DEFAULT_SYSTEM
) -> Iterator[tuple[int, int, list[Scene]]]:
    """Yields (batch index, total batches, scenes) as the story is processed.

    Short stories produce a single batch, so the caller needs no special case.
    If the model or the GPU can't take the context, the budget halves and the
    story is re-batched instead of failing.
    """
    sentences = split_sentences(story)
    if not sentences:
        raise ValueError("Empty story.")

    ceiling, reserved = model_limits()
    budget = ceiling - reserved

    while True:
        ctx = min(
            ceiling,
            max(CTX_FLOOR, estimate_tokens(sentences) + reserved),
        )
        batches = batch_sentences(sentences, budget)

        try:
            for index, batch in enumerate(batches, start=1):
                yield index, len(batches), _call(batch, system_prompt, ctx)
            return
        except Exception as error:
            # Out of memory or context overflow: shrink and start over.
            if budget <= CTX_FLOOR // 2 or not _is_capacity_error(error):
                raise
            budget //= 2
            ceiling = min(ceiling, budget + reserved)


def split_story(story: str, system_prompt: str = DEFAULT_SYSTEM) -> list[Scene]:
    """Blocking version: returns every scene at once."""
    scenes = []
    for _, _, batch in stream_story(story, system_prompt):
        scenes.extend(batch)
    return scenes
