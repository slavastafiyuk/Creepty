"""Splits a story into scenes using a local LLM through Ollama.

The model never reproduces the story: it returns sentence-number ranges and
Python slices the original text. Context size, batching and retries adapt to
whatever model and hardware are available, so no constant needs editing.
"""

import functools
import re
from collections.abc import Iterator

import httpx
import ollama
from pydantic import BaseModel, ValidationError, field_validator

from pipeline.moods import Mood, normalize_mood
from pipeline.scenes import Scene

MODEL = "gemma4:e4b"
KEEP_ALIVE = "5m"
# A hung local service must not keep cancellation/shutdown pending indefinitely.
client = ollama.Client(timeout=httpx.Timeout(180.0, connect=5.0))
cleanup_client = ollama.Client(timeout=httpx.Timeout(30.0, connect=5.0))
MAX_BATCH_ATTEMPTS = 3


class InvalidSceneResponse(ValueError):
    pass
_model_in_use = False

TOKENS_PER_WORD = 1.6    # rough estimate, includes the numbering overhead
CTX_FLOOR = 4096         # never go below this, even on tiny models
CTX_CAP = 32768          # past this, batching beats a huge context window
MAX_SCENE_SENTENCES = 5  # matches "usually 2 to 5 sentences"; any boundary
                          # the model returns larger than this gets split
                          # further so narration text stays TTS-sized


DEFAULT_SYSTEM = (
    "You split horror stories into scenes for a narrated video. "
    "You receive the story as numbered sentences. "
    "Return scenes as ranges of sentence numbers, covering every sentence "
    "in order with no gaps and no overlaps. "
    "A scene is one continuous visual moment, usually 2 to 5 sentences. "
    "Prefer keeping sentences from the same paragraph together. "
    "Treat a paragraph break as a strong signal to start a new scene, "
    "especially when it marks a new thought, moment or narrative beat. "
    "Start a new scene when the place, the time, the subject or the "
    "narrative moment changes. "
    "Do not split a paragraph unnecessarily just to reach a certain "
    "number of sentences. "
    "Write image_prompt and mood in English, whatever the story language. "
    "Choose exactly one mood: mysterious, fear, dread, passion, tense, calm."
)


class Boundary(BaseModel):
    start: int          # first sentence number of the scene, 1-based
    end: int            # last sentence number, inclusive
    image_prompt: str   # visual description for the image model
    mood: Mood

    @field_validator("mood", mode="before")
    @classmethod
    def canonical_mood(cls, value):
        return normalize_mood(value) if isinstance(value, str) else value


class Boundaries(BaseModel):
    scenes: list[Boundary]


@functools.cache
def model_limits(model: str = MODEL) -> tuple[int, int]:
    """Reads the model's real context length and reserves room for the answer.

    Cached: this hits Ollama once per process, not once per call.
    """
    try:
        info = client.show(model).modelinfo or {}
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

def _cap_length(start: int, end: int) -> list[tuple[int, int]]:
    """Breaks a sentence range into pieces of at most MAX_SCENE_SENTENCES."""
    pieces = []
    cursor = start
    while cursor <= end:
        piece_end = min(cursor + MAX_SCENE_SENTENCES - 1, end)
        pieces.append((cursor, piece_end))
        cursor = piece_end + 1
    return pieces


def repair(raw: list[Boundary], total: int) -> list[Boundary]:
    """Forces the model's ranges into a clean partition of the sentences.

    Models drop, repeat or overshoot indices. Clamping is cheaper and more
    predictable than retrying the call. Oversized scenes (the model
    ignoring "usually 2 to 5 sentences") are also split further here, since
    a scene that's too long produces narration too long for reliable TTS.
    """
    fixed, cursor = [], 1
    for scene in sorted(raw, key=lambda s: s.start):
        if cursor > total:
            break
        start = cursor                       # always continue where we left off
        end = min(max(scene.end, start), total)
        fixed.extend(
            Boundary(start=s, end=e, image_prompt=scene.image_prompt, mood=scene.mood)
            for s, e in _cap_length(start, end)
        )
        cursor = end + 1
    if cursor <= total:
        # absorb anything the model forgot, in scene-sized pieces too
        image_prompt = fixed[-1].image_prompt if fixed else ""
        mood = fixed[-1].mood if fixed else ""
        fixed.extend(
            Boundary(start=s, end=e, image_prompt=image_prompt, mood=mood)
            for s, e in _cap_length(cursor, total)
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

    global _model_in_use
    _model_in_use = True
    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": numbered},
        ],
        format=Boundaries.model_json_schema(),
        options={"temperature": 0.2, "num_ctx": ctx, "num_predict": 4096},
        keep_alive=KEEP_ALIVE,
    )

    if getattr(response, "done_reason", None) == "length":
        raise InvalidSceneResponse("Scene response was truncated by the output limit.")
    try:
        raw = Boundaries.model_validate_json(response.message.content).scenes
    except ValidationError as error:
        raise InvalidSceneResponse("Scene model returned invalid JSON or scene fields.") from error
    if not raw or any(not scene.image_prompt.strip() for scene in raw):
        raise InvalidSceneResponse("Scene response contains no usable visual prompts.")

    return [_scene_from(sentences, b) for b in repair(raw, len(sentences))]


def _is_capacity_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        word in text
        for word in ("memory", "context", "oom", "allocate", "too large")
    )


def unload_model() -> None:
    """Release only a model used by this process, before the next GPU phase."""
    global _model_in_use
    if _model_in_use:
        cleanup_client.generate(model=MODEL, prompt="", keep_alive=0)
        _model_in_use = False


def stream_story(
    story: str, system_prompt: str = DEFAULT_SYSTEM, *, cancelled=None
) -> Iterator[tuple[int, int, list[Scene]]]:
    """Keep the model resident across batches; retry only unfinished text.

    Context stays fixed unless a capacity error requires a smaller window.
    Closing the iterator also unloads the model, including on cancellation.
    """
    sentences = split_sentences(story)
    if not sentences:
        raise ValueError("Empty story.")

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError("Scene planning cancelled.")

    check_cancelled()
    ceiling, reserved = model_limits()
    budget = ceiling - reserved
    ctx = min(ceiling, max(CTX_FLOOR, estimate_tokens(sentences) + reserved))
    offset = completed = failures = 0
    sentence_limit = len(sentences)
    primary_error = None

    try:
        while offset < len(sentences):
            check_cancelled()
            batches = [part[i:i + sentence_limit]
                       for part in batch_sentences(sentences[offset:], budget)
                       for i in range(0, len(part), sentence_limit)]
            total = completed + len(batches)
            for batch in batches:
                check_cancelled()
                try:
                    scenes = _call(batch, system_prompt, ctx)
                    check_cancelled()
                except Exception as error:
                    check_cancelled()
                    failures += 1
                    capacity = _is_capacity_error(error)
                    invalid = isinstance(error, InvalidSceneResponse)
                    if failures >= MAX_BATCH_ATTEMPTS or not (capacity or invalid):
                        raise
                    if capacity:
                        if budget <= CTX_FLOOR // 2:
                            raise
                        budget //= 2
                        ctx = max(CTX_FLOOR, min(ctx, budget + reserved))
                    if invalid:
                        sentence_limit = max(1, len(batch) // 2)
                    break
                failures = 0
                offset += len(batch)
                completed += 1
                yield completed, total, scenes
            else:
                return
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            unload_model()
        except Exception as error:
            if primary_error is None:
                raise RuntimeError(
                    "Couldn't unload the scene model before the next GPU phase."
                ) from error
            primary_error.add_note(f"Scene model cleanup also failed: {error}")


def split_story(story: str, system_prompt: str = DEFAULT_SYSTEM) -> list[Scene]:
    """Blocking version: returns every scene at once."""
    scenes = []
    for _, _, batch in stream_story(story, system_prompt):
        scenes.extend(batch)
    return scenes
