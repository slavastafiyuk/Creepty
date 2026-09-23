"""Generates narration audio locally through Parler-TTS.

Runs on CPU, so it never competes with ComfyUI for VRAM. Unlike a plain
TTS model, Parler-TTS is steered with a natural-language description of
how the line should be delivered, so each scene's mood (mysterious,
fearful, passionate, ...) is turned into a description and baked into
the performance itself, not just a speed multiplier.
"""

import functools
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from num2words import num2words
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

OUTPUT_DIR = Path(__file__).parent.parent / "output"

# The multilingual checkpoint was trained on named speakers, which keeps
# the narrator's voice identity consistent across scenes even as the
# mood description changes.
MODEL_NAME = "parler-tts/parler-tts-mini-multilingual-v1.1"
SPEAKER = "Daniel"   # trained speaker name; swap for another trained voice
DEVICE = "cpu" #"cuda:0" if torch.cuda.is_available() else "cpu"
DESCRIPTION_MODEL_NAME = "google/flan-t5-large"  # this checkpoint's text_encoder

# Straight-apostrophe / straight-quote normalization: Parler-TTS was
# trained mostly on plain ASCII punctuation and mispronounces curly
# quotes and other smart punctuation. Em/en dashes become a comma pause
# rather than a literal dash, which TTS models handle far more reliably.
_NORMALIZE = str.maketrans({
    "\u2018": "'", "\u2019": "'",   # ‘ ’
    "\u201c": '"', "\u201d": '"',  # “ ”
    "\u2013": ",", "\u2014": ",",  # – —
    "\u2026": "...",               # …
})

# 4-digit numbers in this range read as spoken years ("two thousand",
# "nineteen eighty-four") rather than as a plain cardinal number.
_YEAR_RANGE = range(1000, 2100)
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")

# Parler-TTS caps out at 512 prompt tokens; past that it still generates
# but the output degrades into skipped/repeated/garbled speech instead
# of erroring. Stay well under that per chunk.
MAX_PROMPT_TOKENS = 400

# Fixed seed: keeps the narrator's tone stable across separate generate_voice
# calls instead of drifting scene to scene.
SEED = 42

# The DAC audio codec this model uses runs at 86 frames/second, so audio
# tokens map to real seconds at that rate. The generation budget is sized
# to the actual chunk length instead of a fixed cap, so short lines don't
# waste time and long lines don't get cut off mid-sentence.
FRAME_RATE = 86
WORDS_PER_SECOND = 2.2       # conservative speech-rate estimate
SECONDS_BUFFER = 6           # headroom for slow/dread deliveries, pauses
ABSOLUTE_MAX_SECONDS = 90    # hard ceiling per chunk, safety net only

# Fixed part of the description: identity + recording quality.
BASE_VOICE = (
    f"{SPEAKER}'s voice is deep and masculine. "
    "The recording is very clear, high quality, very close up, "
    "and has no background noise."
)

# Mood -> delivery instructions appended to BASE_VOICE. Add more moods here
# as your stories need them; the key should match scene.mood exactly
# (case-insensitive).
MOOD_DESCRIPTIONS = {
    "mysterious": (
        "He speaks slowly and quietly, with controlled intensity, "
        "as if revealing a dangerous secret."
    ),

    "fear": (
        "He speaks quickly with an unsteady, frightened voice. "
        "His breathing becomes tense and his words occasionally catch."
    ),

    "dread": (
        "He speaks very slowly and quietly, with a heavy, ominous tone "
        "and long pauses between important phrases."
    ),

    "passion": (
        "He speaks with strong emotion and conviction, becoming more "
        "intense as he continues."
    ),

    "tense": (
        "He speaks quickly and precisely, sounding nervous and under "
        "constant pressure while trying to remain controlled."
    ),

    "calm": (
        "He speaks at a natural, steady pace in a calm and controlled voice."
    ),
}
DEFAULT_MOOD = "calm"


@functools.cache
def _model() -> PreTrainedModel:
    # Loaded once per process and kept warm; CPU inference is slow enough
    # that reloading per call would be very costly.
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
    model.eval()  # disables dropout; without this, quality is worse and
                  # non-deterministic even with a fixed seed
    return model


@functools.cache
def _prompt_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(MODEL_NAME)


@functools.cache
def _description_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(DESCRIPTION_MODEL_NAME, legacy=False)


def _build_description(mood: str) -> str:
    key = (mood or "").strip().lower()
    mood_text = MOOD_DESCRIPTIONS.get(key, MOOD_DESCRIPTIONS[DEFAULT_MOOD])
    return f"{BASE_VOICE} {mood_text}"


def _expand_number(match: re.Match) -> str:
    digits = match.group(0).replace(",", "")
    if "." in digits:
        return num2words(float(digits))
    value = int(digits)
    if len(digits) == 4 and value in _YEAR_RANGE:
        return num2words(value, to="year")
    return num2words(value)


def _normalize(text: str) -> str:
    text = text.translate(_NORMALIZE)
    # Hyphenated compounds ("near-miss") are frequently mispronounced;
    # a space reads far more reliably without changing the meaning.
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    # Bare digits ("2000") get spelled out; Parler-TTS doesn't expand
    # numbers on its own and reads them inconsistently otherwise.
    text = _NUMBER_RE.sub(_expand_number, text)
    return text


def _split_for_budget(text: str) -> list[str]:
    """Breaks text into sentence-level chunks that fit MAX_PROMPT_TOKENS.

    A rough word-count proxy for tokens is enough here: scenes are
    already short, this only kicks in for the rare oversized one.
    """
    sentences = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks, current, count = [], [], 0
    for sentence in sentences:
        tokens = len(sentence.split())
        if count + tokens > MAX_PROMPT_TOKENS and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += tokens
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def _max_new_tokens_for(word_count: int) -> int:
    """Sizes the generation budget to the actual text instead of a fixed cap."""
    estimated_seconds = word_count / WORDS_PER_SECOND + SECONDS_BUFFER
    seconds = min(estimated_seconds, ABSOLUTE_MAX_SECONDS)
    return int(seconds * FRAME_RATE)


def generate_voice(text: str, name: str, mood: str = DEFAULT_MOOD) -> str:
    """Renders text to speech and saves it as output/<name>.wav.

    mood selects the delivery style via MOOD_DESCRIPTIONS - pass the
    scene's own mood field so each scene gets its own performance.
    """
    text = _normalize(text.strip())
    if not text:
        raise ValueError("Empty text.")

    description = _build_description(mood)
    model = _model()
    desc = _description_tokenizer()(description, return_tensors="pt").to(DEVICE)

    audio_chunks = []
    for chunk in _split_for_budget(text):
        prompt = _prompt_tokenizer()(chunk, return_tensors="pt").to(DEVICE)
        target_tokens = _max_new_tokens_for(len(chunk.split()))

        torch.manual_seed(SEED)
        with torch.inference_mode():
            generation = model.generate(
                input_ids=desc.input_ids,
                attention_mask=desc.attention_mask,
                prompt_input_ids=prompt.input_ids,
                prompt_attention_mask=prompt.attention_mask,
                do_sample=True,
                temperature=0.9,
                max_new_tokens=target_tokens,
            )
        audio_chunks.append(generation.cpu().numpy().squeeze().astype(np.float32))

    audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]

    path = OUTPUT_DIR / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, model.config.sampling_rate)
    return str(path)
