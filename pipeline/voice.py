"""Generates narration locally through Qwen3-TTS CustomVoice.

Runs on CPU so ComfyUI can use the GPU simultaneously.

The same predefined speaker is used for every scene, keeping one narrator
throughout the entire video. Mood instructions change delivery/emotion
without changing narrator identity.
"""

import functools
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from num2words import num2words

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

DEVICE = "cpu"
DTYPE = torch.float32
LANGUAGE = "English"

# Fixed narrator for the entire video.
# Ryan = native English male voice.
SPEAKER = "Ryan"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

_NORMALIZE = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": ",",
    "\u2014": ",",
    "\u2026": "...",
})

_YEAR_RANGE = range(1000, 2100)
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


# ---------------------------------------------------------------------------
# Voice style
# ---------------------------------------------------------------------------

BASE_STYLE = (
    "Narrate like a professional cinematic horror storyteller. "
    "Keep the voice natural, intimate and controlled. "
    "Speak clearly with realistic pacing and natural pauses at punctuation. "
    "The recording should feel close, clean and studio quality."
)

MOOD_DESCRIPTIONS = {
    "mysterious": (
        "Speak slowly and quietly with controlled intensity, "
        "as if revealing a dangerous secret."
    ),

    "fear": (
        "Sound frightened and increasingly uneasy. "
        "Use restrained panic and tense breathing while remaining clear."
    ),

    "dread": (
        "Speak slowly with a heavy ominous feeling. "
        "Use deliberate pauses and restrained emotion."
    ),

    "passion": (
        "Speak with strong emotion and conviction, "
        "gradually becoming more intense."
    ),

    "tense": (
        "Speak slightly faster with nervous controlled energy, "
        "as if something dangerous could happen at any moment."
    ),

    "calm": (
        "Speak naturally at a steady pace with a calm, controlled delivery."
    ),
}

DEFAULT_MOOD = "calm"


# ---------------------------------------------------------------------------
# Cached model
# ---------------------------------------------------------------------------

@functools.cache
def _model() -> Qwen3TTSModel:
    model = Qwen3TTSModel.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        dtype=DTYPE,
    )

    print("Qwen narrator:", SPEAKER)
    print("Supported speakers:", model.get_supported_speakers())

    return model


# ---------------------------------------------------------------------------
# Style instruction
# ---------------------------------------------------------------------------

def _build_instruction(mood: str) -> str:
    key = (mood or "").strip().lower()

    mood_text = MOOD_DESCRIPTIONS.get(
        key,
        MOOD_DESCRIPTIONS[DEFAULT_MOOD],
    )

    return f"{BASE_STYLE} {mood_text}"


# ---------------------------------------------------------------------------
# Number normalization
# ---------------------------------------------------------------------------

def _expand_number(match: re.Match) -> str:
    digits = match.group(0).replace(",", "")

    if digits.endswith("."):
        number = digits[:-1]

        if len(number) == 4 and int(number) in _YEAR_RANGE:
            return num2words(int(number), to="year") + "."

        return num2words(int(number)) + "."

    if "." in digits:
        return num2words(float(digits))

    value = int(digits)

    if len(digits) == 4 and value in _YEAR_RANGE:
        return num2words(value, to="year")

    return num2words(value)


def _normalize(text: str) -> str:
    text = text.translate(_NORMALIZE)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    text = _NUMBER_RE.sub(_expand_number, text)

    return text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(
    text: str,
    mood: str,
) -> tuple[np.ndarray, int]:

    model = _model()
    instruction = _build_instruction(mood)

    with torch.inference_mode():
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=LANGUAGE,
            speaker=SPEAKER,
            instruct=instruction,

            # Keep some natural variation while preserving speaker identity.
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    audio = np.asarray(
        wavs[0],
        dtype=np.float32,
    )

    return audio, sample_rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_voice(
    text: str,
    name: str,
    mood: str = DEFAULT_MOOD,
) -> str:

    text = _normalize(text.strip())

    if not text:
        raise ValueError("Empty text.")

    # One scene = one generation.
    #
    # Do NOT split sentences/chunks here:
    # separate generations can create changes in prosody and audible seams.
    audio, sample_rate = _generate(
        text,
        mood,
    )

    if audio.size == 0:
        raise RuntimeError("No audio was generated.")

    path = OUTPUT_DIR / f"{name}.wav"

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sf.write(
        str(path),
        audio,
        sample_rate,
        subtype="PCM_16",
    )

    return str(path)