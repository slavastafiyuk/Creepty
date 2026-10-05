"""Generates narration locally through Qwen3-TTS CustomVoice.

Runs on CUDA in BF16. The caller releases ComfyUI before narration.

The same predefined speaker is used for every scene, keeping one narrator
throughout the entire video. Mood instructions change delivery/emotion
without changing narrator identity.
"""

import functools
import gc
import json
import re
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from num2words import num2words
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

from pipeline.audio_validation import InvalidAudioError, inspect_audio, validate_audio
from pipeline.moods import normalize_mood


OUTPUT_DIR = Path(__file__).parent.parent / "output"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"

DEVICE = "cuda:0"
DTYPE = torch.bfloat16
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
_NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?(?!\w)")
_TIME_RE = re.compile(r"(?<![\w:])([01]?\d|2[0-3]):([0-5]\d)(?![\w:])")


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

def _complete_snapshot(path: str) -> bool:
    """Check required files, including every shard, before using offline cache."""
    root = Path(path)
    required = (
        "config.json", "generation_config.json", "preprocessor_config.json",
        "tokenizer_config.json", "vocab.json", "merges.txt",
        "speech_tokenizer/config.json", "speech_tokenizer/preprocessor_config.json",
    )
    if not all((root / name).is_file() for name in required):
        return False
    for directory in (root, root / "speech_tokenizer"):
        if (directory / "model.safetensors").is_file():
            continue
        index = directory / "model.safetensors.index.json"
        if not index.is_file():
            return False
        shards = json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()
        if not shards or not all((directory / shard).is_file() for shard in shards):
            return False
    return True


@functools.cache
def _model() -> Qwen3TTSModel:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Narration requires CUDA. Run setup.bat to install the GPU dependencies."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The narration model requires a GPU with BF16 support.")

    # A local snapshot avoids tokenizer metadata requests on every load.
    if Path(MODEL_NAME).is_dir():
        source = MODEL_NAME
    else:
        try:
            source = snapshot_download(MODEL_NAME, local_files_only=True)
            if not _complete_snapshot(source):
                source = snapshot_download(MODEL_NAME)
        except LocalEntryNotFoundError:
            source = snapshot_download(MODEL_NAME)

    model = Qwen3TTSModel.from_pretrained(
        source,
        device_map=DEVICE,
        dtype=DTYPE,
        attn_implementation="sdpa",
    )

    # Keep the main model in BF16, but decode the final waveform in FP32.
    model.model.speech_tokenizer.model.decoder.to(
        device=DEVICE,
        dtype=torch.float32,
    )

    print("Qwen narrator:", SPEAKER)
    print("Supported speakers:", model.get_supported_speakers())

    return model


def unload_model() -> None:
    """Release the cached narrator after a batch, including allocator memory."""
    _model.cache_clear()
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Style instruction
# ---------------------------------------------------------------------------

def _build_instruction(mood: str) -> str:
    key = normalize_mood(mood)

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

    if "." in digits:
        whole, fraction = digits.split(".", 1)
        return num2words(int(whole)) + " point " + " ".join(num2words(int(d)) for d in fraction)

    value = int(digits)

    if len(digits) == 4 and value in _YEAR_RANGE:
        return num2words(value, to="year")

    return num2words(value)


def _normalize(text: str) -> str:
    # Preserve numerical ranges before converting narrative dashes to pauses.
    text = re.sub(r"(?<!\w)(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)(?!\w)",
                  lambda match: f"{match[1]} to {match[2]}", text)
    text = text.translate(_NORMALIZE)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)
    def time_words(match):
        hour, minute = map(int, match.groups())
        if minute == 0:
            return num2words(hour) + " o'clock"
        return num2words(hour) + (" oh " if minute < 10 else " ") + num2words(minute)
    text = _TIME_RE.sub(time_words, text)
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
    *, cancelled=None,
) -> str:

    text = _normalize(text.strip())

    if not text:
        raise ValueError("Empty text.")

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError("Narration cancelled.")

    # Keep whole-scene prosody. Retry once only when signal validation fails.
    for attempt in range(2):
        check_cancelled()
        audio, sample_rate = _generate(text, mood)
        check_cancelled()
        try:
            validate_audio(audio, sample_rate)
        except InvalidAudioError as error:
            if attempt == 0:
                continue
            raise RuntimeError(f"Narration failed audio validation after two attempts: {error}") from error
        break

    path = OUTPUT_DIR / f"{name}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".wav", delete=False) as stream:
            temporary = Path(stream.name)
        sf.write(str(temporary), audio, sample_rate, subtype="PCM_16")
        inspect_audio(temporary)  # Validate the encoded file before publishing it.
        check_cancelled()
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(path)
