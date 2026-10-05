"""Signal checks shared by generation and project recovery.

These detect empty, corrupt, silent and nearly flat output; they do not measure
pronunciation or replace listening. Thresholds allow quiet narration.
"""
import numpy as np
import soundfile as sf

MIN_DURATION = 0.1
MIN_SIGNAL_RMS = 1e-5
MIN_PEAK = 2 / 32768


class InvalidAudioError(ValueError):
    pass


def validate_audio(audio: np.ndarray, sample_rate: int) -> float:
    audio = np.asarray(audio)
    if not isinstance(sample_rate, (int, np.integer)) or sample_rate <= 0:
        raise InvalidAudioError("Invalid audio sample rate.")
    if audio.ndim not in (1, 2) or audio.size == 0 or (audio.ndim == 2 and audio.shape[1] not in (1, 2)):
        raise InvalidAudioError("No usable audio samples were generated.")
    if not np.isfinite(audio).all():
        raise InvalidAudioError("Audio contains non-finite samples.")
    duration = len(audio) / sample_rate
    if duration < MIN_DURATION:
        raise InvalidAudioError("Generated audio is too short (less than 0.1 seconds).")
    if np.max(np.abs(audio)) < MIN_PEAK or np.max(np.std(audio, axis=0, dtype=np.float64)) < MIN_SIGNAL_RMS:
        raise InvalidAudioError("Generated audio is silent or has no meaningful signal.")
    if np.max(np.abs(audio)) > 1:
        raise InvalidAudioError("Generated audio exceeds the PCM amplitude range.")
    return duration


def inspect_audio(path) -> float:
    audio, sample_rate = sf.read(str(path), dtype="float32")
    return validate_audio(audio, sample_rate)
