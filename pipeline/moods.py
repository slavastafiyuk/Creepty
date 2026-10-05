"""Shared vocabulary for scene planning and narrator delivery."""
import re
from typing import Literal

Mood = Literal["mysterious", "fear", "dread", "passion", "tense", "calm"]
MOODS = ("mysterious", "fear", "dread", "passion", "tense", "calm")
ALIASES = {
    "mystery": "mysterious", "secretive": "mysterious",
    "fearful": "fear", "frightened": "fear", "scared": "fear", "terrified": "fear",
    "ominous": "dread", "foreboding": "dread", "horror": "dread",
    "passionate": "passion", "intense": "passion", "emotional": "passion",
    "suspenseful": "tense", "suspense": "tense", "tension": "tense", "anxious": "tense",
    "peaceful": "calm", "neutral": "calm",
}


def normalize_mood(value: str) -> Mood:
    # Free-form historical moods use the first recognized word, deterministically.
    for word in re.findall(r"[a-z]+", (value or "").lower()):
        if word in MOODS:
            return word
        if word in ALIASES:
            return ALIASES[word]
    return "calm"
