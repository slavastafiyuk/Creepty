"""Splits a story into scenes using a local LLM through Ollama."""

import ollama
from pydantic import BaseModel

MODEL = "gemma4:e4b"

DEFAULT_SYSTEM = (
    "You split horror stories into scenes for a narrated short video. "
    "A scene is one continuous visual moment, 2 to 4 sentences long. "
    "Copy the original wording exactly, never rewrite or summarise it. "
    "Write image_prompt and mood in English, whatever the story language."
)


class Scene(BaseModel):
    text: str           # verbatim slice of the story, used for narration
    image_prompt: str   # visual description for the image model
    mood: str           # atmosphere, drives the sound design


class Scenes(BaseModel):
    scenes: list[Scene]


def split_story(story: str, system_prompt: str = DEFAULT_SYSTEM) -> list[Scene]:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": story},
        ],
        format=Scenes.model_json_schema(),  # constrains output to the schema
        options={"temperature": 0.2, "num_ctx": 8192},
    )
    scenes = Scenes.model_validate_json(response.message.content).scenes
    if not scenes:
        raise ValueError("The model returned no scenes.")
    return scenes
