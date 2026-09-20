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

def chunk_story(story: str, max_words: int = 400) -> list[str]:
    """Groups paragraphs into chunks small enough for one model call."""
    chunks, current, count = [], [], 0

    for paragraph in [p for p in story.split("\n") if p.strip()]:
        words = len(paragraph.split())
        if count + words > max_words and current:
            chunks.append("\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words

    if current:
        chunks.append("\n".join(current))
    return chunks


def split_long_story(
    story: str, system_prompt: str = DEFAULT_SYSTEM
) -> list[Scene]:
    """Splits a story of any length, one model call per chunk."""
    scenes = []
    for chunk in chunk_story(story):
        scenes.extend(split_story(chunk, system_prompt))
    return scenes