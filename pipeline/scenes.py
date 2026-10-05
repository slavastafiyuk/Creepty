"""Scene identity and asset invalidation, independent of the UI and models."""
import uuid
from pathlib import Path

from pydantic import BaseModel, Field


class Scene(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex, min_length=1)
    revision: int = Field(default=0, ge=0)
    text: str
    image_prompt: str
    mood: str
    image_path: str | None = None
    audio_path: str | None = None
    audio_duration: float | None = Field(default=None, gt=0, allow_inf_nan=False)


def edit_scene(scene: Scene, *, text: str, image_prompt: str, mood: str) -> Scene:
    changes = {"text": text, "image_prompt": image_prompt, "mood": mood}
    if all(getattr(scene, key) == value for key, value in changes.items()):
        return scene
    changes["revision"] = scene.revision + 1
    # Text and mood contribute to both the image prompt and voice instruction.
    if text != scene.text or mood != scene.mood:
        changes.update(image_path=None, audio_path=None, audio_duration=None)
    elif image_prompt != scene.image_prompt:
        changes["image_path"] = None
    return scene.model_copy(update=changes)


def has_asset(scene: Scene, kind: str) -> bool:
    path = getattr(scene, f"{kind}_path")
    try:
        return bool(path and Path(path).is_file() and Path(path).stat().st_size)
    except OSError:
        return False
