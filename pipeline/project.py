"""Versioned, atomic project manifests; paths are relative where possible."""
import json
import os
import tempfile
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from PySide6.QtGui import QImageReader

from pipeline.audio_validation import inspect_audio
from pipeline.scenes import Scene


# noinspection PyDataclass
class Project(BaseModel):
    version: Literal[1] = 1
    run_id: str
    story: str = ""
    system_prompt: str = ""
    scenes: list[Scene] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_scenes_and_safe_run(self):
        if not self.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in self.run_id):
            raise ValueError("Invalid project run identifier.")
        if len({scene.id for scene in self.scenes}) != len(self.scenes):
            raise ValueError("Project contains duplicate scene identifiers.")
        return self


def save_project(path: Path, project: Project) -> None:
    path = Path(path).resolve()
    data = project.model_dump()
    for scene in data["scenes"]:
        for key in ("image_path", "audio_path"):
            if scene[key]:
                asset = Path(scene[key]).resolve()
                try:
                    scene[key] = os.path.relpath(asset, path.parent)
                except ValueError:  # Assets on a different Windows drive.
                    scene[key] = str(asset)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_project(path: Path) -> tuple[Project, list[str]]:
    path = Path(path).resolve()
    project = Project.model_validate_json(path.read_text(encoding="utf-8"))
    warnings = []
    for index, scene in enumerate(project.scenes, 1):
        scene.audio_duration = None  # Recompute from the actual media.
        for kind in ("image", "audio"):
            key = f"{kind}_path"
            value = getattr(scene, key)
            if not value:
                continue
            asset = (path.parent / value).resolve()
            try:
                if kind == "audio":
                    scene.audio_duration = inspect_audio(asset)
                elif QImageReader(str(asset)).read().isNull():
                    raise ValueError("Image is missing or unreadable.")
                setattr(scene, key, str(asset))
            except (OSError, ValueError, RuntimeError) as error:
                setattr(scene, key, None)
                warnings.append(f"Scene {index} {kind}: {error}")
    return project, warnings
