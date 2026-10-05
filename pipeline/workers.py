"""GPU pipeline jobs and immutable scene snapshots used by the desktop UI."""
import time
import uuid
from contextlib import closing
from datetime import datetime

import soundfile as sf
from PySide6.QtCore import QObject, Signal, Slot

from pipeline import comfy_server, images, voice, video
from pipeline.segmenter import Scene, stream_story, unload_model as unload_scenes

# ---------- shared pipeline helpers ----------

def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def build_image_prompt(scene: Scene) -> str:
    """Structured prompt: the Qwen3 text encoder reads labelled fields."""
    lines = [
        f"context: {scene.text.strip()}",
        f"image: {scene.image_prompt.strip()}",
    ]
    if scene.mood.strip():
        lines.append(f"mood: {scene.mood.strip()}")
    return "\n".join(lines)


def asset_name(run_id: str, row: int) -> str:
    # The suffix keeps every render unique, so a moved or regenerated scene
    # never points at a file that belongs to another scene.
    return f"{run_id}/scene_{row + 1:03d}_{uuid.uuid4().hex[:6]}"


def can_render_image(scene: Scene) -> bool:
    return bool(scene.text.strip() and scene.image_prompt.strip())


def can_render_voice(scene: Scene) -> bool:
    return bool(scene.text.strip())


# ---------- workers ----------

class AssetWorker(QObject):
    """Render a batch of images, then a batch of CUDA narration."""

    progress = Signal(str)
    image_progress = Signal(str)
    voice_progress = Signal(str)
    asset_done = Signal(str, int, str, str, float, float)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(
        self, image_jobs: list[tuple[int, Scene]],
        voice_jobs: list[tuple[int, Scene]], run_id: str,
    ):
        super().__init__()
        self.image_jobs = [(row, scene.model_copy(deep=True)) for row, scene in image_jobs]
        self.voice_jobs = [(row, scene.model_copy(deep=True)) for row, scene in voice_jobs]
        self.run_id = run_id
        self.cancelled = False

    def cancel(self):
        # Observed by the worker; never block the UI on an HTTP request.
        self.cancelled = True

    def prepare(self):
        voice.unload_model()
        unload_scenes()  # retry cleanup if an earlier Ollama unload failed

    def _run_images(self):
        total = len(self.image_jobs)
        for index, (row, scene) in enumerate(self.image_jobs, start=1):
            if self.cancelled:
                return
            self.image_progress.emit(f"{index} of {total} (scene {row + 1})")
            start = time.monotonic()
            path = images.generate_image(
                build_image_prompt(scene), asset_name(self.run_id, row),
                cancelled=lambda: self.cancelled,
            )
            self.asset_done.emit(scene.id, scene.revision, "image", path, time.monotonic() - start, 0.0)
        if total:
            self.image_progress.emit("done")

    def _run_voices(self):
        total = len(self.voice_jobs)
        for index, (row, scene) in enumerate(self.voice_jobs, start=1):
            if self.cancelled:
                return
            self.voice_progress.emit(f"{index} of {total} (scene {row + 1})")
            start = time.monotonic()
            path = voice.generate_voice(
                scene.text, asset_name(self.run_id, row), mood=scene.mood,
                cancelled=lambda: self.cancelled,
            )
            self.asset_done.emit(scene.id, scene.revision, "audio", path, time.monotonic() - start, sf.info(path).duration)
        if total:
            self.voice_progress.emit("done")

    def generate_assets(self):
        if self.cancelled:
            return
        if self.image_jobs:
            self.progress.emit("Starting ComfyUI...")
            if not comfy_server.start():
                raise RuntimeError(
                    f"Couldn't start ComfyUI in {comfy_server.COMFY_DIR}."
                )
            self._run_images()
        if self.voice_jobs and not self.cancelled:
            self.progress.emit("Preparing GPU for narration...")
            comfy_server.release_vram()
            self._run_voices()

    @Slot()
    def run(self):
        failure = None
        try:
            if not self.cancelled:
                self.prepare()
                self.generate_assets()
        except InterruptedError as error:
            if not self.cancelled:
                failure = str(error)
        except Exception as error:
            failure = str(error)
        finally:
            # Clear exception tracebacks before releasing cached CUDA tensors.
            cleanups = [voice.unload_model, unload_scenes]
            if failure or self.cancelled:
                cleanups.append(comfy_server.stop)
            for cleanup in cleanups:
                try:
                    cleanup()
                except Exception as error:
                    failure = f"{failure + '; ' if failure else ''}GPU cleanup: {error}"
        if failure:
            self.failed.emit(f"Failed: {failure}")
        else:
            self.finished.emit("Cancelled." if self.cancelled else "Done.")


class Generator(AssetWorker):
    """Plan scenes, then use the same GPU phases as individual regeneration."""

    scenes_ready = Signal(int, list)

    def __init__(self, story: str, prompt: str, run_id: str):
        super().__init__([], [], run_id)
        self.story = story
        self.prompt = prompt
        self.scenes: list[Scene] = []

    def prepare(self):
        super().prepare()
        self.progress.emit("Preparing GPU for scene planning...")
        comfy_server.release_vram()
        if self.cancelled:
            return
        with closing(stream_story(self.story, self.prompt, cancelled=lambda: self.cancelled)) as batches:
            for index, total, scenes in batches:
                if self.cancelled:
                    return
                self.progress.emit(f"Splitting scenes (batch {index} of {total})...")
                self.scenes.extend(scenes)
                self.scenes_ready.emit(index, scenes)
                if self.cancelled:
                    return
        self.image_jobs = [
            (row, scene) for row, scene in enumerate(self.scenes)
            if can_render_image(scene)
        ]
        self.voice_jobs = [
            (row, scene) for row, scene in enumerate(self.scenes)
            if can_render_voice(scene)
        ]


class VideoWorker(QObject):
    """Export an immutable scene snapshot as a final MP4."""

    progress = Signal(str)
    finished = Signal(str)
    failed = Signal(str)

    def __init__(self, scenes: list[Scene], output_path: str):
        super().__init__()
        self.scenes = [
            scene.model_copy(deep=True)
            for scene in scenes
        ]
        self.output_path = output_path
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    @Slot()
    def run(self):
        try:
            if self.cancelled:
                self.finished.emit("Cancelled.")
                return

            path = video.export_video(
                self.scenes,
                self.output_path,
                cancelled=lambda: self.cancelled,
                progress=self.progress.emit,
            )

        except InterruptedError as error:
            if self.cancelled:
                self.finished.emit("Cancelled.")
            else:
                self.failed.emit(f"Failed: {error}")
            return

        except Exception as error:
            self.failed.emit(f"Failed: {error}")
            return

        self.finished.emit(
            f"Video exported: {path}"
        )