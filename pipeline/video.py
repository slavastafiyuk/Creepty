"""Final MP4 video composition through FFmpeg.

Each Creepty scene becomes one video clip:

    image + narration -> scene MP4

The scene clips are then concatenated into the final video.

The first implementation intentionally keeps images static. Animation,
transitions and subtitles can be added later without changing the public API.
"""

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from pipeline.audio_validation import inspect_audio
from pipeline.scenes import Scene, has_asset


# ---------------------------------------------------------------------------
# Video configuration
# ---------------------------------------------------------------------------

WIDTH = 1080
HEIGHT = 1920
FPS = 30

VIDEO_CODEC = "libx264"
VIDEO_PRESET = "medium"
VIDEO_CRF = 18

AUDIO_CODEC = "aac"
AUDIO_BITRATE = "192k"
AUDIO_SAMPLE_RATE = 48000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class VideoExportError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# FFmpeg discovery
# ---------------------------------------------------------------------------

def find_ffmpeg() -> str:
    """Return the FFmpeg executable.

    PATH is checked first. On Windows, common WinGet locations are also
    checked so Creepty does not depend entirely on shell PATH refreshes.
    """

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")

        if local_app_data:
            local = Path(local_app_data)

            # Standard WinGet command alias.
            link = (
                local
                / "Microsoft"
                / "WinGet"
                / "Links"
                / "ffmpeg.exe"
            )

            if link.is_file():
                return str(link)

            # Fallback for installations where WinGet created the package
            # correctly but did not expose the command alias.
            packages = (
                local
                / "Microsoft"
                / "WinGet"
                / "Packages"
            )

            if packages.is_dir():
                candidates = list(
                    packages.glob(
                        "Gyan.FFmpeg_*/**/bin/ffmpeg.exe"
                    )
                )

                if candidates:
                    candidates.sort(
                        key=lambda path: path.stat().st_mtime,
                        reverse=True,
                    )
                    return str(candidates[0])

    raise VideoExportError(
        "FFmpeg could not be found. "
        "Run scripts/setup_ffmpeg.bat."
    )


def ffmpeg_available() -> bool:
    try:
        find_ffmpeg()
        return True
    except VideoExportError:
        return False


# ---------------------------------------------------------------------------
# Scene validation
# ---------------------------------------------------------------------------

def can_export_video(scenes: list[Scene]) -> bool:
    """True when every scene has both generated assets."""

    return bool(scenes) and all(
        has_asset(scene, "image")
        and has_asset(scene, "audio")
        for scene in scenes
    )


def _validate_scenes(
    scenes: list[Scene],
) -> list[tuple[Path, Path, float]]:
    """Validate all media before starting FFmpeg.

    Returns:
        (image path, audio path, measured audio duration)
        for every scene.
    """

    if not scenes:
        raise VideoExportError(
            "The project contains no scenes."
        )

    validated = []

    for index, scene in enumerate(scenes, start=1):
        if not scene.image_path:
            raise VideoExportError(
                f"Scene {index} has no image."
            )

        if not scene.audio_path:
            raise VideoExportError(
                f"Scene {index} has no narration."
            )

        image = Path(scene.image_path).resolve()
        audio = Path(scene.audio_path).resolve()

        if not image.is_file() or image.stat().st_size == 0:
            raise VideoExportError(
                f"Scene {index} image is missing: {image}"
            )

        if not audio.is_file() or audio.stat().st_size == 0:
            raise VideoExportError(
                f"Scene {index} narration is missing: {audio}"
            )

        try:
            duration = inspect_audio(audio)
        except (OSError, ValueError, RuntimeError) as error:
            raise VideoExportError(
                f"Scene {index} narration is invalid: {error}"
            ) from error

        validated.append(
            (image, audio, duration)
        )

    return validated


# ---------------------------------------------------------------------------
# FFmpeg process
# ---------------------------------------------------------------------------

def _run_ffmpeg(
    command: list[str],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Run FFmpeg while periodically observing cancellation."""

    creation_flags = 0

    if os.name == "nt":
        creation_flags = subprocess.CREATE_NO_WINDOW

    # Use a file rather than PIPE for stderr. FFmpeg can produce enough
    # diagnostic output to fill a pipe while we are polling for cancellation.
    with tempfile.TemporaryFile(
        mode="w+",
        encoding="utf-8",
        errors="replace",
    ) as log:

        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log,
            creationflags=creation_flags,
        )

        try:
            while process.poll() is None:
                if cancelled is not None and cancelled():
                    process.terminate()

                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                    raise InterruptedError(
                        "Video export cancelled."
                    )

                time.sleep(0.1)

            if process.returncode != 0:
                log.seek(0)
                message = log.read().strip()

                if not message:
                    message = (
                        f"FFmpeg exited with code "
                        f"{process.returncode}."
                    )

                # The useful error is normally at the end.
                lines = message.splitlines()
                message = "\n".join(lines[-20:])

                raise VideoExportError(message)

        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


# ---------------------------------------------------------------------------
# Scene rendering
# ---------------------------------------------------------------------------

def _render_scene(
    ffmpeg: str,
    image: Path,
    audio: Path,
    duration: float,
    output: Path,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Render one static-image scene synchronized to its narration."""

    video_filter = (
        f"scale={WIDTH}:{HEIGHT}:"
        "force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},"
        "setsar=1"
    )

    command = [
        ffmpeg,

        "-hide_banner",
        "-loglevel",
        "error",
        "-y",

        # Keep the single image alive for the whole narration.
        "-loop",
        "1",

        "-framerate",
        str(FPS),

        "-i",
        str(image),

        "-i",
        str(audio),

        # Explicit streams avoid accidental metadata/data streams.
        "-map",
        "0:v:0",

        "-map",
        "1:a:0",

        "-vf",
        video_filter,

        "-c:v",
        VIDEO_CODEC,

        "-preset",
        VIDEO_PRESET,

        "-crf",
        str(VIDEO_CRF),

        "-tune",
        "stillimage",

        "-pix_fmt",
        "yuv420p",

        "-r",
        str(FPS),

        "-c:a",
        AUDIO_CODEC,

        "-b:a",
        AUDIO_BITRATE,

        "-ar",
        str(AUDIO_SAMPLE_RATE),

        "-ac",
        "2",

        # Safety bound. The looped image itself has no ending.
        "-t",
        f"{duration:.6f}",

        "-shortest",

        str(output),
    ]

    _run_ffmpeg(
        command,
        cancelled=cancelled,
    )


# ---------------------------------------------------------------------------
# Final concatenation
# ---------------------------------------------------------------------------

def _concat_clips(
    ffmpeg: str,
    directory: Path,
    clips: list[Path],
    output: Path,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Join identically encoded scene clips without re-encoding."""

    concat_file = directory / "concat.txt"

    # Only relative temporary filenames are written here. This deliberately
    # avoids Windows drive-letter and quoting problems in FFmpeg concat files.
    concat_file.write_text(
        "".join(
            f"file '{clip.name}'\n"
            for clip in clips
        ),
        encoding="utf-8",
    )

    command = [
        ffmpeg,

        "-hide_banner",
        "-loglevel",
        "error",
        "-y",

        "-f",
        "concat",

        "-safe",
        "0",

        "-i",
        str(concat_file),

        "-c",
        "copy",

        "-movflags",
        "+faststart",

        str(output),
    ]

    _run_ffmpeg(
        command,
        cancelled=cancelled,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_video(
    scenes: list[Scene],
    output_path: str | Path,
    *,
    cancelled: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> str:
    """Export a Creepty scene list as one vertical MP4.

    Args:
        scenes:
            Ordered Creepty scenes.

        output_path:
            Destination MP4.

        cancelled:
            Optional callback returning True when work should stop.

        progress:
            Optional callback receiving human-readable progress messages.

    Returns:
        Absolute path to the completed MP4.

    The destination is published atomically: an incomplete export never
    replaces an existing finished video.
    """

    def check_cancelled():
        if cancelled is not None and cancelled():
            raise InterruptedError(
                "Video export cancelled."
            )

    def report(message: str):
        if progress is not None:
            progress(message)

    check_cancelled()

    ffmpeg = find_ffmpeg()

    report("Checking video assets...")

    media = _validate_scenes(scenes)

    check_cancelled()

    output = Path(output_path).resolve()

    if output.suffix.lower() != ".mp4":
        output = output.with_suffix(".mp4")

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_output = None

    try:
        with tempfile.TemporaryDirectory(
            prefix="creepty_video_",
            dir=output.parent,
        ) as temporary_directory:

            temp = Path(temporary_directory)
            clips = []

            total = len(media)

            # -----------------------------------------------------------
            # Render individual scenes
            # -----------------------------------------------------------

            for index, (
                image,
                audio,
                duration,
            ) in enumerate(media, start=1):

                check_cancelled()

                report(
                    f"Rendering video scene "
                    f"{index} of {total}..."
                )

                clip = temp / f"scene_{index:04d}.mp4"

                _render_scene(
                    ffmpeg,
                    image,
                    audio,
                    duration,
                    clip,
                    cancelled=cancelled,
                )

                if (
                    not clip.is_file()
                    or clip.stat().st_size == 0
                ):
                    raise VideoExportError(
                        f"Scene {index} produced "
                        f"an empty video file."
                    )

                clips.append(clip)

            check_cancelled()

            # -----------------------------------------------------------
            # Concatenate
            # -----------------------------------------------------------

            report("Joining video scenes...")

            # Keep the temporary final output beside the destination.
            # os.replace() is therefore atomic on the same filesystem.
            handle = tempfile.NamedTemporaryFile(
                dir=output.parent,
                prefix=output.stem + ".",
                suffix=".tmp.mp4",
                delete=False,
            )

            temporary_output = Path(handle.name)
            handle.close()

            _concat_clips(
                ffmpeg,
                temp,
                clips,
                temporary_output,
                cancelled=cancelled,
            )

            check_cancelled()

            if (
                not temporary_output.is_file()
                or temporary_output.stat().st_size == 0
            ):
                raise VideoExportError(
                    "FFmpeg produced an empty final video."
                )

            # -----------------------------------------------------------
            # Publish
            # -----------------------------------------------------------

            os.replace(
                temporary_output,
                output,
            )

            temporary_output = None

        report("Video export complete.")

        return str(output)

    finally:
        if temporary_output is not None:
            temporary_output.unlink(
                missing_ok=True,
            )