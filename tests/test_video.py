import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.scenes import Scene
from pipeline.video import (
    VideoExportError,
    can_export_video,
    export_video,
)


class VideoTests(unittest.TestCase):
    def test_can_export_video_requires_image_and_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            image = root / "image.png"
            audio = root / "audio.wav"

            image.write_bytes(b"image")
            audio.write_bytes(b"audio")

            complete = Scene(
                text="Hello",
                image_prompt="A room",
                mood="calm",
                image_path=str(image),
                audio_path=str(audio),
            )

            missing_audio = complete.model_copy(
                update={"audio_path": None}
            )

            self.assertTrue(
                can_export_video([complete])
            )

            self.assertFalse(
                can_export_video([missing_audio])
            )

    def test_export_rejects_missing_image(self):
        scene = Scene(
            text="Hello",
            image_prompt="A room",
            mood="calm",
            image_path="missing.png",
            audio_path="missing.wav",
        )

        with self.assertRaises(VideoExportError):
            export_video(
                [scene],
                "video.mp4",
            )

    @patch("pipeline.video.find_ffmpeg")
    @patch("pipeline.video._validate_scenes")
    @patch("pipeline.video._render_scene")
    @patch("pipeline.video._concat_clips")
    def test_export_video(
        self,
        concat_clips,
        render_scene,
        validate_scenes,
        find_ffmpeg,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            image = root / "image.png"
            audio = root / "audio.wav"
            output = root / "video.mp4"

            image.write_bytes(b"image")
            audio.write_bytes(b"audio")

            scene = Scene(
                text="Hello",
                image_prompt="A room",
                mood="calm",
                image_path=str(image),
                audio_path=str(audio),
            )

            find_ffmpeg.return_value = "ffmpeg"
            validate_scenes.return_value = [
                (image, audio, 1.0)
            ]

            def fake_render(
                ffmpeg,
                image,
                audio,
                duration,
                output,
                **kwargs,
            ):
                output.write_bytes(b"clip")

            def fake_concat(
                ffmpeg,
                directory,
                clips,
                output,
                **kwargs,
            ):
                output.write_bytes(b"video")

            render_scene.side_effect = fake_render
            concat_clips.side_effect = fake_concat

            result = export_video(
                [scene],
                output,
            )

            self.assertEqual(
                Path(result),
                output.resolve(),
            )

            self.assertTrue(
                output.exists()
            )

            self.assertGreater(
                output.stat().st_size,
                0,
            )


if __name__ == "__main__":
    unittest.main()