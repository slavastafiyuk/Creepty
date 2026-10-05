import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline import workers as app
from pipeline import voice
from pipeline.segmenter import Scene


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.scene = Scene(text="Someone is here.", image_prompt="A dark door", mood="dread")

        def record(name, result=None):
            def run(*args, **kwargs):
                self.events.append(name)
                return result
            return run

        self.unload = self.enterContext(patch.object(voice, "unload_model", side_effect=record("voice_free")))
        self.enterContext(patch.object(app, "unload_scenes", side_effect=record("gemma_free")))
        self.start = self.enterContext(patch.object(app.comfy_server, "start", side_effect=record("image_start", True)))
        self.release = self.enterContext(patch.object(app.comfy_server, "release_vram", side_effect=record("image_free")))
        self.stop = self.enterContext(patch.object(app.comfy_server, "stop", side_effect=record("image_stop")))
        self.image = self.enterContext(patch.object(app.images, "generate_image", side_effect=record("image", "image.png")))
        self.voice = self.enterContext(patch.object(voice, "generate_voice", side_effect=record("voice", "voice.wav")))
        self.enterContext(patch.object(app.sf, "info", return_value=MagicMock(duration=1.0)))
        self.finished = []
        self.failed = []

    def run_worker(self, worker):
        worker.finished.connect(self.finished.append)
        worker.failed.connect(self.failed.append)
        worker.run()
        return worker

    def test_all_images_then_all_voices_and_final_release(self):
        jobs = [(0, self.scene), (1, self.scene)]
        self.run_worker(app.AssetWorker(jobs, jobs, "test"))
        self.assertEqual(self.events, ["voice_free", "gemma_free", "image_start",
                                      "image", "image", "image_free", "voice", "voice", "voice_free", "gemma_free"])
        self.assertEqual(self.finished, ["Done."])
        self.assertFalse(self.failed)

    def test_regeneration_after_image_only_batch_frees_comfy_before_voice(self):
        self.run_worker(app.AssetWorker([(0, self.scene)], [], "test"))
        self.release.assert_not_called()
        self.stop.assert_not_called()
        self.events.clear()
        self.run_worker(app.AssetWorker([], [(0, self.scene)], "test"))
        self.assertEqual(self.events, ["voice_free", "gemma_free", "image_free", "voice", "voice_free", "gemma_free"])

    def test_failed_gpu_handoff_never_starts_voice(self):
        self.release.side_effect = RuntimeError("ComfyUI busy")
        self.run_worker(app.AssetWorker([], [(0, self.scene)], "test"))
        self.voice.assert_not_called()
        self.stop.assert_called_once()
        self.assertEqual(len(self.failed), 1)
        self.assertFalse(self.finished)

    def test_voice_failure_still_frees_gpu_before_failed_signal(self):
        self.voice.side_effect = RuntimeError("CUDA out of memory")
        self.run_worker(app.AssetWorker([], [(0, self.scene)], "test"))
        self.assertEqual(self.events[-3:], ["voice_free", "gemma_free", "image_stop"])
        self.assertIn("out of memory", self.failed[0])

    def test_generator_finishes_planning_and_unload_before_images(self):
        def stream(*args, **kwargs):
            try:
                self.events.append("plan")
                yield 1, 1, [self.scene]
            finally:
                self.events.append("plan_free")

        with patch.object(app, "stream_story", side_effect=stream):
            self.run_worker(app.Generator("story", "instructions", "test"))

        self.assertEqual(self.events, ["voice_free", "gemma_free", "image_free",
                                      "plan", "plan_free", "image_start", "image",
                                      "image_free", "voice", "voice_free", "gemma_free"])

    def test_cancel_during_scene_delivery_closes_iterator_without_rendering(self):
        def stream(*args, **kwargs):
            try:
                yield 1, 2, [self.scene]
                self.fail("Cancelled generator consumed another batch")
            finally:
                self.events.append("plan_free")

        worker = app.Generator("story", "instructions", "test")
        worker.scenes_ready.connect(lambda *args: worker.cancel())

        with patch.object(app, "stream_story", side_effect=stream):
            self.run_worker(worker)

        self.assertIn("plan_free", self.events)
        self.image.assert_not_called()
        self.voice.assert_not_called()
        self.assertEqual(self.finished, ["Cancelled."])

    def test_cancel_during_gpu_handoff_does_not_load_gemma(self):
        worker = app.Generator("story", "instructions", "test")
        self.release.side_effect = worker.cancel

        with patch.object(app, "stream_story") as stream:
            self.run_worker(worker)

        stream.assert_not_called()
        self.assertEqual(self.finished, ["Cancelled."])

    def test_video_worker_exports_video(self):
        progress = []

        def export_video(scenes, output_path, cancelled, progress):
            progress("Rendering video scene 1 of 1...")
            return output_path

        worker = app.VideoWorker([self.scene], "video.mp4")
        worker.progress.connect(progress.append)

        with patch.object(app.video, "export_video", side_effect=export_video) as export:
            self.run_worker(worker)

        export.assert_called_once()
        self.assertEqual(progress, ["Rendering video scene 1 of 1..."])
        self.assertEqual(self.finished, ["Video exported: video.mp4"])
        self.assertFalse(self.failed)

    def test_cancelled_video_worker_does_not_export(self):
        worker = app.VideoWorker([self.scene], "video.mp4")
        worker.cancel()

        with patch.object(app.video, "export_video") as export:
            self.run_worker(worker)

        export.assert_not_called()
        self.assertEqual(self.finished, ["Cancelled."])
        self.assertFalse(self.failed)


class VoiceTests(unittest.TestCase):
    def setUp(self):
        voice._model.cache_clear()
        self.addCleanup(voice._model.cache_clear)

    def test_cuda_bf16_sdpa_model_is_reused_until_unload(self):
        with patch.object(voice.torch.cuda, "is_available", return_value=True), \
             patch.object(voice.torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(voice.torch.cuda, "is_initialized", return_value=True), \
             patch.object(voice.torch.cuda, "empty_cache") as empty, \
             patch.object(voice, "snapshot_download", return_value="cached-model"), \
             patch.object(voice, "_complete_snapshot", return_value=True), \
             patch.object(voice.Qwen3TTSModel, "from_pretrained") as load:
            self.assertIs(voice._model(), voice._model())
            load.assert_called_once_with("cached-model", device_map="cuda:0",
                                         dtype=voice.torch.bfloat16, attn_implementation="sdpa")
            load.return_value.model.speech_tokenizer.model.decoder.to.assert_called_once_with(
                device="cuda:0", dtype=voice.torch.float32)
            voice.unload_model()
            empty.assert_called_once()
            self.assertEqual(voice._model.cache_info().currsize, 0)

    def test_no_silent_cpu_fallback(self):
        with patch.object(voice.torch.cuda, "is_available", return_value=False), \
             patch.object(voice.Qwen3TTSModel, "from_pretrained") as load:
            with self.assertRaisesRegex(RuntimeError, "requires CUDA"):
                voice._model()

        load.assert_not_called()

    def test_partial_snapshot_is_completed_before_loading(self):
        with patch.object(voice.torch.cuda, "is_available", return_value=True), \
             patch.object(voice.torch.cuda, "is_bf16_supported", return_value=True), \
             patch.object(voice, "snapshot_download", side_effect=["partial", "complete"]) as download, \
             patch.object(voice, "_complete_snapshot", return_value=False), \
             patch.object(voice.Qwen3TTSModel, "from_pretrained") as load:
            voice._model()

        self.assertEqual(download.call_count, 2)
        self.assertEqual(load.call_args.args, ("complete",))

    def test_snapshot_check_includes_audio_tokenizer_and_all_shards(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            required = ["config.json", "generation_config.json", "preprocessor_config.json",
                        "tokenizer_config.json", "vocab.json", "merges.txt",
                        "speech_tokenizer/config.json", "speech_tokenizer/preprocessor_config.json"]

            for name in required:
                path = root / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("{}")

            (root / "model.safetensors").write_bytes(b"weights")
            self.assertFalse(voice._complete_snapshot(temp))

            tokenizer = root / "speech_tokenizer"

            (tokenizer / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "part1", "b": "part2"}})
            )

            (tokenizer / "part1").write_bytes(b"weights")
            self.assertFalse(voice._complete_snapshot(temp))

            (tokenizer / "part2").write_bytes(b"weights")
            self.assertTrue(voice._complete_snapshot(temp))


if __name__ == "__main__":
    unittest.main()