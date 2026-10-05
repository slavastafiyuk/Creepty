import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
import soundfile as sf
from PySide6.QtCore import QCoreApplication, Slot
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

import app
from pipeline import project, segmenter, voice, workers
from pipeline.audio_validation import InvalidAudioError, inspect_audio, validate_audio
from pipeline.moods import normalize_mood
from pipeline.scenes import Scene, edit_scene


def signal():
    return (0.1 * np.sin(np.arange(24000) * 2 * np.pi * 220 / 24000)).astype(np.float32)


def scene(**kwargs):
    return Scene(text="A door moved.", image_prompt="A dark door", mood="dread", **kwargs)


class AudioSafetyTests(unittest.TestCase):
    def test_invalid_signals_are_rejected_and_quiet_voice_is_allowed(self):
        for data in (np.zeros(24000), np.full(24000, np.nan), np.full(24000, np.inf),
                     np.full(24000, 0.2), np.array([]), signal()[:20], signal() * 20):
            with self.subTest(shape=data.shape), self.assertRaises(InvalidAudioError):
                validate_audio(data, 24000)
        self.assertEqual(validate_audio(signal() * 0.01, 24000), 1)
        with self.assertRaises(InvalidAudioError):
            validate_audio(signal(), 0)

    def test_invalid_generation_retries_once_and_publishes_valid_pcm(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(voice, "OUTPUT_DIR", Path(temp)), \
             patch.object(voice, "_generate", side_effect=[(np.zeros(24000), 24000), (signal(), 24000)]) as generate:
            path = voice.generate_voice("Room 12, then wait.", "scene", "suspenseful")
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(inspect_audio(path), 1)
            self.assertEqual(sf.info(path).subtype, "PCM_16")
            self.assertEqual(list(Path(temp).iterdir()), [Path(path)])

    def test_two_failed_attempts_preserve_existing_file(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(voice, "OUTPUT_DIR", Path(temp)), \
             patch.object(voice, "_generate", return_value=(np.zeros(24000), 24000)) as generate:
            target = Path(temp) / "scene.wav"
            target.write_bytes(b"previous asset")
            with self.assertRaisesRegex(RuntimeError, "two attempts"):
                voice.generate_voice("A door moved.", "scene")
            self.assertEqual(generate.call_count, 2)
            self.assertEqual(target.read_bytes(), b"previous asset")

    def test_valid_full_scale_pcm_endpoint_survives_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(voice, "OUTPUT_DIR", Path(temp)), \
             patch.object(voice, "_generate", return_value=(signal() * 9.99999, 24000)):
            path = voice.generate_voice("A door moved.", "scene")
            self.assertEqual(inspect_audio(path), 1)
            data, _ = sf.read(path)
            self.assertEqual(data.min(), -1)

    def test_cancel_during_invalid_generation_stops_before_retry(self):
        cancelled = False
        def generate(*args):
            nonlocal cancelled
            cancelled = True
            return np.zeros(24000), 24000
        with tempfile.TemporaryDirectory() as temp, patch.object(voice, "OUTPUT_DIR", Path(temp)), \
             patch.object(voice, "_generate", side_effect=generate) as model:
            with self.assertRaises(InterruptedError):
                voice.generate_voice("A door moved.", "scene", cancelled=lambda: cancelled)
            model.assert_called_once()
            self.assertFalse(list(Path(temp).iterdir()))

    def test_number_expansion_preserves_pauses_and_decimal_precision(self):
        pairs = {
            "Room 12, then turn left.": "Room twelve, then turn left.",
            "She waited 1, 2, 3 seconds.": "She waited one, two, three seconds.",
            "There were 12,345 doors.": "There were twelve thousand, three hundred and forty-five doors.",
            "Wait 2.05 seconds.": "Wait two point zero five seconds.",
            "At 3:05, then 3:15.": "At three oh five, then three fifteen.",
            "Wait 3-5 minutes.": "Wait three to five minutes.",
            "Wait 3–5 minutes.": "Wait three to five minutes.",
            "Wait 3—5 minutes.": "Wait three to five minutes.",
            "At 3:00.": "At three o'clock.",
            "In 1984.": "In nineteen eighty-four.",
        }
        for original, expected in pairs.items():
            with self.subTest(original=original):
                self.assertEqual(voice._normalize(original), expected)

    def test_freeform_moods_use_the_same_canonical_style(self):
        self.assertEqual(normalize_mood("Suspenseful, tense, fearful"), "tense")
        self.assertEqual(voice._build_instruction("Suspenseful, tense, fearful"), voice._build_instruction("tense"))
        self.assertEqual(normalize_mood("frightened"), "fear")
        self.assertEqual(normalize_mood("ominous"), "dread")
        self.assertEqual(normalize_mood("unknown"), "calm")
        boundary = segmenter.Boundary(start=1, end=1, image_prompt="Door", mood="fearful")
        self.assertEqual(boundary.mood, "fear")
        self.assertEqual(len(segmenter.Boundary.model_json_schema()["properties"]["mood"]["enum"]), 6)


class SceneAndProjectTests(unittest.TestCase):
    def test_editing_invalidates_only_affected_assets(self):
        original = scene(image_path="image.png", audio_path="voice.wav", audio_duration=1)
        prompt_edit = edit_scene(original, text=original.text, image_prompt="Another door", mood=original.mood)
        self.assertIsNone(prompt_edit.image_path)
        self.assertEqual(prompt_edit.audio_path, "voice.wav")
        self.assertEqual(prompt_edit.audio_duration, 1)
        for key, value in (("text", "Someone knocked."), ("mood", "fear")):
            changes = {key: value, **{k: getattr(original, k) for k in ("text", "image_prompt", "mood") if k != key}}
            changed = edit_scene(original, **changes)
            self.assertEqual(changed.id, original.id)
            self.assertEqual(changed.revision, 1)
            self.assertIsNone(changed.image_path)
            self.assertIsNone(changed.audio_path)
            self.assertIsNone(changed.audio_duration)
        self.assertIs(edit_scene(original, text=original.text, image_prompt=original.image_prompt, mood=original.mood), original)

    def test_manifest_roundtrip_preserves_order_ids_text_and_media_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wav = root / "voice.wav"
            png = root / "image.png"
            sf.write(wav, signal(), 24000, subtype="PCM_16")
            image = QImage(8, 8, QImage.Format.Format_RGB32)
            image.fill(0)
            self.assertTrue(image.save(str(png)))
            scenes = [scene(image_path=str(png), audio_path=str(wav), audio_duration=99), scene()]
            manifest = root / "project.json"
            project.save_project(manifest, project.Project(run_id="test", story="Source story", system_prompt="Instructions", scenes=scenes))
            data = json.loads(manifest.read_text())
            self.assertEqual(data["scenes"][0]["audio_path"], "voice.wav")
            restored, warnings = project.load_project(manifest)
            self.assertFalse(warnings)
            self.assertEqual([s.id for s in restored.scenes], [s.id for s in scenes])
            self.assertEqual(restored.story, "Source story")
            self.assertEqual(restored.scenes[0].audio_duration, 1)
            self.assertEqual(restored.scenes[0].audio_path, str(wav.resolve()))

    def test_missing_images_and_silent_audio_become_regenerable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wav = root / "silent.wav"
            sf.write(wav, np.zeros(24000), 24000)
            manifest = root / "project.json"
            project.save_project(manifest, project.Project(run_id="test", scenes=[scene(image_path=str(root / "missing.png"), audio_path=str(wav), audio_duration=1)]))
            restored, warnings = project.load_project(manifest)
            self.assertEqual(len(warnings), 2)
            self.assertIsNone(restored.scenes[0].image_path)
            self.assertIsNone(restored.scenes[0].audio_path)
            self.assertIsNone(restored.scenes[0].audio_duration)

    def test_failed_atomic_save_keeps_previous_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp) / "project.json"
            manifest.write_text("previous content")
            with patch.object(project.os, "replace", side_effect=OSError("disk failure")), self.assertRaises(OSError):
                project.save_project(manifest, project.Project(run_id="test", scenes=[scene()]))
            self.assertEqual(manifest.read_text(), "previous content")
            self.assertEqual(list(Path(temp).iterdir()), [manifest])

    def test_invalid_version_duplicate_identity_and_unsafe_run_are_rejected(self):
        for data in ({"run_id": "../escape"}, {"version": 2, "run_id": "test"},
                     {"run_id": "test", "scenes": [scene(id="same"), scene(id="same")]}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                project.Project.model_validate(data)


class PlanningSafetyTests(unittest.TestCase):
    def setUp(self):
        segmenter._model_in_use = False
        self.addCleanup(setattr, segmenter, "_model_in_use", False)
        self.unload = self.enterContext(patch.object(segmenter.cleanup_client, "generate"))
        self.enterContext(patch.object(segmenter, "model_limits", return_value=(8192, 2048)))
        self.enterContext(patch.object(segmenter, "batch_sentences", side_effect=lambda sentences, budget: [sentences[:2]] + ([sentences[2:]] if len(sentences) > 2 else [])))

    @staticmethod
    def response(**kwargs):
        count = len(kwargs["messages"][1]["content"].splitlines())
        body = {"scenes": [{"start": 1, "end": count, "image_prompt": "Door", "mood": "tense"}]}
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)), done_reason="stop")

    def test_truncation_retries_only_unfinished_text_in_smaller_batches(self):
        calls = []
        def chat(**kwargs):
            calls.append(kwargs["messages"][1]["content"])
            response = self.response(**kwargs)
            if len(calls) == 2:
                response.done_reason = "length"
            return response
        with patch.object(segmenter.client, "chat", side_effect=chat):
            batches = list(segmenter.stream_story("One. Two. Three. Four."))
        self.assertEqual(" ".join(s.text for _, _, group in batches for s in group), "One. Two. Three. Four.")
        self.assertEqual(len(calls), 4)
        self.assertEqual(calls[2:], ["1. Three.", "1. Four."])
        self.unload.assert_called_once()

    def test_invalid_json_has_finite_attempts_and_unloads(self):
        response = SimpleNamespace(message=SimpleNamespace(content="{broken"), done_reason="stop")
        with patch.object(segmenter.client, "chat", return_value=response) as chat:
            with self.assertRaises(segmenter.InvalidSceneResponse):
                list(segmenter.stream_story("One."))
            self.assertEqual(chat.call_count, segmenter.MAX_BATCH_ATTEMPTS)
        self.unload.assert_called_once()

    def test_cancel_during_request_prevents_retries_or_results(self):
        cancelled = False
        def chat(**kwargs):
            nonlocal cancelled
            cancelled = True
            return self.response(**kwargs)
        with patch.object(segmenter.client, "chat", side_effect=chat) as request:
            with self.assertRaises(InterruptedError):
                list(segmenter.stream_story("One. Two. Three.", cancelled=lambda: cancelled))
            request.assert_called_once()
        self.unload.assert_called_once()

    def test_timeout_fails_without_retry_and_preserves_cleanup(self):
        with patch.object(segmenter.client, "chat", side_effect=httpx.ReadTimeout("timed out")) as chat:
            with self.assertRaises(httpx.ReadTimeout):
                list(segmenter.stream_story("One."))
            chat.assert_called_once()
        self.unload.assert_called_once()
        self.assertEqual(segmenter.client._client.timeout.read, 180)
        self.assertEqual(segmenter.client._client.timeout.connect, 5)
        self.assertEqual(segmenter.cleanup_client._client.timeout.read, 30)


class BlockingWorker(workers.AssetWorker):
    def __init__(self, failed=False):
        super().__init__([], [], "test")
        self.entered = threading.Event()
        self.release = threading.Event()
        self.fail = failed

    @Slot()
    def run(self):
        self.run_thread_id = threading.get_ident()
        self.entered.set()
        self.release.wait(5)
        (self.failed if self.fail else self.finished).emit("Test complete")


class WindowSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt = QApplication.instance() or QApplication(["tests", "-platform", "offscreen"])
        cls.qt.setQuitOnLastWindowClosed(False)

    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.window = app.Window()
        self.window.project_path = Path(temp) / "project.json"
        self.addCleanup(self.cleanup_window)

    def pump_until(self, condition):
        deadline = time.monotonic() + 5
        while not condition() and time.monotonic() < deadline:
            QCoreApplication.processEvents()
            # QTest.qWait can retain the GIL and starve Python worker threads.
            time.sleep(0.01)
        self.assertTrue(condition())

    def cleanup_window(self):
        worker = self.window.worker
        if worker is not None:
            worker.release.set()
            self.pump_until(lambda: self.window.thread is None)
        self.window.close()
        QCoreApplication.processEvents()
        self.window.deleteLater()

    def test_edits_lock_during_work_and_old_revisions_are_ignored(self):
        original = scene()
        self.window.scenes = [original]
        self.window.refresh_table(keep_row=0)
        self.window.set_busy(True)
        self.assertFalse(self.window.detail_text.isEnabled())
        self.assertFalse(self.window.detail_image.isEnabled())
        self.assertFalse(self.window.detail_mood.isEnabled())
        self.window.detail_text.setPlainText("Stale edit")
        self.assertEqual(self.window.scenes[0].text, original.text)
        self.window.set_busy(False)
        changed = edit_scene(original, text="New narration", image_prompt=original.image_prompt, mood=original.mood)
        self.window.scenes = [changed]
        self.window.asset_ready(original.id, original.revision, "audio", "old.wav", 1, 1)
        self.assertIsNone(self.window.scenes[0].audio_path)
        self.window.scenes.insert(0, scene())
        self.window.refresh_table()
        self.window.asset_ready(changed.id, changed.revision, "audio", "new.wav", 1, 2)
        self.assertEqual(self.window.scenes[1].audio_path, "new.wav")
        self.assertIsNone(self.window.scenes[0].audio_path)

    def test_close_waits_for_running_worker_cleanup_on_success_and_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                self.window._closing = False
                self.window.show()
                worker = BlockingWorker(failed)
                self.window.start_worker(worker)
                self.pump_until(worker.entered.is_set)
                self.assertNotEqual(worker.run_thread_id, threading.get_ident())
                self.assertFalse(self.window.close())
                self.assertFalse(self.window.close())
                self.assertTrue(worker.cancelled)
                self.assertIs(self.window.worker, worker)
                self.assertTrue(self.window.isVisible())
                worker.release.set()
                self.pump_until(lambda: self.window.thread is None and not self.window.isVisible())
                self.assertIsNone(self.window.worker)

    def test_normal_completion_releases_thread_and_allows_next_worker(self):
        for _ in range(2):
            worker = BlockingWorker()
            self.window.start_worker(worker)
            self.pump_until(worker.entered.is_set)
            worker.release.set()
            self.pump_until(lambda: self.window.thread is None)
            self.assertFalse(self.window.busy)
            self.assertIsNone(self.window.worker)

    def test_new_project_save_failure_preserves_current_scenes(self):
        original = scene()
        self.window.scenes = [original]
        self.window.run_id = "old"
        self.window.story.setPlainText("A new story.")
        with patch.object(app, "save_project", side_effect=[None, OSError("disk full")]), \
             patch.object(self.window, "start_worker") as start:
            self.window.generate()
            start.assert_not_called()
        self.assertEqual(self.window.scenes, [original])
        self.assertEqual(self.window.run_id, "old")
        self.assertIn("could not be saved", self.window.status.text())

    def test_asset_dispatch_requires_a_saved_project(self):
        self.window.scenes = [scene()]
        with patch.object(app, "save_project", side_effect=OSError("disk full")), \
             patch.object(self.window, "start_worker") as start:
            self.window.run_assets([0], [])
            start.assert_not_called()

    def test_cancel_reports_persistent_scene_cleanup_failure(self):
        worker = workers.AssetWorker([], [], "test")
        worker.cancel()
        errors = []
        worker.failed.connect(errors.append)
        with patch.object(workers, "unload_scenes", side_effect=RuntimeError("Ollama unload failed")), \
             patch.object(workers.voice, "unload_model"), patch.object(workers.comfy_server, "stop") as stop:
            worker.run()
        self.assertEqual(len(errors), 1)
        self.assertIn("Ollama unload failed", errors[0])
        stop.assert_called_once()

    def test_worker_takes_snapshot_instead_of_mutable_scene_reference(self):
        original = scene()
        worker = workers.AssetWorker([(0, original)], [], "test")
        original.text = "Changed after dispatch"
        self.assertEqual(worker.image_jobs[0][1].text, "A door moved.")

    def test_missing_generation_detects_deleted_files(self):
        self.window.scenes = [scene(image_path="deleted.png", audio_path="deleted.wav")]
        with patch.object(self.window, "run_assets") as render:
            self.window.generate_missing()
            render.assert_called_once_with([0], [0])

    def test_reordering_refreshes_detail_and_persists_scene_order(self):
        first, second = scene(), scene()
        second.text = "The next scene."
        self.window.scenes = [first, second]
        self.window.refresh_table(keep_row=0)
        self.window.move_scene(1)
        self.assertEqual(self.window.detail_text.toPlainText(), first.text)
        self.assertTrue(self.window.autosave())
        loaded, _ = project.load_project(self.window.project_path)
        self.assertEqual([s.id for s in loaded.scenes], [second.id, first.id])


if __name__ == "__main__":
    unittest.main()
