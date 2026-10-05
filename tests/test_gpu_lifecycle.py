import errno
import subprocess
import unittest
from unittest.mock import MagicMock, patch

import requests

from pipeline import comfy_server as comfy, images


def response(data=None):
    result = MagicMock()
    result.json.return_value = data
    return result


def stats(reserved=0, free=7 * 1024**3, cache=0):
    return response({"devices": [{"type": "cuda", "torch_vram_total": reserved,
                                 "vram_free": free, "torch_vram_free": cache}]})


IDLE = {"queue_running": [], "queue_pending": []}


class ComfyLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(comfy, "_process", None))
        self.get = self.enterContext(patch.object(comfy.requests, "get"))
        self.post = self.enterContext(patch.object(comfy.requests, "post"))
        self.enterContext(patch.object(comfy.time, "sleep"))

    def test_owned_server_is_stopped_without_external_free(self):
        process = MagicMock()
        process.poll.return_value = None
        comfy._process = process
        comfy.release_vram()
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=10)
        self.post.assert_not_called()
        self.assertIsNone(comfy._process)

    def test_kill_is_followed_by_wait(self):
        process = MagicMock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("comfy", 10), 0]
        comfy._process = process
        comfy.stop()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)

    def test_external_busy_server_is_untouched(self):
        self.get.return_value = response({"queue_running": [[0, "someone-else"]],
                                         "queue_pending": []})
        with self.assertRaisesRegex(RuntimeError, "busy"):
            comfy.release_vram()
        self.post.assert_not_called()

    def test_free_ack_waits_for_physical_memory_even_with_empty_torch_cache(self):
        self.get.side_effect = [
            response(IDLE), response(IDLE), stats(free=1024**3),
            response(IDLE), stats(),
        ]
        comfy.release_vram()
        self.assertEqual(self.get.call_count, 5)
        self.post.assert_called_once_with(
            images.COMFY_URL + "/free",
            json={"unload_models": True, "free_memory": True}, timeout=10)

    def test_reserved_cache_is_not_counted_as_physical_free_memory(self):
        self.get.side_effect = [
            response(IDLE), response(IDLE),
            stats(reserved=128 * 1024**2, free=comfy.MIN_FREE_BYTES + 64 * 1024**2,
                  cache=128 * 1024**2),
            response(IDLE), stats(),
        ]
        comfy.release_vram()
        self.assertEqual(self.get.call_count, 5)

    def test_unload_timeout_fails_before_next_model(self):
        self.get.side_effect = [response(IDLE), response(IDLE), stats(free=0)]
        with patch.object(comfy.time, "monotonic", side_effect=[0, 0, 31]):
            with self.assertRaises(TimeoutError):
                comfy.release_vram()

    def test_missing_server_is_safe_but_connection_reset_is_not(self):
        refused = requests.ConnectionError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))
        self.get.side_effect = refused
        comfy.release_vram()
        self.get.side_effect = requests.ConnectionError("connection reset")
        with self.assertRaises(requests.ConnectionError):
            comfy.release_vram()
        self.post.assert_not_called()

    def test_transient_http_failure_does_not_orphan_owned_process(self):
        old = MagicMock()
        old.poll.return_value = None
        new = MagicMock()
        new.poll.return_value = None
        comfy._process = old
        with patch.object(comfy.images, "is_available", side_effect=[False, True]), \
             patch.object(comfy, "COMFY_PYTHON") as python, \
             patch.object(comfy.subprocess, "Popen", return_value=new):
            python.exists.return_value = True
            self.assertTrue(comfy.start())
        old.terminate.assert_called_once()
        old.wait.assert_called_once()
        self.assertIs(comfy._process, new)

    def test_startup_timeout_stops_owned_process(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(comfy.images, "is_available", return_value=False), \
             patch.object(comfy, "COMFY_PYTHON") as python, \
             patch.object(comfy.subprocess, "Popen", return_value=process), \
             patch.object(comfy.time, "monotonic", side_effect=[0, 181]):
            python.exists.return_value = True
            self.assertFalse(comfy.start())
        process.terminate.assert_called_once()
        self.assertIsNone(comfy._process)


class ImageCancellationTests(unittest.TestCase):
    def test_cancel_targets_own_job_and_waits_for_it_to_leave_queue(self):
        with patch.object(images.requests, "post") as post, \
             patch.object(images.requests, "get") as get, \
             patch.object(images.time, "sleep"):
            get.side_effect = [
                response({"queue_running": [[0, "ours"]], "queue_pending": []}),
                response({"queue_running": [[1, "other"]], "queue_pending": []}),
            ]
            images.interrupt("ours")
        post.assert_called_once_with(images.COMFY_URL + "/api/jobs/ours/cancel", timeout=5)
        self.assertEqual(get.call_count, 2)

    def test_cancelled_wait_never_polls_history(self):
        with patch.object(images, "interrupt") as cancel, \
             patch.object(images.requests, "get") as get:
            with self.assertRaises(InterruptedError):
                images._wait("ours", lambda: True)
        cancel.assert_called_once_with("ours")
        get.assert_not_called()

    def test_poll_failure_cancels_submitted_job(self):
        with patch.object(images.requests, "get", side_effect=requests.ConnectionError("lost")), \
             patch.object(images, "interrupt") as cancel:
            with self.assertRaises(requests.ConnectionError):
                images._wait("ours")
        cancel.assert_called_once_with("ours")

    def test_cancellation_failure_is_reported(self):
        with patch.object(images, "interrupt", side_effect=RuntimeError("not confirmed")):
            with self.assertRaisesRegex(RuntimeError, "Could not confirm"):
                images._wait("ours", lambda: True)

    def test_image_http_error_is_not_saved_as_png(self):
        submitted = response({"prompt_id": "ours"})
        submitted.status_code = 200
        downloaded = response()
        downloaded.raise_for_status.side_effect = requests.HTTPError("500")
        with patch.object(images.requests, "post", return_value=submitted), \
             patch.object(images.requests, "get", return_value=downloaded), \
             patch.object(images, "_wait", return_value={}), \
             patch.object(images, "_first_image", return_value={}), \
             patch.object(images, "OUTPUT_DIR") as output:
            with self.assertRaises(requests.HTTPError):
                images.generate_image("a corridor", "test", seed=123)
        output.__truediv__.assert_not_called()
