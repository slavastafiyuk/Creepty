import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline import segmenter as seg


def reply(**kwargs):
    count = len(kwargs["messages"][1]["content"].splitlines())
    body = {"scenes": [{"start": 1, "end": count,
                        "image_prompt": "A dark corridor", "mood": "dread"}]}
    return SimpleNamespace(message=SimpleNamespace(content=json.dumps(body)))


class SegmenterTests(unittest.TestCase):
    def setUp(self):
        seg._model_in_use = False
        self.unload = self.enterContext(patch.object(seg.cleanup_client, "generate"))
        self.enterContext(patch.object(seg, "model_limits", return_value=(8192, 2048)))
        self.enterContext(patch.object(seg, "estimate_tokens", return_value=7000))
        self.enterContext(patch.object(
            seg, "batch_sentences",
            side_effect=lambda sentences, budget: [
                sentences[i:i + (2 if budget > 4000 else 1)]
                for i in range(0, len(sentences), 2 if budget > 4000 else 1)
            ],
        ))

    def tearDown(self):
        seg._model_in_use = False

    def test_batches_share_context_and_unload_only_at_end(self):
        with patch.object(seg.client, "chat", side_effect=reply) as chat:
            batches = seg.stream_story("One. Two. Three. Four.")
            next(batches)
            self.unload.assert_not_called()
            next(batches)
            self.unload.assert_not_called()
            with self.assertRaises(StopIteration):
                next(batches)
        self.assertEqual([c.kwargs["options"]["num_ctx"] for c in chat.call_args_list],
                         [8192, 8192])
        self.assertTrue(all(c.kwargs["keep_alive"] == "5m"
                            for c in chat.call_args_list))
        self.unload.assert_called_once_with(model=seg.MODEL, prompt="", keep_alive=0)

    def test_capacity_retry_does_not_repeat_completed_text(self):
        attempts = 0
        def chat(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 2:
                raise RuntimeError("out of memory")
            return reply(**kwargs)
        with patch.object(seg.client, "chat", side_effect=chat):
            result = list(seg.stream_story("One. Two. Three. Four."))
        self.assertEqual([b[0] for b in result], [1, 2, 3])
        self.assertEqual(" ".join(s.text for _, _, scenes in result for s in scenes),
                         "One. Two. Three. Four.")
        self.unload.assert_called_once()

    def test_closing_iterator_unloads_model(self):
        with patch.object(seg.client, "chat", side_effect=reply):
            batches = seg.stream_story("One. Two. Three.")
            next(batches)
            batches.close()
        self.unload.assert_called_once()
        self.assertFalse(seg._model_in_use)

    def test_generation_error_is_preserved_when_cleanup_also_fails(self):
        self.unload.side_effect = RuntimeError("unload failed")
        original = ValueError("invalid response")
        with patch.object(seg.client, "chat", side_effect=original):
            with self.assertRaises(ValueError) as caught:
                list(seg.stream_story("One."))
        self.assertIs(caught.exception, original)
        self.assertIn("unload failed", original.__notes__[0])
        self.assertTrue(seg._model_in_use)

    def test_cleanup_failure_blocks_next_phase_and_can_be_retried(self):
        self.unload.side_effect = RuntimeError("service unavailable")
        with patch.object(seg.client, "chat", side_effect=reply):
            with self.assertRaisesRegex(RuntimeError, "Couldn't unload"):
                list(seg.stream_story("One."))
        self.unload.side_effect = None
        seg.unload_model()
        self.assertFalse(seg._model_in_use)
