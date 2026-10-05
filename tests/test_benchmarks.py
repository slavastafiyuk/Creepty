import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks import benchmark_pipeline as benchmark
from benchmarks import benchmark_finish as report


def voice_result(seconds=10):
    return {"returncode": 0, "model_load_s": 2, "process_wall_s": seconds * 3 + 2,
            "samples": [{"sample": i, "wall_s": seconds, "duration_s": 5,
                         "rtf": seconds / 5, "sample_rate": 24000}
                        for i in range(3)]}


class BenchmarkTests(unittest.TestCase):
    def test_selected_gpu_profile_uses_moved_runner_and_completes(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(benchmark, "ROOT", Path(temp)), \
             patch.object(benchmark, "write_manifest"), \
             patch.object(benchmark, "Monitor"), \
             patch.object(benchmark, "event"), \
             patch.object(benchmark, "llm_benchmark") as llm, \
             patch.object(benchmark, "start_comfy", return_value=("process", "log")), \
             patch.object(benchmark, "stop_comfy") as stop, \
             patch.object(benchmark, "run_images", return_value=[{}, {}, {}]), \
             patch.object(benchmark, "run_voice", return_value=voice_result()) as voice:
            self.assertEqual(benchmark.main(["--skip-llm", "--profiles", "sequential_gpu"]), 0)
            llm.assert_not_called()
            stop.assert_called_once_with(("process", "log"))
            self.assertEqual(voice.call_args.args[1], "gpu")
            summary = json.loads(next(Path(temp).glob("output/benchmarks/*/summary.json")).read_text())
            self.assertEqual(list(summary), ["sequential_gpu"])
            self.assertTrue(summary["sequential_gpu"]["complete"])

    def test_failed_voice_worker_makes_profile_fail(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(benchmark, "ROOT", Path(temp)), \
             patch.object(benchmark, "write_manifest"), \
             patch.object(benchmark, "Monitor"), \
             patch.object(benchmark, "event"), \
             patch.object(benchmark, "start_comfy", return_value=("process", "log")), \
             patch.object(benchmark, "stop_comfy"), \
             patch.object(benchmark, "run_images", return_value=[{}, {}, {}]), \
             patch.object(benchmark, "run_voice", return_value={"returncode": 1}):
            self.assertEqual(benchmark.main(["--skip-llm", "--profiles", "sequential_gpu"]), 1)
            summary = json.loads(next(Path(temp).glob("output/benchmarks/*/summary.json")).read_text())
            self.assertFalse(summary["sequential_gpu"]["complete"])

    def test_voice_completion_rejects_missing_duplicate_or_zero_duration_samples(self):
        result = voice_result()
        self.assertTrue(benchmark.voice_complete(result))
        duplicate = copy.deepcopy(result)
        duplicate["samples"][2]["sample"] = 1
        self.assertFalse(benchmark.voice_complete(duplicate))
        result["samples"][0]["duration_s"] = 0
        self.assertFalse(benchmark.voice_complete(result))
        self.assertFalse(benchmark.voice_complete({"returncode": 0, "samples": []}))

    def test_report_reads_existing_results_without_model_execution(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(benchmark, "ROOT", Path(temp)), \
             patch.object(benchmark, "event"), \
             patch.object(benchmark, "run_voice") as voice, \
             patch.object(benchmark, "start_comfy") as comfy:
            root = Path(temp)
            source = root / "pipeline"
            isolated = root / "voice"
            source.mkdir()
            isolated.mkdir()
            (source / "summary.json").write_text(json.dumps({
                "sequential_gpu": {"wall_s": 45, "images": [{}, {}, {}], "voice": voice_result()},
            }))
            (isolated / "summary.json").write_text(json.dumps({
                "cpu": voice_result(30), "gpu": voice_result(10),
            }))
            self.assertEqual(report.main(["--pipeline-run", str(source), "--voice-run", str(isolated)]), 0)
            text = next(root.glob("output/benchmarks/report_*/report.md")).read_text(encoding="utf-8")
            self.assertIn("**3.00×**", text)
            self.assertIn("No Ollama results", text)
            self.assertNotIn("8302", text)
            voice.assert_not_called()
            comfy.assert_not_called()

    def test_report_keeps_failures_out_of_timing_comparisons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            report.write_report(root, root, {
                "sequential_cpu": {"wall_s": 1, "error": "failed"},
            }, None, {"cpu": {"returncode": 1}, "gpu": {"returncode": 0, "samples": []}}, {})
            text = (root / "report.md").read_text(encoding="utf-8")
            self.assertIn("Incomplete or not measured", text)
            self.assertNotIn("GPU synthesis speedup", text)
            self.assertNotIn("| 1.00 |", text)

    def test_incompatible_run_seeds_are_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(benchmark, "ROOT", Path(temp)), \
             contextlib.redirect_stderr(io.StringIO()):
            root = Path(temp)
            source, isolated = root / "pipeline", root / "voice"
            for path, summary, seed in (
                (source, {"sequential_gpu": {}}, 1),
                (isolated, {"cpu": {}}, 2),
            ):
                path.mkdir()
                (path / "summary.json").write_text(json.dumps(summary))
                (path / "manifest.json").write_text(json.dumps({"seeds": [seed]}))
            with self.assertRaises(SystemExit) as caught:
                report.main(["--pipeline-run", str(source), "--voice-run", str(isolated)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse((root / "output").exists())

    def test_parallel_only_report_prefers_pipeline_result_over_failed_voice_run(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(benchmark, "ROOT", Path(temp)), \
             patch.object(benchmark, "event"):
            root = Path(temp)
            source, isolated = root / "pipeline", root / "voice"
            source.mkdir()
            isolated.mkdir()
            (source / "summary.json").write_text(json.dumps({
                "parallel_cpu": {"wall_s": 45, "images": [{}, {}, {}], "voice": voice_result()},
            }))
            (isolated / "summary.json").write_text(json.dumps({
                "cpu": voice_result(), "parallel_cpu": {"error": "out of memory"},
            }))
            self.assertEqual(report.main(["--pipeline-run", str(source), "--voice-run", str(isolated)]), 0)
            text = next(root.glob("output/benchmarks/report_*/report.md")).read_text(encoding="utf-8")
            self.assertIn("| Images + CPU voice in parallel | 45.00 | Complete |", text)

    def test_isolated_voice_directory_is_not_a_pipeline_run(self):
        with tempfile.TemporaryDirectory() as temp, \
             contextlib.redirect_stderr(io.StringIO()):
            root = Path(temp)
            (root / "summary.json").write_text(json.dumps({"cpu": voice_result()}))
            with self.assertRaises(SystemExit) as caught:
                report.main(["--pipeline-run", str(root)])
            self.assertEqual(caught.exception.code, 2)
