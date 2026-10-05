"""Build an English report from existing benchmark runs; never loads models."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

if __package__:
    from . import benchmark_pipeline as b
else:
    import benchmark_pipeline as b


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_run(path):
    path = path.resolve()
    summary = read_json(path / "summary.json")
    if not isinstance(summary, dict):
        raise ValueError(f"{path}/summary.json must contain an object")
    manifest_path = path / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    return path, summary, manifest


def metric(value):
    return f"{value:.2f}" if isinstance(value, (int, float)) else "n/a"


def write_report(run, pipeline_run, pipeline, voice_run, isolated, manifests):
    """Incomplete measurements are retained but excluded from speedups."""
    cpu, gpu = isolated.get("cpu", {}), isolated.get("gpu", {})
    profiles = {
        "Images then CPU voice": pipeline.get("sequential_cpu", pipeline.get("cpu", {})),
        "Images then GPU voice": pipeline.get("sequential_gpu", pipeline.get("gpu", {})),
        "Images + CPU voice in parallel": pipeline.get("parallel_cpu", isolated.get("parallel_cpu", {})),
    }
    llm_path = voice_run / "llm.json" if voice_run else pipeline_run / "llm.json"
    if not llm_path.exists():
        llm_path = pipeline_run / "llm.json"
    llm = read_json(llm_path) if llm_path.exists() else []
    b.write_json(run / "consolidated.json", {
        "pipeline_run": str(pipeline_run),
        "voice_run": str(voice_run) if voice_run else None,
        "pipeline": pipeline, "isolated_and_parallel": isolated,
        "llm": llm, "manifests": manifests,
    })
    lines = [
        "# Creepty benchmark report", "",
        f"Generated {datetime.now().isoformat(timespec='seconds')} from existing results.",
        "No models were loaded and no new timing measurements were taken.", "",
        "## Full pipeline", "",
        "| Profile | Elapsed seconds | Status |", "|---|---:|---|",
    ]
    for label, result in profiles.items():
        complete = b.assets_complete(result)
        status = "Complete" if complete else "Incomplete or not measured"
        # A failed attempt's elapsed time is not a completion time.
        lines.append(f"| {label} | {metric(result.get('wall_s')) if complete else 'n/a'} | {status} |")
    lines += [
        "", "Pipeline timings include ComfyUI startup and voice worker startup. "
        "The sequential GPU profile also stops ComfyUI before loading the narrator.",
        "", "## Isolated narration", "",
        "| Mode | Synthesis seconds | Audio seconds | Model load seconds | Process seconds | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    totals = {}
    for mode, result in (("CPU FP32", cpu), ("GPU BF16", gpu)):
        complete = b.voice_complete(result)
        samples = result.get("samples", [])
        synthesis = sum(s.get("wall_s", 0) for s in samples)
        duration = sum(s.get("duration_s", 0) for s in samples)
        if complete and synthesis > 0 and duration > 0:
            totals[mode] = (synthesis, duration)
            lines.append(f"| {mode} | {synthesis:.2f} | {duration:.2f} | "
                         f"{metric(result.get('model_load_s'))} | "
                         f"{metric(result.get('process_wall_s'))} | Complete |")
        else:
            lines.append(f"| {mode} | n/a | n/a | n/a | n/a | Incomplete or not measured |")
    if len(totals) == 2:
        lines += ["", f"GPU synthesis speedup over CPU: "
                  f"**{totals['CPU FP32'][0] / totals['GPU BF16'][0]:.2f}×**. "
                  "Audio duration and numerical output can differ between CPU and GPU."]
    lines += [
        "", "## Ollama", "",
        "| Profile | Sample | Context | Wall seconds | Load seconds | Valid JSON | Exact coverage |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for item in llm:
        load_ns = item.get("response", {}).get("load_duration")
        load_s = load_ns / 1e9 if isinstance(load_ns, (int, float)) else None
        lines.append(f"| {item.get('profile', 'n/a')} | {item.get('sample', 'n/a')} | "
                     f"{item.get('ctx', 'n/a')} | {metric(item.get('wall_s'))} | "
                     f"{metric(load_s)} | {item.get('json_valid', 'n/a')} | "
                     f"{item.get('exact_coverage', 'n/a')} |")
    if not llm:
        lines += ["", "No Ollama results were available; that stage may have been skipped or failed."]
    lines += [
        "", "## Samples and interpretation", "",
        "| Source | Scene | Asset |", "|---|---:|---|",
    ]
    with (run / "voice_samples.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["mode", "sample", "wall_s", "duration_s", "rtf", "sample_rate", "path"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode, result in (("cpu", cpu), ("gpu", gpu)):
            for sample in result.get("samples", []):
                writer.writerow({"mode": mode, **{k: sample.get(k) for k in fields if k != "mode"}})
                if sample.get("path"):
                    path = Path(sample["path"]).as_posix()
                    lines.append(f"| Isolated {mode} | {sample.get('sample', 'n/a')} | [WAV]({path}) |")
    for label, result in profiles.items():
        for sample in result.get("images", []):
            if sample.get("path"):
                path = Path(sample["path"]).as_posix()
                lines.append(f"| {label} | {sample.get('sample', 'n/a')} | [PNG]({path}) |")
        for sample in result.get("voice", {}).get("samples", []):
            if sample.get("path"):
                path = Path(sample["path"]).as_posix()
                lines.append(f"| {label} | {sample.get('sample', 'n/a')} | [WAV]({path}) |")
    lines += [
        "", "Each profile uses three fixed scenes in one pass. Results are diagnostics, "
        "not repeated-trial statistics or a guarantee for longer stories.",
        "WAV hashes, image pixel hashes, audio metrics, model device/dtype, package versions, "
        "and sampled system resource usage remain in the source JSON files. "
        "Numerical checks do not establish perceptual quality.",
        "Hardware and configurations may differ between supplied runs; inspect their manifests "
        "and model metadata before comparing them. Missing historical metadata is not inferred.",
        "", f"Pipeline source: {pipeline_run.as_posix()}",
        f"Voice source: {voice_run.as_posix() if voice_run else 'not supplied'}",
        "", "[Consolidated data](consolidated.json) · [Isolated voice samples](voice_samples.csv)",
    ]
    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-run", required=True, type=Path,
                        help="Completed benchmark_pipeline output directory")
    parser.add_argument("--voice-run", type=Path,
                        help="Optional benchmark_resume output with isolated narration")
    args = parser.parse_args(argv)
    try:
        pipeline_run, pipeline, pipeline_manifest = load_run(args.pipeline_run)
        voice_run, isolated, voice_manifest = (
            load_run(args.voice_run) if args.voice_run else (None, {}, {})
        )
        named_profiles = any(key in pipeline for key in
                             ("sequential_cpu", "sequential_gpu", "parallel_cpu"))
        historical_profiles = any(
            isinstance(pipeline.get(key), dict)
            and any(field in pipeline[key] for field in ("voice", "images", "error"))
            for key in ("cpu", "gpu")
        )
        if not (named_profiles or historical_profiles):
            raise ValueError("--pipeline-run does not contain pipeline results")
        if args.voice_run and not any(key in isolated for key in ("cpu", "gpu")):
            raise ValueError("--voice-run does not contain isolated voice results")
        for key in ("scenes", "seeds", "workflow_sha256"):
            if key in pipeline_manifest and key in voice_manifest:
                if pipeline_manifest[key] != voice_manifest[key]:
                    raise ValueError(f"Runs use different {key}; choose comparable runs")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    run = b.ROOT / "output" / "benchmarks" / (
        "report_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    run.mkdir(parents=True)
    write_report(run, pipeline_run, pipeline, voice_run, isolated,
                 {"pipeline": pipeline_manifest, "voice": voice_manifest})
    b.event(run, "report_ready", path=str(run / "report.md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
