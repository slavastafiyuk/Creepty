"""Complete diagnostics using cached Qwen weights and a fixed LLM context."""
import argparse
import concurrent.futures
from datetime import datetime
import time
import traceback
if __package__:
    from . import benchmark_pipeline as b
else:
    import benchmark_pipeline as b

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run fresh isolated CPU/GPU narration tests, then parallel assets.")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Ollama measurements")
    args = parser.parse_args(argv)
    run=b.ROOT/"output"/"benchmarks"/("resume_"+datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True)
    b.event(run,"resume_start",output_dir=str(run))
    b.write_manifest(run, "Isolated GPU/CPU narration and parallel assets")
    results={}
    if not args.skip_llm:
        try:
            with b.Monitor(run,"llm"):
                b.llm_benchmark(run)
        except Exception:
            b.event(run,"llm_error",error=traceback.format_exc())
    for mode in ["gpu","cpu"]:
        folder=run/("isolated_"+mode)
        folder.mkdir()
        try:
            with b.Monitor(folder,"resources"):
                results[mode]=b.run_voice(folder,mode,timeout=900)
        except Exception:
            results[mode]={"error":traceback.format_exc()}
        b.write_json(run/"summary.json",results)
        b.event(run,"isolated_complete",mode=mode,result=results[mode])
    cpu=results.get("cpu",{})
    if b.voice_complete(cpu):
        folder=run/"parallel_cpu"
        folder.mkdir()
        pair=None
        start=time.perf_counter()
        try:
            with b.Monitor(folder,"resources"):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    vf=pool.submit(b.run_voice,folder,"cpu",900)
                    pair=b.start_comfy(folder)
                    imgs=b.run_images(folder)
                    voc=vf.result()
                results["parallel_cpu"]={"wall_s":time.perf_counter()-start,"images":imgs,"voice":voc}
        except Exception:
            results["parallel_cpu"]={"wall_s":time.perf_counter()-start,"error":traceback.format_exc()}
        finally:
            if pair:
                b.stop_comfy(pair)
        b.write_json(run/"summary.json",results)
    b.event(run,"resume_done",output_dir=str(run))


    return 0 if (b.voice_complete(results.get("cpu", {}))
                 and b.voice_complete(results.get("gpu", {}))
                 and b.assets_complete(results.get("parallel_cpu", {}))) else 1


if __name__ == "__main__":
    raise SystemExit(main())
