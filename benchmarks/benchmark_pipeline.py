"""Reproducible local Creepty benchmark. Does not change production settings."""
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import psutil
import requests

SCENES = [
    {"text": "Someone was breathing behind the locked door.", "mood": "mysterious",
     "image_prompt": "A locked wooden door at the end of a narrow abandoned hallway, peeling pale wallpaper, a thin line of cold moonlight beneath the door, cinematic medium shot."},
    {"text": "Mara raised her flashlight. At the end of the corridor, a shadow moved against the light.", "mood": "tense",
     "image_prompt": "Mara, a woman with short dark hair wearing a red raincoat, holds a flashlight in an abandoned hospital corridor. A human shaped shadow stretches toward her against the beam. Cinematic wide shot."},
    {"text": "The footsteps stopped outside her bedroom. She held her breath and watched the handle turn, slowly, from the other side.", "mood": "dread",
     "image_prompt": "Close up of an old brass bedroom door handle beginning to turn in a dark room, chipped white paint, cold blue moonlight and a narrow warm light beneath the door, cinematic shallow depth of field."},
]
SEEDS = [14731, 29841, 39217]
LOCK = threading.Lock()

def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def event(run, name, **data):
    item = {"time": datetime.now().isoformat(), "event": name, **data}
    line = json.dumps(item, ensure_ascii=False)
    with LOCK:
        with (run / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    print(line, flush=True)

class Monitor:
    def __init__(self, run, label):
        self.run, self.label = run, label
        self.stop_event = threading.Event()
        self.samples = []
    def __enter__(self):
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        return self
    def loop(self):
        while not self.stop_event.is_set():
            m = psutil.virtual_memory()
            s = {"t": time.time(), "ram_used_gib": m.used / 2**30, "ram_available_gib": m.available / 2**30,
                 "cpu_percent": psutil.cpu_percent()}
            try:
                p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
                v = p.stdout.strip().split(",")
                s.update(vram_mib=int(v[0]), gpu_percent=int(v[1]))
            except Exception:
                pass
            self.samples.append(s)
            self.stop_event.wait(2)
    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(5)
        write_json(self.run / (self.label + "_resources.json"), self.samples)

def voice_worker(run, mode):
    t = time.perf_counter()
    import numpy as np
    import soundfile as sf
    import torch
    from pipeline import voice
    import_seconds = time.perf_counter() - t
    voice.DEVICE = "cuda:0" if mode == "gpu" else "cpu"
    voice.DTYPE = torch.bfloat16 if mode == "gpu" else torch.float32
    voice.OUTPUT_DIR = run / "audio" / mode
    t = time.perf_counter()
    model = voice._model()
    if mode == "gpu":
        torch.cuda.synchronize()
    model_load_s = time.perf_counter() - t
    meta = {"model": voice.MODEL_NAME, "mode": mode, "import_s": import_seconds, "model_load_s": model_load_s,
            "torch": torch.__version__, "threads": torch.get_num_threads(),
            "dtype": str(next(model.model.parameters()).dtype), "device": str(next(model.model.parameters()).device),
            "attention": str(getattr(model.model.config, "_attn_implementation", None)), "samples": []}
    write_json(run / ("voice_" + mode + ".json"), meta)
    event(run, "voice_model_loaded", **{k:v for k,v in meta.items() if k != "samples"})
    for i, scene in enumerate(SCENES):
        torch.manual_seed(SEEDS[i])
        np.random.seed(SEEDS[i])
        event(run, "voice_sample_start", mode=mode, sample=i)
        t = time.perf_counter()
        path = voice.generate_voice(scene["text"], "sample_%02d" % i, mood=scene["mood"])
        if mode == "gpu":
            torch.cuda.synchronize()
        wall = time.perf_counter() - t
        audio, sr = sf.read(path, dtype="float32")
        duration = len(audio) / sr
        item = {"sample": i, "text": scene["text"], "normalized_text": voice._normalize(scene["text"]),
                "mood": scene["mood"], "seed": SEEDS[i], "wall_s": wall, "duration_s": duration,
                "rtf": wall/duration, "sample_rate": sr, "peak": float(np.max(np.abs(audio))),
                "rms": float(np.sqrt(np.mean(audio**2))), "finite": bool(np.isfinite(audio).all()),
                "saturation_fraction": float(np.mean(np.abs(audio) >= 0.9999)), "path": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        meta["samples"].append(item)
        write_json(run / ("voice_" + mode + ".json"), meta)
        event(run, "voice_sample_done", **item)
    return meta

def run_voice(run, mode, timeout=900):
    log = (run / ("voice_" + mode + ".log")).open("w", encoding="utf-8")
    p = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                          "--worker", mode, "--run", str(run)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=subprocess.CREATE_NO_WINDOW, env=os.environ.copy())
    t = time.perf_counter()
    try:
        while p.poll() is None:
            if time.perf_counter()-t > timeout:
                raise TimeoutError("Voice benchmark exceeded %s seconds" % timeout)
            if psutil.virtual_memory().available < 700 * 2**20:
                raise RuntimeError("Benchmark stopped: less than 700 MiB RAM available")
            time.sleep(1)
        wall = time.perf_counter()-t
        result_file = run / ("voice_" + mode + ".json")
        result = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else {}
        result.update(process_wall_s=wall, returncode=p.returncode)
        if p.returncode:
            result["error_log"] = str(run / ("voice_" + mode + ".log"))
        write_json(result_file, result)
        event(run, "voice_process_done", mode=mode, wall_s=wall, returncode=p.returncode)
        return result
    finally:
        if p.poll() is None:
            p.terminate()
            p.wait(20)
        log.close()

def llm_benchmark(run):
    from pipeline import segmenter
    base = requests.get("http://127.0.0.1:11434/api/ps", timeout=10).json()
    if base.get("models"):
        event(run, "llm_skipped", reason="An Ollama model was already resident; avoid changing user session")
        return
    stories = [
        "Mara reached the abandoned hospital after midnight. Rain dripped from her red coat onto the tiles. She followed a faint tapping sound down the corridor. Every door was locked.\n\nAt the nurses' station, a telephone began to ring. She lifted the receiver and heard her own voice whisper her name. Then the tapping stopped. Someone was breathing behind her.",
        "The old house had stood empty for twenty years. Daniel found a photograph on the kitchen table. It showed him asleep in his apartment the previous night. He turned it over and found tomorrow's date.\n\nUpstairs, a music box began to play. He climbed the stairs, holding the photograph. His bedroom door stood at the end of the landing. From inside, a child asked why he had come back."
    ]
    results = []
    try:
        for profile, keep in [("unload_each", 0), ("resident", "5m")]:
            for i, story in enumerate(stories):
                sentences = segmenter.split_sentences(story)
                ceiling, reserved = segmenter.model_limits()
                ctx = min(ceiling, max(segmenter.CTX_FLOOR, max(segmenter.estimate_tokens(segmenter.split_sentences(s)) for s in stories)+reserved))
                payload = {"model": segmenter.MODEL, "stream": False,
                    "messages": [{"role":"system","content":segmenter.DEFAULT_SYSTEM},
                                 {"role":"user","content":"\n".join(f"{j}. {s}" for j,s in enumerate(sentences,1))}],
                    "format": segmenter.Boundaries.model_json_schema(),
                    "options": {"temperature":0.2,"num_ctx":ctx,"num_predict":4096,"seed":4421},
                    "keep_alive": keep}
                event(run, "llm_start", profile=profile, sample=i, ctx=ctx)
                t = time.perf_counter()
                response = requests.post("http://127.0.0.1:11434/api/chat", json=payload, timeout=600)
                response.raise_for_status()
                data = response.json()
                item = {"profile":profile,"sample":i,"wall_s":time.perf_counter()-t,"ctx":ctx,
                        "response":data, "input":payload}
                try:
                    raw = segmenter.Boundaries.model_validate_json(data["message"]["content"]).scenes
                    covered = [n for b in raw for n in range(b.start,b.end+1)]
                    item.update(json_valid=True, scenes=len(raw), exact_coverage=covered==list(range(1,len(sentences)+1)))
                except Exception as exc:
                    item.update(json_valid=False,validation_error=str(exc))
                results.append(item)
                write_json(run / "llm.json", results)
                event(run,"llm_done",profile=profile,sample=i,wall_s=item["wall_s"],
                      load_s=data.get("load_duration",0)/1e9,eval_count=data.get("eval_count"),
                      done_reason=data.get("done_reason"),json_valid=item["json_valid"])
    finally:
        requests.post("http://127.0.0.1:11434/api/generate",
                      json={"model":segmenter.MODEL,"keep_alive":0},timeout=90)

def start_comfy(run):
    from pipeline.comfy_server import COMFY_DIR, COMFY_PYTHON
    port = 8189
    try:
        requests.get(f"http://127.0.0.1:{port}/system_stats",timeout=1)
    except requests.ConnectionError:
        pass
    else:
        raise RuntimeError("Benchmark port 8189 is already in use")
    log = (run / "comfy.log").open("w",encoding="utf-8")
    p = subprocess.Popen([str(COMFY_PYTHON),"-u","main.py","--port",str(port),
                          "--output-directory",str(run / "comfy_output")],cwd=COMFY_DIR,
                         stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    t=time.perf_counter()
    try:
        while time.perf_counter()-t < 180:
            if p.poll() is not None:
                raise RuntimeError("ComfyUI startup failed; see " + str(run / "comfy.log"))
            try:
                r=requests.get(f"http://127.0.0.1:{port}/system_stats",timeout=2)
                if r.ok:
                    event(run,"comfy_ready",startup_s=time.perf_counter()-t,pid=p.pid)
                    return p,log
            except requests.RequestException:
                pass
            time.sleep(1)
        raise TimeoutError("ComfyUI startup timed out")
    except BaseException:
        p.terminate()
        p.wait(20)
        log.close()
        raise

def stop_comfy(pair):
    p,log=pair
    if p.poll() is None:
        p.terminate()
        p.wait(30)
    log.close()

def run_images(run):
    from pipeline import images
    from PIL import Image
    results=[]
    base="http://127.0.0.1:8189"
    for i,scene in enumerate(SCENES):
        prompt=f"context: {scene['text']}\nimage: {scene['image_prompt']}\nmood: {scene['mood']}"
        graph=images._build(prompt,SEEDS[i])
        graph["78"]["inputs"]["filename_prefix"]="benchmark/sample_%02d"%i
        event(run,"image_start",sample=i,seed=SEEDS[i])
        t=time.perf_counter()
        r=requests.post(base+"/prompt",json={"prompt":graph,"client_id":"creepty-benchmark"},timeout=30)
        r.raise_for_status()
        prompt_id=r.json()["prompt_id"]
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            h=requests.get(base+"/history/"+prompt_id,timeout=10)
            h.raise_for_status()
            entry=h.json().get(prompt_id)
            if entry:
                if entry.get("status",{}).get("status_str")=="error":
                    write_json(run/("image_error_%s.json"%i),entry)
                    raise RuntimeError("ComfyUI generation failed")
                break
            time.sleep(1)
        else:
            requests.post(base+"/interrupt",json={"prompt_id":prompt_id},timeout=10)
            raise TimeoutError("Image timed out")
        info=images._first_image(entry["outputs"])
        r=requests.get(base+"/view",params=info,timeout=60)
        r.raise_for_status()
        path=run/("image_%02d.png"%i)
        path.write_bytes(r.content)
        wall=time.perf_counter()-t
        with Image.open(path) as im:
            im.load()
            size=list(im.size)
            pixels=hashlib.sha256(im.tobytes()).hexdigest()
        cached=[]
        for kind, details in entry.get("status",{}).get("messages",[]):
            if kind=="execution_cached":
                cached.extend(details.get("nodes",[]))
        item={"sample":i,"wall_s":wall,"seed":SEEDS[i],"size":size,"pixel_sha256":pixels,
              "sampler_cached":"77:81" in cached,"cached_nodes":cached,"path":str(path),"prompt_id":prompt_id}
        if item["sampler_cached"]:
            raise RuntimeError("Full image cache hit would invalidate timing")
        results.append(item)
        write_json(run/"images.json",results)
        write_json(run/("image_history_%02d.json"%i),entry)
        event(run,"image_done",**item)
    return results

def voice_complete(result):
    samples = result.get("samples", [])
    return (not result.get("error") and result.get("returncode") == 0
            and len(samples) == len(SCENES)
            and {item.get("sample") for item in samples} == set(range(len(SCENES)))
            and all(item.get("wall_s", 0) > 0 and item.get("duration_s", 0) > 0
                    for item in samples))


def assets_complete(result):
    return (not result.get("error") and voice_complete(result.get("voice", {}))
            and len(result.get("images", [])) == len(SCENES))


def write_manifest(run, method):
    packages = {name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "qwen-tts", "numpy", "psutil", "Pillow")}
    write_json(run / "manifest.json", {
        "scenes": SCENES, "seeds": SEEDS, "packages": packages, "method": method,
        "ram_total_gib": psutil.virtual_memory().total / 2**30,
        "cpu_logical": psutil.cpu_count(),
        "comfy_dir": os.environ.get("CREEPTY_COMFY_DIR", r"C:\AI\ComfyUI"),
        "workflow_sha256": hashlib.sha256(
            (ROOT / "pipeline/workflows/image.json").read_bytes()).hexdigest(),
        "cache": "Fresh ComfyUI server per profile; sampler cache hits rejected",
        "sampling": "Application sampling parameters with fixed per-scene seeds",
        "scope": "Three different scenes, one pass; not repeated-trial statistics",
    })


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=["cpu", "gpu"], help=argparse.SUPPRESS)
    parser.add_argument("--run", help=argparse.SUPPRESS)
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip Ollama measurements")
    parser.add_argument("--profiles", nargs="+",
                        choices=["sequential_cpu", "sequential_gpu", "parallel_cpu"],
                        default=["sequential_cpu", "sequential_gpu", "parallel_cpu"],
                        help="Asset profiles to measure (default: all three)")
    args=parser.parse_args(argv)
    if args.worker and not args.run:
        parser.error("--worker requires --run")
    if args.run and not args.worker:
        parser.error("--run is only valid with --worker")
    if args.worker:
        voice_worker(Path(args.run),args.worker)
        return
    run=ROOT/"output"/"benchmarks"/datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True)
    write_manifest(run, "Full pipeline comparison")
    event(run,"benchmark_start",output_dir=str(run))
    if not args.skip_llm:
        try:
            with Monitor(run,"llm"):
                llm_benchmark(run)
        except Exception:
            event(run,"llm_error",error=traceback.format_exc())
    results={}
    for profile in dict.fromkeys(args.profiles):
        folder=run/profile
        folder.mkdir()
        pair=None
        event(run,"profile_start",profile=profile)
        t=time.perf_counter()
        try:
            with Monitor(folder,"resources"):
                if profile=="parallel_cpu":
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        vf=pool.submit(run_voice,folder,"cpu")
                        pair=start_comfy(folder)
                        imgs=run_images(folder)
                        voc=vf.result()
                else:
                    pair=start_comfy(folder)
                    imgs=run_images(folder)
                    if profile=="sequential_gpu":
                        stop_comfy(pair)
                        pair=None
                    voc=run_voice(folder,"gpu" if profile=="sequential_gpu" else "cpu")
                results[profile]={"wall_s":time.perf_counter()-t,"images":imgs,"voice":voc}
        except Exception:
            results[profile]={"wall_s":time.perf_counter()-t,"error":traceback.format_exc()}
        finally:
            if pair:
                stop_comfy(pair)
        results[profile]["complete"] = assets_complete(results[profile])
        write_json(run/"summary.json",results)
        event(run,"profile_done",profile=profile,wall_s=results[profile]["wall_s"],
              error=results[profile].get("error"),voice_returncode=results[profile].get("voice",{}).get("returncode"))
    event(run,"benchmark_done",output_dir=str(run))
    return 0 if all(result["complete"] for result in results.values()) else 1

if __name__=="__main__":
    raise SystemExit(main())
