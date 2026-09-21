"""Generates scene images through the local ComfyUI HTTP API.

Model-agnostic: the workflow is any graph exported in API format with the
literal text {{PROMPT}} where the prompt goes. Swapping models means
re-exporting the JSON, never touching this file.
"""

import copy
import functools
import json
import random
import time
import uuid
from pathlib import Path

import requests

COMFY_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = Path(__file__).parent / "workflows" / "image.json"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

PLACEHOLDER = "{{PROMPT}}"
SEED_KEYS = {"seed", "noise_seed"}

# Appended to every prompt so all scenes share one look.
STYLE = (
    "cinematic horror film still, dark atmosphere, film grain, "
    "dramatic low-key lighting, highly detailed"
)
TIMEOUT = 600   # seconds before giving up on one image

CLIENT_ID = str(uuid.uuid4())


@functools.cache
def _template() -> dict:
    workflow = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    if PLACEHOLDER not in json.dumps(workflow):
        raise ValueError(f"{WORKFLOW_PATH.name} has no {PLACEHOLDER} field.")
    return workflow


def _build(prompt: str, seed: int) -> dict:
    workflow = copy.deepcopy(_template())
    text = f"{prompt}\nstyle: {STYLE}"


    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key, value in inputs.items():
            if isinstance(value, str) and PLACEHOLDER in value:
                inputs[key] = value.replace(PLACEHOLDER, text)
            elif key in SEED_KEYS and isinstance(value, int):
                inputs[key] = seed
    return workflow


def is_available() -> bool:
    try:
        requests.get(f"{COMFY_URL}/system_stats", timeout=2).raise_for_status()
        return True
    except requests.RequestException:
        return False


def interrupt():
    """Stops whatever ComfyUI is currently rendering."""
    try:
        requests.post(f"{COMFY_URL}/interrupt", timeout=5)
    except requests.RequestException:
        pass


def _wait(prompt_id: str) -> dict:
    deadline = time.monotonic() + TIMEOUT
    while time.monotonic() < deadline:
        history = requests.get(
            f"{COMFY_URL}/history/{prompt_id}", timeout=10
        ).json()
        if prompt_id in history:
            entry = history[prompt_id]
            if entry.get("status", {}).get("status_str") == "error":
                raise RuntimeError("ComfyUI failed while generating the image.")
            return entry["outputs"]
        time.sleep(1)
    raise TimeoutError("ComfyUI took too long to generate the image.")


def _first_image(outputs: dict) -> dict:
    for node_output in outputs.values():
        for image in node_output.get("images", []):
            if image.get("type") == "output":
                return {
                    "filename": image["filename"],
                    "subfolder": image["subfolder"],
                    "type": image["type"],
                }
    raise RuntimeError("ComfyUI returned no image.")


def generate_image(prompt: str, name: str, seed: int | None = None) -> str:
    """Renders one image and saves it as output/<name>.png."""
    workflow = _build(
        prompt, seed if seed is not None else random.getrandbits(32)
    )

    response = requests.post(
        f"{COMFY_URL}/prompt",
        json={"prompt": workflow, "client_id": CLIENT_ID},
        timeout=30,
    )
    if response.status_code != 200:
        raise RuntimeError(f"ComfyUI rejected the workflow: {response.text}")

    outputs = _wait(response.json()["prompt_id"])
    data = requests.get(
        f"{COMFY_URL}/view", params=_first_image(outputs), timeout=60
    ).content

    path = OUTPUT_DIR / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path)
