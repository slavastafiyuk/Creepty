"""Starts and stops the local ComfyUI server from inside the app."""

import errno
import os
import subprocess
import time
from pathlib import Path

import requests

from pipeline import images

COMFY_DIR = Path(os.environ.get("CREEPTY_COMFY_DIR", r"C:\AI\ComfyUI"))
COMFY_PYTHON = COMFY_DIR / "venv" / "scripts" / "python.exe"
STARTUP_TIMEOUT = 180   # first launch loads CUDA and can be slow
RELEASE_TIMEOUT = 30
RELEASE_RESERVED_BYTES = 256 * 1024**2
# Measured Qwen peak was ~4.5 GiB above desktop usage; leave a margin.
MIN_FREE_BYTES = 5632 * 1024**2

_process: subprocess.Popen | None = None


def start() -> bool:
    """Launches ComfyUI unless one is already running. Returns True when ready."""
    global _process

    if images.is_available():
        return True   # already running, e.g. started by hand: don't own it

    if not COMFY_PYTHON.exists():
        return False

    # A transient HTTP failure must not overwrite a still-running child.
    if _process is not None:
        stop()

    _process = subprocess.Popen(
        [str(COMFY_PYTHON), "main.py", "--port", "8188"],
        cwd=COMFY_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,   # no extra console window
    )

    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _process.poll() is not None:
            _process = None
            return False   # crashed on startup
        if images.is_available():
            return True
        time.sleep(1)
    stop()
    return False


def stop():
    """Stops ComfyUI only if this app started it."""
    global _process
    if _process and _process.poll() is None:
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=10)
    _process = None


def _connection_refused(error: BaseException) -> bool:
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, OSError) and (
            current.errno in (errno.ECONNREFUSED, 10061)
            or getattr(current, "winerror", None) == 10061
        ):
            return True
        pending.extend(
            item for item in (
                current.__cause__, current.__context__,
                getattr(current, "reason", None), *current.args,
            ) if isinstance(item, BaseException)
        )
    return False


def release_vram() -> None:
    """Stop our server, or verify an idle external server's asynchronous unload.

    An external client can still submit work afterwards; its server is never
    killed or interrupted. Fail rather than load another model while it is busy.
    """
    if _process is not None and _process.poll() is None:
        stop()
        return

    def check_idle():
        response = requests.get(f"{images.COMFY_URL}/queue", timeout=5)
        response.raise_for_status()
        queue = response.json()
        if queue["queue_running"] or queue["queue_pending"]:
            raise RuntimeError(
                "ComfyUI is busy. Finish its queued jobs before switching GPU models."
            )

    try:
        check_idle()
    except requests.ConnectionError as error:
        if _connection_refused(error):
            return  # no server listening; a reset/timeout must not be ignored
        raise

    response = requests.post(
        f"{images.COMFY_URL}/free",
        json={"unload_models": True, "free_memory": True},
        timeout=10,
    )
    response.raise_for_status()
    deadline = time.monotonic() + RELEASE_TIMEOUT
    while time.monotonic() < deadline:
        check_idle()
        response = requests.get(f"{images.COMFY_URL}/system_stats", timeout=5)
        response.raise_for_status()
        devices = response.json()["devices"]
        cuda_devices = [device for device in devices if device["type"] == "cuda"]
        if cuda_devices and all(
            device["torch_vram_total"] <= RELEASE_RESERVED_BYTES
            and device["vram_free"] - device["torch_vram_free"] >= MIN_FREE_BYTES
            for device in cuda_devices
        ):
            return
        time.sleep(0.25)
    raise TimeoutError("ComfyUI did not release its GPU models in time.")
