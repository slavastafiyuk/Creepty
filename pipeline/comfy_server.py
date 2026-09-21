"""Starts and stops the local ComfyUI server from inside the app."""

import os
import subprocess
import time
from pathlib import Path

from pipeline import images

COMFY_DIR = Path(os.environ.get("CREEPTY_COMFY_DIR", r"C:\AI\ComfyUI"))
COMFY_PYTHON = COMFY_DIR / "venv" / "Scripts" / "python.exe"
STARTUP_TIMEOUT = 180   # first launch loads CUDA and can be slow

_process: subprocess.Popen | None = None


def start() -> bool:
    """Launches ComfyUI unless one is already running. Returns True when ready."""
    global _process

    if images.is_available():
        return True   # already running, e.g. started by hand: don't own it

    if not COMFY_PYTHON.exists():
        return False

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
            return False   # crashed on startup
        if images.is_available():
            return True
        time.sleep(1)
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
    _process = None
