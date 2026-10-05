@echo off
setlocal
cd /d "%~dp0.."

echo === Creepty GPU environment ===
if not exist ".venv\Scripts\python.exe" (
    py -3.12 -m venv .venv
    if errorlevel 1 exit /b 1
)
set "CREEPTY_PYTHON=%CD%\.venv\Scripts\python.exe"
"%CREEPTY_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

rem Use the exact CUDA build validated by the local benchmark.
"%CREEPTY_PYTHON%" -m pip install "torch==2.3.1+cu121" "torchaudio==2.3.1+cu121" --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 exit /b 1
"%CREEPTY_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
"%CREEPTY_PYTHON%" -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable: check the NVIDIA driver'; assert torch.cuda.is_bf16_supported(), 'BF16 requires a compatible GPU'; print('GPU ready:', torch.cuda.get_device_name(0))"
if errorlevel 1 exit /b 1

echo === Downloading narrator weights without loading a model ===
"%CREEPTY_PYTHON%" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')"
if errorlevel 1 exit /b 1
echo Ready. Run .venv\Scripts\python app.py
exit /b 0
