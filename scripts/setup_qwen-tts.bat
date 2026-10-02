@echo off
setlocal

echo ============================================
echo  Creepty setup - Qwen3-TTS ^& dependencies
echo ============================================

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python 3.12 or newer first:
    echo https://www.python.org/downloads/
    exit /b 1
)

echo.
echo === Checking Python version ===
python -c "import sys; exit(0 if sys.version_info >= (3,12) else 1)"
if errorlevel 1 (
    echo Python 3.12 or newer is required.
    python --version
    exit /b 1
)

echo.
echo === Upgrading pip ===
python -m pip install --upgrade pip
if errorlevel 1 (
    echo FAILED to upgrade pip.
    exit /b 1
)

echo.
echo === Installing PyTorch CPU build ===
python -m pip install "torch~=2.14.0" torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo FAILED to install PyTorch.
    exit /b 1
)

echo.
echo === Installing Creepty dependencies ===
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo FAILED to install requirements.txt.
    exit /b 1
)

echo.
echo === Pre-downloading Qwen3-TTS model weights ===
echo This runs once now instead of downloading the model on first generation.

python -c "from qwen_tts import Qwen3TTSModel; print('Downloading Qwen3-TTS...'); model=Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', device_map='cpu'); print('Qwen3-TTS model ready.')"

if errorlevel 1 (
    echo.
    echo WARNING: Model pre-download failed.
    echo The model will download automatically when Creepty first generates narration.
) else (
    echo Qwen3-TTS model downloaded successfully.
)

echo.
echo ============================================
echo  Setup complete.
echo  You can now run Creepty.
echo ============================================

exit /b 0