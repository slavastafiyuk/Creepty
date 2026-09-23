@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Creepty setup - Parler-TTS ^& dependencies
echo ============================================

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found on PATH.
    echo Install Python 3.10 or newer first: https://www.python.org/downloads/
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
echo === Installing PyTorch (CPU build) ===
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo FAILED to install torch.
    exit /b 1
)

echo.
echo === Installing Parler-TTS ===
echo (descript-audiotools ships a post-checkout git hook; the clone is
echo  disallowed by default in recent Git versions unless this is set)
set GIT_CLONE_PROTECTION_ACTIVE=false
python -m pip install git+https://github.com/huggingface/parler-tts.git
set INSTALL_RESULT=%errorlevel%
set GIT_CLONE_PROTECTION_ACTIVE=
if not "%INSTALL_RESULT%"=="0" (
    echo FAILED to install parler-tts.
    exit /b 1
)

echo.
echo === Installing remaining Python packages ===
python -m pip install PySide6 soundfile numpy
if errorlevel 1 (
    echo FAILED to install PySide6, soundfile or numpy.
    exit /b 1
)

echo.
echo === Pre-downloading Parler-TTS model weights ===
echo This runs once now instead of stalling the app on first use.
python -c "from parler_tts import ParlerTTSForConditionalGeneration; from transformers import AutoTokenizer; m='parler-tts/parler-tts-mini-multilingual-v1.1'; model=ParlerTTSForConditionalGeneration.from_pretrained(m); AutoTokenizer.from_pretrained(m); AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)"
if errorlevel 1 (
    echo Model pre-download failed - it will download automatically on first generation instead.
)

echo.
echo Setup complete. You can now run the app.
exit /b 0
