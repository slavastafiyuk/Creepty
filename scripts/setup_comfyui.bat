@echo off
setlocal
if not defined CREEPTY_COMFY_DIR set "CREEPTY_COMFY_DIR=C:\AI\ComfyUI"
set "COMFY_DIR=%CREEPTY_COMFY_DIR%"
set "MODELS=%COMFY_DIR%\models"
set "COMFY_PYTHON=%COMFY_DIR%\venv\Scripts\python.exe"

echo === ComfyUI in %COMFY_DIR% ===
if not exist "%COMFY_DIR%" (
    git clone https://github.com/comfyanonymous/ComfyUI.git "%COMFY_DIR%"
) else (
    git -C "%COMFY_DIR%" pull
)
if errorlevel 1 (
    echo FAILED to get ComfyUI. Is Git installed?
    exit /b 1
)

if not exist "%COMFY_PYTHON%" (
    py -3.12 -m venv "%COMFY_DIR%\venv"
    if errorlevel 1 exit /b 1
)
if not exist "%COMFY_PYTHON%" (
    echo FAILED to create the ComfyUI Python environment.
    exit /b 1
)
"%COMFY_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%COMFY_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
"%COMFY_PYTHON%" -m pip install -r "%COMFY_DIR%\requirements.txt"
if errorlevel 1 exit /b 1

echo === FLUX.2 Klein 4B models ===
call :get "%MODELS%\diffusion_models\flux-2-klein-4b-fp8.safetensors" https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors || exit /b 1
call :get "%MODELS%\text_encoders\qwen_3_4b.safetensors" https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors || exit /b 1
call :get "%MODELS%\vae\flux2-vae.safetensors" https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors || exit /b 1

echo ComfyUI ready.
exit /b 0

:get
if exist "%~1" for %%I in ("%~1") do if %%~zI GTR 0 (
    echo skip %~nx1
    exit /b 0
)
for %%I in ("%~1") do if not exist "%%~dpI" mkdir "%%~dpI"
if errorlevel 1 exit /b 1
curl --location --fail --retry 2 --output "%~1.part" "%~2"
if errorlevel 1 (
    del /q "%~1.part" 2>nul
    echo FAILED %~nx1. If it is the BFL model, accept its terms on Hugging Face and rerun.
    exit /b 1
)
if not exist "%~1.part" (
    echo FAILED: download did not create %~nx1.
    exit /b 1
)
for %%I in ("%~1.part") do if %%~zI LEQ 0 (
    del /q "%~1.part" 2>nul
    echo FAILED: downloaded file is empty: %~nx1.
    exit /b 1
)
move /y "%~1.part" "%~1" >nul
if errorlevel 1 exit /b 1
exit /b 0
