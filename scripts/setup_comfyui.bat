@echo off
setlocal
if not defined CREEPTY_COMFY_DIR set CREEPTY_COMFY_DIR=C:\AI\ComfyUI
set COMFY_DIR=%CREEPTY_COMFY_DIR%
set MODELS=%COMFY_DIR%\models

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

if not exist "%COMFY_DIR%\venv" py -3.12 -m venv "%COMFY_DIR%\venv"
call "%COMFY_DIR%\venv\Scripts\activate"
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r "%COMFY_DIR%\requirements.txt"
call deactivate

echo === FLUX.2 Klein 4B models ===
call :get "%MODELS%\diffusion_models\flux-2-klein-4b-fp8.safetensors" https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8/resolve/main/flux-2-klein-4b-fp8.safetensors
call :get "%MODELS%\text_encoders\qwen_3_4b.safetensors" https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors
call :get "%MODELS%\vae\flux2-vae.safetensors" https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors

echo ComfyUI ready.
exit /b 0

:get
if exist %1 (
    echo skip %~nx1
    exit /b 0
)
curl -L --fail -o %1 %2
if errorlevel 1 (
    del %1 2>nul
    echo FAILED %~nx1. If it's the BFL model, accept its terms on Hugging Face and rerun.
)
exit /b 0
