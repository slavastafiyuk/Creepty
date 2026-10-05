@echo off
call "%~dp0scripts\setup_qwen-tts.bat" || exit /b 1
call "%~dp0scripts\setup_ollama.bat" || exit /b 1
call "%~dp0scripts\setup_comfyui.bat" || exit /b 1
call "%~dp0scripts\setup_ffmpeg.bat" || exit /b 1
echo.
echo All done. Run: .venv\Scripts\python app.py
