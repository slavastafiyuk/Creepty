@echo off
setlocal
set MODEL=gemma4:e4b

echo === Ollama ===
where ollama >nul 2>nul
if errorlevel 1 (
    winget install -e --id Ollama.Ollama --accept-source-agreements --accept-package-agreements
    rem winget doesn't refresh PATH in this window, so call it by full path.
    set "OLLAMA=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
) else (
    set OLLAMA=ollama
)

"%OLLAMA%" pull %MODEL%
if errorlevel 1 (
    echo FAILED to pull %MODEL%. Is the Ollama service running?
    exit /b 1
)

echo Ollama ready.
exit /b 0
