@echo off
setlocal

echo === FFmpeg ===

rem Already available?
where ffmpeg >nul 2>nul
if not errorlevel 1 (
    ffmpeg -version >nul 2>nul
    if errorlevel 1 (
        echo FAILED: ffmpeg was found but could not run.
        exit /b 1
    )

    echo FFmpeg ready.
    exit /b 0
)

rem FFmpeg is not installed. Install through winget.
where winget >nul 2>nul
if errorlevel 1 (
    echo FAILED: winget is required to install FFmpeg.
    exit /b 1
)

echo Installing FFmpeg...
winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
if errorlevel 1 (
    echo FAILED to install FFmpeg.
    exit /b 1
)

rem winget portable applications are normally exposed here.
set "FFMPEG=%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"

if exist "%FFMPEG%" (
    "%FFMPEG%" -version >nul 2>nul
    if errorlevel 1 (
        echo FAILED: FFmpeg was installed but could not run.
        exit /b 1
    )

    echo FFmpeg ready.
    exit /b 0
)

rem Fall back to PATH in case winget refreshed it.
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo FAILED: FFmpeg was installed but is not available on PATH.
    echo Open a new terminal and run setup.bat again.
    exit /b 1
)

ffmpeg -version >nul 2>nul
if errorlevel 1 (
    echo FAILED: FFmpeg could not run.
    exit /b 1
)

echo FFmpeg ready.
exit /b 0