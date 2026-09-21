@echo off
setlocal

echo === espeak-ng ===
reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall" /s /f "eSpeak NG" >nul 2>nul
if not errorlevel 1 (
    echo skip espeak-ng
    exit /b 0
)

set MSI=%TEMP%\espeak-ng-setup.msi
set URL=https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi

echo Downloading espeak-ng...
powershell -Command "Invoke-WebRequest -Uri '%URL%' -OutFile '%MSI%' -UseBasicParsing"
if not exist "%MSI%" (
    echo FAILED to download espeak-ng.
    exit /b 1
)

echo Installing espeak-ng...
msiexec /i "%MSI%" /qn
del "%MSI%" 2>nul

echo espeak-ng ready.
exit /b 0
