@echo off
setlocal
rem Project root is one level above this script, wherever it's called from.
cd /d "%~dp0.."

echo === Creepty app ===
if not exist .venv py -3.12 -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
call deactivate

echo App ready.
exit /b 0
