@echo off
REM One-time setup: creates .venv and installs dependencies
python -m venv .venv || py -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Setup done. Next: python check_models.py   then   run.bat
pause
