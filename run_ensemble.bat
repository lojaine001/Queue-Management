@echo off
set PYTHONIOENCODING=utf-8
cd /d "%~dp0Queue-Management-System-v2-main\Queue-Management-System-v2-main"
"%~dp0Queue-Management-System-v2-main\Queue-Management-System-v2-main\venv\Scripts\python.exe" ensemble_predict.py >> "%~dp0ensemble_predict.log" 2>&1
