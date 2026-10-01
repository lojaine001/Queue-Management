@echo off
cd /d "%~dp0"
start "Pipeline A - Checkout Lanes Camera (Exit)" "%~dp0.venv\Scripts\python.exe" main.py --execution_provider openvino --view-img
