@echo off
title IQMS - Run Live Camera Streams
echo ==========================================================
echo        STARTING IQMS LIVE CAMERA STREAMS (OPENCV)
echo ==========================================================

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cameras.ps1"
ping 127.0.0.1 -n 3 >nul

set "ROOT=%~dp0"
set "HD_DIR=%ROOT%Head-Detector"
set "HD_PY=%HD_DIR%\.venv\Scripts\python.exe"
set "QM_DIR=%ROOT%Queue-Management-System-v2-main\Queue-Management-System-v2-main"
set "QM_PY=%QM_DIR%\venv\Scripts\python.exe"

echo Launching Pipeline A (Checkout Lanes / Exit Camera)...
start "Pipeline A - Checkout Lanes Camera (Exit)" /d "%HD_DIR%" "%HD_PY%" main.py --execution_provider openvino --view-img

ping 127.0.0.1 -n 3 >nul

echo Launching Pipeline B (Entrance Demographics / Entrance Camera)...
start "Pipeline B - Entrance Demographics Camera (Entrance)" /d "%QM_DIR%" "%QM_PY%" queue_management_v2.py --device openvino --view-img

echo.
echo ==========================================================
echo    OpenCV Windows Launched:
echo    1. "Detection Application YOLOv9" (Exit / Checkout Lanes)
echo    2. "Queue Management System"      (Entrance Demographics)
echo ==========================================================

