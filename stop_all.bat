@echo off
title IQMS - Stop All Services
echo ==========================================================
echo               STOPPING ALL IQMS SERVICES
echo ==========================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_all.ps1"
echo.
pause
