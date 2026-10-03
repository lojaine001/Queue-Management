@echo off
title IQMS - Start All Services
echo ==========================================================
echo               STARTING ALL IQMS SERVICES
echo ==========================================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_all.ps1" %*
echo.
pause
