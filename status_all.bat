@echo off
title IQMS - System Status
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0status_all.ps1"
echo.
pause
