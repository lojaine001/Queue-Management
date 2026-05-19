@echo off
setlocal

cd /d "%~dp0"

if "%~1"=="" goto usage

python run_simulator.py %*
exit /b %errorlevel%

:usage
echo Usage:
echo   run_simulator.bat batch [normal_day^|lunch_rush^|evening_rush]
echo   run_simulator.bat test
echo   run_simulator.bat dashboard
echo   run_simulator.bat viewer [normal_day^|lunch_rush^|evening_rush]
echo   run_simulator.bat calibrate [--days 30] [--output report^|json^|py]
exit /b 1
