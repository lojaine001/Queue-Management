@echo off
title IQMS - Stop Camera Streams
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop_cameras.ps1"
