<#
.SYNOPSIS
    Starts both IQMS live camera processing pipelines with interactive OpenCV desktop preview windows.
.DESCRIPTION
    1. Pipeline A: Checkout Lanes (Head-Detector / Exit Camera) -> OpenCV window "Detection Application YOLOv9"
    2. Pipeline B: Entrance Demographics (queue_management_v2 / Entrance Camera) -> OpenCV window "Queue Management System"
    Both run in visible command windows using their designated virtualenvs.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$rootDir = $PSScriptRoot
if (-not $rootDir) { $rootDir = Get-Location }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "       STARTING IQMS LIVE CAMERA STREAMS (OPENCV GUI)     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Clean up any existing camera instances to prevent mutex locks or socket conflicts
Write-Host "[1/3] Stopping any previous camera instances..." -ForegroundColor Yellow
& (Join-Path $rootDir "stop_cameras.ps1")
Start-Sleep -Seconds 1

# 2. Setup paths and verify virtual environments
$hdDir    = Join-Path $rootDir "Head-Detector"
$hdVenvPy = Join-Path $hdDir ".venv\Scripts\python.exe"
$qmDir    = Join-Path $rootDir "Queue-Management-System-v2-main\Queue-Management-System-v2-main"
$qmVenvPy = Join-Path $qmDir "venv\Scripts\python.exe"

if (-not (Test-Path $hdVenvPy)) {
    Write-Host "[ERROR] Pipeline A Python virtualenv not found at: $hdVenvPy" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $qmVenvPy)) {
    Write-Host "[ERROR] Pipeline B Python virtualenv not found at: $qmVenvPy" -ForegroundColor Red
    exit 1
}

# 3. Launch Pipeline A (Checkout Lanes / Exit Camera)
Write-Host "[2/3] Launching Pipeline A (Checkout Lanes / Exit Camera)..." -ForegroundColor Yellow
$procA = Start-Process -FilePath $hdVenvPy -ArgumentList "main.py --execution_provider openvino --view-img" -WorkingDirectory $hdDir -PassThru
Write-Host "      [OK] Pipeline A launched (PID: $($procA.Id), OpenCV window: Detection Application YOLOv9)" -ForegroundColor Green

Start-Sleep -Seconds 2

# 4. Launch Pipeline B (Entrance Demographics / Entrance Camera)
Write-Host "[3/3] Launching Pipeline B (Entrance Demographics / Entrance Camera)..." -ForegroundColor Yellow
$procB = Start-Process -FilePath $qmVenvPy -ArgumentList "queue_management_v2.py --device openvino --view-img" -WorkingDirectory $qmDir -PassThru
Write-Host "      [OK] Pipeline B launched (PID: $($procB.Id), OpenCV window: Queue Management System)" -ForegroundColor Green

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "          LIVE CAMERA STREAMS ACTIVATED SUCCESSFULLY      " -ForegroundColor Green
Write-Host "  Look at your desktop for the following OpenCV windows:  " -ForegroundColor White
Write-Host "  1. 'Detection Application YOLOv9' (Exit / Checkout Lanes)" -ForegroundColor Cyan
Write-Host "  2. 'Queue Management System'      (Entrance Demographics) " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Green
