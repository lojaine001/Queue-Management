<#
.SYNOPSIS
    Stops both IQMS camera video processing pipelines cleanly.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "SilentlyContinue"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "             STOPPING IQMS CAMERA PIPELINES               " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$cameraPatterns = @(
    "queue_management_v2\.py",
    "Head-Detector",
    "main\.py",
    "execution_provider",
    "Pipeline A",
    "Pipeline B"
)

$allPy = Get-CimInstance Win32_Process -Filter "name = 'python.exe' or name = 'cmd.exe'" -ErrorAction SilentlyContinue
$stoppedCount = 0

foreach ($p in $allPy) {
    $cmd = $p.CommandLine
    if (-not $cmd) { continue }
    foreach ($pat in $cameraPatterns) {
        if ($cmd -match $pat) {
            Write-Host "[-] Terminating camera process (PID: $($p.ProcessId), Script: $pat)..." -ForegroundColor Yellow
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            $stoppedCount++
            break
        }
    }
}

# Release any lingering mutex lock files
$rootDir = $PSScriptRoot
if (-not $rootDir) { $rootDir = Get-Location }
Remove-Item -Path (Join-Path $rootDir ".runtime\*.lock") -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 1
Write-Host "[OK] Camera pipelines stopped ($stoppedCount processes terminated)." -ForegroundColor Green
