<#
.SYNOPSIS
    Stops all IQMS services cleanly (API, Scheduler, Dashboard, Web, ngrok tunnel, and Camera Pipelines).
.DESCRIPTION
    1. Kills processes recorded in logs/pids.json
    2. Finds and terminates remaining python processes running IQMS scripts
    3. Stops the ngrok agent tunnelling port 3001
    4. Kills any processes still bound to ports 8000, 8501, 3001
    5. Cleans up stale runtime locks (.runtime/*.lock)
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "SilentlyContinue"
$rootDir = $PSScriptRoot
if (-not $rootDir) { $rootDir = Get-Location }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "             STOPPING ALL IQMS SERVICES                   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Stop processes from pids.json
$pidFile = Join-Path $rootDir "logs\pids.json"
if (Test-Path $pidFile) {
    try {
        $pidsData = Get-Content $pidFile -Raw | ConvertFrom-Json
        foreach ($prop in $pidsData.PSObject.Properties) {
            $name = $prop.Name
            $targetPid = $prop.Value
            if ($targetPid -and (Get-Process -Id $targetPid -ErrorAction SilentlyContinue)) {
                Write-Host "[-] Stopping $name (PID: $targetPid)..." -ForegroundColor Yellow
                Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
            }
        }
        Remove-Item -Path $pidFile -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Warning "Could not parse $($pidFile): $_"
    }
}

# 2. Kill by known script patterns
$scriptPatterns = @(
    "run_scheduler\.py",
    "queue_management_v2\.py",
    "Head-Detector",
    "main\.py",
    "execution_provider",
    "streamlit run dashboard\.py",
    "uvicorn api:app",
    "vite",
    "Pipeline A",
    "Pipeline B"
)

$allPy = Get-CimInstance Win32_Process -Filter "name = 'python.exe' or name = 'node.exe' or name = 'cmd.exe'" -ErrorAction SilentlyContinue
foreach ($p in $allPy) {
    $cmd = $p.CommandLine
    if (-not $cmd) { continue }
    foreach ($pat in $scriptPatterns) {
        if ($cmd -match $pat) {
            Write-Host "[-] Terminating lingering process ($pat, PID: $($p.ProcessId))..." -ForegroundColor Yellow
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            break
        }
    }
}

# Orphaned multiprocessing workers from this project's venvs (e.g. a uvicorn
# child) inherit the parent's listening socket and keep port 8000 bound after
# the parent is killed. Their command line is only "spawn_main(...)", so match
# on this project's path to leave other Python programs alone.
$rootPattern = [regex]::Escape($rootDir)
$forks = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "multiprocessing-fork" -and $_.CommandLine -match $rootPattern }
foreach ($p in $forks) {
    Write-Host "[-] Terminating orphaned worker (PID: $($p.ProcessId))..." -ForegroundColor Yellow
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}

# ngrok tunnel for IQMS-Web -- only the agent forwarding to port 3001, so any
# other ngrok session on this machine is left alone
$ngrokProcs = Get-CimInstance Win32_Process -Filter "name = 'ngrok.exe'" -ErrorAction SilentlyContinue
foreach ($p in $ngrokProcs) {
    if ($p.CommandLine -match "\b3001\b") {
        Write-Host "[-] Stopping ngrok tunnel (PID: $($p.ProcessId))..." -ForegroundColor Yellow
        Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
    }
}

# 3. Verify & clear ports: 8000 (FastAPI), 8501 (Streamlit), 3001 (IQMS-Web)
$ports = @(8000, 8501, 3001)
foreach ($port in $ports) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($conn in $conns) {
        if ($conn.OwningProcess -gt 0) {
            Write-Host "[-] Port $port occupied by PID $($conn.OwningProcess) - terminating..." -ForegroundColor Yellow
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

# 4. Remove stale lock files
Write-Host "[*] Cleaning lock files..." -ForegroundColor Gray
Remove-Item -Path (Join-Path $rootDir ".runtime\*.lock") -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 1

# 5. Final check
Write-Host "`n--- Verification ---" -ForegroundColor Cyan
$activePorts = @()
foreach ($port in $ports) {
    $listening = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($listening) {
        Write-Host "[!] Port $port is still in use by PID $($listening.OwningProcess)" -ForegroundColor Red
        $activePorts += $port
    } else {
        Write-Host "[OK] Port $port is free" -ForegroundColor Green
    }
}

$ngrokLeft = Get-CimInstance Win32_Process -Filter "name = 'ngrok.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -match "\b3001\b" }
if ($ngrokLeft) {
    Write-Host "[!] ngrok tunnel still running (PID: $($ngrokLeft.ProcessId -join ', '))" -ForegroundColor Red
    $activePorts += "ngrok"
} else {
    Write-Host "[OK] ngrok tunnel is stopped" -ForegroundColor Green
}

if ($activePorts.Count -eq 0) {
    Write-Host "`n[SUCCESS] All IQMS services and ports have been cleanly stopped." -ForegroundColor Green
} else {
    Write-Host "`n[WARNING] Some ports are still active: $($activePorts -join ', ')" -ForegroundColor Yellow
}
