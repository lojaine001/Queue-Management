<#
.SYNOPSIS
    Displays comprehensive health and status of all IQMS components.
#>

$ErrorActionPreference = "SilentlyContinue"
$rootDir = $PSScriptRoot
if (-not $rootDir) { $rootDir = Get-Location }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "              IQMS SYSTEM STATUS OVERVIEW                 " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. PostgreSQL
$pg = Get-Service -Name *postgres* -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pg -and $pg.Status -eq "Running") {
    Write-Host "[OK] Database:      PostgreSQL is Running ($($pg.Name))" -ForegroundColor Green
} else {
    Write-Host "[!] Database:      PostgreSQL service is NOT running!" -ForegroundColor Red
}

# 2. Ports Check
function Check-Port($name, $port, $url) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conn) {
        Write-Host "[OK] $name`:".PadRight(20) "Port $port [Listening] (PID: $($conn.OwningProcess)) -> $url" -ForegroundColor Green
    } else {
        Write-Host "[-] $name`:".PadRight(20) "Port $port [Offline]" -ForegroundColor Red
    }
}

Check-Port "FastAPI Backend" 8000 "http://localhost:8000/health"
Check-Port "Streamlit Dashboard" 8501 "http://localhost:8501"
Check-Port "IQMS-Web Frontend" 3001 "http://localhost:3001"

# ngrok tunnel (public URL for IQMS-Web; /api is proxied through it to 8000)
try {
    $tunnel = (Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2).tunnels |
        Where-Object { $_.config.addr -match ":3001$" } | Select-Object -First 1
} catch { $tunnel = $null }
if ($tunnel) {
    Write-Host "[OK] ngrok Tunnel:".PadRight(20) "$($tunnel.public_url) -> localhost:3001" -ForegroundColor Green
} else {
    Write-Host "[-] ngrok Tunnel:".PadRight(20) "Offline" -ForegroundColor Red
}

# 3. Auto-Predict Scheduler Status
$schedJson = Join-Path $rootDir ".runtime\scheduler.json"
if (Test-Path $schedJson) {
    try {
        $sched = Get-Content $schedJson -Raw | ConvertFrom-Json
        $hb = [datetime]::Parse($sched.heartbeat)
        $diffSec = ([datetime]::UtcNow - $hb.ToUniversalTime()).TotalSeconds
        if ($diffSec -lt 120 -and (Get-Process -Id $sched.pid -ErrorAction SilentlyContinue)) {
            Write-Host "[OK] ML Scheduler:       Running (PID: $($sched.pid), Heartbeat: $([math]::Round($diffSec))s ago)" -ForegroundColor Green
        } else {
            Write-Host "[!] ML Scheduler:       Stale or inactive (Heartbeat: $([math]::Round($diffSec))s ago)" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "[-] ML Scheduler:       Could not read status" -ForegroundColor Yellow
    }
} else {
    Write-Host "[-] ML Scheduler:       No runtime heartbeat found" -ForegroundColor Red
}

# 4. Camera Pipelines
$pyProcs = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction SilentlyContinue
$pipeA = $pyProcs | Where-Object { $_.CommandLine -match "Head-Detector" -and $_.CommandLine -match "main.py" } | Select-Object -First 1
$pipeB = $pyProcs | Where-Object { $_.CommandLine -match "queue_management_v2.py" } | Select-Object -First 1

if ($pipeA) {
    Write-Host "[OK] Pipeline A (Exit):  Running (PID: $($pipeA.ProcessId), Camera: Bosch_Camera_exit)" -ForegroundColor Green
} else {
    Write-Host "[-] Pipeline A (Exit):  Offline" -ForegroundColor Red
}

if ($pipeB) {
    Write-Host "[OK] Pipeline B (Enter): Running (PID: $($pipeB.ProcessId), Camera: Bosch_Camera_Entrance)" -ForegroundColor Green
} else {
    Write-Host "[-] Pipeline B (Enter): Offline" -ForegroundColor Red
}

# 5. RTSP Camera Ping
Write-Host "`n--- Physical Camera Network Check ---" -ForegroundColor Cyan
$cams = @(
    @{ Name = "Entrance (192.168.169.1:554)"; IP = "192.168.169.1"; Port = 554 },
    @{ Name = "Exit/Lanes (192.168.1.48:554)";  IP = "192.168.1.48";  Port = 554 }
)
foreach ($c in $cams) {
    $t = Test-NetConnection -ComputerName $c.IP -Port $c.Port -WarningAction SilentlyContinue
    if ($t.TcpTestSucceeded) {
        Write-Host "[OK] $($c.Name) reachable" -ForegroundColor Green
    } else {
        Write-Host "[!] $($c.Name) NOT REACHABLE" -ForegroundColor Yellow
    }
}

Write-Host "==========================================================" -ForegroundColor Cyan
