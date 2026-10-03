<#
.SYNOPSIS
    Starts all IQMS platform services cleanly in the correct order with designated virtualenvs.
.DESCRIPTION
    1. Checks PostgreSQL service
    2. Runs stop_all.ps1 to prevent port conflicts or duplicate runs
    3. Starts FastAPI (port 8000) as background daemon
    4. Starts Auto-Predict Scheduler (3m interval) and the Web Push alert checker as background daemons
    5. Starts Streamlit Dashboard (port 8501) as background daemon
    6. Starts IQMS-Web React App (port 3001) as background daemon
    7. Starts the ngrok tunnel to IQMS-Web (public URL; /api is proxied to 8000 by Vite)
    8. Starts Pipeline A (Head-Detector / Checkout lanes) - View Img ON by default (visible window)
    9. Starts Pipeline B (Queue-Management-v2 / Entrance) - View Img ON by default (visible window)
   10. Performs health check on ports and endpoints

    The tunnel domain defaults to wildfowl-agenda-curve.ngrok-free.dev; override it
    with the NGROK_DOMAIN environment variable.
#>

[CmdletBinding()]
param(
    [switch]$SkipCameras,
    [switch]$NoViewImg,     # When set, disables camera preview windows (headless mode)
    [switch]$NoTunnel       # When set, does not start the ngrok tunnel
)

$ErrorActionPreference = "Stop"
$rootDir = $PSScriptRoot
if (-not $rootDir) { $rootDir = Get-Location }
$logDir = Join-Path $rootDir "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# View Img is ON by default unless -NoViewImg is explicitly passed
$viewImgOn = -not $NoViewImg

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "             STARTING ALL IQMS SERVICES                   " -ForegroundColor Cyan
Write-Host "  View Image Windows: $(if ($viewImgOn) { 'ON (Visible Windows)' } else { 'OFF (Headless)' })" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Database check
Write-Host "[1/8] Checking PostgreSQL database..." -ForegroundColor Yellow
$pgService = Get-Service -Name *postgres* -ErrorAction SilentlyContinue | Select-Object -First 1
if ($pgService) {
    if ($pgService.Status -ne "Running") {
        Write-Host "      Starting PostgreSQL service ($($pgService.Name))..." -ForegroundColor Yellow
        Start-Service -Name $pgService.Name -ErrorAction SilentlyContinue
    }
    Write-Host "      [OK] PostgreSQL is active ($($pgService.Name))" -ForegroundColor Green
} else {
    Write-Host "      [!] Warning: No Windows PostgreSQL service found, checking port 5432..." -ForegroundColor Yellow
    $pgPort = Get-NetTCPConnection -LocalPort 5432 -State Listen -ErrorAction SilentlyContinue
    if (-not $pgPort) {
        Write-Host "      [ERROR] PostgreSQL port 5432 is not listening! Please start PostgreSQL first." -ForegroundColor Red
        exit 1
    }
}

# 2. Clean teardown of existing services
Write-Host "`n[2/8] Cleaning existing processes and locks..." -ForegroundColor Yellow
& (Join-Path $rootDir "stop_all.ps1")

# 3. Path definitions
$qmDir       = Join-Path $rootDir "Queue-Management-System-v2-main\Queue-Management-System-v2-main"
$qmVenvPy    = Join-Path $qmDir "venv\Scripts\python.exe"
$hdDir       = Join-Path $rootDir "Head-Detector"
$hdVenvPy    = Join-Path $hdDir ".venv\Scripts\python.exe"
$webDir      = Join-Path $rootDir "IQMS-Web"
$ngrokDomain = if ($env:NGROK_DOMAIN) { $env:NGROK_DOMAIN } else { "wildfowl-agenda-curve.ngrok-free.dev" }

# Validate virtualenvs
foreach ($py in @($qmVenvPy, $hdVenvPy)) {
    if (-not (Test-Path $py)) {
        Write-Host "[ERROR] Virtualenv Python not found at: $py" -ForegroundColor Red
        exit 1
    }
}

# Helper to launch detached background services or visible window applications
$servicePids = @{}

function Start-ServiceProcess {
    param(
        [string]$Title,
        [string]$WorkingDir,
        [string]$FilePath,
        [string]$Arguments,
        [string]$OutLog,
        [string]$ErrLog,
        [bool]$Visible = $false
    )
    
    if ($Visible) {
        # Launch independent visible interactive window using cmd /c start with explicit working directory (/d)
        $cmdLine = "/c start `"$Title`" /d `"$WorkingDir`" `"$FilePath`" $Arguments"
        $proc = Start-Process -FilePath "cmd.exe" -ArgumentList $cmdLine -WorkingDirectory $WorkingDir -PassThru
        return $proc.Id
    } else {
        # Launch detached background daemon with direct native log redirection
        $argList = if ($Arguments) { $Arguments.Split(" ") } else { @() }
        $proc = Start-Process -FilePath $FilePath -ArgumentList $Arguments -WorkingDirectory $WorkingDir `
            -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -WindowStyle Hidden -PassThru
        return $proc.Id
    }
}

# 4. Start FastAPI (Port 8000)
Write-Host "`n[3/8] Launching FastAPI Backend (Port 8000)..." -ForegroundColor Yellow
$servicePids["api"] = Start-ServiceProcess -Title "FastAPI Backend" `
    -WorkingDir $qmDir `
    -FilePath $qmVenvPy `
    -Arguments "-m uvicorn api:app --host 0.0.0.0 --port 8000" `
    -OutLog (Join-Path $logDir "api.log") `
    -ErrLog (Join-Path $logDir "api_err.log") `
    -Visible $false
Write-Host "      [OK] FastAPI launched (PID: $($servicePids['api']))" -ForegroundColor Green

# 5. Start Auto-Predict Scheduler (Background ML Forecaster)
Write-Host "`n[4/8] Launching Auto-Predict ML Scheduler (every 3m)..." -ForegroundColor Yellow
$servicePids["scheduler"] = Start-ServiceProcess -Title "ML Scheduler" `
    -WorkingDir $rootDir `
    -FilePath $qmVenvPy `
    -Arguments "run_scheduler.py --interval 3 --days 60 --source REAL" `
    -OutLog (Join-Path $logDir "scheduler.log") `
    -ErrLog (Join-Path $logDir "scheduler_err.log") `
    -Visible $false
Write-Host "      [OK] ML Scheduler launched (PID: $($servicePids['scheduler']))" -ForegroundColor Green

# Web Push alert checker (sends phone notifications when a device's threshold is crossed)
Write-Host "      Launching Web Push alert checker..." -ForegroundColor Yellow
$servicePids["push_alerts"] = Start-ServiceProcess -Title "Push Alert Checker" `
    -WorkingDir $qmDir `
    -FilePath $qmVenvPy `
    -Arguments "-u push_alert_checker.py" `
    -OutLog (Join-Path $logDir "push_alerts.log") `
    -ErrLog (Join-Path $logDir "push_alerts_err.log") `
    -Visible $false
Write-Host "      [OK] Push alert checker launched (PID: $($servicePids['push_alerts']))" -ForegroundColor Green

# 6. Start Streamlit Operator Dashboard (Port 8501)
Write-Host "`n[5/8] Launching Streamlit Operator Dashboard (Port 8501)..." -ForegroundColor Yellow
$servicePids["dashboard"] = Start-ServiceProcess -Title "Streamlit Dashboard" `
    -WorkingDir $qmDir `
    -FilePath $qmVenvPy `
    -Arguments "-m streamlit run dashboard.py --server.port=8501 --server.headless=true" `
    -OutLog (Join-Path $logDir "dashboard.log") `
    -ErrLog (Join-Path $logDir "dashboard_err.log") `
    -Visible $false
Write-Host "      [OK] Streamlit Dashboard launched (PID: $($servicePids['dashboard']))" -ForegroundColor Green

# 7. Start IQMS-Web Frontend (Port 3001)
Write-Host "`n[6/8] Launching IQMS-Web React App (Port 3001)..." -ForegroundColor Yellow
$servicePids["web"] = Start-ServiceProcess -Title "IQMS-Web" `
    -WorkingDir $webDir `
    -FilePath "cmd.exe" `
    -Arguments "/c npm run dev" `
    -OutLog (Join-Path $logDir "web.log") `
    -ErrLog (Join-Path $logDir "web_err.log") `
    -Visible $false
Write-Host "      [OK] IQMS-Web launched (PID: $($servicePids['web']))" -ForegroundColor Green

# 8. Start ngrok tunnel to IQMS-Web (Port 3001)
$ngrokCmd = Get-Command ngrok -ErrorAction SilentlyContinue
if ($NoTunnel) {
    Write-Host "`n[7/8] ngrok tunnel skipped (-NoTunnel flag set)" -ForegroundColor Gray
} elseif (-not $ngrokCmd) {
    Write-Host "`n[7/8] [!] ngrok not found on PATH - tunnel not started" -ForegroundColor Yellow
} else {
    Write-Host "`n[7/8] Launching ngrok tunnel (https://$ngrokDomain -> localhost:3001)..." -ForegroundColor Yellow
    try {
        $servicePids["ngrok"] = Start-ServiceProcess -Title "ngrok Tunnel" `
            -WorkingDir $rootDir `
            -FilePath $ngrokCmd.Source `
            -Arguments "http --url=https://$ngrokDomain 3001 --log=stdout" `
            -OutLog (Join-Path $logDir "ngrok.log") `
            -ErrLog (Join-Path $logDir "ngrok_err.log") `
            -Visible $false
        Write-Host "      [OK] ngrok launched (PID: $($servicePids['ngrok']))" -ForegroundColor Green
    } catch {
        Write-Host "      [!] ngrok failed to start: $_" -ForegroundColor Yellow
    }
}

# 9. Start Camera Pipelines
if (-not $SkipCameras) {
    Write-Host "`n[8/8] Launching Camera Video Processing Pipelines (View Img: $(if ($viewImgOn) { 'ON (Visible Windows)' } else { 'OFF (Headless)' }))..." -ForegroundColor Yellow
    
    # Pipeline A: Checkout Lanes / Head-Detector
    $hdArgs = "main.py --execution_provider openvino $(if ($viewImgOn) { '--view-img' } else { '--no-view-img' })"
    $servicePids["camera_exit"] = Start-ServiceProcess -Title "Pipeline A - Checkout Lanes Camera (Exit)" `
        -WorkingDir $hdDir `
        -FilePath $hdVenvPy `
        -Arguments $hdArgs `
        -OutLog (Join-Path $logDir "camera_exit.log") `
        -ErrLog (Join-Path $logDir "camera_exit_err.log") `
        -Visible $viewImgOn
    Write-Host "      [OK] Pipeline A (Checkout Lanes) launched (PID: $($servicePids['camera_exit']))" -ForegroundColor Green

    # Pipeline B: Entrance Demographics / Queue Management v2
    $qmArgs = "queue_management_v2.py --device openvino $(if ($viewImgOn) { '--view-img' } else { '--no-view-img' })"
    $servicePids["camera_entrance"] = Start-ServiceProcess -Title "Pipeline B - Entrance Demographics Camera (Entrance)" `
        -WorkingDir $qmDir `
        -FilePath $qmVenvPy `
        -Arguments $qmArgs `
        -OutLog (Join-Path $logDir "camera_entrance.log") `
        -ErrLog (Join-Path $logDir "camera_entrance_err.log") `
        -Visible $viewImgOn
    Write-Host "      [OK] Pipeline B (Entrance Demographics) launched (PID: $($servicePids['camera_entrance']))" -ForegroundColor Green
} else {
    Write-Host "`n[8/8] Camera video processing skipped (-SkipCameras flag set)" -ForegroundColor Gray
}

# Save active PIDs to logs/pids.json for clean tracking and teardown
$servicePids | ConvertTo-Json | Set-Content -Path (Join-Path $logDir "pids.json") -Encoding UTF8

# Warmup wait loop: Wait for ports to become active (up to 30s)
Write-Host "`n[*] Waiting for services to initialize ports..." -ForegroundColor Cyan
for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 1
    $stPort  = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
    $apiPort = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    $webPort = Get-NetTCPConnection -LocalPort 3001 -State Listen -ErrorAction SilentlyContinue
    if ($stPort -and $apiPort -and $webPort) {
        Write-Host "      All web and API ports active in ${i}s." -ForegroundColor Green
        break
    }
}

# Public tunnel URL, from the ngrok agent's local API (up to 10s)
$publicUrl = $null
if ($servicePids.ContainsKey("ngrok")) {
    for ($i = 1; $i -le 10 -and -not $publicUrl; $i++) {
        try {
            $publicUrl = ((Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 2).tunnels |
                Where-Object { $_.config.addr -match ":3001$" } | Select-Object -First 1).public_url
        } catch { }
        if (-not $publicUrl) { Start-Sleep -Seconds 1 }
    }
    if (-not $publicUrl) {
        Write-Host "      [!] ngrok tunnel did not come up - see $(Join-Path $logDir 'ngrok.log')" -ForegroundColor Yellow
    }
}

# Final Status Check
& (Join-Path $rootDir "status_all.ps1")

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "           ALL SERVICES SUCCESSFULLY STARTED              " -ForegroundColor Green
Write-Host "  - IQMS-Web App:        http://localhost:3001            " -ForegroundColor White
Write-Host "  - Public (ngrok):      $(if ($publicUrl) { $publicUrl } else { 'not running' })" -ForegroundColor White
Write-Host "  - Streamlit Dashboard: http://localhost:8501            " -ForegroundColor White
Write-Host "  - FastAPI Health:      http://localhost:8000/health     " -ForegroundColor White
Write-Host "  - Camera Windows:      $(if ($viewImgOn) { 'Visible on Desktop' } else { 'Headless' })" -ForegroundColor White
Write-Host "  - Logs Directory:      $logDir                          " -ForegroundColor White
Write-Host "==========================================================" -ForegroundColor Green
