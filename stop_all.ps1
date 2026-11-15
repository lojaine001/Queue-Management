<#
.SYNOPSIS
    Stops all IQMS services cleanly (API, Scheduler, Dashboard, Web, ngrok tunnel, and Camera Pipelines).
.DESCRIPTION
    Only processes that belong to this project are stopped; other programs on the
    machine are left alone, even when they run a main.py, a vite server, or hold
    one of the IQMS ports (that case is reported, not killed).

    1. Finds this project's processes: python / node started from, or pointing into,
       this folder, plus everything they started
    2. Adds IQMS services whose launcher is gone (matched on their exact command)
    3. Stops Caddy (tools\bin\caddy.exe) and the ngrok agent tunnelling port 3001
    4. Reports any process still bound to ports 8000, 8501, 3001
    5. Cleans up stale runtime locks (.runtime/*.lock)

    Run with -WhatIf to list what would be stopped without stopping anything.
#>

[CmdletBinding(SupportsShouldProcess)]
param()

$ErrorActionPreference = "SilentlyContinue"
$rootDir = $PSScriptRoot
if (-not $rootDir) { $rootDir = Get-Location }

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "             STOPPING ALL IQMS SERVICES                   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# A path belongs to this project when it is inside the project folder.
$rootFull  = [System.IO.Path]::GetFullPath("$rootDir").TrimEnd('\')
$underRoot = [regex]::Escape($rootFull + '\')

# Loaded with -WhatIf switched off, so a dry run does not list the module's own setup.
$whatIfWas = $WhatIfPreference; $WhatIfPreference = $false
Import-Module CimCmdlets
$WhatIfPreference = $whatIfWas
$allProcs = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
$byPid = @{}
foreach ($p in $allProcs) { $byPid[[int]$p.ProcessId] = $p }

# The shell running this script (and whatever started it, e.g. start_all) is never stopped.
$protected = @{}
$cur = $byPid[[int]$PID]
while ($cur -and -not $protected.ContainsKey([int]$cur.ProcessId)) {
    $protected[[int]$cur.ProcessId] = $true
    $cur = $byPid[[int]$cur.ParentProcessId]
}

function Test-IqmsProcess($p) {
    # python / node whose program or command line points into this project
    # (the venv interpreters, multiprocessing workers, the IQMS-Web dev server),
    # and the `cmd /c start` wrappers start_all.ps1 uses for the camera windows.
    if (-not $p) { return $false }
    $cmd = "$($p.CommandLine)"
    if ($p.Name -eq 'cmd.exe') { return ($cmd -match '/c\s+start\s+"' -and $cmd -match $underRoot) }
    if ($p.Name -notin @('python.exe', 'pythonw.exe', 'node.exe')) { return $false }
    return (("$($p.ExecutablePath)" -match $underRoot) -or ($cmd -match $underRoot))
}

# 1. This project's processes, and everything they started.
#    A venv python.exe is only a launcher: the real interpreter is its child and
#    shows no project path, so children are collected by parent.
$targets = @{}
foreach ($p in $allProcs) {
    if (Test-IqmsProcess $p) { $targets[[int]$p.ProcessId] = "in project folder" }
}
do {
    $added = $false
    foreach ($p in $allProcs) {
        $id = [int]$p.ProcessId
        if ($targets.ContainsKey($id)) { continue }
        $parent = $byPid[[int]$p.ParentProcessId]
        # CreationDate check: a parent PID can be reused by a newer, unrelated process.
        if ($parent -and $targets.ContainsKey([int]$parent.ProcessId) -and $p.CreationDate -ge $parent.CreationDate) {
            $targets[$id] = "started by PID $($parent.ProcessId)"
            $added = $true
        }
    }
} while ($added)

# 2. IQMS services whose launcher is gone. They show no project path, so they are
#    matched on the exact command start_all.ps1 uses, and only when they have no
#    living parent (a copy someone runs from a terminal elsewhere is left alone).
$orphanSignatures = @(
    'run_scheduler\.py\s+--interval',
    'push_alert_checker\.py',
    'queue_management_v2\.py\s+--device',
    'main\.py\s+--execution_provider',
    'uvicorn\s+api:app\b.*--port\s+8000',
    'streamlit\s+run\s+dashboard\.py\b.*--server\.port=8501'
)
foreach ($p in $allProcs) {
    $id = [int]$p.ProcessId
    if ($p.Name -notin @('python.exe', 'pythonw.exe') -or $targets.ContainsKey($id) -or -not $p.CommandLine) { continue }
    $parent = $byPid[[int]$p.ParentProcessId]
    if ($parent -and $parent.CreationDate -le $p.CreationDate) { continue }   # parent still alive
    foreach ($sig in $orphanSignatures) {
        if ($p.CommandLine -match $sig) { $targets[$id] = "IQMS service without launcher"; break }
    }
}

foreach ($id in @($targets.Keys | Sort-Object)) {
    if ($protected.ContainsKey($id)) { continue }
    $p = $byPid[$id]
    Write-Host "[-] Stopping $($p.Name) (PID: $id, $($targets[$id]))..." -ForegroundColor Yellow
    Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
}

# logs\pids.json only lists what start_all.ps1 launched; those processes are found
# above, and a PID read from an old file may now belong to another program.
Remove-Item -Path (Join-Path $rootDir "logs\pids.json") -Force -ErrorAction SilentlyContinue

# Caddy serving IQMS-Web -- only this project's copy (tools\bin\caddy.exe)
$caddyPath = [regex]::Escape((Join-Path $rootDir "tools\bin\caddy.exe"))
$caddyProcs = Get-CimInstance Win32_Process -Filter "name = 'caddy.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -and $_.ExecutablePath -match "^$caddyPath$" }
foreach ($p in $caddyProcs) {
    Write-Host "[-] Stopping Caddy (PID: $($p.ProcessId))..." -ForegroundColor Yellow
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

# 3. Remove stale lock files
Write-Host "[*] Cleaning lock files..." -ForegroundColor Gray
Remove-Item -Path (Join-Path $rootDir ".runtime\*.lock") -Force -ErrorAction SilentlyContinue

Start-Sleep -Seconds 1

# 4. Final check: 8000 (FastAPI), 8501 (Streamlit), 3001 (IQMS-Web).
#    A port held by another program is reported, never killed.
Write-Host "`n--- Verification ---" -ForegroundColor Cyan
$ports = @(8000, 8501, 3001)
$activePorts = @()
foreach ($port in $ports) {
    $holders = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.OwningProcess -gt 0 } | Select-Object -ExpandProperty OwningProcess -Unique)
    if (-not $holders.Count) {
        Write-Host "[OK] Port $port is free" -ForegroundColor Green
        continue
    }
    foreach ($holder in $holders) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $holder" -ErrorAction SilentlyContinue
        $isCaddy = $proc.ExecutablePath -and $proc.ExecutablePath -match "^$caddyPath$"
        if ($targets.ContainsKey([int]$holder) -or (Test-IqmsProcess $proc) -or $isCaddy) {
            Write-Host "[!] Port $port is still in use by IQMS process PID $holder" -ForegroundColor Red
        } else {
            Write-Host "[!] Port $port is in use by another program, left running: $($proc.Name) (PID $holder)" -ForegroundColor Red
            Write-Host "      $($proc.CommandLine)" -ForegroundColor DarkGray
        }
    }
    $activePorts += $port
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
