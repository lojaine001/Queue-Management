<#
.SYNOPSIS
    Restarts the background services that die silently (scheduler, push alert checker).
.DESCRIPTION
    Run every 5 minutes by the Windows task IQMS_Watchdog (through tools\run_hidden.vbs).
    Does nothing unless the system is meant to be running (FastAPI listening on 8000),
    so a deliberate stop_all is respected.

    Scheduler: restarted only when its recorded process is gone AND it was not stopped
    on purpose (a dashboard Stop leaves status "stopped" or a matching stop.json).
    It restarts with the last interval/days/source the scheduler was running with.
    Push alert checker: restarted when no instance is running.

    Every action is appended to logs\watchdog.log; quiet runs write nothing.
#>

$ErrorActionPreference = "Stop"
$qm      = Split-Path $PSScriptRoot -Parent
$qm2     = Join-Path $qm "Queue-Management-System-v2-main\Queue-Management-System-v2-main"
$py      = Join-Path $qm2 "venv\Scripts\python.exe"
$logDir  = Join-Path $qm "logs"
$runtime = Join-Path $qm ".runtime"
$pidFile = Join-Path $logDir "pids.json"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Write-Log([string]$msg) {
    Add-Content -Path (Join-Path $logDir "watchdog.log") -Value ("{0:yyyy-MM-dd HH:mm:ss} {1}" -f (Get-Date), $msg) -Encoding UTF8
}
function Read-Json([string]$path) {
    try { if (Test-Path $path) { return Get-Content $path -Raw | ConvertFrom-Json } } catch { }
    return $null
}
function Save-Pid([string]$name, [int]$procId) {
    $pids = Read-Json $pidFile
    if (-not $pids) { $pids = [pscustomobject]@{} }
    $pids | Add-Member -NotePropertyName $name -NotePropertyValue $procId -Force
    $pids | ConvertTo-Json | Set-Content -Path $pidFile -Encoding UTF8
}

try {
    # System deliberately stopped (stop_all) or not started yet: leave it alone.
    if (-not (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) { exit 0 }

    # --- Scheduler -----------------------------------------------------------
    $state = Read-Json (Join-Path $runtime "scheduler.json")
    $stop  = Read-Json (Join-Path $runtime "stop.json")
    $alive = $false
    if ($state -and $state.pid) {
        $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$state.pid)" -ErrorAction SilentlyContinue
        $alive = [bool]($proc -and $proc.CommandLine -match "run_scheduler\.py")
    }
    $stoppedOnPurpose = $state -and ($state.status -eq "stopped" -or ($stop -and $stop.owner -eq $state.owner))
    if (-not $alive -and -not $stoppedOnPurpose) {
        $interval = if ($state -and $state.interval_min) { [int]$state.interval_min } else { 3 }
        $days     = if ($state -and $state.days) { [int]$state.days } else { 60 }
        $source   = if ($state -and $state.source) { $state.source } else { "REAL" }
        $p = Start-Process -FilePath $py -ArgumentList "run_scheduler.py --interval $interval --days $days --source $source" `
            -WorkingDirectory $qm -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $logDir "scheduler_watchdog.log") `
            -RedirectStandardError (Join-Path $logDir "scheduler_watchdog_err.log")
        Save-Pid "scheduler" $p.Id
        $last = if ($state) { "last status '$($state.status)', last heartbeat $($state.heartbeat)" } else { "no scheduler.json" }
        Write-Log "RESTARTED scheduler (PID $($p.Id), every ${interval}m, ${days}d, $source) - previous process gone; $last"
    } elseif ($alive -and $state.heartbeat) {
        $age = ([datetime]::UtcNow - ([datetime]::Parse($state.heartbeat)).ToUniversalTime()).TotalMinutes
        if ($age -gt 5) { Write-Log "WARN scheduler PID $($state.pid) alive but heartbeat is $([math]::Round($age)) min old (hung?)" }
    }

    # --- Push alert checker ----------------------------------------------------
    $push = @(Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "push_alert_checker\.py" -and $_.CommandLine -match "venv" })
    if ($push.Count -eq 0) {
        $p = Start-Process -FilePath $py -ArgumentList "-u push_alert_checker.py" -WorkingDirectory $qm2 -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $logDir "push_alerts_watchdog.log") `
            -RedirectStandardError (Join-Path $logDir "push_alerts_watchdog_err.log")
        Save-Pid "push_alerts" $p.Id
        Write-Log "RESTARTED push alert checker (PID $($p.Id)) - no instance was running"
    } elseif ($push.Count -gt 1) {
        Write-Log "WARN $($push.Count) push alert checkers running - phones get duplicate alerts (run stop_all/start_all)"
    }
} catch {
    Write-Log "ERROR watchdog failed: $($_.Exception.Message)"
    exit 1
}
