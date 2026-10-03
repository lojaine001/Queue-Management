<#
.SYNOPSIS
    Regression checks for the IQMS platform (run after any structural change).
.DESCRIPTION
    Default: static checks + live checks against whatever is already running.
    Safe on the production machine: nothing is stopped or restarted.

    Static
      S1  every .ps1 (QM root, tools\, outer forwarders) parses
      S2  QM launch scripts have no leftover "Queue-Management\" paths and
          every venv / entry script / web dependency they use exists
      S3  outer forwarders point at existing Queue-Management scripts
      S4  IQMS-Web copies (QM vs outer) are identical          (WARN on drift)
      S5  entry-point Python files compile with their own venv
      S6  unit tests (prediction\ + tests\): failing set equals
          tools\regression_baseline.txt
      S7  IQMS-Web production build succeeds
      S8  outer gitlink == Queue-Management HEAD, working trees clean (WARN)
    Live (skipped when the services are not running)
      L1  ports 8000 / 8501 / 3001 listening
      L2  /health, web root and /api proxy answer 200
      L3  ngrok tunnel up for 3001 and public /api/health answers 200
      L4  scheduler heartbeat < 120s, no last_error, last success < 15 min
      L5  /forecast updated within 15 min
      L6  camera pipelines running and snapshots refreshed < 120s

    -Runtime   also runs the stop/start cycle first: stop_all must free every
               port and the tunnel, then start_all brings everything back
               (add -SkipCameras to leave the cameras off). Leaves the system
               running.
    -UpdateBaseline  records the current failing unit tests as the new baseline.

    Exit code 0 = no FAIL. A report is written to logs\regression_<time>.txt.
.EXAMPLE
    tools\regression_test.ps1
    tools\regression_test.ps1 -Runtime
#>

[CmdletBinding()]
param(
    [switch]$Runtime,
    [switch]$SkipCameras,
    [switch]$UpdateBaseline
)

$ErrorActionPreference = "Continue"
$qm      = Split-Path $PSScriptRoot -Parent
$outer   = Split-Path $qm -Parent
$qm2     = Join-Path $qm "Queue-Management-System-v2-main\Queue-Management-System-v2-main"
$qmPy    = Join-Path $qm2 "venv\Scripts\python.exe"
$hdPy    = Join-Path $qm "Head-Detector\.venv\Scripts\python.exe"
$webDir  = Join-Path $qm "IQMS-Web"
$logDir  = Join-Path $qm "logs"
$baseline = Join-Path $PSScriptRoot "regression_baseline.txt"
$ngrokHeader = @{ "ngrok-skip-browser-warning" = "1" }
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$results = New-Object System.Collections.Generic.List[object]
function Add-Result([string]$id, [string]$status, [string]$detail) {
    $results.Add([pscustomobject]@{ Id = $id; Status = $status; Detail = $detail })
    $color = switch ($status) { "PASS" { "Green" } "WARN" { "Yellow" } "SKIP" { "Gray" } default { "Red" } }
    Write-Host ("[{0}] {1,-4} {2}" -f $status, $id, $detail) -ForegroundColor $color
}
function Test-Port([int]$port) { [bool](Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) }
function Get-Status([string]$url, [hashtable]$headers = @{}) {
    try { (Invoke-WebRequest $url -Headers $headers -UseBasicParsing -TimeoutSec 15).StatusCode } catch { 0 }
}
function Get-Tunnel {
    try {
        (Invoke-RestMethod "http://127.0.0.1:4040/api/tunnels" -TimeoutSec 3).tunnels |
            Where-Object { $_.config.addr -match ":3001$" } | Select-Object -First 1
    } catch { $null }
}
$launchScripts = @("start_all", "stop_all", "status_all", "run_cameras", "stop_cameras")

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "                IQMS REGRESSION CHECKS                    " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# ---------------------------------------------------------------- runtime cycle
if ($Runtime) {
    Write-Host "`n--- Runtime cycle (stop / start) ---" -ForegroundColor Cyan
    & (Join-Path $qm "stop_all.ps1") *>&1 | Out-Null
    $busy = @(8000, 8501, 3001 | Where-Object { Test-Port $_ })
    $ngrokLeft = @(Get-CimInstance Win32_Process -Filter "name = 'ngrok.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "\b3001\b" })
    if ($busy.Count -eq 0 -and $ngrokLeft.Count -eq 0 -and -not (Test-Path (Join-Path $logDir "pids.json"))) {
        Add-Result "R1" "PASS" "stop_all freed 8000/8501/3001, stopped the tunnel, removed pids.json"
    } else {
        Add-Result "R1" "FAIL" "after stop_all: busy ports [$($busy -join ',')], ngrok left $($ngrokLeft.Count)"
    }
    $startArgs = @{}
    if ($SkipCameras) { $startArgs["SkipCameras"] = $true }
    & (Join-Path $qm "start_all.ps1") @startArgs *>&1 | Out-Null
    $pids = $null
    try { $pids = Get-Content (Join-Path $logDir "pids.json") -Raw | ConvertFrom-Json } catch { }
    if ($pids -and $pids.api -and $pids.web) {
        Add-Result "R2" "PASS" "start_all launched: $(($pids.PSObject.Properties.Name) -join ', ')"
    } else {
        Add-Result "R2" "FAIL" "start_all did not write a complete logs\pids.json"
    }
    # let the scheduler write its first heartbeat and the cameras their first frames
    Start-Sleep -Seconds 20
}

# ---------------------------------------------------------------- static checks
Write-Host "`n--- Static checks ---" -ForegroundColor Cyan

# S1 parse
$ps1 = @(Get-ChildItem $qm -Filter *.ps1) + @(Get-ChildItem $PSScriptRoot -Filter *.ps1) +
       @($launchScripts | ForEach-Object { Get-Item (Join-Path $outer "$_.ps1") -ErrorAction SilentlyContinue })
$parseErrors = @()
foreach ($f in $ps1) {
    $tok = $null; $err = $null
    [System.Management.Automation.Language.Parser]::ParseFile($f.FullName, [ref]$tok, [ref]$err) | Out-Null
    if ($err.Count) { $parseErrors += "$($f.Name): $($err[0].Message)" }
}
if ($parseErrors.Count) { Add-Result "S1" "FAIL" ($parseErrors -join " | ") }
else { Add-Result "S1" "PASS" "$($ps1.Count) PowerShell scripts parse" }

# S2 paths
function Find-StalePaths([string]$dir) {
    @(Get-ChildItem -Path $dir -File | Where-Object { $_.Extension -in ".ps1", ".bat" } |
        Select-String -Pattern 'Queue-Management\\' | ForEach-Object { "$($_.Filename):$($_.LineNumber)" })
}
$leftover = Find-StalePaths $qm
$required = @($qmPy, $hdPy,
    (Join-Path $qm "run_scheduler.py"), (Join-Path $qm "Head-Detector\main.py"),
    (Join-Path $qm2 "api.py"), (Join-Path $qm2 "dashboard.py"), (Join-Path $qm2 "queue_management_v2.py"),
    (Join-Path $qm2 "ensemble_predict.py"), (Join-Path $qm "run_ensemble.bat"), (Join-Path $qm "run_ensemble_hidden.vbs"),
    (Join-Path $webDir "package.json"), (Join-Path $webDir "node_modules\.bin\vite.cmd"))
$missing = @($required | Where-Object { -not (Test-Path $_) } | ForEach-Object { $_.Replace("$qm\", "") })
if ($leftover.Count -or $missing.Count) {
    Add-Result "S2" "FAIL" ("leftover paths: [$($leftover -join ', ')] missing: [$($missing -join ', ')]")
} else { Add-Result "S2" "PASS" "no stale paths; $($required.Count) required files present" }

# S3 forwarders
$badFwd = @()
foreach ($n in $launchScripts) {
    $f = Join-Path $outer "$n.ps1"
    if (-not (Test-Path $f)) { $badFwd += "$n.ps1 missing"; continue }
    if (-not (Select-String -Path $f -SimpleMatch "Queue-Management\$n.ps1" -Quiet)) { $badFwd += "$n.ps1 does not forward" }
}
if ($badFwd.Count) { Add-Result "S3" "FAIL" ($badFwd -join ", ") }
else { Add-Result "S3" "PASS" "outer forwarders -> Queue-Management for $($launchScripts.Count) scripts" }

# S4 IQMS-Web parity
& (Join-Path $PSScriptRoot "sync_iqms_web.ps1") *>&1 | Out-Null
if ($LASTEXITCODE -eq 0) { Add-Result "S4" "PASS" "IQMS-Web copies identical" }
else { Add-Result "S4" "WARN" ("IQMS-Web copies drifted - run tools\sync_iqms_web.ps1 for details") }

# S5 compile entry points
$compileFail = @()
foreach ($pair in @(
        @($qmPy, (Join-Path $qm "run_scheduler.py")), @($qmPy, (Join-Path $qm2 "api.py")),
        @($qmPy, (Join-Path $qm2 "dashboard.py")), @($qmPy, (Join-Path $qm2 "queue_management_v2.py")),
        @($qmPy, (Join-Path $qm2 "ensemble_predict.py")), @($qmPy, (Join-Path $qm2 "push_alert_checker.py")),
        @($hdPy, (Join-Path $qm "Head-Detector\main.py")))) {
    & $pair[0] -m py_compile $pair[1] 2>$null
    if ($LASTEXITCODE -ne 0) { $compileFail += (Split-Path $pair[1] -Leaf) }
}
if ($compileFail.Count) { Add-Result "S5" "FAIL" "do not compile: $($compileFail -join ', ')" }
else { Add-Result "S5" "PASS" "7 entry-point files compile" }

# S6 unit tests vs baseline
Push-Location $qm
$testOut = ""
foreach ($suite in @("prediction", "tests")) {
    $testOut += & $qmPy -m unittest discover -s $suite -t . -p "test_*.py" 2>&1 | Out-String
}
Pop-Location
$ran = 0
foreach ($m in [regex]::Matches($testOut, "Ran (\d+) tests?")) { $ran += [int]$m.Groups[1].Value }
$failing = @([regex]::Matches($testOut, "(?m)^(?:FAIL|ERROR): (.+)$") | ForEach-Object { $_.Groups[1].Value.Trim() } | Sort-Object -Unique)
if ($UpdateBaseline) {
    $failing | Set-Content -Path $baseline -Encoding UTF8
    Add-Result "S6" "PASS" "baseline updated: $ran tests, $($failing.Count) known failures"
} elseif ($ran -eq 0) {
    Add-Result "S6" "FAIL" "unit tests did not run"
} else {
    $known = @()
    if (Test-Path $baseline) { $known = @(Get-Content $baseline | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) }
    $new   = @($failing | Where-Object { $known -notcontains $_ })
    $fixed = @($known | Where-Object { $failing -notcontains $_ })
    if ($new.Count) { Add-Result "S6" "FAIL" "$ran tests, NEW failures: $($new -join '; ')" }
    elseif ($fixed.Count) { Add-Result "S6" "PASS" "$ran tests, no new failures; now passing: $($fixed -join '; ') (run -UpdateBaseline)" }
    else { Add-Result "S6" "PASS" "$ran tests, failures unchanged vs baseline ($($known.Count) known)" }
}

# S7 web build
$buildDir = Join-Path $env:TEMP "iqms-regression-build"
Push-Location $webDir
$buildOut = & cmd.exe /c "npx vite build --outDir `"$buildDir`" --emptyOutDir 2>&1"
$buildOk = ($LASTEXITCODE -eq 0)
Pop-Location
Remove-Item $buildDir -Recurse -Force -ErrorAction SilentlyContinue
if ($buildOk) { Add-Result "S7" "PASS" "IQMS-Web production build succeeds" }
else { Add-Result "S7" "FAIL" "vite build failed: $(($buildOut | Select-Object -Last 3) -join ' ')" }

# S8 git consistency
$gitlink = (git -C $outer ls-tree HEAD Queue-Management 2>$null) -split "\s+" | Select-Object -Index 2
$qmHead  = git -C $qm rev-parse HEAD 2>$null
$dirty = @()
foreach ($repo in @($outer, $qm)) {
    $changes = git -C $repo status --porcelain 2>$null
    if ($changes) { $dirty += (Split-Path $repo -Leaf) }
}
if ($gitlink -ne $qmHead) { Add-Result "S8" "WARN" "outer gitlink $($gitlink.Substring(0,7)) != Queue-Management HEAD $($qmHead.Substring(0,7)) (commit the outer repo)" }
elseif ($dirty.Count) { Add-Result "S8" "WARN" "gitlink in sync; uncommitted changes in: $($dirty -join ', ')" }
else { Add-Result "S8" "PASS" "gitlink in sync, working trees clean" }

# ---------------------------------------------------------------- live checks
Write-Host "`n--- Live checks ---" -ForegroundColor Cyan
$up = @{ 8000 = (Test-Port 8000); 8501 = (Test-Port 8501); 3001 = (Test-Port 3001) }
if (-not ($up[8000] -or $up[8501] -or $up[3001])) {
    Add-Result "L*" "SKIP" "services not running (start_all.ps1, or re-run with -Runtime)"
} else {
    $down = @($up.Keys | Where-Object { -not $up[$_] } | Sort-Object)
    if ($down.Count) { Add-Result "L1" "FAIL" "not listening: $($down -join ', ')" }
    else { Add-Result "L1" "PASS" "8000, 8501, 3001 listening" }

    $codes = [ordered]@{
        "api /health"   = (Get-Status "http://localhost:8000/health")
        "web /"         = (Get-Status "http://localhost:3001/")
        "web /api/health" = (Get-Status "http://localhost:3001/api/health")
    }
    $bad = @($codes.Keys | Where-Object { $codes[$_] -ne 200 } | ForEach-Object { "$_=$($codes[$_])" })
    if ($bad.Count) { Add-Result "L2" "FAIL" ($bad -join ", ") } else { Add-Result "L2" "PASS" "api, web and /api proxy return 200" }

    $tunnel = Get-Tunnel
    if (-not $tunnel) { Add-Result "L3" "WARN" "no ngrok tunnel for 3001 (started with -NoTunnel?)" }
    else {
        $pub = Get-Status "$($tunnel.public_url)/api/health" $ngrokHeader
        if ($pub -eq 200) { Add-Result "L3" "PASS" "$($tunnel.public_url) -> 3001, public /api/health 200" }
        else { Add-Result "L3" "FAIL" "$($tunnel.public_url) public /api/health returned $pub" }
    }

    $schedFile = Join-Path $qm ".runtime\scheduler.json"
    if (-not (Test-Path $schedFile)) { Add-Result "L4" "FAIL" "no .runtime\scheduler.json" }
    else {
        $s = Get-Content $schedFile -Raw | ConvertFrom-Json
        $hbAge = ([datetime]::UtcNow - ([datetime]::Parse($s.heartbeat)).ToUniversalTime()).TotalSeconds
        $okAge = if ($s.last_success) { ([datetime]::UtcNow - ([datetime]::Parse($s.last_success)).ToUniversalTime()).TotalMinutes } else { 1e9 }
        if ($hbAge -gt 120) { Add-Result "L4" "FAIL" "scheduler heartbeat $([math]::Round($hbAge))s old" }
        elseif ($s.last_error) { Add-Result "L4" "FAIL" "scheduler last_error: $($s.last_error)" }
        elseif ($okAge -gt 15) { Add-Result "L4" "WARN" "heartbeat ok, but last success $([math]::Round($okAge)) min ago (a run may still be in progress)" }
        else { Add-Result "L4" "PASS" "heartbeat $([math]::Round($hbAge))s, last success $([math]::Round($okAge,1)) min ago" }
    }

    try {
        $f = Invoke-RestMethod "http://localhost:8000/forecast" -TimeoutSec 15
        $fAge = ([datetime]::UtcNow - ([datetime]::Parse($f.updated_at)).ToUniversalTime()).TotalMinutes
        if ($fAge -le 15) { Add-Result "L5" "PASS" "forecast updated $([math]::Round($fAge,1)) min ago" }
        else { Add-Result "L5" "WARN" "forecast is $([math]::Round($fAge)) min old (outside opening hours?)" }
    } catch { Add-Result "L5" "FAIL" "/forecast failed: $($_.Exception.Message)" }

    $py = Get-CimInstance Win32_Process -Filter "name = 'python.exe'" -ErrorAction SilentlyContinue
    $pipeA = [bool]($py | Where-Object { $_.CommandLine -match "Head-Detector" -and $_.CommandLine -match "main\.py" })
    $pipeB = [bool]($py | Where-Object { $_.CommandLine -match "queue_management_v2\.py" })
    if (-not ($pipeA -or $pipeB)) { Add-Result "L6" "SKIP" "camera pipelines not running" }
    else {
        $stale = @(Get-ChildItem (Join-Path $qm2 "snapshots\latest_*.jpg") -ErrorAction SilentlyContinue |
            Where-Object { ((Get-Date) - $_.LastWriteTime).TotalSeconds -gt 120 } | ForEach-Object { $_.Name })
        if (-not ($pipeA -and $pipeB)) { Add-Result "L6" "FAIL" "pipeline A running=$pipeA, B running=$pipeB" }
        elseif ($stale.Count) { Add-Result "L6" "FAIL" "snapshots older than 120s: $($stale -join ', ')" }
        else { Add-Result "L6" "PASS" "pipelines A and B running, snapshots fresh" }
    }
}

# ---------------------------------------------------------------- summary
$fails = @($results | Where-Object Status -eq "FAIL").Count
$warns = @($results | Where-Object Status -eq "WARN").Count
$report = Join-Path $logDir ("regression_{0:yyyyMMdd_HHmmss}.txt" -f (Get-Date))
$results | Format-Table -AutoSize | Out-String -Width 300 | Set-Content -Path $report -Encoding UTF8
Write-Host "`n==========================================================" -ForegroundColor Cyan
$summaryColor = if ($fails) { "Red" } elseif ($warns) { "Yellow" } else { "Green" }
Write-Host ("  {0} PASS, {1} WARN, {2} FAIL   (report: {3})" -f @($results | Where-Object Status -eq "PASS").Count, $warns, $fails, $report) -ForegroundColor $summaryColor
Write-Host "==========================================================" -ForegroundColor Cyan
if ($fails) { exit 1 } else { exit 0 }
