<#
.SYNOPSIS
    Sets up IQMS on a fresh Windows machine from a clone of this repository.

.DESCRIPTION
    Safe to run again: every step is skipped when it is already done.
      1. finds Python 3.10
      2. creates the venvs and installs the pinned packages
           Head-Detector\.venv                                    checkout pipeline
           Queue-Management-System-v2-main\...\venv               entrance, API, dashboard, forecast
           simulator\.venv                                        simulator (skip with -SkipSimulator)
      3. creates .env from .env.example (edit DB_PASSWORD afterwards) and adds a random
         IQMS_API_KEY if it has none (the value is never printed)
      4. installs the web app (npm ci in IQMS-Web) and downloads Caddy, the web server
         that serves it in production (tools\bin\caddy.exe, pinned version, SHA-512 checked)
      5. creates the database from .env if it does not exist and checks TimescaleDB
         (the apps create their tables themselves on first start)
      6. downloads missing detector models from Hugging Face (tools\models.json)
      7. with -RegisterTasks: the daily 07:00 regression check task

    Not done here (see README.md): installing Python, PostgreSQL + TimescaleDB, Node.js,
    ngrok and the NPU driver, and the camera settings in config.yml.

.PARAMETER Python
    Path to a Python 3.10 python.exe. Default: found automatically.
.PARAMETER Exact
    Install the complete production package sets (requirements.lock.txt) instead of only
    what the code imports at production's versions.
.PARAMETER ModelsRepo
    Hugging Face repo holding the models, e.g. "your-account/iqms-models". Default: the
    IQMS_MODELS_HF_REPO environment variable, then huggingface_repo in tools\models.json.
    For a private repo set HF_TOKEN.
.PARAMETER DryRun
    Only print what would be done.

.EXAMPLE
    .\tools\install.ps1
.EXAMPLE
    .\tools\install.ps1 -ModelsRepo your-account/iqms-models -RegisterTasks
#>
[CmdletBinding()]
param(
    [string]$Python,
    [switch]$Exact,
    [switch]$SkipSimulator,
    [switch]$SkipWeb,
    [switch]$RegisterTasks,
    [string]$ModelsRepo,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$qmDir = Join-Path $root "Queue-Management-System-v2-main\Queue-Management-System-v2-main"
$problems = @()

function Step([string]$text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Ok([string]$text) { Write-Host "   [OK] $text" -ForegroundColor Green }
function Todo([string]$text) { Write-Host "   [!]  $text" -ForegroundColor Yellow; $script:problems += $text }
function Run([string]$what, [scriptblock]$action) {
    if ($DryRun) { Write-Host "   would: $what" -ForegroundColor DarkGray; return }
    Write-Host "   $what" -ForegroundColor Gray
    $global:LASTEXITCODE = 0
    & $action
    if ($LASTEXITCODE) { throw "Failed ($LASTEXITCODE): $what" }
}

function Read-DotEnv([string]$path) {
    $values = @{}
    if (Test-Path $path) {
        foreach ($line in Get-Content $path) {
            if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { $values[$Matches[1]] = $Matches[2].Trim('"', "'") }
        }
    }
    return $values
}

# 1. Python 3.10 -------------------------------------------------------------------
Step "Python 3.10"
$candidates = @()
if ($Python) { $candidates += $Python }
try { $candidates += (& py -3.10 -c "import sys; print(sys.executable)" 2>$null) } catch { }
$candidates += "C:\Program Files\Python310\python.exe", (Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe")
$py = $null
foreach ($c in $candidates | Where-Object { $_ -and (Test-Path $_) }) {
    if ((& $c -c "import sys; print('%d.%d' % sys.version_info[:2])") -eq "3.10") { $py = $c; break }
}
if (-not $py) { throw "Python 3.10 not found. Install it (python.org, 64-bit) or pass -Python <path to python.exe>." }
Ok "$py"

# 2. Virtual environments ----------------------------------------------------------
$venvs = @(
    @{ Name = "checkout pipeline"; Dir = (Join-Path $root "Head-Detector"); Venv = ".venv" },
    @{ Name = "entrance, API, dashboard, forecast"; Dir = $qmDir; Venv = "venv" }
)
if (-not $SkipSimulator) { $venvs += @{ Name = "simulator"; Dir = (Join-Path $root "simulator"); Venv = ".venv" } }

foreach ($v in $venvs) {
    Step "venv: $($v.Name)"
    $venvDir = Join-Path $v.Dir $v.Venv
    $venvPy = Join-Path $venvDir "Scripts\python.exe"
    if (Test-Path $venvPy) { Ok "exists: $venvDir" }
    else { Run "create $venvDir" { & $py -m venv $venvDir } }
    $req = Join-Path $v.Dir "requirements.txt"
    $lock = Join-Path $v.Dir "requirements.lock.txt"
    if ($Exact) { Run "pip install -r requirements.lock.txt" { & $venvPy -m pip install -r $lock } }
    else { Run "pip install -r requirements.txt -c requirements.lock.txt" { & $venvPy -m pip install -r $req -c $lock } }
}

# 3. .env ---------------------------------------------------------------------------
Step ".env"
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) { Ok "exists" }
else {
    Run "copy .env.example -> .env" { Copy-Item (Join-Path $root ".env.example") $envFile }
    Todo "Edit .env: set DB_PASSWORD (and the other settings) for this machine."
}

# 3b. API key for the endpoints that change data (never printed) ---------------------
Step "API key (IQMS_API_KEY in .env)"
if ((Test-Path $envFile) -and ((Get-Content $envFile) -match '^\s*IQMS_API_KEY\s*=\s*\S+')) { Ok "set" }
elseif (Test-Path $envFile) {
    Run "add a random IQMS_API_KEY to .env" {
        $bytes = New-Object byte[] 32
        [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
        $key = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        $text = [IO.File]::ReadAllText($envFile)
        if ($text -match '(?m)^\s*IQMS_API_KEY\s*=\s*$') { $text = $text -replace '(?m)^\s*IQMS_API_KEY\s*=\s*$', "IQMS_API_KEY=$key" }
        else { if ($text -and -not $text.EndsWith("`n")) { $text += "`r`n" }; $text += "IQMS_API_KEY=$key`r`n" }
        [IO.File]::WriteAllText($envFile, $text, (New-Object System.Text.UTF8Encoding $false))
    }
    if (-not $DryRun) { Ok "added (value not shown; restart the API and Caddy to use it)" }
}

# 4. Web app ------------------------------------------------------------------------
if (-not $SkipWeb) {
    Step "web app (IQMS-Web)"
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) { Todo "npm not found: install Node.js, then run 'npm ci' in IQMS-Web." }
    elseif (Test-Path (Join-Path $root "IQMS-Web\node_modules\.bin\vite.cmd")) { Ok "node_modules present" }
    else { Run "npm ci in IQMS-Web" { Push-Location (Join-Path $root "IQMS-Web"); try { npm ci } finally { Pop-Location } } }

    # Caddy serves the built app in production (start_all.ps1); pinned and checksum-verified.
    $caddyVersion = "2.11.7"
    $caddyExe = Join-Path $root "tools\bin\caddy.exe"
    $haveCaddy = (Test-Path $caddyExe) -and ((& $caddyExe version 2>$null) -match "^v$([regex]::Escape($caddyVersion))\b")
    if ($haveCaddy) { Ok "Caddy $caddyVersion present" }
    else {
        Run "download Caddy $caddyVersion to tools\bin" {
            $base = "https://github.com/caddyserver/caddy/releases/download/v$caddyVersion"
            $zipName = "caddy_${caddyVersion}_windows_amd64.zip"
            $tmp = Join-Path ([IO.Path]::GetTempPath()) ("caddy_" + [guid]::NewGuid().ToString("N"))
            New-Item -ItemType Directory -Force $tmp | Out-Null
            try {
                $zip = Join-Path $tmp $zipName
                Invoke-WebRequest -Uri "$base/$zipName" -OutFile $zip -UseBasicParsing
                $sums = (Invoke-WebRequest -Uri "$base/caddy_${caddyVersion}_checksums.txt" -UseBasicParsing).Content
                if ($sums -is [byte[]]) { $sums = [Text.Encoding]::UTF8.GetString($sums) }
                $expected = ($sums -split "`n" | Where-Object { $_ -match [regex]::Escape($zipName) } | Select-Object -First 1) -replace '\s.*$', ''
                $actual = (Get-FileHash $zip -Algorithm SHA512).Hash.ToLower()
                if (-not $expected -or $actual -ne $expected.Trim().ToLower()) { throw "Caddy download does not match the published SHA-512 checksum." }
                Expand-Archive $zip -DestinationPath $tmp -Force
                New-Item -ItemType Directory -Force (Split-Path $caddyExe -Parent) | Out-Null
                Copy-Item (Join-Path $tmp "caddy.exe") $caddyExe -Force
            } finally { Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue }
        }
        if (-not $DryRun) { Ok "Caddy $caddyVersion installed (checksum verified)" }
    }
}

# 5. Database -----------------------------------------------------------------------
Step "database"
$cfg = Read-DotEnv $envFile
$dbName = if ($cfg.DB_NAME) { $cfg.DB_NAME } else { "iqms" }
$psql = Get-ChildItem "C:\Program Files\PostgreSQL\*\bin\psql.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -First 1
if (-not $psql) { Todo "PostgreSQL not found (C:\Program Files\PostgreSQL). Install PostgreSQL 17 and TimescaleDB, then run this script again." }
elseif (-not $cfg.DB_PASSWORD -or $cfg.DB_PASSWORD -eq "change-me") { Todo "Set DB_PASSWORD in .env, then run this script again to create the database." }
else {
    $env:PGPASSWORD = $cfg.DB_PASSWORD
    $pgArgs = @("-h", $(if ($cfg.DB_HOST) { $cfg.DB_HOST } else { "localhost" }), "-p", $(if ($cfg.DB_PORT) { $cfg.DB_PORT } else { "5432" }),
                "-U", $(if ($cfg.DB_USER) { $cfg.DB_USER } else { "postgres" }), "-tA")
    $exists = & $psql.FullName @pgArgs -d postgres -c "SELECT 1 FROM pg_database WHERE datname = '$dbName';"
    if ($LASTEXITCODE) { Todo "Cannot connect to PostgreSQL with the .env settings." }
    else {
        if ($exists -eq "1") { Ok "database '$dbName' exists" }
        else { Run "create database $dbName" { & $psql.FullName @pgArgs -d postgres -c "CREATE DATABASE $dbName;" } }
        $ts = & $psql.FullName @pgArgs -d postgres -c "SELECT 1 FROM pg_available_extensions WHERE name = 'timescaledb';"
        if ($ts -eq "1") { Ok "TimescaleDB available" }
        else { Todo "TimescaleDB is not installed in PostgreSQL. Install it; the apps need time_bucket and hypertables." }

        # Tables come from the migrations in db\ (db\README.md), not from the apps.
        if ($ts -eq "1") {
            $alembic = @((Join-Path $qmDir "venv\Scripts\python.exe"), "-m", "alembic", "-c", (Join-Path $root "db\alembic.ini"))
            # "marked" + "has tables": 01 = tables from before db\ but no version (production before the switch).
            $state = & $psql.FullName @pgArgs -d $dbName -c "SELECT (to_regclass('public.alembic_version') IS NOT NULL)::int::text || (to_regclass('public.entrance_events') IS NOT NULL)::int::text;"
            if ($state -eq "01") {
                Todo "Database '$dbName' has tables but no migration version. Mark it once (changes no table): cd `"$root`"; & `"$($alembic[0])`" -m alembic -c `"$($alembic[4])`" stamp 0001_baseline"
            }
            else {
                Push-Location $root
                try { Run "database migrations (alembic upgrade head)" { & $alembic[0] $alembic[1..($alembic.Count - 1)] upgrade head } }
                finally { Pop-Location }
            }
        }
        else {
            Push-Location $root
            try { Run "database migrations (alembic upgrade head)" { & $alembic[0] $alembic[1..($alembic.Count - 1)] upgrade head } }
            finally { Pop-Location }
        }
    }
    Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue
}

# 6. Models -------------------------------------------------------------------------
Step "detector models"
$manifest = Get-Content (Join-Path $PSScriptRoot "models.json") -Raw | ConvertFrom-Json
$repo = if ($ModelsRepo) { $ModelsRepo } elseif ($env:IQMS_MODELS_HF_REPO) { $env:IQMS_MODELS_HF_REPO } else { $manifest.huggingface_repo }
foreach ($m in $manifest.models) {
    $target = Join-Path $root $m.path
    $name = Split-Path $m.path -Leaf
    if (-not (Test-Path $target)) {
        if (-not $m.required) { Write-Host "   [-]  optional, not present: $name" -ForegroundColor DarkGray; continue }
        if (-not $repo) { Todo "Missing $name ($($m.used_by)). Copy it to $($m.path), or pass -ModelsRepo <huggingface repo>."; continue }
        $url = "https://huggingface.co/$repo/resolve/main/$name"
        Run "download $name from $repo" {
            New-Item -ItemType Directory -Force (Split-Path $target -Parent) | Out-Null
            $headers = @{}
            if ($env:HF_TOKEN) { $headers["Authorization"] = "Bearer $env:HF_TOKEN" }
            Invoke-WebRequest -Uri $url -OutFile $target -Headers $headers -UseBasicParsing
        }
    }
    if ((Test-Path $target) -and $m.sha256) {
        $hash = (Get-FileHash $target -Algorithm SHA256).Hash.ToLower()
        if ($hash -ne $m.sha256.ToLower()) { Todo "$name does not match the expected sha256 (wrong or damaged file)." }
        else { Ok "$name (sha256 checked)" }
    }
    elseif (Test-Path $target) { Ok $name }
}

# 7. Scheduled task -----------------------------------------------------------------
if ($RegisterTasks) {
    Step "scheduled task IQMS_RegressionCheck (daily 07:00)"
    if (Get-ScheduledTask -TaskName "IQMS_RegressionCheck" -ErrorAction SilentlyContinue) { Ok "already registered" }
    else {
        Run "register IQMS_RegressionCheck" {
            $action = New-ScheduledTaskAction -Execute "wscript.exe" `
                -Argument ('//B //Nologo "{0}" "{1}"' -f (Join-Path $PSScriptRoot "run_hidden.vbs"), (Join-Path $PSScriptRoot "regression_test.ps1"))
            $trigger = New-ScheduledTaskTrigger -Daily -At 7am
            Register-ScheduledTask -TaskName "IQMS_RegressionCheck" -Action $action -Trigger $trigger `
                -Description "IQMS daily regression check (tools\regression_test.ps1)" | Out-Null
        }
    }
}

# Summary ---------------------------------------------------------------------------
Write-Host ""
if ($problems.Count) {
    Write-Host "Done, with $($problems.Count) thing(s) left to do:" -ForegroundColor Yellow
    $problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
} else {
    Write-Host "Done. Start everything with .\start_all.ps1" -ForegroundColor Green
}
