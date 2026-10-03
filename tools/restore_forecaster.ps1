<#
.SYNOPSIS
    Puts back the previous ensemble_predict.py (the forecaster) in one command.
.DESCRIPTION
    Restores Queue-Management-System-v2-main\...\ensemble_predict.py from a git commit
    (default: 4b42e14, the last version before the Sep 27 refactor was merged in on
    2026-10-03) and runs the unit tests. The scheduler launches ensemble_predict.py as a
    fresh process on every run, so the restored version is used from the next run on --
    no restart needed.

    The old version keeps working with the current model files (it uses the flat model
    files and retrains on every run); the forecast_runs table the new version adds is
    simply left unused. To make the restore permanent, commit the restored file.
.EXAMPLE
    tools\restore_forecaster.ps1
    tools\restore_forecaster.ps1 -Commit <sha>
#>

[CmdletBinding()]
param([string]$Commit = "4b42e14")

$ErrorActionPreference = "Stop"
$qm   = Split-Path $PSScriptRoot -Parent
$rel  = "Queue-Management-System-v2-main/Queue-Management-System-v2-main/ensemble_predict.py"
$file = Join-Path $qm $rel

git -C $qm cat-file -e "${Commit}:$rel" 2>$null
if ($LASTEXITCODE -ne 0) { throw "Commit $Commit has no $rel" }
Copy-Item $file "$file.before_restore" -Force
# git restore writes the working copy only (nothing staged) and applies the repo's line-ending rules
git -C $qm restore --source $Commit --worktree -- $rel
if ($LASTEXITCODE -ne 0) { throw "git restore failed" }
Write-Host "[OK] ensemble_predict.py restored from $Commit (previous file kept as ensemble_predict.py.before_restore)" -ForegroundColor Green

$py = Join-Path $qm "Queue-Management-System-v2-main\Queue-Management-System-v2-main\venv\Scripts\python.exe"
& $py -m py_compile $file
if ($LASTEXITCODE -ne 0) { throw "Restored file does not compile" }
Write-Host "[OK] compiles. The next scheduler run (within its interval) uses the restored version." -ForegroundColor Green
Write-Host "     Check with: tools\regression_test.ps1   |   make permanent: git commit $rel" -ForegroundColor Gray
