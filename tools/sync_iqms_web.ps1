<#
.SYNOPSIS
    Compares (and optionally syncs) the two IQMS-Web copies:
    Queue-Management\IQMS-Web (the one start_all.ps1 runs) and the outer
    Retail-Wait-Prediction\IQMS-Web (where lojaine/dev web changes land).
.DESCRIPTION
    Only source files are considered: whatever git tracks or would track in each
    copy (node_modules, dist, .env and anything else gitignored is skipped).

    Default (no -From): report only. Exit code 0 = identical, 1 = drift.
    -From Outer : copy outer -> Queue-Management (e.g. after merging lojaine/dev)
    -From QM    : copy Queue-Management -> outer
    Files missing on the source side are reported, never deleted.
.EXAMPLE
    tools\sync_iqms_web.ps1
    tools\sync_iqms_web.ps1 -From Outer
#>

[CmdletBinding()]
param(
    [ValidateSet("Outer", "QM")]
    [string]$From
)

$ErrorActionPreference = "Stop"
$qmRoot    = Split-Path $PSScriptRoot -Parent
$outerRoot = Split-Path $qmRoot -Parent

function Get-WebFiles([string]$repo) {
    # Tracked + untracked-but-not-ignored, relative to the repo root
    $files = git -C $repo ls-files --cached --others --exclude-standard -- IQMS-Web
    if ($LASTEXITCODE -ne 0) { throw "git ls-files failed in $repo" }
    return @($files | Where-Object { Test-Path (Join-Path $repo $_) -PathType Leaf })
}

$qmFiles    = Get-WebFiles $qmRoot
$outerFiles = Get-WebFiles $outerRoot

$all = @($qmFiles + $outerFiles | Sort-Object -Unique)
$differs = @(); $onlyQM = @(); $onlyOuter = @()
foreach ($f in $all) {
    $q = Join-Path $qmRoot $f
    $o = Join-Path $outerRoot $f
    $inQ = $qmFiles -contains $f
    $inO = $outerFiles -contains $f
    if ($inQ -and $inO) {
        if ((Get-FileHash $q).Hash -ne (Get-FileHash $o).Hash) { $differs += $f }
    } elseif ($inQ) {
        $onlyQM += $f
    } else {
        $onlyOuter += $f
    }
}

Write-Host "IQMS-Web: $($qmFiles.Count) files in Queue-Management, $($outerFiles.Count) in outer"
foreach ($f in $differs)   { Write-Host "  [differs]       $f" -ForegroundColor Yellow }
foreach ($f in $onlyQM)    { Write-Host "  [only in QM]    $f" -ForegroundColor Yellow }
foreach ($f in $onlyOuter) { Write-Host "  [only in outer] $f" -ForegroundColor Yellow }

$drift = $differs.Count + $onlyQM.Count + $onlyOuter.Count
if ($drift -eq 0) {
    Write-Host "[OK] Both copies are identical." -ForegroundColor Green
    exit 0
}

if (-not $From) {
    Write-Host "[!] $drift file(s) drifted. Re-run with -From Outer or -From QM to sync." -ForegroundColor Yellow
    exit 1
}

$src = if ($From -eq "Outer") { $outerRoot } else { $qmRoot }
$dst = if ($From -eq "Outer") { $qmRoot } else { $outerRoot }
$toCopy = if ($From -eq "Outer") { $differs + $onlyOuter } else { $differs + $onlyQM }
foreach ($f in $toCopy) {
    $target = Join-Path $dst $f
    New-Item -ItemType Directory -Force -Path (Split-Path $target -Parent) | Out-Null
    Copy-Item -Path (Join-Path $src $f) -Destination $target -Force
    Write-Host "  [copied]        $f" -ForegroundColor Green
}
$left = if ($From -eq "Outer") { $onlyQM } else { $onlyOuter }
if ($left.Count) {
    Write-Host "[!] $($left.Count) file(s) exist only on the destination side; left untouched (delete by hand if intended)." -ForegroundColor Yellow
}
Write-Host "[OK] Copied $($toCopy.Count) file(s) from $From. Review with git diff, then commit." -ForegroundColor Green
