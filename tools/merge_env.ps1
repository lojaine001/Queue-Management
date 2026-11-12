<#
.SYNOPSIS
    Merges any extra .env files in the project into the one at the project root.

.DESCRIPTION
    python-dotenv (find_dotenv) uses the FIRST .env it finds, walking up from a
    program's folder. A .env inside a subfolder therefore hides the root .env
    from every program started there, and settings that exist only in the root
    file silently do not apply to them.

    Without -Apply this only reports, by setting NAME (values are never shown):
      add        only in the extra file: will be copied into the root .env
      same       in both with the same value: nothing to do
      CONFLICT   in both with different values: you decide; nothing is changed

    With -Apply (refused while there is a conflict):
      1. backs up the root .env and each extra .env to $HOME\iqms_audit\
      2. appends the "add" lines to the root .env, unchanged
      3. moves each extra .env out of the project (into $HOME\iqms_audit\)
    Restart afterwards (.\start_all.ps1) so every program reads the root .env.

.EXAMPLE
    .\tools\merge_env.ps1            # report only
.EXAMPLE
    .\tools\merge_env.ps1 -Apply
#>
[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$BackupDir = (Join-Path $HOME "iqms_audit")
)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$rootEnv = Join-Path $root ".env"

function Read-EnvLines([string]$path) {
    $map = [ordered]@{}
    foreach ($line in [IO.File]::ReadAllLines($path)) {
        if ($line -match '^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') {
            $map[$Matches[1]] = @{ Value = $Matches[2].Trim('"', "'"); Line = $line.Trim() }
        }
    }
    return $map
}

function Find-ExtraEnv([string]$dir) {
    Get-ChildItem -Path $dir -Filter ".env" -File -Recurse -Force -Depth 4 -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName -ne $rootEnv -and $_.FullName -notmatch '\\(node_modules|venv|\.venv|\.git)\\' }
}

if (-not (Test-Path $rootEnv)) { throw "No .env at the project root ($rootEnv). Create it from .env.example first." }
$extras = @(Find-ExtraEnv $root)
if (-not $extras.Count) { Write-Host "Only one .env (the project root). Nothing to merge." -ForegroundColor Green; exit 0 }

$rootMap = Read-EnvLines $rootEnv
$plan = @()
foreach ($file in $extras) {
    $rel = $file.FullName.Substring($root.Length + 1)
    $map = Read-EnvLines $file.FullName
    Write-Host "`n$rel ($($map.Count) settings)" -ForegroundColor Cyan
    foreach ($name in $map.Keys) {
        if (-not $rootMap.Contains($name)) { $action = "add" }
        elseif ($rootMap[$name].Value -eq $map[$name].Value) { $action = "same" }
        else { $action = "CONFLICT" }
        $color = @{ add = "Green"; same = "Gray"; CONFLICT = "Red" }[$action]
        Write-Host ("   {0,-9} {1}" -f $action, $name) -ForegroundColor $color
        $plan += [pscustomobject]@{ File = $file.FullName; Rel = $rel; Name = $name; Action = $action; Line = $map[$name].Line }
    }
}

$adds = @($plan | Where-Object Action -eq "add")
$conflicts = @($plan | Where-Object Action -eq "CONFLICT")
# The same name added from two extra files with different values is a conflict too.
foreach ($group in ($adds | Group-Object Name | Where-Object Count -gt 1)) {
    if (@($group.Group.Line | Select-Object -Unique).Count -gt 1) { $conflicts += $group.Group[0] }
}

Write-Host "`n$($adds.Count) to add, $(@($plan | Where-Object Action -eq 'same').Count) already the same, $($conflicts.Count) conflict(s)."
if ($conflicts.Count) {
    Write-Host "Resolve each CONFLICT first: keep the right value in the root .env and delete that line from the other file. Nothing was changed." -ForegroundColor Red
    exit 1
}
if (-not $Apply) { Write-Host "Report only. Run again with -Apply to merge." -ForegroundColor Yellow; exit 0 }

$backupDir = $BackupDir
New-Item -ItemType Directory -Force $backupDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
Copy-Item $rootEnv (Join-Path $backupDir "root_env_before_merge_$stamp.env")

$text = [IO.File]::ReadAllText($rootEnv)
if ($text -and -not $text.EndsWith("`n")) { $text += "`r`n" }
foreach ($group in ($adds | Group-Object Rel)) {
    $text += "`r`n# Moved from $($group.Name) on $(Get-Date -Format 'yyyy-MM-dd') (tools\merge_env.ps1)`r`n"
    foreach ($item in ($group.Group | Sort-Object Name -Unique)) { $text += "$($item.Line)`r`n" }
}
[IO.File]::WriteAllText($rootEnv, $text, (New-Object System.Text.UTF8Encoding $false))

foreach ($file in $extras) {
    $safeName = ($file.FullName.Substring($root.Length + 1) -replace '[\\/:]', '_')
    Move-Item $file.FullName (Join-Path $backupDir "moved_${safeName}_$stamp.env")
}
Write-Host "Merged. Backups and the moved file(s) are in $backupDir. Now restart: .\start_all.ps1" -ForegroundColor Green
