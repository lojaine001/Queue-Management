<#
.SYNOPSIS
    Saves the API's answers for past days (which do not change), to check that a code
    change did not change what the API returns.

.DESCRIPTION
    Before a change:   .\tools\api_snapshot.ps1 -Out $HOME\iqms_audit\api_before
    After restarting:  .\tools\api_snapshot.ps1 -Out $HOME\iqms_audit\api_after -CompareWith $HOME\iqms_audit\api_before

    Only read-only endpoints are called, for yesterday, last week and last month, plus
    a few deliberately invalid requests (their error answers must not change either).
    Timestamps such as "time" or "as_of" are blanked before comparing.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Out,
    [string]$CompareWith,
    [string]$Base = "http://localhost:8000",
    [string]$Day = (Get-Date).AddDays(-1).ToString("yyyy-MM-dd")
)

$ErrorActionPreference = "Stop"
$week = ([datetime]$Day).AddDays(-7).ToString("yyyy-MM-dd")
$month = ([datetime]$Day).AddMonths(-1).ToString("yyyy-MM-dd")
$requests = [ordered]@{
    "day-recap"           = "/day-recap?date=$Day"
    "day-wait-chart"      = "/day-wait-chart?date=$Day"
    "export-preview-day"  = "/export/preview?period=day&date=$Day"
    "export-preview-week" = "/export/preview?period=week&date=$week&lang=en"
    "export-month-csv"    = "/export?period=month&date=$month&lang=en"
    "export-bad-period"   = "/export/preview?period=year&date=$Day"
    "export-bad-format"   = "/export?period=day&date=$Day&format=xlsx"
    "recap-bad-date"      = "/day-recap?date=not-a-date"
    "set-lanes-no-key"    = "POST /set-lanes"
    "unknown-route"       = "/no-such-route"
}
$volatile = '"(time|generated_at|as_of|now|age_seconds|updated_at|server_time)"\s*:\s*"?[^,}\]]*"?'

New-Item -ItemType Directory -Force $Out | Out-Null
foreach ($name in $requests.Keys) {
    $req = $requests[$name]
    $method = "GET"; $path = $req
    if ($req -like "POST *") { $method = "POST"; $path = $req.Substring(5) }
    $raw = & curl.exe -s -X $method -H "Content-Type: application/json" $(if ($method -eq "POST") { "-d", "{}" }) -w "`n%{http_code}" "$Base$path"
    $text = ($raw -join "`n") -replace $volatile, '"$1":"<t>"'
    [IO.File]::WriteAllText((Join-Path $Out "$name.txt"), $text, (New-Object System.Text.UTF8Encoding $false))
}
Write-Host "Saved $($requests.Count) answers to $Out (day $Day)" -ForegroundColor Green

if ($CompareWith) {
    $diff = 0
    foreach ($name in $requests.Keys) {
        $a = Join-Path $CompareWith "$name.txt"; $b = Join-Path $Out "$name.txt"
        if (-not (Test-Path $a)) { Write-Host "  MISSING  $name (not in $CompareWith)" -ForegroundColor Yellow; $diff++; continue }
        $same = [IO.File]::ReadAllText($a) -ceq [IO.File]::ReadAllText($b)
        $code = ([IO.File]::ReadAllLines($b) | Select-Object -Last 1)
        if ($same) { Write-Host "  SAME  $code  $name" -ForegroundColor Green } else { Write-Host "  DIFF  $code  $name" -ForegroundColor Red; $diff++ }
    }
    if ($diff) { Write-Host "$diff answer(s) differ - compare the files in $CompareWith and $Out" -ForegroundColor Red; exit 1 }
    Write-Host "All $($requests.Count) answers identical." -ForegroundColor Green
}
