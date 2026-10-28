<#
.SYNOPSIS
  Record one evaluation clip from a store camera, independently of the live pipeline.

.DESCRIPTION
  Opens a second RTSP connection to the same stream the pipeline reads and copies the
  video packets to disk (ffmpeg -c copy: no decoding, near-zero CPU, live loop untouched).
  Audio is dropped. Camera address and credentials are read from the pipeline's own
  config.yml, so they never appear on the command line or in shell history.

  Writes <camera>_<yyyyMMdd_HHmmss>[_label].mkv plus a .json sidecar holding the wall-clock
  start time (needed to line the clip up with DB rows and the frame log), file size,
  sha256 and the ffprobe stream info.

.EXAMPLE
  .\record_clip.ps1 -Camera checkout -Minutes 20 -Label peak
.EXAMPLE
  .\record_clip.ps1 -Camera checkout -Minutes 1 -Label sizetest   # estimate MB per minute first
.EXAMPLE
  .\record_clip.ps1 -Camera entrance -DryRun                      # check config parsing only
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet('checkout', 'entrance')][string]$Camera,
    [ValidateRange(1, 60)][int]$Minutes = 20,
    [string]$Label = '',
    [string]$OutDir = (Join-Path $PSScriptRoot 'data\raw'),
    [double]$MinFreeGB = 3,
    [int]$RetentionDays = 30,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
if ($Camera -eq 'checkout') {
    $cfgPath = Join-Path $repoRoot 'Head-Detector\config.yml'
} else {
    $cfgPath = Join-Path $repoRoot 'Queue-Management-System-v2-main\Queue-Management-System-v2-main\config.yml'
}
if (-not (Test-Path $cfgPath)) { throw "Camera config not found: $cfgPath" }
$cfgText = [IO.File]::ReadAllText($cfgPath)

function Get-YamlScalar([string]$text, [string]$key) {
    $m = [regex]::Match($text, "(?m)^$key\s*:\s*(.*?)\s*$")
    if (-not $m.Success) { return '' }
    return $m.Groups[1].Value.Trim([char[]]@("'", '"'))
}

$camId    = Get-YamlScalar $cfgText 'camID'
$ip       = Get-YamlScalar $cfgText 'ip_address'
$user     = Get-YamlScalar $cfgText 'username'
$password = Get-YamlScalar $cfgText 'password'
if (-not $ip) { throw "ip_address missing in $cfgPath" }

# Same link-building rule as main.py / queue_management_v2.py, so we record exactly what the pipeline reads.
if ($ip.StartsWith('http')) {
    $url = $ip
} elseif ($user -and $password) {
    $url = "rtsp://${user}:${password}@$ip"
} else {
    $url = "rtsp://$ip"
}
$redactedUrl = if ($ip.StartsWith('http')) { $ip } else { "rtsp://***@$ip" }

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$name = "${Camera}_$stamp"
if ($Label) { $name = "${name}_$Label" }
$outFile = Join-Path $OutDir "$name.mkv"
$sidecar = Join-Path $OutDir "$name.json"

$drive = (Resolve-Path $OutDir).Drive
$freeGB = [math]::Round($drive.Free / 1GB, 2)

Write-Host "[REC] camera=$Camera camID=$camId stream=$redactedUrl"
Write-Host "[REC] duration=${Minutes} min -> $outFile"
Write-Host "[REC] free space on $($drive.Name): ${freeGB} GB (minimum $MinFreeGB GB)"

if ($DryRun) { Write-Host '[REC] Dry run: nothing recorded.'; return }
if ($freeGB -lt $MinFreeGB) { throw "Not enough free disk space (${freeGB} GB < $MinFreeGB GB)." }

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) { throw 'ffmpeg not found on PATH. Install it with: winget install Gyan.FFmpeg' }

$ffArgs = @(
    '-hide_banner', '-loglevel', 'warning',   # warning level keeps the URL (with password) out of the console
    '-rtsp_transport', 'tcp',
    '-i', $url,
    '-t', ($Minutes * 60),
    '-map', '0:v:0', '-c', 'copy',             # video only, no re-encoding
    '-y', $outFile
)

$startUtc = [DateTimeOffset]::UtcNow
& $ffmpeg.Source @ffArgs
$exitCode = $LASTEXITCODE
$endUtc = [DateTimeOffset]::UtcNow

if (-not (Test-Path $outFile)) { throw "ffmpeg produced no file (exit code $exitCode)." }

$probe = $null
$ffprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
if ($ffprobe) {
    $probeJson = & $ffprobe.Source -v error -select_streams v:0 -count_packets `
        -show_entries 'stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_packets:format=duration' `
        -of json $outFile
    $probe = ($probeJson -join "`n") | ConvertFrom-Json
}

$sizeMB = [math]::Round((Get-Item $outFile).Length / 1MB, 1)
$meta = [ordered]@{
    camera          = $Camera
    cam_id          = $camId
    file            = (Split-Path $outFile -Leaf)
    label           = $Label
    start_epoch     = $startUtc.ToUnixTimeMilliseconds() / 1000.0
    start_local     = $startUtc.ToLocalTime().ToString('yyyy-MM-dd HH:mm:ss.fff zzz')
    end_epoch       = $endUtc.ToUnixTimeMilliseconds() / 1000.0
    start_note      = 'start_epoch is taken just before ffmpeg connects; the first frame can be 0.5-2 s later. Refine with the on-screen camera clock if needed.'
    requested_sec   = $Minutes * 60
    ffmpeg_exit     = $exitCode
    size_mb         = $sizeMB
    sha256          = (Get-FileHash -Algorithm SHA256 $outFile).Hash.ToLower()
    delete_after    = $startUtc.ToLocalTime().AddDays($RetentionDays).ToString('yyyy-MM-dd')
    probe           = $probe
}
$json = $meta | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText($sidecar, $json, (New-Object Text.UTF8Encoding $false))

$elapsedMin = [math]::Max(($endUtc - $startUtc).TotalMinutes, 0.01)
Write-Host "[REC] done: $sizeMB MB in $([math]::Round($elapsedMin,1)) min (~$([math]::Round($sizeMB / $elapsedMin,1)) MB/min), exit=$exitCode"
Write-Host "[REC] sidecar: $sidecar"
