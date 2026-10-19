param(
    [double]$PollInterval = 3.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
$LogFile = Join-Path $LogDir "discord_bot.log"

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = (Join-Path $ProjectRoot "src")
$env:PYTHONIOENCODING = "utf-8"

Write-Output "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] Starting Discord Polling Bot..." | Out-File -FilePath $LogFile -Append -Encoding utf8
& $Python -u -m stock_ai.cloud.discord_bot *>> $LogFile
