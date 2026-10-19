param(
    [switch]$ForceWebhook
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "logs"
if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}
$LogFile = Join-Path $LogDir "cloud_pipeline.log"

Set-Location -LiteralPath $ProjectRoot
$env:PYTHONPATH = (Join-Path $ProjectRoot "src")

$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$Msg = "[$Timestamp] Starting Stock-AI Cloud Pipeline (Scheduled/Manual)..."
Write-Output $Msg | Out-File -FilePath $LogFile -Append -Encoding utf8
Write-Host $Msg

$ArgsList = @("-u", "-m", "stock_ai.cloud.pipeline")
if ($ForceWebhook) {
    $ArgsList += "--force-webhook"
}

& $Python @ArgsList *>> $LogFile
$ExitCode = $LASTEXITCODE

$EndTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
$EndMsg = "[$EndTimestamp] Stock-AI Cloud Pipeline finished with exit code $ExitCode."
Write-Output $EndMsg | Out-File -FilePath $LogFile -Append -Encoding utf8
Write-Host $EndMsg

exit $ExitCode
