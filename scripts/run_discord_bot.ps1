param(
    [double]$PollInterval = 3.0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
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

# 1. Single-Instance Mutex Lock (Prevent duplicate running processes)
$MutexName = "Local\StockAI_DiscordBot_SingleInstance_Mutex"
$CreatedNew = $false
try {
    $Mutex = [System.Threading.Mutex]::new($true, $MutexName, [ref]$CreatedNew)
} catch {
    $CreatedNew = $false
}

if (-not $CreatedNew) {
    try {
        $Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Output "[$Timestamp] [SingleInstance] Another Discord Polling Bot instance is already running. Exiting duplicate process." | Out-File -FilePath $LogFile -Append -Encoding utf8 -ErrorAction SilentlyContinue
    } catch {}
    exit 0
}

# 2. Immortal Daemon Loop: Automatically restart Python process if it ever terminates
try {
    $Timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Output "[$Timestamp] [Watchdog] Acquired single-instance lock. Entering immortal supervision loop." | Out-File -FilePath $LogFile -Append -Encoding utf8

    while ($true) {
        $StartTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Output "[$StartTime] Starting Discord Polling Bot..." | Out-File -FilePath $LogFile -Append -Encoding utf8

        # Execute python bot synchronously
        & $Python -u -m stock_ai.cloud.discord_bot *>> $LogFile
        $ExitCode = $LASTEXITCODE

        $EndTime = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        Write-Output "[$EndTime] [Watchdog Warning] Discord Polling Bot terminated (ExitCode: $ExitCode). Automatically restarting in 5 seconds..." | Out-File -FilePath $LogFile -Append -Encoding utf8
        Start-Sleep -Seconds 5
    }
} finally {
    if ($null -ne $Mutex -and $CreatedNew) {
        try {
            $Mutex.ReleaseMutex()
            $Mutex.Dispose()
        } catch {}
    }
}
