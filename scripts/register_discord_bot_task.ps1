[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "stock-ai discord bot",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$WrapperScript = Join-Path $ScriptDir "run_discord_bot.ps1"
$HiddenRunner = Join-Path $ScriptDir "run_hidden.vbs"

$WscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"
$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

$Arguments = @(
    "//nologo",
    "`"$HiddenRunner`"",
    "`"$PowerShellExe`"",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$WrapperScript`""
) -join " "

Write-Host "Registering Scheduled Task: $TaskName"
Write-Host "Action: $WscriptExe $Arguments"

$Action = New-ScheduledTaskAction -Execute $WscriptExe -Argument $Arguments -WorkingDirectory $ProjectRoot

# Triggers:
# 1. At user logon
$LogonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# 2. Recurring trigger repeating every 5 minutes for 10 years (re-spawns if closed or wakes from sleep)
$RepeatTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)

$Triggers = @($LogonTrigger, $RepeatTrigger)

# Settings:
# - ExecutionTimeLimit: [TimeSpan]::Zero (Infinite run time, never kill after 3 days)
# - RestartCount: 999, RestartInterval: 1 min
# - MultipleInstances: IgnoreNew (single instance protected by mutex)
# - StartWhenAvailable: Run immediately if missed during sleep/off
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -StartWhenAvailable

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $ExistingTask) {
    if (-not $Force) {
        Write-Host "Task '$TaskName' already exists. Use -Force to update."
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Triggers -Settings $Settings -Description "Runs stock-ai interactive Discord Polling Bot permanently in the background with immortal watchdog" | Out-Null
Write-Host "Task '$TaskName' registered successfully with immortal watchdog configuration."
