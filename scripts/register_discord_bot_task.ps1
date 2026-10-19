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
Write-Host "Trigger: At log on of current user"
Write-Host "Action: $WscriptExe $Arguments"

$Action = New-ScheduledTaskAction -Execute $WscriptExe -Argument $Arguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit (New-TimeSpan -Days 365)

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $ExistingTask) {
    if (-not $Force) {
        Write-Host "Task '$TaskName' already exists. Use -Force to update."
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Description "Runs stock-ai interactive Discord Polling Bot permanently in the background" | Out-Null
Write-Host "Task '$TaskName' registered successfully."
