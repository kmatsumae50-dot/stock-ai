param(
    [string]$TaskName = "stock-ai daily predict",
    [string]$Time = "08:30",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$DailyPredictScript = Join-Path $ScriptDir "daily_predict.ps1"

if (-not (Test-Path -LiteralPath $DailyPredictScript)) {
    throw "daily_predict.ps1 was not found: $DailyPredictScript"
}

try {
    $ParsedTime = [datetime]::ParseExact($Time, "HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)
} catch {
    throw "Time must use HH:mm format, for example 08:30."
}

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $ExistingTask) {
    if (-not $Force) {
        $Answer = Read-Host "Task '$TaskName' already exists. Overwrite it? [y/N]"
        if ($Answer -notin @("y", "Y", "yes", "YES")) {
            Write-Host "Registration cancelled."
            exit 0
        }
    }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$Arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$DailyPredictScript`"",
    "-FetchNews",
    "-DailyNewsProfiles",
    "-NewsLimit", "3"
) -join " "

$Action = New-ScheduledTaskAction `
    -Execute $PowerShellExe `
    -Argument $Arguments `
    -WorkingDirectory $ProjectRoot

$Trigger = New-ScheduledTaskTrigger -Daily -At $ParsedTime
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 3)

$Description = "Runs stock-ai daily prediction with FetchNews, DailyNewsProfiles, and NewsLimit 3."

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description $Description | Out-Null

Write-Host "Registered task: $TaskName"
Write-Host "Project root: $ProjectRoot"
Write-Host "Schedule: daily at $Time"
Write-Host "Command: .\scripts\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3"
Write-Host "StartWhenAvailable: enabled"
