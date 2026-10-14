[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "stock-ai autonomous research",
    [string]$Time = "12:00",
    [string]$OperationsTaskName = "stock-ai autonomous operations",
    [double]$DependencyWaitMinutes = 30,
    [int]$DependencyPollSeconds = 60,
    [double]$OperationsFreshnessMinutes = 180,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$ExpectedRoot = "C:\Users\kmats\Documents\ai-stock\stock-ai"
$WrapperScript = Join-Path $ScriptDir "run_autonomous_research.ps1"
if ($ProjectRoot -ne $ExpectedRoot) { throw "Refusing to register outside the formal project root. Actual: $ProjectRoot" }
if (-not (Test-Path -LiteralPath $WrapperScript)) { throw "Autonomous Research wrapper was not found: $WrapperScript" }
if ($DependencyWaitMinutes -lt 0) { throw "DependencyWaitMinutes cannot be negative." }
if ($DependencyPollSeconds -le 0) { throw "DependencyPollSeconds must be positive." }
if ($OperationsFreshnessMinutes -le 0) { throw "OperationsFreshnessMinutes must be positive." }
try {
    $ParsedTime = [datetime]::ParseExact($Time, "HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)
} catch {
    throw "Time must use HH:mm format, for example 12:00."
}

$HiddenRunner = Join-Path $ScriptDir "run_hidden.vbs"
if (-not (Test-Path -LiteralPath $HiddenRunner)) { throw "run_hidden.vbs was not found: $HiddenRunner" }

$WscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"
$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$Arguments = @(
    "//nologo",
    "`"$HiddenRunner`"",
    "`"$PowerShellExe`"",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$WrapperScript`"",
    "-OperationsTaskName", "`"$OperationsTaskName`"",
    "-DependencyWaitMinutes", $DependencyWaitMinutes.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DependencyPollSeconds", $DependencyPollSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-OperationsFreshnessMinutes", $OperationsFreshnessMinutes.ToString([System.Globalization.CultureInfo]::InvariantCulture)
) -join " "
$Description = "Runs gated read-only stock-ai Autonomous Research after healthy Autonomous Operations. No automatic retry or production change."

Write-Host "Desired task: $TaskName"
Write-Host "Schedule: daily at $Time"
Write-Host "Command: $WscriptExe $Arguments"
Write-Host "Working directory: $ProjectRoot"
Write-Host "Dependency: $OperationsTaskName (wait up to $DependencyWaitMinutes minutes, poll every $DependencyPollSeconds seconds)"
Write-Host "Operations report freshness: same day and at most $OperationsFreshnessMinutes minutes old"
Write-Host "StartWhenAvailable: enabled"
Write-Host "MultipleInstances: IgnoreNew"
Write-Host "Power: battery execution allowed"
Write-Host "Automatic retry: disabled for all exit codes, including 2, 3, 124, and 125"
Write-Host "Execution time limit: 130 minutes; wrapper timeout remains authoritative at 120 minutes"
Write-Host "Security: current interactive user, limited run level; elevation may be required to register on some PCs"

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -ne $ExistingTask) {
    $ExistingAction = @($ExistingTask.Actions) | Select-Object -First 1
    $ExistingTrigger = @($ExistingTask.Triggers) | Select-Object -First 1
    $ExistingTime = if ($ExistingTrigger.StartBoundary) { ([datetime]$ExistingTrigger.StartBoundary).ToString("HH:mm") } else { $null }
    $Differences = @()
    if ($ExistingAction.Execute -ne $WscriptExe) { $Differences += "command" }
    if ($ExistingAction.Arguments -ne $Arguments) { $Differences += "arguments" }
    if ($ExistingAction.WorkingDirectory -ne $ProjectRoot) { $Differences += "working_directory" }
    if ($ExistingTime -ne $Time) { $Differences += "trigger_time" }
    if ($ExistingTask.Settings.MultipleInstances.ToString() -ne "IgnoreNew") { $Differences += "multiple_instances" }
    if (-not [bool]$ExistingTask.Settings.StartWhenAvailable) { $Differences += "start_when_available" }
    if ([bool]$ExistingTask.Settings.DisallowStartIfOnBatteries -or [bool]$ExistingTask.Settings.StopIfGoingOnBatteries) { $Differences += "power_settings" }
    if ($ExistingTask.Settings.ExecutionTimeLimit -ne "PT2H10M") { $Differences += "execution_time_limit" }
    if ([int]$ExistingTask.Settings.RestartCount -ne 0) { $Differences += "restart_policy" }
    if ($ExistingTask.Description -ne $Description) { $Differences += "description" }
    if ($Differences.Count -eq 0) {
        Write-Host "Existing task already matches the requested configuration; no change was made."
        exit 0
    }
    Write-Warning "Existing task differs in: $($Differences -join ', ')"
    if (-not $Force -and -not $WhatIfPreference) {
        throw "Use -Force to update the existing task after reviewing the displayed differences."
    }
}

if (-not $PSCmdlet.ShouldProcess($TaskName, "Register or update Autonomous Research scheduled task")) { exit 0 }

$Action = New-ScheduledTaskAction -Execute $WscriptExe -Argument $Arguments -WorkingDirectory $ProjectRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $ParsedTime
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 130)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description $Description `
    -Force | Out-Null

Write-Host "Registered task: $TaskName"
Write-Host "The daily prediction and Autonomous Operations tasks were not changed."
