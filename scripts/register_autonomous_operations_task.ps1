[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "stock-ai autonomous operations",
    [string]$Time = "11:15",
    [string]$DependencyTaskName = "stock-ai daily predict",
    [double]$DependencyWaitMinutes = 30,
    [int]$DependencyPollSeconds = 60,
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$ExpectedRoot = "C:\Users\kmats\Documents\ai-stock\stock-ai"
$WrapperScript = Join-Path $ScriptDir "run_autonomous_operations.ps1"
if ($ProjectRoot -ne $ExpectedRoot) {
    throw "Refusing to register outside the formal project root. Actual: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $WrapperScript)) {
    throw "Autonomous Operations wrapper was not found: $WrapperScript"
}
if ($DependencyWaitMinutes -lt 0) {
    throw "DependencyWaitMinutes cannot be negative."
}
if ($DependencyPollSeconds -le 0) {
    throw "DependencyPollSeconds must be positive."
}
try {
    $ParsedTime = [datetime]::ParseExact($Time, "HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)
} catch {
    throw "Time must use HH:mm format, for example 11:15."
}

$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$Arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$WrapperScript`"",
    "-DependencyTaskName", "`"$DependencyTaskName`"",
    "-DependencyWaitMinutes", $DependencyWaitMinutes.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DependencyPollSeconds", $DependencyPollSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
) -join " "
$Description = "Runs read-only stock-ai Autonomous Operations after daily_predict. Exit 2 means human review; no automatic retry."

Write-Host "Desired task: $TaskName"
Write-Host "Schedule: daily at $Time"
Write-Host "Command: $PowerShellExe $Arguments"
Write-Host "Working directory: $ProjectRoot"
Write-Host "Dependency: $DependencyTaskName (wait up to $DependencyWaitMinutes minutes, poll every $DependencyPollSeconds seconds)"
Write-Host "StartWhenAvailable: enabled"
Write-Host "MultipleInstances: IgnoreNew"
Write-Host "Automatic retry: disabled (exit code 2 means human review required)"
Write-Host "Security: current interactive user, limited run level; elevation may be required to register on some PCs"

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
$ExactMatch = $false
if ($null -ne $ExistingTask) {
    $ExistingAction = @($ExistingTask.Actions) | Select-Object -First 1
    $ExistingTrigger = @($ExistingTask.Triggers) | Select-Object -First 1
    $ExistingTime = if ($ExistingTrigger.StartBoundary) {
        ([datetime]$ExistingTrigger.StartBoundary).ToString("HH:mm")
    } else {
        $null
    }
    $ExactMatch = (
        $ExistingAction.Execute -eq $PowerShellExe -and
        $ExistingAction.Arguments -eq $Arguments -and
        $ExistingAction.WorkingDirectory -eq $ProjectRoot -and
        $ExistingTime -eq $Time -and
        $ExistingTask.Settings.MultipleInstances.ToString() -eq "IgnoreNew" -and
        [bool]$ExistingTask.Settings.StartWhenAvailable -and
        -not [bool]$ExistingTask.Settings.DisallowStartIfOnBatteries -and
        -not [bool]$ExistingTask.Settings.StopIfGoingOnBatteries -and
        $ExistingTask.Settings.ExecutionTimeLimit -eq "PT1H" -and
        [int]$ExistingTask.Settings.RestartCount -eq 0 -and
        $ExistingTask.Description -eq $Description
    )
    if ($ExactMatch) {
        Write-Host "Existing task already matches the requested configuration; no change was made."
        exit 0
    }
    Write-Warning "An existing task with this name differs from the requested configuration."
    if (-not $Force -and -not $WhatIfPreference) {
        throw "Use -Force to update the existing task after reviewing the displayed configuration."
    }
}

if (-not $PSCmdlet.ShouldProcess($TaskName, "Register or update Autonomous Operations scheduled task")) {
    exit 0
}

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
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description $Description `
    -Force | Out-Null

Write-Host "Registered task: $TaskName"
