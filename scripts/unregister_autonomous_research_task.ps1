[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "stock-ai autonomous research",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ExpectedTaskName = "stock-ai autonomous research"
if ($TaskName -ne $ExpectedTaskName) {
    throw "Refusing to unregister any task except '$ExpectedTaskName'."
}

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $ExistingTask) {
    Write-Host "Task is not registered: $TaskName"
    exit 0
}
if (-not $Force -and -not $WhatIfPreference) {
    $Answer = Read-Host "Delete scheduled task '$TaskName'? [y/N]"
    if ($Answer -notin @("y", "Y", "yes", "YES")) {
        Write-Host "Unregister cancelled. No task was changed."
        exit 0
    }
}
if ($PSCmdlet.ShouldProcess($TaskName, "Unregister Autonomous Research scheduled task")) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Deleted task: $TaskName"
    Write-Host "The daily prediction and Autonomous Operations tasks were not changed."
}
