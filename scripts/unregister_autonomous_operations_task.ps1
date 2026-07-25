[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$TaskName = "stock-ai autonomous operations",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $ExistingTask) {
    Write-Host "Task is not registered: $TaskName"
    exit 0
}

if (-not $Force -and -not $WhatIfPreference) {
    $Answer = Read-Host "Delete scheduled task '$TaskName'? [y/N]"
    if ($Answer -notin @("y", "Y", "yes", "YES")) {
        Write-Host "Unregister cancelled. No other task was changed."
        exit 0
    }
}

if ($PSCmdlet.ShouldProcess($TaskName, "Unregister Autonomous Operations scheduled task")) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Deleted task: $TaskName"
    Write-Host "The stock-ai daily predict task was not changed."
}

