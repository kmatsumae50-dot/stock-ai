param(
    [string]$TaskName = "stock-ai daily predict",
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $ExistingTask) {
    Write-Host "Task is not registered: $TaskName"
    exit 0
}

if (-not $Force) {
    $Answer = Read-Host "Delete scheduled task '$TaskName'? [y/N]"
    if ($Answer -notin @("y", "Y", "yes", "YES")) {
        Write-Host "Unregister cancelled."
        exit 0
    }
}

Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "Deleted task: $TaskName"
