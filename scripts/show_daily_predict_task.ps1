param(
    [string]$TaskName = "stock-ai daily predict",
    [switch]$AsJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
function Format-TaskDate {
    param([object]$Value)

    if ($null -eq $Value) {
        return $null
    }
    try {
        $DateValue = [datetime]$Value
        if ($DateValue.Year -lt 2000) {
            return $null
        }
        return $DateValue.ToString("yyyy-MM-dd HH:mm:ss")
    } catch {
        return $Value.ToString()
    }
}

if ($null -eq $Task) {
    $Result = [ordered]@{
        registered = $false
        task_name = $TaskName
        state = $null
        next_run_time = $null
        last_run_time = $null
        last_result = $null
        command = $null
        working_directory = $null
    }
} else {
    $Info = Get-ScheduledTaskInfo -TaskName $TaskName
    $Action = @($Task.Actions) | Select-Object -First 1
    $Result = [ordered]@{
        registered = $true
        task_name = $TaskName
        state = $Task.State.ToString()
        next_run_time = Format-TaskDate $Info.NextRunTime
        last_run_time = Format-TaskDate $Info.LastRunTime
        last_result = $Info.LastTaskResult
        command = if ($Action) { "$($Action.Execute) $($Action.Arguments)" } else { $null }
        working_directory = if ($Action) { $Action.WorkingDirectory } else { $null }
    }
}

if ($AsJson) {
    $Result | ConvertTo-Json -Depth 4
    exit 0
}

if (-not $Result.registered) {
    Write-Host "Task is not registered: $TaskName"
    exit 0
}

Write-Host "Task name: $($Result.task_name)"
Write-Host "Registered: $($Result.registered)"
Write-Host "State: $($Result.state)"
Write-Host "Next run time: $($Result.next_run_time)"
Write-Host "Last run time: $($Result.last_run_time)"
Write-Host "Last result: $($Result.last_result)"
Write-Host "Working directory: $($Result.working_directory)"
Write-Host "Command: $($Result.command)"
