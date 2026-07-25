param(
    [string]$TaskName = "stock-ai autonomous research",
    [switch]$AsJson
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Format-TaskDate {
    param([object]$Value)
    if ($null -eq $Value) { return $null }
    try {
        $DateValue = [datetime]$Value
        if ($DateValue.Year -lt 2000) { return $null }
        return $DateValue.ToString("yyyy-MM-dd HH:mm:ss")
    } catch {
        return $Value.ToString()
    }
}

$Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($null -eq $Task) {
    $Result = [ordered]@{ registered = $false; task_name = $TaskName }
} else {
    $Info = Get-ScheduledTaskInfo -TaskName $TaskName
    $Action = @($Task.Actions) | Select-Object -First 1
    $Trigger = @($Task.Triggers) | Select-Object -First 1
    $Result = [ordered]@{
        registered = $true
        task_name = $TaskName
        state = $Task.State.ToString()
        enabled = $Task.State.ToString() -ne "Disabled"
        trigger = if ($Trigger.StartBoundary) { "Daily " + ([datetime]$Trigger.StartBoundary).ToString("HH:mm") } else { $null }
        next_run_time = Format-TaskDate $Info.NextRunTime
        last_run_time = Format-TaskDate $Info.LastRunTime
        last_result = $Info.LastTaskResult
        command = if ($Action) { $Action.Execute } else { $null }
        arguments = if ($Action) { $Action.Arguments } else { $null }
        working_directory = if ($Action) { $Action.WorkingDirectory } else { $null }
        start_when_available = [bool]$Task.Settings.StartWhenAvailable
        multiple_instances = $Task.Settings.MultipleInstances.ToString()
        allow_start_if_on_batteries = -not [bool]$Task.Settings.DisallowStartIfOnBatteries
        stop_if_going_on_batteries = [bool]$Task.Settings.StopIfGoingOnBatteries
        restart_count = [int]$Task.Settings.RestartCount
        execution_time_limit = $Task.Settings.ExecutionTimeLimit
        description = $Task.Description
    }
}

if ($AsJson) {
    $Result | ConvertTo-Json -Depth 4
    exit 0
}
$Result.GetEnumerator() | ForEach-Object { Write-Host "$($_.Key): $($_.Value)" }
