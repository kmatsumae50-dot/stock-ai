function Wait-AutonomousDependencyTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskName,
        [Parameter(Mandatory = $true)]
        [double]$WaitMinutes,
        [Parameter(Mandatory = $true)]
        [int]$PollSeconds,
        [scriptblock]$TaskReader = {
            param([string]$Name)
            Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue
        },
        [scriptblock]$TaskInfoReader = {
            param([string]$Name)
            Get-ScheduledTaskInfo -TaskName $Name -ErrorAction SilentlyContinue
        },
        [scriptblock]$SleepAction = {
            param([int]$Seconds)
            Start-Sleep -Seconds $Seconds
        },
        [scriptblock]$NowProvider = { Get-Date }
    )

    if ($WaitMinutes -lt 0) {
        throw "WaitMinutes cannot be negative."
    }
    if ($PollSeconds -le 0) {
        throw "PollSeconds must be positive."
    }

    $LogLines = @()
    $Task = & $TaskReader $TaskName
    if ($null -eq $Task) {
        return [pscustomobject]@{
            ShouldProceed = $true
            Status = "not_registered_proceeding"
            LastResult = $null
            LogLines = @("dependency_status: not_registered_proceeding")
        }
    }

    $WaitStartedAt = & $NowProvider
    while ($Task.State.ToString() -eq "Running") {
        $ElapsedMinutes = ((& $NowProvider) - $WaitStartedAt).TotalMinutes
        if ($ElapsedMinutes -ge $WaitMinutes) {
            $LogLines += "dependency_status: wait_timeout"
            $LogLines += "dependency_state: Running"
            return [pscustomobject]@{
                ShouldProceed = $false
                Status = "wait_timeout"
                LastResult = $null
                LogLines = $LogLines
            }
        }
        $LogLines += "dependency_wait: daily predict is Running; checking again in $PollSeconds seconds"
        & $SleepAction $PollSeconds
        $Task = & $TaskReader $TaskName
        if ($null -eq $Task) {
            $LogLines += "dependency_status: disappeared_while_waiting_proceeding"
            return [pscustomobject]@{
                ShouldProceed = $true
                Status = "disappeared_while_waiting_proceeding"
                LastResult = $null
                LogLines = $LogLines
            }
        }
    }

    $Info = & $TaskInfoReader $TaskName
    $LastResult = if ($null -ne $Info) { $Info.LastTaskResult } else { $null }
    $LogLines += "dependency_status: ready_to_proceed"
    $LogLines += "dependency_state: $($Task.State)"
    if ($null -ne $Info) {
        $LogLines += "dependency_last_run_time: $($Info.LastRunTime.ToString('o'))"
        $LogLines += "dependency_last_result: $LastResult"
        if ($LastResult -ne 0) {
            $LogLines += "dependency_warning: last result is nonzero; Autonomous Operations will diagnose it without retrying daily_predict"
        }
    }
    return [pscustomobject]@{
        ShouldProceed = $true
        Status = "ready_to_proceed"
        LastResult = $LastResult
        LogLines = $LogLines
    }
}

