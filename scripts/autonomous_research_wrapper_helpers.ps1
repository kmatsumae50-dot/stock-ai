function Get-LatestAutonomousOperationsReport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportDir
    )

    if (-not (Test-Path -LiteralPath $ReportDir)) {
        return $null
    }
    return Get-ChildItem -LiteralPath $ReportDir -Filter "operations_*.json" -File |
        Where-Object { $_.BaseName -match '^operations_\d{4}-\d{2}-\d{2}_\d{4}(_\d+)?$' } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

function Test-AutonomousOperationsReportForResearch {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportDir,
        [Parameter(Mandatory = $true)]
        [double]$FreshnessMinutes,
        [datetime]$DependencyLastRunTime,
        [scriptblock]$NowProvider = { Get-Date }
    )

    $Now = & $NowProvider
    $ReportFile = Get-LatestAutonomousOperationsReport -ReportDir $ReportDir
    if ($null -eq $ReportFile) {
        return [pscustomobject]@{
            Healthy = $false
            Status = "operations_report_missing"
            ReportPath = $null
            CycleCompletedAt = $null
            AgeMinutes = $null
            Reasons = @("operations_report_missing")
        }
    }

    try {
        $Report = Get-Content -LiteralPath $ReportFile.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        return [pscustomobject]@{
            Healthy = $false
            Status = "operations_report_unreadable"
            ReportPath = $ReportFile.FullName
            CycleCompletedAt = $null
            AgeMinutes = $null
            Reasons = @("operations_report_unreadable")
        }
    }

    $Reasons = @()
    $CycleCompletedAt = $null
    try {
        $CycleCompletedAt = [datetime]$Report.cycle_completed_at
    } catch {
        $Reasons += "operations_cycle_completed_at_invalid"
    }
    $AgeMinutes = if ($null -ne $CycleCompletedAt) {
        ($Now - $CycleCompletedAt).TotalMinutes
    } else {
        $null
    }
    if ($null -ne $CycleCompletedAt) {
        if ($CycleCompletedAt.Date -ne $Now.Date -or $AgeMinutes -lt -5 -or $AgeMinutes -gt $FreshnessMinutes) {
            $Reasons += "operations_report_stale"
        }
        if ($DependencyLastRunTime.Year -ge 2000 -and $CycleCompletedAt -lt $DependencyLastRunTime.AddMinutes(-5)) {
            $Reasons += "operations_report_predates_latest_task_run"
        }
    }
    if ($Report.stop_reason -ne "completed_read_only_cycle") {
        $Reasons += "operations_cycle_not_completed"
    }
    if ([int]($Report.health_summary.action_required_count) -gt 0) {
        $Reasons += "operations_action_required"
    }
    if ([int]($Report.health_summary.unknown_source_count) -gt 0 -or @($Report.unknown_source_findings).Count -gt 0) {
        $Reasons += "operations_unknown_source"
    }
    if (@($Report.missing_live_findings).Count -gt 0) {
        $Reasons += "operations_missing_live_findings"
    }
    if ($Report.data_freshness.freshness_status -ne "OK" -or [bool]$Report.data_freshness.stale_at_run_time) {
        $Reasons += "operations_data_stale"
    }
    if (-not [bool]$Report.database_validation.readable) {
        $Reasons += "operations_database_unreadable"
    }

    return [pscustomobject]@{
        Healthy = $Reasons.Count -eq 0
        Status = if ($Reasons.Count -eq 0) { "operations_report_healthy" } else { "operations_report_unhealthy" }
        ReportPath = $ReportFile.FullName
        CycleCompletedAt = $CycleCompletedAt
        AgeMinutes = $AgeMinutes
        Reasons = $Reasons
    }
}

function Wait-AutonomousResearchDependency {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TaskName,
        [Parameter(Mandatory = $true)]
        [string]$ReportDir,
        [Parameter(Mandatory = $true)]
        [double]$WaitMinutes,
        [Parameter(Mandatory = $true)]
        [int]$PollSeconds,
        [Parameter(Mandatory = $true)]
        [double]$FreshnessMinutes,
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

    if ($WaitMinutes -lt 0) { throw "WaitMinutes cannot be negative." }
    if ($PollSeconds -le 0) { throw "PollSeconds must be positive." }
    if ($FreshnessMinutes -le 0) { throw "FreshnessMinutes must be positive." }

    $LogLines = @()
    $Task = & $TaskReader $TaskName
    if ($null -eq $Task) {
        return [pscustomobject]@{
            ShouldProceed = $false
            Status = "operations_task_not_registered"
            ExitCode = 1
            LastResult = $null
            ReportPath = $null
            LogLines = @("dependency_status: operations_task_not_registered")
        }
    }

    $WaitStartedAt = & $NowProvider
    while ($Task.State.ToString() -eq "Running") {
        $ElapsedMinutes = ((& $NowProvider) - $WaitStartedAt).TotalMinutes
        if ($ElapsedMinutes -ge $WaitMinutes) {
            $LogLines += "dependency_status: wait_timeout"
            return [pscustomobject]@{
                ShouldProceed = $false
                Status = "wait_timeout"
                ExitCode = 125
                LastResult = $null
                ReportPath = $null
                LogLines = $LogLines
            }
        }
        $LogLines += "dependency_wait: Autonomous Operations is Running; checking again in $PollSeconds seconds"
        & $SleepAction $PollSeconds
        $Task = & $TaskReader $TaskName
        if ($null -eq $Task) {
            $LogLines += "dependency_status: operations_task_disappeared"
            return [pscustomobject]@{
                ShouldProceed = $false
                Status = "operations_task_disappeared"
                ExitCode = 1
                LastResult = $null
                ReportPath = $null
                LogLines = $LogLines
            }
        }
    }

    $Info = & $TaskInfoReader $TaskName
    if ($null -eq $Info) {
        $LogLines += "dependency_status: operations_task_info_unavailable"
        return [pscustomobject]@{
            ShouldProceed = $false
            Status = "operations_task_info_unavailable"
            ExitCode = 1
            LastResult = $null
            ReportPath = $null
            LogLines = $LogLines
        }
    }

    $LastResult = [int]$Info.LastTaskResult
    $LastRunTime = [datetime]$Info.LastRunTime
    $Now = & $NowProvider
    $LogLines += "dependency_last_run_time: $($LastRunTime.ToString('o'))"
    $LogLines += "dependency_last_result: $LastResult"
    if ($LastResult -eq 2) {
        $LogLines += "dependency_status: operations_human_review_required"
        return [pscustomobject]@{
            ShouldProceed = $false
            Status = "operations_human_review_required"
            ExitCode = 3
            LastResult = $LastResult
            ReportPath = $null
            LogLines = $LogLines
        }
    }
    if ($LastResult -ne 0) {
        $LogLines += "dependency_status: operations_technical_failure"
        return [pscustomobject]@{
            ShouldProceed = $false
            Status = "operations_technical_failure"
            ExitCode = 1
            LastResult = $LastResult
            ReportPath = $null
            LogLines = $LogLines
        }
    }
    $TaskAgeMinutes = ($Now - $LastRunTime).TotalMinutes
    if ($LastRunTime.Year -lt 2000 -or $LastRunTime.Date -ne $Now.Date -or $TaskAgeMinutes -gt $FreshnessMinutes) {
        $LogLines += "dependency_status: operations_task_run_stale"
        return [pscustomobject]@{
            ShouldProceed = $false
            Status = "operations_task_run_stale"
            ExitCode = 3
            LastResult = $LastResult
            ReportPath = $null
            LogLines = $LogLines
        }
    }

    $ReportResult = Test-AutonomousOperationsReportForResearch `
        -ReportDir $ReportDir `
        -FreshnessMinutes $FreshnessMinutes `
        -DependencyLastRunTime $LastRunTime `
        -NowProvider $NowProvider
    $LogLines += "dependency_report_path: $($ReportResult.ReportPath)"
    $LogLines += "dependency_report_status: $($ReportResult.Status)"
    if ($null -ne $ReportResult.CycleCompletedAt) {
        $LogLines += "dependency_cycle_completed_at: $($ReportResult.CycleCompletedAt.ToString('o'))"
        $LogLines += "dependency_report_age_minutes: $([Math]::Round($ReportResult.AgeMinutes, 2))"
    }
    if (-not $ReportResult.Healthy) {
        $LogLines += "dependency_report_reasons: $(@($ReportResult.Reasons) -join ',')"
        return [pscustomobject]@{
            ShouldProceed = $false
            Status = $ReportResult.Status
            ExitCode = 3
            LastResult = $LastResult
            ReportPath = $ReportResult.ReportPath
            LogLines = $LogLines
        }
    }

    $LogLines += "dependency_status: ready_to_research"
    return [pscustomobject]@{
        ShouldProceed = $true
        Status = "ready_to_research"
        ExitCode = 0
        LastResult = $LastResult
        ReportPath = $ReportResult.ReportPath
        LogLines = $LogLines
    }
}
