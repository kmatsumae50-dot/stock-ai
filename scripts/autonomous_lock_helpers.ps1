function Get-AutonomousLockState {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LockPath,
        [Parameter(Mandatory = $true)]
        [double]$StaleAfterMinutes,
        [scriptblock]$NowProvider = { Get-Date },
        [scriptblock]$ProcessReader = {
            param([int]$ProcessId)
            Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
        }
    )

    if ($StaleAfterMinutes -le 0) {
        throw "StaleAfterMinutes must be positive."
    }
    if (-not (Test-Path -LiteralPath $LockPath)) {
        return [pscustomobject]@{
            Classification = "lock_missing"
            CanRecover = $false
            Pid = $null
            CreatedAt = $null
            AgeMinutes = $null
            Reason = "lock_missing"
        }
    }

    try {
        $Payload = Get-Content -LiteralPath $LockPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $OwnerPid = [int]$Payload.pid
        $CreatedAt = [datetimeoffset]::Parse([string]$Payload.created_at)
        if ($OwnerPid -le 0) { throw "Lock PID must be positive." }
    } catch {
        return [pscustomobject]@{
            Classification = "lock_unreadable"
            CanRecover = $false
            Pid = $null
            CreatedAt = $null
            AgeMinutes = $null
            Reason = "Lock payload must contain a valid pid and created_at."
        }
    }

    $Owner = & $ProcessReader $OwnerPid
    if ($null -ne $Owner) {
        return [pscustomobject]@{
            Classification = "lock_active"
            CanRecover = $false
            Pid = $OwnerPid
            CreatedAt = $CreatedAt
            AgeMinutes = [Math]::Max(0, ((& $NowProvider) - $CreatedAt.LocalDateTime).TotalMinutes)
            Reason = "lock_owner_process_is_active"
        }
    }

    $AgeMinutes = ((& $NowProvider) - $CreatedAt.LocalDateTime).TotalMinutes
    if ($AgeMinutes -lt 0 -or $AgeMinutes -lt $StaleAfterMinutes) {
        return [pscustomobject]@{
            Classification = "lock_orphaned_recent"
            CanRecover = $false
            Pid = $OwnerPid
            CreatedAt = $CreatedAt
            AgeMinutes = $AgeMinutes
            Reason = "lock_owner_missing_but_stale_threshold_not_met"
        }
    }

    return [pscustomobject]@{
        Classification = "lock_stale_candidate"
        CanRecover = $true
        Pid = $OwnerPid
        CreatedAt = $CreatedAt
        AgeMinutes = $AgeMinutes
        Reason = "lock_owner_missing_and_stale_threshold_met"
    }
}

function Move-AutonomousStaleLock {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LockPath,
        [Parameter(Mandatory = $true)]
        [string]$QuarantineDirectory,
        [Parameter(Mandatory = $true)]
        [psobject]$LockState
    )

    if (-not [bool]$LockState.CanRecover -or $LockState.Classification -ne "lock_stale_candidate") {
        throw "Only a verified stale lock candidate can be quarantined."
    }
    New-Item -ItemType Directory -Path $QuarantineDirectory -Force | Out-Null
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmssfff"
    $Name = [System.IO.Path]::GetFileName($LockPath)
    $Destination = Join-Path $QuarantineDirectory "$Name.$Stamp.stale"
    Move-Item -LiteralPath $LockPath -Destination $Destination
    return [pscustomobject]@{
        Status = "stale_lock_quarantined"
        OriginalPath = $LockPath
        QuarantinePath = $Destination
        Pid = $LockState.Pid
        AgeMinutes = $LockState.AgeMinutes
    }
}
