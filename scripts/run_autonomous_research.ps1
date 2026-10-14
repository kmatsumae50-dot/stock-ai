[CmdletBinding()]
param(
    [string]$Ticker = "7203.T",
    [double]$MaxRuntimeMinutes = 120,
    [string]$OperationsTaskName = "stock-ai autonomous operations",
    [double]$DependencyWaitMinutes = 30,
    [int]$DependencyPollSeconds = 60,
    [double]$OperationsFreshnessMinutes = 180,
    [double]$StaleLockMinutes = 150,
    [switch]$Force,
    [switch]$SkipShadow,
    [switch]$SkipOffline
)

$ErrorActionPreference = "Stop"
$ProjectRoot = "C:\Users\kmats\Documents\ai-stock\stock-ai"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "main.py"
$ReportDir = Join-Path $ProjectRoot "reports\autonomous"
$LogDir = Join-Path $ProjectRoot "logs\autonomous"
$LockPath = Join-Path $ReportDir ".research.lock"
$HelperScript = Join-Path $ProjectRoot "scripts\autonomous_research_wrapper_helpers.ps1"
$LockHelperScript = Join-Path $ProjectRoot "scripts\autonomous_lock_helpers.ps1"

if ((Resolve-Path $PSScriptRoot).Path -ne (Join-Path $ProjectRoot "scripts")) {
    throw "Refusing to run outside the formal project root: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Formal project virtual environment is missing: $Python"
}
if (-not (Test-Path -LiteralPath $HelperScript)) {
    throw "Research wrapper helper is missing: $HelperScript"
}
if (-not (Test-Path -LiteralPath $LockHelperScript)) {
    throw "Autonomous lock helper is missing: $LockHelperScript"
}
if ($MaxRuntimeMinutes -le 0 -or $MaxRuntimeMinutes -gt 120) {
    throw "MaxRuntimeMinutes must be in (0, 120]."
}
if ($DependencyWaitMinutes -lt 0) { throw "DependencyWaitMinutes cannot be negative." }
if ($DependencyPollSeconds -le 0) { throw "DependencyPollSeconds must be positive." }
if ($OperationsFreshnessMinutes -le 0) { throw "OperationsFreshnessMinutes must be positive." }
if ($StaleLockMinutes -le $MaxRuntimeMinutes) {
    throw "StaleLockMinutes must be greater than MaxRuntimeMinutes."
}
. $HelperScript
. $LockHelperScript

New-Item -ItemType Directory -Force -Path $ReportDir, $LogDir | Out-Null
Set-Location -LiteralPath $ProjectRoot
$stamp = Get-Date -Format "yyyy-MM-dd_HHmm"
$logPath = Join-Path $LogDir "research_$stamp.log"
$PreflightLogLines = @()

if (Test-Path -LiteralPath $LockPath) {
    $LockState = Get-AutonomousLockState -LockPath $LockPath -StaleAfterMinutes $StaleLockMinutes
    if (-not [bool]$LockState.CanRecover) {
        $Message = "Research lock stop: classification=$($LockState.Classification), reason=$($LockState.Reason), path=$LockPath"
        $Message | Tee-Object -FilePath $logPath -Append
        exit 125
    }
    $Recovery = Move-AutonomousStaleLock `
        -LockPath $LockPath `
        -QuarantineDirectory (Join-Path $LogDir "stale_locks") `
        -LockState $LockState
    $PreflightLogLines += "lock_recovery_status: $($Recovery.Status)"
    $PreflightLogLines += "lock_recovery_age_minutes: $([Math]::Round($Recovery.AgeMinutes, 2))"
    $PreflightLogLines += "lock_recovery_quarantine_path: $($Recovery.QuarantinePath)"
}

$lockPayload = @{ pid = $PID; created_at = (Get-Date).ToString("o"); wrapper = $true } | ConvertTo-Json
Set-Content -LiteralPath $LockPath -Value $lockPayload -Encoding UTF8

try {
    $dependency = Wait-AutonomousResearchDependency `
        -TaskName $OperationsTaskName `
        -ReportDir $ReportDir `
        -WaitMinutes $DependencyWaitMinutes `
        -PollSeconds $DependencyPollSeconds `
        -FreshnessMinutes $OperationsFreshnessMinutes
    $PreflightLogLines + @(
        "Research dependency gate",
        "operations_task_name: $OperationsTaskName",
        "dependency_wait_minutes: $DependencyWaitMinutes",
        "dependency_poll_seconds: $DependencyPollSeconds",
        "operations_freshness_minutes: $OperationsFreshnessMinutes"
    ) + @($dependency.LogLines) | Tee-Object -FilePath $logPath -Append
    if (-not [bool]$dependency.ShouldProceed) {
        $ExitClassification = if ([int]$dependency.ExitCode -eq 3) {
            "safe_operations_unhealthy_skip"
        } else {
            "technical_or_concurrency_dependency_stop"
        }
        "dependency_exit_classification: $ExitClassification" |
            Tee-Object -FilePath $logPath -Append
        "Research was not started because dependency_status=$($dependency.Status)." |
            Tee-Object -FilePath $logPath -Append
        exit ([int]$dependency.ExitCode)
    }

    $arguments = @(
        $Main,
        "autonomous-research",
        "--ticker", $Ticker,
        "--max-runtime-minutes", $MaxRuntimeMinutes
    )
    if ($Force) { $arguments += "--force" }
    if ($SkipShadow) { $arguments += "--skip-shadow" }
    if ($SkipOffline) { $arguments += "--skip-offline" }

    $env:STOCK_AI_RESEARCH_WRAPPER_LOCK_HELD = "1"
    $stdoutPath = "$logPath.stdout.tmp"
    $stderrPath = "$logPath.stderr.tmp"
    $argumentText = ($arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + ($_ -replace '"', '\"') + '"' } else { [string]$_ }
    }) -join ' '
    $process = Start-Process -FilePath $Python -ArgumentList $argumentText -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddMinutes($MaxRuntimeMinutes)
    while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 1
        $process.Refresh()
    }
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        "Research exceeded the wrapper runtime limit of $MaxRuntimeMinutes minutes." |
            Tee-Object -FilePath $logPath -Append
        if (Test-Path $stdoutPath) { Get-Content $stdoutPath | Tee-Object -FilePath $logPath -Append }
        if (Test-Path $stderrPath) { Get-Content $stderrPath | Tee-Object -FilePath $logPath -Append }
        exit 124
    }
    if (Test-Path $stdoutPath) { Get-Content $stdoutPath | Tee-Object -FilePath $logPath -Append }
    if (Test-Path $stderrPath) { Get-Content $stderrPath | Tee-Object -FilePath $logPath -Append }
    $exitCode = $process.ExitCode
    Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    exit $exitCode
} finally {
    Remove-Item Env:\STOCK_AI_RESEARCH_WRAPPER_LOCK_HELD -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}
