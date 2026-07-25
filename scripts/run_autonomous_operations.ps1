param(
    [string]$Ticker = "7203.T",
    [string]$StatePath = "reports\autonomous\operations_state.json",
    [string]$TaskQueuePath = "reports\autonomous\operations_task_queue.json",
    [string]$ReportDir = "reports\autonomous",
    [double]$MaxRuntimeMinutes = 10,
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
if ($ProjectRoot -ne $ExpectedRoot) {
    throw "Refusing to run outside the formal project root. Actual: $ProjectRoot"
}
if ($MaxRuntimeMinutes -le 0) {
    throw "MaxRuntimeMinutes must be positive."
}
if ($DependencyWaitMinutes -lt 0) {
    throw "DependencyWaitMinutes cannot be negative."
}
if ($DependencyPollSeconds -le 0) {
    throw "DependencyPollSeconds must be positive."
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project virtual environment Python was not found: $Python"
}
$HelperScript = Join-Path $ScriptDir "autonomous_operations_wrapper_helpers.ps1"
if (-not (Test-Path -LiteralPath $HelperScript)) {
    throw "Autonomous Operations wrapper helper was not found: $HelperScript"
}
. $HelperScript

function Resolve-ProjectPath {
    param([string]$Value)
    if ([System.IO.Path]::IsPathRooted($Value)) {
        return [System.IO.Path]::GetFullPath($Value)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Value))
}

$ResolvedStatePath = Resolve-ProjectPath $StatePath
$ResolvedTaskQueuePath = Resolve-ProjectPath $TaskQueuePath
$ResolvedReportDir = Resolve-ProjectPath $ReportDir
$LogDir = Join-Path $ProjectRoot "logs\autonomous"
$LockPath = Join-Path $ResolvedReportDir ".operations.lock"
New-Item -ItemType Directory -Path $ResolvedReportDir -Force | Out-Null
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

if (Test-Path -LiteralPath $LockPath) {
    if (-not $Force) {
        throw "Autonomous Operations is already locked: $LockPath"
    }
    $OwnerIsActive = $false
    try {
        $LockData = Get-Content -LiteralPath $LockPath -Raw | ConvertFrom-Json
        if ($null -ne $LockData.pid) {
            $OwnerIsActive = $null -ne (Get-Process -Id ([int]$LockData.pid) -ErrorAction SilentlyContinue)
        }
    } catch {
        $OwnerIsActive = $false
    }
    if ($OwnerIsActive) {
        throw "The existing lock belongs to an active process: $LockPath"
    }
    Remove-Item -LiteralPath $LockPath -Force
}

$LockStream = $null
$ExitCode = 1
$WrapperStartedAt = Get-Date
try {
    $LockStream = [System.IO.File]::Open(
        $LockPath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::Write,
        [System.IO.FileShare]::None
    )
    $LockPayload = [System.Text.Encoding]::UTF8.GetBytes(
        (@{ pid = $PID; created_at = (Get-Date).ToString("o") } | ConvertTo-Json -Compress)
    )
    $LockStream.Write($LockPayload, 0, $LockPayload.Length)
    $LockStream.Flush()

    $Stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
    $LogPath = Join-Path $LogDir "operations_$Stamp.log"
    $DependencyLogLines = @(
        "dependency_task_name: $DependencyTaskName",
        "dependency_wait_minutes: $DependencyWaitMinutes",
        "dependency_poll_seconds: $DependencyPollSeconds"
    )
    $DependencyResult = Wait-AutonomousDependencyTask `
        -TaskName $DependencyTaskName `
        -WaitMinutes $DependencyWaitMinutes `
        -PollSeconds $DependencyPollSeconds
    $DependencyLogLines += @($DependencyResult.LogLines)
    $RunAutonomous = [bool]$DependencyResult.ShouldProceed
    if (-not $RunAutonomous) {
        $ExitCode = 125
    }

    $Arguments = @(
        "main.py",
        "autonomous-operations",
        "--ticker", $Ticker,
        "--dry-run",
        "--state-path", $ResolvedStatePath,
        "--task-queue-path", $ResolvedTaskQueuePath,
        "--report-dir", $ResolvedReportDir,
        "--max-runtime-minutes", $MaxRuntimeMinutes.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    if ($Force) {
        $Arguments += "--force"
    }

    if ($RunAutonomous) {
        $QuotedArguments = ($Arguments | ForEach-Object {
            '"' + $_.Replace('"', '\"') + '"'
        }) -join " "
        $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
        $StartInfo.FileName = $Python
        $StartInfo.Arguments = $QuotedArguments
        $StartInfo.WorkingDirectory = $ProjectRoot
        $StartInfo.UseShellExecute = $false
        $StartInfo.CreateNoWindow = $true
        $StartInfo.RedirectStandardOutput = $true
        $StartInfo.RedirectStandardError = $true
        $Process = New-Object System.Diagnostics.Process
        $Process.StartInfo = $StartInfo
        if (-not $Process.Start()) {
            throw "Failed to start Autonomous Operations."
        }
        $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
        $StderrTask = $Process.StandardError.ReadToEndAsync()

        $TimeoutMilliseconds = [int][Math]::Ceiling($MaxRuntimeMinutes * 60 * 1000)
        if (-not $Process.WaitForExit($TimeoutMilliseconds)) {
            try { $Process.Kill() } catch { }
            $Process.WaitForExit()
            $ExitCode = 124
            $TimeoutMessage = "Autonomous Operations exceeded the wrapper timeout."
        } else {
            $Process.WaitForExit()
            $ExitCode = [int]$Process.ExitCode
            $TimeoutMessage = $null
        }
        $StdoutText = $StdoutTask.GetAwaiter().GetResult()
        $StderrText = $StderrTask.GetAwaiter().GetResult()
        if ($null -ne $TimeoutMessage) {
            $StderrText = "$StderrText`r`n$TimeoutMessage".Trim()
        }
    } else {
        $StdoutText = ""
        $StderrText = "Dependency wait timed out; Autonomous Operations was not started."
    }

    $LogLines = @(
        "Autonomous Operations wrapper",
        "started_at: $($WrapperStartedAt.ToString('o'))",
        "completed_at: $((Get-Date).ToString('o'))",
        "project_root: $ProjectRoot",
        "ticker: $Ticker",
        "max_runtime_minutes: $MaxRuntimeMinutes",
        "exit_code: $ExitCode",
        ""
    )
    $LogLines += $DependencyLogLines
    $LogLines += @(
        "",
        "STDOUT"
    )
    if (-not [string]::IsNullOrWhiteSpace($StdoutText)) {
        $LogLines += $StdoutText.TrimEnd() -split "`r?`n"
    }
    $LogLines += @("", "STDERR")
    if (-not [string]::IsNullOrWhiteSpace($StderrText)) {
        $LogLines += $StderrText.TrimEnd() -split "`r?`n"
    }
    $LogLines | Set-Content -LiteralPath $LogPath -Encoding UTF8
    $LogLines | ForEach-Object { Write-Host $_ }
} finally {
    if ($null -ne $LockStream) {
        $LockStream.Dispose()
    }
    if (Test-Path -LiteralPath $LockPath) {
        Remove-Item -LiteralPath $LockPath -Force
    }
}

exit $ExitCode
