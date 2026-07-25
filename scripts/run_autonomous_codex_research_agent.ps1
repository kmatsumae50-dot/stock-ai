[CmdletBinding()]
param(
    [string]$Ticker = "7203.T",
    [double]$MaxRuntimeMinutes = 180,
    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$ExpectedRoot = "C:\Users\kmats\Documents\ai-stock\stock-ai"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Main = Join-Path $ProjectRoot "main.py"
$LogDir = Join-Path $ProjectRoot "logs\autonomous"
$LockPath = Join-Path $ProjectRoot "reports\autonomous\.codex_research_agent_wrapper.lock"

if ($ProjectRoot -ne $ExpectedRoot) {
    throw "Refusing to run outside the formal project root. Actual: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Formal project virtual environment is missing: $Python"
}
if ($MaxRuntimeMinutes -le 0 -or $MaxRuntimeMinutes -gt 240) {
    throw "MaxRuntimeMinutes must be in (0, 240]."
}
if (Test-Path -LiteralPath $LockPath) {
    Write-Host "Codex Research Agent wrapper lock exists: $LockPath"
    exit 125
}

New-Item -ItemType Directory -Force -Path $LogDir, (Split-Path -Parent $LockPath) | Out-Null
Set-Location -LiteralPath $ProjectRoot
$lockPayload = @{ pid = $PID; created_at = (Get-Date).ToString("o"); wrapper = $true } | ConvertTo-Json
Set-Content -LiteralPath $LockPath -Value $lockPayload -Encoding UTF8
$stamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$logPath = Join-Path $LogDir "codex_agent_$stamp.log"

try {
    $arguments = @(
        $Main,
        "autonomous-codex-agent",
        "--ticker", $Ticker,
        "--max-runtime-minutes", $MaxRuntimeMinutes.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    if ($DryRun) { $arguments += "--dry-run" }
    $argumentText = ($arguments | ForEach-Object {
        if ($_ -match "\s") { '"' + ($_ -replace '"', '\"') + '"' } else { [string]$_ }
    }) -join " "
    $startInfo = New-Object System.Diagnostics.ProcessStartInfo
    $startInfo.FileName = $Python
    $startInfo.Arguments = $argumentText
    $startInfo.WorkingDirectory = $ProjectRoot
    $startInfo.UseShellExecute = $false
    $startInfo.CreateNoWindow = $true
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $startInfo
    if (-not $process.Start()) {
        throw "Failed to start Autonomous Codex Research Agent."
    }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $timeoutMilliseconds = [int][Math]::Ceiling($MaxRuntimeMinutes * 60 * 1000)
    if (-not $process.WaitForExit($timeoutMilliseconds)) {
        try { $process.Kill() } catch { }
        $process.WaitForExit()
        "Codex Research Agent exceeded the wrapper timeout of $MaxRuntimeMinutes minutes." |
            Tee-Object -FilePath $logPath -Append
        exit 124
    }
    $process.WaitForExit()
    $stdoutText = $stdoutTask.GetAwaiter().GetResult()
    $stderrText = $stderrTask.GetAwaiter().GetResult()
    if (-not [string]::IsNullOrWhiteSpace($stdoutText)) {
        $stdoutText.TrimEnd() -split "`r?`n" | Tee-Object -FilePath $logPath -Append
    }
    if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
        $stderrText.TrimEnd() -split "`r?`n" | Tee-Object -FilePath $logPath -Append
    }
    $exitCode = [int]$process.ExitCode
    "exit_code: $exitCode" | Tee-Object -FilePath $logPath -Append
    exit $exitCode
} finally {
    Remove-Item -LiteralPath $LockPath -Force -ErrorAction SilentlyContinue
}
