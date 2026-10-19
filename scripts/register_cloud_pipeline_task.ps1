param(
    [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$RunnerScript = Join-Path $ScriptDir "run_cloud_pipeline.ps1"
$HiddenRunner = Join-Path $ScriptDir "run_hidden.vbs"

if (-not (Test-Path -LiteralPath $RunnerScript)) {
    throw "run_cloud_pipeline.ps1 was not found: $RunnerScript"
}
if (-not (Test-Path -LiteralPath $HiddenRunner)) {
    throw "run_hidden.vbs was not found: $HiddenRunner"
}

$WscriptExe = Join-Path $env:SystemRoot "System32\wscript.exe"
$PowerShellExe = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$Arguments = @(
    "//nologo",
    "`"$HiddenRunner`"",
    "`"$PowerShellExe`"",
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$RunnerScript`""
) -join " "

$TasksToRegister = @(
    @{
        Name = "stock-ai cloud pipeline morning"
        Time = "07:30"
        Description = "Stock-AI morning market open pipeline (Daily 07:30 JST, StartWhenAvailable)."
    },
    @{
        Name = "stock-ai cloud pipeline evening"
        Time = "16:00"
        Description = "Stock-AI evening market close pipeline (Daily 16:00 JST, StartWhenAvailable)."
    }
)

foreach ($t in $TasksToRegister) {
    $TaskName = $t.Name
    $TimeStr = $t.Time
    $ParsedTime = [datetime]::ParseExact($TimeStr, "HH:mm", [System.Globalization.CultureInfo]::InvariantCulture)
    $Description = $t.Description

    Write-Host "Registering task: $TaskName (Daily at $TimeStr JST)..."

    $ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $ExistingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    }

    $Action = New-ScheduledTaskAction `
        -Execute $WscriptExe `
        -Argument $Arguments `
        -WorkingDirectory $ProjectRoot

    $Trigger = New-ScheduledTaskTrigger -Daily -At $ParsedTime
    $Settings = New-ScheduledTaskSettingsSet `
        -StartWhenAvailable `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Hours 2)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description $Description | Out-Null

    Write-Host "✅ Registered: $TaskName (Daily $TimeStr, StartWhenAvailable=True)"
}

Write-Host "`nAll Stock-AI daily scheduled tasks are registered successfully."
