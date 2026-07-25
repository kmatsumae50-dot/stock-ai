param(
    [string]$Ticker = "7203.T",
    [int]$Tail = 40,
    [switch]$Json
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$PythonPath = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Error "Missing project Python: $PythonPath"
    exit 1
}

$Arguments = @(
    "main.py",
    "health-check",
    "--ticker",
    $Ticker,
    "--expected-root",
    $ProjectRoot,
    "--tail",
    "$Tail"
)

if ($Json.IsPresent) {
    $Arguments += "--json"
}

& $PythonPath @Arguments
exit $LASTEXITCODE
