param(
    [string[]]$Ticker = @(),
    [string]$TickerFile = "",
    [string]$Start = "2010-01-01",
    [string]$MulticlassThresholdProfile = "original_5_1",
    [string]$MulticlassWeightMode = "none",
    [switch]$FetchNews,
    [string]$NewsQuery = "",
    [string]$NewsQueryProfile = "",
    [switch]$DailyNewsProfiles,
    [switch]$AllNewsProfiles,
    [int]$NewsLimit = 10,
    [switch]$ContinueOnError
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding
$env:PYTHONIOENCODING = "utf-8"

$ScriptDir = Split-Path -Parent $PSCommandPath
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")).Path
$LogDir = Join-Path $ProjectRoot "logs"
$LogPath = Join-Path $LogDir "daily_predict.log"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location -LiteralPath $ProjectRoot

function Write-Log {
    param([object]$Message)

    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $Text = if ($null -eq $Message) { "" } else { $Message.ToString() }
    $Line = "[$Timestamp] $Text"
    Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
    Write-Host $Line
}

function Resolve-TickerFile {
    if (-not [string]::IsNullOrWhiteSpace($TickerFile)) {
        if ([System.IO.Path]::IsPathRooted($TickerFile)) {
            return $TickerFile
        }
        return (Join-Path $ProjectRoot $TickerFile)
    }

    return (Join-Path $ProjectRoot "configs\daily_predict_tickers.txt")
}

function Get-TargetTickers {
    $Items = @()

    if ($Ticker.Count -gt 0) {
        $Items += $Ticker
    } else {
        $ResolvedTickerFile = Resolve-TickerFile
        if (Test-Path -LiteralPath $ResolvedTickerFile) {
            foreach ($Line in Get-Content -LiteralPath $ResolvedTickerFile) {
                $Body = ($Line -split "#", 2)[0]
                foreach ($Item in ($Body -split ",")) {
                    $Clean = $Item.Trim()
                    if (-not [string]::IsNullOrWhiteSpace($Clean)) {
                        $Items += $Clean
                    }
                }
            }
        }
    }

    if ($Items.Count -eq 0) {
        $Items += "7203.T"
    }

    return @($Items | ForEach-Object { $_.Trim() } | Where-Object { $_ } | Select-Object -Unique)
}

function Test-VenvOrigin {
    param([string]$VenvDir)

    $ConfigPath = Join-Path $VenvDir "pyvenv.cfg"
    if (-not (Test-Path -LiteralPath $ConfigPath)) {
        return
    }

    $ConfigText = Get-Content -LiteralPath $ConfigPath -Raw
    if ($ConfigText -match "command\s*=.*\s-m\s+venv\s+(?<path>.+)") {
        $OriginalPath = $Matches["path"].Trim()
        try {
            $Expected = [System.IO.Path]::GetFullPath($VenvDir).TrimEnd("\")
            $Actual = [System.IO.Path]::GetFullPath($OriginalPath).TrimEnd("\")
            if ($Actual -ne $Expected) {
                Write-Log "WARNING: .venv was created at '$OriginalPath'. If Python fails, recreate .venv in '$ProjectRoot'."
            }
        } catch {
            Write-Log "WARNING: Could not validate .venv origin: $($_.Exception.Message)"
        }
    }
}

function Enable-ProjectVenv {
    $VenvDir = Join-Path $ProjectRoot ".venv"
    $ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"

    if (-not (Test-Path -LiteralPath $ActivateScript)) {
        throw "Missing virtual environment activation script: $ActivateScript"
    }

    Test-VenvOrigin -VenvDir $VenvDir
    . $ActivateScript

    $PythonCommand = Get-Command python -ErrorAction Stop
    Write-Log "Python command: $($PythonCommand.Source)"
    & python --version *>&1 | ForEach-Object { Write-Log $_ }
    if ($LASTEXITCODE -ne 0) {
        throw "The activated Python failed to start. Recreate .venv in the current project folder."
    }
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )

    Write-Log "START: $Name"
    $PreviousErrorActionPreference = $ErrorActionPreference
    try {
        # Native Python warnings on stderr, such as LightGBM/SHAP UserWarning, are logged
        # but should not fail the job unless Python exits with a non-zero code.
        $ErrorActionPreference = "Continue"
        & python @Arguments *>&1 | ForEach-Object { Write-Log $_ }
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }
    if ($ExitCode -ne 0) {
        throw "$Name failed with exit code $ExitCode"
    }
    Write-Log "DONE: $Name"
}

function Invoke-OptionalPythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments,
        [string]$WarnOnText = ""
    )

    Write-Log "NEWS START: $Name"
    $PreviousErrorActionPreference = $ErrorActionPreference
    $Output = @()
    try {
        $ErrorActionPreference = "Continue"
        $Output = & python @Arguments *>&1
        $ExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $PreviousErrorActionPreference
    }

    foreach ($Line in $Output) {
        Write-Log $Line
    }

    if ($ExitCode -ne 0) {
        Write-Log "NEWS WARN: $Name failed with exit code $ExitCode"
        return $false
    }

    $OutputText = ($Output | Out-String)
    if (-not [string]::IsNullOrWhiteSpace($WarnOnText) -and $OutputText -match $WarnOnText) {
        Write-Log "NEWS WARN: $Name completed with quality warning"
        return $false
    }

    Write-Log "NEWS OK: $Name"
    return $true
}

try {
    Write-Log "============================================================"
    Write-Log "daily_predict started"
    Write-Log "Project root: $ProjectRoot"
    Write-Log "Log path: $LogPath"
    Write-Log "Prediction storage mode: same ticker/date/target/model rows are overwritten in SQLite."
    Write-Log "Multiclass threshold profile: $MulticlassThresholdProfile"
    Write-Log "Multiclass weight mode: $MulticlassWeightMode"
    Write-Log "FetchNews enabled: $($FetchNews.IsPresent)"
    Write-Log "News query: $(if ([string]::IsNullOrWhiteSpace($NewsQuery)) { '<ticker default>' } else { $NewsQuery })"
    Write-Log "News query profile: $(if ([string]::IsNullOrWhiteSpace($NewsQueryProfile)) { '<none>' } else { $NewsQueryProfile })"
    Write-Log "Daily news profiles: $($DailyNewsProfiles.IsPresent)"
    Write-Log "All news profiles: $($AllNewsProfiles.IsPresent)"
    $NewsProfileMode = if ($AllNewsProfiles.IsPresent) {
        "all profiles"
    } elseif ($DailyNewsProfiles.IsPresent) {
        "daily profiles"
    } elseif (-not [string]::IsNullOrWhiteSpace($NewsQueryProfile)) {
        "single profile"
    } else {
        "single query/default"
    }
    Write-Log "News profile mode: $NewsProfileMode"
    Write-Log "News limit: $NewsLimit"

    if ($DailyNewsProfiles.IsPresent -and $AllNewsProfiles.IsPresent) {
        throw "Use either -DailyNewsProfiles or -AllNewsProfiles, not both."
    }

    Enable-ProjectVenv
    $TargetTickers = Get-TargetTickers
    Write-Log "Tickers: $($TargetTickers -join ', ')"
    Invoke-PythonStep "download-external" @("main.py", "download-external", "--start", $Start)

    $FailedTickers = @()
    foreach ($CurrentTicker in $TargetTickers) {
        try {
            Write-Log "----- ticker: $CurrentTicker -----"
            Invoke-PythonStep "download $CurrentTicker" @("main.py", "download", "--ticker", $CurrentTicker, "--start", $Start)
            if ($FetchNews.IsPresent) {
                $EffectiveNewsLimit = [Math]::Max(1, $NewsLimit)
                $FetchArgs = @("main.py", "fetch-news", "--ticker", $CurrentTicker, "--limit", "$EffectiveNewsLimit")
                if ($AllNewsProfiles.IsPresent) {
                    $FetchArgs += @("--all-profiles")
                } elseif ($DailyNewsProfiles.IsPresent) {
                    $FetchArgs += @("--daily-profiles")
                } elseif (-not [string]::IsNullOrWhiteSpace($NewsQueryProfile)) {
                    $FetchArgs += @("--query-profile", $NewsQueryProfile)
                }
                if (-not $AllNewsProfiles.IsPresent -and -not $DailyNewsProfiles.IsPresent -and -not [string]::IsNullOrWhiteSpace($NewsQuery)) {
                    $FetchArgs += @("--query", $NewsQuery)
                }
                $FetchOk = Invoke-OptionalPythonStep "fetch-news $NewsProfileMode $CurrentTicker" $FetchArgs
                $CheckOk = Invoke-OptionalPythonStep "check-news $CurrentTicker" @("main.py", "check-news", "--ticker", $CurrentTicker) "news quality status:\s*(WARN|FAIL)"
                if (-not $FetchOk -or -not $CheckOk) {
                    Write-Log "NEWS WARN: optional news step had warnings for $CurrentTicker; continuing prediction flow"
                }
            } else {
                Write-Log "NEWS SKIP: FetchNews disabled for $CurrentTicker"
            }
            Invoke-PythonStep "build-features $CurrentTicker" @("main.py", "build-features", "--ticker", $CurrentTicker)
            Invoke-PythonStep "data-freshness $CurrentTicker" @("main.py", "data-freshness", "--ticker", $CurrentTicker)
            try {
                Invoke-PythonStep "train-binary $CurrentTicker" @("main.py", "train-binary", "--ticker", $CurrentTicker)
                Invoke-PythonStep "predict-binary $CurrentTicker" @("main.py", "predict-binary", "--ticker", $CurrentTicker)
                Invoke-PythonStep "train-multiclass $CurrentTicker" @("main.py", "train-multiclass", "--ticker", $CurrentTicker, "--threshold-profile", $MulticlassThresholdProfile, "--weight-mode", $MulticlassWeightMode)
                Invoke-PythonStep "predict-multiclass $CurrentTicker" @("main.py", "predict-multiclass", "--ticker", $CurrentTicker, "--threshold-profile", $MulticlassThresholdProfile, "--weight-mode", $MulticlassWeightMode)
                Write-Log "Binary and multiclass prediction saves confirmed for $CurrentTicker"
            } catch {
                Write-Log "C2C WARN: Binary/multiclass step for $CurrentTicker had an issue: $($_.Exception.Message); proceeding to overnight step"
            }

            # Overnight (Close-to-Open) shadow production pillar prediction
            try {
                $OvernightModelPath = Join-Path $ProjectRoot "models\artifacts\lightgbm_overnight_binary_${CurrentTicker}.joblib"
                if (Test-Path -LiteralPath $OvernightModelPath) {
                    Invoke-PythonStep "predict-overnight $CurrentTicker" @("main.py", "predict-overnight", "--ticker", $CurrentTicker)
                    Write-Log "Overnight prediction save confirmed for $CurrentTicker"
                } else {
                    Write-Log "OVERNIGHT SKIP: Model artifact not found for $CurrentTicker ($OvernightModelPath)"
                }
            } catch {
                Write-Log "OVERNIGHT WARN: predict-overnight for $CurrentTicker failed: $($_.Exception.Message); continuing main flow"
            }
        } catch {
            Write-Log "ERROR for ${CurrentTicker}: $($_.Exception.Message)"
            Write-Log $_.ScriptStackTrace
            $FailedTickers += $CurrentTicker
            if (-not $ContinueOnError.IsPresent) {
                throw
            }
        }
    }

    if ($FailedTickers.Count -gt 0) {
        throw "Failed tickers: $($FailedTickers -join ', ')"
    }

    # Overnight Paper Trader portfolio simulation & ledger update
    try {
        Write-Log "START: run-overnight-paper-trader"
        Invoke-PythonStep "run-overnight-paper-trader" @("main.py", "run-overnight-paper-trader")
        Write-Log "DONE: run-overnight-paper-trader"
    } catch {
        Write-Log "PAPER TRADER WARN: run-overnight-paper-trader failed: $($_.Exception.Message); continuing"
    }

    Write-Log "daily_predict finished successfully"
    exit 0
} catch {
    Write-Log "daily_predict failed: $($_.Exception.Message)"
    Write-Log $_.ScriptStackTrace
    exit 1
}

