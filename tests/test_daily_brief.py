from types import SimpleNamespace

from stock_ai.app.main import _build_today_signal


def _risk_scores(surge: int = 20, crash: int = 20):
    return {
        "surge": SimpleNamespace(score=surge),
        "crash": SimpleNamespace(score=crash),
    }


def _percentiles(surge_bucket: str = "Low", crash_bucket: str = "Low"):
    values = {"Low": 0.30, "Medium": 0.75, "High": 0.92}
    return {
        "surge": {
            "percentile": values[surge_bucket],
            "relative_bucket": surge_bucket,
        },
        "crash": {
            "percentile": values[crash_bucket],
            "relative_bucket": crash_bucket,
        },
    }


def test_today_signal_bullish_when_binary_and_surge_risk_align():
    signal = _build_today_signal(
        prediction_label="up",
        up_probability=0.62,
        down_probability=0.38,
        risk_scores=_risk_scores(surge=45, crash=20),
        risk_percentiles=_percentiles(surge_bucket="High", crash_bucket="Low"),
        regime_state=SimpleNamespace(name="Bull Trend"),
        anomaly_context=SimpleNamespace(score=10),
    )

    assert signal["signal"] == "Bullish"
    assert signal["confidence"] == "High"


def test_today_signal_high_risk_when_binary_down_and_crash_risk_high():
    signal = _build_today_signal(
        prediction_label="down",
        up_probability=0.37,
        down_probability=0.63,
        risk_scores=_risk_scores(surge=20, crash=50),
        risk_percentiles=_percentiles(surge_bucket="Low", crash_bucket="High"),
        regime_state=SimpleNamespace(name="High Volatility"),
        anomaly_context=SimpleNamespace(score=20),
    )

    assert signal["signal"] == "High Risk"
    assert signal["confidence"] == "High"


def test_today_signal_caution_when_binary_up_but_crash_risk_high():
    signal = _build_today_signal(
        prediction_label="up",
        up_probability=0.61,
        down_probability=0.39,
        risk_scores=_risk_scores(surge=20, crash=50),
        risk_percentiles=_percentiles(surge_bucket="Low", crash_bucket="High"),
        regime_state=SimpleNamespace(name="High Volatility"),
        anomaly_context=SimpleNamespace(score=20),
    )

    assert signal["signal"] == "Caution"
    assert signal["confidence"] == "Medium"


def test_today_signal_neutral_when_binary_probabilities_are_close():
    signal = _build_today_signal(
        prediction_label="up",
        up_probability=0.51,
        down_probability=0.49,
        risk_scores=_risk_scores(surge=20, crash=20),
        risk_percentiles=_percentiles(surge_bucket="Low", crash_bucket="Low"),
        regime_state=SimpleNamespace(name="Sideways"),
        anomaly_context=SimpleNamespace(score=10),
    )

    assert signal["signal"] == "Neutral"
    assert signal["confidence"] == "Low"


def test_today_signal_watch_only_when_both_tail_risks_are_high():
    signal = _build_today_signal(
        prediction_label="up",
        up_probability=0.62,
        down_probability=0.38,
        risk_scores=_risk_scores(surge=60, crash=60),
        risk_percentiles=_percentiles(surge_bucket="High", crash_bucket="High"),
        regime_state=SimpleNamespace(name="High Volatility"),
        anomaly_context=SimpleNamespace(score=60),
    )

    assert signal["signal"] == "Watch Only"
    assert signal["confidence"] == "Medium"


def test_today_signal_warn_health_keeps_market_signal_but_adds_note():
    signal = _build_today_signal(
        prediction_label="up",
        up_probability=0.62,
        down_probability=0.38,
        risk_scores=_risk_scores(surge=45, crash=20),
        risk_percentiles=_percentiles(surge_bucket="High", crash_bucket="Low"),
        regime_state=SimpleNamespace(name="Bull Trend"),
        anomaly_context=SimpleNamespace(score=10),
        operations_status="WARN",
    )

    assert signal["signal"] == "Bullish"
    assert "Operations health is WARN" in signal["reason"]


def test_today_signal_fail_health_forces_watch_only():
    signal = _build_today_signal(
        prediction_label="up",
        up_probability=0.62,
        down_probability=0.38,
        risk_scores=_risk_scores(surge=45, crash=20),
        risk_percentiles=_percentiles(surge_bucket="High", crash_bucket="Low"),
        regime_state=SimpleNamespace(name="Bull Trend"),
        anomaly_context=SimpleNamespace(score=10),
        operations_status="FAIL",
    )

    assert signal["signal"] == "Watch Only"
    assert signal["confidence"] == "Low"
