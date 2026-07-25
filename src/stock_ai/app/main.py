from __future__ import annotations

import json
import importlib
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from stock_ai.backtest.prediction_history import (
    diagnose_missing_live_dates,
    evaluate_prediction_history,
    summarize_prediction_operations,
)
from stock_ai.backtest.append_only_history import (
    build_append_only_health_visibility,
    build_append_only_run_detail,
    summarize_append_only_history_for_ui,
)
from stock_ai.config import PROJECT_ROOT, load_settings
from stock_ai.data.universe import TickerInfo, load_universe
from stock_ai.explain.reason_builder import build_basic_reasons
from stock_ai.features.missingness import classify_feature_missingness
from stock_ai.features.technical import add_technical_features, available_technical_columns
from stock_ai.health.checks import run_health_check
from stock_ai.labels.targets import BINARY_CLASS_NAMES, CLASS_NAMES
from stock_ai.storage.parquet_store import read_parquet
from stock_ai.storage.sqlite_store import read_latest_predictions
from stock_ai.utils.safe_cast import safe_float, safe_int


st.set_page_config(page_title="stock-ai", layout="wide")

LOGGER = logging.getLogger(__name__)

MULTICLASS_PROBABILITY_FIELDS = [
    (4, "surge", "probability_surge", "#2f855a"),
    (3, "up", "probability_up", "#68a357"),
    (2, "flat", "probability_flat", "#8a8f98"),
    (1, "down", "probability_down", "#d98c45"),
    (0, "crash", "probability_crash", "#d95f5f"),
]

RISK_MULTICLASS_OPTIONS = [
    ("original_5_1", "none"),
    ("practical_3_0_5", "balanced"),
    ("practical_3_0_5", "manual_tail_boost"),
    ("quantile", "balanced"),
]
DEFAULT_RISK_MULTICLASS_OPTION = ("practical_3_0_5", "balanced")
DAILY_BRIEF_RISK_BUCKET_MODE = "quantile_60_90"
PRODUCTION_BINARY_MODEL_NAME = "lightgbm_binary"
PRODUCTION_MULTICLASS_MODEL_NAME = "lightgbm_multiclass"


def main() -> None:
    settings = load_settings()
    universe = load_universe()

    st.title("stock-ai")
    st.caption("Japanese stock analysis MVP")

    ticker = _select_ticker(universe)
    data, source = _load_dataset(ticker, settings)

    if data.empty:
        st.info("No data yet. Run download and build-features first.")
        st.code(
            f"python main.py download --ticker {ticker} --start 2010-01-01\n"
            f"python main.py build-features --ticker {ticker}",
            language="powershell",
        )
        return

    data = _prepare_for_display(data)
    data = _add_regime_context(data)
    latest = data.iloc[-1]

    _render_status(ticker, source, data, latest)

    (
        daily_brief_tab,
        chart_tab,
        indicators_tab,
        prediction_tab,
        history_tab,
        backtest_tab,
        evaluation_tab,
        model_compare_tab,
        ablation_tab,
        similar_tab,
        cross_ticker_tab,
        lead_lag_tab,
        anomaly_tab,
        operations_tab,
        data_tab,
    ) = st.tabs(
        [
            "Daily Brief",
            "Chart",
            "Technical",
            "Prediction",
            "Prediction History",
            "Backtest",
            "Evaluation",
            "Model Compare",
            "Ablation",
            "Similar Patterns",
            "Cross-Ticker Similarity",
            "Lead-Lag",
            "Anomaly",
            "Operations",
            "Data",
        ]
    )

    # Render lightweight operational views before expensive analysis tabs.
    with data_tab:
        _render_data_tab(ticker, data, settings)

    with history_tab:
        _render_prediction_history_tab(ticker, data, settings.paths.database_path, settings)

    with daily_brief_tab:
        _render_daily_brief_tab(ticker, data, settings)

    with chart_tab:
        _render_chart_tab(ticker, data, latest, source)

    with indicators_tab:
        _render_indicator_tab(ticker, data, latest, source)

    with prediction_tab:
        _render_prediction_tab(ticker, data, latest, settings.paths.database_path, settings)

    with backtest_tab:
        _render_backtest_tab(ticker, data, settings)

    with evaluation_tab:
        _render_evaluation_tab(ticker, data, settings)

    with model_compare_tab:
        _render_model_compare_tab(ticker, data, settings)

    with ablation_tab:
        _render_ablation_tab(ticker, data, settings)

    with similar_tab:
        _render_similar_patterns_tab(ticker, data)

    with cross_ticker_tab:
        _render_cross_ticker_similarity_tab(ticker, data, settings)

    with lead_lag_tab:
        _render_lead_lag_tab(ticker, data, settings)

    with anomaly_tab:
        _render_anomaly_tab(ticker, data, settings)

    with operations_tab:
        _render_operations_tab(ticker, data, settings)


def _select_ticker(universe: list[TickerInfo]) -> str:
    options = [ticker.code for ticker in universe]
    names = {ticker.code: ticker.name for ticker in universe}

    selected = st.sidebar.selectbox(
        "Ticker",
        options=options,
        format_func=lambda code: f"{code}  {names.get(code, '')}".strip(),
        key="sidebar_ticker_select",
    )
    manual = st.sidebar.text_input("Manual ticker", value="", key="sidebar_manual_ticker_input")
    st.sidebar.caption("Example: 7203.T")
    return manual.strip() or selected


def _load_dataset(ticker: str, settings) -> tuple[pd.DataFrame, str]:
    feature_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
    raw_path = settings.paths.raw_data_dir / f"{ticker}.parquet"

    if feature_path.exists():
        data = read_parquet(feature_path)
        data = _attach_external_market_features(data, settings)
        data = _attach_earnings_event_features(data, ticker, settings)
        data = _attach_news_features(data, ticker, settings)
        return _attach_narrative_features(data, ticker, settings), "features"
    if raw_path.exists():
        raw = read_parquet(raw_path)
        features = add_technical_features(
            raw,
            moving_average_windows=settings.features.moving_average_windows,
            rsi_window=settings.features.rsi_window,
            volatility_window=settings.features.volatility_window,
            bollinger_window=settings.features.bollinger_window,
        )
        features = _attach_external_market_features(features, settings)
        features = _attach_earnings_event_features(features, ticker, settings)
        features = _attach_news_features(features, ticker, settings)
        return _attach_narrative_features(features, ticker, settings), "raw + temporary features"
    return pd.DataFrame(), "missing"


def _prepare_for_display(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"])
    return result.sort_values("date").reset_index(drop=True)


def _attach_external_market_features(data: pd.DataFrame, settings) -> pd.DataFrame:
    try:
        return _external_module().add_external_market_features(data, settings.paths.raw_data_dir)
    except Exception as error:
        st.caption(f"External market features could not be loaded: {error}")
        return data.copy()


def _attach_earnings_event_features(data: pd.DataFrame, ticker: str, settings) -> pd.DataFrame:
    try:
        events = _events_module()
        return events.add_earnings_event_features(
            data,
            events.default_events_dir(settings.paths.raw_data_dir.parent),
            ticker,
        )
    except Exception as error:
        st.caption(f"Earnings event features could not be loaded: {error}")
        return data.copy()


def _attach_news_features(data: pd.DataFrame, ticker: str, settings) -> pd.DataFrame:
    try:
        news = _news_module()
        return news.add_news_features(
            data,
            news.default_news_dir(settings.paths.raw_data_dir.parent),
            ticker,
        )
    except Exception as error:
        st.caption(f"News features could not be loaded: {error}")
        return data.copy()


def _attach_narrative_features(data: pd.DataFrame, ticker: str, settings) -> pd.DataFrame:
    try:
        narrative = _narrative_module()
        news = _news_module()
        return narrative.add_narrative_features(
            data,
            news.default_news_dir(settings.paths.raw_data_dir.parent),
            ticker,
        )
    except Exception as error:
        st.caption(f"Narrative features could not be loaded: {error}")
        return data.copy()


def _add_regime_context(data: pd.DataFrame) -> pd.DataFrame:
    try:
        return _regime_module().add_regime_labels(data)
    except Exception as error:
        st.caption(f"Market regime could not be calculated: {error}")
        return data.copy()


def _render_status(ticker: str, source: str, data: pd.DataFrame, latest: pd.Series) -> None:
    latest_date = pd.to_datetime(latest["date"]).strftime("%Y-%m-%d")
    return_1d = latest.get("return_1d")

    cols = st.columns(5)
    cols[0].metric("Ticker", ticker)
    cols[1].metric("Latest date", latest_date)
    cols[2].metric("Close", f"{latest['close']:,.0f}")
    cols[3].metric(
        "1 day return",
        "-" if pd.isna(return_1d) else f"{return_1d * 100:.2f}%",
    )

    st.caption(f"Data source: {source} / rows: {len(data):,}")
    st.caption(f"Project root: {PROJECT_ROOT}")


def _build_price_chart(data: pd.DataFrame, ticker: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.74, 0.26],
    )
    fig.add_trace(
        go.Candlestick(
            x=data["date"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name=ticker,
        ),
        row=1,
        col=1,
    )

    for column, label in (("ma_5", "MA 5"), ("ma_25", "MA 25"), ("ma_75", "MA 75")):
        if column in data.columns:
            fig.add_trace(
                go.Scatter(x=data["date"], y=data[column], mode="lines", name=label),
                row=1,
                col=1,
            )

    fig.add_trace(
        go.Bar(x=data["date"], y=data["volume"], name="Volume", marker_color="#7f8c8d"),
        row=2,
        col=1,
    )
    fig.update_layout(height=620, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def _render_technical_price_overview(
    ticker: str,
    data: pd.DataFrame,
    latest: pd.Series,
    source: str,
) -> None:
    st.subheader("Technical / Price Overview")
    st.caption(
        "This view describes the latest price and technical state. "
        "Use Prediction for forecasts and Anomaly for unusual-condition details."
    )
    price_cols = st.columns(5)
    price_cols[0].metric("Latest feature date", _pattern_latest_feature_date(data))
    price_cols[1].metric("Ticker", ticker)
    price_cols[2].metric("Latest close", _format_price_value(latest.get("close")))
    price_cols[3].metric("Return 1D", _format_percent(latest.get("return_1d")))
    price_cols[4].metric("Current trend", _technical_trend_label(latest))

    indicator_cols = st.columns(4)
    indicator_cols[0].metric("MA25 difference", _format_percent(latest.get("close_vs_ma_25")))
    indicator_cols[1].metric("RSI 14", _format_number(latest.get("rsi_14"), digits=1))
    indicator_cols[2].metric("Volatility 20", _format_percent(latest.get("volatility_20")))
    indicator_cols[3].metric("Regime", str(latest.get("regime") or "Not available"))

    st.caption(f"Technical summary: {_technical_state_summary(latest)}")
    st.caption(
        f"Data freshness: latest available row {_pattern_latest_feature_date(data)} / "
        f"source {source} / {len(data):,} rows."
    )
    anomaly_note = (
        f"Stored anomaly state: {latest.get('anomaly_level')} "
        f"({_format_number(latest.get('anomaly_score'), digits=0)}/100)."
        if pd.notna(latest.get("anomaly_score"))
        else "Anomaly details are calculated and displayed in the Anomaly tab."
    )
    st.caption(f"Regime / anomaly context: {latest.get('regime') or 'Unknown regime'}. {anomaly_note}")

    with st.expander("How to read technical / chart analysis", expanded=False):
        st.write("- Technical indicators describe price conditions using historical and current data.")
        st.write("- Do not use a single indicator as a standalone trading decision.")
        st.write("- Moving averages, RSI, and volatility are state indicators, not outcome guarantees.")
        st.write("- Read them together with Prediction, Risk, Anomaly, Regime, and News.")
        st.write("- Start with the overview and open detailed charts or tables only when needed.")


def _render_chart_tab(
    ticker: str,
    data: pd.DataFrame,
    latest: pd.Series,
    source: str,
) -> None:
    _render_technical_price_overview(ticker, data, latest, source)
    st.subheader("Price / Trend Summary")
    st.write(_technical_state_summary(latest))

    range_options = {
        "3 months": 66,
        "6 months": 132,
        "1 year": 252,
        "2 years": 504,
        "All available": None,
    }
    selected_range = st.selectbox(
        "Chart range",
        options=list(range_options),
        index=2,
        key="chart_price_range_select",
    )
    chart_data = _technical_chart_range(data, range_options[selected_range])
    st.caption(f"Main chart: {selected_range} / {len(chart_data):,} rows")
    st.plotly_chart(_build_price_chart(chart_data, ticker), use_container_width=True)

    with st.expander("Price feature snapshot", expanded=False):
        st.dataframe(
            _technical_feature_snapshot(data, rows=20),
            use_container_width=True,
            hide_index=True,
        )


def _render_indicator_tab(
    ticker: str,
    data: pd.DataFrame,
    latest: pd.Series,
    source: str,
) -> None:
    _render_technical_price_overview(ticker, data, latest, source)
    st.subheader("Technical Indicator Summary")
    cols = st.columns(3)
    cols[0].metric("RSI 14", _format_number(latest.get("rsi_14"), digits=1))
    cols[1].metric("MACD", _format_number(latest.get("macd"), digits=2))
    cols[2].metric("MA25 diff", _format_percent(latest.get("close_vs_ma_25")))

    indicator_data = _technical_chart_range(data, 504)
    indicator_fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("RSI", "MACD"),
    )
    if "rsi_14" in indicator_data.columns:
        indicator_fig.add_trace(
            go.Scatter(
                x=indicator_data["date"],
                y=indicator_data["rsi_14"],
                mode="lines",
                name="RSI 14",
            ),
            row=1,
            col=1,
        )
        indicator_fig.add_hline(y=70, line_dash="dot", line_color="#c0392b", row=1, col=1)
        indicator_fig.add_hline(y=30, line_dash="dot", line_color="#2980b9", row=1, col=1)

    if {"macd", "macd_signal", "macd_hist"}.issubset(indicator_data.columns):
        indicator_fig.add_trace(
            go.Scatter(
                x=indicator_data["date"],
                y=indicator_data["macd"],
                mode="lines",
                name="MACD",
            ),
            row=2,
            col=1,
        )
        indicator_fig.add_trace(
            go.Scatter(
                x=indicator_data["date"],
                y=indicator_data["macd_signal"],
                mode="lines",
                name="Signal",
            ),
            row=2,
            col=1,
        )
        indicator_fig.add_trace(
            go.Bar(
                x=indicator_data["date"],
                y=indicator_data["macd_hist"],
                name="Hist",
                marker_color="#95a5a6",
            ),
            row=2,
            col=1,
        )

    indicator_fig.update_layout(height=540, margin=dict(l=20, r=20, t=40, b=20))
    st.caption("Indicator charts use the latest two years of available rows for a lighter initial view.")
    st.plotly_chart(indicator_fig, use_container_width=True)

    with st.expander("Feature Snapshot", expanded=False):
        st.dataframe(
            _technical_feature_snapshot(data, rows=20),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Raw technical feature table", expanded=False):
        st.dataframe(
            _technical_feature_snapshot(data, rows=100),
            use_container_width=True,
            hide_index=True,
        )


def _technical_chart_range(data: pd.DataFrame, rows: int | None) -> pd.DataFrame:
    if data.empty or rows is None:
        return data
    return data.tail(max(1, int(rows))).copy()


def _technical_feature_snapshot(data: pd.DataFrame, *, rows: int) -> pd.DataFrame:
    feature_view = [
        column
        for column in (
            "date",
            "close",
            "return_1d",
            "ma_5",
            "ma_25",
            "ma_75",
            "close_vs_ma_5",
            "close_vs_ma_25",
            "close_vs_ma_75",
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "volume_change_1d",
            "volatility_20",
            "bb_percent_b",
            "regime",
        )
        if column in data.columns
    ]
    display = data[feature_view].tail(max(1, int(rows))).copy()
    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return display.sort_values("date", ascending=False) if "date" in display.columns else display


def _format_price_value(value) -> str:
    numeric = safe_float(value)
    return "-" if numeric is None else f"{numeric:,.0f}"


def _technical_trend_label(latest: pd.Series) -> str:
    ma_5 = safe_float(latest.get("ma_5"))
    ma_25 = safe_float(latest.get("ma_25"))
    ma_75 = safe_float(latest.get("ma_75"))
    close = safe_float(latest.get("close"))
    if None in (ma_5, ma_25, ma_75, close):
        return "Not available"
    if close > ma_5 > ma_25 > ma_75:
        return "Bullish alignment"
    if close < ma_5 < ma_25 < ma_75:
        return "Bearish alignment"
    if close >= ma_25:
        return "Above MA25"
    return "Below MA25"


def _technical_state_summary(latest: pd.Series) -> str:
    trend = _technical_trend_label(latest)
    rsi = safe_float(latest.get("rsi_14"))
    if rsi is None:
        rsi_text = "RSI is unavailable"
    elif rsi >= 70:
        rsi_text = f"RSI {rsi:.1f} is in a high zone"
    elif rsi <= 30:
        rsi_text = f"RSI {rsi:.1f} is in a low zone"
    else:
        rsi_text = f"RSI {rsi:.1f} is in the middle range"
    volatility = _format_percent(latest.get("volatility_20"))
    return f"Trend is {trend}; {rsi_text}; 20-day volatility is {volatility}."


def _render_daily_brief_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("Daily Brief")
    st.caption(
        "Start here for today's quick read. Use the detailed tabs when you want to inspect the evidence behind each item."
    )
    st.caption(
        "Similarity snapshots in this brief use a lightweight recent lookback; the Similar Patterns and Cross-Ticker tabs remain the full research views."
    )

    brief = _build_daily_brief_context(ticker, data, settings)
    signal = brief.get("signal", {})
    signal_name = str(signal.get("signal") or "Watch Only")
    confidence = str(signal.get("confidence") or "Low")
    signal_reason = str(signal.get("reason") or "Saved predictions or context are incomplete.")

    _render_daily_brief_overview(brief, signal_name, confidence)
    _render_daily_brief_reading_guide()
    _render_daily_brief_attention(brief)
    _render_daily_brief_next_check(brief)

    st.subheader("Today Signal")
    if signal_name == "Bullish":
        st.success(f"Today Signal: {signal_name} / Confidence: {confidence}")
    elif signal_name in {"Bearish", "High Risk", "Watch Only"}:
        st.warning(f"Today Signal: {signal_name} / Confidence: {confidence}")
    else:
        st.info(f"Today Signal: {signal_name} / Confidence: {confidence}")
    st.write(signal_reason)
    st.caption(_daily_brief_today_signal_validation_text(ticker, signal_name))

    st.subheader("Prediction Summary")
    st.caption("Check the Prediction tab for full probability charts, SHAP drivers, and Integrated Reasoning.")
    headline_cols = st.columns(5)
    headline_cols[0].metric("Feature date", _format_date(brief.get("prediction_feature_date")))
    headline_cols[1].metric("Binary", brief.get("binary_label") or "-")
    headline_cols[2].metric(
        "Up / Down",
        f"{_format_percent(brief.get('up_probability'))} / {_format_percent(brief.get('down_probability'))}",
    )
    headline_cols[3].metric("Surge Risk", _daily_brief_risk_metric(brief, "surge"))
    headline_cols[4].metric("Crash Risk", _daily_brief_risk_metric(brief, "crash"))
    st.caption(
        "Date semantics: Feature date is the data row used by the model. "
        f"Target close date is the next trading date after it: "
        f"{_format_date(brief.get('prediction_target_date'))}."
    )

    with st.expander("Prediction details", expanded=False):
        prediction_cols = st.columns([1, 1])
        with prediction_cols[0]:
            st.caption("Binary Prediction")
            st.dataframe(_daily_brief_binary_frame(brief), use_container_width=True, hide_index=True)
        with prediction_cols[1]:
            st.caption("Multiclass Prediction")
            multiclass_frame = _daily_brief_multiclass_frame(brief)
            if multiclass_frame.empty:
                st.info("Multiclass prediction is not available for the daily default profile.")
            else:
                st.dataframe(multiclass_frame, use_container_width=True, hide_index=True)

    st.subheader("Risk Summary")
    st.caption("Check Prediction for risk reasons and Evaluation for Risk Validation / Calibration.")
    st.caption("Current risk percentiles compare today's Risk Scores with the historical Risk Validation distribution.")
    risk_frame = _daily_brief_risk_display_frame(brief.get("risk_percentiles"))
    if not risk_frame.empty:
        st.dataframe(risk_frame, use_container_width=True, hide_index=True)

    context_rows = _daily_brief_context_rows(brief)
    st.subheader("Market Context")
    st.caption("Use the dedicated Similar Patterns, Cross-Ticker, Lead-Lag, Anomaly, and Data tabs for details.")
    st.dataframe(
        pd.DataFrame(_daily_brief_filter_context_rows(context_rows, "market")),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Event / News Context")
    st.caption("Use Prediction for event/news/narrative details, and Anomaly for special-condition checks.")
    st.dataframe(
        pd.DataFrame(_daily_brief_filter_context_rows(context_rows, "event")),
        use_container_width=True,
        hide_index=True,
    )
    important_news = _daily_brief_important_news_frame(brief.get("news_context"), data)
    if not important_news.empty:
        st.caption("Recent important news")
        st.dataframe(
            important_news,
            use_container_width=True,
            hide_index=True,
            column_config=_news_url_column_config(important_news),
        )
        st.caption(
            "Price moves are measured after the news date, not treated as causality. "
            "Details: Evaluation > News Impact Validation."
        )
    news_impact_text = brief.get("news_impact_text")
    if news_impact_text:
        st.caption(f"News Impact Validation: {news_impact_text}")
    source_type_messages = brief.get("source_type_news_impact_messages") or []
    if source_type_messages:
        st.caption("Source Type News Impact")
        for message in source_type_messages[:3]:
            st.write(f"- {message}")
        st.caption("Details: Evaluation > News Impact Validation > Source Types.")
    profile_messages = brief.get("profile_news_impact_messages") or []
    if profile_messages:
        st.caption("Profile News Impact")
        for message in profile_messages[:2]:
            st.write(f"- {message}")
        st.caption(
            "Details: Evaluation > News Impact Validation > Profile News Impact. "
            "For daily NewsLimit decisions, see Operations > Daily News Profile Recommendation."
        )

    operations = brief.get("operations", {})
    st.subheader("Operations Snapshot")
    op_cols = st.columns(4)
    op_cols[0].metric("Health", operations.get("overall_status") or "-")
    op_cols[1].metric("daily_predict", operations.get("daily_status") or "-")
    op_cols[2].metric("Latest binary", _format_date(operations.get("latest_binary_date")))
    op_cols[3].metric("Latest multiclass", _format_date(operations.get("latest_multiclass_date")))
    known_warn = operations.get("known_warn")
    if known_warn:
        st.caption(f"Known WARN: {known_warn}")


def _build_daily_brief_context(ticker: str, data: pd.DataFrame, settings) -> dict[str, object]:
    predictions = read_latest_predictions(settings.paths.database_path)
    if predictions.empty:
        predictions = pd.DataFrame()
    else:
        predictions = predictions[predictions["ticker"] == ticker].copy()

    binary_predictions = _filter_prediction_rows(
        predictions,
        target="next_close_binary",
        model_name_contains="binary",
    )
    latest_binary = binary_predictions.iloc[0] if not binary_predictions.empty else None
    binary_metrics = _load_binary_metrics(ticker)
    top_features = []
    if latest_binary is not None:
        top_features = _parse_top_features(latest_binary.get("top_features"))
    if not top_features and binary_metrics:
        top_features = binary_metrics.get("top_features", []) or []

    daily_multiclass_profile = "original_5_1"
    daily_multiclass_weight = "none"
    multiclass_model_name = _multiclass_model_name(daily_multiclass_profile, daily_multiclass_weight)
    multiclass_predictions = _filter_prediction_rows(
        predictions,
        target="next_close_multiclass",
        model_name=multiclass_model_name,
    )
    latest_multiclass = multiclass_predictions.iloc[0] if not multiclass_predictions.empty else None

    risk_multiclass_profile, risk_multiclass_weight = DEFAULT_RISK_MULTICLASS_OPTION
    risk_model_name = _multiclass_model_name(risk_multiclass_profile, risk_multiclass_weight)
    risk_multiclass_predictions = _filter_prediction_rows(
        predictions,
        target="next_close_multiclass",
        model_name=risk_model_name,
    )
    latest_risk_multiclass = (
        risk_multiclass_predictions.iloc[0] if not risk_multiclass_predictions.empty else None
    )

    external_context = _external_market_context(data)
    earnings_context = _earnings_event_context(data)
    news_context = _news_context(data, ticker, settings)
    narrative_context = _narrative_context(data)
    regime_state = _current_regime_state(data)
    similar_patterns, similar_summary = _daily_brief_similarity_context(data)
    cross_patterns, cross_summary = _daily_brief_cross_ticker_context(ticker, data, settings)
    _, lead_lag_ranking = _lead_lag_context(ticker, data, settings)
    _, anomaly_context, _, anomaly_summary = _anomaly_context(data)

    risk_scores = _risk_score_module().build_surge_crash_scores(
        data=data,
        shap_items=top_features,
        similar_summary=similar_summary,
        cross_ticker_summary=cross_summary,
        lead_lag_summary=lead_lag_ranking,
        anomaly_context=anomaly_context,
    )
    risk_scores = _augment_risk_scores_with_multiclass(
        risk_scores,
        latest_risk_multiclass,
        profile=risk_multiclass_profile,
        weight_mode=risk_multiclass_weight,
    )
    risk_percentiles = _current_risk_percentile_context(
        ticker,
        data,
        settings,
        risk_scores,
        risk_multiclass_profile,
        risk_multiclass_weight,
        DAILY_BRIEF_RISK_BUCKET_MODE,
    )

    prediction_label = "unknown"
    up_probability = None
    down_probability = None
    if latest_binary is not None:
        prediction_label = _prediction_class_name(
            latest_binary.get("predicted_class"),
            latest_binary.get("target"),
        )
        up_probability = latest_binary.get("probability_up")
        down_probability = latest_binary.get("probability_down")

    result = {
        "latest_binary": latest_binary,
        "latest_multiclass": latest_multiclass,
        "latest_risk_multiclass": latest_risk_multiclass,
        "latest_prediction_date": _daily_brief_latest_prediction_date(latest_binary, latest_multiclass),
        "binary_label": prediction_label,
        "up_probability": up_probability,
        "down_probability": down_probability,
        "top_features": top_features,
        "multiclass_profile": daily_multiclass_profile,
        "multiclass_weight": daily_multiclass_weight,
        "risk_multiclass_profile": risk_multiclass_profile,
        "risk_multiclass_weight": risk_multiclass_weight,
        "risk_scores": risk_scores,
        "risk_percentiles": risk_percentiles,
        "external_context": external_context,
        "earnings_context": earnings_context,
        "news_context": news_context,
        "news_impact_text": _daily_brief_news_impact_text(data),
        "source_type_news_impact_messages": _daily_brief_source_type_news_impact_messages(
            ticker,
            data,
            settings,
        ),
        "profile_news_impact_messages": _daily_brief_profile_news_impact_messages(
            ticker,
            data,
            settings,
        ),
        "narrative_context": narrative_context,
        "regime_state": regime_state,
        "similar_patterns": similar_patterns,
        "similar_summary": similar_summary,
        "cross_patterns": cross_patterns,
        "cross_summary": cross_summary,
        "lead_lag_ranking": lead_lag_ranking,
        "anomaly_context": anomaly_context,
        "anomaly_summary": anomaly_summary,
        "operations": _daily_brief_operations_snapshot(ticker),
    }
    result["prediction_feature_date"] = result["latest_prediction_date"]
    result["prediction_target_date"] = _next_trading_date_after(
        data,
        result.get("prediction_feature_date"),
    )
    result["prediction_saved_at"] = _latest_prediction_saved_at_from_rows(
        [latest_binary, latest_multiclass],
    )
    result["signal"] = _build_today_signal(
        prediction_label=prediction_label,
        up_probability=up_probability,
        down_probability=down_probability,
        risk_scores=risk_scores,
        risk_percentiles=risk_percentiles,
        regime_state=regime_state,
        anomaly_context=anomaly_context,
        operations_status=str(result.get("operations", {}).get("overall_status") or ""),
    )
    return result


def _build_today_signal(
    *,
    prediction_label: str,
    up_probability,
    down_probability,
    risk_scores: dict[str, object],
    risk_percentiles: dict[str, dict[str, object]],
    regime_state,
    anomaly_context,
    operations_status: str = "",
) -> dict[str, str]:
    signal_validation = _today_signal_validation_module()
    return signal_validation.build_today_signal(
        prediction_label=prediction_label,
        up_probability=up_probability,
        down_probability=down_probability,
        surge_score=getattr(risk_scores.get("surge"), "score", None),
        crash_score=getattr(risk_scores.get("crash"), "score", None),
        surge_percentile=risk_percentiles.get("surge", {}).get("percentile"),
        crash_percentile=risk_percentiles.get("crash", {}).get("percentile"),
        surge_bucket=str(risk_percentiles.get("surge", {}).get("relative_bucket") or "Unknown"),
        crash_bucket=str(risk_percentiles.get("crash", {}).get("relative_bucket") or "Unknown"),
        regime_name=getattr(regime_state, "name", None) or "Unknown regime",
        anomaly_score=getattr(anomaly_context, "score", None),
        operations_status=operations_status,
    )


def _today_signal_operations_note(operations_status: str) -> str:
    if str(operations_status).upper() == "WARN":
        return " Operations health is WARN, so confirm the Operations Snapshot before relying on it."
    return ""


def _daily_brief_today_signal_validation_text(ticker: str, signal_name: str) -> str:
    summary = st.session_state.get(f"today_signal_validation_summary_{ticker}")
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        module = _today_signal_validation_module()
        context = module.current_signal_context_text(summary, signal_name)
        suggestions = module.build_signal_improvement_suggestions(summary)
        improvement = module.current_signal_improvement_text(suggestions, signal_name)
        return " ".join(part for part in (context, improvement) if part)
    return "Run Evaluation > Today Signal Validation to see how this signal behaved historically."


def _daily_brief_latest_prediction_date(
    latest_binary: pd.Series | None,
    latest_multiclass: pd.Series | None,
):
    dates = []
    for row in (latest_binary, latest_multiclass):
        if row is None:
            continue
        value = row.get("date") if "date" in row else row.get("prediction_date")
        timestamp = pd.to_datetime(value, errors="coerce")
        if not pd.isna(timestamp):
            dates.append(timestamp)
    return max(dates) if dates else None


def _next_trading_date_after(data: pd.DataFrame | None, feature_date):
    if data is None or data.empty or "date" not in data.columns:
        return None
    timestamp = pd.to_datetime(feature_date, errors="coerce")
    if pd.isna(timestamp):
        return None
    dates = pd.to_datetime(data["date"], errors="coerce").dropna().dt.normalize().drop_duplicates()
    if dates.empty:
        return None
    future_dates = dates[dates > timestamp.normalize()].sort_values()
    if future_dates.empty:
        return None
    return future_dates.iloc[0]


def _latest_prediction_saved_at_from_rows(rows: list[pd.Series | None]):
    saved_at_values = []
    for row in rows:
        if row is None:
            continue
        timestamp = pd.to_datetime(row.get("created_at"), errors="coerce")
        if not pd.isna(timestamp):
            saved_at_values.append(timestamp)
    return max(saved_at_values) if saved_at_values else None


def _daily_brief_risk_metric(brief: dict[str, object], risk_type: str) -> str:
    risk_scores = brief.get("risk_scores", {}) or {}
    risk_percentiles = brief.get("risk_percentiles", {}) or {}
    score = risk_scores.get(risk_type)
    value = safe_float(getattr(score, "score", None))
    percentile = safe_float(risk_percentiles.get(risk_type, {}).get("percentile"))
    if value is None and percentile is None:
        return "-"
    return f"{_format_number(value, digits=0)} / {_risk_percentile_position_text(percentile)}"


def _render_daily_brief_overview(
    brief: dict[str, object],
    signal_name: str,
    confidence: str,
) -> None:
    st.subheader("Today Overview")
    st.caption("Check direction, risk, and data freshness first. Use the sections below for details.")

    operations = brief.get("operations", {}) or {}
    overview_cols = st.columns(4)
    overview_cols[0].metric("今日の総合シグナル", f"{signal_name} / {confidence}")
    overview_cols[1].metric(
        "Binary direction",
        f"{brief.get('binary_label') or '-'} / {_format_percent(_daily_brief_binary_confidence(brief))}",
    )
    overview_cols[2].metric(
        "Multiclass",
        (
            f"{_daily_brief_multiclass_label(brief)} / "
            f"{_format_percent(_daily_brief_multiclass_confidence(brief))}"
        ),
    )
    overview_cols[3].metric("Risk", _daily_brief_risk_summary(brief))

    status_cols = st.columns(4)
    status_cols[0].metric("Latest feature date", _format_date(operations.get("latest_feature_date")))
    status_cols[1].metric("Target close date", _format_date(brief.get("prediction_target_date")))
    status_cols[2].metric("News Health", _daily_brief_news_health_label(operations))
    status_cols[3].metric("daily_predict", _daily_brief_daily_status_label(operations))


def _render_daily_brief_reading_guide() -> None:
    with st.expander("How to read this brief", expanded=False):
        st.write(
            "- Direction: Binary is the next close direction view. Multiclass adds surge/flat/crash context."
        )
        st.write("- Confidence: A probability-like strength, not a guarantee. Read with Risk and Context.")
        st.write("- Risk: A caution signal for surge, crash, or unusual conditions. Use score and percentile together.")
        st.write("- News: Entry point for recent news, feature quality, and impact validation.")
        st.write("- Operations: Check whether data refresh and daily prediction are running normally.")


def _render_daily_brief_attention(brief: dict[str, object]) -> None:
    st.subheader("Attention / 注意点")
    items = _daily_brief_attention_items(brief)
    if not items:
        st.success("No immediate action is required. Operation can continue when only known warnings remain.")
        return
    for level, message in items:
        if level == "warning":
            st.warning(message)
        elif level == "info":
            st.info(message)
        else:
            st.caption(message)


def _render_daily_brief_next_check(brief: dict[str, object]) -> None:
    operations = brief.get("operations", {}) or {}
    st.subheader("Next Check")
    next_run = _format_datetime(operations.get("next_daily_predict_run"))
    st.write(f"- 次囁Edaily_predict 予宁E {next_run}")
    st.write("- 次に見る場所: Operations > Data Collection Monitor / NewsLimit 3 Evaluation")
    st.write(
        "- NewsLimit 3 and the 4-profile daily setup are still under observation. Re-evaluate after more fetch history is collected."
    )


def _daily_brief_binary_confidence(brief: dict[str, object]) -> float | None:
    latest_binary = brief.get("latest_binary")
    if latest_binary is not None:
        confidence = safe_float(latest_binary.get("confidence"))
        if confidence is not None:
            return confidence
    return max(
        safe_float(brief.get("up_probability")) or 0.0,
        safe_float(brief.get("down_probability")) or 0.0,
    ) or None


def _daily_brief_multiclass_label(brief: dict[str, object]) -> str:
    latest_multiclass = brief.get("latest_multiclass")
    if latest_multiclass is None:
        return "-"
    return _prediction_class_name(
        latest_multiclass.get("predicted_class"),
        latest_multiclass.get("target"),
    )


def _daily_brief_multiclass_confidence(brief: dict[str, object]) -> float | None:
    latest_multiclass = brief.get("latest_multiclass")
    if latest_multiclass is None:
        return None
    confidence = safe_float(latest_multiclass.get("confidence"))
    if confidence is not None:
        return confidence
    probabilities = [
        value
        for value in _multiclass_probability_map(latest_multiclass).values()
        if value is not None
    ]
    return max(probabilities) if probabilities else None


def _daily_brief_risk_summary(brief: dict[str, object]) -> str:
    surge = _daily_brief_risk_metric(brief, "surge")
    crash = _daily_brief_risk_metric(brief, "crash")
    return f"Surge {surge} / Crash {crash}"


def _daily_brief_news_health_label(operations: dict[str, object]) -> str:
    status = str(operations.get("news_status") or "-")
    rows = _format_number(operations.get("news_rows"), digits=0)
    unknown = safe_int(operations.get("unknown_source_count")) or 0
    if unknown:
        return f"{status}; unknown {unknown}"
    return f"{status}; rows {rows}"


def _daily_brief_daily_status_label(operations: dict[str, object]) -> str:
    status = str(operations.get("daily_status") or "-")
    latest_success = _format_datetime(operations.get("latest_daily_success"))
    if latest_success == "-":
        return status
    return f"{status}; {latest_success}"


def _daily_brief_attention_items(brief: dict[str, object]) -> list[tuple[str, str]]:
    operations = brief.get("operations", {}) or {}
    items: list[tuple[str, str]] = []

    action_required_count = safe_int(operations.get("action_required_count")) or 0
    known_missed_count = safe_int(operations.get("known_missed_count")) or 0
    known_ok_count = safe_int(operations.get("known_ok_count")) or 0
    action_required_items = operations.get("action_required_items", []) or []
    known_missed_items = operations.get("known_missed_items", []) or []
    known_ok_items = operations.get("known_ok_items", []) or []
    if action_required_count:
        items.append(
            (
                "warning",
                "Action required: "
                f"{action_required_count}. Check Operations > Health Warnings Summary.",
            )
        )
        names = _health_warning_names(action_required_items)
        if names:
            items.append(("caption", f"Action required items: {names}"))
    elif known_missed_count or known_ok_count:
        known_missed_names = _health_warning_names(known_missed_items)
        names = _health_warning_names(known_ok_items)
        parts = []
        if known_missed_count:
            parts.append(f"Known missed: {known_missed_names or known_missed_count}")
        if known_ok_count:
            parts.append(f"Known OK: {names or known_ok_count}")
        items.append(
            (
                "info",
                f"Action required: 0. {'; '.join(parts)}. Status: operation can continue.",
            )
        )

    unknown_sources = safe_int(operations.get("unknown_source_count")) or 0
    if unknown_sources:
        items.append(
            (
                "warning",
                f"unknown_source_count is {unknown_sources}. Check Operations > Unknown Sources before relying on source weighting.",
            )
        )

    overall_status = str(operations.get("overall_status") or "").upper()
    daily_status = str(operations.get("daily_status") or "").upper()
    if overall_status == "WARN" and daily_status == "OK":
        items.append(
            (
                "info",
                "health-check is WARN, but latest daily_predict succeeded. Operation can continue if only known warnings remain.",
            )
        )
    elif overall_status == "FAIL" or daily_status == "FAIL":
        items.append(
            (
                "warning",
                "health-check or daily_predict is FAIL. Check the latest log in Operations.",
            )
        )

    for warning in operations.get("warn_items", []) or []:
        name = str(warning.get("name") or "")
        message = str(warning.get("message") or "")
        if "TOPIX" in name:
            items.append(("caption", f"TOPIX fallback: {message}"))
        elif "Missing live dates" in name:
            items.append(
                (
                    "caption",
                    "Live feature-date coverage is checked only after the configured operation start date. "
                    "Earlier feature dates are outside the live-operation window, not missing predictions.",
                )
            )
        elif name:
            items.append(("caption", f"{name}: {message}"))

    return items


def _health_warning_names(items: list[dict[str, object]]) -> str:
    names = [
        str(item.get("name") or "-")
        for item in items
        if isinstance(item, dict)
    ]
    return ", ".join(names[:4])


def _daily_brief_binary_frame(brief: dict[str, object]) -> pd.DataFrame:
    latest_binary = brief.get("latest_binary")
    if latest_binary is None:
        return pd.DataFrame([{"Item": "Status", "Value": "Binary prediction is not available."}])
    return pd.DataFrame(
        [
            {"Item": "Feature date", "Value": _format_date(latest_binary.get("date"))},
            {"Item": "Target close date", "Value": _format_date(brief.get("prediction_target_date"))},
            {"Item": "Saved at", "Value": _format_datetime(latest_binary.get("created_at"))},
            {"Item": "Predicted direction", "Value": brief.get("binary_label") or "-"},
            {"Item": "Up probability", "Value": _format_percent(brief.get("up_probability"))},
            {"Item": "Down probability", "Value": _format_percent(brief.get("down_probability"))},
            {"Item": "Confidence", "Value": _format_percent(latest_binary.get("confidence"))},
        ]
    )


def _daily_brief_multiclass_frame(brief: dict[str, object]) -> pd.DataFrame:
    latest_multiclass = brief.get("latest_multiclass")
    if latest_multiclass is None:
        return pd.DataFrame()
    probability_map = _multiclass_probability_map(latest_multiclass)
    rows = [
        {"Item": "Feature date", "Value": _format_date(latest_multiclass.get("date"))},
        {"Item": "Target close date", "Value": _format_date(brief.get("prediction_target_date"))},
        {"Item": "Saved at", "Value": _format_datetime(latest_multiclass.get("created_at"))},
        {"Item": "Profile", "Value": brief.get("multiclass_profile") or "-"},
        {"Item": "Weight mode", "Value": brief.get("multiclass_weight") or "-"},
        {
            "Item": "Predicted class",
            "Value": _prediction_class_name(
                latest_multiclass.get("predicted_class"),
                latest_multiclass.get("target"),
            ),
        },
        {"Item": "Confidence", "Value": _format_percent(latest_multiclass.get("confidence"))},
    ]
    for label in ("surge", "up", "flat", "down", "crash"):
        rows.append({"Item": f"{label} probability", "Value": _format_percent(probability_map.get(label))})
    return pd.DataFrame(rows)


def _daily_brief_risk_display_frame(risk_percentiles) -> pd.DataFrame:
    return pd.DataFrame(_current_risk_percentile_display_rows(risk_percentiles))


def _daily_brief_context_rows(brief: dict[str, object]) -> list[dict[str, str]]:
    external_context = brief.get("external_context")
    earnings_context = brief.get("earnings_context")
    news_context = brief.get("news_context")
    narrative_context = brief.get("narrative_context")
    regime_state = brief.get("regime_state")
    anomaly_context = brief.get("anomaly_context")
    similar_summary = brief.get("similar_summary")
    cross_summary = brief.get("cross_summary")
    lead_lag_ranking = brief.get("lead_lag_ranking")

    return [
        {
            "Area": "Regime",
            "Today": _brief_regime_text(regime_state),
            "How to read": "Current market state for interpreting all predictions.",
            "Detail tab": "Prediction / Evaluation",
        },
        {
            "Area": "Anomaly",
            "Today": _brief_anomaly_text(anomaly_context),
            "How to read": "Unusual price, volume, volatility, sector, market, news, or narrative behavior.",
            "Detail tab": "Anomaly",
        },
        {
            "Area": "Similar Patterns",
            "Today": _brief_similarity_text(similar_summary),
            "How to read": "How similar Toyota historical windows behaved afterward.",
            "Detail tab": "Similar Patterns",
        },
        {
            "Area": "Cross-Ticker Similarity",
            "Today": _brief_cross_ticker_text(cross_summary),
            "How to read": "Whether Toyota resembles past peer-stock windows.",
            "Detail tab": "Cross-Ticker Similarity",
        },
        {
            "Area": "Lead-Lag",
            "Today": _brief_lead_lag_text(lead_lag_ranking),
            "How to read": "Whether peers, Nikkei, TOPIX, or USDJPY may be leading Toyota.",
            "Detail tab": "Lead-Lag",
        },
        {
            "Area": "News / Narrative",
            "Today": _brief_news_narrative_text(news_context, narrative_context),
            "How to read": "Manual news and detected themes as context, not standalone predictions.",
            "Detail tab": "Prediction",
        },
        {
            "Area": "Earnings",
            "Today": _brief_earnings_text(earnings_context),
            "How to read": "Separates normal days from earnings/event-sensitive days.",
            "Detail tab": "Prediction",
        },
        {
            "Area": "External Market",
            "Today": _brief_external_market_text(external_context),
            "How to read": "USDJPY, Nikkei, TOPIX and broad market context for Toyota.",
            "Detail tab": "Prediction / Data",
        },
        {
            "Area": "Auto Sector",
            "Today": _brief_auto_sector_text(external_context),
            "How to read": "Honda, Nissan, Tesla, BYD and sector aggregate context.",
            "Detail tab": "Prediction / Cross-Ticker Similarity",
        },
    ]


def _daily_brief_filter_context_rows(
    rows: list[dict[str, str]],
    section: str,
) -> list[dict[str, str]]:
    market_areas = {
        "Regime",
        "Similar Patterns",
        "Cross-Ticker Similarity",
        "Lead-Lag",
        "External Market",
        "Auto Sector",
    }
    event_areas = {"Anomaly", "News / Narrative", "Earnings"}
    allowed = market_areas if section == "market" else event_areas
    return [row for row in rows if row.get("Area") in allowed]


def _daily_brief_important_news_frame(news_context, data: pd.DataFrame) -> pd.DataFrame:
    if news_context is None:
        return pd.DataFrame()
    items = getattr(news_context, "important_items", []) or []
    if not items:
        return pd.DataFrame()
    frame = pd.DataFrame(items).head(3).copy()
    try:
        frame = _news_impact_module().add_news_following_price_moves(frame, data)
    except Exception as error:
        LOGGER.debug("Could not add Daily Brief news price moves: %s", error, exc_info=True)
    display_columns = [
        column
        for column in (
            "date",
            "base_trading_date",
            "title",
            "source",
            "url",
            "source_type",
            "category",
            "sentiment_score",
            "importance_score",
            "return_5d",
            "max_down_20d",
            "future_data_note",
        )
        if column in frame.columns
    ]
    display = frame[display_columns].copy()
    display["quality_note"] = frame.apply(_news_quality_note, axis=1)
    for column in ("sentiment_score", "importance_score"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    for column in ("return_5d", "max_down_20d"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("date", "base_trading_date"):
        if column in display.columns:
            display[column] = pd.to_datetime(display[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return display.rename(
        columns={
            "date": "Date",
            "base_trading_date": "Base Trading Date",
            "title": "Title",
            "source": "Source",
            "url": "URL",
            "source_type": "Source Type",
            "category": "Category",
            "sentiment_score": "Sentiment",
            "importance_score": "Importance",
            "return_5d": "Return 5D",
            "max_down_20d": "Max Down 20D",
            "future_data_note": "Future Data Note",
            "quality_note": "Quality Note",
        }
    )


def _news_quality_note(row: pd.Series) -> str:
    notes: list[str] = []
    title = str(row.get("title") or "").strip()
    summary = str(row.get("summary") or "").strip()
    if not title:
        notes.append("missing title")
    try:
        if _news_quality_module().is_mojibake_suspect(f"{title} {summary}"):
            notes.append("mojibake suspect")
    except Exception as error:
        LOGGER.debug("Could not check news quality note: %s", error, exc_info=True)
    return ", ".join(notes)


def _daily_brief_similarity_context(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        similarity = _similarity_module()
        brief_data = data.tail(252).copy()
        patterns = similarity.find_similar_patterns(
            brief_data,
            window_size=20,
            top_n=5,
            forward_days=(1, 5, 20),
        )
        summary = similarity.summarize_forward_returns(patterns, forward_days=(1, 5, 20))
        return patterns, summary
    except Exception as error:
        _log_optional_context_error("daily brief similar patterns", error)
        return pd.DataFrame(), pd.DataFrame()


def _daily_brief_cross_ticker_context(
    ticker: str,
    data: pd.DataFrame,
    settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        cross = _cross_ticker_module()
        frames = cross.load_cross_ticker_frames(
            raw_data_dir=settings.paths.raw_data_dir,
            feature_data_dir=settings.paths.feature_data_dir,
        )
        frames = {symbol: frame.tail(252).copy() for symbol, frame in frames.items()}
        patterns = cross.find_cross_ticker_similar_patterns(
            target_data=data.tail(252).copy(),
            candidate_frames=frames,
            target_ticker=ticker,
            window_size=20,
            top_n=5,
            forward_days=(1, 5, 20),
        )
        summary = cross.summarize_cross_ticker_patterns(patterns, forward_days=(1, 5, 20))
        return patterns, summary
    except Exception as error:
        _log_optional_context_error("daily brief cross-ticker similarity", error)
        return pd.DataFrame(), pd.DataFrame()


def _brief_regime_text(regime_state) -> str:
    if regime_state is None:
        return "Unavailable"
    name = getattr(regime_state, "name", "Unknown")
    level = getattr(regime_state, "level", "")
    return f"{name} {f'({level})' if level else ''}".strip()


def _brief_anomaly_text(anomaly_context) -> str:
    if anomaly_context is None:
        return "No anomaly context available."
    score = safe_float(getattr(anomaly_context, "score", None))
    level = getattr(anomaly_context, "level", "Unknown")
    types = getattr(anomaly_context, "types", []) or []
    if isinstance(types, str):
        type_text = types
    else:
        type_text = ", ".join(str(item) for item in types[:3]) if types else "No major type"
    return f"{_format_number(score, digits=0)}/100 {level}; {type_text}."


def _brief_similarity_text(summary) -> str:
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return "No similar-pattern summary available."
    row = _brief_summary_row(summary, "5D") or _brief_summary_row(summary, "1D")
    if row is None:
        return "No forward-return summary available."
    return (
        f"{row.get('horizon')} average return {_format_percent(row.get('average_return'))}; "
        f"up rate {_format_percent(row.get('positive_rate'))}."
    )


def _brief_cross_ticker_text(summary) -> str:
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return "No cross-ticker summary available."
    row = _brief_summary_row(summary, "5D") or _brief_summary_row(summary, "1D")
    if row is None:
        return "No peer forward-return summary available."
    return (
        f"{row.get('horizon')} Toyota avg {_format_percent(row.get('toyota_average_return'))}; "
        f"peer avg {_format_percent(row.get('peer_average_return'))}."
    )


def _brief_summary_row(summary: pd.DataFrame, horizon: str) -> dict[str, object] | None:
    if summary.empty or "horizon" not in summary.columns:
        return None
    rows = summary[summary["horizon"].astype(str) == horizon]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _brief_lead_lag_text(ranking) -> str:
    if not isinstance(ranking, pd.DataFrame) or ranking.empty:
        return "No lead-lag ranking available."
    row = ranking.iloc[0]
    label = row.get("ticker") or row.get("symbol") or row.get("name") or row.get("candidate") or "Top signal"
    lag = row.get("lag")
    signal = row.get("current_signal") or row.get("direction") or "-"
    score = row.get("score")
    lag_text = f" lag {lag}" if lag is not None else ""
    return f"{label}{lag_text}: {signal}, score {_format_number(score, digits=1)}."


def _brief_news_narrative_text(news_context, narrative_context) -> str:
    parts: list[str] = []
    if news_context is not None:
        metrics = getattr(news_context, "metrics", {}) or {}
        if metrics.get("negative_news_shock"):
            shock = "negative shock"
        elif metrics.get("positive_news_shock"):
            shock = "positive shock"
        else:
            shock = "no shock"
        parts.append(
            f"News 3D {_format_number(metrics.get('news_count_3d'), digits=0)}, "
            f"sentiment {_format_number(metrics.get('avg_sentiment_3d'), digits=2)}, {shock}"
        )
    if narrative_context is not None:
        dominant = getattr(narrative_context, "dominant_label", "None")
        metrics = getattr(narrative_context, "metrics", {}) or {}
        parts.append(
            f"Narrative {dominant}, trend {_format_number(metrics.get('dominant_narrative_trend'), digits=2)}"
        )
    return "; ".join(parts) if parts else "No news or narrative context available."


def _brief_earnings_text(earnings_context) -> str:
    if earnings_context is None:
        return "No earnings context available."
    next_date = getattr(earnings_context, "next_earnings_date", None) or "-"
    days_to = getattr(earnings_context, "days_to_earnings", None)
    state = getattr(earnings_context, "event_state", "normal")
    post_mode = "post-earnings mode" if getattr(earnings_context, "post_earnings_mode", False) else "normal mode"
    return f"{state}; next {next_date}; days to earnings {days_to if days_to is not None else '-'}; {post_mode}."


def _brief_external_market_text(external_context) -> str:
    if external_context is None:
        return "No external market context available."
    metrics = getattr(external_context, "metrics", {}) or {}
    return (
        f"USDJPY {getattr(external_context, 'usdjpy_status', '-')}, "
        f"market {getattr(external_context, 'market_trend', '-')}; "
        f"USDJPY 1D {_format_percent(metrics.get('usdjpy_return_1d'))}, "
        f"Nikkei 1D {_format_percent(metrics.get('n225_return_1d'))}."
    )


def _brief_auto_sector_text(external_context) -> str:
    if external_context is None:
        return "No auto-sector context available."
    metrics = getattr(external_context, "metrics", {}) or {}
    return (
        f"Sector {getattr(external_context, 'auto_sector_trend', '-')}; "
        f"Tesla {getattr(external_context, 'tesla_trend', '-')}, "
        f"Honda {getattr(external_context, 'honda_trend', '-')}; "
        f"sector return {_format_percent(metrics.get('sector_mean_return'))}."
    )


def _daily_brief_operations_snapshot(ticker: str) -> dict[str, object]:
    try:
        report = _load_health_report_for_ui(ticker)
    except Exception as error:
        LOGGER.debug("Could not load Daily Brief operations snapshot: %s", error, exc_info=True)
        return {"overall_status": "Unavailable", "daily_status": "Unavailable", "known_warn": str(error)}

    daily_item = _health_item(report, "daily_predict", "Latest run")
    binary_item = _health_item(report, "SQLite", "Latest binary prediction")
    multiclass_item = _health_item(report, "SQLite", "Latest multiclass prediction")
    feature_item = _health_item(report, "Data", "Feature summary")
    news_item = _health_item(report, "News", "News CSV quality")
    scheduler_item = _health_item(report, "daily_predict", "Windows scheduled task")
    warn_items = [
        item
        for item in report.get("items", []) or []
        if str(item.get("status")) == "WARN"
    ]
    known_warn = "; ".join(f"{item.get('name')}: {item.get('message')}" for item in warn_items[:2])
    warn_details = [
        {"name": item.get("name"), "message": item.get("message")}
        for item in warn_items[:5]
    ]
    warning_summary = report.get("warning_summary") or {}
    action_required = warning_summary.get("action_required", []) or []
    known_missed = warning_summary.get("known_missed", []) or []
    known_ok = warning_summary.get("known_ok", []) or []
    return {
        "overall_status": report.get("overall_status"),
        "daily_status": _health_status(daily_item),
        "latest_daily_success": _health_detail(daily_item, "latest_success"),
        "latest_binary_date": _health_detail(binary_item, "date"),
        "latest_multiclass_date": _health_detail(multiclass_item, "date"),
        "latest_feature_date": _health_detail(feature_item, "latest_feature_date"),
        "news_status": _health_status(news_item),
        "news_rows": _health_detail(news_item, "rows"),
        "unknown_source_count": _health_detail(news_item, "unknown_source_count"),
        "latest_news_date": _health_detail(news_item, "latest_news_date"),
        "latest_news_fetched_at": _health_detail(news_item, "latest_fetched_at"),
        "next_daily_predict_run": _health_detail(scheduler_item, "next_run_time"),
        "known_warn": known_warn,
        "warn_items": warn_details,
        "action_required_count": warning_summary.get("action_required_count", len(action_required)),
        "known_missed_count": warning_summary.get("known_missed_count", len(known_missed)),
        "known_ok_count": warning_summary.get("known_ok_count", len(known_ok)),
        "action_required_items": action_required[:5],
        "known_missed_items": known_missed[:5],
        "known_ok_items": known_ok[:5],
    }


def _daily_brief_news_impact_text(data: pd.DataFrame) -> str:
    try:
        module = _news_impact_module()
        frame = module.build_news_impact_frame(data)
        if frame.empty:
            return ""
        summaries = module.build_news_impact_summaries(frame)
        return module.current_news_impact_context(frame, summaries)
    except Exception as error:
        LOGGER.debug("Could not build Daily Brief news impact text: %s", error, exc_info=True)
        return ""


def _daily_brief_source_type_news_impact_messages(
    ticker: str,
    data: pd.DataFrame,
    settings,
) -> list[str]:
    try:
        module = _news_impact_module()
        news_dir = _news_module().default_news_dir(settings.paths.raw_data_dir.parent)
        source_type_frame = module.summarize_source_type_impact(data, news_dir, ticker)
        return module.build_source_type_impact_messages(source_type_frame, max_items=3)
    except Exception as error:
        LOGGER.debug(
            "Could not build Daily Brief source-type news impact messages: %s",
            error,
            exc_info=True,
        )
        return []


def _daily_brief_profile_news_impact_messages(
    ticker: str,
    data: pd.DataFrame,
    settings,
) -> list[str]:
    try:
        module = _news_impact_module()
        news_dir = _news_module().default_news_dir(settings.paths.raw_data_dir.parent)
        profile_frame = module.build_profile_news_impact(data, news_dir, ticker)
        recommendations = module.build_daily_news_profile_recommendations(profile_frame)
        messages = module.build_daily_news_profile_recommendation_messages(
            recommendations,
            max_items=2,
        )
        if messages:
            return messages
        return module.build_profile_news_impact_messages(profile_frame, max_items=2)
    except Exception as error:
        LOGGER.debug(
            "Could not build Daily Brief profile news impact messages: %s",
            error,
            exc_info=True,
        )
        return []


def _render_prediction_tab(
    ticker: str,
    data: pd.DataFrame,
    latest: pd.Series,
    database_path,
    settings,
) -> None:
    st.caption(
        "Suggested reading order: 1. Current Prediction, 2. Multiclass Prediction, "
        "3. Risk Scores, 4. Integrated Reasoning, 5. Context Details."
    )

    actual_label = _class_label_from_value(latest.get("target_next_close_class"))
    basic_reasons = list(build_basic_reasons(latest))

    predictions = read_latest_predictions(database_path)
    if predictions.empty:
        predictions = pd.DataFrame()
    else:
        predictions = predictions[predictions["ticker"] == ticker].copy()

    binary_predictions = _filter_prediction_rows(
        predictions,
        target="next_close_binary",
        model_name_contains="binary",
    )
    selected_multiclass_profile = _select_multiclass_threshold_profile(
        key="prediction_multiclass_threshold_profile_select",
    )
    selected_multiclass_weight_mode = _select_multiclass_weight_mode(
        key="prediction_multiclass_weight_mode_select",
    )
    selected_multiclass_model_name = _multiclass_model_name(
        selected_multiclass_profile,
        selected_multiclass_weight_mode,
    )
    multiclass_predictions = _filter_prediction_rows(
        predictions,
        target="next_close_multiclass",
        model_name=selected_multiclass_model_name,
    )
    latest_multiclass = (
        multiclass_predictions.iloc[0] if not multiclass_predictions.empty else None
    )
    risk_multiclass_profile, risk_multiclass_weight_mode = _select_risk_multiclass_setting(
        key="prediction_risk_multiclass_setting_select",
    )
    risk_bucket_mode = _select_risk_bucket_mode(
        key="prediction_risk_bucket_mode_select",
    )
    risk_multiclass_model_name = _multiclass_model_name(
        risk_multiclass_profile,
        risk_multiclass_weight_mode,
    )
    risk_multiclass_predictions = _filter_prediction_rows(
        predictions,
        target="next_close_multiclass",
        model_name=risk_multiclass_model_name,
    )
    latest_risk_multiclass = (
        risk_multiclass_predictions.iloc[0]
        if not risk_multiclass_predictions.empty
        else None
    )
    binary_metrics = _load_binary_metrics(ticker)
    latest_binary = binary_predictions.iloc[0] if not binary_predictions.empty else None
    up_probability = None
    down_probability = None
    prediction_label = "unknown"
    top_features: list[dict[str, object]] = []

    _render_prediction_overview(
        ticker=ticker,
        data=data,
        latest_binary=latest_binary,
        latest_multiclass=latest_multiclass,
        selected_multiclass_profile=selected_multiclass_profile,
        selected_multiclass_weight_mode=selected_multiclass_weight_mode,
        actual_label=actual_label,
        basic_reasons=basic_reasons,
    )

    st.subheader("Current Prediction")
    st.caption("Binary Prediction")
    if latest_binary is not None:
        up_probability = latest_binary.get("probability_up")
        down_probability = latest_binary.get("probability_down")
        prediction_label = _prediction_class_name(
            latest_binary.get("predicted_class"),
            latest_binary.get("target"),
        )

        probability_cols = st.columns([1, 1, 1])
        probability_cols[0].metric("Up probability", _format_percent(up_probability))
        probability_cols[1].metric("Down probability", _format_percent(down_probability))
        probability_cols[2].metric("Prediction", prediction_label)
        st.plotly_chart(
            _build_probability_chart(up_probability, down_probability),
            use_container_width=True,
        )

        if binary_metrics:
            metric_cols = st.columns(4)
            metrics = binary_metrics.get("metrics", {})
            metric_cols[0].metric("Accuracy", _format_percent(metrics.get("accuracy")))
            metric_cols[1].metric("F1", _format_percent(metrics.get("f1")))
            metric_cols[2].metric("AUC", _format_percent(metrics.get("auc")))
            metric_cols[3].metric("Test rows", _format_number(metrics.get("rows"), digits=0))

        top_features = _parse_top_features(latest_binary.get("top_features"))
        if not top_features:
            top_features = binary_metrics.get("top_features", []) if binary_metrics else []
        if top_features:
            with st.expander("SHAP / explanation details", expanded=False):
                if _has_shap_contributions(top_features):
                    _render_shap_explanation(top_features, prediction_label)
                else:
                    st.caption("Feature importance top 10")
                    st.plotly_chart(
                        _build_feature_contribution_chart(top_features),
                        use_container_width=True,
                    )
                    st.dataframe(pd.DataFrame(top_features), use_container_width=True)

        global_importance = binary_metrics.get("top_features", []) if binary_metrics else []
        if global_importance and global_importance != top_features:
            with st.expander("Global feature importance"):
                st.caption("Training-level importance. This is different from the current sample's SHAP reason.")
                st.plotly_chart(
                    _build_feature_contribution_chart(global_importance),
                    use_container_width=True,
                )
                st.dataframe(pd.DataFrame(global_importance), use_container_width=True)

        st.caption("Prediction rationale")
        st.write(
            _build_prediction_rationale(
                latest=latest,
                prediction_label=prediction_label,
                up_probability=up_probability,
                down_probability=down_probability,
                top_features=top_features,
                metrics=binary_metrics.get("metrics", {}) if binary_metrics else {},
            )
        )

    else:
        st.info("Binary prediction is not available. Run predict-binary first.")
        st.code(
            f"python main.py train-binary --ticker {ticker}\n"
            f"python main.py predict-binary --ticker {ticker}",
            language="powershell",
        )

    if not multiclass_predictions.empty:
        _render_multiclass_prediction_section(
            latest_multiclass,
            metrics=_load_multiclass_metrics(
                ticker,
                selected_multiclass_profile,
                selected_multiclass_weight_mode,
            ),
            latest_binary=latest_binary,
            threshold_profile=selected_multiclass_profile,
            weight_mode=selected_multiclass_weight_mode,
        )
    else:
        st.subheader("Multiclass Prediction")
        st.caption(
            f"Threshold profile: {selected_multiclass_profile} / "
            f"weight mode: {selected_multiclass_weight_mode}"
        )
        st.info("Multiclass prediction is not available. Run predict-multiclass first.")
        st.code(
            f"python main.py train-multiclass --ticker {ticker} --threshold-profile {selected_multiclass_profile} --weight-mode {selected_multiclass_weight_mode}\n"
            f"python main.py predict-multiclass --ticker {ticker} --threshold-profile {selected_multiclass_profile} --weight-mode {selected_multiclass_weight_mode}",
            language="powershell",
        )

    if latest_binary is not None:
        _render_integrated_reasoning(
            data=data,
            prediction_label=prediction_label,
            up_probability=up_probability,
            down_probability=down_probability,
            top_features=top_features,
            ticker=ticker,
            settings=settings,
            multiclass_prediction=latest_multiclass,
            risk_multiclass_prediction=latest_risk_multiclass,
            risk_multiclass_profile=risk_multiclass_profile,
            risk_multiclass_weight_mode=risk_multiclass_weight_mode,
            risk_bucket_mode=risk_bucket_mode,
        )

    if not predictions.empty:
        predictions["predicted_label"] = predictions.apply(_prediction_label, axis=1)
        with st.expander("Detailed prediction records", expanded=False):
            st.dataframe(predictions, use_container_width=True)


def _render_prediction_overview(
    *,
    ticker: str,
    data: pd.DataFrame,
    latest_binary: pd.Series | None,
    latest_multiclass: pd.Series | None,
    selected_multiclass_profile: str,
    selected_multiclass_weight_mode: str,
    actual_label: int | None,
    basic_reasons: list[str],
) -> None:
    st.subheader("Prediction Overview")
    st.caption(
        "Daily Brief is the quick read. This tab is for checking the saved prediction rows, "
        "model details, probabilities, and explanations behind that overview."
    )

    latest_feature_date = _prediction_latest_date_value(latest_binary, latest_multiclass)
    target_close_date = _next_trading_date_after(data, latest_feature_date)
    overview_cols = st.columns(5)
    overview_cols[0].metric("Feature date", _format_date(latest_feature_date))
    overview_cols[1].metric("Target close date", _format_date(target_close_date))
    overview_cols[2].metric("Binary", _prediction_overview_binary_label(latest_binary))
    overview_cols[3].metric("Binary confidence", _prediction_row_percent(latest_binary, "confidence"))
    overview_cols[4].metric("Source", _prediction_source_label(latest_binary, latest_multiclass))

    multiclass_cols = st.columns(4)
    multiclass_cols[0].metric("Multiclass", _prediction_overview_multiclass_label(latest_multiclass))
    multiclass_cols[1].metric("Multiclass confidence", _prediction_row_percent(latest_multiclass, "confidence"))
    multiclass_cols[2].metric("Profile", selected_multiclass_profile)
    multiclass_cols[3].metric("Weight mode", selected_multiclass_weight_mode)

    freshness = _prediction_data_freshness(ticker, data)
    freshness_cols = st.columns(5)
    freshness_cols[0].metric("Latest feature date", _format_date(freshness.get("latest_feature_date")))
    freshness_cols[1].metric("Latest binary date", _format_date(freshness.get("latest_binary_date")))
    freshness_cols[2].metric("Latest multiclass date", _format_date(freshness.get("latest_multiclass_date")))
    freshness_cols[3].metric("daily_predict", freshness.get("daily_status") or "-")
    freshness_cols[4].metric("Health", freshness.get("overall_status") or "-")
    if _prediction_health_attention(freshness):
        st.info(_prediction_health_attention(freshness))

    st.caption("How to read")
    st.write(
        "- Feature date: the saved prediction row's input data date. "
        "Target close date: the next trading date after the feature date. "
        "Binary: next business day close direction, up or down. "
        " Multiclass: rough surge/up/flat/down/crash class. "
        " Confidence: probability-like strength, not a guarantee. "
        " Source: live means a daily operation saved prediction."
    )

    with st.expander("Current rule-based context", expanded=False):
        actual_text = CLASS_NAMES.get(actual_label, "pending") if actual_label is not None else "pending"
        st.metric("Actual next-close label", actual_text)
        if basic_reasons:
            st.caption("Rule-based reasons")
            for reason in basic_reasons:
                st.write(f"- {reason}")

    rows = [
        _prediction_overview_record("Binary", latest_binary, data),
        _prediction_overview_record("Multiclass", latest_multiclass, data),
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _prediction_overview_record(
    label: str,
    row: pd.Series | None,
    data: pd.DataFrame | None = None,
) -> dict[str, object]:
    if row is None:
        return {
            "Type": label,
            "Feature date": "-",
            "Target close date": "-",
            "Prediction": "not available",
            "Confidence": "-",
            "Model": "-",
            "Target": "-",
            "Source": "-",
            "Saved at": "-",
        }
    feature_date = row.get("date")
    return {
        "Type": label,
        "Feature date": _format_date(feature_date),
        "Target close date": _format_date(_next_trading_date_after(data, feature_date)),
        "Prediction": _prediction_class_name(row.get("predicted_class"), row.get("target")),
        "Confidence": _format_percent(row.get("confidence")),
        "Model": row.get("model_name") or "-",
        "Target": row.get("target") or "-",
        "Source": row.get("prediction_type") or "live",
        "Saved at": _format_datetime(row.get("created_at")),
    }


def _prediction_overview_binary_label(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    return _prediction_class_name(row.get("predicted_class"), row.get("target"))


def _prediction_overview_multiclass_label(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    return _prediction_class_name(row.get("predicted_class"), row.get("target"))


def _prediction_row_percent(row: pd.Series | None, column: str) -> str:
    if row is None:
        return "-"
    return _format_percent(row.get(column))


def _prediction_source_label(
    latest_binary: pd.Series | None,
    latest_multiclass: pd.Series | None,
) -> str:
    sources = []
    for row in (latest_binary, latest_multiclass):
        if row is None:
            continue
        sources.append(str(row.get("prediction_type") or "live"))
    if not sources:
        return "-"
    return " / ".join(dict.fromkeys(sources))


def _prediction_latest_date(
    latest_binary: pd.Series | None,
    latest_multiclass: pd.Series | None,
) -> str:
    return _format_date(_prediction_latest_date_value(latest_binary, latest_multiclass))


def _prediction_latest_date_value(
    latest_binary: pd.Series | None,
    latest_multiclass: pd.Series | None,
):
    return _daily_brief_latest_prediction_date(latest_binary, latest_multiclass)


def _prediction_data_freshness(ticker: str, data: pd.DataFrame) -> dict[str, object]:
    fallback_date = data["date"].max() if "date" in data.columns and not data.empty else None
    snapshot = _daily_brief_operations_snapshot(ticker)
    if not snapshot.get("latest_feature_date"):
        snapshot["latest_feature_date"] = fallback_date
    return snapshot


def _prediction_health_attention(freshness: dict[str, object]) -> str:
    overall = str(freshness.get("overall_status") or "").upper()
    daily = str(freshness.get("daily_status") or "").upper()
    if overall == "WARN" and daily == "OK":
        return (
            "Health is WARN, but latest daily_predict is OK. "
            "If warnings are TOPIX fallback and historical live prediction gaps only, operation can continue."
        )
    if overall == "FAIL" or daily == "FAIL":
        return "Health or daily_predict is failing. Check Operations before relying on this prediction."
    return ""


def _filter_prediction_rows(
    predictions: pd.DataFrame,
    *,
    target: str,
    model_name_contains: str | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    target_match = pd.Series(False, index=predictions.index)
    if "target" in predictions.columns:
        target_match = predictions["target"].astype(str) == target

    model_match = pd.Series(False, index=predictions.index)
    if "model_name" in predictions.columns:
        if model_name is not None:
            model_match = predictions["model_name"].astype(str) == model_name
        elif model_name_contains is not None:
            model_match = predictions["model_name"].astype(str).str.contains(
                model_name_contains,
                case=False,
                na=False,
            )

    if model_name is not None:
        return predictions[target_match & model_match].copy()
    return predictions[target_match | model_match].copy()


def _render_prediction_history_tab(ticker: str, data: pd.DataFrame, database_path, settings) -> None:
    st.caption(
        "This tab currently evaluates binary live/backfill predictions only. "
        "Multiclass history evaluation will be added later."
    )
    st.caption(
        "Saved prediction date fields are feature dates: the data row used by the model. "
        "The target close date is the next trading date after that feature date."
    )
    st.caption(
        "Post-start missing live feature dates mean no real daily prediction row was saved for those "
        "feature dates. Feature dates before the configured live operation start date are outside "
        "the live-operation window."
    )
    all_saved_predictions = _read_all_prediction_history(database_path, ticker=ticker)
    all_predictions = _read_prediction_history(
        database_path,
        ticker=ticker,
        target="next_close_binary",
        model_name=PRODUCTION_BINARY_MODEL_NAME,
    )
    operations = summarize_prediction_operations(
        all_predictions,
        data,
        ticker=ticker,
        lookback_trading_days=60,
        live_prediction_start_date=settings.operations.live_prediction_start_date,
    )
    missing_diagnostics = diagnose_missing_live_dates(
        all_saved_predictions,
        data,
        operations.get("missing_prediction_dates", []) or [],
        ticker=ticker,
        known_missed_live_dates=settings.operations.known_missed_live_dates,
    )
    _render_prediction_history_overview(
        all_saved_predictions=all_saved_predictions,
        binary_predictions=all_predictions,
        operations=operations,
        ticker=ticker,
    )
    _render_saved_prediction_browser(all_saved_predictions, data)
    _render_append_only_prediction_history(database_path, ticker)

    if all_predictions.empty:
        st.info("No saved binary prediction history yet.")
        st.code(
            f"python main.py predict-binary --ticker {ticker}",
            language="powershell",
        )
        return

    source_options = {
        "Live only": "live",
        "Backfill only": "backfill",
        "Live + Backfill": None,
    }
    source_label = st.radio(
        "History source",
        options=list(source_options),
        horizontal=True,
        key="prediction_history_source_radio",
    )
    selected_source = source_options[source_label]
    if selected_source is None:
        predictions = all_predictions.copy()
    elif "prediction_type" in all_predictions.columns:
        predictions = all_predictions[
            all_predictions["prediction_type"].fillna("live") == selected_source
        ].copy()
    else:
        predictions = all_predictions.copy() if selected_source == "live" else pd.DataFrame()

    with st.expander("Prediction operations and missing live dates", expanded=True):
        _render_prediction_history_operations(
            operations,
            ticker=ticker,
            missing_diagnostics=missing_diagnostics,
        )
        _render_daily_predict_log_summary()

    if predictions.empty:
        st.info(f"No {source_label.lower()} prediction rows are available yet.")
        return

    history, metrics = evaluate_prediction_history(predictions, data)
    if history.empty:
        st.info("Prediction history exists, but it could not be matched with price data yet.")
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric("Binary evaluated rows", metrics["total_predictions"])
    metric_cols[1].metric("Confirmed", metrics["settled_predictions"])
    metric_cols[2].metric("Pending", metrics["pending_predictions"])
    metric_cols[3].metric("Correct", metrics["correct_predictions"])

    accuracy_cols = st.columns(4)
    accuracy_cols[0].metric("Accuracy", _format_percent(metrics["accuracy"]))
    accuracy_cols[1].metric("Up accuracy", _format_percent(metrics["up_prediction_accuracy"]))
    accuracy_cols[2].metric("Down accuracy", _format_percent(metrics["down_prediction_accuracy"]))
    accuracy_cols[3].metric("Avg confidence", _format_percent(metrics["average_confidence"]))

    confidence_accuracy = metrics.get("confidence_accuracy")
    if isinstance(confidence_accuracy, pd.DataFrame) and not confidence_accuracy.empty:
        st.caption("Accuracy by confidence bucket")
        st.plotly_chart(
            _build_confidence_accuracy_chart(confidence_accuracy),
            use_container_width=True,
        )
        display_confidence = confidence_accuracy.copy()
        display_confidence["accuracy"] = display_confidence["accuracy"].map(_format_percent)
        with st.expander("Confidence bucket details", expanded=False):
            st.dataframe(display_confidence, use_container_width=True)

    display_columns = [
        "date",
        "ticker",
        "prediction_type",
        "created_at",
        "prediction_direction",
        "probability_up",
        "probability_down",
        "confidence",
        "actual_return",
        "actual_direction",
        "status",
        "is_correct",
    ]
    display = history[[column for column in display_columns if column in history.columns]].copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    if "created_at" in display.columns:
        display["created_at"] = pd.to_datetime(display["created_at"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    for column in ("probability_up", "probability_down", "confidence", "actual_return"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    display["is_correct"] = display.apply(
        lambda row: _format_correctness(row.get("is_correct"), row.get("status")),
        axis=1,
    )
    display["status"] = display["status"].map(
        lambda value: "Confirmed" if value == "confirmed" else "Pending"
    )
    display = display.rename(
        columns={
            "date": "Feature date",
            "ticker": "Ticker",
            "prediction_type": "Source",
            "created_at": "Saved at",
            "prediction_direction": "Prediction",
            "probability_up": "Up probability",
            "probability_down": "Down probability",
            "confidence": "Confidence",
            "actual_return": "Actual next-day return",
            "actual_direction": "Actual",
            "status": "Status",
            "is_correct": "Result",
        }
    )
    st.caption(
        "Binary evaluated rows counts only the selected production binary rows below. "
        "It is not the total number of saved SQLite prediction records."
    )
    st.caption("Saved predictions are matched with the next available trading day in local data.")
    with st.expander("Binary prediction history", expanded=True):
        st.dataframe(display, use_container_width=True)


def _render_append_only_prediction_history(database_path, ticker: str) -> None:
    append_only = summarize_append_only_history_for_ui(database_path, ticker=ticker)
    summary = append_only["summary"]
    run_summary = append_only["run_summary"]
    record_summary = append_only["record_summary"]
    evaluation_summary = append_only["evaluation_summary"]
    comparison = append_only["comparison"]
    runs = append_only["runs"]
    records = append_only["records"]
    feature_date_summary = append_only["feature_date_summary"]
    comparison_rows = append_only["comparison_rows"]

    with st.expander("Append-only Run History", expanded=False):
        st.caption(
            "Append-only records are an audit/history view. They are not yet the source of truth "
            "for Prediction History metrics."
        )
        st.caption(
            "Production metrics and health-check still use the existing predictions snapshot. "
            "Dual-write is currently manual opt-in and not scheduled."
        )

        run_cols = st.columns(5)
        run_cols[0].metric("Total runs", _format_number(run_summary["total_runs"], digits=0))
        run_cols[1].metric("Total records", _format_number(record_summary["total_records"], digits=0))
        run_cols[2].metric("Official records", _format_number(record_summary["official_records"], digits=0))
        run_cols[3].metric("Latest records", _format_number(record_summary["latest_records"], digits=0))
        run_cols[4].metric("Existing snapshot rows", _format_number(summary["existing_predictions_count"], digits=0))

        latest_cols = st.columns(5)
        latest_cols[0].metric("Latest run type", run_summary["latest_run_type"] or "-")
        latest_cols[1].metric("Latest source", run_summary["latest_source"] or "-")
        latest_cols[2].metric("Latest feature date", _format_date(run_summary["latest_feature_date"]))
        latest_cols[3].metric("Data freshness", run_summary["latest_data_freshness_status"] or "-")
        latest_cols[4].metric("Stale at run", _format_enabled(run_summary["latest_stale_at_run_time"]))

        st.caption(f"Latest run id: {run_summary['latest_run_id'] or '-'}")
        st.caption(f"Latest run timestamp: {_format_datetime(run_summary['latest_run_timestamp'])}")
        if run_summary.get("latest_notes"):
            st.caption(f"Latest notes: {run_summary['latest_notes']}")

        eval_cols = st.columns(5)
        eval_cols[0].metric("Pending", _format_number(evaluation_summary["pending_count"], digits=0))
        eval_cols[1].metric("Hit", _format_number(evaluation_summary["hit_count"], digits=0))
        eval_cols[2].metric("Miss", _format_number(evaluation_summary["miss_count"], digits=0))
        eval_cols[3].metric("Unevaluated", _format_number(evaluation_summary["unevaluated_count"], digits=0))
        eval_cols[4].metric(
            "Latest pending target",
            _format_date(evaluation_summary["latest_pending_target_close_date"]),
        )
        st.caption(
            "Evaluation status is read-only here. This view does not fetch actuals, update records, "
            "or change production metrics."
        )

        matched_count = _append_only_matched_count(comparison_rows)
        mismatch_count = _append_only_mismatch_count(comparison_rows)
        compare_cols = st.columns(4)
        compare_cols[0].metric("Snapshot official", _format_number(comparison["snapshot_official_count"], digits=0))
        compare_cols[1].metric("Append official", _format_number(comparison["append_official_count"], digits=0))
        compare_cols[2].metric("Matched records", _format_number(matched_count, digits=0))
        compare_cols[3].metric("Mismatches", _format_number(mismatch_count, digits=0))
        if comparison.get("matches"):
            st.success("Append-only latest official records match the current predictions snapshot.")
        else:
            st.warning(
                "Append-only latest official records do not fully match the current predictions snapshot. "
                "Review details before using append-only for any later phase."
            )

        if feature_date_summary.empty:
            st.info("No append-only feature-date run history is available yet.")
        else:
            st.caption("Feature-date run history")
            st.dataframe(
                _append_only_feature_date_display_frame(feature_date_summary),
                use_container_width=True,
                hide_index=True,
            )

        _render_append_only_run_detail(database_path, ticker, runs)

        with st.expander("Append-only run summary", expanded=False):
            if runs.empty:
                st.info("No append-only runs are available yet.")
            else:
                st.dataframe(
                    _append_only_runs_display_frame(runs),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Append-only record summary", expanded=False):
            if records.empty:
                st.info("No append-only records are available yet.")
            else:
                st.dataframe(
                    _append_only_records_display_frame(records),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Append-only vs snapshot comparison", expanded=False):
            if comparison_rows.empty:
                st.info("No append-only comparison rows are available yet.")
            else:
                st.dataframe(
                    _append_only_comparison_display_frame(comparison_rows),
                    use_container_width=True,
                    hide_index=True,
                )


def _render_append_only_run_detail(database_path, ticker: str, runs: pd.DataFrame) -> None:
    st.caption("Run ID drilldown")
    if runs.empty:
        st.info("No append-only run is available for run_id detail yet.")
        return

    ordered_runs = runs.copy()
    ordered_runs["_run_timestamp"] = pd.to_datetime(
        ordered_runs.get("run_timestamp"),
        errors="coerce",
    )
    ordered_runs = ordered_runs.sort_values(
        ["_run_timestamp", "created_at", "run_id"],
        ascending=[False, False, False],
    )
    run_options = ordered_runs["run_id"].astype(str).tolist()
    selected_run_id = st.selectbox(
        "Select append-only run_id",
        options=run_options,
        index=0,
        key="append_only_run_detail_select",
    )
    detail = build_append_only_run_detail(
        database_path,
        ticker=ticker,
        run_id=selected_run_id,
    )
    selected_run = detail["run"]
    selected_records = detail["record_table"]
    selected_summary = detail["run_summary"]
    selected_compare = detail["snapshot_compare"]

    with st.expander("Selected run detail", expanded=True):
        st.caption(
            "この表示はappend-only監査履歴です。現時点ではPrediction History metricsや"
            "health-checkの判定基準ではありません。"
        )
        st.caption(
            "daily_predictへの自動dual-writeはまだ有効化していません。"
            "既存predictions snapshotとの比較は監査目的です。"
        )

        summary_cols = st.columns(6)
        summary_cols[0].metric("Records", _format_number(selected_summary["record_count"], digits=0))
        summary_cols[1].metric(
            "Official",
            _format_number(selected_summary["official_record_count"], digits=0),
        )
        summary_cols[2].metric(
            "Latest",
            _format_number(selected_summary["latest_record_count"], digits=0),
        )
        summary_cols[3].metric(
            "Binary",
            _format_number(selected_summary["binary_record_count"], digits=0),
        )
        summary_cols[4].metric(
            "Multiclass",
            _format_number(selected_summary["multiclass_record_count"], digits=0),
        )
        summary_cols[5].metric("Pending", _format_number(selected_summary["pending_count"], digits=0))

        result_cols = st.columns(4)
        result_cols[0].metric("Hit", _format_number(selected_summary["hit_count"], digits=0))
        result_cols[1].metric("Miss", _format_number(selected_summary["miss_count"], digits=0))
        result_cols[2].metric(
            "Unevaluated",
            _format_number(selected_summary["unevaluated_count"], digits=0),
        )
        result_cols[3].metric("Snapshot mismatches", _format_number(selected_compare["mismatch_count"], digits=0))

        st.caption(f"Targets: {selected_summary['targets_included'] or '-'}")
        st.caption(f"Models: {selected_summary['model_names_included'] or '-'}")

        metadata = _append_only_selected_run_metadata_frame(selected_run)
        if not metadata.empty:
            st.caption("Selected run metadata")
            st.dataframe(metadata, use_container_width=True, hide_index=True)

        compare_cols = st.columns(5)
        compare_cols[0].metric(
            "Snapshot official",
            _format_number(selected_compare["snapshot_official_count"], digits=0),
        )
        compare_cols[1].metric("Run records", _format_number(selected_compare["run_record_count"], digits=0))
        compare_cols[2].metric("Matched", _format_number(selected_compare["matched_count"], digits=0))
        compare_cols[3].metric(
            "Missing in append-only",
            _format_number(selected_compare["missing_in_append_only_count"], digits=0),
        )
        compare_cols[4].metric(
            "Missing in snapshot",
            _format_number(selected_compare["missing_in_snapshot_count"], digits=0),
        )
        if selected_compare.get("matches"):
            st.success("Selected run records match the current official predictions snapshot.")
        else:
            st.warning("Selected run differs from the current official predictions snapshot. Review the table below.")

        st.caption("Selected run records")
        if selected_records.empty:
            st.info("No append-only records are linked to this run_id.")
        else:
            st.dataframe(
                _append_only_records_display_frame(selected_records),
                use_container_width=True,
                hide_index=True,
            )

        compare_rows = pd.DataFrame(selected_compare.get("rows", []))
        st.caption("Selected run vs current official snapshot")
        if compare_rows.empty:
            st.info("No comparison rows are available for this run_id.")
        else:
            st.dataframe(
                _append_only_selected_run_compare_display_frame(compare_rows),
                use_container_width=True,
                hide_index=True,
            )


def _append_only_selected_run_metadata_frame(run: dict[str, object]) -> pd.DataFrame:
    if not run:
        return pd.DataFrame()
    rows = [
        ("run_id", run.get("run_id")),
        ("run_timestamp", _format_datetime(run.get("run_timestamp"))),
        ("ticker", run.get("ticker")),
        ("run_type", run.get("run_type")),
        ("source", run.get("source")),
        ("command_name", run.get("command_name")),
        ("working_directory", run.get("working_directory")),
        ("latest_feature_date", _format_date(run.get("latest_feature_date"))),
        ("latest_raw_date", _format_date(run.get("latest_raw_date"))),
        ("expected_latest_trading_date", _format_date(run.get("expected_latest_trading_date"))),
        ("data_freshness_status", run.get("data_freshness_status")),
        ("stale_at_run_time", _format_enabled(run.get("stale_at_run_time"))),
        ("created_at", _format_datetime(run.get("created_at"))),
        ("notes", run.get("notes")),
    ]
    return pd.DataFrame(rows, columns=["Field", "Value"])


def _append_only_matched_count(comparison_rows: pd.DataFrame) -> int:
    if comparison_rows.empty or "match" not in comparison_rows.columns:
        return 0
    return int(comparison_rows["match"].fillna(False).astype(bool).sum())


def _append_only_mismatch_count(comparison_rows: pd.DataFrame) -> int:
    if comparison_rows.empty or "match" not in comparison_rows.columns:
        return 0
    return int((~comparison_rows["match"].fillna(False).astype(bool)).sum())


def _append_only_feature_date_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("feature_date", "latest_run_timestamp"):
        if column in display.columns:
            display[column] = display[column].map(_format_datetime if column.endswith("timestamp") else _format_date)
    return display.rename(
        columns={
            "feature_date": "Feature date",
            "run_count": "Runs",
            "run_ids": "Run ids",
            "record_count": "Records",
            "official_records": "Official",
            "latest_records": "Latest",
            "binary_records": "Binary",
            "multiclass_records": "Multiclass",
            "run_types": "Run types",
            "manual_run_ids": "Manual run ids",
            "scheduled_run_ids": "Scheduled run ids",
            "official_run_ids": "Official run ids",
            "sources": "Sources",
            "latest_run_timestamp": "Latest run timestamp",
            "latest_run_id": "Latest run id",
        }
    )


def _append_only_runs_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("run_timestamp", "created_at"):
        if column in display.columns:
            display[column] = display[column].map(_format_datetime)
    columns = [
        "run_id",
        "run_timestamp",
        "ticker",
        "run_type",
        "command_name",
        "source",
        "data_freshness_status",
        "latest_raw_date",
        "latest_feature_date",
        "expected_latest_trading_date",
        "stale_at_run_time",
        "notes",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "run_id": "Run id",
            "run_timestamp": "Run timestamp",
            "ticker": "Ticker",
            "run_type": "Run type",
            "command_name": "Command",
            "source": "Source",
            "data_freshness_status": "Data freshness",
            "latest_raw_date": "Latest raw date",
            "latest_feature_date": "Latest feature date",
            "expected_latest_trading_date": "Expected latest trading date",
            "stale_at_run_time": "Stale at run",
            "notes": "Notes",
        }
    )


def _append_only_records_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("feature_date", "target_close_date", "evaluated_at"):
        if column in display.columns:
            display[column] = display[column].map(_format_datetime if column == "evaluated_at" else _format_date)
    if "confidence" in display.columns:
        display["confidence"] = display["confidence"].map(_format_percent)
    columns = [
        "feature_date",
        "target_close_date",
        "target",
        "model_name",
        "model_type",
        "predicted_label",
        "predicted_direction",
        "confidence",
        "source",
        "is_official",
        "is_latest",
        "result",
        "actual_return",
        "evaluated_at",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "feature_date": "Feature date",
            "target_close_date": "Target close date",
            "target": "Target",
            "model_name": "Model",
            "model_type": "Model type",
            "predicted_label": "Predicted label",
            "predicted_direction": "Predicted direction",
            "confidence": "Confidence",
            "source": "Source",
            "is_official": "Official",
            "is_latest": "Latest",
            "result": "Result",
            "actual_return": "Actual return",
            "evaluated_at": "Evaluated at",
        }
    )


def _append_only_comparison_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    columns = [
        "feature_date",
        "target",
        "model_name",
        "snapshot_predicted_label",
        "append_predicted_label",
        "snapshot_confidence",
        "append_confidence",
        "append_run_id",
        "append_run_timestamp",
        "match",
        "reason",
    ]
    display = display[[column for column in columns if column in display.columns]]
    for column in ("snapshot_confidence", "append_confidence"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "append_run_timestamp" in display.columns:
        display["append_run_timestamp"] = display["append_run_timestamp"].map(_format_datetime)
    return display.rename(
        columns={
            "feature_date": "Feature date",
            "target": "Target",
            "model_name": "Model",
            "snapshot_predicted_label": "Snapshot label",
            "append_predicted_label": "Append-only label",
            "snapshot_confidence": "Snapshot confidence",
            "append_confidence": "Append-only confidence",
            "append_run_id": "Append-only run id",
            "append_run_timestamp": "Append-only run timestamp",
            "match": "Match",
            "reason": "Reason",
        }
    )


def _append_only_selected_run_compare_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    columns = [
        "feature_date",
        "target",
        "model_name",
        "snapshot_predicted_label",
        "append_predicted_label",
        "snapshot_direction",
        "append_direction",
        "snapshot_confidence",
        "append_confidence",
        "confidence_delta",
        "label_match",
        "direction_match",
        "confidence_match",
        "match",
        "reason",
    ]
    display = display[[column for column in columns if column in display.columns]]
    for column in ("snapshot_confidence", "append_confidence", "confidence_delta"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "feature_date": "Feature date",
            "target": "Target",
            "model_name": "Model",
            "snapshot_predicted_label": "Snapshot label",
            "append_predicted_label": "Run label",
            "snapshot_direction": "Snapshot direction",
            "append_direction": "Run direction",
            "snapshot_confidence": "Snapshot confidence",
            "append_confidence": "Run confidence",
            "confidence_delta": "Confidence delta",
            "label_match": "Label match",
            "direction_match": "Direction match",
            "confidence_match": "Confidence match",
            "match": "Match",
            "reason": "Reason",
        }
    )


def _render_prediction_history_overview(
    *,
    all_saved_predictions: pd.DataFrame,
    binary_predictions: pd.DataFrame,
    operations: dict[str, object],
    ticker: str,
) -> None:
    st.subheader("Prediction History Overview")
    st.caption(
        "This tab is for saved prediction records. live rows are real daily operation records; "
        "backfill rows are historical inspection rows created later. The saved date is the feature date, "
        "not the database save time or an explicit target date."
    )
    latest_live_binary = _latest_prediction_history_row(
        all_saved_predictions,
        target="next_close_binary",
        prediction_type="live",
        model_name=PRODUCTION_BINARY_MODEL_NAME,
    )
    latest_live_multiclass = _latest_prediction_history_row(
        all_saved_predictions,
        target="next_close_multiclass",
        prediction_type="live",
        model_name=PRODUCTION_MULTICLASS_MODEL_NAME,
    )
    operations_snapshot = _daily_brief_operations_snapshot(ticker)
    scope_summary = _prediction_history_scope_summary(all_saved_predictions)

    overview_cols = st.columns(5)
    overview_cols[0].metric("Latest live feature date", _format_date(operations.get("latest_prediction_date")))
    overview_cols[1].metric("Latest binary", _prediction_history_label(latest_live_binary))
    overview_cols[2].metric("Binary confidence", _prediction_row_percent(latest_live_binary, "confidence"))
    overview_cols[3].metric("Latest multiclass", _prediction_history_label(latest_live_multiclass))
    overview_cols[4].metric("Multiclass confidence", _prediction_row_percent(latest_live_multiclass, "confidence"))

    count_cols = st.columns(6)
    count_cols[0].metric("Total saved rows", _format_number(scope_summary["total_rows"], digits=0))
    count_cols[1].metric("Production live rows", _format_number(scope_summary["production_live_rows"], digits=0))
    count_cols[2].metric("Latest feature-date rows", _format_number(scope_summary["latest_feature_date_rows"], digits=0))
    count_cols[3].metric("Experimental rows", _format_number(scope_summary["experimental_live_rows"], digits=0))
    count_cols[4].metric("Backfill rows", _format_number(scope_summary["backfill_rows"], digits=0))
    count_cols[5].metric("Latest saved_at", _format_datetime(_latest_prediction_saved_at(all_saved_predictions)))

    status_cols = st.columns(4)
    status_cols[0].metric("Binary production live", _format_number(scope_summary["binary_production_live_rows"], digits=0))
    status_cols[1].metric(
        "Multiclass production live",
        _format_number(scope_summary["multiclass_production_live_rows"], digits=0),
    )
    status_cols[2].metric("daily_predict", operations_snapshot.get("daily_status") or "-")
    status_cols[3].metric("Health", operations_snapshot.get("overall_status") or "-")

    st.caption(
        "Latest feature-date rows are only the rows for the newest feature date, usually binary + multiclass = 2. "
        "They are not the total saved prediction history."
    )
    st.caption(
        "Current SQLite storage keeps one row per feature date / ticker / target / model. "
        "If daily_predict runs again with the same feature date, that row is updated rather than appended."
    )
    with st.expander("Record scope details", expanded=False):
        st.dataframe(
            _prediction_history_scope_frame(all_saved_predictions),
            use_container_width=True,
            hide_index=True,
        )

    note = _prediction_history_health_note(operations_snapshot)
    if note:
        st.info(note)
    _render_prediction_history_missing_note(operations)

    if binary_predictions.empty and not all_saved_predictions.empty:
        st.caption("Binary evaluation rows are missing, but other saved prediction rows are available below.")


def _is_production_prediction_row(row: pd.Series) -> bool:
    target = str(row.get("target") or "")
    model_name = str(row.get("model_name") or "")
    return (
        target == "next_close_binary"
        and model_name == PRODUCTION_BINARY_MODEL_NAME
    ) or (
        target == "next_close_multiclass"
        and model_name == PRODUCTION_MULTICLASS_MODEL_NAME
    )


def _prediction_history_scope_summary(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {
            "total_rows": 0,
            "live_rows": 0,
            "backfill_rows": 0,
            "production_live_rows": 0,
            "binary_production_live_rows": 0,
            "multiclass_production_live_rows": 0,
            "experimental_live_rows": 0,
            "latest_feature_date": None,
            "latest_feature_date_rows": 0,
            "latest_feature_date_production_rows": 0,
        }

    frame = rows.copy()
    prediction_type = frame.get("prediction_type", pd.Series(index=frame.index, dtype=str)).fillna("live").astype(str)
    targets = frame.get("target", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    models = frame.get("model_name", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    dates = pd.to_datetime(frame.get("date"), errors="coerce")
    live_mask = prediction_type.eq("live")
    backfill_mask = prediction_type.eq("backfill")
    binary_prod_mask = targets.eq("next_close_binary") & models.eq(PRODUCTION_BINARY_MODEL_NAME)
    multiclass_prod_mask = targets.eq("next_close_multiclass") & models.eq(PRODUCTION_MULTICLASS_MODEL_NAME)
    production_mask = binary_prod_mask | multiclass_prod_mask
    latest_feature_date = dates.dropna().max() if not dates.dropna().empty else None
    latest_mask = dates.eq(latest_feature_date) if latest_feature_date is not None else pd.Series(False, index=frame.index)

    return {
        "total_rows": int(len(frame)),
        "live_rows": int(live_mask.sum()),
        "backfill_rows": int(backfill_mask.sum()),
        "production_live_rows": int((live_mask & production_mask).sum()),
        "binary_production_live_rows": int((live_mask & binary_prod_mask).sum()),
        "multiclass_production_live_rows": int((live_mask & multiclass_prod_mask).sum()),
        "experimental_live_rows": int((live_mask & ~production_mask).sum()),
        "latest_feature_date": latest_feature_date,
        "latest_feature_date_rows": int(latest_mask.sum()),
        "latest_feature_date_production_rows": int((latest_mask & production_mask).sum()),
    }


def _prediction_history_scope_frame(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Scope",
        "Rows",
        "Min feature date",
        "Max feature date",
        "Latest saved at",
        "Meaning",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    frame = rows.copy()
    prediction_type = frame.get("prediction_type", pd.Series(index=frame.index, dtype=str)).fillna("live").astype(str)
    targets = frame.get("target", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    models = frame.get("model_name", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    live_mask = prediction_type.eq("live")
    backfill_mask = prediction_type.eq("backfill")
    production_mask = (
        targets.eq("next_close_binary") & models.eq(PRODUCTION_BINARY_MODEL_NAME)
    ) | (
        targets.eq("next_close_multiclass") & models.eq(PRODUCTION_MULTICLASS_MODEL_NAME)
    )
    binary_mask = targets.eq("next_close_binary") & models.eq(PRODUCTION_BINARY_MODEL_NAME)
    multiclass_mask = targets.eq("next_close_multiclass") & models.eq(PRODUCTION_MULTICLASS_MODEL_NAME)

    scopes = [
        (
            "All saved rows",
            pd.Series(True, index=frame.index),
            "Everything in SQLite predictions for this ticker.",
        ),
        (
            "Production live rows",
            live_mask & production_mask,
            "Rows used for normal daily operation history.",
        ),
        (
            "Binary production live",
            live_mask & binary_mask,
            "Rows used by the binary accuracy section below.",
        ),
        (
            "Multiclass production live",
            live_mask & multiclass_mask,
            "Current production multiclass rows; reference only for direction.",
        ),
        (
            "Experimental live rows",
            live_mask & ~production_mask,
            "Past multiclass variants saved as live; hidden from production scope by default.",
        ),
        (
            "Backfill rows",
            backfill_mask,
            "Historical inspection rows created after the fact.",
        ),
    ]
    records = []
    for scope, mask, meaning in scopes:
        subset = frame[mask].copy()
        if subset.empty:
            records.append(
                {
                    "Scope": scope,
                    "Rows": 0,
                    "Min feature date": "-",
                    "Max feature date": "-",
                    "Latest saved at": "-",
                    "Meaning": meaning,
                }
            )
            continue
        dates = pd.to_datetime(subset.get("date"), errors="coerce")
        saved_at = pd.to_datetime(subset.get("created_at"), errors="coerce")
        records.append(
            {
                "Scope": scope,
                "Rows": int(len(subset)),
                "Min feature date": _format_date(dates.min()),
                "Max feature date": _format_date(dates.max()),
                "Latest saved at": _format_datetime(saved_at.max()),
                "Meaning": meaning,
            }
        )
    return pd.DataFrame(records, columns=columns)


def _render_prediction_history_missing_note(operations: dict[str, object]) -> None:
    missing_count = safe_int(operations.get("missing_prediction_count"))
    if missing_count is None:
        missing_count = len(operations.get("missing_prediction_dates", []) or [])
    pre_start_count = safe_int(
        operations.get(
            "pre_start_excluded_count",
            operations.get("ignored_pre_start_missing_count"),
        )
    ) or 0
    start_date = _format_date(operations.get("live_prediction_start_date"))
    if missing_count <= 0:
        if pre_start_count > 0:
            st.caption(
                f"Feature dates before {start_date} are outside the configured live-operation "
                f"window and are excluded from missing checks ({pre_start_count} dates)."
            )
        return
    st.caption(
        f"Missing live feature dates after {start_date}: {missing_count} dates. "
        "Feature dates before the configured operation start are excluded from this count. "
        "It is separate from backfill and does not mean current daily_predict is failing."
    )


def _render_saved_prediction_browser(all_saved_predictions: pd.DataFrame, data: pd.DataFrame) -> None:
    st.subheader("Saved Prediction Browser")
    st.caption(
        "Use this for record inspection. Feature date is the input data date; target close date is inferred "
        "from the next trading date in the local feature data. Binary accuracy evaluation remains below."
    )
    if all_saved_predictions.empty:
        st.info("No saved prediction records are available yet.")
        return

    filtered, scope_label = _prediction_history_filter_controls(all_saved_predictions)
    st.caption(
        f"Current browser scope: {scope_label}. "
        "Use 'All saved rows' only when you intentionally want production, experimental, and backfill rows together."
    )
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.caption("Confidence over time")
        st.plotly_chart(_build_prediction_history_confidence_chart(filtered), use_container_width=True)
    with chart_cols[1]:
        st.caption("Predicted class over time")
        st.plotly_chart(_build_prediction_history_class_chart(filtered), use_container_width=True)

    with st.expander("Latest live predictions in current scope", expanded=True):
        latest_live = filtered[filtered.get("prediction_type", pd.Series(dtype=str)).fillna("live") == "live"].copy()
        if latest_live.empty:
            st.info("No live rows match the current browser filters.")
        else:
            st.caption(
                "This is a record browser, not the binary accuracy total. "
                "The newest feature date normally has two production rows: binary and multiclass."
            )
            st.dataframe(
                _prediction_history_records_display_frame(latest_live.head(30), data),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Multiclass prediction history", expanded=False):
        multiclass_rows = filtered[filtered["target"].astype(str).eq("next_close_multiclass")].copy()
        if multiclass_rows.empty:
            st.info("No multiclass prediction records match the current filters.")
        else:
            st.dataframe(
                _prediction_history_records_display_frame(multiclass_rows, data),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Backfill records", expanded=False):
        backfill_rows = filtered[
            filtered.get("prediction_type", pd.Series(dtype=str)).fillna("live").astype(str).eq("backfill")
        ].copy()
        if backfill_rows.empty:
            st.info("No backfill records match the current filters.")
        else:
            st.dataframe(
                _prediction_history_records_display_frame(backfill_rows, data),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Raw prediction records", expanded=False):
        st.dataframe(
            _prediction_history_records_display_frame(filtered, data),
            use_container_width=True,
            hide_index=True,
        )


def _prediction_history_filter_controls(rows: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    filters = st.columns(6)
    scope_label = filters[0].selectbox(
        "Record scope",
        options=[
            "Production live",
            "Production live + backfill",
            "Experimental / variants",
            "Backfill only",
            "All saved rows",
        ],
        index=0,
        key="prediction_history_browser_scope",
    )
    scoped = _apply_prediction_history_record_scope(rows, scope_label)
    source = filters[1].selectbox(
        "Source",
        options=["all", "live", "backfill"],
        index=0,
        key="prediction_history_browser_source",
    )
    target_label = filters[2].selectbox(
        "Prediction type",
        options=["all", "binary", "multiclass"],
        index=0,
        key="prediction_history_browser_target",
    )
    model_options = ["all"] + sorted(scoped.get("model_name", pd.Series(dtype=str)).dropna().astype(str).unique())
    model_name = filters[3].selectbox(
        "Model name",
        options=model_options,
        index=0,
        key="prediction_history_browser_model",
    )
    confidence_threshold = filters[4].slider(
        "Min confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        key="prediction_history_browser_confidence",
    )

    date_range = _prediction_history_date_range(scoped)
    selected_range = filters[5].date_input(
        "Feature date range",
        value=date_range,
        key="prediction_history_browser_date_range",
    )

    filtered = scoped.copy()
    if source != "all" and "prediction_type" in filtered.columns:
        filtered = filtered[filtered["prediction_type"].fillna("live").astype(str).eq(source)].copy()
    if target_label != "all" and "target" in filtered.columns:
        target = "next_close_binary" if target_label == "binary" else "next_close_multiclass"
        filtered = filtered[filtered["target"].astype(str).eq(target)].copy()
    if model_name != "all" and "model_name" in filtered.columns:
        filtered = filtered[filtered["model_name"].astype(str).eq(model_name)].copy()
    if "confidence" in filtered.columns and confidence_threshold > 0:
        confidence = pd.to_numeric(filtered["confidence"], errors="coerce")
        filtered = filtered[confidence >= confidence_threshold].copy()
    filtered = _filter_prediction_history_date_range(filtered, selected_range)
    return _sort_prediction_history_rows(filtered), scope_label


def _apply_prediction_history_record_scope(rows: pd.DataFrame, scope_label: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    frame = rows.copy()
    prediction_type = frame.get("prediction_type", pd.Series(index=frame.index, dtype=str)).fillna("live").astype(str)
    targets = frame.get("target", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    models = frame.get("model_name", pd.Series(index=frame.index, dtype=str)).fillna("").astype(str)
    production_mask = (
        targets.eq("next_close_binary") & models.eq(PRODUCTION_BINARY_MODEL_NAME)
    ) | (
        targets.eq("next_close_multiclass") & models.eq(PRODUCTION_MULTICLASS_MODEL_NAME)
    )
    live_mask = prediction_type.eq("live")
    backfill_mask = prediction_type.eq("backfill")

    if scope_label == "Production live":
        return frame[live_mask & production_mask].copy()
    if scope_label == "Production live + backfill":
        return frame[production_mask].copy()
    if scope_label == "Experimental / variants":
        return frame[live_mask & ~production_mask].copy()
    if scope_label == "Backfill only":
        return frame[backfill_mask].copy()
    return frame


def _prediction_history_date_range(rows: pd.DataFrame):
    if rows.empty or "date" not in rows.columns:
        today = pd.Timestamp.today().date()
        return (today, today)
    dates = pd.to_datetime(rows["date"], errors="coerce").dropna()
    if dates.empty:
        today = pd.Timestamp.today().date()
        return (today, today)
    return (dates.min().date(), dates.max().date())


def _filter_prediction_history_date_range(rows: pd.DataFrame, selected_range) -> pd.DataFrame:
    if rows.empty or "date" not in rows.columns:
        return rows
    if not isinstance(selected_range, (tuple, list)) or len(selected_range) != 2:
        return rows
    start, end = selected_range
    dates = pd.to_datetime(rows["date"], errors="coerce")
    return rows[(dates.dt.date >= start) & (dates.dt.date <= end)].copy()


def _sort_prediction_history_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    result = rows.copy()
    result["_sort_date"] = pd.to_datetime(result.get("date"), errors="coerce")
    result["_sort_created_at"] = pd.to_datetime(result.get("created_at"), errors="coerce")
    result = result.sort_values(["_sort_date", "_sort_created_at"], ascending=[False, False])
    return result.drop(columns=["_sort_date", "_sort_created_at"], errors="ignore").reset_index(drop=True)


def _prediction_history_records_display_frame(
    rows: pd.DataFrame,
    data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    display = rows.copy()
    display["prediction_label"] = display.apply(_prediction_label, axis=1)
    if "date" in display.columns:
        display["target_close_date"] = display["date"].map(
            lambda value: _next_trading_date_after(data, value)
        )
    for column in ("date",):
        if column in display.columns:
            display[column] = pd.to_datetime(display[column], errors="coerce").dt.strftime("%Y-%m-%d")
    if "target_close_date" in display.columns:
        display["target_close_date"] = pd.to_datetime(
            display["target_close_date"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    if "created_at" in display.columns:
        display["created_at"] = pd.to_datetime(display["created_at"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    for column in (
        "confidence",
        "probability_surge",
        "probability_up",
        "probability_flat",
        "probability_down",
        "probability_crash",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    columns = [
        "date",
        "target_close_date",
        "ticker",
        "target",
        "model_name",
        "prediction_type",
        "prediction_label",
        "confidence",
        "probability_up",
        "probability_down",
        "probability_surge",
        "probability_flat",
        "probability_crash",
        "created_at",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "date": "Feature date",
            "target_close_date": "Target close date",
            "ticker": "Ticker",
            "target": "Target",
            "model_name": "Model",
            "prediction_type": "Source",
            "prediction_label": "Prediction",
            "confidence": "Confidence",
            "probability_up": "Up",
            "probability_down": "Down",
            "probability_surge": "Surge",
            "probability_flat": "Flat",
            "probability_crash": "Crash",
            "created_at": "Saved at",
        }
    )


def _build_prediction_history_confidence_chart(rows: pd.DataFrame) -> go.Figure:
    frame = _prediction_history_chart_frame(rows)
    fig = go.Figure()
    if frame.empty:
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=30))
        return fig
    for target, group in frame.groupby("target_label"):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["confidence"],
                mode="lines+markers",
                name=str(target),
            )
        )
    fig.update_layout(
        height=280,
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis_title="Feature date",
        yaxis_title="Confidence",
        margin=dict(l=20, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_prediction_history_class_chart(rows: pd.DataFrame) -> go.Figure:
    frame = _prediction_history_chart_frame(rows)
    fig = go.Figure()
    if frame.empty:
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=10, b=30))
        return fig
    for target, group in frame.groupby("target_label"):
        fig.add_trace(
            go.Scatter(
                x=group["date"],
                y=group["predicted_class"],
                mode="markers",
                name=str(target),
                text=group["prediction_label"],
                hovertemplate="Date=%{x}<br>Class=%{text}<extra></extra>",
            )
        )
    fig.update_layout(
        height=280,
        xaxis_title="Feature date",
        yaxis_title="Predicted class id",
        margin=dict(l=20, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _prediction_history_chart_frame(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    frame = rows.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce")
    frame["confidence"] = pd.to_numeric(frame.get("confidence"), errors="coerce")
    frame["predicted_class"] = pd.to_numeric(frame.get("predicted_class"), errors="coerce")
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return frame
    frame["target_label"] = frame.get("target", "").map(_prediction_history_target_label)
    frame["prediction_label"] = frame.apply(_prediction_label, axis=1)
    return frame.sort_values("date").tail(200)


def _prediction_history_target_label(value) -> str:
    text = str(value or "")
    if "multiclass" in text:
        return "multiclass"
    if "binary" in text:
        return "binary"
    return text or "unknown"


def _latest_prediction_history_row(
    rows: pd.DataFrame,
    *,
    target: str,
    prediction_type: str | None = None,
    model_name: str | None = None,
) -> pd.Series | None:
    if rows.empty or "target" not in rows.columns:
        return None
    filtered = rows[rows["target"].astype(str).eq(target)].copy()
    if prediction_type is not None and "prediction_type" in filtered.columns:
        filtered = filtered[filtered["prediction_type"].fillna("live").astype(str).eq(prediction_type)].copy()
    if model_name is not None and "model_name" in filtered.columns:
        filtered = filtered[filtered["model_name"].astype(str).eq(model_name)].copy()
    filtered = _sort_prediction_history_rows(filtered)
    if filtered.empty:
        return None
    return filtered.iloc[0]


def _prediction_history_label(row: pd.Series | None) -> str:
    if row is None:
        return "-"
    return _prediction_class_name(row.get("predicted_class"), row.get("target"))


def _prediction_history_source_count(rows: pd.DataFrame, source: str) -> int:
    if rows.empty or "prediction_type" not in rows.columns:
        return 0
    return int(rows["prediction_type"].fillna("live").astype(str).eq(source).sum())


def _latest_prediction_saved_at(rows: pd.DataFrame):
    if rows.empty or "created_at" not in rows.columns:
        return None
    created = pd.to_datetime(rows["created_at"], errors="coerce").dropna()
    if created.empty:
        return None
    return created.max()


def _prediction_history_health_note(operations_snapshot: dict[str, object]) -> str:
    overall = str(operations_snapshot.get("overall_status") or "").upper()
    daily = str(operations_snapshot.get("daily_status") or "").upper()
    if overall == "WARN" and daily == "OK":
        return (
            "Health is WARN, but latest daily_predict is OK. "
            "If warnings are TOPIX fallback and historical live prediction gaps only, operation can continue."
        )
    if overall == "FAIL" or daily == "FAIL":
        return "Health or daily_predict is failing. Check Operations before relying on saved prediction history."
    return ""


def _render_prediction_history_operations(
    operations: dict[str, object],
    ticker: str,
    missing_diagnostics: pd.DataFrame | None = None,
) -> None:
    st.subheader("Prediction Operations")
    missing_dates = operations.get("missing_prediction_dates", []) or []
    backfilled_missing = operations.get("backfilled_missing_dates", []) or []
    backfillable_missing = operations.get("backfillable_missing_dates", []) or []
    pre_start_count = safe_int(
        operations.get(
            "pre_start_excluded_count",
            operations.get("ignored_pre_start_missing_count"),
        )
    ) or 0
    start_date = _format_date(operations.get("live_prediction_start_date"))
    latest_prediction_date = _format_date(operations.get("latest_prediction_date"))
    latest_saved_at = _format_datetime(operations.get("latest_saved_at"))
    next_confirmable = _format_date(operations.get("next_confirmable_prediction_date"))
    known_missed_dates = _known_missed_count_from_diagnostics(missing_diagnostics)

    cols = st.columns(6)
    cols[0].metric("Latest feature date", latest_prediction_date)
    cols[1].metric("Latest saved at", latest_saved_at)
    cols[2].metric("Post-start missing dates", _format_number(len(missing_dates), digits=0))
    cols[3].metric("Known missed dates", _format_number(known_missed_dates, digits=0))
    cols[4].metric("Backfill available", _format_number(len(backfillable_missing), digits=0))
    cols[5].metric("Next confirmable date", next_confirmable)

    st.caption(
        f"live_prediction_start_date is {start_date}. Feature dates before this start date are "
        "outside the live-operation window and are not counted as missing predictions."
    )
    if pre_start_count:
        st.caption(f"Pre-operation feature dates excluded from missing checks: {pre_start_count}.")
    st.caption(
        "Post-start missing live feature dates are trading dates where the app does not have a "
        "real live row using that feature date. Backfill can create pseudo predictions for research, "
        "but it does not turn those feature dates into real live predictions."
    )
    if backfilled_missing:
        st.caption(
            f"{len(backfilled_missing)} missing live feature dates already have backfill rows for "
            "history inspection."
        )

    if missing_dates:
        with st.expander("Post-start missing live feature dates"):
            backfilled_missing_set = set(backfilled_missing)
            display = pd.DataFrame(
                {
                    "missing_feature_date": [
                        pd.Timestamp(date).strftime("%Y-%m-%d") for date in missing_dates
                    ],
                    "backfill_status": [
                        "already backfilled" if date in backfilled_missing_set else "can attempt"
                        for date in missing_dates
                    ],
                }
            )
            st.caption("Expected feature dates use the local feature dataset's trading dates.")
            st.dataframe(display, use_container_width=True, hide_index=True)
            details = _known_missed_details_from_diagnostics(missing_diagnostics)
            if not details.empty:
                st.caption("Known missed details")
                st.dataframe(details, use_container_width=True, hide_index=True)
        _render_missing_live_date_diagnostics(missing_diagnostics)
        first_backfillable = operations.get("first_backfillable_missing_date")
        if first_backfillable is not None:
            start_date = pd.Timestamp(first_backfillable).strftime("%Y-%m-%d")
            st.caption(
                "Backfill is for historical inspection only. It trains each row using prior rows, "
                "but it is not a live operational prediction and should not be mixed with live results."
            )
            st.code(
                f"python main.py backfill-predictions --ticker {ticker} --start {start_date}",
                language="powershell",
            )


def _known_missed_count_from_diagnostics(diagnostics: pd.DataFrame | None) -> int:
    if diagnostics is None or diagnostics.empty or "known_missed" not in diagnostics.columns:
        return 0
    return int(diagnostics["known_missed"].fillna(False).astype(bool).sum())


def _known_missed_details_from_diagnostics(diagnostics: pd.DataFrame | None) -> pd.DataFrame:
    if diagnostics is None or diagnostics.empty or "known_missed" not in diagnostics.columns:
        return pd.DataFrame()
    known = diagnostics[diagnostics["known_missed"].fillna(False).astype(bool)].copy()
    if known.empty:
        return pd.DataFrame()
    rows = []
    for _, row in known.iterrows():
        rows.append(
            {
                "Feature date": _format_date(row.get("missing_date")),
                "Reason code": row.get("known_reason_code") or row.get("inferred_reason") or "-",
                "Reason": row.get("known_reason") or "-",
                "Action": row.get("known_action") or "-",
                "Allow backfill": _format_enabled(row.get("known_allow_backfill")),
                "Note": row.get("recommended_action") or "-",
            }
        )
    return pd.DataFrame(rows)


def _render_missing_live_date_diagnostics(diagnostics: pd.DataFrame | None) -> None:
    st.subheader("Missing Live Feature Date Diagnostics")
    if diagnostics is None or diagnostics.empty:
        st.caption("No missing live feature dates to diagnose.")
        return
    display = diagnostics.copy()
    for column in ("missing_date", "previous_live_prediction_date", "next_live_prediction_date"):
        if column in display.columns:
            display[column] = pd.to_datetime(display[column], errors="coerce").dt.strftime("%Y-%m-%d")
    if "previous_live_saved_at" in display.columns:
        display["previous_live_saved_at"] = pd.to_datetime(
            display["previous_live_saved_at"],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    display = display.rename(
        columns={
            "missing_date": "Missing feature date",
            "in_feature_data": "Feature row exists",
            "any_prediction_exists": "Any prediction exists",
            "binary_live_exists": "Binary live exists",
            "multiclass_live_exists": "Multiclass live exists",
            "backfill_exists": "Backfill exists",
            "source_breakdown": "Source breakdown",
            "previous_live_prediction_date": "Previous live feature date",
            "previous_live_saved_at": "Previous live saved at",
            "next_live_prediction_date": "Next live feature date",
            "inferred_reason": "Inferred reason",
            "known_missed": "Known missed",
            "known_reason_code": "Known reason code",
            "known_reason": "Known reason",
            "known_action": "Known action",
            "known_allow_backfill": "Known backfill allowed",
            "recommended_action": "Recommended action",
        }
    )
    st.caption(
        "Diagnostics separate real live gaps from backfill rows and from runs that saved an older "
        "feature date because newer market data was not yet available."
    )
    summary_rows = diagnostics[
        [
            "missing_date",
            "inferred_reason",
            "known_missed",
            "known_reason_code",
            "known_action",
            "known_allow_backfill",
            "recommended_action",
        ]
    ].copy()
    for _, row in summary_rows.iterrows():
        known_status = "Known missed" if bool(row.get("known_missed")) else "Action required"
        reason_code = row.get("known_reason_code") or row.get("inferred_reason")
        action = row.get("known_action") or row.get("recommended_action")
        backfill_text = "yes" if bool(row.get("known_allow_backfill")) else "no"
        st.info(
            f"{_format_date(row.get('missing_date'))}: Status: {known_status}. "
            f"Reason: {reason_code}. Action: {action}. Backfill allowed: {backfill_text}."
        )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_daily_predict_log_summary() -> None:
    log_path = PROJECT_ROOT / "logs" / "daily_predict.log"
    summary = _daily_predict_log_summary(log_path)
    cols = st.columns(2)
    cols[0].metric("Last daily job success", summary.get("last_success") or "-")
    cols[1].metric("Last daily job event", summary.get("last_event") or "-")
    if summary.get("last_error"):
        st.warning(f"Last daily job error: {summary['last_error']}")
    st.caption(f"Daily log: {log_path}")
    st.caption(
        "If old project paths still appear in this log, rotate or clear logs/daily_predict.log "
        "after re-registering the Windows scheduled task."
    )


def _daily_predict_log_summary(log_path: Path) -> dict[str, str | None]:
    if not log_path.exists():
        return {"last_success": None, "last_event": "log not found", "last_error": None}
    try:
        lines = log_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return {"last_success": None, "last_event": f"log read failed: {error}", "last_error": None}

    latest_run_lines = _latest_daily_predict_run_lines(lines)
    last_success = _last_log_line_containing(
        latest_run_lines,
        "daily_predict finished successfully",
    )
    last_error = _last_log_line_containing(latest_run_lines, "daily_predict failed")
    last_event = next((line for line in reversed(latest_run_lines) if line.strip()), None)
    return {
        "last_success": last_success,
        "last_event": last_event,
        "last_error": last_error,
    }


def _latest_daily_predict_run_lines(lines: list[str]) -> list[str]:
    for index in range(len(lines) - 1, -1, -1):
        if "daily_predict started" in lines[index]:
            return lines[index:]
    return lines


def _last_log_line_containing(lines: list[str], pattern: str) -> str | None:
    return next((line for line in reversed(lines) if pattern in line), None)


def _render_model_evaluation_overview(ticker: str, data: pd.DataFrame, *, section: str) -> None:
    binary_bundle = _load_binary_metrics(ticker)
    binary_metrics = binary_bundle.get("metrics", {}) if isinstance(binary_bundle, dict) else {}
    multiclass_bundle = _load_multiclass_metrics(ticker)
    multiclass_metrics = (
        multiclass_bundle.get("metrics", {}) if isinstance(multiclass_bundle, dict) else {}
    )

    latest_evaluation_date = _latest_model_metrics_timestamp(ticker)
    latest_data_date = _latest_data_date(data)
    sample_size = _model_evaluation_sample_size(binary_metrics, multiclass_metrics)
    best_model = _model_compare_best_model_from_session(ticker)

    st.subheader("Model Evaluation Overview")
    st.caption(
        f"{section} uses existing saved metrics and in-session results. "
        "No model training or scoring logic is changed by this view."
    )
    cols = st.columns(5)
    cols[0].metric("Latest evaluation", latest_evaluation_date or "-")
    cols[1].metric("Latest data", latest_data_date or "-")
    cols[2].metric("Binary F1", _format_percent(binary_metrics.get("f1")))
    cols[3].metric("Multiclass Macro F1", _format_percent(multiclass_metrics.get("macro_f1")))
    cols[4].metric("Sample size", _format_number(sample_size, digits=0))

    model_cols = st.columns(4)
    model_cols[0].metric("Binary Accuracy", _format_percent(binary_metrics.get("accuracy")))
    model_cols[1].metric("Binary AUC", _format_percent(binary_metrics.get("auc")))
    model_cols[2].metric(
        "Crash / Surge Recall",
        (
            f"{_format_percent(_multiclass_class_metric(multiclass_metrics, 'crash', 'recall'))} / "
            f"{_format_percent(_multiclass_class_metric(multiclass_metrics, 'surge', 'recall'))}"
        ),
    )
    model_cols[3].metric("Best model in session", best_model or "-")

    caveats = _model_evaluation_caveats(binary_metrics, multiclass_metrics)
    if caveats:
        st.caption(" / ".join(caveats))

    with st.expander("How to read model evaluation"):
        st.write("- Accuracy alone is not enough; compare it with F1, AUC, and return simulation.")
        st.write("- F1 balances precision and recall, especially when up/down labels are uneven.")
        st.write("- AUC shows ranking quality across thresholds, not one fixed decision rule.")
        st.write("- Macro F1 treats every multiclass label equally, so weak crash/surge detection is visible.")
        st.write("- Surge/crash recall relates to missed rare-event warnings; higher recall can also mean more false alarms.")
        st.write("- Ablation is a feature-group sensitivity check, not proof of causality.")
        st.write("- Backtest is historical simulation and does not guarantee future performance.")


def _latest_model_metrics_timestamp(ticker: str) -> str | None:
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    if not metrics_dir.exists():
        return None
    metric_paths = list(metrics_dir.glob(f"*{ticker}.json"))
    if not metric_paths:
        return None
    latest_mtime = max(path.stat().st_mtime for path in metric_paths)
    return pd.Timestamp.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M")


def _latest_data_date(data: pd.DataFrame) -> str | None:
    if data.empty or "date" not in data.columns:
        return None
    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().strftime("%Y-%m-%d")


def _model_evaluation_sample_size(
    binary_metrics: dict[str, object],
    multiclass_metrics: dict[str, object],
) -> int | None:
    values = [
        safe_int(binary_metrics.get("rows")),
        safe_int(multiclass_metrics.get("rows")),
    ]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _model_compare_best_model_from_session(ticker: str) -> str | None:
    result = st.session_state.get(f"model_compare_{ticker}")
    score_table = getattr(result, "score_table", pd.DataFrame())
    best = _best_model_row(score_table, "f1")
    if best is None:
        return None
    return str(best.get("model_name"))


def _model_evaluation_caveats(
    binary_metrics: dict[str, object],
    multiclass_metrics: dict[str, object],
) -> list[str]:
    caveats: list[str] = []
    if not binary_metrics:
        caveats.append("Binary metrics are not available yet")
    if not multiclass_metrics:
        caveats.append("Multiclass metrics are not available yet")
    crash_support = safe_int(_multiclass_class_metric(multiclass_metrics, "crash", "support"))
    surge_support = safe_int(_multiclass_class_metric(multiclass_metrics, "surge", "support"))
    if crash_support is not None and crash_support < 30:
        caveats.append("Crash samples are limited")
    if surge_support is not None and surge_support < 30:
        caveats.append("Surge samples are limited")
    return caveats


def _render_backtest_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    walk_forward = _walk_forward_backtest_module()
    _render_model_evaluation_overview(ticker, data, section="Backtest")
    st.caption(
        "Walk-forward backtest retrains on past data only, then predicts the next monthly test window."
    )
    st.caption(
        "This is separate from Prediction History, which evaluates only predictions that were actually saved day by day."
    )

    if data.empty or "date" not in data.columns:
        st.info("Backtest needs a feature dataset with dates.")
        return

    date_series = pd.to_datetime(data["date"], errors="coerce").dropna()
    if date_series.empty:
        st.info("Backtest needs valid dates.")
        return

    min_date = date_series.min().date()
    max_date = date_series.max().date()
    default_start = max(pd.Timestamp(min_date) + pd.DateOffset(years=3), date_series.min()).date()
    default_start = min(default_start, max_date)

    controls = st.columns(5)
    start_date = controls[0].date_input(
        "Start",
        value=default_start,
        min_value=min_date,
        max_value=max_date,
        key="backtest_start_date",
    )
    end_date = controls[1].date_input(
        "End",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key="backtest_end_date",
    )
    threshold = controls[2].slider(
        "Buy threshold",
        min_value=0.50,
        max_value=0.90,
        value=0.55,
        step=0.01,
        key="backtest_buy_threshold",
    )
    train_years = controls[3].number_input(
        "Train years",
        min_value=1,
        max_value=8,
        value=3,
        step=1,
        key="backtest_train_years",
    )
    test_step = controls[4].selectbox(
        "Test step",
        options=[1, 2, 3],
        index=0,
        format_func=lambda value: f"{value} month",
        key="backtest_test_step",
    )
    if start_date > end_date:
        st.warning("Start date must be before end date.")
        return

    result_key = f"walk_forward_backtest_{ticker}"
    config_key = f"walk_forward_backtest_config_{ticker}"
    if st.button("Run walk-forward backtest", type="primary", key="backtest_run_button"):
        anomaly_enriched, _, _, _ = _anomaly_context(data)
        config = walk_forward.WalkForwardConfig(
            ticker=ticker,
            train_years=int(train_years),
            test_step_months=int(test_step),
            probability_threshold=float(threshold),
            start_date=str(start_date),
            end_date=str(end_date),
        )
        with st.spinner("Running walk-forward backtest..."):
            try:
                result = walk_forward.run_walk_forward_backtest(anomaly_enriched, config=config)
            except Exception as error:
                st.error(f"Backtest failed: {error}")
                return
        st.session_state[result_key] = result
        st.session_state[config_key] = config

    result = st.session_state.get(result_key)
    config = st.session_state.get(config_key)
    if result is None:
        st.info("Set the period and threshold, then run the walk-forward backtest.")
        return
    if result.predictions.empty:
        st.warning("Backtest produced no prediction rows. Try a later start date or a shorter train window.")
        return

    with st.expander("Current Model Summary", expanded=True):
        _render_backtest_metrics(result.metrics)
        st.plotly_chart(_build_backtest_equity_chart(result.equity_curve), use_container_width=True)

    with st.expander("Backtest Details", expanded=True):
        st.caption("Monthly performance and regime/risk slices for the current walk-forward run.")
        st.subheader("Monthly Performance")
        st.dataframe(_performance_display_frame(result.monthly_performance), use_container_width=True, hide_index=True)

        summary_cols = st.columns(3)
        with summary_cols[0]:
            st.subheader("Regime Performance")
            st.dataframe(_performance_display_frame(result.regime_performance), use_container_width=True, hide_index=True)
        with summary_cols[1]:
            st.subheader("Surge Risk")
            st.dataframe(
                _performance_display_frame(result.surge_risk_performance),
                use_container_width=True,
                hide_index=True,
            )
        with summary_cols[2]:
            st.subheader("Crash Risk")
            st.dataframe(
                _performance_display_frame(result.crash_risk_performance),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Raw metrics / details"):
        st.caption("Individual backtest prediction rows. Use this for audits, not as the first reading surface.")
        st.dataframe(_backtest_predictions_display_frame(result.predictions), use_container_width=True, hide_index=True)

    if st.button("Save backtest CSV", key="backtest_save_csv_button"):
        output_dir = PROJECT_ROOT / "reports" / "backtests"
        threshold_value = getattr(config, "probability_threshold", threshold)
        paths = walk_forward.write_backtest_csvs(result, output_dir, ticker=ticker, threshold=threshold_value)
        st.success("Backtest CSV files saved.")
        st.dataframe(
            pd.DataFrame(
                [{"artifact": key, "path": str(path)} for key, path in paths.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )


def _render_backtest_metrics(metrics: dict[str, object]) -> None:
    st.subheader("Backtest Metrics")
    metric_cols = st.columns(5)
    metric_cols[0].metric("Rows", _format_number(metrics.get("rows"), digits=0))
    metric_cols[1].metric("Accuracy", _format_percent(metrics.get("accuracy")))
    metric_cols[2].metric("F1", _format_percent(metrics.get("f1")))
    metric_cols[3].metric("AUC", _format_percent(metrics.get("auc")))
    metric_cols[4].metric("Precision", _format_percent(metrics.get("precision")))

    result_cols = st.columns(5)
    result_cols[0].metric("Recall", _format_percent(metrics.get("recall")))
    result_cols[1].metric("Up accuracy", _format_percent(metrics.get("up_prediction_accuracy")))
    result_cols[2].metric("Down accuracy", _format_percent(metrics.get("down_prediction_accuracy")))
    result_cols[3].metric("Trades", _format_number(metrics.get("trade_count"), digits=0))
    result_cols[4].metric("Win rate", _format_percent(metrics.get("win_rate")))

    return_cols = st.columns(3)
    return_cols[0].metric("Strategy return", _format_percent(metrics.get("cumulative_return")))
    return_cols[1].metric("Buy & Hold", _format_percent(metrics.get("buy_hold_return")))
    return_cols[2].metric("Max drawdown", _format_percent(metrics.get("max_drawdown")))

    confusion = metrics.get("confusion_matrix", {}) or {}
    confusion_frame = pd.DataFrame(
        [
            {"Actual": "Down", "Predicted Down": confusion.get("tn", 0), "Predicted Up": confusion.get("fp", 0)},
            {"Actual": "Up", "Predicted Down": confusion.get("fn", 0), "Predicted Up": confusion.get("tp", 0)},
        ]
    )
    st.caption("Confusion Matrix")
    st.dataframe(confusion_frame, use_container_width=True, hide_index=True)


def _build_backtest_equity_chart(equity: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if equity.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["strategy_cumulative_return"],
            mode="lines",
            name="Strategy",
            line=dict(color="#3a8f6b", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["buy_hold_cumulative_return"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#6c8dbf", width=2),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        title="Cumulative Return",
        height=360,
        yaxis_tickformat=".1%",
        margin=dict(l=20, r=20, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _performance_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in ("strategy_return", "buy_hold_return", "win_rate", "average_trade_return"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "month": "Month",
            "regime": "Regime",
            "surge_risk_level": "Surge Risk",
            "crash_risk_level": "Crash Risk",
            "rows": "Rows",
            "trade_count": "Trades",
            "strategy_return": "Strategy Return",
            "buy_hold_return": "Buy & Hold",
            "win_rate": "Win Rate",
            "average_trade_return": "Avg Trade Return",
        }
    )


def _backtest_predictions_display_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    display_columns = [
        "date",
        "prediction_direction",
        "actual_direction",
        "probability_up",
        "probability_down",
        "trade",
        "actual_return",
        "strategy_return",
        "strategy_cumulative_return",
        "buy_hold_cumulative_return",
        "regime",
        "surge_risk_level",
        "crash_risk_level",
        "train_start_date",
        "train_end_date",
    ]
    display = predictions[[column for column in display_columns if column in predictions.columns]].copy()
    for column in ("date", "train_start_date", "train_end_date"):
        if column in display.columns:
            display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d")
    for column in (
        "probability_up",
        "probability_down",
        "actual_return",
        "strategy_return",
        "strategy_cumulative_return",
        "buy_hold_cumulative_return",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "trade" in display.columns:
        display["trade"] = display["trade"].map(lambda value: "Buy" if bool(value) else "No trade")
    return display.rename(
        columns={
            "date": "Date",
            "prediction_direction": "Prediction",
            "actual_direction": "Actual",
            "probability_up": "Up Probability",
            "probability_down": "Down Probability",
            "trade": "Action",
            "actual_return": "Actual Return",
            "strategy_return": "Strategy Return",
            "strategy_cumulative_return": "Strategy Cum Return",
            "buy_hold_cumulative_return": "Buy & Hold Cum Return",
            "regime": "Regime",
            "surge_risk_level": "Surge Risk",
            "crash_risk_level": "Crash Risk",
            "train_start_date": "Train Start",
            "train_end_date": "Train End",
        }
    )


def _render_evaluation_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    evaluation = _evaluation_module()
    st.caption(
        "Condition-level evaluation for Backtest and Prediction History. Use this to find where the model is strong or weak."
    )

    _render_multiclass_evaluation_section(ticker, data, settings)
    _render_model_decision_notes_section(ticker)
    _render_risk_validation_section(ticker, data, settings)
    _render_news_impact_validation_section(ticker, data, settings)
    _render_today_signal_validation_section(ticker, data, settings)
    _render_offline_model_experiments_section(ticker)
    _render_binary_up_detection_experiments_section(ticker)
    _render_binary_shadow_evaluation_section(ticker)

    st.subheader("Binary Evaluation")
    source = st.radio(
        "Evaluation source",
        options=["Backtest", "Prediction History"],
        horizontal=True,
        key="evaluation_source_radio",
    )
    if source == "Backtest":
        result = st.session_state.get(f"walk_forward_backtest_{ticker}")
        if result is None or result.predictions.empty:
            st.info("Run a walk-forward backtest first, then return here.")
            return
        eval_frame = evaluation.prepare_backtest_evaluation_frame(result.predictions)
    else:
        predictions = _read_prediction_history(
            settings.paths.database_path,
            ticker=ticker,
            target="next_close_binary",
            model_name="lightgbm_binary",
        )
        if predictions.empty:
            st.info("No saved Prediction History is available yet.")
            return
        anomaly_data, _, _, _ = _anomaly_context(data)
        history, _metrics = evaluate_prediction_history(predictions, anomaly_data)
        eval_frame = evaluation.prepare_prediction_history_evaluation_frame(history, anomaly_data)

    if eval_frame.empty:
        st.warning("No confirmed evaluation rows are available for this source.")
        return

    overall = evaluation.summarize_rows(eval_frame)
    summaries = evaluation.build_evaluation_summaries(eval_frame)
    _render_evaluation_overview(overall)

    st.subheader("Model Strengths / Weak Conditions")
    for message in evaluation.build_weakness_summary(summaries, overall, min_count=5):
        st.write(f"- {message}")

    st.subheader("Condition Breakdown")
    slice_options = {
        "Year": "year",
        "Month": "month",
        "Regime": "regime",
        "Surge Risk": "surge_risk_level",
        "Crash Risk": "crash_risk_level",
        "Anomaly Type": "anomaly_type",
        "Narrative": "dominant_narrative",
        "Earnings Event": "earnings_event_state",
        "News Sentiment": "news_sentiment_bucket",
        "Confidence": "confidence_bucket",
    }
    selected_label = st.selectbox(
        "Slice",
        options=list(slice_options),
        index=2,
        key="evaluation_slice_select",
    )
    selected_key = slice_options[selected_label]
    selected_summary = summaries.get(selected_key, pd.DataFrame())
    if selected_summary.empty:
        st.info(f"No rows available for {selected_label}.")
    else:
        st.plotly_chart(
            _build_evaluation_bar_chart(selected_summary, selected_key, selected_label),
            use_container_width=True,
        )
        st.dataframe(_evaluation_summary_display_frame(selected_summary), use_container_width=True, hide_index=True)

    st.subheader("Monthly Accuracy / Return Heatmaps")
    heat_cols = st.columns(2)
    with heat_cols[0]:
        st.plotly_chart(
            _build_monthly_heatmap(eval_frame, value_column="is_correct", title="Monthly Accuracy"),
            use_container_width=True,
        )
    with heat_cols[1]:
        st.plotly_chart(
            _build_monthly_heatmap(eval_frame, value_column="strategy_return", title="Monthly Strategy Return"),
            use_container_width=True,
        )

    confidence_summary = summaries.get("confidence_bucket", pd.DataFrame())
    if not confidence_summary.empty:
        st.subheader("Confidence Accuracy")
        st.plotly_chart(
            _build_confidence_evaluation_chart(confidence_summary),
            use_container_width=True,
        )

    _render_accuracy_diagnostics_section(
        ticker=ticker,
        data=data,
        settings=settings,
        source=source,
        backtest_eval_frame=eval_frame if source == "Backtest" else None,
    )


def _render_model_decision_notes_section(ticker: str) -> None:
    with st.expander("Model Decision Notes", expanded=False):
        st.caption(
            "Current production judgement snapshot. This is a record of how to read the existing "
            "models and operations; it does not change production models or prediction history."
        )
        report = _latest_model_decision_notes_report()
        if report is None:
            st.info("No model decision notes report is available yet.")
            st.caption("Expected report path:")
            st.code(
                f"reports/evaluation/model_decision_notes_{pd.Timestamp.today().strftime('%Y-%m-%d')}.md",
                language="text",
            )
            st.caption(f"Ticker context: {ticker}")
            return
        st.caption(f"Latest report: {report.name}")
        try:
            st.markdown(report.read_text(encoding="utf-8"))
        except Exception as error:
            st.warning(f"Could not read model decision notes: {error}")


def _latest_model_decision_notes_report() -> Path | None:
    report_dir = PROJECT_ROOT / "reports" / "evaluation"
    files = sorted(report_dir.glob("model_decision_notes_*.md"), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def _render_offline_model_experiments_section(ticker: str) -> None:
    with st.expander("Offline Model Experiments", expanded=False):
        st.caption(
            "Offline experiments compare binary model candidates on historical walk-forward rows. "
            "They do not change production models, daily_predict, SQLite prediction history, or feature generation."
        )
        report = _latest_offline_model_experiment_report()
        if report is None:
            st.info("No offline experiment report is available yet.")
            st.code(
                f"python main.py offline-model-experiments --ticker {ticker}",
                language="powershell",
            )
            return
        csv_path, json_path = report
        st.caption(f"Latest report: {csv_path.name}")
        try:
            summary = pd.read_csv(csv_path)
        except Exception as error:
            st.warning(f"Could not read offline experiment report: {error}")
            return
        conclusions = _offline_model_experiment_conclusions(json_path)
        if conclusions:
            st.caption("Conclusions")
            for conclusion in conclusions:
                st.write(f"- {conclusion}")
        if summary.empty:
            st.info("Offline experiment report is empty.")
            return
        display = _offline_model_experiment_display_frame(summary)
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Use this as a research comparison only. A production model change should be a separate task "
            "after reviewing sample size, leakage risk, and baseline deltas."
        )


def _latest_offline_model_experiment_report() -> tuple[Path, Path | None] | None:
    report_dir = PROJECT_ROOT / "reports" / "evaluation"
    files = sorted(report_dir.glob("offline_model_experiments_*.csv"), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    csv_path = files[-1]
    json_path = csv_path.with_suffix(".json")
    return csv_path, json_path if json_path.exists() else None


def _offline_model_experiment_conclusions(json_path: Path | None) -> list[str]:
    if json_path is None:
        return []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    conclusions = payload.get("conclusions", [])
    return [str(item) for item in conclusions if str(item).strip()]


def _offline_model_experiment_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "candidate_name",
        "sample_count",
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "down_precision",
        "predicted_up_ratio",
        "predicted_down_ratio",
        "model_minus_always_up",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
        "notes",
        "caveats",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    percent_columns = [
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "down_precision",
        "predicted_up_ratio",
        "predicted_down_ratio",
        "model_minus_always_up",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
    ]
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "candidate_name": "Candidate",
            "sample_count": "Rows",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "up_recall": "Up Recall",
            "down_recall": "Down Recall",
            "up_precision": "Up Precision",
            "down_precision": "Down Precision",
            "predicted_up_ratio": "Predicted Up",
            "predicted_down_ratio": "Predicted Down",
            "model_minus_always_up": "Model - Always Up",
            "model_minus_always_down": "Model - Always Down",
            "model_minus_previous_day_direction": "Model - Previous Day",
            "model_minus_moving_average_trend_only": "Model - MA Trend",
            "notes": "Notes",
            "caveats": "Caveats",
        }
    )


def _render_binary_up_detection_experiments_section(ticker: str) -> None:
    with st.expander("Binary Up Detection Experiments", expanded=False):
        st.caption(
            "Offline walk-forward experiments focused on improving up detection. "
            "They are read-only reports and do not change production models, daily_predict, SQLite history, "
            "or feature generation."
        )
        report = _latest_binary_up_detection_report()
        if report is None:
            st.info("No binary up-detection experiment report is available yet.")
            st.code(
                f"python main.py binary-up-detection-experiments --ticker {ticker}",
                language="powershell",
            )
            return
        csv_path, json_path, md_path = report
        st.caption(f"Latest report: {csv_path.name}")
        try:
            summary = pd.read_csv(csv_path)
        except Exception as error:
            st.warning(f"Could not read binary up-detection report: {error}")
            return

        conclusions = _binary_up_detection_conclusions(json_path)
        if conclusions:
            st.caption("Conclusions")
            for conclusion in conclusions:
                st.write(f"- {conclusion}")
        if md_path is not None:
            with st.expander("Report notes", expanded=False):
                try:
                    st.markdown(md_path.read_text(encoding="utf-8"))
                except Exception as error:
                    st.warning(f"Could not read markdown report: {error}")
        if summary.empty:
            st.info("Binary up-detection report is empty.")
            return
        historical = summary[summary.get("period", pd.Series(dtype=str)).astype(str).eq("Historical all")].copy()
        if historical.empty:
            historical = summary.copy()
        st.dataframe(
            _binary_up_detection_display_frame(historical),
            use_container_width=True,
            hide_index=True,
        )
        live = summary[summary.get("period", pd.Series(dtype=str)).astype(str).eq("Live only")].copy()
        if not live.empty:
            st.caption("Live saved prediction reference")
            st.dataframe(
                _binary_up_detection_display_frame(live),
                use_container_width=True,
                hide_index=True,
            )
        st.caption(
            "Use this as an offline research view. Any production model or threshold change should be a separate task."
        )


def _latest_binary_up_detection_report() -> tuple[Path, Path | None, Path | None] | None:
    report_dir = PROJECT_ROOT / "reports" / "evaluation"
    files = sorted(report_dir.glob("binary_up_detection_experiments_*.csv"), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    csv_path = files[-1]
    json_path = csv_path.with_suffix(".json")
    md_path = csv_path.with_suffix(".md")
    return (
        csv_path,
        json_path if json_path.exists() else None,
        md_path if md_path.exists() else None,
    )


def _binary_up_detection_conclusions(json_path: Path | None) -> list[str]:
    if json_path is None:
        return []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    conclusions = payload.get("conclusions", [])
    return [str(item) for item in conclusions if str(item).strip()]


def _binary_up_detection_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_name",
        "period",
        "sample_count",
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "down_precision",
        "predicted_up_ratio",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
        "constraint_passed",
        "notes",
        "caveats",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    percent_columns = [
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "down_precision",
        "predicted_up_ratio",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
    ]
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "candidate_name": "Candidate",
            "period": "Period",
            "sample_count": "Rows",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "up_recall": "Up Recall",
            "down_recall": "Down Recall",
            "up_precision": "Up Precision",
            "down_precision": "Down Precision",
            "predicted_up_ratio": "Predicted Up",
            "model_minus_always_down": "Model - Always Down",
            "model_minus_previous_day_direction": "Model - Previous Day",
            "model_minus_moving_average_trend_only": "Model - MA Trend",
            "constraint_passed": "Constraint Passed",
            "notes": "Notes",
            "caveats": "Caveats",
        }
    )


def _render_binary_shadow_evaluation_section(ticker: str) -> None:
    with st.expander("Binary Shadow Evaluation", expanded=False):
        st.caption(
            "Manual shadow comparison for production_current and class_weight_balanced. "
            "Shadow rows are saved only under reports/evaluation and are not mixed into SQLite Prediction History."
        )
        csv_path = PROJECT_ROOT / "reports" / "evaluation" / "binary_shadow_predictions.csv"
        latest_md = _latest_binary_shadow_summary_report()
        if not csv_path.exists():
            st.info("No binary shadow predictions are available yet.")
            st.code(
                f"python main.py binary-shadow-predict --ticker {ticker}\n"
                f"python main.py binary-shadow-evaluate --ticker {ticker}",
                language="powershell",
            )
            return
        try:
            rows = pd.read_csv(csv_path)
        except Exception as error:
            st.warning(f"Could not read binary shadow predictions: {error}")
            return
        if rows.empty:
            st.info("Binary shadow prediction file is empty.")
            return
        st.caption(f"Prediction file: {csv_path.name}")
        if latest_md is not None:
            st.caption(f"Latest summary: {latest_md.name}")
        summary = _binary_shadow_summary_display(rows)
        st.dataframe(summary, use_container_width=True, hide_index=True)
        latest = rows.sort_values("feature_date").tail(10).copy()
        st.caption("Latest Shadow Predictions")
        st.dataframe(_binary_shadow_prediction_display(latest), use_container_width=True, hide_index=True)
        pending = rows[rows.get("result", pd.Series(dtype=str)).astype(str).eq("pending")].copy()
        if not pending.empty:
            st.caption("Pending")
            st.dataframe(_binary_shadow_prediction_display(pending.tail(10)), use_container_width=True, hide_index=True)
        if latest_md is not None:
            with st.expander("Markdown summary", expanded=False):
                try:
                    st.markdown(latest_md.read_text(encoding="utf-8"))
                except Exception as error:
                    st.warning(f"Could not read binary shadow markdown summary: {error}")
        st.caption(
            "Run manually for now. Do not schedule this until enough confirmed shadow rows justify automation."
        )


def _latest_binary_shadow_summary_report() -> Path | None:
    report_dir = PROJECT_ROOT / "reports" / "evaluation"
    files = sorted(report_dir.glob("binary_shadow_summary_*.md"), key=lambda path: path.stat().st_mtime)
    if not files:
        return None
    return files[-1]


def _binary_shadow_summary_display(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    result = []
    for candidate, group in rows.groupby("candidate_name", dropna=False):
        confirmed = group[group.get("result", pd.Series(dtype=str)).isin(["hit", "miss"])].copy()
        pending = group[group.get("result", pd.Series(dtype=str)).eq("pending")].copy()
        result.append(
            {
                "Candidate": candidate,
                "Rows": len(group),
                "Confirmed": len(confirmed),
                "Pending": len(pending),
                "Hit rate": _format_percent(confirmed["result"].eq("hit").mean()) if not confirmed.empty else "-",
                "Latest feature date": _format_date(pd.to_datetime(group.get("feature_date"), errors="coerce").max()),
            }
        )
    return pd.DataFrame(result)


def _binary_shadow_prediction_display(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_date",
        "target_close_date",
        "candidate_name",
        "predicted_direction",
        "confidence",
        "probability_up",
        "actual_direction",
        "actual_return",
        "result",
        "caveat",
    ]
    display = rows[[column for column in columns if column in rows.columns]].copy()
    for column in ("confidence", "probability_up", "actual_return"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "feature_date": "Feature date",
            "target_close_date": "Target close date",
            "candidate_name": "Candidate",
            "predicted_direction": "Prediction",
            "confidence": "Confidence",
            "probability_up": "Probability Up",
            "actual_direction": "Actual",
            "actual_return": "Actual Return",
            "result": "Result",
            "caveat": "Caveat",
        }
    )


def _render_accuracy_diagnostics_section(
    *,
    ticker: str,
    data: pd.DataFrame,
    settings,
    source: str,
    backtest_eval_frame: pd.DataFrame | None = None,
) -> None:
    diagnostics = _accuracy_diagnostics_module()
    st.subheader("Accuracy Diagnostics")
    st.caption(
        "Overall accuracy around 50% does not automatically mean the model is useless. "
        "Check whether it beats simple baselines, improves at higher confidence, works in specific regimes, "
        "or fails in particular market conditions before judging the average."
    )
    st.caption(
        "Saved prediction rows use the feature date. The target close date below is inferred as the next "
        "available trading day in local data; rows without that next close are excluded from confirmed metrics."
    )

    multiclass_frame = pd.DataFrame()
    historical_frame = pd.DataFrame()
    if source == "Prediction History":
        source_options = {
            "Live only": "live",
            "Backfill only": "backfill",
            "Live + Backfill": None,
        }
        selected_source_label = st.radio(
            "Accuracy diagnostics source",
            options=list(source_options),
            horizontal=True,
            key="accuracy_diagnostics_source_radio",
        )
        selected_prediction_type = source_options[selected_source_label]
        all_predictions = _read_all_prediction_history(settings.paths.database_path, ticker=ticker)
        anomaly_data, _, _, _ = _anomaly_context(_add_regime_context(data))
        binary_frame = diagnostics.build_binary_diagnostic_frame(
            all_predictions,
            anomaly_data,
            prediction_type=selected_prediction_type,
        )
        multiclass_frame = diagnostics.build_multiclass_diagnostic_frame(
            all_predictions,
            anomaly_data,
            prediction_type=selected_prediction_type,
        )
        backtest_result = st.session_state.get(f"walk_forward_backtest_{ticker}")
        if backtest_result is not None and not backtest_result.predictions.empty:
            historical_frame = diagnostics.build_walk_forward_binary_diagnostic_frame(
                backtest_result.predictions,
                anomaly_data,
            )
    else:
        anomaly_data, _, _, _ = _anomaly_context(_add_regime_context(data))
        binary_frame = (
            diagnostics.build_walk_forward_binary_diagnostic_frame(backtest_eval_frame, anomaly_data)
            if backtest_eval_frame is not None
            else pd.DataFrame()
        )
        historical_frame = binary_frame.copy()

    if binary_frame.empty and multiclass_frame.empty:
        st.info("No rows are available for accuracy diagnostics yet.")
        return

    target_summary = diagnostics.summarize_evaluation_targets(binary_frame, multiclass_frame)
    summary_cols = st.columns(6)
    summary_cols[0].metric("Binary evaluated", _format_number(target_summary.binary_confirmed_rows, digits=0))
    summary_cols[1].metric("Binary pending", _format_number(target_summary.binary_pending_rows, digits=0))
    summary_cols[2].metric("Multiclass evaluated", _format_number(target_summary.multiclass_confirmed_rows, digits=0))
    summary_cols[3].metric("Multiclass pending", _format_number(target_summary.multiclass_pending_rows, digits=0))
    summary_cols[4].metric("Live rows", _format_number(target_summary.live_rows, digits=0))
    summary_cols[5].metric("Backfill rows", _format_number(target_summary.backfill_rows, digits=0))
    st.caption(
        f"Evaluated period: {target_summary.evaluated_start or '-'} to "
        f"{target_summary.evaluated_end or '-'}. Binary metrics use confirmed next-close rows only."
    )

    with st.expander("Overall vs Baselines", expanded=True):
        baselines = diagnostics.baseline_accuracy_frame(binary_frame)
        if baselines.empty:
            st.info("Baseline comparison is not available for this source yet.")
        else:
            st.dataframe(_accuracy_baseline_display_frame(baselines), use_container_width=True, hide_index=True)
        source_breakdown = diagnostics.source_breakdown_frame(binary_frame)
        if not source_breakdown.empty:
            st.caption("Live / backfill breakdown")
            st.dataframe(
                _accuracy_source_breakdown_display_frame(source_breakdown),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Historical / Live Comparison", expanded=True):
        comparison_frames = {}
        if source == "Prediction History":
            comparison_frames[selected_source_label] = binary_frame
        else:
            comparison_frames["Historical walk-forward"] = historical_frame
            live_predictions = _read_all_prediction_history(settings.paths.database_path, ticker=ticker)
            live_frame = diagnostics.build_binary_diagnostic_frame(
                live_predictions,
                anomaly_data,
                prediction_type="live",
            )
            if not live_frame.empty:
                comparison_frames["Live saved"] = live_frame
        if source == "Prediction History" and not historical_frame.empty:
            comparison_frames["Historical walk-forward"] = historical_frame

        comparison = diagnostics.source_comparison_frame(comparison_frames)
        if comparison.empty:
            st.info("Historical comparison is not available yet. Run a walk-forward backtest first.")
        else:
            st.dataframe(_accuracy_summary_display_frame(comparison), use_container_width=True, hide_index=True)

        period = diagnostics.period_summary_frame(binary_frame)
        if not period.empty:
            st.caption("Period comparison for the currently selected diagnostics source")
            st.dataframe(_accuracy_summary_display_frame(period), use_container_width=True, hide_index=True)
        if source == "Prediction History" and historical_frame.empty:
            st.caption("Run a walk-forward backtest to compare saved live rows with historical rows.")

    with st.expander("Probability Distribution", expanded=False):
        distribution = diagnostics.confidence_distribution_frame(binary_frame)
        if distribution.empty:
            st.info("Probability distribution is not available yet.")
        else:
            st.dataframe(_accuracy_probability_display_frame(distribution), use_container_width=True, hide_index=True)

    with st.expander("Confidence Bucket Accuracy", expanded=True):
        confidence = diagnostics.confidence_threshold_frame(binary_frame)
        if confidence.empty:
            st.info("Confidence diagnostics are not available yet.")
        else:
            st.dataframe(_accuracy_confidence_display_frame(confidence), use_container_width=True, hide_index=True)

    with st.expander("Regime / Anomaly Accuracy", expanded=True):
        regime = diagnostics.regime_accuracy_frame(binary_frame)
        anomaly = diagnostics.anomaly_accuracy_frame(binary_frame)
        reg_cols = st.columns(2)
        with reg_cols[0]:
            st.caption("Regime")
            st.dataframe(_accuracy_context_display_frame(regime), use_container_width=True, hide_index=True)
        with reg_cols[1]:
            st.caption("Anomaly level")
            st.dataframe(_accuracy_context_display_frame(anomaly), use_container_width=True, hide_index=True)

    with st.expander("Class-wise Performance", expanded=True):
        class_cols = st.columns(2)
        with class_cols[0]:
            st.caption("Binary class performance")
            binary_class = diagnostics.binary_class_performance_frame(binary_frame)
            if binary_class.empty:
                st.info("Binary class-wise performance is not available.")
            else:
                st.dataframe(
                    _accuracy_binary_class_display_frame(binary_class),
                    use_container_width=True,
                    hide_index=True,
                )
        with class_cols[1]:
            st.caption("Multiclass precision / recall")
            multiclass_perf = diagnostics.multiclass_performance_frame(multiclass_frame)
            if multiclass_perf.empty:
                st.info("Multiclass saved-history diagnostics are not available for this source.")
            else:
                st.dataframe(
                    _accuracy_multiclass_display_frame(multiclass_perf),
                    use_container_width=True,
                    hide_index=True,
                )
        confusion = diagnostics.multiclass_confusion_matrix(multiclass_frame)
        if not confusion.empty and int(confusion.to_numpy().sum()) > 0:
            st.caption("Multiclass confusion matrix")
            st.dataframe(confusion, use_container_width=True)
        flat_summary = diagnostics.multiclass_flat_summary_frame(multiclass_frame)
        if not flat_summary.empty:
            st.caption("Multiclass flat bias")
            st.dataframe(_accuracy_flat_summary_display_frame(flat_summary), use_container_width=True, hide_index=True)

    with st.expander("Biggest Hits / Misses", expanded=True):
        hits_misses = diagnostics.biggest_hits_misses_frame(binary_frame)
        if hits_misses.empty:
            st.info("No confirmed hit/miss rows are available.")
        else:
            st.dataframe(_accuracy_hits_misses_display_frame(hits_misses), use_container_width=True, hide_index=True)


def _accuracy_baseline_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("model_accuracy", "baseline_accuracy", "model_minus_baseline"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "sample_count" in display.columns:
        display["sample_count"] = display["sample_count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "baseline": "Baseline",
            "model_accuracy": "Model Accuracy",
            "baseline_accuracy": "Baseline Accuracy",
            "model_minus_baseline": "Model - Baseline",
            "sample_count": "Sample Count",
            "evaluated_start": "Start",
            "evaluated_end": "End",
        }
    )


def _accuracy_source_breakdown_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("rows", "confirmed_rows"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    if "accuracy" in display.columns:
        display["accuracy"] = display["accuracy"].map(_format_percent)
    return display.rename(
        columns={
            "source": "Source",
            "rows": "Rows",
            "confirmed_rows": "Confirmed Rows",
            "accuracy": "Accuracy",
        }
    )


def _accuracy_summary_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("sample_count",):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    percent_columns = (
        "model_accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "predicted_up_accuracy",
        "predicted_down_accuracy",
        "actual_up_rate",
        "actual_down_rate",
        "predicted_up_rate",
        "predicted_down_rate",
        "average_confidence",
        "confidence_min",
        "confidence_max",
        "confidence_median",
        "confidence_ge_55_rate",
        "always_up",
        "always_down",
        "previous_day_direction",
        "moving_average_trend_only",
        "class_distribution_baseline",
    )
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "source": "Source",
            "period": "Period",
            "sample_count": "Sample Count",
            "warning": "Warning",
            "evaluated_start": "Start",
            "evaluated_end": "End",
            "model_accuracy": "Model Accuracy",
            "balanced_accuracy": "Balanced Accuracy",
            "up_recall": "Up Recall",
            "down_recall": "Down Recall",
            "predicted_up_accuracy": "Predicted Up Accuracy",
            "predicted_down_accuracy": "Predicted Down Accuracy",
            "actual_up_rate": "Actual Up Rate",
            "actual_down_rate": "Actual Down Rate",
            "predicted_up_rate": "Predicted Up Rate",
            "predicted_down_rate": "Predicted Down Rate",
            "average_confidence": "Avg Confidence",
            "confidence_min": "Confidence Min",
            "confidence_max": "Confidence Max",
            "confidence_median": "Confidence Median",
            "confidence_ge_55_rate": "Confidence >=55%",
            "always_up": "Always Up",
            "always_down": "Always Down",
            "previous_day_direction": "Previous Day",
            "moving_average_trend_only": "MA Trend",
            "class_distribution_baseline": "Class Distribution",
        }
    )


def _accuracy_probability_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("count", "count_48_52", "count_ge_55"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    for column in ("min", "max", "mean", "median"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "metric": "Metric",
            "count": "Count",
            "min": "Min",
            "max": "Max",
            "mean": "Mean",
            "median": "Median",
            "count_48_52": "48-52% Count",
            "count_ge_55": ">=55% Count",
        }
    )


def _accuracy_flat_summary_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("sample_count", "flat_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    for column in ("flat_rate", "flat_avg_actual_return", "flat_avg_abs_actual_return"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "sample_count": "Sample Count",
            "flat_count": "Flat Count",
            "flat_rate": "Flat Rate",
            "flat_avg_actual_return": "Flat Avg Return",
            "flat_avg_abs_actual_return": "Flat Avg Abs Return",
        }
    )


def _accuracy_confidence_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "threshold" in display.columns:
        display["threshold"] = display["threshold"].map(lambda value: f">= {_format_percent(value)}")
        display.loc[display["threshold"].eq(">= 0.00%"), "threshold"] = "All predictions"
    if "sample_count" in display.columns:
        display["sample_count"] = display["sample_count"].map(lambda value: _format_number(value, digits=0))
    for column in ("accuracy", "precision", "recall", "hit_rate", "average_confidence", "coverage"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "threshold": "Confidence",
            "sample_count": "Sample Count",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "hit_rate": "Hit Rate",
            "average_confidence": "Avg Confidence",
            "coverage": "Coverage",
        }
    )


def _accuracy_context_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "sample_count" in display.columns:
        display["sample_count"] = display["sample_count"].map(lambda value: _format_number(value, digits=0))
    for column in ("accuracy", "up_accuracy", "down_accuracy", "average_confidence"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "regime": "Regime",
            "anomaly_level": "Anomaly Level",
            "sample_count": "Sample Count",
            "accuracy": "Accuracy",
            "up_accuracy": "Predicted Up Accuracy",
            "down_accuracy": "Predicted Down Accuracy",
            "average_confidence": "Avg Confidence",
        }
    )


def _accuracy_binary_class_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "sample_count" in display.columns:
        display["sample_count"] = display["sample_count"].map(lambda value: _format_number(value, digits=0))
    if "value" in display.columns:
        display["value"] = display["value"].map(_format_percent)
    return display.rename(
        columns={
            "metric": "Metric",
            "class": "Class",
            "sample_count": "Sample Count",
            "value": "Value",
        }
    )


def _accuracy_multiclass_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("precision", "recall"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("support", "predicted_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "class_id": "Class ID",
            "class": "Class",
            "precision": "Precision",
            "recall": "Recall",
            "support": "Support",
            "predicted_count": "Predicted Count",
        }
    )


def _accuracy_hits_misses_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("actual_return", "confidence"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "anomaly_score" in display.columns:
        display["anomaly_score"] = display["anomaly_score"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "category": "Category",
            "feature_date": "Feature Date",
            "target_close_date": "Target Close Date",
            "predicted_class": "Predicted",
            "actual_direction": "Actual",
            "actual_return": "Actual Return",
            "confidence": "Confidence",
            "regime": "Regime",
            "anomaly_score": "Anomaly Score",
            "top_features": "Top Features",
        }
    )


def _render_multiclass_evaluation_section(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("Multiclass Evaluation")
    threshold_profile = _select_multiclass_threshold_profile(
        key="evaluation_multiclass_threshold_profile_select",
    )
    weight_mode = _select_multiclass_weight_mode(
        key="evaluation_multiclass_weight_mode_select",
    )
    st.caption(f"Selected threshold profile: {threshold_profile} / weight mode: {weight_mode}")
    _render_threshold_profile_comparison(ticker, data, settings)

    metrics_bundle = _load_multiclass_metrics(ticker, threshold_profile, weight_mode)
    metrics = metrics_bundle.get("metrics", {}) if metrics_bundle else {}
    if not isinstance(metrics, dict) or not metrics:
        st.info("Multiclass metrics are not available. Run train-multiclass first.")
        st.code(
            f"python main.py train-multiclass --ticker {ticker} --threshold-profile {threshold_profile} --weight-mode {weight_mode}",
            language="powershell",
        )
        _render_label_distribution_analysis(data, settings, threshold_profile)
        return

    label_thresholds = metrics_bundle.get("label_thresholds") or metrics.get("label_thresholds")
    if isinstance(label_thresholds, dict) and label_thresholds:
        st.caption("Trained label thresholds")
        st.dataframe(
            _multiclass_threshold_display_frame(label_thresholds),
            use_container_width=True,
            hide_index=True,
        )

    overview_cols = st.columns(6)
    overview_cols[0].metric("Rows", _format_number(metrics.get("rows"), digits=0))
    overview_cols[1].metric("Accuracy", _format_percent(metrics.get("accuracy")))
    overview_cols[2].metric("Macro F1", _format_percent(metrics.get("macro_f1")))
    overview_cols[3].metric("Weighted F1", _format_percent(metrics.get("weighted_f1")))
    overview_cols[4].metric(
        "Crash recall",
        _format_percent(_multiclass_class_metric(metrics, "crash", "recall")),
    )
    overview_cols[5].metric(
        "Surge recall",
        _format_percent(_multiclass_class_metric(metrics, "surge", "recall")),
    )

    tail_cols = st.columns(4)
    tail_cols[0].metric(
        "Crash precision",
        _format_percent(_multiclass_class_metric(metrics, "crash", "precision")),
    )
    tail_cols[1].metric(
        "Surge precision",
        _format_percent(_multiclass_class_metric(metrics, "surge", "precision")),
    )
    tail_cols[2].metric(
        "Crash samples",
        _format_number(_multiclass_class_metric(metrics, "crash", "support"), digits=0),
    )
    tail_cols[3].metric(
        "Surge samples",
        _format_number(_multiclass_class_metric(metrics, "surge", "support"), digits=0),
    )

    for message in _multiclass_evaluation_summary(metrics, _load_binary_metrics(ticker)):
        st.write(f"- {message}")

    class_frame = _multiclass_per_class_frame(metrics)
    if not class_frame.empty:
        st.caption("Class-level precision / recall / F1")
        st.dataframe(
            _multiclass_per_class_display_frame(class_frame),
            use_container_width=True,
            hide_index=True,
        )

    confusion_frame = _multiclass_confusion_frame(metrics)
    if not confusion_frame.empty:
        st.caption("Confusion Matrix")
        matrix_cols = st.columns([1.2, 1])
        with matrix_cols[0]:
            st.plotly_chart(
                _build_multiclass_confusion_heatmap(confusion_frame),
                use_container_width=True,
            )
        with matrix_cols[1]:
            st.dataframe(confusion_frame, use_container_width=True)

    return_frame, error = _multiclass_test_prediction_returns(
        ticker,
        data,
        settings,
        threshold_profile,
        weight_mode,
    )
    if not return_frame.empty:
        return_summary = _multiclass_return_summary_frame(return_frame)
        st.caption("Next-day return by predicted 5-class label on the chronological test split")
        st.plotly_chart(
            _build_multiclass_return_chart(return_summary),
            use_container_width=True,
        )
        st.dataframe(
            _multiclass_return_display_frame(return_summary),
            use_container_width=True,
            hide_index=True,
        )
    elif error:
        st.caption(f"Return-by-class check is unavailable: {error}")

    _render_label_distribution_analysis(data, settings, threshold_profile)


def _render_risk_validation_section(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("Risk Score Validation")
    st.caption(
        "This section checks whether Surge / Crash Risk Scores relate to future returns. "
        "It is not a trading backtest; it validates the risk score itself."
    )
    risk_profile, risk_weight_mode = _select_risk_multiclass_setting(
        key="evaluation_risk_validation_multiclass_setting_select",
    )
    risk_model_name = _multiclass_model_name(risk_profile, risk_weight_mode)
    saved_multiclass = _read_prediction_history(
        settings.paths.database_path,
        ticker=ticker,
        target="next_close_multiclass",
        model_name=risk_model_name,
    )

    risk_validation = _risk_validation_module()
    config = risk_validation.RiskValidationConfig(
        min_history_rows=st.number_input(
            "Risk validation minimum history rows",
            min_value=20,
            max_value=252,
            value=60,
            step=10,
            key="risk_validation_min_history_rows",
        )
    )
    frame = risk_validation.build_risk_validation_frame(
        data,
        config=config,
        multiclass_predictions=saved_multiclass,
    )
    if frame.empty:
        st.info("Risk validation is not available for the current dataset.")
        return

    bucket_mode = st.selectbox(
        "Risk bucket mode",
        options=risk_validation.bucket_mode_options(),
        index=risk_validation.bucket_mode_options().index(risk_validation.DEFAULT_BUCKET_MODE),
        format_func=lambda mode: _risk_bucket_mode_label(risk_validation, mode),
        key="risk_validation_bucket_mode_select",
    )
    calibrated_frame = risk_validation.apply_bucket_mode(frame, bucket_mode)

    surge_summary = risk_validation.summarize_risk_buckets(
        frame,
        risk_type="surge",
        bucket_mode=bucket_mode,
    )
    crash_summary = risk_validation.summarize_risk_buckets(
        frame,
        risk_type="crash",
        bucket_mode=bucket_mode,
    )
    distribution = risk_validation.risk_score_distribution(frame)
    current_risk_scores = _latest_risk_scores_from_validation_frame(frame)
    surge_mode_comparison = risk_validation.compare_bucket_modes(frame, risk_type="surge")
    crash_mode_comparison = risk_validation.compare_bucket_modes(frame, risk_type="crash")
    surge_high = risk_validation.high_risk_days(
        frame,
        risk_type="surge",
        bucket_mode=bucket_mode,
        limit=20,
    )
    crash_high = risk_validation.high_risk_days(
        frame,
        risk_type="crash",
        bucket_mode=bucket_mode,
        limit=20,
    )

    _render_risk_evaluation_overview(
        ticker=ticker,
        frame=frame,
        calibrated_frame=calibrated_frame,
        distribution=distribution,
        current_risk_scores=current_risk_scores,
        risk_profile=risk_profile,
        risk_weight_mode=risk_weight_mode,
        bucket_mode=bucket_mode,
        risk_validation=risk_validation,
    )
    _render_risk_evaluation_reading_guide()

    if _risk_high_count(surge_summary) == 0 or _risk_high_count(crash_summary) == 0:
        st.warning("High bucket has no samples. Consider quantile-based calibration.")
    for message in risk_validation.build_risk_validation_messages(surge_summary, crash_summary):
        st.write(f"- {message}")

    with st.expander("Current Risk Summary", expanded=True):
        _render_current_risk_summary_table(
            frame=frame,
            calibrated_frame=calibrated_frame,
            current_risk_scores=current_risk_scores,
            risk_validation=risk_validation,
        )

    with st.expander("Risk Validation", expanded=True):
        summary_cols = st.columns(2)
        with summary_cols[0]:
            st.caption("Surge Risk bucket performance")
            st.plotly_chart(
                _build_risk_validation_bucket_chart(surge_summary, risk_type="surge"),
                use_container_width=True,
            )
            st.dataframe(
                _risk_validation_summary_display_frame(surge_summary, risk_type="surge"),
                use_container_width=True,
                hide_index=True,
            )
        with summary_cols[1]:
            st.caption("Crash Risk bucket performance")
            st.plotly_chart(
                _build_risk_validation_bucket_chart(crash_summary, risk_type="crash"),
                use_container_width=True,
            )
            st.dataframe(
                _risk_validation_summary_display_frame(crash_summary, risk_type="crash"),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Risk Calibration", expanded=False):
        st.caption("Bucket mode comparison")
        mode_cols = st.columns(2)
        with mode_cols[0]:
            st.dataframe(
                _risk_bucket_mode_comparison_display_frame(surge_mode_comparison, risk_type="surge"),
                use_container_width=True,
                hide_index=True,
            )
        with mode_cols[1]:
            st.dataframe(
                _risk_bucket_mode_comparison_display_frame(crash_mode_comparison, risk_type="crash"),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Risk Bucket / Percentile Details", expanded=False):
        if distribution.empty:
            st.info("Risk Score distribution is not available.")
        else:
            dist_cols = st.columns([1.2, 1])
            with dist_cols[0]:
                st.plotly_chart(
                    _build_risk_score_histogram(frame, current_scores=current_risk_scores),
                    use_container_width=True,
                )
                if current_risk_scores:
                    st.caption("Vertical markers show the latest available Risk Score in this validation frame.")
            with dist_cols[1]:
                st.dataframe(
                    _risk_score_distribution_display_frame(distribution),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("Today Signal Risk Context", expanded=False):
        st.caption(
            "Today Signal uses Risk Score percentile context in Daily Brief. "
            "This section shows recent High bucket days from the selected calibration mode."
        )
        high_cols = st.columns(2)
        with high_cols[0]:
            st.caption("Recent Surge Risk High days")
            if surge_high.empty:
                st.info("No Surge Risk High days are available.")
            else:
                st.dataframe(
                    _risk_high_days_display_frame(surge_high, risk_type="surge"),
                    use_container_width=True,
                    hide_index=True,
                )
        with high_cols[1]:
            st.caption("Recent Crash Risk High days")
            if crash_high.empty:
                st.info("No Crash Risk High days are available.")
            else:
                st.dataframe(
                    _risk_high_days_display_frame(crash_high, risk_type="crash"),
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("Raw risk tables / details", expanded=False):
        st.caption("Risk Score vs future 5-day return")
        st.plotly_chart(_build_risk_validation_scatter(calibrated_frame), use_container_width=True)


def _render_news_impact_validation_section(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("News Impact Validation")
    st.caption(
        "This validates news features directly. Ablation checks whether the whole news feature group "
        "helps the model; this section checks how news signals relate to future returns and Risk Scores."
    )
    news_status = _news_fetch_status(ticker, settings)
    _render_news_evaluation_overview(news_status)
    _render_news_evaluation_reading_guide()

    module = _news_impact_module()
    min_history_rows = st.number_input(
        "News impact minimum history rows",
        min_value=20,
        max_value=252,
        value=60,
        step=10,
        key="news_impact_min_history_rows",
    )
    config = module.NewsImpactConfig(min_history_rows=int(min_history_rows))
    frame = module.build_news_impact_frame(data, config=config)
    if frame.empty:
        st.info("News Impact Validation is not available yet. Build features after adding news data.")
        return

    summaries = module.build_news_impact_summaries(frame, config=config)
    st.session_state[f"news_impact_frame_{ticker}"] = frame
    st.session_state[f"news_impact_summaries_{ticker}"] = summaries

    latest_context = module.current_news_impact_context(frame, summaries)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Validation rows", _format_number(len(frame), digits=0))
    metric_cols[1].metric(
        "News days",
        _format_number((pd.to_numeric(frame["news_count_1d"], errors="coerce") > 0).sum(), digits=0),
    )
    metric_cols[2].metric(
        "Negative flag days",
        _format_number((pd.to_numeric(frame["negative_news_flag"], errors="coerce") > 0).sum(), digits=0),
    )
    metric_cols[3].metric(
        "Positive flag days",
        _format_number((pd.to_numeric(frame["positive_news_flag"], errors="coerce") > 0).sum(), digits=0),
    )
    metric_cols[4].metric(
        "Shock days",
        _format_number(
            (
                (pd.to_numeric(frame["negative_news_shock"], errors="coerce") > 0)
                | (pd.to_numeric(frame["positive_news_shock"], errors="coerce") > 0)
            ).sum(),
            digits=0,
        ),
    )
    st.caption(latest_context)
    for message in module.build_news_impact_messages(summaries):
        st.write(f"- {message}")

    news_dir = _news_module().default_news_dir(settings.paths.raw_data_dir.parent)
    with st.expander("News Health / Data Quality", expanded=False):
        _render_news_evaluation_quality_details(news_status)

    with st.expander("News Impact Validation details", expanded=True):
        _render_news_impact_detail_tabs(module, data, news_dir, ticker, summaries, config)

    with st.expander("Price Move Rankings", expanded=False):
        _render_news_price_move_ranking_section(module, data, news_dir, ticker)

    with st.expander("Category / Source Analysis", expanded=False):
        _render_category_price_move_ranking_section(module, data, news_dir, ticker, config)
        _render_category_source_type_impact_section(module, data, news_dir, ticker, config)

    with st.expander("Profile / Query Evaluation", expanded=False):
        _render_profile_news_impact_section(module, data, news_dir, ticker, config)

    with st.expander("NewsLimit / Daily Profile Operations", expanded=False):
        _render_news_evaluation_operations_guide(news_status)


def _render_news_evaluation_overview(news_status: dict[str, object]) -> None:
    st.subheader("News Evaluation Overview")
    health = news_status.get("health", {}) or {}
    history = _fetch_history_frame(news_status)
    recommendation_rows = news_status.get("profile_recommendations") or []

    cols = st.columns(5)
    cols[0].metric("News rows", _format_number(news_status.get("count"), digits=0))
    cols[1].metric("Latest news date", _format_date(news_status.get("latest_date")))
    cols[2].metric("Latest fetched_at", news_status.get("latest_fetched_at") or "-")
    cols[3].metric("Unknown sources", _format_number(health.get("unknown_source_count"), digits=0))
    cols[4].metric("Duplicates", _format_number(health.get("duplicate_count"), digits=0))

    ops_cols = st.columns(4)
    ops_cols[0].metric("Observation days", _format_number(_fetch_history_observation_days(history), digits=0))
    ops_cols[1].metric("Daily profiles", _format_number(_current_daily_profile_count(), digits=0))
    ops_cols[2].metric("Current NewsLimit", _latest_news_limit_label(history))
    ops_cols[3].metric("Profile recommendation", _news_profile_recommendation_summary(recommendation_rows))

    unknown_sources = safe_int(health.get("unknown_source_count")) or 0
    if unknown_sources:
        st.warning("Unknown sources are present. Check Operations > Unknown Sources before reading News Impact strongly.")
    elif str(health.get("status") or "").upper() == "OK":
        st.caption("News CSV quality is OK. News Impact remains exploratory, not causal proof.")


def _render_news_evaluation_reading_guide() -> None:
    with st.expander("How to read news evaluation", expanded=False):
        st.write("- News Impact is not causal proof; it checks price moves after news.")
        st.write("- Price Move Ranking is exploratory, for finding articles worth reviewing.")
        st.write("- Category / Source / Profile views help judge which news cuts may be useful.")
        st.write("- Do not over-trust categories or profiles with small sample sizes.")
        st.write("- NewsLimit 3 Evaluation is an operational data-quality view, not proof of predictive power.")


def _render_news_evaluation_quality_details(news_status: dict[str, object]) -> None:
    health = news_status.get("health", {}) or {}
    quality_rows = [
        {"Item": "News rows", "Value": _format_number(news_status.get("count"), digits=0)},
        {"Item": "Latest news date", "Value": _format_date(news_status.get("latest_date"))},
        {"Item": "Latest fetched_at", "Value": news_status.get("latest_fetched_at") or "-"},
        {"Item": "Health status", "Value": health.get("status") or "-"},
        {"Item": "Duplicate count", "Value": _format_number(health.get("duplicate_count"), digits=0)},
        {"Item": "Invalid dates", "Value": _format_number(health.get("invalid_date_count"), digits=0)},
        {"Item": "Mojibake suspects", "Value": _format_number(health.get("mojibake_suspect_count"), digits=0)},
        {"Item": "Unknown sources", "Value": _format_number(health.get("unknown_source_count"), digits=0)},
    ]
    st.dataframe(pd.DataFrame(quality_rows), use_container_width=True, hide_index=True)
    _render_unknown_sources_section(news_status)

    quality = pd.DataFrame(news_status.get("profile_quality") or [])
    if not quality.empty:
        st.caption("Profile quality summary")
        st.dataframe(
            _news_profile_quality_display_frame(quality),
            use_container_width=True,
            hide_index=True,
        )


def _render_news_impact_detail_tabs(
    module,
    data: pd.DataFrame,
    news_dir,
    ticker: str,
    summaries: dict[str, pd.DataFrame],
    config,
) -> None:
    tabs = st.tabs(
        [
            "Volume / Sentiment",
            "Flags / Shocks",
            "Feature Relationships",
            "Categories",
            "Source Types",
            "Query Profiles",
        ]
    )
    with tabs[0]:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.caption("News volume buckets")
            volume = summaries.get("volume", pd.DataFrame())
            st.dataframe(_news_impact_display_frame(volume), use_container_width=True, hide_index=True)
        with col2:
            st.caption("Sentiment buckets")
            sentiment = summaries.get("sentiment", pd.DataFrame())
            st.dataframe(_news_impact_display_frame(sentiment), use_container_width=True, hide_index=True)
            _render_news_examples_for_rows(
                module,
                data,
                news_dir,
                ticker,
                sentiment,
                dimension="sentiment_bucket",
                label_column="group",
                title_prefix="sentiment bucket",
                config=config,
            )
        with col3:
            st.caption("Importance buckets")
            importance = summaries.get("importance", pd.DataFrame())
            st.dataframe(_news_impact_display_frame(importance), use_container_width=True, hide_index=True)
            _render_news_examples_for_rows(
                module,
                data,
                news_dir,
                ticker,
                importance,
                dimension="importance_bucket",
                label_column="group",
                title_prefix="importance bucket",
                config=config,
            )
        chart_frame = pd.concat(
            [
                volume.assign(section="Volume") if not volume.empty else pd.DataFrame(),
                sentiment.assign(section="Sentiment") if not sentiment.empty else pd.DataFrame(),
                importance.assign(section="Importance") if not importance.empty else pd.DataFrame(),
            ],
            ignore_index=True,
        )
        if not chart_frame.empty:
            st.plotly_chart(_build_news_impact_return_chart(chart_frame), use_container_width=True)

    with tabs[1]:
        events = summaries.get("events", pd.DataFrame())
        if events.empty:
            st.info("No news flag/shock rows are available.")
        else:
            st.dataframe(_news_event_impact_display_frame(events), use_container_width=True, hide_index=True)
            st.plotly_chart(_build_news_event_chart(events), use_container_width=True)
            _render_news_examples_for_rows(
                module,
                data,
                news_dir,
                ticker,
                events,
                dimension="event",
                label_column="event",
                title_prefix="event",
                config=config,
            )

    with tabs[2]:
        relationships = summaries.get("feature_relationships", pd.DataFrame())
        if relationships.empty:
            st.info("No feature relationship rows are available.")
        else:
            st.dataframe(
                _news_feature_relationship_display_frame(relationships),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        category_frame = module.summarize_category_impact(data, news_dir, ticker, config=config)
        if category_frame.empty:
            st.info("No category-level news impact rows are available.")
        else:
            st.caption(
                "Category-level rows use the category inferred or registered on each news item. "
                "This helps separate earnings, recall/safety, macro, EV/hybrid, and other themes."
            )
            st.dataframe(
                _news_profile_impact_display_frame(category_frame),
                use_container_width=True,
                hide_index=True,
            )
            _render_news_examples_for_rows(
                module,
                data,
                news_dir,
                ticker,
                category_frame,
                dimension="category",
                label_column="category",
                title_prefix="category",
                config=config,
            )

    with tabs[4]:
        source_type_frame = module.summarize_source_type_impact(data, news_dir, ticker, config=config)
        if source_type_frame.empty:
            st.info("No source-type news impact rows are available.")
        else:
            st.caption(
                "Source-type rows use credibility metadata from configs/news_sources.yaml. "
                "They are meant as context, not as proof that a source is always right. "
                "Daily Brief shows a short top source-type summary; use this table for details."
            )
            st.dataframe(
                _news_profile_impact_display_frame(source_type_frame),
                use_container_width=True,
                hide_index=True,
            )
            _render_news_examples_for_rows(
                module,
                data,
                news_dir,
                ticker,
                source_type_frame,
                dimension="source_type",
                label_column="source_type",
                title_prefix="source type",
                config=config,
            )

    with tabs[5]:
        profile_frame = module.summarize_query_profile_impact(data, news_dir, ticker, config=config)
        if profile_frame.empty:
            st.info("No news query profile configuration is available.")
        else:
            st.caption(
                "Profile-level rows use the query_profile column for newly fetched news. "
                "Older/manual rows may appear as untracked."
            )
            st.dataframe(
                _news_profile_impact_display_frame(profile_frame),
                use_container_width=True,
                hide_index=True,
            )


def _render_news_evaluation_operations_guide(news_status: dict[str, object]) -> None:
    history = _fetch_history_frame(news_status)
    st.caption(
        "This section is a light Evaluation-side pointer. "
        "Use Operations for the full Data Collection Monitor and NewsLimit 3 Evaluation."
    )
    cols = st.columns(5)
    cols[0].metric("Daily profiles", _format_number(_current_daily_profile_count(), digits=0))
    cols[1].metric("Current NewsLimit", _latest_news_limit_label(history))
    cols[2].metric("Observation days", _format_number(_fetch_history_observation_days(history), digits=0))
    cols[3].metric("Latest duplicate rate", _latest_fetch_duplicate_rate(history))
    st.code(
        ".\\scripts\\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3",
        language="powershell",
    )
    st.caption(
        "NewsLimit 3 / 4-profile setup is under observation. Do not change settings here; re-evaluate after more fetch_history."
    )


def _fetch_history_observation_days(history: pd.DataFrame) -> int:
    if history.empty:
        return 0
    date_column = "date" if "date" in history.columns else "fetched_at"
    if date_column not in history.columns:
        return 0
    dates = pd.to_datetime(history[date_column], errors="coerce", format="mixed").dropna()
    if dates.empty:
        return 0
    return int(dates.dt.date.nunique())


def _latest_news_limit_label(history: pd.DataFrame) -> str:
    if history.empty:
        return "-"
    for column in ("limit", "news_limit", "requested_limit"):
        if column in history.columns:
            values = history[column].dropna()
            if not values.empty:
                return str(values.iloc[-1])
    fetched = pd.to_numeric(history.get("fetched_count"), errors="coerce") if "fetched_count" in history.columns else None
    if fetched is not None and not fetched.dropna().empty:
        latest = fetched.dropna().iloc[-1]
        if latest <= 3:
            return "3"
    return "3"


def _latest_fetch_duplicate_rate(history: pd.DataFrame) -> str:
    if history.empty:
        return "-"
    latest = history.tail(1).iloc[0]
    duplicates = safe_float(latest.get("duplicate_count"))
    fetched = safe_float(latest.get("fetched_count"))
    if fetched is None or fetched <= 0 or duplicates is None:
        return "-"
    return _format_percent(duplicates / fetched)


def _news_profile_recommendation_summary(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "-"
    counts: dict[str, int] = {}
    for row in rows:
        recommendation = str(row.get("recommendation") or row.get("daily_operation_recommendation") or "-")
        counts[recommendation] = counts.get(recommendation, 0) + 1
    if not counts:
        return "-"
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(f"{name}: {count}" for name, count in ordered[:2])


def _render_today_signal_validation_section(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("Today Signal Validation")
    st.caption(
        "This validates the Daily Brief decision itself. It combines walk-forward binary predictions "
        "with historical Risk Scores, then checks future returns by Today Signal."
    )

    today_signal = _today_signal_validation_module()
    risk_validation = _risk_validation_module()
    bucket_modes = risk_validation.bucket_mode_options()
    default_index = bucket_modes.index(DAILY_BRIEF_RISK_BUCKET_MODE)
    bucket_mode = st.selectbox(
        "Today Signal risk calibration",
        options=bucket_modes,
        index=default_index,
        format_func=lambda mode: _risk_bucket_mode_label(risk_validation, mode),
        key="today_signal_validation_bucket_mode_select",
    )

    result_key = f"walk_forward_backtest_{ticker}"
    result = st.session_state.get(result_key)
    if result is None or result.predictions.empty:
        st.info(
            "Today Signal Validation needs walk-forward binary predictions. "
            "Run Backtest first, or run the validation backtest here."
        )
        if st.button("Run Today Signal validation backtest", key="today_signal_validation_run_button"):
            walk_forward = _walk_forward_backtest_module()
            anomaly_enriched, _, _, _ = _anomaly_context(data)
            config = walk_forward.WalkForwardConfig(ticker=ticker)
            with st.spinner("Running walk-forward predictions for Today Signal Validation..."):
                result = walk_forward.run_walk_forward_backtest(anomaly_enriched, config=config)
            st.session_state[result_key] = result
            st.session_state[f"walk_forward_backtest_config_{ticker}"] = config
        else:
            return

    if result is None or result.predictions.empty:
        st.warning("No walk-forward prediction rows were produced.")
        return

    risk_profile, risk_weight_mode = DEFAULT_RISK_MULTICLASS_OPTION
    risk_model_name = _multiclass_model_name(risk_profile, risk_weight_mode)
    saved_multiclass = _read_prediction_history(
        settings.paths.database_path,
        ticker=ticker,
        target="next_close_multiclass",
        model_name=risk_model_name,
    )
    risk_frame = risk_validation.build_risk_validation_frame(
        data,
        multiclass_predictions=saved_multiclass,
    )
    validation_frame = today_signal.build_today_signal_validation_frame(
        result.predictions,
        risk_frame,
        data=data,
        bucket_mode=bucket_mode,
    )
    if validation_frame.empty:
        st.warning("Today Signal Validation has no overlapping backtest and risk-score rows.")
        return

    summary = today_signal.summarize_today_signals(validation_frame)
    st.session_state[f"today_signal_validation_summary_{ticker}"] = summary
    st.session_state[f"today_signal_validation_frame_{ticker}"] = validation_frame

    st.caption(
        "Risk Validation checks Risk Score alone. This section checks the combined Daily Brief signal "
        "using the selected risk calibration."
    )
    overview_cols = st.columns(5)
    overview_cols[0].metric("Validation rows", _format_number(len(validation_frame), digits=0))
    overview_cols[1].metric(
        "High Risk count",
        _format_number(_today_signal_summary_value(summary, "High Risk", "count"), digits=0),
    )
    overview_cols[2].metric(
        "High Risk avg 5D",
        _format_percent(_today_signal_summary_value(summary, "High Risk", "average_return_5d")),
    )
    overview_cols[3].metric(
        "Bullish avg 5D",
        _format_percent(_today_signal_summary_value(summary, "Bullish", "average_return_5d")),
    )
    overview_cols[4].metric(
        "Bullish 5D up rate",
        _format_percent(_today_signal_summary_value(summary, "Bullish", "up_rate_5d")),
    )

    for message in today_signal.build_today_signal_validation_messages(summary):
        st.write(f"- {message}")

    suggestions = today_signal.build_signal_improvement_suggestions(summary)
    st.subheader("Signal Improvement Suggestions")
    st.caption(
        "Rule-based suggestions from validation results. This does not change Today Signal logic yet."
    )
    if suggestions.empty:
        st.info("No improvement suggestions are available yet.")
    else:
        st.dataframe(
            _today_signal_improvement_display_frame(suggestions),
            use_container_width=True,
            hide_index=True,
        )

    comparison, distribution, changes, recommendation = today_signal.compare_today_signal_configs(
        result.predictions,
        risk_frame,
        data=data,
        bucket_mode=bucket_mode,
    )
    st.subheader("Today Signal Config Experiments")
    st.caption(
        "Compares candidate rule configs against the same validation data. "
        "This is read-only and does not rewrite configs/today_signal.yaml."
    )
    if comparison.empty:
        st.info("No Today Signal config experiment results are available.")
    else:
        recommended_name = str(recommendation.get("config_name") or "-")
        adoption = today_signal.build_config_adoption_recommendations(comparison)
        st.subheader("Adoption Recommendation")
        st.success(f"Recommended candidate: {recommended_name}")
        st.caption(str(recommendation.get("reason") or "No recommendation reason is available."))
        if not adoption.empty:
            recommended_adoption = adoption[adoption["config_name"] == recommended_name]
            if not recommended_adoption.empty:
                st.info(str(recommended_adoption.iloc[0].get("adoption_reason") or ""))
            st.dataframe(
                _today_signal_adoption_display_frame(adoption),
                use_container_width=True,
                hide_index=True,
            )

        st.caption("Config experiment score table")
        st.dataframe(
            _today_signal_config_experiment_display_frame(comparison),
            use_container_width=True,
            hide_index=True,
        )
        if not distribution.empty:
            st.plotly_chart(
                _build_today_signal_config_distribution_chart(distribution),
                use_container_width=True,
            )
            st.caption("Signal distribution by candidate")
            st.dataframe(
                _today_signal_config_distribution_display_frame(distribution),
                use_container_width=True,
                hide_index=True,
            )
        recommended_changes = (
            changes[changes["config_name"] == recommended_name].copy()
            if not changes.empty and recommended_name != "-"
            else pd.DataFrame()
        )
        if not recommended_changes.empty:
            st.caption(
                "Manual adoption guide. These are the values to copy into configs/today_signal.yaml "
                "if you choose this candidate later."
            )
            st.dataframe(
                _today_signal_config_changes_display_frame(recommended_changes),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("The recommended candidate does not require manual config changes from current.")

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(_build_today_signal_return_chart(summary), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(_build_today_signal_count_chart(summary), use_container_width=True)

    st.caption("Signal-level performance")
    st.dataframe(
        _today_signal_summary_display_frame(summary),
        use_container_width=True,
        hide_index=True,
    )

    high_risk_rows = validation_frame[validation_frame["today_signal"] == "High Risk"].copy()
    if not high_risk_rows.empty:
        st.caption("Recent High Risk signals")
        st.dataframe(
            _today_signal_validation_rows_display_frame(high_risk_rows.tail(20)),
            use_container_width=True,
            hide_index=True,
        )


def _multiclass_class_metric(metrics: dict[str, object], class_name: str, metric_name: str):
    direct_value = metrics.get(f"{class_name}_{metric_name}")
    if direct_value is not None:
        return direct_value
    per_class = metrics.get("per_class", {})
    if not isinstance(per_class, dict):
        return None
    values = per_class.get(class_name, {})
    if not isinstance(values, dict):
        return None
    return values.get(metric_name)


def _risk_bucket_mode_label(risk_validation, bucket_mode: str) -> str:
    description = risk_validation.bucket_mode_description(bucket_mode)
    return f"{bucket_mode} - {description}"


def _risk_high_count(summary: pd.DataFrame) -> int:
    if summary.empty or "bucket" not in summary.columns:
        return 0
    rows = summary[summary["bucket"] == "High"]
    if rows.empty:
        return 0
    return safe_int(rows.iloc[0].get("count")) or 0


def _render_risk_evaluation_overview(
    *,
    ticker: str,
    frame: pd.DataFrame,
    calibrated_frame: pd.DataFrame,
    distribution: pd.DataFrame,
    current_risk_scores: dict[str, float],
    risk_profile: str,
    risk_weight_mode: str,
    bucket_mode: str,
    risk_validation,
) -> None:
    st.subheader("Risk Evaluation Overview")
    surge_percentile = risk_validation.latest_risk_percentile(frame, risk_type="surge")
    crash_percentile = risk_validation.latest_risk_percentile(frame, risk_type="crash")
    overview_cols = st.columns(5)
    overview_cols[0].metric(
        "Surge Risk",
        _risk_overview_score_text(current_risk_scores.get("surge"), surge_percentile),
    )
    overview_cols[1].metric(
        "Crash Risk",
        _risk_overview_score_text(current_risk_scores.get("crash"), crash_percentile),
    )
    overview_cols[2].metric("Calibration", _risk_bucket_mode_short_label(bucket_mode))
    overview_cols[3].metric("Validation rows", _format_number(len(frame), digits=0))
    overview_cols[4].metric("Latest eval date", _risk_latest_evaluation_date(frame))

    high_cols = st.columns(4)
    high_cols[0].metric(
        "Surge High days",
        _format_number((calibrated_frame["surge_risk_bucket"] == "High").sum(), digits=0),
    )
    high_cols[1].metric(
        "Crash High days",
        _format_number((calibrated_frame["crash_risk_bucket"] == "High").sum(), digits=0),
    )
    high_cols[2].metric(
        "Saved multiclass days",
        _format_number(frame.get("multiclass_signal_used", pd.Series(dtype=bool)).sum(), digits=0),
    )
    high_cols[3].metric("Risk model", f"{risk_profile} / {risk_weight_mode}")

    st.caption(
        "To avoid future-data leakage, this MVP recomputes historical rule-based Risk Scores. "
        "Saved multiclass predictions are merged only on dates where they already exist."
    )
    if distribution.empty:
        st.caption("Risk Score distribution is not available yet.")
    else:
        st.caption(_risk_distribution_context_text(distribution))

    health_note = _risk_evaluation_health_note(ticker)
    if health_note:
        st.info(health_note)


def _render_risk_evaluation_reading_guide() -> None:
    with st.expander("How to read risk evaluation", expanded=False):
        st.write("- Risk Score is a caution level, not a trading instruction.")
        st.write("- Surge / Crash Risk is a warning indicator for sharp moves, separate from direction prediction.")
        st.write("- Calibration checks whether score levels match observed events or future returns.")
        st.write("- Do not over-trust buckets with small sample sizes, especially High bucket.")
        st.write("- Percentile shows how high the current score is versus historical distribution.")
        st.write("- If health-check WARN only contains TOPIX fallback and known live gaps, operation can continue.")


def _render_current_risk_summary_table(
    *,
    frame: pd.DataFrame,
    calibrated_frame: pd.DataFrame,
    current_risk_scores: dict[str, float],
    risk_validation,
) -> None:
    rows = []
    for risk_type in ("surge", "crash"):
        latest_score = current_risk_scores.get(risk_type)
        percentile = risk_validation.latest_risk_percentile(frame, risk_type=risk_type)
        bucket_column = f"{risk_type}_risk_bucket"
        bucket = "-"
        if not calibrated_frame.empty and bucket_column in calibrated_frame.columns:
            clean = calibrated_frame.dropna(subset=["date"]).sort_values("date")
            if not clean.empty:
                bucket = str(clean.iloc[-1].get(bucket_column) or "-")
        rows.append(
            {
                "Risk": risk_type.title(),
                "Latest Score": _format_number(latest_score, digits=0),
                "Percentile": _risk_percentile_position_text(percentile),
                "Selected Bucket": bucket,
                "Meaning": "Surge warning" if risk_type == "surge" else "Crash warning",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _risk_overview_score_text(score: float | None, percentile: float | None) -> str:
    return f"{_format_number(score, digits=0)} / {_risk_percentile_position_text(percentile)}"


def _risk_bucket_mode_short_label(bucket_mode: str) -> str:
    if bucket_mode == "fixed_33_66":
        return "Fixed 33/66"
    if bucket_mode == "quantile_50_85":
        return "Quantile 50/85"
    if bucket_mode == "quantile_60_90":
        return "Quantile 60/90"
    return bucket_mode


def _risk_latest_evaluation_date(frame: pd.DataFrame) -> str:
    if frame.empty or "date" not in frame.columns:
        return "-"
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return "-"
    return _format_date(dates.max())


def _risk_distribution_context_text(distribution: pd.DataFrame) -> str:
    parts = []
    for _, row in distribution.iterrows():
        risk_type = str(row.get("risk_type") or "-").title()
        parts.append(
            f"{risk_type}: mean {_format_number(row.get('mean'), digits=1)}, "
            f"p90 {_format_number(row.get('p90'), digits=1)}, max {_format_number(row.get('max'), digits=1)}"
        )
    return "Risk distribution snapshot: " + "; ".join(parts)


def _risk_evaluation_health_note(ticker: str) -> str:
    try:
        report = _load_health_report_for_ui(ticker)
    except Exception as error:
        LOGGER.debug("Could not load health note for risk evaluation: %s", error, exc_info=True)
        return ""
    overall = str(report.get("overall_status") or "").upper()
    if overall != "WARN":
        return ""
    warn_items = [
        item
        for item in report.get("items", []) or []
        if str(item.get("status")) == "WARN"
    ]
    names = " / ".join(str(item.get("name") or "") for item in warn_items)
    if "TOPIX" in names and "Missing live dates" in names:
        return "health-check is WARN, but current operations can continue when warnings are TOPIX fallback and known missed live feature dates only."
    return "health-check is WARN. Check Operations if this risk evaluation looks stale or incomplete."


def _risk_bucket_mode_comparison_display_frame(
    frame: pd.DataFrame,
    *,
    risk_type: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in (
        "high_event_rate",
        "high_average_return_5d",
        "high_average_return_20d",
        "high_extreme_20d",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("high_count", "medium_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    event_label = "High Surge Rate" if risk_type == "surge" else "High Crash Rate"
    extreme_label = "High Max Upside 20D" if risk_type == "surge" else "High Max Drawdown 20D"
    title = "Surge" if risk_type == "surge" else "Crash"
    display.insert(0, "risk_type", title)
    return display.rename(
        columns={
            "risk_type": "Risk",
            "bucket_mode": "Bucket Mode",
            "description": "Definition",
            "high_count": "High Count",
            "medium_count": "Medium Count",
            "high_event_rate": event_label,
            "high_average_return_5d": "High Avg 5D Return",
            "high_average_return_20d": "High Avg 20D Return",
            "high_extreme_20d": extreme_label,
            "interpretability": "Interpretability",
        }
    )


def _risk_score_distribution_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in ("mean", "max", "p50", "p60", "p85", "p90", "p95"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=1))
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "risk_type": "Risk",
            "count": "Count",
            "mean": "Mean",
            "max": "Max",
            "p50": "P50",
            "p60": "P60",
            "p85": "P85",
            "p90": "P90",
            "p95": "P95",
        }
    )


def _news_impact_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in (
        "average_return_1d",
        "average_return_5d",
        "average_return_20d",
        "up_rate_1d",
        "large_down_rate_20d",
        "large_up_rate_20d",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in (
        "average_crash_risk",
        "average_surge_risk",
        "average_crash_risk_change_1d",
        "average_surge_risk_change_1d",
        "average_source_credibility",
        "average_source_weight",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=1))
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(columns=_news_impact_column_labels())


def _news_event_impact_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = _news_impact_display_frame(frame)
    if "Event" in display.columns:
        display["Event"] = display["Event"].map(lambda value: str(value).replace("_", " ").title())
    return display


def _news_profile_impact_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = _news_impact_display_frame(frame)
    return display.rename(
        columns={
            "query_profile": "Query Profile",
            "tracking_status": "Tracking Status",
            "query": "Query",
            "ticker": "Ticker",
            "storage_ticker": "Storage Ticker",
            "category": "Category",
            "source_type": "Source Type",
            "news_dir": "News Dir",
        }
    )


def _render_profile_news_impact_section(
    module,
    data: pd.DataFrame,
    news_dir,
    ticker: str,
    config,
) -> None:
    st.subheader("Profile News Impact")
    st.caption(
        "News Profile Quality checks fetch efficiency and duplicate rate. "
        "This section checks price moves and Risk Score tendencies after news collected by each profile. "
        "Treat it as exploratory analysis because the news sample is still small."
    )
    mode_rows = module.profile_news_impact_mode_rows()
    mode_labels = [row["label"] for row in mode_rows]
    label_to_mode = {row["label"]: row["mode"] for row in mode_rows}
    selected_label = st.selectbox(
        "Profile impact ranking type",
        mode_labels,
        index=0,
        key="profile_news_impact_ranking_type",
    )
    ranking = module.build_profile_news_impact(
        data,
        news_dir,
        ticker,
        ranking_mode=label_to_mode.get(selected_label, "average_return_5d_desc"),
        config=config,
    )
    if ranking.empty:
        st.info("No profile-level news impact rows are available yet.")
        return

    low_count = ranking[pd.to_numeric(ranking["news_count"], errors="coerce").fillna(0) < 5]
    if not low_count.empty:
        st.caption(
            "Some profiles have few saved news rows. Use this as a profile review aid, "
            "not a statistical conclusion."
        )
    st.dataframe(
        _profile_news_impact_display_frame(ranking),
        use_container_width=True,
        hide_index=True,
    )

    review_rows = ranking[ranking["recommendation"].astype(str).eq("review_needed")].copy()
    if not review_rows.empty:
        st.caption("Daily profile review notes")
        for _, row in review_rows.head(5).iterrows():
            st.write(f"- {row.get('daily_decision_note')}")

    st.caption("Representative news by profile")
    for _, row in ranking.head(6).iterrows():
        profile = str(row.get("query_profile") or "")
        if not profile:
            continue
        examples = module.profile_news_impact_examples(data, news_dir, ticker, profile, limit=3)
        with st.expander(f"Show representative news: profile = {profile}", expanded=False):
            if examples.empty:
                st.info("No representative news examples are available for this profile.")
                continue
            st.caption("These are price moves after the news date; they do not imply causality.")
            display = _news_price_move_ranking_display_frame(examples)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config=_news_url_column_config(display),
            )
            _render_news_example_price_move_charts(
                module,
                data,
                examples,
                chart_key_prefix=f"profile_news_impact_{profile}",
            )


def _profile_news_impact_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in (
        "average_return_1d",
        "average_return_5d",
        "average_return_20d",
        "average_max_up_20d",
        "average_max_down_20d",
        "large_up_rate",
        "large_down_rate",
        "duplicate_rate",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in (
        "average_sentiment_score",
        "average_importance_score",
        "average_source_credibility_score",
        "average_source_market_impact_weight",
        "average_crash_risk",
        "average_surge_risk",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    for column in ("news_count", "total_fetched_rows", "total_saved_rows", "added_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    columns = [
        "query_profile",
        "group",
        "enabled_for_daily",
        "language",
        "region",
        "news_count",
        "total_fetched_rows",
        "added_count",
        "total_saved_rows",
        "duplicate_rate",
        "quality_status",
        "recommendation",
        "average_return_5d",
        "average_max_down_20d",
        "large_down_rate",
        "large_up_rate",
        "average_crash_risk",
        "average_surge_risk",
        "average_importance_score",
        "average_source_credibility_score",
        "latest_fetched_at",
        "daily_decision_note",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "query_profile": "Query Profile",
            "group": "Group",
            "enabled_for_daily": "Daily",
            "language": "Language",
            "region": "Region",
            "news_count": "News Count",
            "total_fetched_rows": "Fetched",
            "added_count": "Added",
            "total_saved_rows": "Saved Rows",
            "duplicate_rate": "Duplicate Rate",
            "quality_status": "Quality",
            "recommendation": "Recommendation",
            "average_return_5d": "Avg Return 5D",
            "average_max_down_20d": "Avg Max Down 20D",
            "large_down_rate": "Large Down Rate",
            "large_up_rate": "Large Up Rate",
            "average_crash_risk": "Avg Crash Risk",
            "average_surge_risk": "Avg Surge Risk",
            "average_importance_score": "Avg Importance",
            "average_source_credibility_score": "Avg Source Credibility",
            "latest_fetched_at": "Latest Fetched At",
            "daily_decision_note": "Daily Decision Note",
        }
    )


def _render_news_price_move_ranking_section(module, data: pd.DataFrame, news_dir, ticker: str) -> None:
    st.subheader("News Price Move Ranking")
    st.caption(
        "This ranks saved news by price moves after the news date. "
        "It is for research discovery and does not imply causality."
    )
    mode_rows = module.news_price_move_ranking_mode_rows()
    mode_labels = [row["label"] for row in mode_rows]
    label_to_mode = {row["label"]: row["mode"] for row in mode_rows}
    controls = st.columns([2, 1])
    selected_label = controls[0].selectbox(
        "Ranking type",
        mode_labels,
        index=0,
        key="news_price_move_ranking_type",
    )
    ranking_limit = controls[1].number_input(
        "Ranking rows",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
        key="news_price_move_ranking_rows",
    )
    ranking = module.build_news_price_move_ranking(
        data,
        news_dir,
        ticker,
        ranking_mode=label_to_mode.get(selected_label, "return_5d_desc"),
        limit=int(ranking_limit),
    )
    if ranking.empty:
        st.info("No saved news is available for price-move ranking yet.")
        return

    if len(ranking) < 20:
        st.caption(
            "The saved news sample is still small, so treat this as a research-candidate list, "
            "not a statistical conclusion."
        )
    display = _news_price_move_ranking_display_frame(ranking)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=_news_url_column_config(display),
    )
    with st.expander("Show charts for top ranked news", expanded=False):
        _render_news_example_price_move_charts(
            module,
            data,
            ranking,
            chart_key_prefix=f"ranking_{label_to_mode.get(selected_label, 'return_5d_desc')}",
        )


def _news_price_move_ranking_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in (
        "return_1d",
        "return_5d",
        "return_20d",
        "max_up_20d",
        "max_down_20d",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("sentiment_score", "importance_score", "source_credibility_score"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    return display.rename(
        columns={
            "rank": "Rank",
            "date": "Date",
            "base_trading_date": "Base Trading Date",
            "title": "Title",
            "source": "Source",
            "source_type": "Source Type",
            "category": "Category",
            "sentiment_score": "Sentiment",
            "importance_score": "Importance",
            "source_credibility_score": "Source Credibility",
            "return_1d": "Return 1D",
            "return_5d": "Return 5D",
            "return_20d": "Return 20D",
            "max_up_20d": "Max Up 20D",
            "max_down_20d": "Max Down 20D",
            "url": "URL",
            "future_data_note": "Future Data Note",
        }
    )


def _render_category_price_move_ranking_section(
    module,
    data: pd.DataFrame,
    news_dir,
    ticker: str,
    config,
) -> None:
    st.subheader("Category Price Move Ranking")
    st.caption(
        "This groups saved news by category and ranks the price moves after those news dates. "
        "Use it to find research themes, not to infer causality."
    )
    mode_rows = module.category_price_move_ranking_mode_rows()
    mode_labels = [row["label"] for row in mode_rows]
    label_to_mode = {row["label"]: row["mode"] for row in mode_rows}
    selected_label = st.selectbox(
        "Category ranking type",
        mode_labels,
        index=0,
        key="category_price_move_ranking_type",
    )
    ranking = module.build_category_price_move_ranking(
        data,
        news_dir,
        ticker,
        ranking_mode=label_to_mode.get(selected_label, "average_return_5d_desc"),
        config=config,
    )
    if ranking.empty:
        st.info("No category-level price move ranking is available yet.")
        return

    low_count_categories = ranking[pd.to_numeric(ranking["news_count"], errors="coerce").fillna(0) < 5]
    if not low_count_categories.empty:
        st.caption(
            "Some categories have few news items. Treat category ranking as exploratory research, "
            "not a statistical conclusion."
        )
    st.dataframe(
        _category_price_move_ranking_display_frame(ranking),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Show representative news by category", expanded=False):
        for category in ranking["category"].dropna().astype(str).head(8):
            examples = module.category_price_move_examples(data, news_dir, ticker, category, limit=3)
            st.caption(f"Representative news: {category}")
            if examples.empty:
                st.info("No representative news examples are available for this category.")
                continue
            display = _news_price_move_ranking_display_frame(examples)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config=_news_url_column_config(display),
            )
            _render_news_example_price_move_charts(
                module,
                data,
                examples,
                chart_key_prefix=f"category_ranking_{category}",
            )


def _category_price_move_ranking_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    percent_columns = (
        "average_return_1d",
        "average_return_5d",
        "average_return_20d",
        "average_max_up_20d",
        "average_max_down_20d",
        "large_up_rate",
        "large_down_rate",
    )
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in (
        "average_sentiment_score",
        "average_importance_score",
        "average_source_credibility",
        "average_crash_risk",
        "average_surge_risk",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    if "news_count" in display.columns:
        display["news_count"] = display["news_count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "category": "Category",
            "news_count": "News Count",
            "average_sentiment_score": "Avg Sentiment",
            "average_importance_score": "Avg Importance",
            "average_source_credibility": "Avg Source Credibility",
            "average_return_1d": "Avg Return 1D",
            "average_return_5d": "Avg Return 5D",
            "average_return_20d": "Avg Return 20D",
            "average_max_up_20d": "Avg Max Up 20D",
            "average_max_down_20d": "Avg Max Down 20D",
            "large_up_rate": "Large Up Rate",
            "large_down_rate": "Large Down Rate",
            "average_crash_risk": "Avg Crash Risk",
            "average_surge_risk": "Avg Surge Risk",
        }
    )


def _render_category_source_type_impact_section(
    module,
    data: pd.DataFrame,
    news_dir,
    ticker: str,
    config,
) -> None:
    st.subheader("Category x Source Type Impact")
    st.caption(
        "This groups saved news by category and source type. It is exploratory and does not imply "
        "that the news caused the later price move."
    )
    mode_rows = module.category_source_type_impact_mode_rows()
    mode_labels = [row["label"] for row in mode_rows]
    label_to_mode = {row["label"]: row["mode"] for row in mode_rows}
    controls = st.columns([2, 2])
    selected_label = controls[0].selectbox(
        "Category x source ranking type",
        mode_labels,
        index=0,
        key="category_source_type_impact_ranking_type",
    )
    ranking = module.build_category_source_type_impact(
        data,
        news_dir,
        ticker,
        ranking_mode=label_to_mode.get(selected_label, "average_return_5d_desc"),
        config=config,
    )
    if ranking.empty:
        st.info("No category x source_type impact data is available yet.")
        return

    low_count = ranking[pd.to_numeric(ranking["news_count"], errors="coerce").fillna(0) < 5]
    if not low_count.empty:
        st.caption(
            "Some category x source_type combinations have few news items. Treat them as research "
            "candidates, not statistical conclusions."
        )
    st.dataframe(
        _category_source_type_impact_display_frame(ranking),
        use_container_width=True,
        hide_index=True,
    )

    metric_rows = module.category_source_type_heatmap_metric_rows()
    metric_labels = [row["label"] for row in metric_rows]
    label_to_metric = {row["label"]: row["metric"] for row in metric_rows}
    selected_metric_label = controls[1].selectbox(
        "Heatmap metric",
        metric_labels,
        index=0,
        key="category_source_type_heatmap_metric",
    )
    heatmap = module.category_source_type_heatmap_frame(
        ranking,
        metric=label_to_metric.get(selected_metric_label, "average_return_5d"),
    )
    if not heatmap.empty:
        st.plotly_chart(
            _build_category_source_type_heatmap(heatmap, title=selected_metric_label),
            use_container_width=True,
        )

    with st.expander("Show representative news by category x source type", expanded=False):
        for row in ranking.head(8).itertuples(index=False):
            category = str(getattr(row, "category", "other") or "other")
            source_type = str(getattr(row, "source_type", "unknown") or "unknown")
            examples = module.category_source_type_examples(
                data,
                news_dir,
                ticker,
                category,
                source_type,
                limit=3,
            )
            st.caption(f"Representative news: {category} x {source_type}")
            if examples.empty:
                st.info("No representative news examples are available for this combination.")
                continue
            display = _news_price_move_ranking_display_frame(examples)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config=_news_url_column_config(display),
            )
            _render_news_example_price_move_charts(
                module,
                data,
                examples,
                chart_key_prefix=f"category_source_{category}_{source_type}",
            )


def _category_source_type_impact_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    percent_columns = (
        "average_return_1d",
        "average_return_5d",
        "average_return_20d",
        "average_max_up_20d",
        "average_max_down_20d",
        "large_up_rate",
        "large_down_rate",
    )
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in (
        "average_sentiment_score",
        "average_importance_score",
        "average_source_credibility",
        "average_source_market_impact_weight",
        "average_crash_risk",
        "average_surge_risk",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    if "news_count" in display.columns:
        display["news_count"] = display["news_count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "category": "Category",
            "source_type": "Source Type",
            "news_count": "News Count",
            "average_sentiment_score": "Avg Sentiment",
            "average_importance_score": "Avg Importance",
            "average_source_credibility": "Avg Source Credibility",
            "average_source_market_impact_weight": "Avg Source Weight",
            "average_return_1d": "Avg Return 1D",
            "average_return_5d": "Avg Return 5D",
            "average_return_20d": "Avg Return 20D",
            "average_max_up_20d": "Avg Max Up 20D",
            "average_max_down_20d": "Avg Max Down 20D",
            "large_up_rate": "Large Up Rate",
            "large_down_rate": "Large Down Rate",
            "average_crash_risk": "Avg Crash Risk",
            "average_surge_risk": "Avg Surge Risk",
        }
    )


def _build_category_source_type_heatmap(frame: pd.DataFrame, *, title: str) -> go.Figure:
    z_values = frame.to_numpy(dtype=float)
    fig = go.Figure(
        data=go.Heatmap(
            z=z_values,
            x=[str(value) for value in frame.columns],
            y=[str(value) for value in frame.index],
            colorscale="RdBu",
            reversescale=True,
            colorbar={"title": title},
            hovertemplate="Category=%{y}<br>Source Type=%{x}<br>Value=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Category x Source Type: {title}",
        height=max(260, 80 + len(frame.index) * 45),
        margin={"l": 80, "r": 30, "t": 55, "b": 60},
    )
    return fig


def _render_news_examples_for_rows(
    module,
    data: pd.DataFrame,
    news_dir,
    ticker: str,
    rows: pd.DataFrame,
    *,
    dimension: str,
    label_column: str,
    title_prefix: str,
    config,
    max_groups: int = 6,
) -> None:
    if rows.empty or label_column not in rows.columns:
        return

    labels = [
        str(value)
        for value in rows[label_column].dropna().astype(str).head(max_groups).tolist()
        if str(value).strip()
    ]
    if not labels:
        return

    st.caption("Example news linked to these aggregate rows")
    for label in labels:
        examples = module.news_examples_for_impact(
            data,
            news_dir,
            ticker,
            dimension=dimension,
            value=label,
            config=config,
            limit=5,
        )
        with st.expander(f"Show example news: {title_prefix} = {label}", expanded=False):
            if examples.empty:
                st.info("No matching news examples are available for this row.")
                continue
            st.caption("These are price moves after the news date; they do not imply causality.")
            display = _news_examples_display_frame(examples)
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config=_news_url_column_config(display),
            )
            _render_news_example_price_move_charts(
                module,
                data,
                examples,
                chart_key_prefix=f"{dimension}_{label}",
            )


def _news_examples_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in ("sentiment_score", "importance_score"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    for column in ("return_1d", "return_5d", "return_20d", "max_up_20d", "max_down_20d"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "date": "Date",
            "base_trading_date": "Base Trading Date",
            "title": "Title",
            "source": "Source",
            "source_type": "Source Type",
            "category": "Category",
            "sentiment_score": "Sentiment",
            "importance_score": "Importance",
            "return_1d": "Return 1D",
            "return_5d": "Return 5D",
            "return_20d": "Return 20D",
            "max_up_20d": "Max Up 20D",
            "max_down_20d": "Max Down 20D",
            "sentiment_reason": "Sentiment Reason",
            "importance_reason": "Importance Reason",
            "url": "URL",
            "quality_note": "Quality Note",
            "future_data_note": "Future Data Note",
        }
    )


def _render_news_example_price_move_charts(
    module,
    data: pd.DataFrame,
    examples: pd.DataFrame,
    *,
    chart_key_prefix: str,
) -> None:
    st.caption("Small charts show cumulative return from the base trading date.")
    for index, row in examples.head(5).iterrows():
        news_date = row.get("date")
        chart_frame = module.build_news_price_move_chart_frame(data, news_date, pre_days=5, post_days=20)
        title = str(row.get("title") or "News item")
        base_date = row.get("base_trading_date") or "-"
        with st.container():
            st.caption(f"{title[:120]} | news date {news_date or '-'} | base trading date {base_date}")
            note = str(row.get("future_data_note") or "").strip()
            if note:
                st.caption(f"Future data note: {note}")
            if chart_frame.empty:
                st.info("Price move chart is not available for this news item.")
                continue
            st.plotly_chart(
                _build_news_example_price_move_chart(chart_frame),
                use_container_width=True,
                key=f"news_example_chart_{_safe_widget_key(chart_key_prefix)}_{index}",
            )


def _build_news_example_price_move_chart(frame: pd.DataFrame) -> go.Figure:
    chart = frame.copy()
    chart["date"] = pd.to_datetime(chart["date"], errors="coerce")
    chart["cumulative_return"] = pd.to_numeric(chart["cumulative_return"], errors="coerce")
    chart["close"] = pd.to_numeric(chart["close"], errors="coerce")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart["day_offset"],
            y=chart["cumulative_return"],
            mode="lines+markers",
            name="Cumulative return",
            customdata=chart[["date", "close"]],
            hovertemplate=(
                "Day %{x}<br>"
                "Date %{customdata[0]|%Y-%m-%d}<br>"
                "Close %{customdata[1]:.2f}<br>"
                "Return %{y:.2%}<extra></extra>"
            ),
            line={"color": "#3b7ddd", "width": 2},
            marker={"size": 5},
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#8a8f98")
    fig.add_vline(x=0, line_dash="dash", line_color="#d95f5f")
    fig.update_layout(
        height=230,
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        xaxis_title="Trading days from base date",
        yaxis_title="Cumulative return",
        yaxis_tickformat=".1%",
        showlegend=False,
    )
    return fig


def _safe_widget_key(value: object) -> str:
    text = str(value or "item")
    return "".join(character if character.isalnum() else "_" for character in text)[:80]


def _news_url_column_config(frame: pd.DataFrame) -> dict[str, object]:
    if "URL" not in frame.columns:
        return {}
    return {"URL": st.column_config.LinkColumn("URL")}


def _news_feature_relationship_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "correlation" in display.columns:
        display["correlation"] = display["correlation"].map(lambda value: _format_number(value, digits=3))
    if "valid_rows" in display.columns:
        display["valid_rows"] = display["valid_rows"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "feature": "Feature",
            "target": "Target",
            "correlation": "Correlation",
            "valid_rows": "Valid Rows",
        }
    )


def _news_impact_column_labels() -> dict[str, str]:
    return {
        "group": "Group",
        "event": "Event",
        "count": "Count",
        "average_return_1d": "Avg 1D Return",
        "average_return_5d": "Avg 5D Return",
        "average_return_20d": "Avg 20D Return",
        "up_rate_1d": "1D Up Rate",
        "large_down_rate_20d": "Large Down Rate",
        "large_up_rate_20d": "Large Up Rate",
        "average_crash_risk": "Avg Crash Risk",
        "average_surge_risk": "Avg Surge Risk",
        "average_crash_risk_change_1d": "Avg Crash Risk Change 1D",
        "average_surge_risk_change_1d": "Avg Surge Risk Change 1D",
        "average_source_credibility": "Avg Source Credibility",
        "average_source_weight": "Avg Source Weight",
    }


def _build_news_impact_return_chart(frame: pd.DataFrame) -> go.Figure:
    chart = frame.copy()
    chart["average_return_5d"] = pd.to_numeric(chart["average_return_5d"], errors="coerce")
    chart["label"] = chart["section"].astype(str) + ": " + chart["group"].astype(str)
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart["label"],
                y=chart["average_return_5d"],
                marker_color="#3b7ddd",
                text=[_format_percent(value) for value in chart["average_return_5d"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="News Buckets: Average 5D Return",
        xaxis_title="Bucket",
        yaxis_title="Average 5D Return",
        height=360,
        margin=dict(l=20, r=20, t=50, b=80),
    )
    return fig


def _build_news_event_chart(frame: pd.DataFrame) -> go.Figure:
    chart = frame.copy()
    chart["average_return_5d"] = pd.to_numeric(chart["average_return_5d"], errors="coerce")
    chart["event"] = chart["event"].astype(str).str.replace("_", " ").str.title()
    fig = go.Figure(
        data=[
            go.Bar(
                x=chart["event"],
                y=chart["average_return_5d"],
                marker_color="#6aa84f",
                text=[_format_percent(value) for value in chart["average_return_5d"]],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="News Flags / Shocks: Average 5D Return",
        xaxis_title="Event",
        yaxis_title="Average 5D Return",
        height=360,
        margin=dict(l=20, r=20, t=50, b=80),
    )
    return fig


def _today_signal_summary_value(summary: pd.DataFrame, signal: str, column: str):
    row = _today_signal_summary_row(summary, signal)
    if row is None:
        return None
    return row.get(column)


def _today_signal_summary_row(summary: pd.DataFrame, signal: str) -> pd.Series | None:
    if summary.empty or "today_signal" not in summary.columns:
        return None
    rows = summary[summary["today_signal"] == signal]
    if rows.empty:
        return None
    return rows.iloc[0]


def _today_signal_summary_display_frame(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    display = summary.copy()
    for column in (
        "average_return_1d",
        "average_return_5d",
        "average_return_20d",
        "up_rate_1d",
        "up_rate_5d",
        "max_drawdown_20d",
        "max_upside_20d",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "today_signal": "Today Signal",
            "count": "Count",
            "average_return_1d": "Avg 1D Return",
            "average_return_5d": "Avg 5D Return",
            "average_return_20d": "Avg 20D Return",
            "up_rate_1d": "1D Up Rate",
            "up_rate_5d": "5D Up Rate",
            "max_drawdown_20d": "Max Drawdown 20D",
            "max_upside_20d": "Max Upside 20D",
        }
    )


def _today_signal_improvement_display_frame(suggestions: pd.DataFrame) -> pd.DataFrame:
    if suggestions.empty:
        return suggestions
    display = suggestions.copy()
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "today_signal": "Signal",
            "count": "Count",
            "issue": "Main Issue",
            "suggestion": "Improvement Candidate",
            "config_keys": "Config to Review",
            "priority": "Priority",
        }
    )


def _today_signal_config_experiment_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in (
        "bullish_average_return_5d",
        "bullish_up_rate_5d",
        "high_risk_average_return_5d",
        "high_risk_max_drawdown_20d",
        "neutral_share",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in (
        "validation_rows",
        "bullish_count",
        "high_risk_count",
        "caution_count",
        "watch_only_count",
        "caution_watch_count",
        "neutral_count",
        "high_priority_suggestions",
        "medium_priority_suggestions",
        "improvement_suggestion_count",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "config_name": "Config",
            "validation_rows": "Rows",
            "bullish_count": "Bullish Count",
            "bullish_average_return_5d": "Bullish Avg 5D",
            "bullish_up_rate_5d": "Bullish 5D Up Rate",
            "high_risk_count": "High Risk Count",
            "high_risk_average_return_5d": "High Risk Avg 5D",
            "high_risk_max_drawdown_20d": "High Risk Max Drawdown 20D",
            "caution_count": "Caution Count",
            "watch_only_count": "Watch Only Count",
            "caution_watch_count": "Caution + Watch Count",
            "neutral_count": "Neutral Count",
            "neutral_share": "Neutral Share",
            "high_priority_suggestions": "High Suggestions",
            "medium_priority_suggestions": "Medium Suggestions",
            "improvement_suggestion_count": "Improvement Flags",
        }
    )


def _today_signal_adoption_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in (
        "bullish_average_return_5d",
        "high_risk_average_return_5d",
        "high_risk_max_drawdown_20d",
        "neutral_share",
        "caution_watch_share",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in (
        "adoption_score",
        "validation_rows",
        "improvement_suggestion_count",
        "high_priority_suggestions",
    ):
        if column in display.columns:
            digits = 2 if column == "adoption_score" else 0
            display[column] = display[column].map(lambda value, d=digits: _format_number(value, digits=d))
    return display.rename(
        columns={
            "config_name": "Config",
            "adoption_rating": "Adoption Rating",
            "adoption_score": "Score",
            "adoption_reason": "Reason",
            "validation_rows": "Rows",
            "bullish_average_return_5d": "Bullish Avg 5D",
            "high_risk_average_return_5d": "High Risk Avg 5D",
            "high_risk_max_drawdown_20d": "High Risk Max Drawdown 20D",
            "neutral_share": "Neutral Share",
            "caution_watch_share": "Caution / Watch Share",
            "improvement_suggestion_count": "Improvement Flags",
            "high_priority_suggestions": "High Priority Flags",
        }
    )


def _today_signal_config_distribution_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    if "share" in display.columns:
        display["share"] = display["share"].map(_format_percent)
    return display.rename(
        columns={
            "config_name": "Config",
            "today_signal": "Today Signal",
            "count": "Count",
            "share": "Share",
        }
    )


def _today_signal_config_changes_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in ("value", "current_value"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=3))
    return display.rename(
        columns={
            "config_name": "Config",
            "config_key": "Config Key",
            "value": "Candidate Value",
            "current_value": "Current Value",
            "change": "Change",
        }
    )


def _today_signal_validation_rows_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "date",
        "today_signal",
        "signal_confidence",
        "probability_up",
        "probability_down",
        "surge_risk_score",
        "crash_risk_score",
        "return_1d",
        "return_5d",
        "return_20d",
        "future_min_return_20d",
        "future_max_return_20d",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in (
        "probability_up",
        "probability_down",
        "return_1d",
        "return_5d",
        "return_20d",
        "future_min_return_20d",
        "future_max_return_20d",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("surge_risk_score", "crash_risk_score"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "date": "Date",
            "today_signal": "Today Signal",
            "signal_confidence": "Confidence",
            "probability_up": "Up Probability",
            "probability_down": "Down Probability",
            "surge_risk_score": "Surge Risk",
            "crash_risk_score": "Crash Risk",
            "return_1d": "1D Return",
            "return_5d": "5D Return",
            "return_20d": "20D Return",
            "future_min_return_20d": "Min Return Next 20D",
            "future_max_return_20d": "Max Return Next 20D",
        }
    )


def _build_today_signal_return_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return fig
    frame = summary.copy()
    frame["average_return_5d"] = pd.to_numeric(frame["average_return_5d"], errors="coerce")
    colors = {
        "Bullish": "#2f855a",
        "Bearish": "#d98c45",
        "Neutral": "#8a8f98",
        "Caution": "#4f78a8",
        "High Risk": "#d95f5f",
        "Watch Only": "#6d5c8d",
    }
    fig.add_trace(
        go.Bar(
            x=frame["today_signal"],
            y=frame["average_return_5d"],
            marker_color=[colors.get(signal, "#8a8f98") for signal in frame["today_signal"]],
            text=[_format_percent(value) for value in frame["average_return_5d"]],
            textposition="outside",
            hovertemplate="Signal=%{x}<br>Avg 5D return=%{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        height=320,
        yaxis_tickformat=".1%",
        xaxis_title="Today Signal",
        yaxis_title="Average 5D return",
        margin=dict(l=20, r=20, t=10, b=40),
    )
    return fig


def _build_today_signal_count_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return fig
    frame = summary.copy()
    frame["count"] = pd.to_numeric(frame["count"], errors="coerce").fillna(0)
    fig.add_trace(
        go.Bar(
            x=frame["today_signal"],
            y=frame["count"],
            marker_color="#6c8dbf",
            text=[_format_number(value, digits=0) for value in frame["count"]],
            textposition="outside",
            hovertemplate="Signal=%{x}<br>Count=%{y}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        xaxis_title="Today Signal",
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=10, b=40),
    )
    return fig


def _build_today_signal_config_distribution_chart(distribution: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if distribution.empty:
        return fig
    frame = distribution.copy()
    frame["share"] = pd.to_numeric(frame["share"], errors="coerce").fillna(0.0)
    for signal in ("Bullish", "Bearish", "Neutral", "Caution", "High Risk", "Watch Only"):
        subset = frame[frame["today_signal"] == signal]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=subset["config_name"],
                y=subset["share"],
                name=signal,
                hovertemplate="Config=%{x}<br>Share=%{y:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=360,
        yaxis_tickformat=".0%",
        xaxis_title="Config candidate",
        yaxis_title="Signal share",
        margin=dict(l=20, r=20, t=10, b=40),
        legend_title="Signal",
    )
    return fig


def _latest_risk_scores_from_validation_frame(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty or "date" not in frame.columns:
        return {}
    clean = frame.copy()
    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean = clean.dropna(subset=["date"]).sort_values("date")
    if clean.empty:
        return {}
    latest = clean.iloc[-1]
    scores: dict[str, float] = {}
    for risk_type in ("surge", "crash"):
        value = safe_float(latest.get(f"{risk_type}_risk_score"))
        if value is not None:
            scores[risk_type] = value
    return scores


def _risk_validation_summary_display_frame(frame: pd.DataFrame, *, risk_type: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    percent_columns = [
        "average_return_1d",
        "average_return_5d",
        "average_return_20d",
        "direction_rate",
        "event_rate",
    ]
    extreme_column = "max_upside_20d" if risk_type == "surge" else "max_drawdown_20d"
    percent_columns.append(extreme_column)
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("count", "multiclass_signal_days"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    if "average_score" in display.columns:
        display["average_score"] = display["average_score"].map(lambda value: _format_number(value, digits=1))
    direction_label = "Actual Up Rate" if risk_type == "surge" else "Actual Down Rate"
    event_label = "Surge Event Rate" if risk_type == "surge" else "Crash Event Rate"
    extreme_label = "Max Upside 20D" if risk_type == "surge" else "Max Drawdown 20D"
    return display.rename(
        columns={
            "bucket": "Bucket",
            "count": "Count",
            "average_score": "Avg Score",
            "average_return_1d": "Avg 1D Return",
            "average_return_5d": "Avg 5D Return",
            "average_return_20d": "Avg 20D Return",
            "direction_rate": direction_label,
            "event_rate": event_label,
            extreme_column: extreme_label,
            "multiclass_signal_days": "Saved Multiclass Days",
        }
    )


def _risk_high_days_display_frame(frame: pd.DataFrame, *, risk_type: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    score_column = f"{risk_type}_risk_score"
    bucket_column = f"{risk_type}_risk_bucket"
    columns = [
        "date",
        score_column,
        bucket_column,
        "return_1d",
        "return_5d",
        "return_20d",
        "future_min_return_20d",
        "future_max_return_20d",
        "multiclass_signal_used",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in (
        "return_1d",
        "return_5d",
        "return_20d",
        "future_min_return_20d",
        "future_max_return_20d",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if score_column in display.columns:
        display[score_column] = display[score_column].map(lambda value: _format_number(value, digits=0))
    if "multiclass_signal_used" in display.columns:
        display["multiclass_signal_used"] = display["multiclass_signal_used"].map(
            lambda value: "Yes" if bool(value) else ""
        )
    return display.rename(
        columns={
            "date": "Date",
            score_column: "Risk Score",
            bucket_column: "Bucket",
            "return_1d": "1D Return",
            "return_5d": "5D Return",
            "return_20d": "20D Return",
            "future_min_return_20d": "Min Return Next 20D",
            "future_max_return_20d": "Max Return Next 20D",
            "multiclass_signal_used": "Saved Multiclass Used",
        }
    )


def _build_risk_validation_bucket_chart(summary: pd.DataFrame, *, risk_type: str) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return fig
    color = "#2f855a" if risk_type == "surge" else "#d95f5f"
    fig.add_trace(
        go.Bar(
            x=summary["bucket"],
            y=summary["average_return_5d"],
            marker_color=color,
            text=[_format_percent(value) for value in summary["average_return_5d"]],
            textposition="outside",
            customdata=summary[["count", "event_rate"]],
            hovertemplate=(
                "Bucket=%{x}<br>"
                "Avg 5D return=%{y:.2%}<br>"
                "Count=%{customdata[0]}<br>"
                "Event rate=%{customdata[1]:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        height=280,
        yaxis_tickformat=".1%",
        xaxis_title=f"{risk_type.title()} Risk bucket",
        yaxis_title="Average 5D return",
        margin=dict(l=20, r=20, t=10, b=40),
    )
    return fig


def _build_risk_score_histogram(
    frame: pd.DataFrame,
    *,
    current_scores: dict[str, float] | None = None,
) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    current_scores = current_scores or {}
    for column, label, color in (
        ("surge_risk_score", "Surge Risk Score", "#2f855a"),
        ("crash_risk_score", "Crash Risk Score", "#d95f5f"),
    ):
        if column not in frame.columns:
            continue
        scores = pd.to_numeric(frame[column], errors="coerce").dropna()
        if scores.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=scores,
                name=label,
                marker_color=color,
                opacity=0.62,
                xbins=dict(start=0, end=100, size=5),
                hovertemplate="Score=%{x}<br>Count=%{y}<extra></extra>",
            )
        )
    for risk_type, label, color, annotation_position in (
        ("surge", "Current Surge", "#1f7a4f", "top left"),
        ("crash", "Current Crash", "#b83b3b", "top right"),
    ):
        current_value = safe_float(current_scores.get(risk_type))
        if current_value is None:
            continue
        fig.add_vline(
            x=current_value,
            line_width=2,
            line_dash="dash",
            line_color=color,
            annotation_text=f"{label}: {_format_number(current_value, digits=0)}",
            annotation_position=annotation_position,
        )
    fig.update_layout(
        barmode="overlay",
        height=280,
        xaxis=dict(title="Risk Score", range=[0, 100]),
        yaxis_title="Count",
        margin=dict(l=20, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_risk_validation_scatter(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return fig
    scatter_frame = frame.dropna(subset=["return_5d"]).copy()
    if scatter_frame.empty:
        return fig
    fig.add_trace(
        go.Scatter(
            x=scatter_frame["surge_risk_score"],
            y=scatter_frame["return_5d"],
            mode="markers",
            name="Surge Risk",
            marker=dict(color="#2f855a", size=6, opacity=0.45),
            customdata=scatter_frame[["date", "surge_risk_bucket"]],
            hovertemplate=(
                "Date=%{customdata[0]}<br>"
                "Bucket=%{customdata[1]}<br>"
                "Surge score=%{x}<br>"
                "5D return=%{y:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=scatter_frame["crash_risk_score"],
            y=scatter_frame["return_5d"],
            mode="markers",
            name="Crash Risk",
            marker=dict(color="#d95f5f", size=6, opacity=0.45),
            customdata=scatter_frame[["date", "crash_risk_bucket"]],
            hovertemplate=(
                "Date=%{customdata[0]}<br>"
                "Bucket=%{customdata[1]}<br>"
                "Crash score=%{x}<br>"
                "5D return=%{y:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        height=360,
        xaxis_title="Risk Score",
        yaxis_title="Future 5D return",
        yaxis_tickformat=".1%",
        margin=dict(l=20, r=20, t=10, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_threshold_profile_comparison(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("Threshold Profile Comparison")
    comparison = _multiclass_profile_metrics_comparison(ticker, data, settings)
    if comparison.empty:
        st.info("No threshold profile comparison is available yet.")
        return

    recommendation = _recommend_threshold_profile(comparison)
    recommended_profile = recommendation.get("profile")
    recommended_weight_mode = recommendation.get("weight_mode")
    if recommended_profile:
        st.success(
            "Recommended multiclass setting for Phase 2 review: "
            f"{recommended_profile} / {recommended_weight_mode}. "
            f"{recommendation.get('reason', '')}"
        )
    else:
        st.warning("No recommended profile yet. Train at least one multiclass profile first.")

    for message in _threshold_profile_summary_messages(comparison):
        st.write(f"- {message}")
    for message in _tail_detection_summary_messages(comparison):
        st.write(f"- {message}")

    st.plotly_chart(
        _build_threshold_profile_score_chart(comparison),
        use_container_width=True,
    )
    st.dataframe(
        _multiclass_profile_metrics_display_frame(comparison),
        use_container_width=True,
        hide_index=True,
    )
    distribution = _threshold_profile_distribution_comparison(data)
    if not distribution.empty:
        with st.expander("Class distribution by threshold profile"):
            st.dataframe(
                _threshold_profile_distribution_display_frame(distribution),
                use_container_width=True,
                hide_index=True,
            )


def _multiclass_profile_metrics_comparison(
    ticker: str,
    data: pd.DataFrame,
    settings,
) -> pd.DataFrame:
    labels = _label_distribution_module()
    rows: list[dict[str, object]] = []
    for profile in labels.threshold_profile_options():
        scheme = labels.get_threshold_profile(profile)
        distribution_summary = {}
        try:
            distribution_analysis = labels.analyze_label_distribution(data, scheme)
            distribution_summary = distribution_analysis.get("summary", {})
        except Exception as error:
            LOGGER.debug("Could not compare label distribution for %s: %s", profile, error)

        for weight_mode in labels.weight_mode_options():
            bundle = _load_multiclass_metrics(ticker, profile, weight_mode)
            metrics = bundle.get("metrics", {}) if bundle else {}
            metrics = metrics if isinstance(metrics, dict) else {}
            return_frame, _error = _multiclass_test_prediction_returns(
                ticker,
                data,
                settings,
                profile,
                weight_mode,
            )
            return_summary = _multiclass_return_summary_frame(return_frame)
            return_metrics = _profile_return_metrics(return_summary)
            rows.append(
                {
                    "profile": profile,
                    "weight_mode": weight_mode,
                    "model_name": _multiclass_model_name(profile, weight_mode),
                    "profile_role": _threshold_profile_role(profile),
                    "trained": bool(metrics),
                    "accuracy": safe_float(metrics.get("accuracy")),
                    "macro_f1": safe_float(metrics.get("macro_f1")),
                    "weighted_f1": safe_float(metrics.get("weighted_f1")),
                    "crash_precision": safe_float(
                        _multiclass_class_metric(metrics, "crash", "precision")
                    ),
                    "crash_recall": safe_float(
                        _multiclass_class_metric(metrics, "crash", "recall")
                    ),
                    "surge_precision": safe_float(
                        _multiclass_class_metric(metrics, "surge", "precision")
                    ),
                    "surge_recall": safe_float(
                        _multiclass_class_metric(metrics, "surge", "recall")
                    ),
                    "flat_precision": safe_float(
                        _multiclass_class_metric(metrics, "flat", "precision")
                    ),
                    "flat_recall": safe_float(_multiclass_class_metric(metrics, "flat", "recall")),
                    "min_class_count": safe_float(distribution_summary.get("min_class_count")),
                    "imbalance_ratio": safe_float(distribution_summary.get("imbalance_ratio")),
                    "trainability": distribution_summary.get("trainability"),
                    "predicted_crash_count": return_metrics.get("predicted_crash_count"),
                    "predicted_surge_count": return_metrics.get("predicted_surge_count"),
                    "predicted_crash_avg_return": return_metrics.get("predicted_crash_avg_return"),
                    "predicted_surge_avg_return": return_metrics.get("predicted_surge_avg_return"),
                    "predicted_flat_avg_abs_return": return_metrics.get(
                        "predicted_flat_avg_abs_return"
                    ),
                    "rows": safe_float(metrics.get("rows")),
                }
            )
    return pd.DataFrame(rows)


def _profile_return_metrics(return_summary: pd.DataFrame) -> dict[str, float | None]:
    if return_summary.empty:
        return {
            "predicted_crash_count": None,
            "predicted_surge_count": None,
            "predicted_crash_avg_return": None,
            "predicted_surge_avg_return": None,
            "predicted_flat_avg_abs_return": None,
        }

    def row_value(label: str, column: str) -> float | None:
        rows = return_summary[return_summary["predicted_label"] == label]
        if rows.empty:
            return None
        return safe_float(rows.iloc[0].get(column))

    return {
        "predicted_crash_count": row_value("crash", "count"),
        "predicted_surge_count": row_value("surge", "count"),
        "predicted_crash_avg_return": row_value("crash", "average_next_return"),
        "predicted_surge_avg_return": row_value("surge", "average_next_return"),
        "predicted_flat_avg_abs_return": row_value("flat", "average_absolute_return"),
    }


def _threshold_profile_distribution_comparison(data: pd.DataFrame) -> pd.DataFrame:
    labels = _label_distribution_module()
    rows: list[dict[str, object]] = []
    for profile in labels.threshold_profile_options():
        scheme = labels.get_threshold_profile(profile)
        try:
            analysis = labels.analyze_label_distribution(data, scheme)
        except Exception as error:
            LOGGER.debug("Could not build profile distribution table for %s: %s", profile, error)
            continue
        distribution = analysis.get("distribution")
        if not isinstance(distribution, pd.DataFrame) or distribution.empty:
            continue
        for _, item in distribution.iterrows():
            rows.append(
                {
                    "profile": profile,
                    "class": item.get("class"),
                    "count": item.get("count"),
                    "share": item.get("share"),
                }
            )
    return pd.DataFrame(rows)


def _threshold_profile_distribution_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    if "share" in display.columns:
        display["share"] = display["share"].map(_format_percent)
    return display.rename(
        columns={
            "profile": "Profile",
            "class": "Class",
            "count": "Count",
            "share": "Share",
        }
    )


def _threshold_profile_role(profile: str) -> str:
    if profile == "original_5_1":
        return "strict event definition"
    if profile == "practical_3_0_5":
        return "operational candidate"
    if profile == "quantile":
        return "research-balanced candidate"
    return "candidate"


def _multiclass_profile_metrics_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in (
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "crash_precision",
        "crash_recall",
        "surge_precision",
        "surge_recall",
        "flat_precision",
        "flat_recall",
        "predicted_crash_avg_return",
        "predicted_surge_avg_return",
        "predicted_flat_avg_abs_return",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("rows", "min_class_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    for column in ("predicted_crash_count", "predicted_surge_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    if "imbalance_ratio" in display.columns:
        display["imbalance_ratio"] = display["imbalance_ratio"].map(
            lambda value: _format_number(value, digits=2)
        )
    if "trained" in display.columns:
        display["trained"] = display["trained"].map(lambda value: "Yes" if value else "No")
    return display.rename(
        columns={
            "profile": "Profile",
            "weight_mode": "Weight Mode",
            "model_name": "Model Name",
            "profile_role": "Role",
            "trained": "Trained",
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "weighted_f1": "Weighted F1",
            "crash_precision": "Crash Precision",
            "crash_recall": "Crash Recall",
            "surge_precision": "Surge Precision",
            "surge_recall": "Surge Recall",
            "flat_precision": "Flat Precision",
            "flat_recall": "Flat Recall",
            "min_class_count": "Min Class Count",
            "imbalance_ratio": "Imbalance Ratio",
            "trainability": "Learning Ease",
            "predicted_crash_count": "Pred Crash Count",
            "predicted_surge_count": "Pred Surge Count",
            "predicted_crash_avg_return": "Pred Crash Avg Return",
            "predicted_surge_avg_return": "Pred Surge Avg Return",
            "predicted_flat_avg_abs_return": "Pred Flat Avg Abs Return",
            "rows": "Rows",
        }
    )


def _threshold_profile_summary_messages(comparison: pd.DataFrame) -> list[str]:
    if comparison.empty:
        return []
    messages: list[str] = []
    for profile in ("original_5_1", "practical_3_0_5", "quantile"):
        row = _profile_comparison_row(comparison, profile)
        if row is None:
            continue
        min_count = safe_float(row.get("min_class_count")) or 0.0
        imbalance = safe_float(row.get("imbalance_ratio"))
        macro_f1 = safe_float(row.get("macro_f1"))
        tail_recall = _tail_recall_score(row)
        if profile == "original_5_1":
            messages.append(
                "original_5_1 keeps a strict crash/surge definition, but its tail-event "
                f"sample count is low (min class count {_format_number(min_count, digits=0)})."
            )
        elif profile == "practical_3_0_5":
            messages.append(
                "practical_3_0_5 is the operational candidate: it loosens thresholds enough "
                f"to improve label balance while keeping absolute move meaning. Macro F1 is {_format_percent(macro_f1)}."
            )
        elif profile == "quantile":
            messages.append(
                "quantile is research-balanced: class balance is strongest, but crash/surge "
                "become relative tail buckets rather than absolute event definitions."
            )
        if imbalance is not None and imbalance >= 20:
            messages.append(f"{profile} has severe class imbalance ({_format_number(imbalance, digits=2)}x).")
        if tail_recall == 0:
            messages.append(f"{profile} currently has zero average crash/surge recall in saved metrics.")
    return messages


def _recommend_threshold_profile(comparison: pd.DataFrame) -> dict[str, str | None]:
    if comparison.empty:
        return {"profile": None, "weight_mode": None, "reason": None}
    candidates = comparison[comparison.get("trained", False).astype(bool)].copy()
    if candidates.empty:
        return {"profile": None, "weight_mode": None, "reason": None}

    candidates["tail_recall_score"] = candidates.apply(_tail_recall_score, axis=1)
    candidates["tail_precision_score"] = candidates.apply(_tail_precision_score, axis=1)
    candidates["tail_prediction_bonus"] = candidates.apply(_tail_prediction_bonus, axis=1)
    candidates["macro_score"] = candidates["macro_f1"].map(lambda value: safe_float(value) or 0.0)
    candidates["weighted_score"] = candidates["weighted_f1"].map(lambda value: safe_float(value) or 0.0)
    candidates["usable_label_bonus"] = candidates.apply(_profile_label_balance_bonus, axis=1)
    candidates["role_bonus"] = candidates["profile"].map(
        lambda profile: 0.05 if profile == "practical_3_0_5" else 0.03 if profile == "quantile" else 0.0
    )
    candidates["weight_bonus"] = candidates["weight_mode"].map(
        lambda mode: 0.03 if mode == "balanced" else 0.02 if mode == "manual_tail_boost" else 0.0
    )
    candidates["recommendation_score"] = (
        candidates["macro_score"] * 0.45
        + candidates["tail_recall_score"] * 0.25
        + candidates["tail_precision_score"] * 0.10
        + candidates["weighted_score"] * 0.10
        + candidates["tail_prediction_bonus"]
        + candidates["usable_label_bonus"]
        + candidates["role_bonus"]
        + candidates["weight_bonus"]
    )
    best = candidates.sort_values("recommendation_score", ascending=False).iloc[0]
    profile = str(best.get("profile"))
    weight_mode = str(best.get("weight_mode") or "none")
    if profile == "practical_3_0_5" and weight_mode != "none":
        reason = (
            "It keeps an operational threshold definition while testing rare-event weighting "
            "for crash/surge detection."
        )
    elif profile == "practical_3_0_5":
        reason = "It is the strongest operational candidate because balance improves without losing absolute threshold meaning."
    elif profile == "quantile":
        reason = "It is best as a research-balanced candidate when class balance matters more than absolute event meaning."
    else:
        reason = "It preserves the strict original event definition, but tail-event sample scarcity should be watched."
    return {"profile": profile, "weight_mode": weight_mode, "reason": reason}


def _profile_label_balance_bonus(row: pd.Series) -> float:
    min_count = safe_float(row.get("min_class_count")) or 0.0
    imbalance = safe_float(row.get("imbalance_ratio"))
    if min_count >= 100 and (imbalance is None or imbalance < 10):
        return 0.08
    if min_count >= 50 and (imbalance is None or imbalance < 20):
        return 0.04
    if min_count < 30:
        return -0.08
    return 0.0


def _tail_recall_score(row: pd.Series) -> float:
    values = [
        safe_float(row.get("crash_recall")),
        safe_float(row.get("surge_recall")),
    ]
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _tail_precision_score(row: pd.Series) -> float:
    values = [
        safe_float(row.get("crash_precision")),
        safe_float(row.get("surge_precision")),
    ]
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else 0.0


def _tail_prediction_bonus(row: pd.Series) -> float:
    crash_count = safe_float(row.get("predicted_crash_count")) or 0.0
    surge_count = safe_float(row.get("predicted_surge_count")) or 0.0
    if crash_count > 0 and surge_count > 0:
        return 0.04
    if crash_count > 0 or surge_count > 0:
        return 0.02
    return -0.04


def _tail_detection_summary_messages(comparison: pd.DataFrame) -> list[str]:
    if comparison.empty:
        return []
    trained = comparison[comparison.get("trained", False).astype(bool)].copy()
    if trained.empty:
        return ["Tail detection summary is unavailable until at least one multiclass setting is trained."]

    messages: list[str] = []
    no_tail = trained[
        (trained.get("predicted_crash_count", 0).fillna(0) <= 0)
        & (trained.get("predicted_surge_count", 0).fillna(0) <= 0)
    ]
    if len(no_tail) == len(trained):
        messages.append("All trained settings currently predict no crash/surge rows on the test split.")
    elif not no_tail.empty:
        settings = [
            f"{row.profile}/{row.weight_mode}"
            for row in no_tail[["profile", "weight_mode"]].itertuples(index=False)
        ]
        messages.append(f"These settings still predict no crash/surge rows: {', '.join(settings)}.")

    weighted = trained[trained["weight_mode"].isin(["balanced", "manual_tail_boost"])]
    if not weighted.empty:
        best_tail = weighted.assign(tail_recall=weighted.apply(_tail_recall_score, axis=1)).sort_values(
            "tail_recall",
            ascending=False,
        ).iloc[0]
        messages.append(
            "Rare-event weighting is a warning-signal experiment: "
            f"{best_tail.get('profile')}/{best_tail.get('weight_mode')} has average crash/surge recall "
            f"{_format_percent(best_tail.get('tail_recall'))}, but precision should be watched for false alarms."
        )
    else:
        messages.append(
            "Train practical_3_0_5 with balanced or manual_tail_boost to check whether crash/surge predictions appear."
        )
    return messages


def _profile_comparison_row(comparison: pd.DataFrame, profile: str) -> pd.Series | None:
    rows = comparison[comparison["profile"] == profile]
    if rows.empty:
        return None
    trained = rows[rows.get("trained", False).astype(bool)].copy()
    if trained.empty:
        return rows.iloc[0]
    trained["_summary_score"] = trained["macro_f1"].map(lambda value: safe_float(value) or 0.0)
    return trained.sort_values("_summary_score", ascending=False).iloc[0]


def _build_threshold_profile_score_chart(comparison: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if comparison.empty:
        return fig
    frame = comparison.copy()
    if "weight_mode" in frame.columns:
        frame["setting"] = frame["profile"].astype(str) + " / " + frame["weight_mode"].astype(str)
    else:
        frame["setting"] = frame["profile"].astype(str)
    metrics = [
        ("macro_f1", "Macro F1"),
        ("weighted_f1", "Weighted F1"),
        ("crash_recall", "Crash Recall"),
        ("surge_recall", "Surge Recall"),
        ("flat_recall", "Flat Recall"),
    ]
    for column, label in metrics:
        if column not in comparison.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=frame["setting"],
                y=frame[column],
                name=label,
                hovertemplate="Setting=%{x}<br>Score=%{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="group",
        height=360,
        yaxis_tickformat=".0%",
        xaxis_title="Threshold profile / weight mode",
        yaxis_title="Score",
        margin=dict(l=20, r=20, t=10, b=60),
    )
    return fig


def _multiclass_threshold_display_frame(thresholds: dict[str, object]) -> pd.DataFrame:
    rows = [
        {"Boundary": "crash <=", "Value": thresholds.get("crash_threshold")},
        {"Boundary": "down <=", "Value": thresholds.get("down_threshold")},
        {"Boundary": "up >=", "Value": thresholds.get("up_threshold")},
        {"Boundary": "surge >=", "Value": thresholds.get("surge_threshold")},
    ]
    display = pd.DataFrame(rows)
    display["Value"] = display["Value"].map(_format_percent)
    return display


def _multiclass_per_class_frame(metrics: dict[str, object]) -> pd.DataFrame:
    per_class = metrics.get("per_class", {})
    if not isinstance(per_class, dict) or not per_class:
        return pd.DataFrame()

    total = safe_float(metrics.get("rows")) or sum(
        safe_float(values.get("support")) or 0.0
        for values in per_class.values()
        if isinstance(values, dict)
    )
    rows: list[dict[str, object]] = []
    for class_id in sorted(CLASS_NAMES):
        class_name = CLASS_NAMES[class_id]
        values = per_class.get(class_name, {})
        if not isinstance(values, dict):
            values = {}
        support = safe_float(values.get("support")) or 0.0
        rows.append(
            {
                "class_order": class_id,
                "class": class_name,
                "support": support,
                "support_share": support / total if total else None,
                "precision": safe_float(values.get("precision")),
                "recall": safe_float(values.get("recall")),
                "f1": safe_float(values.get("f1")),
                "rare_event": class_name in {"crash", "surge"},
            }
        )
    return pd.DataFrame(rows).sort_values("class_order")


def _multiclass_per_class_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.drop(columns=["class_order"], errors="ignore").copy()
    for column in ("support_share", "precision", "recall", "f1"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "support" in display.columns:
        display["support"] = display["support"].map(lambda value: _format_number(value, digits=0))
    if "rare_event" in display.columns:
        display["rare_event"] = display["rare_event"].map(lambda value: "Yes" if value else "")
    return display.rename(
        columns={
            "class": "Class",
            "support": "Samples",
            "support_share": "Sample Share",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "rare_event": "Rare Event",
        }
    )


def _multiclass_confusion_frame(metrics: dict[str, object]) -> pd.DataFrame:
    matrix = metrics.get("confusion_matrix")
    labels = [CLASS_NAMES[class_id] for class_id in sorted(CLASS_NAMES)]
    if not isinstance(matrix, list) or len(matrix) != len(labels):
        return pd.DataFrame()
    try:
        return pd.DataFrame(
            matrix,
            index=[f"Actual {label}" for label in labels],
            columns=[f"Pred {label}" for label in labels],
        )
    except ValueError:
        return pd.DataFrame()


def _build_multiclass_confusion_heatmap(confusion_frame: pd.DataFrame) -> go.Figure:
    values = confusion_frame.to_numpy(dtype=float)
    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=list(confusion_frame.columns),
            y=list(confusion_frame.index),
            colorscale="Blues",
            text=values.astype(int).astype(str),
            texttemplate="%{text}",
            hovertemplate="Actual=%{y}<br>Predicted=%{x}<br>Count=%{z}<extra></extra>",
        )
    )
    fig.update_layout(
        height=430,
        xaxis_title="Predicted class",
        yaxis_title="Actual class",
        margin=dict(l=20, r=20, t=10, b=70),
    )
    return fig


def _multiclass_evaluation_summary(
    metrics: dict[str, object],
    binary_metrics_bundle: dict[str, object],
) -> list[str]:
    messages: list[str] = []
    rows = safe_float(metrics.get("rows")) or 0.0
    macro_f1 = safe_float(metrics.get("macro_f1"))
    weighted_f1 = safe_float(metrics.get("weighted_f1"))
    crash_support = safe_float(_multiclass_class_metric(metrics, "crash", "support")) or 0.0
    surge_support = safe_float(_multiclass_class_metric(metrics, "surge", "support")) or 0.0
    crash_recall = safe_float(_multiclass_class_metric(metrics, "crash", "recall"))
    surge_recall = safe_float(_multiclass_class_metric(metrics, "surge", "recall"))
    flat_recall = safe_float(_multiclass_class_metric(metrics, "flat", "recall"))

    if rows and (crash_support + surge_support) / rows < 0.05:
        messages.append(
            "Crash / surge samples are rare in the current test split, so tail-event metrics are unstable."
        )
    if macro_f1 is not None and weighted_f1 is not None and weighted_f1 - macro_f1 >= 0.10:
        messages.append(
            "Weighted F1 is much higher than Macro F1, which suggests the model is leaning on common classes."
        )
    if (crash_recall == 0 or surge_recall == 0) and (crash_support > 0 or surge_support > 0):
        messages.append("Tail-event detection is currently weak because crash or surge recall is zero.")
    if flat_recall is not None and flat_recall >= 0.80:
        messages.append("Flat is being captured often, but this can hide missed up/down/tail classes.")

    binary_metrics = binary_metrics_bundle.get("metrics", {}) if binary_metrics_bundle else {}
    binary_accuracy = safe_float(binary_metrics.get("accuracy")) if isinstance(binary_metrics, dict) else None
    multiclass_accuracy = safe_float(metrics.get("accuracy"))
    if binary_accuracy is not None and multiclass_accuracy is not None:
        if binary_accuracy > multiclass_accuracy:
            messages.append(
                "Binary remains the cleaner direction model; multiclass should be treated as tail-risk context for now."
            )
        else:
            messages.append(
                "Multiclass accuracy is competitive with binary, so its class detail may add useful context."
            )

    return messages or ["Multiclass metrics are available, but no strong pattern stands out yet."]


def _multiclass_test_prediction_returns(
    ticker: str,
    data: pd.DataFrame,
    settings,
    threshold_profile: str | None = None,
    weight_mode: str | None = None,
) -> tuple[pd.DataFrame, str | None]:
    labels = _label_distribution_module()
    active_profile = labels.normalize_threshold_profile(threshold_profile)
    active_weight_mode = labels.normalize_weight_mode(weight_mode)
    model_path = settings.paths.model_dir / (
        f"{labels.multiclass_artifact_stem(ticker, active_profile, active_weight_mode)}.joblib"
    )
    if not model_path.exists():
        return pd.DataFrame(), f"model file is missing: {model_path}"
    try:
        from stock_ai.models.predict import load_model_bundle
        from stock_ai.validation.time_split import chronological_train_valid_test_split

        bundle = load_model_bundle(model_path)
        model = bundle.get("model")
        feature_columns = list(bundle.get("feature_columns") or [])
        if model is None or not feature_columns:
            return pd.DataFrame(), "model bundle does not include model and feature columns."

        metadata = bundle.get("metadata", {}) or {}
        scheme = labels.get_threshold_profile(active_profile)
        dataset = labels.ensure_next_close_return(data)
        train_data, _valid_data, _test_data = chronological_train_valid_test_split(dataset)
        label_thresholds = metadata.get("label_thresholds")
        if not isinstance(label_thresholds, dict):
            label_thresholds = labels.resolve_label_thresholds(train_data, scheme)
        dataset = labels.add_target_for_scheme(dataset, scheme, thresholds=label_thresholds)
        _train_data, _valid_data, test_data = chronological_train_valid_test_split(dataset)

        required_columns = feature_columns + ["target_next_close_class", "next_close_return", "date"]
        missing_columns = [column for column in required_columns if column not in test_data.columns]
        if missing_columns:
            return pd.DataFrame(), f"missing columns: {', '.join(missing_columns[:5])}"

        eval_data = test_data.dropna(subset=required_columns).copy()
        eval_data = eval_data[eval_data["target_next_close_class"] >= 0]
        if eval_data.empty:
            return pd.DataFrame(), "chronological test split has no evaluable rows."

        predictions = model.predict(eval_data[feature_columns]).astype(int)
        result = eval_data[["date", "target_next_close_class", "next_close_return"]].copy()
        result["predicted_class"] = predictions
        result["actual_label"] = result["target_next_close_class"].map(CLASS_NAMES)
        result["predicted_label"] = result["predicted_class"].map(CLASS_NAMES)
        result["is_correct"] = result["target_next_close_class"].astype(int) == result["predicted_class"].astype(int)
        result["is_positive_return"] = result["next_close_return"] > 0
        result["absolute_return"] = result["next_close_return"].abs()
        return result, None
    except Exception as error:
        LOGGER.debug("Could not build multiclass return summary: %s", error, exc_info=True)
        return pd.DataFrame(), str(error)


def _multiclass_return_summary_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    summary = (
        predictions.groupby("predicted_label", dropna=False)
        .agg(
            count=("predicted_label", "size"),
            average_next_return=("next_close_return", "mean"),
            median_next_return=("next_close_return", "median"),
            average_absolute_return=("absolute_return", "mean"),
            win_rate=("is_positive_return", "mean"),
            accuracy=("is_correct", "mean"),
        )
        .reset_index()
    )
    summary["class_order"] = summary["predicted_label"].map(
        {label: class_id for class_id, label in CLASS_NAMES.items()}
    )
    return summary.sort_values("class_order")


def _multiclass_return_display_frame(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    display = summary.drop(columns=["class_order"], errors="ignore").copy()
    for column in (
        "average_next_return",
        "median_next_return",
        "average_absolute_return",
        "win_rate",
        "accuracy",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "predicted_label": "Predicted Class",
            "count": "Predictions",
            "average_next_return": "Avg Next Return",
            "median_next_return": "Median Next Return",
            "average_absolute_return": "Avg Abs Return",
            "win_rate": "Win Rate",
            "accuracy": "Accuracy",
        }
    )


def _build_multiclass_return_chart(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if summary.empty:
        return fig
    colors = {
        "crash": "#d95f5f",
        "down": "#d98c45",
        "flat": "#8a8f98",
        "up": "#68a357",
        "surge": "#2f855a",
    }
    fig.add_trace(
        go.Bar(
            x=summary["predicted_label"],
            y=summary["average_next_return"],
            marker_color=[colors.get(label, "#8a8f98") for label in summary["predicted_label"]],
            text=[_format_percent(value) for value in summary["average_next_return"]],
            textposition="outside",
            hovertemplate="Predicted=%{x}<br>Avg next return=%{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        height=320,
        yaxis_tickformat=".1%",
        xaxis_title="Predicted 5-class label",
        yaxis_title="Average next-day return",
        margin=dict(l=20, r=20, t=10, b=40),
    )
    return fig


def _render_label_distribution_analysis(
    data: pd.DataFrame,
    settings,
    threshold_profile: str | None = None,
) -> None:
    st.subheader("Label Distribution Analysis")
    st.caption(
        "This is diagnostic only. It does not change the currently trained 5-class model thresholds."
    )
    labels = _label_distribution_module()
    active_profile = labels.normalize_threshold_profile(threshold_profile)
    current_scheme = labels.get_threshold_profile(active_profile)
    try:
        current = labels.analyze_label_distribution(data, current_scheme)
    except Exception as error:
        LOGGER.debug("Could not build label distribution analysis: %s", error, exc_info=True)
        st.warning("Label distribution analysis is unavailable for the current data.")
        st.caption(str(error))
        return

    current_tab, year_tab, regime_tab, candidate_tab = st.tabs(
        ["Current Labels", "By Year", "By Regime", "Threshold Candidates"]
    )

    with current_tab:
        _render_current_label_distribution(current)

    labeled_data = current.get("labeled_data")
    if not isinstance(labeled_data, pd.DataFrame):
        labeled_data = pd.DataFrame()

    with year_tab:
        year_frame = labels.distribution_by_group(labeled_data, "year")
        _render_group_label_distribution(year_frame, group_column="year", title="Year")

    with regime_tab:
        regime_frame = labels.distribution_by_group(labeled_data, "regime")
        _render_group_label_distribution(regime_frame, group_column="regime", title="Regime")

    with candidate_tab:
        candidate_schemes = [
            labels.get_threshold_profile(profile)
            for profile in labels.threshold_profile_options()
        ]
        comparison = labels.compare_label_schemes(data, candidate_schemes)
        if comparison.empty:
            st.info("No threshold candidate comparison is available.")
        else:
            st.plotly_chart(
                _build_label_scheme_share_chart(comparison),
                use_container_width=True,
            )
            st.dataframe(
                _label_scheme_comparison_display_frame(comparison),
                use_container_width=True,
                hide_index=True,
            )


def _label_distribution_module():
    from stock_ai.labels import distribution

    return distribution


def _select_multiclass_threshold_profile(*, key: str) -> str:
    labels = _label_distribution_module()
    options = labels.threshold_profile_options()
    selected = st.selectbox(
        "Multiclass threshold profile",
        options=options,
        index=options.index(labels.DEFAULT_THRESHOLD_PROFILE),
        format_func=_threshold_profile_label,
        key=key,
    )
    st.caption(_threshold_profile_usage_text(selected))
    return selected


def _select_multiclass_weight_mode(*, key: str) -> str:
    labels = _label_distribution_module()
    options = labels.weight_mode_options()
    selected = st.selectbox(
        "Multiclass weight mode",
        options=options,
        index=options.index(labels.DEFAULT_WEIGHT_MODE),
        format_func=_weight_mode_label,
        key=key,
    )
    st.caption(_weight_mode_usage_text(selected))
    return selected


def _select_risk_multiclass_setting(*, key: str) -> tuple[str, str]:
    default_index = RISK_MULTICLASS_OPTIONS.index(DEFAULT_RISK_MULTICLASS_OPTION)
    selected = st.selectbox(
        "Risk Score multiclass signal",
        options=RISK_MULTICLASS_OPTIONS,
        index=default_index,
        format_func=lambda option: _risk_multiclass_setting_label(option[0], option[1]),
        key=key,
    )
    st.caption(
        "This setting is used only as a Surge / Crash early-warning signal. "
        "It does not replace the binary main prediction or daily_predict default."
    )
    return selected


def _select_risk_bucket_mode(*, key: str) -> str:
    risk_validation = _risk_validation_module()
    options = risk_validation.bucket_mode_options()
    selected = st.selectbox(
        "Risk bucket calibration",
        options=options,
        index=options.index(risk_validation.DEFAULT_BUCKET_MODE),
        format_func=lambda mode: _risk_bucket_mode_label(risk_validation, mode),
        key=key,
    )
    st.caption(
        "Fixed buckets use absolute score levels. Quantile buckets show where the current score sits in historical distribution."
    )
    return selected


def _threshold_profile_label(profile: str) -> str:
    labels = _label_distribution_module()
    try:
        scheme = labels.get_threshold_profile(profile)
    except ValueError:
        return str(profile)
    return f"{scheme.key} - {scheme.description}"


def _risk_multiclass_setting_label(profile: str, weight_mode: str) -> str:
    label = _risk_multiclass_setting_text(profile, weight_mode)
    if profile == "practical_3_0_5" and weight_mode == "balanced":
        return f"{label} - recommended warning signal"
    if profile == "practical_3_0_5" and weight_mode == "manual_tail_boost":
        return f"{label} - stronger tail-warning experiment"
    if profile == "quantile" and weight_mode == "balanced":
        return f"{label} - research-balanced tail bucket"
    return f"{label} - baseline"


def _risk_multiclass_setting_text(profile: str | None, weight_mode: str | None) -> str:
    active_profile = profile or "unknown_profile"
    active_weight_mode = weight_mode or "unknown_weight"
    return f"{active_profile} / {active_weight_mode} 5-class model"


def _weight_mode_label(weight_mode: str) -> str:
    if weight_mode == "none":
        return "none - no rare-event weighting"
    if weight_mode == "balanced":
        return "balanced - inverse-frequency class weighting"
    if weight_mode == "manual_tail_boost":
        return "manual_tail_boost - crash/surge warning boost"
    return str(weight_mode)


def _threshold_profile_usage_text(profile: str) -> str:
    if profile == "original_5_1":
        return "Strict event definition. Best for preserving the original crash/surge meaning, but tail samples are scarce."
    if profile == "practical_3_0_5":
        return "Operational candidate. Looser absolute thresholds improve trainability while keeping price-move meaning."
    if profile == "quantile":
        return "Research-balanced candidate. Class balance improves, but crash/surge become relative tail buckets."
    return "5-class threshold profile."


def _weight_mode_usage_text(weight_mode: str) -> str:
    if weight_mode == "none":
        return "Baseline setting. Keeps existing multiclass behavior and model names unchanged."
    if weight_mode == "balanced":
        return "Uses class-frequency balancing. This may improve rare-class recall, with more false alarms possible."
    if weight_mode == "manual_tail_boost":
        return "Boosts crash/surge manually for warning-signal research. Expect recall/precision trade-offs."
    return "Multiclass class weighting mode."


def _multiclass_model_name(
    profile: str | None = None,
    weight_mode: str | None = None,
) -> str:
    return _label_distribution_module().multiclass_model_name(profile, weight_mode)


def _render_current_label_distribution(analysis: dict[str, object]) -> None:
    distribution = analysis.get("distribution")
    summary = analysis.get("summary")
    if not isinstance(distribution, pd.DataFrame) or distribution.empty:
        st.info("No label distribution is available.")
        return
    if not isinstance(summary, dict):
        summary = {}

    metric_cols = st.columns(4)
    metric_cols[0].metric("Labeled rows", _format_number(summary.get("total"), digits=0))
    metric_cols[1].metric(
        "Min class count",
        _format_number(summary.get("min_class_count"), digits=0),
    )
    metric_cols[2].metric(
        "Imbalance ratio",
        _format_number(summary.get("imbalance_ratio"), digits=2),
    )
    metric_cols[3].metric("Learning ease", str(summary.get("trainability") or "-"))

    warnings = summary.get("warnings") or []
    if warnings:
        st.warning(" ".join(str(message) for message in warnings))

    st.plotly_chart(_build_label_distribution_chart(distribution), use_container_width=True)
    st.dataframe(
        _label_distribution_display_frame(distribution),
        use_container_width=True,
        hide_index=True,
    )


def _render_group_label_distribution(
    frame: pd.DataFrame,
    *,
    group_column: str,
    title: str,
) -> None:
    if frame.empty:
        st.info(f"No {title} label distribution is available.")
        return
    st.plotly_chart(
        _build_group_label_share_chart(frame, group_column=group_column, title=title),
        use_container_width=True,
    )
    st.dataframe(
        _group_label_distribution_display_frame(frame, group_column=group_column),
        use_container_width=True,
        hide_index=True,
    )


def _label_distribution_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "share" in display.columns:
        display["share"] = display["share"].map(_format_percent)
    if "count" in display.columns:
        display["count"] = display["count"].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "class_id": "Class ID",
            "class": "Class",
            "count": "Count",
            "share": "Share",
        }
    )


def _group_label_distribution_display_frame(
    frame: pd.DataFrame,
    *,
    group_column: str,
) -> pd.DataFrame:
    display = frame.copy()
    for class_name in _ordered_class_names():
        share_column = f"{class_name}_share"
        if share_column in display.columns:
            display[share_column] = display[share_column].map(_format_percent)
    return display.rename(
        columns={
            group_column: group_column.title(),
            **{class_name: f"{class_name} count" for class_name in _ordered_class_names()},
            **{
                f"{class_name}_share": f"{class_name} share"
                for class_name in _ordered_class_names()
            },
            "total": "Total",
        }
    )


def _label_scheme_comparison_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for class_name in _ordered_class_names():
        share_column = f"{class_name}_share"
        count_column = f"{class_name}_count"
        if share_column in display.columns:
            display[share_column] = display[share_column].map(_format_percent)
        if count_column in display.columns:
            display[count_column] = display[count_column].map(
                lambda value: _format_number(value, digits=0)
            )
    for column in ("total", "min_class_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    if "imbalance_ratio" in display.columns:
        display["imbalance_ratio"] = display["imbalance_ratio"].map(
            lambda value: _format_number(value, digits=2)
        )
    return display.rename(
        columns={
            "scheme": "Scheme",
            "description": "Rule",
            "trainability": "Learning Ease",
            "total": "Total",
            "min_class_count": "Min Class Count",
            "imbalance_ratio": "Imbalance Ratio",
            **{
                f"{class_name}_count": f"{class_name} count"
                for class_name in _ordered_class_names()
            },
            **{
                f"{class_name}_share": f"{class_name} share"
                for class_name in _ordered_class_names()
            },
        }
    )


def _build_label_distribution_chart(frame: pd.DataFrame) -> go.Figure:
    colors = _class_color_map()
    fig = go.Figure(
        go.Bar(
            x=frame["class"],
            y=frame["count"],
            marker_color=[colors.get(label, "#8a8f98") for label in frame["class"]],
            text=[_format_percent(value) for value in frame["share"]],
            textposition="outside",
            hovertemplate="Class=%{x}<br>Count=%{y}<br>Share=%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        xaxis_title="5-class label",
        yaxis_title="Rows",
        margin=dict(l=20, r=20, t=10, b=40),
    )
    return fig


def _build_group_label_share_chart(
    frame: pd.DataFrame,
    *,
    group_column: str,
    title: str,
) -> go.Figure:
    fig = go.Figure()
    colors = _class_color_map()
    x_values = frame[group_column].astype(str)
    for class_name in _ordered_class_names():
        share_column = f"{class_name}_share"
        if share_column not in frame.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=x_values,
                y=frame[share_column],
                name=class_name,
                marker_color=colors.get(class_name, "#8a8f98"),
                hovertemplate=(
                    f"{title}=%{{x}}<br>Class={class_name}"
                    "<br>Share=%{y:.2%}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        height=360,
        yaxis_tickformat=".0%",
        xaxis_title=title,
        yaxis_title="Label share",
        margin=dict(l=20, r=20, t=10, b=60),
    )
    return fig


def _build_label_scheme_share_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    colors = _class_color_map()
    for class_name in _ordered_class_names():
        share_column = f"{class_name}_share"
        if share_column not in frame.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=frame["scheme"],
                y=frame[share_column],
                name=class_name,
                marker_color=colors.get(class_name, "#8a8f98"),
                hovertemplate="Scheme=%{x}<br>Share=%{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack",
        height=340,
        yaxis_tickformat=".0%",
        xaxis_title="Threshold candidate",
        yaxis_title="Class share",
        margin=dict(l=20, r=20, t=10, b=80),
    )
    return fig


def _ordered_class_names() -> list[str]:
    return [CLASS_NAMES[class_id] for class_id in sorted(CLASS_NAMES)]


def _class_color_map() -> dict[str, str]:
    return {
        "crash": "#d95f5f",
        "down": "#d98c45",
        "flat": "#8a8f98",
        "up": "#68a357",
        "surge": "#2f855a",
    }


def _render_evaluation_overview(metrics: dict[str, object]) -> None:
    st.subheader("Evaluation Overview")
    cols = st.columns(5)
    cols[0].metric("Rows", _format_number(metrics.get("count"), digits=0))
    cols[1].metric("Accuracy", _format_percent(metrics.get("accuracy")))
    cols[2].metric("F1", _format_percent(metrics.get("f1")))
    cols[3].metric("Precision", _format_percent(metrics.get("precision")))
    cols[4].metric("Recall", _format_percent(metrics.get("recall")))

    return_cols = st.columns(5)
    return_cols[0].metric("AUC", _format_percent(metrics.get("auc")))
    return_cols[1].metric("Avg Return", _format_percent(metrics.get("average_return")))
    return_cols[2].metric("Cum Return", _format_percent(metrics.get("cumulative_return")))
    return_cols[3].metric("Win Rate", _format_percent(metrics.get("win_rate")))
    return_cols[4].metric("Max Drawdown", _format_percent(metrics.get("max_drawdown")))


def _evaluation_summary_display_frame(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    display = summary.copy()
    for column in (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "average_return",
        "cumulative_return",
        "win_rate",
        "max_drawdown",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "year": "Year",
            "month": "Month",
            "regime": "Regime",
            "surge_risk_level": "Surge Risk",
            "crash_risk_level": "Crash Risk",
            "anomaly_type": "Anomaly Type",
            "dominant_narrative": "Narrative",
            "earnings_event_state": "Earnings Event",
            "news_sentiment_bucket": "News Sentiment",
            "confidence_bucket": "Confidence",
            "count": "Count",
            "accuracy": "Accuracy",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
            "auc": "AUC",
            "average_return": "Avg Return",
            "cumulative_return": "Cum Return",
            "win_rate": "Win Rate",
            "max_drawdown": "Max Drawdown",
        }
    )


def _build_evaluation_bar_chart(
    summary: pd.DataFrame,
    group_column: str,
    label: str,
) -> go.Figure:
    chart = summary.copy()
    chart = chart.sort_values("accuracy", ascending=True).tail(15)
    fig = go.Figure(
        go.Bar(
            x=chart["accuracy"],
            y=chart[group_column].astype(str),
            orientation="h",
            marker_color="#6c8dbf",
            customdata=chart[["count", "cumulative_return"]],
            hovertemplate=(
                f"{label}=%{{y}}<br>"
                "Accuracy=%{x:.2%}<br>"
                "Rows=%{customdata[0]}<br>"
                "Cum Return=%{customdata[1]:.2%}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=max(320, 60 + len(chart) * 30),
        xaxis_title="Accuracy",
        xaxis_tickformat=".0%",
        yaxis_title=label,
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _build_monthly_heatmap(eval_frame: pd.DataFrame, value_column: str, title: str) -> go.Figure:
    rows = eval_frame.copy()
    rows["year"] = rows["date"].dt.year
    rows["month_num"] = rows["date"].dt.month
    if value_column == "is_correct":
        rows[value_column] = rows["predicted_class"] == rows["actual_class"]
        values = rows.pivot_table(index="year", columns="month_num", values=value_column, aggfunc="mean")
        colorbar_title = "Accuracy"
    else:
        values = rows.pivot_table(index="year", columns="month_num", values=value_column, aggfunc=lambda x: (1 + x.fillna(0.0)).prod() - 1)
        colorbar_title = "Return"
    values = values.reindex(columns=list(range(1, 13)))
    fig = go.Figure(
        go.Heatmap(
            z=values.values,
            x=[str(month) for month in values.columns],
            y=[str(year) for year in values.index],
            zmid=0 if value_column != "is_correct" else None,
            colorscale="RdBu" if value_column != "is_correct" else "Blues",
            colorbar=dict(title=colorbar_title, tickformat=".0%"),
            hovertemplate="Year=%{y}<br>Month=%{x}<br>Value=%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=max(280, 70 + len(values.index) * 28),
        xaxis_title="Month",
        yaxis_title="Year",
        margin=dict(l=20, r=20, t=45, b=30),
    )
    return fig


def _build_confidence_evaluation_chart(confidence_summary: pd.DataFrame) -> go.Figure:
    frame = confidence_summary.copy().sort_values("confidence_bucket")
    fig = go.Figure(
        go.Bar(
            x=frame["confidence_bucket"].astype(str),
            y=frame["accuracy"],
            marker_color="#3a8f6b",
            customdata=frame[["count"]],
            hovertemplate="Confidence=%{x}<br>Accuracy=%{y:.2%}<br>Rows=%{customdata[0]}<extra></extra>",
        )
    )
    fig.update_layout(
        height=300,
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        xaxis_title="Confidence bucket",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _render_model_compare_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    compare = _model_compare_module()
    _render_model_evaluation_overview(ticker, data, section="Model Compare")
    st.caption(
        "Compares several binary models on the same chronological train / validation / test split."
    )
    st.caption(
        "Random splits are not used. Dummy Classifier is included to check whether ML adds value."
    )

    if data.empty or "date" not in data.columns:
        st.info("Model comparison needs a feature dataset with dates.")
        return

    controls = st.columns(4)
    threshold = controls[0].slider(
        "Buy threshold",
        0.50,
        0.90,
        0.55,
        0.01,
        key="model_compare_buy_threshold",
    )
    train_ratio = controls[1].slider(
        "Train ratio",
        0.50,
        0.80,
        0.70,
        0.05,
        key="model_compare_train_ratio",
    )
    valid_ratio = controls[2].slider(
        "Validation ratio",
        0.10,
        0.30,
        0.15,
        0.05,
        key="model_compare_validation_ratio",
    )
    include_xgboost = controls[3].checkbox(
        "Include XGBoost",
        value=True,
        key="model_compare_include_xgboost",
    )
    if train_ratio + valid_ratio >= 0.95:
        st.warning("Train + validation ratio is too high. Leave enough rows for the test period.")
        return

    result_key = f"model_compare_{ticker}"
    config_key = f"model_compare_config_{ticker}"
    if st.button("Run model comparison", type="primary", key="model_compare_run_button"):
        anomaly_enriched, _, _, _ = _anomaly_context(data)
        with st.spinner("Comparing models on the same time split..."):
            try:
                result = compare.run_model_comparison(
                    anomaly_enriched,
                    ticker=ticker,
                    probability_threshold=float(threshold),
                    train_ratio=float(train_ratio),
                    valid_ratio=float(valid_ratio),
                    include_xgboost=bool(include_xgboost),
                )
            except Exception as error:
                st.error(f"Model comparison failed: {error}")
                return
        st.session_state[result_key] = result
        st.session_state[config_key] = {
            "threshold": threshold,
            "train_ratio": train_ratio,
            "valid_ratio": valid_ratio,
            "include_xgboost": include_xgboost,
        }

    result = st.session_state.get(result_key)
    if result is None:
        st.info("Set the split and threshold, then run the model comparison.")
        return
    if result.predictions.empty or result.score_table.empty:
        st.warning("Model comparison produced no usable test predictions.")
        if not result.skipped_models.empty:
            st.dataframe(result.skipped_models, use_container_width=True, hide_index=True)
        return

    with st.expander("Current Model Summary", expanded=True):
        _render_model_compare_split_summary(result.split_summary, len(result.feature_columns))
        _render_model_compare_takeaways(result.score_table)

    with st.expander("Model Compare", expanded=True):
        st.caption("Model scores use the same chronological split and feature set.")
        st.subheader("Model Scores")
        st.dataframe(
            _model_compare_score_display_frame(result.score_table),
            use_container_width=True,
            hide_index=True,
        )

        if not result.skipped_models.empty:
            with st.expander("Skipped models"):
                st.dataframe(result.skipped_models, use_container_width=True, hide_index=True)

        ranking_metrics = list(result.metric_rankings["metric"].dropna().unique())
        selected_metric = st.selectbox(
            "Ranking metric",
            options=ranking_metrics,
            index=ranking_metrics.index("accuracy") if "accuracy" in ranking_metrics else 0,
            key="model_compare_ranking_metric_select",
        )
        st.plotly_chart(
            _build_model_compare_ranking_chart(result.metric_rankings, selected_metric),
            use_container_width=True,
        )

    with st.expander("Backtest Details", expanded=True):
        st.subheader("Cumulative Return")
        st.plotly_chart(
            _build_model_compare_return_chart(result.predictions),
            use_container_width=True,
        )

        detail_cols = st.columns(2)
        model_options = list(result.score_table["model_name"])
        selected_model = detail_cols[0].selectbox(
            "Confusion Matrix Model",
            options=model_options,
            key="model_compare_confusion_model_select",
        )
        with detail_cols[0]:
            st.dataframe(
                _model_compare_confusion_frame(result.score_table, selected_model),
                use_container_width=True,
                hide_index=True,
            )
        with detail_cols[1]:
            st.dataframe(
                compare.model_characteristics_frame(),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Condition-level details"):
        st.subheader("Condition Strengths")
        condition_options = {
            "Regime": "regime",
            "Surge Risk": "surge_risk_level",
            "Crash Risk": "crash_risk_level",
            "Anomaly Type": "anomaly_type",
            "Narrative": "dominant_narrative",
            "Earnings Event": "earnings_event_state",
            "News Sentiment": "news_sentiment_bucket",
            "Confidence": "confidence_bucket",
        }
        condition_cols = st.columns(3)
        condition_label = condition_cols[0].selectbox(
            "Condition",
            options=list(condition_options),
            key="model_compare_condition_select",
        )
        condition_metric = condition_cols[1].selectbox(
            "Metric",
            options=["accuracy", "f1", "precision", "recall", "auc", "cumulative_return", "win_rate"],
            key="model_compare_condition_metric_select",
        )
        min_count = condition_cols[2].number_input(
            "Minimum rows",
            1,
            100,
            5,
            1,
            key="model_compare_condition_min_rows",
        )
        condition_key = condition_options[condition_label]
        condition_table = compare.compare_condition_performance(
            result.predictions,
            group_column=condition_key,
            metric=condition_metric,
            min_count=int(min_count),
        )
        if condition_table.empty:
            st.info("No condition-level comparison is available for this slice yet.")
        else:
            st.plotly_chart(
                _build_condition_compare_chart(
                    condition_table,
                    group_column=condition_key,
                    metric=condition_metric,
                    label=condition_label,
                ),
                use_container_width=True,
            )
            st.dataframe(
                _condition_compare_display_frame(condition_table),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Raw metrics / details"):
        st.subheader("Test Predictions")
        model_options = list(result.score_table["model_name"])
        selected_prediction_model = st.selectbox(
            "Prediction rows",
            options=model_options,
            key=f"model_compare_prediction_rows_model_{ticker}",
        )
        prediction_rows = result.predictions[result.predictions["model_name"] == selected_prediction_model]
        st.dataframe(
            _model_compare_predictions_display_frame(prediction_rows.tail(100)),
            use_container_width=True,
            hide_index=True,
        )


def _render_model_compare_split_summary(split_summary: dict[str, object], feature_count: int) -> None:
    st.subheader("Split Summary")
    cols = st.columns(4)
    cols[0].metric("Features", _format_number(feature_count, digits=0))
    cols[1].metric("Train rows", _format_number(split_summary.get("train_rows"), digits=0))
    cols[2].metric("Validation rows", _format_number(split_summary.get("valid_rows"), digits=0))
    cols[3].metric("Test rows", _format_number(split_summary.get("test_rows"), digits=0))

    period = pd.DataFrame(
        [
            {
                "Split": "Train",
                "Start": split_summary.get("train_start_date"),
                "End": split_summary.get("train_end_date"),
            },
            {
                "Split": "Validation",
                "Start": split_summary.get("valid_start_date"),
                "End": split_summary.get("valid_end_date"),
            },
            {
                "Split": "Test",
                "Start": split_summary.get("test_start_date"),
                "End": split_summary.get("test_end_date"),
            },
        ]
    )
    for column in ("Start", "End"):
        period[column] = pd.to_datetime(period[column], errors="coerce").dt.strftime("%Y-%m-%d")
    st.dataframe(period, use_container_width=True, hide_index=True)


def _render_model_compare_takeaways(score_table: pd.DataFrame) -> None:
    st.subheader("Takeaways")
    messages = _model_compare_takeaways(score_table)
    for message in messages:
        st.write(f"- {message}")


def _model_compare_takeaways(score_table: pd.DataFrame) -> list[str]:
    messages: list[str] = []
    for metric, label in (
        ("accuracy", "accuracy"),
        ("f1", "F1"),
        ("cumulative_return", "simulated return"),
    ):
        best = _best_model_row(score_table, metric)
        if best is not None:
            messages.append(
                f"{best.get('model_name')} is currently strongest by {label} "
                f"({_format_model_compare_metric(metric, best.get(metric))})."
            )

    dummy = score_table[score_table["model_name"] == "Dummy Classifier"]
    best_accuracy = _best_model_row(score_table, "accuracy")
    if not dummy.empty and best_accuracy is not None:
        dummy_accuracy = safe_float(dummy.iloc[0].get("accuracy"))
        best_accuracy_value = safe_float(best_accuracy.get("accuracy"))
        if dummy_accuracy is not None and best_accuracy_value is not None:
            edge = best_accuracy_value - dummy_accuracy
            messages.append(
                "Best model accuracy is "
                f"{_format_percent(edge)} above Dummy Classifier."
            )
    return messages[:4] or ["No stable comparison signal is available yet."]


def _best_model_row(score_table: pd.DataFrame, metric: str):
    if score_table.empty or metric not in score_table.columns:
        return None
    rows = score_table.copy()
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    rows = rows.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if rows.empty:
        return None
    return rows.iloc[0]


def _format_model_compare_metric(metric: str, value) -> str:
    if metric in {
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "up_prediction_accuracy",
        "down_prediction_accuracy",
        "cumulative_return",
        "buy_hold_return",
        "max_drawdown",
        "win_rate",
    }:
        return _format_percent(value)
    return _format_number(value, digits=0)


def _model_compare_score_display_frame(score_table: pd.DataFrame) -> pd.DataFrame:
    if score_table.empty:
        return score_table
    columns = [
        "model_name",
        "rows",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "up_prediction_accuracy",
        "down_prediction_accuracy",
        "cumulative_return",
        "buy_hold_return",
        "max_drawdown",
        "win_rate",
        "trade_count",
    ]
    display = score_table[[column for column in columns if column in score_table.columns]].copy()
    for column in (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "up_prediction_accuracy",
        "down_prediction_accuracy",
        "cumulative_return",
        "buy_hold_return",
        "max_drawdown",
        "win_rate",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("rows", "trade_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "model_name": "Model",
            "rows": "Rows",
            "accuracy": "Accuracy",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
            "auc": "AUC",
            "up_prediction_accuracy": "Up Pred Acc",
            "down_prediction_accuracy": "Down Pred Acc",
            "cumulative_return": "Strategy Return",
            "buy_hold_return": "Buy & Hold",
            "max_drawdown": "Max Drawdown",
            "win_rate": "Win Rate",
            "trade_count": "Trades",
        }
    )


def _build_model_compare_ranking_chart(rankings: pd.DataFrame, metric: str) -> go.Figure:
    rows = rankings[rankings["metric"] == metric].copy()
    rows = rows.sort_values("value", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=rows["value"],
            y=rows["model_name"],
            orientation="h",
            marker_color="#4f78a8",
            text=rows["value"].map(lambda value: _format_model_compare_metric(metric, value)),
            textposition="auto",
            hovertemplate="Model=%{y}<br>Value=%{x}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(280, 80 + len(rows) * 42),
        xaxis_title=metric,
        yaxis_title="Model",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    if metric != "trade_count":
        fig.update_xaxes(tickformat=".0%")
    return fig


def _build_model_compare_return_chart(predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if predictions.empty:
        return fig
    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    for model_name, model_rows in rows.groupby("model_name", dropna=False):
        model_rows = model_rows.sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=model_rows["date"],
                y=model_rows["strategy_cumulative_return"],
                mode="lines",
                name=str(model_name),
            )
        )
    buy_hold = rows.sort_values("date").drop_duplicates("date")
    fig.add_trace(
        go.Scatter(
            x=buy_hold["date"],
            y=buy_hold["buy_hold_cumulative_return"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#555555", width=2, dash="dot"),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        height=380,
        yaxis_title="Cumulative return",
        yaxis_tickformat=".1%",
        margin=dict(l=20, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _model_compare_confusion_frame(score_table: pd.DataFrame, model_name: str) -> pd.DataFrame:
    row = score_table[score_table["model_name"] == model_name]
    if row.empty:
        return pd.DataFrame()
    confusion = row.iloc[0].get("confusion_matrix", {}) or {}
    if not isinstance(confusion, dict):
        confusion = {}
    return pd.DataFrame(
        [
            {
                "Actual": "Down",
                "Predicted Down": confusion.get("tn", 0),
                "Predicted Up": confusion.get("fp", 0),
            },
            {
                "Actual": "Up",
                "Predicted Down": confusion.get("fn", 0),
                "Predicted Up": confusion.get("tp", 0),
            },
        ]
    )


def _build_condition_compare_chart(
    table: pd.DataFrame,
    group_column: str,
    metric: str,
    label: str,
) -> go.Figure:
    fig = go.Figure()
    if table.empty:
        return fig
    rows = table.copy()
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    rows = rows.dropna(subset=[metric])
    top_groups = rows.groupby(group_column)["count"].sum().sort_values(ascending=False).head(8).index
    rows = rows[rows[group_column].isin(top_groups)]
    for model_name, model_rows in rows.groupby("model_name", dropna=False):
        model_rows = model_rows.sort_values(group_column)
        fig.add_trace(
            go.Bar(
                x=model_rows[group_column].astype(str),
                y=model_rows[metric],
                name=str(model_name),
                customdata=model_rows[["count"]],
                hovertemplate=(
                    f"{label}=%{{x}}<br>"
                    f"{metric}=%{{y:.2%}}<br>"
                    "Rows=%{customdata[0]}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="group",
        height=380,
        xaxis_title=label,
        yaxis_title=metric,
        yaxis_tickformat=".0%",
        margin=dict(l=20, r=20, t=10, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _condition_compare_display_frame(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table
    display = table.copy()
    for column in (
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "average_return",
        "cumulative_return",
        "win_rate",
        "max_drawdown",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "model_name": "Model",
            "regime": "Regime",
            "surge_risk_level": "Surge Risk",
            "crash_risk_level": "Crash Risk",
            "anomaly_type": "Anomaly Type",
            "dominant_narrative": "Narrative",
            "earnings_event_state": "Earnings Event",
            "news_sentiment_bucket": "News Sentiment",
            "confidence_bucket": "Confidence",
            "count": "Rows",
            "accuracy": "Accuracy",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
            "auc": "AUC",
            "average_return": "Avg Return",
            "cumulative_return": "Cum Return",
            "win_rate": "Win Rate",
            "max_drawdown": "Max Drawdown",
        }
    )


def _model_compare_predictions_display_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    display = _backtest_predictions_display_frame(predictions)
    if "Model" not in display.columns and "model_name" in predictions.columns:
        display.insert(0, "Model", predictions["model_name"].values)
    return display


def _render_ablation_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    ablation = _ablation_module()
    _render_model_evaluation_overview(ticker, data, section="Ablation Analysis")
    st.caption(
        "Feature-group ablation checks which information blocks help prediction and simulated returns."
    )
    st.caption(
        "Each configuration uses the same chronological split and the same model type."
    )

    if data.empty or "date" not in data.columns:
        st.info("Ablation needs a feature dataset with dates.")
        return

    controls = st.columns(4)
    model_name = controls[0].selectbox(
        "Model",
        options=["LightGBM", "Random Forest", "Logistic Regression"],
        key="ablation_model_select",
    )
    threshold = controls[1].slider(
        "Buy threshold",
        0.50,
        0.90,
        0.55,
        0.01,
        key="ablation_buy_threshold",
    )
    train_ratio = controls[2].slider(
        "Train ratio",
        0.50,
        0.80,
        0.70,
        0.05,
        key="ablation_train_ratio",
    )
    valid_ratio = controls[3].slider(
        "Validation ratio",
        0.10,
        0.30,
        0.15,
        0.05,
        key="ablation_validation_ratio",
    )
    if train_ratio + valid_ratio >= 0.95:
        st.warning("Train + validation ratio is too high. Leave enough rows for the test period.")
        return

    result_key = f"feature_ablation_{ticker}"
    if st.button("Run ablation", type="primary", key="ablation_run_button"):
        anomaly_enriched, _, _, _ = _anomaly_context(data)
        with st.spinner("Running feature-group ablation..."):
            try:
                result = ablation.run_feature_ablation(
                    anomaly_enriched,
                    ticker=ticker,
                    model_name=model_name,
                    probability_threshold=float(threshold),
                    train_ratio=float(train_ratio),
                    valid_ratio=float(valid_ratio),
                )
            except Exception as error:
                st.error(f"Ablation failed: {error}")
                return
        st.session_state[result_key] = result

    result = st.session_state.get(result_key)
    if result is None:
        st.info("Set the model and split, then run the ablation.")
        return
    if result.score_table.empty:
        st.warning("Ablation produced no usable results.")
        if not result.skipped_configs.empty:
            st.dataframe(result.skipped_configs, use_container_width=True, hide_index=True)
        return

    with st.expander("Current Model Summary", expanded=True):
        _render_model_compare_split_summary(
            result.split_summary,
            int(result.score_table["feature_count"].max()),
        )
        _render_ablation_feature_groups(result.feature_groups)

        st.subheader("Useful Feature Groups")
        for message in result.summary_messages:
            st.write(f"- {message}")

    with st.expander("Ablation Analysis", expanded=True):
        st.caption("Compare full features, limited feature sets, and one-group-removed variants.")
        st.subheader("Ablation Scores")
        st.dataframe(
            _ablation_score_display_frame(result.score_table),
            use_container_width=True,
            hide_index=True,
        )

        metric_options = [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "auc",
            "up_prediction_accuracy",
            "down_prediction_accuracy",
            "cumulative_return",
            "win_rate",
        ]
        metric = st.selectbox(
            "Ablation metric",
            options=metric_options,
            index=metric_options.index("f1"),
            key="ablation_metric_select",
        )
        chart_cols = st.columns(2)
        with chart_cols[0]:
            st.plotly_chart(
                _build_ablation_delta_chart(result.score_table, metric),
                use_container_width=True,
            )
        with chart_cols[1]:
            st.plotly_chart(
                _build_ablation_heatmap(result.score_table),
                use_container_width=True,
            )

    with st.expander("Feature group impact"):
        st.subheader("Group Impact")
        if result.group_impact.empty:
            st.info("Group impact is not available for the current result.")
        else:
            st.plotly_chart(
                _build_ablation_group_impact_chart(result.group_impact),
                use_container_width=True,
            )
            st.dataframe(
                _ablation_group_impact_display_frame(result.group_impact),
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("Backtest Details"):
        if result.predictions.empty:
            st.info("No return simulation rows are available for this ablation result.")
        else:
            st.subheader("Cumulative Return by Feature Set")
            st.plotly_chart(
                _build_ablation_return_chart(result.predictions),
                use_container_width=True,
            )

    if not result.skipped_configs.empty:
        with st.expander("Raw metrics / skipped configurations"):
            st.dataframe(result.skipped_configs, use_container_width=True, hide_index=True)


def _render_ablation_feature_groups(feature_groups: dict[str, list[str]]) -> None:
    rows = [
        {
            "Feature Group": group,
            "Feature Count": len(columns),
            "Examples": ", ".join(columns[:6]),
        }
        for group, columns in feature_groups.items()
        if columns
    ]
    if not rows:
        return
    with st.expander("Detected feature groups"):
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _ablation_score_display_frame(score_table: pd.DataFrame) -> pd.DataFrame:
    if score_table.empty:
        return score_table
    columns = [
        "config_label",
        "feature_count",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "up_prediction_accuracy",
        "down_prediction_accuracy",
        "cumulative_return",
        "buy_hold_return",
        "max_drawdown",
        "win_rate",
        "trade_count",
        "f1_delta_vs_full",
        "auc_delta_vs_full",
        "cumulative_return_delta_vs_full",
    ]
    display = score_table[[column for column in columns if column in score_table.columns]].copy()
    percent_columns = [
        "accuracy",
        "f1",
        "precision",
        "recall",
        "auc",
        "up_prediction_accuracy",
        "down_prediction_accuracy",
        "cumulative_return",
        "buy_hold_return",
        "max_drawdown",
        "win_rate",
        "f1_delta_vs_full",
        "auc_delta_vs_full",
        "cumulative_return_delta_vs_full",
    ]
    for column in percent_columns:
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("feature_count", "trade_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    return display.rename(
        columns={
            "config_label": "Feature Set",
            "feature_count": "Features",
            "accuracy": "Accuracy",
            "f1": "F1",
            "precision": "Precision",
            "recall": "Recall",
            "auc": "AUC",
            "up_prediction_accuracy": "Up Pred Acc",
            "down_prediction_accuracy": "Down Pred Acc",
            "cumulative_return": "Strategy Return",
            "buy_hold_return": "Buy & Hold",
            "max_drawdown": "Max Drawdown",
            "win_rate": "Win Rate",
            "trade_count": "Trades",
            "f1_delta_vs_full": "F1 vs Full",
            "auc_delta_vs_full": "AUC vs Full",
            "cumulative_return_delta_vs_full": "Return vs Full",
        }
    )


def _build_ablation_delta_chart(score_table: pd.DataFrame, metric: str) -> go.Figure:
    delta_column = f"{metric}_delta_vs_full"
    fig = go.Figure()
    if score_table.empty or delta_column not in score_table.columns:
        return fig
    rows = score_table.copy()
    rows[delta_column] = pd.to_numeric(rows[delta_column], errors="coerce")
    rows = rows.dropna(subset=[delta_column]).sort_values(delta_column)
    colors = rows[delta_column].map(lambda value: "#3a8f6b" if value >= 0 else "#d95f5f")
    fig.add_trace(
        go.Bar(
            x=rows[delta_column],
            y=rows["config_label"],
            orientation="h",
            marker_color=colors,
            text=rows[delta_column].map(lambda value: _format_model_compare_metric(metric, value)),
            textposition="auto",
            hovertemplate="Feature set=%{y}<br>Delta=%{x:.2%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        title=f"{metric} delta vs all features",
        height=max(340, 80 + len(rows) * 28),
        xaxis_title=f"{metric} delta",
        xaxis_tickformat=".1%",
        yaxis_title="Feature set",
        margin=dict(l=20, r=20, t=45, b=30),
    )
    return fig


def _build_ablation_heatmap(score_table: pd.DataFrame) -> go.Figure:
    metrics = ["accuracy", "f1", "precision", "recall", "auc", "cumulative_return", "win_rate"]
    available = [metric for metric in metrics if metric in score_table.columns]
    fig = go.Figure()
    if score_table.empty or not available:
        return fig
    rows = score_table.copy()
    values = rows[available].apply(pd.to_numeric, errors="coerce")
    fig.add_trace(
        go.Heatmap(
            z=values.values,
            x=[metric.replace("_", " ").title() for metric in available],
            y=rows["config_label"],
            colorscale="Blues",
            colorbar=dict(title="Metric", tickformat=".0%"),
            hovertemplate="Feature set=%{y}<br>Metric=%{x}<br>Value=%{z:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Metric heatmap",
        height=max(340, 80 + len(rows) * 28),
        margin=dict(l=20, r=20, t=45, b=60),
    )
    return fig


def _build_ablation_group_impact_chart(group_impact: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if group_impact.empty:
        return fig
    rows = group_impact.copy()
    for column, label, color in (
        ("f1_drop_when_removed", "F1 drop when removed", "#d95f5f"),
        ("f1_gain_vs_technical", "F1 gain vs technical", "#3a8f6b"),
    ):
        if column not in rows.columns:
            continue
        fig.add_trace(
            go.Bar(
                x=rows["group_label"],
                y=pd.to_numeric(rows[column], errors="coerce"),
                name=label,
                marker_color=color,
                hovertemplate="Group=%{x}<br>Value=%{y:.2%}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        barmode="group",
        height=340,
        yaxis_title="F1 impact",
        yaxis_tickformat=".1%",
        margin=dict(l=20, r=20, t=10, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _ablation_group_impact_display_frame(group_impact: pd.DataFrame) -> pd.DataFrame:
    if group_impact.empty:
        return group_impact
    display = group_impact.copy()
    for column in (
        "f1_drop_when_removed",
        "auc_drop_when_removed",
        "return_drop_when_removed",
        "f1_gain_vs_technical",
        "auc_gain_vs_technical",
        "return_gain_vs_technical",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "feature_count" in display.columns:
        display["feature_count"] = display["feature_count"].map(
            lambda value: _format_number(value, digits=0)
        )
    return display.rename(
        columns={
            "group_label": "Group",
            "feature_count": "Features",
            "f1_drop_when_removed": "F1 Drop Removed",
            "auc_drop_when_removed": "AUC Drop Removed",
            "return_drop_when_removed": "Return Drop Removed",
            "f1_gain_vs_technical": "F1 Gain vs Technical",
            "auc_gain_vs_technical": "AUC Gain vs Technical",
            "return_gain_vs_technical": "Return Gain vs Technical",
        }
    )


def _build_ablation_return_chart(predictions: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if predictions.empty:
        return fig
    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    for config_label, config_rows in rows.groupby("config_label", dropna=False):
        config_rows = config_rows.sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=config_rows["date"],
                y=config_rows["strategy_cumulative_return"],
                mode="lines",
                name=str(config_label),
            )
        )
    buy_hold = rows.sort_values("date").drop_duplicates("date")
    fig.add_trace(
        go.Scatter(
            x=buy_hold["date"],
            y=buy_hold["buy_hold_cumulative_return"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="#555555", width=2, dash="dot"),
        )
    )
    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        height=380,
        yaxis_title="Cumulative return",
        yaxis_tickformat=".1%",
        margin=dict(l=20, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _read_prediction_history(database_path, ticker: str, target: str, model_name: str) -> pd.DataFrame:
    from stock_ai.storage import sqlite_store

    reloaded_store = importlib.reload(sqlite_store)
    read_history = getattr(reloaded_store, "read_prediction_history", None)
    if read_history is None:
        return pd.DataFrame()
    return read_history(
        database_path,
        ticker=ticker,
        target=target,
        model_name=model_name,
    )


def _read_all_prediction_history(database_path, ticker: str) -> pd.DataFrame:
    from stock_ai.storage import sqlite_store

    reloaded_store = importlib.reload(sqlite_store)
    read_history = getattr(reloaded_store, "read_prediction_history", None)
    if read_history is None:
        return pd.DataFrame()
    return read_history(database_path, ticker=ticker)


def _similarity_module():
    from stock_ai.similarity import chart_similarity

    return importlib.reload(chart_similarity)


def _cross_ticker_module():
    from stock_ai.cross_ticker import similarity

    return importlib.reload(similarity)


def _lead_lag_module():
    from stock_ai.lead_lag import analysis

    return importlib.reload(analysis)


def _anomaly_module():
    from stock_ai.anomaly import rule_based

    return importlib.reload(rule_based)


def _walk_forward_backtest_module():
    from stock_ai.backtest import walk_forward

    return importlib.reload(walk_forward)


def _evaluation_module():
    from stock_ai.evaluation import performance

    return importlib.reload(performance)


def _accuracy_diagnostics_module():
    from stock_ai.evaluation import accuracy_diagnostics

    return importlib.reload(accuracy_diagnostics)


def _risk_validation_module():
    from stock_ai.evaluation import risk_validation

    return importlib.reload(risk_validation)


def _news_impact_module():
    from stock_ai.evaluation import news_impact

    return importlib.reload(news_impact)


def _today_signal_validation_module():
    from stock_ai.evaluation import today_signal_validation

    return importlib.reload(today_signal_validation)


def _model_compare_module():
    from stock_ai.model_compare import compare

    return importlib.reload(compare)


def _ablation_module():
    from stock_ai.ablation import feature_ablation

    return importlib.reload(feature_ablation)


def _shap_reason_module():
    from stock_ai.explain import shap_reason

    return importlib.reload(shap_reason)


def _integrated_reason_module():
    from stock_ai.explain import integrated_reason

    return importlib.reload(integrated_reason)


def _risk_score_module():
    from stock_ai.explain import risk_score

    return importlib.reload(risk_score)


def _external_module():
    from stock_ai.external import market_data

    return importlib.reload(market_data)


def _events_module():
    from stock_ai.events import earnings

    return importlib.reload(earnings)


def _news_module():
    from stock_ai.news import manual_csv

    return importlib.reload(manual_csv)


def _news_fetcher_module():
    from stock_ai.news import fetcher

    return importlib.reload(fetcher)


def _news_quality_module():
    from stock_ai.news import quality

    return importlib.reload(quality)


def _news_query_profiles_module():
    from stock_ai.news import query_profiles

    return importlib.reload(query_profiles)


def _news_profile_quality_module():
    from stock_ai.news import profile_quality

    return importlib.reload(profile_quality)


def _news_sources_module():
    from stock_ai.news import sources

    return importlib.reload(sources)


def _narrative_module():
    from stock_ai.narrative import rule_based

    return importlib.reload(rule_based)


def _regime_module():
    from stock_ai.regime import rule_based

    return importlib.reload(rule_based)


def _external_market_context(data: pd.DataFrame):
    try:
        return _external_module().build_external_market_context(data)
    except Exception as error:
        _log_optional_context_error("external market", error)
        return None


def _earnings_event_context(data: pd.DataFrame):
    try:
        return _events_module().build_earnings_event_context(data)
    except Exception as error:
        _log_optional_context_error("earnings event", error)
        return None


def _news_context(data: pd.DataFrame, ticker: str, settings):
    try:
        news = _news_module()
        return news.build_news_context(
            data,
            news.default_news_dir(settings.paths.raw_data_dir.parent),
            ticker,
        )
    except Exception as error:
        _log_optional_context_error("news", error)
        return None


def _narrative_context(data: pd.DataFrame):
    try:
        return _narrative_module().build_narrative_context(data)
    except Exception as error:
        _log_optional_context_error("narrative", error)
        return None


def _current_regime_state(data: pd.DataFrame):
    try:
        return _regime_module().classify_latest_regime(data)
    except Exception as error:
        _log_optional_context_error("market regime", error)
        return None


def _log_optional_context_error(name: str, error: Exception) -> None:
    LOGGER.debug("Optional %s context is unavailable: %s", name, error, exc_info=True)


def _current_regime_name(data: pd.DataFrame) -> str | None:
    state = _current_regime_state(data)
    if state is not None:
        name = getattr(state, "name", None)
        if name:
            return str(name)
    if "regime" in data.columns and not data.empty:
        value = data.iloc[-1].get("regime")
        if pd.notna(value):
            return str(value)
    return None


def _current_event_state(data: pd.DataFrame) -> str | None:
    if "earnings_event_state" not in data.columns or data.empty:
        return None
    value = data.iloc[-1].get("earnings_event_state")
    if pd.isna(value):
        return None
    return str(value)


def _current_narrative_key(data: pd.DataFrame) -> str | None:
    if "dominant_narrative" not in data.columns or data.empty:
        return None
    value = data.iloc[-1].get("dominant_narrative")
    if pd.isna(value) or not str(value):
        return None
    return str(value)


def _prediction_similarity_context(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    similarity = _similarity_module()
    anomaly_data, _, _, _ = _anomaly_context(data)
    current_regime = _current_regime_name(data)
    current_event_state = _current_event_state(data)
    current_narrative = _current_narrative_key(data)
    patterns = similarity.find_similar_patterns(
        anomaly_data,
        window_size=20,
        top_n=5,
        forward_days=(1, 5, 20),
        regime_column="regime" if "regime" in data.columns else None,
        prefer_regime=current_regime,
        event_state_column="earnings_event_state" if "earnings_event_state" in data.columns else None,
        prefer_event_state=current_event_state,
        narrative_column="dominant_narrative" if "dominant_narrative" in data.columns else None,
        prefer_narrative=current_narrative,
    )
    summary = similarity.summarize_forward_returns(patterns, forward_days=(1, 5, 20))
    return patterns, summary


def _cross_ticker_similarity_context(
    ticker: str,
    data: pd.DataFrame,
    settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = _cross_ticker_module()
    patterns = cross.find_cross_ticker_similar_patterns_from_storage(
        target_data=data,
        raw_data_dir=settings.paths.raw_data_dir,
        feature_data_dir=settings.paths.feature_data_dir,
        target_ticker=ticker,
        window_size=20,
        top_n=5,
        forward_days=(1, 5, 20),
    )
    summary = cross.summarize_cross_ticker_patterns(patterns, forward_days=(1, 5, 20))
    return patterns, summary


def _lead_lag_context(
    ticker: str,
    data: pd.DataFrame,
    settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lead_lag = _lead_lag_module()
    metrics = lead_lag.analyze_lead_lag_from_storage(
        target_data=data,
        raw_data_dir=settings.paths.raw_data_dir,
        feature_data_dir=settings.paths.feature_data_dir,
        lags=(1, 3, 5, 10),
    )
    ranking = lead_lag.rank_lead_lag_signals(metrics, top_n=7)
    return metrics, ranking


def _anomaly_context(data: pd.DataFrame):
    try:
        anomaly = _anomaly_module()
        enriched = anomaly.add_anomaly_scores(data)
        context = anomaly.build_anomaly_context(enriched)
        events = anomaly.find_anomaly_events(enriched, min_score=50, top_n=100)
        summary = anomaly.summarize_anomaly_forward_returns(events, forward_days=(1, 5, 20))
        return enriched, context, events, summary
    except Exception:
        return data.copy(), None, pd.DataFrame(), pd.DataFrame()


def _render_pattern_analysis_overview(
    *,
    data: pd.DataFrame,
    ticker: str,
    analysis_name: str,
    result_count: int,
    top_summary: str,
    comparison_summary: str,
    lead_lag_summary: str,
    caveat: str,
) -> None:
    st.subheader("Pattern / Cross-Ticker Overview")
    st.caption(
        f"{analysis_name} is supporting context for the prediction, not a standalone forecast."
    )
    cols = st.columns(5)
    cols[0].metric("Latest feature date", _pattern_latest_feature_date(data))
    cols[1].metric("Analyzed ticker", ticker)
    cols[2].metric("Available results", _format_number(result_count, digits=0))
    cols[3].metric("Comparison scope", comparison_summary)
    cols[4].metric("Lead-Lag", lead_lag_summary)
    st.caption(f"Top summary: {top_summary}")
    st.caption(f"Caveat: {caveat}")

    with st.expander("How to read pattern analysis", expanded=False):
        st.write("- Similar Patterns searches for historical shapes resembling the current window.")
        st.write("- A similar shape does not mean the following price move will repeat.")
        st.write("- Cross-Ticker compares Toyota with other stocks and market-related series.")
        st.write("- Lead-Lag is exploratory timing evidence and does not prove causality.")
        st.write("- Small sample counts or weak correlations should not be treated as strong signals.")
        st.write("- Use Prediction for the latest forecast and Integrated Reasoning for the combined view.")


def _pattern_latest_feature_date(data: pd.DataFrame) -> str:
    if data.empty or "date" not in data.columns:
        return "-"
    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    if dates.empty:
        return "-"
    return dates.max().strftime("%Y-%m-%d")


def _similar_pattern_top_summary(result: pd.DataFrame) -> str:
    if result.empty:
        return "No similar historical window is available."
    best = result.iloc[0]
    end_date = _format_date(best.get("similar_end_date"))
    similarity = _format_number(best.get("similarity"), digits=3)
    return f"Best historical match ends {end_date} with similarity {similarity}."


def _cross_ticker_scope(patterns: pd.DataFrame) -> str:
    if patterns.empty:
        return "No peers available"
    names = patterns.get("ticker_name", patterns.get("ticker", pd.Series(dtype="object")))
    unique_names = [str(value) for value in names.dropna().unique() if str(value).strip()]
    return ", ".join(unique_names[:3]) if unique_names else "Peer tickers"


def _cross_ticker_top_summary(patterns: pd.DataFrame) -> str:
    if patterns.empty:
        return "No cross-ticker match is available."
    best = patterns.iloc[0]
    name = best.get("ticker_name") or best.get("ticker") or "peer"
    end_date = _format_date(best.get("similar_end_date"))
    similarity = _format_number(best.get("similarity"), digits=3)
    return f"Top match is {name}, ending {end_date}, with similarity {similarity}."


def _lead_lag_top_summary(ranking: pd.DataFrame) -> str:
    if ranking.empty:
        return "No lead-lag relationship is available."
    best = ranking.iloc[0]
    name = best.get("name") or best.get("symbol") or "candidate"
    lag = safe_int(best.get("lag"))
    correlation = _format_number(best.get("correlation"), digits=3)
    signal = str(best.get("current_signal") or "neutral")
    return f"Top relationship: {name}, {lag or '-'}D, correlation {correlation}, current signal {signal}."


def _render_similar_patterns_tab(ticker: str, data: pd.DataFrame) -> None:
    window_size = 20
    forward_days = (1, 5, 20)
    similarity = _similarity_module()
    anomaly_data, anomaly_context, _, _ = _anomaly_context(data)
    current_regime = _current_regime_name(data)
    current_event_state = _current_event_state(data)
    current_narrative = _current_narrative_key(data)
    result = similarity.find_similar_patterns(
        anomaly_data,
        window_size=window_size,
        top_n=5,
        forward_days=forward_days,
        regime_column="regime" if "regime" in data.columns else None,
        prefer_regime=current_regime,
        event_state_column="earnings_event_state" if "earnings_event_state" in data.columns else None,
        prefer_event_state=current_event_state,
        narrative_column="dominant_narrative" if "dominant_narrative" in data.columns else None,
        prefer_narrative=current_narrative,
    )

    _render_pattern_analysis_overview(
        data=data,
        ticker=ticker,
        analysis_name="Similar Patterns",
        result_count=len(result),
        top_summary=_similar_pattern_top_summary(result),
        comparison_summary="Same ticker history",
        lead_lag_summary="See Lead-Lag tab",
        caveat="Historical similarity is descriptive and does not guarantee the same future return.",
    )
    st.subheader("Similar Pattern Summary")
    st.caption(f"Latest {window_size} trading days compared with earlier windows in the same ticker.")
    st.caption("Features: Close return, Volume change, RSI, MACD, Volatility.")
    if current_regime:
        st.caption(f"Current regime: {current_regime}. Same-regime windows are prioritized.")
    if current_event_state:
        st.caption(
            f"Current event state: {current_event_state}. Same-event windows are prioritized."
        )
    if current_narrative:
        st.caption(
            f"Current narrative: {current_narrative}. Same-narrative windows are prioritized."
        )
    if anomaly_context is not None and getattr(anomaly_context, "score", 0) >= 40:
        st.caption(
            f"Current anomaly: {getattr(anomaly_context, 'score', 0)}/100 "
            f"({getattr(anomaly_context, 'level', 'Unknown')}). "
            "Matched-window anomaly scores are shown below."
        )
    if result.empty:
        st.info("Not enough historical data or valid features to find similar patterns yet.")
        return

    summary = similarity.summarize_forward_returns(result, forward_days=forward_days)
    _render_similarity_summary(summary)
    st.write(_build_similarity_rationale(result.iloc[0]))

    display_columns = [
        "similar_start_date",
        "similar_end_date",
        "similarity",
        "regime",
        "regime_match",
        "event_state",
        "event_state_match",
        "narrative",
        "narrative_match",
        "anomaly_score",
        "anomaly_types",
        "return_1d_after",
        "return_5d_after",
        "return_20d_after",
        "similar_features",
    ]
    display = result[[column for column in display_columns if column in result.columns]].copy()
    display["similar_start_date"] = pd.to_datetime(display["similar_start_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    display["similar_end_date"] = pd.to_datetime(display["similar_end_date"]).dt.strftime(
        "%Y-%m-%d"
    )
    display["similarity"] = display["similarity"].map(lambda value: f"{value:.4f}")
    if "regime_match" in display.columns:
        display["regime_match"] = display["regime_match"].map(
            lambda value: "Yes" if str(value).lower() == "true" else "No"
        )
    if "event_state_match" in display.columns:
        display["event_state_match"] = display["event_state_match"].map(
            lambda value: "Yes" if str(value).lower() == "true" else "No"
        )
    if "narrative_match" in display.columns:
        display["narrative_match"] = display["narrative_match"].map(
            lambda value: "Yes" if str(value).lower() == "true" else "No"
        )
    for column in ("return_1d_after", "return_5d_after", "return_20d_after"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    display = display.rename(
        columns={
            "similar_start_date": "Similar start",
            "similar_end_date": "Similar end",
            "similarity": "Similarity",
            "regime": "Regime",
            "regime_match": "Same regime",
            "event_state": "Event state",
            "event_state_match": "Same event",
            "narrative": "Narrative",
            "narrative_match": "Same narrative",
            "anomaly_score": "Anomaly score",
            "anomaly_types": "Anomaly types",
            "return_1d_after": "After 1D",
            "return_5d_after": "After 5D",
            "return_20d_after": "After 20D",
            "similar_features": "Similar features",
        }
    )
    with st.expander("Similar Pattern Details", expanded=False):
        st.plotly_chart(_build_similarity_ranking_chart(result), use_container_width=True)
        st.dataframe(display, use_container_width=True)

    with st.expander("Current vs selected historical pattern", expanded=False):
        selected_index = _select_similar_pattern(result)
        selected_pattern = result.iloc[selected_index]
        comparison = similarity.build_pattern_comparison_frame(
            data,
            selected_pattern,
            window_size=window_size,
            forward_days=max(forward_days),
        )
        if not comparison.empty:
            st.plotly_chart(
                _build_pattern_comparison_chart(comparison, window_size=window_size),
                use_container_width=True,
            )
            st.write(_build_similarity_rationale(selected_pattern))

    if not summary.empty:
        with st.expander("Forward return statistics and raw summary", expanded=False):
            st.plotly_chart(_build_forward_return_chart(result), use_container_width=True)
            display_summary = summary.copy()
            for column in (
                "average_return",
                "median_return",
                "positive_rate",
                "best_return",
                "worst_return",
            ):
                if column in display_summary.columns:
                    display_summary[column] = display_summary[column].map(_format_percent)
            st.dataframe(display_summary, use_container_width=True)


def _render_cross_ticker_similarity_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    cross = _cross_ticker_module()
    patterns, summary = _cross_ticker_similarity_context(ticker, data, settings)
    _, anomaly_context, _, _ = _anomaly_context(data)

    _render_pattern_analysis_overview(
        data=data,
        ticker=ticker,
        analysis_name="Cross-Ticker Similarity",
        result_count=len(patterns),
        top_summary=_cross_ticker_top_summary(patterns),
        comparison_summary=_cross_ticker_scope(patterns),
        lead_lag_summary="See Lead-Lag tab",
        caveat="Peer similarity is contextual evidence and does not imply Toyota will copy the peer move.",
    )
    st.subheader("Cross-Ticker Similarity")
    st.caption("Latest 20 Toyota trading days compared with 20-day windows in auto-related tickers.")
    st.caption("Peers: Toyota, Honda, Nissan, Tesla, BYD. Available local parquet files are used.")
    if anomaly_context is not None and getattr(anomaly_context, "score", 0) >= 40:
        st.caption(
            f"Current anomaly context: {getattr(anomaly_context, 'score', 0)}/100 "
            f"({getattr(anomaly_context, 'level', 'Unknown')}). "
            "Read cross-ticker matches together with the Anomaly tab."
        )
    if patterns.empty:
        st.info(
            "Cross-ticker patterns are not available yet. Run download-external or download peer "
            "tickers, then build features again."
        )
        st.code(
            "python main.py download-external --start 2010-01-01\n"
            "python main.py build-features --ticker 7203.T",
            language="powershell",
        )
        return

    st.write(cross.build_cross_ticker_reason(patterns, summary))
    _render_cross_ticker_summary(summary)

    display_columns = [
        "ticker_name",
        "ticker",
        "similar_start_date",
        "similar_end_date",
        "similarity",
        "peer_return_1d_after",
        "peer_return_5d_after",
        "peer_return_20d_after",
        "toyota_return_1d_after",
        "toyota_return_5d_after",
        "toyota_return_20d_after",
        "similar_features",
    ]
    display = patterns[[column for column in display_columns if column in patterns.columns]].copy()
    for column in ("similar_start_date", "similar_end_date"):
        if column in display.columns:
            display[column] = pd.to_datetime(display[column]).dt.strftime("%Y-%m-%d")
    if "similarity" in display.columns:
        display["similarity"] = display["similarity"].map(lambda value: f"{value:.4f}")
    for column in (
        "peer_return_1d_after",
        "peer_return_5d_after",
        "peer_return_20d_after",
        "toyota_return_1d_after",
        "toyota_return_5d_after",
        "toyota_return_20d_after",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    display = display.rename(
        columns={
            "ticker_name": "Similar ticker",
            "ticker": "Ticker",
            "similar_start_date": "Similar start",
            "similar_end_date": "Similar end",
            "similarity": "Similarity",
            "peer_return_1d_after": "Peer after 1D",
            "peer_return_5d_after": "Peer after 5D",
            "peer_return_20d_after": "Peer after 20D",
            "toyota_return_1d_after": "Toyota same-date 1D",
            "toyota_return_5d_after": "Toyota same-date 5D",
            "toyota_return_20d_after": "Toyota same-date 20D",
            "similar_features": "Similar features",
        }
    )
    with st.expander("Cross-Ticker ranking and raw similarity table", expanded=False):
        st.plotly_chart(_build_cross_ticker_ranking_chart(patterns), use_container_width=True)
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_lead_lag_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    lead_lag = _lead_lag_module()
    metrics, ranking = _lead_lag_context(ticker, data, settings)

    _render_pattern_analysis_overview(
        data=data,
        ticker=ticker,
        analysis_name="Lead-Lag",
        result_count=len(ranking),
        top_summary=_lead_lag_top_summary(ranking),
        comparison_summary=f"{metrics['symbol'].nunique() if 'symbol' in metrics.columns else 0} candidates",
        lead_lag_summary=(
            str(ranking.iloc[0].get("current_signal") or "neutral")
            if not ranking.empty
            else "Not available"
        ),
        caveat="Correlation and timing relationships are exploratory and are not proof of causality.",
    )
    st.subheader("Lead-Lag Summary")
    st.caption(
        "Candidate 1-day returns are compared with Toyota future returns over 1, 3, 5, and 10 trading days."
    )
    st.caption(
        "This is exploratory correlation analysis, not proof of causality. Available joined external features or local parquet files are used."
    )
    if metrics.empty:
        st.info(
            "Lead-lag analysis is not available yet. Download external market and peer data, then rebuild features."
        )
        st.code(
            "python main.py download-external --start 2010-01-01\n"
            "python main.py build-features --ticker 7203.T",
            language="powershell",
        )
        return

    st.write(lead_lag.build_lead_lag_reason(metrics, ranking))

    impact = _lead_lag_impact_summary_frame(ranking)
    if not impact.empty:
        st.caption("Leading indicators at a glance")
        st.dataframe(impact, use_container_width=True, hide_index=True)

    with st.expander("Leading Indicator Ranking", expanded=False):
        st.dataframe(_lead_lag_display_frame(ranking), use_container_width=True, hide_index=True)

    with st.expander("Lag Correlation Heatmap", expanded=False):
        st.plotly_chart(_build_lead_lag_heatmap(metrics), use_container_width=True)

    with st.expander("Raw lead-lag details", expanded=False):
        detail = _lead_lag_display_frame(metrics)
        st.dataframe(detail, use_container_width=True, hide_index=True)


def _lead_lag_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display_columns = [
        "rank",
        "name",
        "symbol",
        "lag",
        "score",
        "sample_size",
        "correlation",
        "direction_agreement",
        "latest_candidate_return",
        "current_signal",
        "large_move_threshold",
        "toyota_up_probability_after_large_up",
        "toyota_down_probability_after_large_down",
        "toyota_avg_return_after_large_up",
        "toyota_avg_return_after_large_down",
    ]
    display = frame[[column for column in display_columns if column in frame.columns]].copy()
    for column in (
        "direction_agreement",
        "latest_candidate_return",
        "large_move_threshold",
        "toyota_up_probability_after_large_up",
        "toyota_down_probability_after_large_down",
        "toyota_avg_return_after_large_up",
        "toyota_avg_return_after_large_down",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("correlation", "score"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=3))
    return display.rename(
        columns={
            "rank": "Rank",
            "name": "Candidate",
            "symbol": "Symbol",
            "lag": "Lag",
            "score": "Score",
            "sample_size": "Samples",
            "correlation": "Correlation",
            "direction_agreement": "Direction agreement",
            "latest_candidate_return": "Latest return",
            "current_signal": "Current signal",
            "large_move_threshold": "Large move threshold",
            "toyota_up_probability_after_large_up": "Toyota up after candidate surge",
            "toyota_down_probability_after_large_down": "Toyota down after candidate drop",
            "toyota_avg_return_after_large_up": "Toyota avg after surge",
            "toyota_avg_return_after_large_down": "Toyota avg after drop",
        }
    )


def _build_lead_lag_heatmap(metrics: pd.DataFrame) -> go.Figure:
    heatmap = metrics.copy()
    heatmap["lag_label"] = heatmap["lag"].map(lambda value: f"{int(value)}D")
    pivot = heatmap.pivot_table(
        index="name",
        columns="lag_label",
        values="correlation",
        aggfunc="first",
    )
    ordered_columns = [column for column in ("1D", "3D", "5D", "10D") if column in pivot.columns]
    pivot = pivot[ordered_columns]

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            zmid=0,
            colorscale="RdBu",
            colorbar=dict(title="Corr"),
            hovertemplate="Candidate=%{y}<br>Lag=%{x}<br>Correlation=%{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(320, 60 + len(pivot.index) * 34),
        margin=dict(l=20, r=20, t=10, b=30),
        xaxis_title="Future Toyota horizon",
        yaxis_title="Candidate",
    )
    return fig


def _lead_lag_impact_summary_frame(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty:
        return pd.DataFrame(columns=["Candidate", "Summary"])
    rows: list[dict[str, str]] = []
    for row in ranking.itertuples(index=False):
        name = getattr(row, "name", "Candidate")
        lag = getattr(row, "lag", "-")
        corr = _format_number(getattr(row, "correlation", None), digits=3)
        after_up = _format_percent(getattr(row, "toyota_up_probability_after_large_up", None))
        after_down = _format_percent(getattr(row, "toyota_down_probability_after_large_down", None))
        current_signal = getattr(row, "current_signal", "neutral")
        if current_signal == "bullish":
            current_text = "current move leans bullish"
        elif current_signal == "bearish":
            current_text = "current move leans bearish"
        else:
            current_text = "current move is neutral"
        rows.append(
            {
                "Candidate": str(name),
                "Summary": (
                    f"{name} has its strongest relationship at {lag}D "
                    f"(corr {corr}); Toyota rose {after_up} after large {name} rises "
                    f"and fell {after_down} after large {name} drops. {current_text}."
                ),
            }
        )
    return pd.DataFrame(rows)


def _render_anomaly_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    anomaly = _anomaly_module()
    enriched, context, events, summary = _anomaly_context(data)
    regime_state = _current_regime_state(data)
    regime_history = _recent_regime_history(data)
    recent_anomaly_count = _recent_anomaly_event_count(enriched, events)

    st.subheader("Anomaly / Regime Overview")
    st.caption(
        "Current abnormality and market regime are supporting context for Prediction and Integrated Reasoning."
    )
    overview_cols = st.columns(5)
    overview_cols[0].metric("Latest feature date", _pattern_latest_feature_date(data))
    overview_cols[1].metric("Analyzed ticker", ticker)
    overview_cols[2].metric(
        "Current anomaly",
        getattr(context, "level", "Not available") if context is not None else "Not available",
        f"{getattr(context, 'score', 0)}/100" if context is not None else None,
    )
    overview_cols[3].metric(
        "Current regime",
        getattr(regime_state, "name", "Not available") if regime_state is not None else "Not available",
        getattr(regime_state, "level", None) if regime_state is not None else None,
    )
    overview_cols[4].metric("Recent anomalies", _format_number(recent_anomaly_count, digits=0))
    st.caption("Recent anomaly count uses the latest 60 available feature rows.")
    st.caption(f"Regime transition: {_regime_transition_summary(regime_history)}")
    st.caption(
        "Caveat: unusual conditions and regime labels are descriptive signals, not trade instructions or outcome guarantees."
    )

    with st.expander("How to read anomaly / regime analysis", expanded=False):
        st.write("- Anomaly detects price or feature conditions that differ from their recent baseline.")
        st.write("- An unusual condition is not automatically dangerous or profitable.")
        st.write("- Regime is a rule-based description of the current market environment.")
        st.write("- Regime classification does not guarantee the next price direction.")
        st.write("- Results depend on available samples and thresholds, so do not use them alone.")
        st.write("- Read them together with Prediction, Risk, News, and Similar Patterns.")

    if context is None:
        st.info("Anomaly detection is not available for the current dataset yet.")
    else:
        st.subheader("Current Anomaly Summary")
        _render_anomaly_summary(context, summary, show_heading=False)

    st.subheader("Current Regime Summary")
    if regime_state is None:
        st.info("Market regime is not available for the current dataset yet.")
    else:
        _render_market_regime_context(regime_state, show_heading=False)

    if context is not None:
        with st.expander("Anomaly Details / Component Scores", expanded=False):
            st.plotly_chart(_build_anomaly_component_chart(context), use_container_width=True)

    with st.expander("Recent Anomaly History", expanded=False):
        if events.empty:
            st.info("No historical anomaly events above the MVP threshold were found.")
        else:
            _render_anomaly_forward_summary(summary)
            display = _anomaly_events_display_frame(events)
            st.dataframe(display, use_container_width=True, hide_index=True)

    with st.expander("Regime History / Transition", expanded=False):
        st.caption(_regime_transition_summary(regime_history))
        if regime_history.empty:
            st.info("Regime history is not available.")
        else:
            st.dataframe(
                _regime_history_display_frame(regime_history),
                use_container_width=True,
                hide_index=True,
            )

    if not events.empty:
        with st.expander("Current vs Selected Anomaly", expanded=False):
            selected_label = st.selectbox(
                "Anomaly event",
                options=_anomaly_event_options(events),
                index=0,
                key="anomaly_event_select",
            )
            selected_date = selected_label.split(" | ", maxsplit=1)[0]
            comparison = anomaly.build_anomaly_comparison_frame(
                enriched,
                selected_date,
                lookback=20,
                forward=20,
            )
            if comparison.empty:
                st.info("Comparison chart is not available for the selected event.")
            else:
                st.plotly_chart(
                    _build_anomaly_comparison_chart(comparison, selected_date, ticker),
                    use_container_width=True,
                )

    with st.expander("Raw anomaly / regime table", expanded=False):
        raw_columns = [
            "date",
            "regime",
            "anomaly_score",
            "anomaly_level",
            "anomaly_types",
            "anomaly_reasons",
        ]
        raw = enriched[
            [column for column in raw_columns if column in enriched.columns]
        ].tail(60)
        if raw.empty:
            st.info("Raw anomaly / regime data is not available.")
        else:
            raw = raw.copy()
            if "date" in raw.columns:
                raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.strftime("%Y-%m-%d")
            st.dataframe(raw, use_container_width=True, hide_index=True)


def _recent_anomaly_event_count(
    enriched: pd.DataFrame,
    events: pd.DataFrame,
    *,
    lookback_rows: int = 60,
) -> int:
    if enriched.empty or events.empty or "date" not in enriched.columns or "date" not in events.columns:
        return 0
    recent_dates = pd.to_datetime(enriched["date"], errors="coerce").dropna().sort_values()
    if recent_dates.empty:
        return 0
    cutoff = recent_dates.tail(max(1, lookback_rows)).min()
    event_dates = pd.to_datetime(events["date"], errors="coerce")
    return int(event_dates.ge(cutoff).sum())


def _recent_regime_history(data: pd.DataFrame, *, limit: int = 60) -> pd.DataFrame:
    if data.empty or not {"date", "regime"}.issubset(data.columns):
        return pd.DataFrame(columns=["date", "regime"])
    history = data[["date", "regime"]].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["regime"] = history["regime"].fillna("").astype(str).str.strip()
    history = history.dropna(subset=["date"])
    history = history[history["regime"].ne("")]
    return history.sort_values("date").tail(max(1, limit)).reset_index(drop=True)


def _regime_transition_summary(history: pd.DataFrame) -> str:
    if history.empty or "regime" not in history.columns:
        return "Regime history is not available."
    regimes = history["regime"].fillna("").astype(str).tolist()
    current = regimes[-1]
    run_length = 1
    for value in reversed(regimes[:-1]):
        if value != current:
            break
        run_length += 1
    previous = next(
        (value for value in reversed(regimes[:-run_length]) if value != current),
        None,
    )
    if previous:
        return f"{previous} transitioned to {current}; current regime has continued for {run_length} trading days."
    return f"Current regime is {current} and has continued for {run_length} observed trading days."


def _regime_history_display_frame(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    display = history.copy()
    display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    display["transition"] = display["regime"].ne(display["regime"].shift(1))
    display["transition"] = display["transition"].map(
        lambda value: "Changed" if value else "Continued"
    )
    if not display.empty:
        display.loc[display.index[0], "transition"] = "Start"
    return display.rename(
        columns={
            "date": "Date",
            "regime": "Regime",
            "transition": "Transition",
        }
    ).sort_values("Date", ascending=False)


def _render_anomaly_summary(
    context,
    summary: pd.DataFrame | None = None,
    *,
    show_heading: bool = True,
) -> None:
    if context is None:
        return
    if show_heading:
        st.subheader("Anomaly Summary")
    types = getattr(context, "types", []) or []
    component_scores = getattr(context, "component_scores", {}) or {}
    top_component = "-"
    if component_scores:
        top_component = max(component_scores.items(), key=lambda item: item[1])[0]

    cols = st.columns(4)
    cols[0].metric("Anomaly Score", f"{getattr(context, 'score', 0)}/100")
    cols[1].metric("Level", getattr(context, "level", "Unknown"))
    cols[2].metric("Type", ", ".join(types[:2]) if types else "Normal")
    cols[3].metric("Top Component", top_component)

    summary_text = getattr(context, "summary", "")
    if summary_text:
        st.write(summary_text)
    reasons = getattr(context, "reasons", []) or []
    if reasons:
        st.caption("Anomaly reasons")
        for reason in reasons[:5]:
            st.write(f"- {reason}")

    if summary is not None and not summary.empty:
        row_5d = summary[summary["horizon"] == "5D"]
        if not row_5d.empty:
            st.caption(
                "Historical anomaly 5D average return: "
                f"{_format_percent(row_5d.iloc[0].get('average_return'))}; "
                f"positive rate: {_format_percent(row_5d.iloc[0].get('positive_rate'))}."
            )


def _render_anomaly_forward_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    display = summary.copy()
    for column in ("average_return", "positive_rate", "worst_return", "best_return"):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    display = display.rename(
        columns={
            "horizon": "Horizon",
            "count": "Count",
            "average_return": "Average return",
            "positive_rate": "Positive rate",
            "worst_return": "Worst return",
            "best_return": "Best return",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _anomaly_events_display_frame(events: pd.DataFrame) -> pd.DataFrame:
    display_columns = [
        "date",
        "anomaly_score",
        "anomaly_level",
        "anomaly_types",
        "return_1d",
        "volume_change_1d",
        "volatility_20",
        "anomaly_return_1d_after",
        "anomaly_return_5d_after",
        "anomaly_return_20d_after",
        "anomaly_reasons",
    ]
    display = events[[column for column in display_columns if column in events.columns]].copy()
    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    for column in (
        "return_1d",
        "volume_change_1d",
        "volatility_20",
        "anomaly_return_1d_after",
        "anomaly_return_5d_after",
        "anomaly_return_20d_after",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    return display.rename(
        columns={
            "date": "Date",
            "anomaly_score": "Score",
            "anomaly_level": "Level",
            "anomaly_types": "Types",
            "return_1d": "Return 1D",
            "volume_change_1d": "Volume change",
            "volatility_20": "Volatility",
            "anomaly_return_1d_after": "After 1D",
            "anomaly_return_5d_after": "After 5D",
            "anomaly_return_20d_after": "After 20D",
            "anomaly_reasons": "Reasons",
        }
    )


def _anomaly_event_options(events: pd.DataFrame) -> list[str]:
    options: list[str] = []
    for row in events.head(50).itertuples(index=False):
        date = pd.to_datetime(getattr(row, "date")).strftime("%Y-%m-%d")
        score = getattr(row, "anomaly_score", "-")
        types = getattr(row, "anomaly_types", "") or "Anomaly"
        options.append(f"{date} | score {score} | {types}")
    return options


def _build_anomaly_component_chart(context) -> go.Figure:
    scores = getattr(context, "component_scores", {}) or {}
    frame = pd.DataFrame(
        [{"component": component, "score": score} for component, score in scores.items()]
    ).sort_values("score", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=frame["score"],
            y=frame["component"],
            orientation="h",
            marker_color="#b07aa1",
            text=frame["score"].map(lambda value: f"{value}/100"),
            textposition="auto",
        )
    )
    fig.update_layout(
        height=max(300, 60 + len(frame) * 34),
        xaxis_title="Score",
        yaxis_title="",
        xaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _build_anomaly_comparison_chart(
    comparison: pd.DataFrame,
    selected_date: str,
    ticker: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=comparison["day"],
            y=comparison["current"],
            mode="lines+markers",
            name="Current",
            line=dict(color="#3a8f6b", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=comparison["day"],
            y=comparison["selected_anomaly"],
            mode="lines+markers",
            name=f"Anomaly {selected_date}",
            line=dict(color="#d95f5f", width=2),
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="#777")
    fig.update_layout(
        title=f"{ticker}: Current vs Historical Anomaly",
        height=360,
        xaxis_title="Trading days from anomaly/current date",
        yaxis_title="Close index at day 0 = 100",
        margin=dict(l=20, r=20, t=45, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _render_cross_ticker_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    display = summary.copy()
    for column in (
        "peer_average_return",
        "toyota_average_return",
        "peer_positive_rate",
        "toyota_positive_rate",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    display = display.rename(
        columns={
            "horizon": "Horizon",
            "count": "Count",
            "peer_average_return": "Peer avg return",
            "toyota_average_return": "Toyota avg return",
            "peer_positive_rate": "Peer positive rate",
            "toyota_positive_rate": "Toyota positive rate",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _build_cross_ticker_ranking_chart(patterns: pd.DataFrame) -> go.Figure:
    labels = [
        f"{row.ticker_name} {pd.to_datetime(row.similar_end_date).strftime('%Y-%m-%d')}"
        for row in patterns.itertuples(index=False)
    ]
    fig = go.Figure(
        go.Bar(
            x=patterns["similarity"],
            y=labels,
            orientation="h",
            marker_color="#6c8dbf",
        )
    )
    fig.update_layout(
        height=300,
        xaxis_title="Cosine similarity",
        yaxis_title="Peer window",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _render_similarity_summary(summary: pd.DataFrame) -> None:
    if summary.empty:
        return

    columns = st.columns(min(3, len(summary)))
    for position, row in enumerate(summary.itertuples(index=False)):
        columns[position % len(columns)].metric(
            f"Avg after {row.horizon}",
            _format_percent(row.average_return),
            delta=f"Up rate {_format_percent(row.positive_rate)}",
        )


def _build_similarity_ranking_chart(result: pd.DataFrame) -> go.Figure:
    chart_frame = result.sort_values("similarity", ascending=True).copy()
    chart_frame["window_label"] = chart_frame.apply(_similar_window_label, axis=1)
    fig = go.Figure(
        go.Bar(
            x=chart_frame["similarity"],
            y=chart_frame["window_label"],
            orientation="h",
            marker_color="#4f78a8",
            text=chart_frame["similarity"].map(lambda value: f"{value:.3f}"),
            textposition="outside",
            customdata=chart_frame[["similar_features"]],
            hovertemplate=(
                "Window=%{y}<br>"
                "Similarity=%{x:.4f}<br>"
                "Close features=%{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=300,
        xaxis_title="Cosine similarity",
        yaxis_title="Similar window",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _select_similar_pattern(result: pd.DataFrame) -> int:
    options = list(range(len(result)))
    return st.selectbox(
        "Pattern to compare",
        options=options,
        format_func=lambda index: f"#{index + 1} {_similar_window_label(result.iloc[index])}",
        key="similar_patterns_pattern_select",
    )


def _build_pattern_comparison_chart(comparison: pd.DataFrame, window_size: int) -> go.Figure:
    fig = go.Figure()
    current = comparison[comparison["series"] == "Current"]
    matched = comparison[comparison["series"] == "Matched past"]
    matched_pattern = matched[matched["day"] < window_size]
    matched_future = matched[matched["day"] >= window_size - 1]

    fig.add_trace(
        go.Scatter(
            x=current["day"],
            y=current["normalized_close"],
            mode="lines+markers",
            name="Current 20D",
            line=dict(color="#3a8f6b", width=3),
            customdata=current[["date", "close"]],
            hovertemplate=(
                "Current day=%{x}<br>"
                "Date=%{customdata[0]|%Y-%m-%d}<br>"
                "Close=%{customdata[1]:,.0f}<br>"
                "Return=%{y:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=matched_pattern["day"],
            y=matched_pattern["normalized_close"],
            mode="lines+markers",
            name="Matched 20D",
            line=dict(color="#4f78a8", width=3),
            customdata=matched_pattern[["date", "close"]],
            hovertemplate=(
                "Matched day=%{x}<br>"
                "Date=%{customdata[0]|%Y-%m-%d}<br>"
                "Close=%{customdata[1]:,.0f}<br>"
                "Return=%{y:.2%}<extra></extra>"
            ),
        )
    )
    if not matched_future.empty:
        fig.add_trace(
            go.Scatter(
                x=matched_future["day"],
                y=matched_future["normalized_close"],
                mode="lines+markers",
                name="Matched future",
                line=dict(color="#d08c3f", width=2, dash="dot"),
                customdata=matched_future[["date", "close"]],
                hovertemplate=(
                    "Future day=%{x}<br>"
                    "Date=%{customdata[0]|%Y-%m-%d}<br>"
                    "Close=%{customdata[1]:,.0f}<br>"
                    "Return=%{y:.2%}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=window_size - 1, line_dash="dash", line_color="#777777")
    fig.update_layout(
        height=430,
        yaxis_tickformat=".1%",
        xaxis_title="Trading days from pattern start",
        yaxis_title="Close return from pattern start",
        margin=dict(l=20, r=20, t=10, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _build_forward_return_chart(result: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for days in (1, 5, 20):
        column = f"return_{days}d_after"
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce").dropna()
        if values.empty:
            continue
        fig.add_trace(
            go.Box(
                y=values,
                name=f"{days}D",
                boxpoints="all",
                marker_color="#4f78a8",
                hovertemplate=f"After {days}D=%{{y:.2%}}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line_color="#555555", line_width=1)
    fig.update_layout(
        height=300,
        yaxis_tickformat=".1%",
        xaxis_title="Forward horizon",
        yaxis_title="Forward return",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _build_similarity_rationale(pattern: pd.Series) -> str:
    start = pd.to_datetime(pattern.get("similar_start_date")).strftime("%Y-%m-%d")
    end = pd.to_datetime(pattern.get("similar_end_date")).strftime("%Y-%m-%d")
    similarity = safe_float(pattern.get("similarity"))
    features = pattern.get("similar_features") or "unknown"
    after_1d = _format_percent(pattern.get("return_1d_after"))
    after_5d = _format_percent(pattern.get("return_5d_after"))
    after_20d = _format_percent(pattern.get("return_20d_after"))
    regime = pattern.get("regime")
    regime_match = pattern.get("regime_match")
    event_state = pattern.get("event_state")
    event_state_match = pattern.get("event_state_match")
    narrative = pattern.get("narrative")
    narrative_match = pattern.get("narrative_match")
    regime_text = ""
    if regime is not None and pd.notna(regime):
        match_text = (
            "same as current regime"
            if regime_match is True
            else "different from current regime"
        )
        regime_text = f" The matched window regime was {regime} ({match_text})."
    event_text = ""
    if event_state is not None and pd.notna(event_state):
        match_text = (
            "same as current event state"
            if event_state_match is True
            else "different from current event state"
        )
        event_text = f" The matched window event state was {event_state} ({match_text})."
    narrative_text = ""
    if narrative is not None and pd.notna(narrative) and str(narrative):
        match_text = (
            "same as current narrative"
            if narrative_match is True
            else "different from current narrative"
        )
        narrative_text = f" The matched window narrative was {narrative} ({match_text})."
    return (
        f"Most comparable selected window is {start} to {end}. "
        f"Cosine similarity is {_format_number(similarity, digits=4)}. "
        f"Closest features are {features}. "
        f"After that window, returns were 1D {after_1d}, 5D {after_5d}, and 20D {after_20d}."
        f"{regime_text}{event_text}{narrative_text}"
    )


def _similar_window_label(row: pd.Series) -> str:
    start = pd.to_datetime(row.get("similar_start_date")).strftime("%Y-%m-%d")
    end = pd.to_datetime(row.get("similar_end_date")).strftime("%Y-%m-%d")
    return f"{start} to {end}"


def _render_shap_explanation(top_features: list[dict[str, object]], prediction_label: str) -> None:
    shap_reason = _shap_reason_module()
    upward, downward = shap_reason.split_shap_contributions(top_features, top_n=10)

    st.caption("SHAP local explanation for the latest prediction")
    st.caption("Positive SHAP values push the model toward Up. Negative values push it toward Down.")
    shap_cols = st.columns(2)
    with shap_cols[0]:
        st.caption("Upward drivers top 10")
        if upward:
            st.plotly_chart(
                _build_shap_direction_chart(upward, direction="up"),
                use_container_width=True,
            )
            st.dataframe(_shap_display_table(upward), use_container_width=True)
        else:
            st.info("No upward SHAP drivers were stored for this prediction.")

    with shap_cols[1]:
        st.caption("Downward drivers top 10")
        if downward:
            st.plotly_chart(
                _build_shap_direction_chart(downward, direction="down"),
                use_container_width=True,
            )
            st.dataframe(_shap_display_table(downward), use_container_width=True)
        else:
            st.info("No downward SHAP drivers were stored for this prediction.")

    st.caption("SHAP reason")
    st.write(shap_reason.build_shap_reason_text(top_features, prediction_label=prediction_label))


def _render_integrated_reasoning(
    data: pd.DataFrame,
    prediction_label: str,
    up_probability,
    down_probability,
    top_features: list[dict[str, object]],
    ticker: str,
    settings,
    multiclass_prediction: pd.Series | None = None,
    risk_multiclass_prediction: pd.Series | None = None,
    risk_multiclass_profile: str | None = None,
    risk_multiclass_weight_mode: str | None = None,
    risk_bucket_mode: str | None = None,
) -> None:
    integrated_reason = _integrated_reason_module()
    risk_score = _risk_score_module()
    external_context = _external_market_context(data)
    earnings_context = _earnings_event_context(data)
    news_context = _news_context(data, ticker, settings)
    narrative_context = _narrative_context(data)
    regime_state = _current_regime_state(data)
    similar_patterns, similar_summary = _prediction_similarity_context(data)
    cross_patterns, cross_summary = _cross_ticker_similarity_context(ticker, data, settings)
    _, lead_lag_ranking = _lead_lag_context(ticker, data, settings)
    _, anomaly_context, _, anomaly_summary = _anomaly_context(data)
    risk_scores = risk_score.build_surge_crash_scores(
        data=data,
        shap_items=top_features,
        similar_summary=similar_summary,
        cross_ticker_summary=cross_summary,
        lead_lag_summary=lead_lag_ranking,
        anomaly_context=anomaly_context,
    )
    risk_scores = _augment_risk_scores_with_multiclass(
        risk_scores,
        risk_multiclass_prediction,
        profile=risk_multiclass_profile,
        weight_mode=risk_multiclass_weight_mode,
    )
    reasoning = integrated_reason.build_integrated_reasoning(
        prediction_label=prediction_label,
        up_probability=up_probability,
        down_probability=down_probability,
        shap_items=top_features,
        similar_patterns=similar_patterns,
        similar_summary=similar_summary,
        external_context=external_context,
        event_context=earnings_context,
        news_context=news_context,
        narrative_context=narrative_context,
        cross_ticker_patterns=cross_patterns,
        cross_ticker_summary=cross_summary,
        lead_lag_summary=lead_lag_ranking,
        anomaly_context=anomaly_context,
        regime_state=regime_state,
        risk_scores=risk_scores,
    )
    multiclass_text = _multiclass_reasoning_text(multiclass_prediction)
    risk_multiclass_text = _risk_multiclass_reasoning_text(
        risk_multiclass_prediction,
        risk_multiclass_profile,
        risk_multiclass_weight_mode,
    )
    risk_percentiles = _current_risk_percentile_context(
        ticker,
        data,
        settings,
        risk_scores,
        risk_multiclass_profile,
        risk_multiclass_weight_mode,
        risk_bucket_mode,
    )
    risk_validation_text = _risk_validation_context_text(
        ticker,
        data,
        settings,
        risk_multiclass_profile,
        risk_multiclass_weight_mode,
        risk_bucket_mode,
        risk_percentiles,
    )
    risk_conflict_text = _risk_multiclass_conflict_text(
        prediction_label=prediction_label,
        up_probability=up_probability,
        down_probability=down_probability,
        risk_multiclass_prediction=risk_multiclass_prediction,
    )
    news_impact_text = _daily_brief_news_impact_text(data)

    st.subheader("Risk Scores")
    _render_surge_crash_scores(
        risk_scores,
        ticker=ticker,
        risk_multiclass_prediction=risk_multiclass_prediction,
        risk_multiclass_profile=risk_multiclass_profile,
        risk_multiclass_weight_mode=risk_multiclass_weight_mode,
        risk_percentiles=risk_percentiles,
    )
    if risk_conflict_text:
        st.warning(risk_conflict_text)

    st.subheader("Integrated Reasoning")
    cols = st.columns(9)
    cols[0].metric("Confidence", reasoning.confidence_level)
    cols[1].metric("Prediction", _integrated_direction_label(reasoning.prediction_direction))
    cols[2].metric("SHAP", _integrated_direction_label(reasoning.shap_direction))
    cols[3].metric("Similar", _integrated_direction_label(reasoning.similarity_direction))
    cols[4].metric("News", _integrated_direction_label(reasoning.news_direction))
    cols[5].metric("Narrative", _integrated_direction_label(reasoning.narrative_direction))
    cols[6].metric("Cross", _integrated_direction_label(reasoning.cross_ticker_direction))
    cols[7].metric("Lead-Lag", _integrated_direction_label(reasoning.lead_lag_direction))
    cols[8].metric("Anomaly", _integrated_direction_label(reasoning.anomaly_direction))

    evidence = pd.DataFrame(
        [
            {"Evidence": "Probability", "Summary": reasoning.probability_text},
            {"Evidence": "SHAP", "Summary": reasoning.shap_text},
            {"Evidence": "Similar Patterns", "Summary": reasoning.similarity_text},
            {"Evidence": "External Market", "Summary": reasoning.external_text},
            {"Evidence": "Earnings Event", "Summary": reasoning.event_text},
            {"Evidence": "News", "Summary": reasoning.news_text},
            {"Evidence": "News Alignment", "Summary": reasoning.news_alignment_text},
            {"Evidence": "News Impact Validation", "Summary": news_impact_text},
            {"Evidence": "Narrative", "Summary": reasoning.narrative_text},
            {"Evidence": "Cross-Ticker", "Summary": reasoning.cross_ticker_text},
            {"Evidence": "Lead-Lag", "Summary": reasoning.lead_lag_text},
            {"Evidence": "Anomaly", "Summary": reasoning.anomaly_text},
            {"Evidence": "Market Regime", "Summary": reasoning.regime_text},
            {"Evidence": "Multiclass", "Summary": multiclass_text},
            {"Evidence": "Risk Multiclass", "Summary": risk_multiclass_text},
            {"Evidence": "Risk Validation", "Summary": risk_validation_text},
            {"Evidence": "Risk Conflict", "Summary": risk_conflict_text},
            {"Evidence": "Surge / Crash", "Summary": reasoning.risk_text},
            {"Evidence": "Agreement", "Summary": reasoning.agreement},
        ]
    )
    _render_integrated_reasoning_summary(
        reasoning=reasoning,
        multiclass_prediction=multiclass_prediction,
        multiclass_text=multiclass_text,
        risk_multiclass_text=risk_multiclass_text,
        risk_validation_text=risk_validation_text,
        risk_conflict_text=risk_conflict_text,
        news_impact_text=news_impact_text,
    )
    _render_integrated_reasoning_details(evidence)

    with st.expander("Market / sector reasoning details", expanded=False):
        _render_external_market_context(external_context)
        _render_market_regime_context(regime_state)

    with st.expander("News reasoning details", expanded=False):
        _render_earnings_event_context(earnings_context)
        _render_news_context(news_context)
        _render_narrative_context(narrative_context)

    with st.expander("Risk / anomaly / similar pattern details", expanded=False):
        _render_prediction_context_overview(reasoning)
        _render_anomaly_summary(anomaly_context, anomaly_summary)


def _render_prediction_context_overview(reasoning) -> None:
    rows = [
        {"Context": "Similar Patterns", "Summary": reasoning.similarity_text},
        {"Context": "Cross-Ticker Similarity", "Summary": reasoning.cross_ticker_text},
        {"Context": "Lead-Lag", "Summary": reasoning.lead_lag_text},
        {"Context": "Anomaly", "Summary": reasoning.anomaly_text},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_integrated_reasoning_summary(
    *,
    reasoning,
    multiclass_prediction: pd.Series | None,
    multiclass_text: str,
    risk_multiclass_text: str,
    risk_validation_text: str,
    risk_conflict_text: str,
    news_impact_text: str,
) -> None:
    st.subheader("Integrated Reasoning Summary")
    st.caption(
        "This is the short read of why the current prediction looks this way. "
        "Open the detail sections below when you want to inspect each component."
    )

    summary_cols = st.columns(4)
    summary_cols[0].metric(
        "Overall",
        f"{_integrated_direction_label(reasoning.prediction_direction)} / {reasoning.confidence_level}",
    )
    summary_cols[1].metric("Binary alignment", _integrated_direction_label(reasoning.prediction_direction))
    summary_cols[2].metric("Multiclass alignment", _integrated_multiclass_alignment(reasoning, multiclass_prediction))
    summary_cols[3].metric("Risk posture", _integrated_risk_posture(reasoning))

    if reasoning.confidence_level == "High":
        st.success(_compact_reason_text(reasoning.summary))
    elif reasoning.confidence_level == "Medium":
        st.info(_compact_reason_text(reasoning.summary))
    else:
        st.warning(_compact_reason_text(reasoning.summary))

    summary_rows = [
        {
            "Area": "Binary / probability",
            "What it says": _compact_reason_text(reasoning.probability_text, limit=180),
        },
        {
            "Area": "Multiclass",
            "What it says": _compact_reason_text(multiclass_text, limit=180),
        },
        {
            "Area": "Risk",
            "What it says": _compact_reason_text(
                " ".join(part for part in (reasoning.risk_text, risk_multiclass_text) if part),
                limit=220,
            ),
        },
        {
            "Area": "News / narrative",
            "What it says": _compact_reason_text(
                " ".join(
                    part
                    for part in (
                        reasoning.news_text,
                        reasoning.news_alignment_text,
                        news_impact_text,
                        reasoning.narrative_text,
                    )
                    if part
                ),
                limit=240,
            ),
        },
        {
            "Area": "Market / context",
            "What it says": _compact_reason_text(
                " ".join(
                    part
                    for part in (
                        reasoning.external_text,
                        reasoning.regime_text,
                        reasoning.similarity_text,
                        reasoning.cross_ticker_text,
                        reasoning.lead_lag_text,
                        reasoning.anomaly_text,
                    )
                    if part
                ),
                limit=260,
            ),
        },
    ]
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    attention = [
        text
        for text in (
            reasoning.conflict_text,
            risk_conflict_text,
            risk_validation_text,
        )
        if text
    ]
    if attention:
        st.caption("Attention")
        for message in attention[:3]:
            st.warning(_compact_reason_text(message, limit=220))


def _render_integrated_reasoning_details(evidence: pd.DataFrame) -> None:
    groups = {
        "Detailed reasoning components": (
            "Probability",
            "SHAP",
            "Multiclass",
            "Agreement",
        ),
        "News reasoning details": (
            "Earnings Event",
            "News",
            "News Alignment",
            "News Impact Validation",
            "Narrative",
        ),
        "Market / sector reasoning details": (
            "External Market",
            "Market Regime",
            "Cross-Ticker",
            "Lead-Lag",
            "Similar Patterns",
        ),
        "Risk / anomaly reasoning details": (
            "Risk Multiclass",
            "Risk Validation",
            "Risk Conflict",
            "Surge / Crash",
            "Anomaly",
        ),
    }
    for title, names in groups.items():
        subset = evidence[evidence["Evidence"].isin(names)].copy()
        if subset.empty:
            continue
        with st.expander(title, expanded=False):
            st.dataframe(subset, use_container_width=True, hide_index=True)
    with st.expander("Raw integrated reasoning records", expanded=False):
        st.dataframe(evidence, use_container_width=True, hide_index=True)


def _integrated_multiclass_alignment(reasoning, multiclass_prediction: pd.Series | None) -> str:
    if multiclass_prediction is None:
        return "-"
    label = _prediction_class_name(
        multiclass_prediction.get("predicted_class"),
        multiclass_prediction.get("target"),
    )
    if label in {"up", "surge"} and reasoning.prediction_direction == "up":
        return "Aligned"
    if label in {"down", "crash"} and reasoning.prediction_direction == "down":
        return "Aligned"
    if label == "flat":
        return "Flat / cautious"
    return "Mixed"


def _integrated_risk_posture(reasoning) -> str:
    risk_text = str(getattr(reasoning, "risk_text", "") or "").lower()
    if "crash" in risk_text and ("high" in risk_text or "warning" in risk_text):
        return "Crash caution"
    if "surge" in risk_text and ("high" in risk_text or "warning" in risk_text):
        return "Surge watch"
    if risk_text:
        return "Normal watch"
    return "-"


def _compact_reason_text(value: object, *, limit: int = 320) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "-"
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3].rstrip()}..."


def _augment_risk_scores_with_multiclass(
    risk_scores: dict[str, object],
    multiclass_prediction: pd.Series | None,
    *,
    profile: str | None = None,
    weight_mode: str | None = None,
) -> dict[str, object]:
    if multiclass_prediction is None:
        return risk_scores

    result = dict(risk_scores)
    model_context = _risk_multiclass_setting_text(profile, weight_mode)
    predicted_label = _prediction_class_name(
        multiclass_prediction.get("predicted_class"),
        multiclass_prediction.get("target"),
    )

    surge_probability = safe_float(multiclass_prediction.get("probability_surge"))
    up_probability = safe_float(multiclass_prediction.get("probability_up"))
    crash_probability = safe_float(multiclass_prediction.get("probability_crash"))
    down_probability = safe_float(multiclass_prediction.get("probability_down"))

    result = _add_multiclass_risk_signal(
        result,
        key="surge",
        label="surge",
        model_context=model_context,
        tail_probability=surge_probability,
        direction_probability=up_probability,
        predicted_label=predicted_label,
        direction_label="up",
    )
    result = _add_multiclass_risk_signal(
        result,
        key="crash",
        label="crash",
        model_context=model_context,
        tail_probability=crash_probability,
        direction_probability=down_probability,
        predicted_label=predicted_label,
        direction_label="down",
    )
    return result


def _add_multiclass_risk_signal(
    risk_scores: dict[str, object],
    *,
    key: str,
    label: str,
    model_context: str,
    tail_probability: float | None,
    direction_probability: float | None,
    predicted_label: str,
    direction_label: str,
) -> dict[str, object]:
    existing = risk_scores.get(key)
    if existing is None:
        return risk_scores

    extra_points = 0
    reasons: list[str] = []
    features: list[str] = []

    if tail_probability is not None and tail_probability >= 0.20:
        points = 20 if tail_probability >= 0.35 else 10
        extra_points += points
        reasons.append(
            f"{model_context} assigns {_format_percent(tail_probability)} probability to {label}."
        )
        features.append(f"probability_{label}")

    if predicted_label == label:
        extra_points += 15
        reasons.append(f"{model_context} directly predicts {label}.")
        features.append("multiclass_predicted_class")

    if direction_probability is not None and direction_probability >= 0.35:
        points = 10 if direction_probability >= 0.45 else 5
        extra_points += points
        reasons.append(
            f"{model_context} shows elevated {direction_label} probability "
            f"({_format_percent(direction_probability)})."
        )
        features.append(f"probability_{direction_label}")

    if extra_points <= 0:
        return risk_scores

    current_score = safe_int(getattr(existing, "score", 0)) or 0
    new_score = min(100, current_score + extra_points)
    merged_reasons = list(getattr(existing, "reasons", []) or []) + reasons
    merged_features = list(getattr(existing, "features", []) or []) + features
    updated = dict(risk_scores)
    updated[key] = type(existing)(
        name=getattr(existing, "name", label.title()),
        score=new_score,
        level=_risk_level_from_score(new_score),
        reasons=merged_reasons,
        features=sorted(set(merged_features)),
    )
    return updated


def _risk_level_from_score(score: int) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _multiclass_reasoning_text(multiclass_prediction: pd.Series | None) -> str:
    if multiclass_prediction is None:
        return "No saved 5-class prediction is available yet."

    probability_map = _multiclass_probability_map(multiclass_prediction)
    predicted_label = _prediction_class_name(
        multiclass_prediction.get("predicted_class"),
        multiclass_prediction.get("target"),
    )
    confidence = safe_float(multiclass_prediction.get("confidence"))
    surge = probability_map.get("surge")
    crash = probability_map.get("crash")

    parts = [
        f"5-class model currently favors {predicted_label} with {_format_percent(confidence)} confidence."
    ]
    if crash is not None and crash >= 0.20:
        parts.append("Crash probability is elevated, so downside risk should be watched.")
    if surge is not None and surge >= 0.20:
        parts.append("Surge probability is elevated, so upside volatility is also possible.")
    if predicted_label == "flat" and max(surge or 0.0, crash or 0.0) >= 0.15:
        parts.append("The top class is flat, but tail-risk classes are not negligible.")
    return " ".join(parts)


def _risk_multiclass_reasoning_text(
    risk_multiclass_prediction: pd.Series | None,
    profile: str | None,
    weight_mode: str | None,
) -> str:
    model_context = _risk_multiclass_setting_text(profile, weight_mode)
    if risk_multiclass_prediction is None:
        return f"{model_context} is selected for Surge / Crash Risk, but no saved prediction is available yet."

    probability_map = _multiclass_probability_map(risk_multiclass_prediction)
    predicted_label = _prediction_class_name(
        risk_multiclass_prediction.get("predicted_class"),
        risk_multiclass_prediction.get("target"),
    )
    return (
        f"Crash Risk and Surge Risk use {model_context} as an auxiliary rare-event signal. "
        f"It predicts {predicted_label}; crash probability is {_format_percent(probability_map.get('crash'))}, "
        f"down probability is {_format_percent(probability_map.get('down'))}, "
        f"up probability is {_format_percent(probability_map.get('up'))}, "
        f"and surge probability is {_format_percent(probability_map.get('surge'))}."
    )


def _risk_multiclass_conflict_text(
    *,
    prediction_label: str,
    up_probability,
    down_probability,
    risk_multiclass_prediction: pd.Series | None,
) -> str:
    if risk_multiclass_prediction is None:
        return ""

    binary_direction = _binary_direction_label(prediction_label, up_probability, down_probability)
    if binary_direction not in {"up", "down"}:
        return ""

    probability_map = _multiclass_probability_map(risk_multiclass_prediction)
    predicted_label = _prediction_class_name(
        risk_multiclass_prediction.get("predicted_class"),
        risk_multiclass_prediction.get("target"),
    )
    crash_probability = probability_map.get("crash") or 0.0
    surge_probability = probability_map.get("surge") or 0.0
    down_probability_value = probability_map.get("down") or 0.0
    up_probability_value = probability_map.get("up") or 0.0

    if binary_direction == "up" and (
        predicted_label == "crash" or crash_probability >= 0.20 or down_probability_value >= 0.45
    ):
        return (
            "Binary prediction leans upward, but the risk multiclass model shows downside warning "
            f"signals: crash {_format_percent(crash_probability)} and down {_format_percent(down_probability_value)}."
        )
    if binary_direction == "down" and (
        predicted_label == "surge" or surge_probability >= 0.20 or up_probability_value >= 0.45
    ):
        return (
            "Binary prediction leans downward, but the risk multiclass model shows upside warning "
            f"signals: surge {_format_percent(surge_probability)} and up {_format_percent(up_probability_value)}."
        )
    return ""


def _current_risk_percentile_context(
    ticker: str,
    data: pd.DataFrame,
    settings,
    risk_scores: dict[str, object],
    risk_multiclass_profile: str | None,
    risk_multiclass_weight_mode: str | None,
    risk_bucket_mode: str | None,
) -> dict[str, dict[str, object]]:
    try:
        module = _risk_validation_module()
        model_name = _multiclass_model_name(risk_multiclass_profile, risk_multiclass_weight_mode)
        saved_multiclass = _read_prediction_history(
            settings.paths.database_path,
            ticker=ticker,
            target="next_close_multiclass",
            model_name=model_name,
        )
        frame = module.build_risk_validation_frame(
            data,
            multiclass_predictions=saved_multiclass,
        )
        if frame.empty:
            return {}
        context: dict[str, dict[str, object]] = {}
        for risk_type in ("surge", "crash"):
            score = risk_scores.get(risk_type)
            score_value = safe_float(getattr(score, "score", None))
            percentile = module.score_percentile(frame, risk_type=risk_type, score=score_value)
            fixed_bucket = module.bucket_for_score(
                frame,
                risk_type=risk_type,
                score=score_value,
                bucket_mode="fixed_33_66",
            )
            relative_bucket = module.bucket_for_score(
                frame,
                risk_type=risk_type,
                score=score_value,
                bucket_mode=risk_bucket_mode,
            )
            context[risk_type] = {
                "score": score_value,
                "percentile": percentile,
                "fixed_bucket": fixed_bucket,
                "relative_bucket": relative_bucket,
                "bucket_mode": module.normalize_bucket_mode(risk_bucket_mode),
            }
        return context
    except Exception as error:
        LOGGER.debug("Could not build current risk percentile context: %s", error, exc_info=True)
        return {}


def _binary_direction_label(prediction_label: str, up_probability, down_probability) -> str:
    if prediction_label in {"up", "down"}:
        return prediction_label
    up = safe_float(up_probability)
    down = safe_float(down_probability)
    if up is None or down is None:
        return "unknown"
    return "up" if up >= down else "down"


def _risk_validation_context_text(
    ticker: str,
    data: pd.DataFrame,
    settings,
    risk_multiclass_profile: str | None,
    risk_multiclass_weight_mode: str | None,
    risk_bucket_mode: str | None = None,
    risk_percentiles: dict[str, dict[str, object]] | None = None,
) -> str:
    try:
        module = _risk_validation_module()
        model_name = _multiclass_model_name(risk_multiclass_profile, risk_multiclass_weight_mode)
        saved_multiclass = _read_prediction_history(
            settings.paths.database_path,
            ticker=ticker,
            target="next_close_multiclass",
            model_name=model_name,
        )
        frame = module.build_risk_validation_frame(
            data,
            multiclass_predictions=saved_multiclass,
        )
        if frame.empty:
            return ""
        surge_summary = module.summarize_risk_buckets(
            frame,
            risk_type="surge",
            bucket_mode=risk_bucket_mode,
        )
        crash_summary = module.summarize_risk_buckets(
            frame,
            risk_type="crash",
            bucket_mode=risk_bucket_mode,
        )
        messages = module.build_risk_validation_messages(surge_summary, crash_summary)
        calibration_parts: list[str] = []
        risk_percentiles = risk_percentiles or {}
        surge_percentile = safe_float(risk_percentiles.get("surge", {}).get("percentile"))
        crash_percentile = safe_float(risk_percentiles.get("crash", {}).get("percentile"))
        surge_relative = risk_percentiles.get("surge", {}).get("relative_bucket")
        surge_fixed = risk_percentiles.get("surge", {}).get("fixed_bucket")
        crash_relative = risk_percentiles.get("crash", {}).get("relative_bucket")
        crash_fixed = risk_percentiles.get("crash", {}).get("fixed_bucket")
        if surge_percentile is not None and surge_percentile >= 0.90:
            calibration_parts.append(
                f"Current Surge Risk is in the top 10% of historical scores "
                f"(percentile {_format_percent(surge_percentile)}, {surge_relative} by selected calibration)."
            )
        if crash_percentile is not None and crash_percentile >= 0.90:
            calibration_parts.append(
                f"Current Crash Risk is in the top 10% of historical scores "
                f"(percentile {_format_percent(crash_percentile)}, {crash_relative} by selected calibration)."
            )
        if surge_fixed != surge_relative and surge_relative:
            calibration_parts.append(f"Surge Risk is {surge_fixed} by fixed thresholds but {surge_relative} by selected calibration.")
        if crash_fixed != crash_relative and crash_relative:
            calibration_parts.append(f"Crash Risk is {crash_fixed} by fixed thresholds but {crash_relative} by selected calibration.")
        fixed_high_missing = (
            _risk_high_count(surge_summary) == 0 or _risk_high_count(crash_summary) == 0
        )
        if fixed_high_missing and calibration_parts:
            calibration_parts.insert(
                0,
                "Fixed thresholds produce few or no High samples, so quantile calibration gives better context.",
            )
        return " ".join(messages[:2] + calibration_parts)
    except Exception as error:
        LOGGER.debug("Could not build risk validation context: %s", error, exc_info=True)
        return ""


def _render_multiclass_prediction_section(
    latest_multiclass: pd.Series,
    metrics: dict[str, object],
    latest_binary: pd.Series | None,
    threshold_profile: str,
    weight_mode: str,
) -> None:
    st.subheader("Multiclass Prediction")
    probability_map = _multiclass_probability_map(latest_multiclass)
    prediction_label = _prediction_class_name(
        latest_multiclass.get("predicted_class"),
        latest_multiclass.get("target"),
    )
    confidence = safe_float(latest_multiclass.get("confidence"))

    cols = st.columns(3)
    cols[0].metric("5-class prediction", prediction_label)
    cols[1].metric("Confidence", _format_percent(confidence))
    cols[2].metric("Crash / Surge", _multiclass_tail_risk_label(probability_map))
    st.caption(
        "Threshold profile: "
        f"{threshold_profile} / weight mode: {weight_mode} / "
        f"model: {latest_multiclass.get('model_name', '-')}"
    )

    st.plotly_chart(
        _build_multiclass_probability_chart(probability_map),
        use_container_width=True,
    )

    metric_values = metrics.get("metrics", {}) if metrics else {}
    if metric_values:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Accuracy", _format_percent(metric_values.get("accuracy")))
        metric_cols[1].metric("Macro F1", _format_percent(metric_values.get("macro_f1")))
        metric_cols[2].metric("Weighted F1", _format_percent(metric_values.get("weighted_f1")))
        metric_cols[3].metric("Surge recall", _format_percent(metric_values.get("surge_recall")))
        metric_cols[4].metric("Crash recall", _format_percent(metric_values.get("crash_recall")))
        per_class = metric_values.get("per_class", {})
        if isinstance(per_class, dict) and per_class:
            with st.expander("5-class per-class metrics"):
                per_class_frame = pd.DataFrame(per_class).T.reset_index(names="class")
                st.dataframe(per_class_frame, use_container_width=True, hide_index=True)

    _render_multiclass_binary_comparison(latest_multiclass, latest_binary)

    top_features = _parse_top_features(latest_multiclass.get("top_features"))
    if not top_features:
        top_features = metrics.get("top_features", []) if metrics else []
    if top_features:
        st.caption("5-class feature importance top 10")
        st.plotly_chart(_build_feature_contribution_chart(top_features), use_container_width=True)
        st.dataframe(pd.DataFrame(top_features), use_container_width=True)

    st.caption("5-class rationale")
    st.write(_multiclass_prediction_rationale(latest_multiclass, top_features))


def _render_multiclass_binary_comparison(
    latest_multiclass: pd.Series,
    latest_binary: pd.Series | None,
) -> None:
    rows = [
        {
            "Model": "LightGBM multiclass",
            "Prediction": _prediction_class_name(
                latest_multiclass.get("predicted_class"),
                latest_multiclass.get("target"),
            ),
            "Confidence": _format_percent(latest_multiclass.get("confidence")),
            "Up probability": _format_percent(latest_multiclass.get("probability_up")),
            "Down probability": _format_percent(latest_multiclass.get("probability_down")),
        }
    ]
    if latest_binary is not None:
        rows.append(
            {
                "Model": "LightGBM binary",
                "Prediction": _prediction_class_name(
                    latest_binary.get("predicted_class"),
                    latest_binary.get("target"),
                ),
                "Confidence": _format_percent(latest_binary.get("confidence")),
                "Up probability": _format_percent(latest_binary.get("probability_up")),
                "Down probability": _format_percent(latest_binary.get("probability_down")),
            }
        )
    st.caption("Binary model comparison")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_external_market_context(external_context) -> None:
    if external_context is None:
        return

    metrics = getattr(external_context, "metrics", {}) or {}
    st.subheader("External Market")
    cols = st.columns(4)
    cols[0].metric("USDJPY", getattr(external_context, "usdjpy_status", "Unavailable"))
    cols[1].metric("Market Trend", getattr(external_context, "market_trend", "Unavailable"))
    cols[2].metric("USDJPY 1D", _format_percent(metrics.get("usdjpy_return_1d")))
    cols[3].metric("Nikkei 1D", _format_percent(metrics.get("n225_return_1d")))

    sector_cols = st.columns(4)
    sector_cols[0].metric(
        "Auto Sector Trend",
        getattr(external_context, "auto_sector_trend", "Unavailable"),
    )
    sector_cols[1].metric("Tesla Trend", getattr(external_context, "tesla_trend", "Unavailable"))
    sector_cols[2].metric("Honda Trend", getattr(external_context, "honda_trend", "Unavailable"))
    sector_cols[3].metric("Sector Return", _format_percent(metrics.get("sector_mean_return")))

    summary = getattr(external_context, "summary", "")
    if summary:
        st.caption(summary)


def _render_earnings_event_context(earnings_context) -> None:
    if earnings_context is None:
        return

    metrics = getattr(earnings_context, "metrics", {}) or {}
    st.subheader("Earnings Event")
    cols = st.columns(4)
    cols[0].metric(
        "Next earnings",
        getattr(earnings_context, "next_earnings_date", None) or "-",
    )
    days_to = getattr(earnings_context, "days_to_earnings", None)
    cols[1].metric("Days to earnings", "-" if days_to is None else days_to)
    cols[2].metric("Event state", getattr(earnings_context, "event_state", "normal"))
    cols[3].metric(
        "Post mode",
        "On" if getattr(earnings_context, "post_earnings_mode", False) else "Off",
    )

    event_cols = st.columns(3)
    event_cols[0].metric(
        "Pre-event vol",
        _format_percent(metrics.get("earnings_pre_volatility")),
    )
    event_cols[1].metric("Post gap", _format_percent(metrics.get("earnings_post_gap")))
    event_cols[2].metric(
        "Post trend",
        _format_percent(metrics.get("earnings_after_trend_5")),
    )

    summary = getattr(earnings_context, "summary", "")
    if summary:
        st.caption(summary)


def _render_news_context(news_context) -> None:
    if news_context is None:
        return

    metrics = getattr(news_context, "metrics", {}) or {}
    st.subheader("News")
    cols = st.columns(4)
    cols[0].metric("News 1D", _format_number(metrics.get("news_count_1d"), digits=0))
    cols[1].metric("News 3D", _format_number(metrics.get("news_count_3d"), digits=0))
    cols[2].metric("Sentiment 3D", _format_number(metrics.get("avg_sentiment_3d"), digits=2))
    cols[3].metric("Importance 3D", _format_number(metrics.get("max_importance_3d"), digits=2))

    impact_cols = st.columns(4)
    impact_cols[0].metric(
        "Sentiment Trend 3D",
        _format_number(metrics.get("sentiment_trend_3d"), digits=2),
    )
    impact_cols[1].metric(
        "Weighted Sentiment",
        _format_number(metrics.get("importance_weighted_sentiment"), digits=2),
    )
    impact_cols[2].metric(
        "Sector News",
        _format_number(metrics.get("sector_news_strength"), digits=2),
    )
    shock_label = "Negative" if metrics.get("negative_news_shock") else "Positive" if metrics.get("positive_news_shock") else "None"
    impact_cols[3].metric("News Shock", shock_label)

    impact_summary = getattr(news_context, "impact_summary", "")
    if impact_summary:
        st.caption("News Impact Summary")
        st.write(impact_summary)

    summary = getattr(news_context, "summary", "")
    if summary:
        st.caption(summary)

    sentiment_history = getattr(news_context, "sentiment_history", []) or []
    if sentiment_history:
        st.plotly_chart(_build_news_sentiment_chart(sentiment_history), use_container_width=True)

    important_items = getattr(news_context, "important_items", []) or []
    if important_items:
        st.caption("Recent important news")
        important = pd.DataFrame(important_items)
        important = _news_display_frame(important)
        st.dataframe(important, use_container_width=True, hide_index=True)

    recent_items = getattr(news_context, "recent_items", []) or []
    if recent_items:
        display = pd.DataFrame(recent_items)
        display = _news_display_frame(display)
        st.caption("Recent news")
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.caption("No recent manual news items are registered.")


def _news_display_frame(display: pd.DataFrame) -> pd.DataFrame:
    display_columns = [
        "date",
        "ticker",
        "title",
        "source",
        "source_type",
        "source_credibility_score",
        "source_market_impact_weight",
        "category",
        "sentiment_score",
        "importance_score",
        "sentiment_reason",
        "importance_reason",
        "summary",
        "url",
    ]
    display = display[[column for column in display_columns if column in display.columns]].copy()
    for column in ("source_credibility_score", "source_market_impact_weight"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    return display.rename(
        columns={
            "date": "Date",
            "ticker": "Ticker",
            "title": "Title",
            "source": "Source",
            "source_type": "Source Type",
            "source_credibility_score": "Source Credibility",
            "source_market_impact_weight": "Source Weight",
            "category": "Category",
            "sentiment_score": "Sentiment",
            "importance_score": "Importance",
            "sentiment_reason": "Sentiment Reason",
            "importance_reason": "Importance Reason",
            "summary": "Summary",
            "url": "URL",
        }
    )


def _build_news_sentiment_chart(sentiment_history: list[dict[str, object]]) -> go.Figure:
    history = pd.DataFrame(sentiment_history)
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    for column in (
        "avg_sentiment_1d",
        "avg_sentiment_3d",
        "importance_weighted_sentiment",
        "sector_news_strength",
    ):
        if column in history.columns:
            history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)

    figure = go.Figure()
    series = {
        "avg_sentiment_1d": "Sentiment 1D",
        "avg_sentiment_3d": "Sentiment 3D",
        "importance_weighted_sentiment": "Weighted",
        "sector_news_strength": "Sector",
    }
    for column, label in series.items():
        if column not in history.columns:
            continue
        figure.add_trace(
            go.Scatter(
                x=history["date"],
                y=history[column],
                mode="lines+markers",
                name=label,
            )
        )
    figure.add_hline(y=0, line_width=1, line_dash="dot", line_color="#777")
    figure.update_layout(
        title="News Sentiment Trend",
        height=260,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return figure


def _render_narrative_context(narrative_context) -> None:
    if narrative_context is None:
        return

    metrics = getattr(narrative_context, "metrics", {}) or {}
    st.subheader("Current Market Narratives")
    cols = st.columns(4)
    cols[0].metric("Dominant", getattr(narrative_context, "dominant_label", "None"))
    cols[1].metric(
        "Sentiment",
        _format_number(metrics.get("dominant_narrative_sentiment"), digits=2),
    )
    cols[2].metric(
        "Importance",
        _format_number(metrics.get("dominant_narrative_importance"), digits=2),
    )
    cols[3].metric(
        "Trend",
        _format_number(metrics.get("dominant_narrative_trend"), digits=2),
    )

    pressure_cols = st.columns(2)
    pressure_cols[0].metric(
        "Bullish Pressure",
        _format_number(metrics.get("narrative_bullish_pressure"), digits=2),
    )
    pressure_cols[1].metric(
        "Bearish Pressure",
        _format_number(metrics.get("narrative_bearish_pressure"), digits=2),
    )

    summary = getattr(narrative_context, "summary", "")
    if summary:
        st.caption(summary)

    themes = getattr(narrative_context, "themes", []) or []
    if themes:
        display = pd.DataFrame(themes)
        display = display[
            ["label", "news_count", "sentiment", "importance", "trend"]
        ].rename(
            columns={
                "label": "Theme",
                "news_count": "News Count",
                "sentiment": "Sentiment",
                "importance": "Importance",
                "trend": "Trend",
            }
        )
        st.dataframe(display.head(9), use_container_width=True, hide_index=True)


def _render_market_regime_context(regime_state, *, show_heading: bool = True) -> None:
    if regime_state is None:
        return

    name = getattr(regime_state, "name", "Unknown")
    level = getattr(regime_state, "level", "Unknown")
    metrics = getattr(regime_state, "metrics", {}) or {}

    if show_heading:
        st.subheader("Market Regime")
    cols = st.columns(4)
    cols[0].metric("Current Regime", name, level)
    cols[1].metric("Volatility", _format_percent(metrics.get("volatility")))
    cols[2].metric("ATR", _format_percent(metrics.get("atr_pct")))
    cols[3].metric("Volume ratio", _format_number(metrics.get("volume_ratio")))

    reasons = getattr(regime_state, "reasons", []) or []
    features = getattr(regime_state, "features", []) or []
    if reasons:
        st.caption("Regime reasons")
        for reason in reasons[:4]:
            st.write(f"- {reason}")
    if features:
        st.caption(f"Regime features: {', '.join(features)}")


def _render_surge_crash_scores(
    risk_scores: dict[str, object],
    *,
    ticker: str,
    risk_multiclass_prediction: pd.Series | None = None,
    risk_multiclass_profile: str | None = None,
    risk_multiclass_weight_mode: str | None = None,
    risk_percentiles: dict[str, dict[str, object]] | None = None,
) -> None:
    st.caption("Surge / Crash early-warning scores")
    _render_risk_multiclass_signal(
        ticker,
        risk_multiclass_prediction,
        risk_multiclass_profile,
        risk_multiclass_weight_mode,
    )
    _render_current_risk_percentiles(risk_percentiles)
    cols = st.columns(2)
    for column, key, label in (
        (cols[0], "surge", "Surge Risk Score"),
        (cols[1], "crash", "Crash Risk Score"),
    ):
        score = risk_scores.get(key)
        if score is None:
            column.info(f"{label}: unavailable")
            continue
        level = getattr(score, "level", "Unknown")
        value = getattr(score, "score", 0)
        column.metric(label, f"{value}/100", level)
        reasons = getattr(score, "reasons", [])
        features = getattr(score, "features", [])
        if reasons:
            column.write("Reasons")
            for reason in reasons:
                column.write(f"- {reason}")
        if features:
            column.caption(f"Related features: {', '.join(features)}")


def _render_current_risk_percentiles(
    risk_percentiles: dict[str, dict[str, object]] | None,
) -> None:
    rows = _current_risk_percentile_display_rows(risk_percentiles)
    if not rows:
        st.caption("Current Risk percentile is unavailable until Risk Validation has enough history.")
        return

    st.caption("Current Risk percentile against the historical Risk Score distribution")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    differences = [
        f"{row['Risk']}: fixed threshold is {row['Fixed Bucket']}, "
        f"selected calibration is {row['Selected Bucket']}."
        for row in rows
        if row["Fixed Bucket"] not in {"-", row["Selected Bucket"]}
    ]
    if differences:
        st.caption(
            "Fixed and calibrated bucket interpretations differ: " + " ".join(differences)
        )


def _current_risk_percentile_display_rows(
    risk_percentiles: dict[str, dict[str, object]] | None,
) -> list[dict[str, object]]:
    if not risk_percentiles:
        return []

    rows: list[dict[str, object]] = []
    for risk_type, label in (("surge", "Surge Risk"), ("crash", "Crash Risk")):
        item = risk_percentiles.get(risk_type) or {}
        percentile = safe_float(item.get("percentile"))
        score = safe_float(item.get("score"))
        if score is None and percentile is None:
            continue
        rows.append(
            {
                "Risk": label,
                "Score": _format_number(score, digits=0),
                "Historical Percentile": _format_percent(percentile),
                "Historical Position": _risk_percentile_position_text(percentile),
                "Fixed Bucket": item.get("fixed_bucket") or "-",
                "Selected Bucket": item.get("relative_bucket") or "-",
                "Bucket Mode": item.get("bucket_mode") or "-",
            }
        )
    return rows


def _risk_percentile_position_text(percentile) -> str:
    value = safe_float(percentile)
    if value is None:
        return "-"
    if value >= 0.95:
        return "Top 5%"
    if value >= 0.90:
        return "Top 10%"
    if value >= 0.85:
        return "Top 15%"
    if value >= 0.75:
        return "Top 25%"
    if value <= 0.10:
        return "Bottom 10%"
    if value <= 0.25:
        return "Bottom 25%"
    return "Middle range"


def _render_risk_multiclass_signal(
    ticker: str,
    risk_multiclass_prediction: pd.Series | None,
    risk_multiclass_profile: str | None,
    risk_multiclass_weight_mode: str | None,
) -> None:
    profile = risk_multiclass_profile or DEFAULT_RISK_MULTICLASS_OPTION[0]
    weight_mode = risk_multiclass_weight_mode or DEFAULT_RISK_MULTICLASS_OPTION[1]
    st.caption(
        "Risk signal source: "
        f"{profile} / {weight_mode}. This is a rare-event warning input, not the main prediction."
    )
    if risk_multiclass_prediction is None:
        st.info("Risk multiclass prediction is not available for this setting yet.")
        st.code(
            f"python main.py train-multiclass --ticker {ticker} --threshold-profile {profile} --weight-mode {weight_mode}\n"
            f"python main.py predict-multiclass --ticker {ticker} --threshold-profile {profile} --weight-mode {weight_mode}",
            language="powershell",
        )
        return

    probability_map = _multiclass_probability_map(risk_multiclass_prediction)
    predicted_label = _prediction_class_name(
        risk_multiclass_prediction.get("predicted_class"),
        risk_multiclass_prediction.get("target"),
    )
    cols = st.columns(5)
    cols[0].metric("Risk profile", profile)
    cols[1].metric("Weight mode", weight_mode)
    cols[2].metric("Risk 5-class label", predicted_label)
    cols[3].metric("Crash probability", _format_percent(probability_map.get("crash")))
    cols[4].metric("Surge probability", _format_percent(probability_map.get("surge")))


def _integrated_direction_label(direction: str) -> str:
    labels = {
        "up": "Up",
        "down": "Down",
        "mixed": "Mixed",
        "unknown": "Unknown",
    }
    return labels.get(direction, str(direction))


def _build_shap_direction_chart(items: list[dict[str, object]], direction: str) -> go.Figure:
    frame = pd.DataFrame(items).copy()
    if frame.empty:
        return go.Figure()

    frame["label"] = frame.get("label", frame["feature"])
    frame["value"] = frame["value"].map(safe_float)
    frame["shap_value"] = frame["shap_value"].map(safe_float)
    frame = frame.dropna(subset=["shap_value"])
    if frame.empty:
        return go.Figure()

    ascending = direction == "up"
    frame = frame.sort_values("shap_value", ascending=ascending)
    color = "#3a8f6b" if direction == "up" else "#d95f5f"

    fig = go.Figure(
        go.Bar(
            x=frame["shap_value"],
            y=frame["label"],
            orientation="h",
            marker_color=color,
            customdata=frame[["feature", "value"]],
            hovertemplate=(
                "Feature=%{customdata[0]}<br>"
                "Value=%{customdata[1]}<br>"
                "SHAP=%{x:.5f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="#555555", line_width=1)
    fig.update_layout(
        height=max(300, 40 + len(frame) * 34),
        xaxis_title="SHAP value toward Up class",
        yaxis_title="",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _shap_display_table(items: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(items).copy()
    if frame.empty:
        return frame
    display_columns = [
        column
        for column in ("label", "feature", "value", "shap_value", "direction")
        if column in frame.columns
    ]
    display = frame[display_columns].copy()
    display = display.rename(
        columns={
            "label": "Label",
            "feature": "Feature",
            "value": "Value",
            "shap_value": "SHAP value",
            "direction": "Direction",
        }
    )
    return display


def _render_data_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("Data Overview")
    st.caption(
        "Data is for freshness, row counts, missing values, and source status. "
        "Use Prediction for forecasts and Evaluation for validation."
    )

    try:
        report = _load_health_report_for_ui(ticker)
    except Exception as error:
        LOGGER.debug("Could not load health report for Data tab: %s", error, exc_info=True)
        report = {}

    raw_data = _load_raw_data_for_ui(str(settings.paths.raw_data_dir / f"{ticker}.parquet"))
    overview = _build_data_overview(ticker, data, raw_data, report)
    feature_latest_for_events = (
        overview["latest_feature_date"] if overview["latest_feature_date"] != "-" else None
    )
    events_dir = _events_module().default_events_dir(settings.paths.raw_data_dir.parent)
    earnings_summary = _load_earnings_event_summary_for_ui(
        str(events_dir),
        ticker,
        feature_latest_for_events,
    )
    earnings_status = str(earnings_summary.get("status") or overview["earnings_status"])
    warning_summary = report.get("warning_summary") or {}
    news_item = _health_item(report, "News", "News CSV quality")
    news_details = _health_details(news_item)
    topix_item = _health_item(report, "External data", "TOPIX parquet")

    first_row = st.columns(4)
    first_row[0].metric("Ticker", overview["ticker"])
    first_row[1].metric("Latest raw date", overview["latest_raw_date"])
    first_row[2].metric("Latest feature date", overview["latest_feature_date"])
    first_row[3].metric("Data freshness", overview["data_freshness"])

    second_row = st.columns(4)
    second_row[0].metric("Raw rows", overview["raw_rows"])
    second_row[1].metric("Feature rows", overview["feature_rows"])
    second_row[2].metric("News rows", overview["news_rows"])
    second_row[3].metric("Latest news date", overview["latest_news_date"])

    third_row = st.columns(4)
    third_row[0].metric("Latest fetched_at", overview["latest_fetched_at"])
    third_row[1].metric("Earnings data", earnings_status)
    third_row[2].metric("External data", overview["external_status"])
    third_row[3].metric("Health status", overview["health_status"])

    warning_row = st.columns(3)
    warning_row[0].metric(
        "Action required",
        warning_summary.get("action_required_count", 0),
    )
    warning_row[1].metric(
        "Known missed dates",
        warning_summary.get("known_missed_date_count", warning_summary.get("known_missed_count", 0)),
    )
    warning_row[2].metric(
        "Known OK",
        warning_summary.get("known_ok_count", 0),
    )

    action_required = warning_summary.get("action_required") or []
    if action_required:
        st.error(
            "Action required: "
            + ", ".join(str(item.get("name") or "Unknown warning") for item in action_required)
        )
    else:
        st.success("Action required: 0. Data operation can continue.")
    known_ok = warning_summary.get("known_ok") or []
    if known_ok:
        st.info(
            "Known OK / operation can continue: "
            + ", ".join(str(item.get("name") or "Known warning") for item in known_ok)
        )
    known_missed = warning_summary.get("known_missed") or []
    known_missed_details = _known_missed_details_frame(known_missed)
    if not known_missed_details.empty:
        with st.expander("Known missed live date details", expanded=False):
            st.caption(
                "These are known live prediction gaps with explicit reasons. They remain visible but are "
                "not Action required."
            )
            st.dataframe(known_missed_details, use_container_width=True, hide_index=True)

    with st.expander("How to read data status", expanded=False):
        st.write("- Raw data: source price data.")
        st.write("- Feature data: processed features used for prediction.")
        st.write("- External data: support data such as FX, indices, and related tickers.")
        st.write("- News data: source data for news features.")
        st.write("- Earnings / events: source data for event-related features.")
        st.write("- The latest feature date should follow the raw data date.")
        st.write("- If health-check Action required is 0, daily operation can usually continue.")

    st.subheader("Data Freshness Summary")
    freshness_rows = [
        {
            "Data source": "Raw price",
            "Status": _health_status(_health_item(report, "Data", "Toyota raw data")),
            "Latest date": overview["latest_raw_date"],
            "Rows": overview["raw_rows"],
        },
        {
            "Data source": "Feature data",
            "Status": _health_status(_health_item(report, "Data", "Feature summary")),
            "Latest date": overview["latest_feature_date"],
            "Rows": overview["feature_rows"],
        },
        {
            "Data source": "News",
            "Status": _health_status(news_item),
            "Latest date": overview["latest_news_date"],
            "Rows": overview["news_rows"],
        },
        {
            "Data source": "Earnings / events",
            "Status": earnings_status,
            "Latest date": earnings_summary.get("latest_event_date") or "-",
            "Rows": earnings_summary.get("rows") or 0,
        },
        {
            "Data source": "External market",
            "Status": overview["external_status"],
            "Latest date": overview["latest_feature_date"],
            "Rows": "-",
        },
    ]
    st.dataframe(pd.DataFrame(freshness_rows), width="stretch", hide_index=True)

    st.subheader("Raw Price Data")
    raw_cols = st.columns(3)
    raw_cols[0].metric("Status", _health_status(_health_item(report, "Data", "Toyota raw data")))
    raw_cols[1].metric("Rows", overview["raw_rows"])
    raw_cols[2].metric("Latest date", overview["latest_raw_date"])
    with st.expander("Raw price data sample / file details", expanded=False):
        if raw_data.empty:
            st.info("Raw price data is not available.")
        else:
            raw_display_columns = [
                column
                for column in ("date", "ticker", "open", "high", "low", "close", "volume")
                if column in raw_data.columns
            ]
            st.dataframe(
                raw_data[raw_display_columns].tail(100),
                width="stretch",
                hide_index=True,
            )
        st.dataframe(
            _health_section_frame(report, "Data"),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Feature Data")
    feature_cols = st.columns(4)
    feature_cols[0].metric("Status", _health_status(_health_item(report, "Data", "Feature summary")))
    feature_cols[1].metric("Rows", overview["feature_rows"])
    feature_cols[2].metric("Columns", len(data.columns))
    feature_cols[3].metric("Latest date", overview["latest_feature_date"])

    numeric_columns = [
        column
        for column in data.columns
        if column not in {"date", "ticker"} and pd.api.types.is_numeric_dtype(data[column])
    ]
    technical_columns = available_technical_columns(data)
    priority_columns = ["open", "high", "low", "close", "volume", *technical_columns]
    display_columns = ["date", "ticker", *[c for c in priority_columns if c in numeric_columns]]
    with st.expander("Feature snapshot", expanded=False):
        st.dataframe(
            data[display_columns].tail(100),
            width="stretch",
            hide_index=True,
        )

    st.subheader("External Market Data")
    external_frame = _health_section_frame(report, "External data")
    if external_frame.empty:
        st.info("External market status is not available.")
    else:
        st.dataframe(external_frame, width="stretch", hide_index=True)
    if topix_item is not None and _health_status(topix_item) == "WARN":
        st.info(
            "TOPIX: fallback via "
            f"{_health_detail(topix_item, 'fallback_symbols')} / Known OK. "
            "The application continues with the configured fallback data."
        )

    st.subheader("News Data")
    news_cols = st.columns(4)
    news_cols[0].metric("Status", _health_status(news_item))
    news_cols[1].metric("Rows", overview["news_rows"])
    news_cols[2].metric("Latest news date", overview["latest_news_date"])
    news_cols[3].metric(
        "Unknown sources",
        _format_number(news_details.get("unknown_source_count"), digits=0),
    )
    unknown_source_count = safe_int(news_details.get("unknown_source_count")) or 0
    if unknown_source_count > 0:
        st.error(
            f"Unknown news sources: {unknown_source_count}. Review Operations > Unknown Sources."
        )
    else:
        st.success("Unknown news sources: 0")
    with st.expander("News quality details", expanded=False):
        st.dataframe(
            pd.DataFrame(
                [
                    {"Item": "Latest fetched_at", "Value": overview["latest_fetched_at"]},
                    {"Item": "Duplicate rows", "Value": news_details.get("duplicate_count", "-")},
                    {"Item": "Invalid dates", "Value": news_details.get("invalid_date_count", "-")},
                    {
                        "Item": "Mojibake suspects",
                        "Value": news_details.get("mojibake_suspect_count", "-"),
                    },
                    {
                        "Item": "Missing category",
                        "Value": news_details.get("missing_category_count", "-"),
                    },
                    {
                        "Item": "Source credibility range",
                        "Value": news_details.get("source_credibility_range_status", "-"),
                    },
                    {
                        "Item": "Source weight range",
                        "Value": news_details.get("source_weight_range_status", "-"),
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Earnings / Event Data")
    earnings_item = _health_item(report, "Data", "Earnings CSV")
    earnings_cols = st.columns(4)
    earnings_cols[0].metric("CSV status", earnings_status)
    earnings_cols[1].metric(
        "Feature columns",
        _health_status(_health_item(report, "Data", "Earnings feature columns")),
    )
    earnings_cols[2].metric(
        "Latest event date",
        earnings_summary.get("latest_event_date") or "-",
    )
    earnings_cols[3].metric(
        "Next known event date",
        earnings_summary.get("next_known_event_date") or "-",
    )
    _render_health_item_message(earnings_item)
    st.caption(str(earnings_summary.get("message") or "Earnings CSV status is unavailable."))

    event_status_cols = st.columns(4)
    event_status_cols[0].metric(
        "days_to_earnings",
        _earnings_distance_status(data, "days_to_earnings"),
    )
    event_status_cols[1].metric(
        "days_after_earnings",
        _earnings_distance_status(data, "days_after_earnings"),
    )
    event_status_cols[2].metric(
        "Source column",
        "available" if earnings_summary.get("source_column_available") else "not available",
    )
    event_status_cols[3].metric(
        "Future events",
        _format_number(earnings_summary.get("future_event_count"), digits=0),
    )

    if earnings_status in {"header_only", "missing", "empty"}:
        st.info(
            "Earnings event dates are not registered for the current period, so "
            "days_to_earnings / days_after_earnings are treated as not applicable. "
            "This is a data-readiness gap, not a model failure."
        )
    elif earnings_status in {"invalid", "needs_review"}:
        st.warning(
            "One or more earnings CSV files need review before earnings-distance "
            "features should be interpreted."
        )

    with st.expander("Earnings CSV status / file details", expanded=False):
        st.dataframe(
            _earnings_event_files_frame(earnings_summary),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "build-features checks candidate files in order and uses the first file "
            "with usable event rows. Header-only files are skipped."
        )

    with st.expander("Safe earnings CSV template", expanded=False):
        st.caption(
            "Do not add estimated dates. Register only official IR or otherwise "
            "confirmed dates with a source note."
        )
        st.code(
            "date,ticker,event_type,fiscal_year,quarter,period,source,note\n"
            "2026-05-08,7203.T,earnings,2026,Q4,FY2026,"
            "https://example.com/official-ir,official confirmed date",
            language="csv",
        )
        st.caption(
            "Current feature generation requires date or earnings_date. event_type is "
            "optional and defaults to earnings; extra columns such as ticker, source, "
            "fiscal_year, and quarter are safe for future manual tracking."
        )

    st.subheader("Feature Missingness Classification")
    missingness = classify_feature_missingness(data)
    missingness_cols = st.columns(4)
    missingness_cols[0].metric("Total feature columns", missingness.total_feature_columns)
    missingness_cols[1].metric("Columns with missing", missingness.columns_with_missing)
    missingness_cols[2].metric("Expected missing", missingness.expected_missing_count)
    missingness_cols[3].metric("Needs review", missingness.needs_review_count)
    missingness_detail_cols = st.columns(3)
    missingness_detail_cols[0].metric("Optional missing", missingness.optional_missing_count)
    missingness_detail_cols[1].metric("Not applicable", missingness.not_applicable_count)
    missingness_detail_cols[2].metric(
        "Latest row missing",
        missingness.latest_row_missing_count,
    )

    if missingness.needs_review_count:
        st.warning(
            f"{missingness.needs_review_count} feature columns need review. "
            "Check the latest-row and data-source gap reasons below."
        )
    else:
        st.success(
            "No feature columns currently need review. Missing values are mainly "
            "calculation-window, source-history, future-target, or not-applicable gaps."
        )

    classification_groups = (
        ("needs_review", "Needs review", missingness.needs_review_count > 0),
        ("expected_missing", "Expected missing", False),
        ("optional_missing", "Optional missing", False),
        ("not_applicable", "Not applicable", False),
    )
    for classification, label, expanded in classification_groups:
        rows = missingness.rows[
            missingness.rows["classification"].eq(classification)
        ]
        with st.expander(f"{label} ({len(rows)})", expanded=expanded):
            if rows.empty:
                st.caption("No columns in this classification.")
            else:
                st.dataframe(
                    _feature_missingness_display_frame(rows),
                    width="stretch",
                    hide_index=True,
                )

    st.subheader("Data Quality / Missing Values")
    missing_frame = _data_missing_values_frame(data)
    quality_cols = st.columns(3)
    quality_cols[0].metric("Rows with any missing value", int(data.isna().any(axis=1).sum()))
    quality_cols[1].metric("Columns with missing values", len(missing_frame))
    quality_cols[2].metric(
        "Total missing values",
        int(data.isna().sum().sum()),
    )
    st.caption(
        "Missing values can include indicator warm-up periods and optional context columns. "
        "Open the details to identify persistent or unexpected gaps."
    )
    with st.expander("Missing value details", expanded=False):
        if missing_frame.empty:
            st.success("No missing values were found in the current feature frame.")
        else:
            st.dataframe(missing_frame, width="stretch", hide_index=True)

    with st.expander("Raw tables / file details", expanded=False):
        st.caption("Health-check file and source details")
        for section in ("Data", "External data", "News"):
            st.write(f"**{section}**")
            frame = _health_section_frame(report, section)
            if frame.empty:
                st.caption("No details available.")
            else:
                st.dataframe(frame, width="stretch", hide_index=True)


@st.cache_data(ttl=60, show_spinner=False)
def _load_raw_data_for_ui(path: str) -> pd.DataFrame:
    raw_path = Path(path)
    if not raw_path.exists():
        return pd.DataFrame()
    try:
        return read_parquet(raw_path)
    except Exception as error:
        LOGGER.debug("Could not load raw data for Data tab: %s", error, exc_info=True)
        return pd.DataFrame()


def _build_data_overview(
    ticker: str,
    feature_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    report: dict[str, object],
) -> dict[str, object]:
    raw_latest = _latest_data_date(raw_data)
    feature_latest = _latest_data_date(feature_data)
    news_details = _health_details(_health_item(report, "News", "News CSV quality"))
    warning_summary = report.get("warning_summary") or {}
    return {
        "ticker": ticker,
        "latest_raw_date": raw_latest or "-",
        "latest_feature_date": feature_latest or "-",
        "raw_rows": len(raw_data),
        "feature_rows": len(feature_data),
        "news_rows": safe_int(news_details.get("rows")) or 0,
        "latest_news_date": _format_date(news_details.get("latest_news_date")),
        "latest_fetched_at": _format_datetime(news_details.get("latest_fetched_at")),
        "earnings_status": _health_status(_health_item(report, "Data", "Earnings CSV")),
        "external_status": _health_section_status(report, "External data"),
        "data_freshness": _data_freshness_label(raw_latest, feature_latest),
        "health_status": str(report.get("overall_status") or "UNKNOWN"),
        "action_required_count": warning_summary.get("action_required_count", 0),
        "known_missed_count": warning_summary.get("known_missed_count", 0),
        "known_ok_count": warning_summary.get("known_ok_count", 0),
    }


def _data_freshness_label(raw_latest: str | None, feature_latest: str | None) -> str:
    raw_date = pd.to_datetime(raw_latest, errors="coerce")
    feature_date = pd.to_datetime(feature_latest, errors="coerce")
    if pd.isna(raw_date) or pd.isna(feature_date):
        return "not available"
    lag_days = int((raw_date - feature_date).days)
    if lag_days <= 0:
        return "OK"
    if lag_days <= 3:
        return "stale"
    return "missing update"


def _data_missing_values_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing Rate"])
    missing = data.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return pd.DataFrame(columns=["Column", "Missing Count", "Missing Rate"])
    return pd.DataFrame(
        {
            "Column": missing.index,
            "Missing Count": missing.astype(int).values,
            "Missing Rate": (missing / len(data)).map(lambda value: f"{value:.2%}").values,
        }
    )


def _feature_missingness_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    display["missing_rate"] = display["missing_rate"].map(
        lambda value: f"{float(value):.2%}"
    )
    display["latest_missing"] = display["latest_missing"].map(
        lambda value: "Yes" if bool(value) else "No"
    )
    return display.rename(
        columns={
            "column": "Column",
            "classification": "Classification",
            "reason": "Reason",
            "missing_count": "Missing Count",
            "missing_rate": "Missing Rate",
            "latest_missing": "Latest Missing",
            "leading_missing": "Leading Missing",
            "trailing_missing": "Trailing Missing",
        }
    )


@st.cache_data(ttl=60, show_spinner=False)
def _load_earnings_event_summary_for_ui(
    events_dir: str,
    ticker: str,
    as_of_date: str | None,
) -> dict[str, object]:
    try:
        return _events_module().summarize_earnings_event_files(
            events_dir,
            ticker,
            as_of_date=as_of_date,
        )
    except Exception as error:
        LOGGER.debug("Could not summarize earnings event CSV files: %s", error, exc_info=True)
        return {
            "status": "not_available",
            "message": "Earnings CSV status could not be summarized.",
            "files": [],
            "rows": 0,
        }


def _earnings_distance_status(data: pd.DataFrame, column: str) -> str:
    if data.empty or column not in data.columns:
        return "missing column"
    frame = data.sort_values("date") if "date" in data.columns else data
    value = safe_float(frame.iloc[-1].get(column))
    if value is None:
        return "not applicable"
    return f"{int(value)} days"


def _earnings_event_files_frame(summary: dict[str, object]) -> pd.DataFrame:
    rows = summary.get("files")
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(
            columns=[
                "Path",
                "Exists",
                "Status",
                "Rows",
                "Columns",
                "Earliest event date",
                "Latest event date",
                "Next known event date",
                "Invalid dates",
                "Duplicate dates",
                "Future events",
                "Ticker mismatches",
                "Build candidate",
                "Build active",
                "Message",
            ]
        )
    frame = pd.DataFrame(rows)
    if "columns" in frame.columns:
        frame["columns"] = frame["columns"].map(
            lambda value: ", ".join(value) if isinstance(value, list) else value
        )
    if "missing_required_columns" in frame.columns:
        frame["missing_required_columns"] = frame["missing_required_columns"].map(
            lambda value: ", ".join(value) if isinstance(value, list) else value
        )
    keep_columns = [
        "path",
        "exists",
        "status",
        "rows",
        "columns",
        "earliest_event_date",
        "latest_event_date",
        "next_known_event_date",
        "required_columns_present",
        "missing_required_columns",
        "invalid_date_count",
        "duplicate_event_date_count",
        "future_event_count",
        "ticker_mismatch_count",
        "source_column_available",
        "event_type_column_available",
        "build_features_candidate",
        "build_features_active",
        "message",
    ]
    display = frame[[column for column in keep_columns if column in frame.columns]].copy()
    return display.rename(
        columns={
            "path": "Path",
            "exists": "Exists",
            "status": "Status",
            "rows": "Rows",
            "columns": "Columns",
            "earliest_event_date": "Earliest event date",
            "latest_event_date": "Latest event date",
            "next_known_event_date": "Next known event date",
            "required_columns_present": "Required columns present",
            "missing_required_columns": "Missing required columns",
            "invalid_date_count": "Invalid dates",
            "duplicate_event_date_count": "Duplicate dates",
            "future_event_count": "Future events",
            "ticker_mismatch_count": "Ticker mismatches",
            "source_column_available": "Source column",
            "event_type_column_available": "Event type column",
            "build_features_candidate": "Build candidate",
            "build_features_active": "Build active",
            "message": "Message",
        }
    )


def _render_operations_tab(ticker: str, data: pd.DataFrame, settings) -> None:
    st.subheader("System Status")
    st.caption(
        "Read-only operational snapshot from health-check. Use this when daily prediction, "
        "data refresh, or saved predictions look suspicious."
    )

    controls = st.columns([1, 3])
    if controls[0].button("Refresh health-check", key="operations_refresh_health"):
        _load_health_report_for_ui.clear()
    controls[1].code(f"python main.py health-check --ticker {ticker}", language="powershell")

    try:
        with st.spinner("Running health-check..."):
            report = _load_health_report_for_ui(ticker)
    except Exception as error:
        LOGGER.exception("Streamlit health-check rendering failed.")
        st.error("Health-check could not be loaded.")
        st.caption(str(error))
        return

    overall_status = str(report.get("overall_status") or "UNKNOWN")
    generated_at = report.get("generated_at")
    if overall_status == "OK":
        st.success("Overall status: OK")
    elif overall_status == "WARN":
        st.warning("Overall status: WARN. The system is usable, but some items need attention.")
    else:
        st.error("Overall status: FAIL. Check the failed item before relying on automation.")
    st.caption(f"Generated at: {generated_at or '-'}")

    python_item = _health_item(report, "Python / venv", ".venv python")
    stock_ai_item = _health_item(report, "Python / venv", "stock_ai import")
    sqlite_item = _health_item(report, "SQLite", "predictions table")
    daily_item = _health_item(report, "daily_predict", "Latest run")
    daily_news_item = _health_item(report, "daily_predict", "Optional news latest run")
    scheduler_item = _health_item(report, "daily_predict", "Windows scheduled task")
    binary_item = _health_item(report, "SQLite", "Latest binary prediction")
    multiclass_item = _health_item(report, "SQLite", "Latest multiclass prediction")
    missing_item = _health_item(report, "SQLite", "Missing live dates")
    topix_item = _health_item(report, "External data", "TOPIX parquet")

    overview = st.columns(5)
    overview[0].metric("Overall status", overall_status)
    overview[1].metric("Python / venv", _health_status(python_item))
    overview[2].metric("stock_ai import", _health_status(stock_ai_item))
    overview[3].metric("SQLite", _health_status(sqlite_item))
    overview[4].metric("Models", _health_section_status(report, "Models"))

    _render_health_warnings_summary(report)
    _render_append_only_health_visibility(settings.paths.database_path, ticker)

    st.subheader("Prediction / Daily Automation")
    missing_details = _health_details(missing_item)
    daily_details = _health_details(daily_item)
    prediction_cols = st.columns(4)
    prediction_cols[0].metric("Latest binary feature date", _health_detail(binary_item, "date"))
    prediction_cols[1].metric("Latest multiclass feature date", _health_detail(multiclass_item, "date"))
    prediction_cols[2].metric(
        "Post-start missing dates",
        _format_number(
            missing_details.get(
                "missing_prediction_count_after_start",
                missing_details.get("missing_prediction_count"),
            ),
            digits=0,
        ),
    )
    prediction_cols[3].metric("daily_predict", _health_status(daily_item))

    _render_health_item_message(daily_item)
    daily_cols = st.columns(2)
    daily_cols[0].caption("Latest daily_predict success")
    daily_cols[0].code(str(daily_details.get("latest_success") or "-"), language="text")
    daily_cols[1].caption("Latest daily_predict error")
    daily_cols[1].code(str(daily_details.get("latest_error") or "-"), language="text")

    if missing_item is not None:
        st.caption(_health_message(missing_item))
        st.caption(
            "live_prediction_start_date: "
            f"{missing_details.get('live_prediction_start_date') or '-'}; "
            "pre-operation dates excluded: "
            f"{_format_number(missing_details.get('pre_start_excluded_date_count', missing_details.get('ignored_pre_start_missing_dates')), digits=0)}"
        )
    st.caption(
        "daily_predict currently uses original_5_1 by default. To test another multiclass "
        "profile manually, run scripts/daily_predict.ps1 with "
        "-MulticlassThresholdProfile practical_3_0_5 or quantile. Rare-event weighting can "
        "also be tested with -MulticlassWeightMode balanced or manual_tail_boost."
    )
    st.caption(
        "Prediction Risk Scores can use a different saved multiclass setting, such as "
        "practical_3_0_5 / balanced, as an auxiliary warning signal while daily_predict "
        "keeps the conservative original_5_1 / none default."
    )
    _render_task_scheduler_status(scheduler_item)
    _render_today_signal_config_status()

    st.subheader("External Data / Models")
    external_cols = st.columns(4)
    external_cols[0].metric("External data", _health_section_status(report, "External data"))
    external_cols[1].metric("TOPIX status", _health_status(topix_item))
    external_cols[2].metric("TOPIX source", _health_detail(topix_item, "data_status"))
    external_cols[3].metric("Model files", _health_section_status(report, "Models"))

    if topix_item is not None and _health_status(topix_item) == "WARN":
        st.warning(_health_message(topix_item))

    st.subheader("News Fetch Status")
    news_status = _news_fetch_status(ticker, settings)
    news_health = news_status.get("health", {}) or {}
    daily_news_details = _health_details(daily_news_item)
    news_cols = st.columns(4)
    news_cols[0].metric("News rows", _format_number(news_status.get("count"), digits=0))
    news_cols[1].metric("Latest news date", _format_date(news_status.get("latest_date")))
    news_cols[2].metric("Latest fetched_at", news_status.get("latest_fetched_at") or "-")
    news_cols[3].metric("News CSV", "OK" if news_status.get("exists") else "Missing")
    daily_news_cols = st.columns(4)
    daily_news_cols[0].metric(
        "FetchNews in latest daily job",
        _format_enabled(daily_news_details.get("fetch_news_enabled")),
    )
    daily_news_cols[1].metric("Daily news step", _health_status(daily_news_item))
    daily_news_cols[2].metric("News Health", news_health.get("status") or "-")
    daily_news_cols[3].metric(
        "Latest fetch warning",
        "Yes" if daily_news_details.get("latest_news_warning") else "No",
    )
    st.caption(f"Latest daily news profile mode: {daily_news_details.get('news_profile_mode') or '-'}")
    st.caption(f"Latest daily news profile: {daily_news_details.get('news_query_profile') or '-'}")
    st.caption(
        "Latest daily daily-profiles mode: "
        f"{_format_enabled(daily_news_details.get('daily_news_profiles'))}"
    )
    st.caption(
        "Latest daily all-profiles mode: "
        f"{_format_enabled(daily_news_details.get('all_news_profiles'))}"
    )
    st.caption(f"News CSV path: {news_status.get('path') or '-'}")
    st.caption(f"News Health: {news_health.get('status') or '-'} - {news_health.get('message') or '-'}")
    st.caption(
        "daily_predict keeps news fetching optional. Run scripts/daily_predict.ps1 with "
        "-FetchNews to fetch news before build-features."
    )
    latest_news_warning = daily_news_details.get("latest_news_warning")
    if latest_news_warning:
        st.warning(f"Latest fetch-news warning: {latest_news_warning}")
    _render_data_collection_monitor(
        report=report,
        news_status=news_status,
        daily_item=daily_item,
        daily_news_item=daily_news_item,
        binary_item=binary_item,
        multiclass_item=multiclass_item,
    )
    health_cols = st.columns(4)
    health_cols[0].metric("Duplicate rows", _format_number(news_health.get("duplicate_count"), digits=0))
    health_cols[1].metric("Invalid dates", _format_number(news_health.get("invalid_date_count"), digits=0))
    health_cols[2].metric(
        "Mojibake suspects",
        _format_number(news_health.get("mojibake_suspect_count"), digits=0),
    )
    health_cols[3].metric("Empty titles", _format_number(news_health.get("empty_title_count"), digits=0))
    range_cols = st.columns(4)
    range_cols[0].metric("Sentiment range", news_health.get("sentiment_range_status") or "-")
    range_cols[1].metric("Importance range", news_health.get("importance_range_status") or "-")
    range_cols[2].metric("Empty source", _format_number(news_health.get("empty_source_count"), digits=0))
    range_cols[3].metric("Empty url", _format_number(news_health.get("empty_url_count"), digits=0))
    reason_cols = st.columns(3)
    reason_cols[0].metric(
        "Missing category",
        _format_number(news_health.get("missing_category_count"), digits=0),
    )
    reason_cols[1].metric(
        "Missing sentiment reason",
        _format_number(news_health.get("missing_sentiment_reason_count"), digits=0),
    )
    reason_cols[2].metric(
        "Missing importance reason",
        _format_number(news_health.get("missing_importance_reason_count"), digits=0),
    )
    source_cols = st.columns(4)
    source_cols[0].metric(
        "Missing source credibility",
        _format_number(news_health.get("missing_source_credibility_count"), digits=0),
    )
    source_cols[1].metric(
        "Unknown source",
        _format_number(news_health.get("unknown_source_count"), digits=0),
    )
    source_cols[2].metric("Source credibility range", news_health.get("source_credibility_range_status") or "-")
    source_cols[3].metric("Source weight range", news_health.get("source_weight_range_status") or "-")
    _render_unknown_sources_section(news_status)
    source_distribution = news_health.get("source_type_distribution") or {}
    if source_distribution:
        st.caption("Source type distribution")
        st.dataframe(
            pd.DataFrame(
                [
                    {"source_type": key, "count": value}
                    for key, value in source_distribution.items()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
    issue_rows = news_health.get("issue_rows") or []
    if issue_rows:
        st.caption("News quality issue rows")
        st.dataframe(pd.DataFrame(issue_rows), use_container_width=True, hide_index=True)
    st.code(
        f'python main.py fetch-news --ticker {ticker} --query "Toyota stock"',
        language="powershell",
    )
    st.code(
        f"python main.py refresh-news-scores --ticker {ticker} --dry-run",
        language="powershell",
    )
    profile_rows = news_status.get("query_profiles") or []
    if profile_rows:
        st.caption("Available news query profiles")
        profile_frame = pd.DataFrame(profile_rows)
        st.dataframe(profile_frame, use_container_width=True, hide_index=True)
        if "enabled_for_daily" in profile_frame.columns:
            daily_profiles = profile_frame[
                profile_frame["enabled_for_daily"].astype(str).str.lower().eq("true")
            ].copy()
            st.caption("Daily-enabled news query profiles")
            st.dataframe(daily_profiles, use_container_width=True, hide_index=True)
        if "group" in profile_frame.columns:
            st.caption("News query profiles by group")
            group_frame = (
                profile_frame.groupby("group", dropna=False)["name"]
                .apply(lambda values: ", ".join(str(value) for value in values))
                .reset_index(name="profiles")
            )
            st.dataframe(group_frame, use_container_width=True, hide_index=True)
        st.code(
            f"python main.py fetch-news --ticker {ticker} --query-profile toyota_stock",
            language="powershell",
        )
        st.code(
            f"python main.py fetch-news --ticker {ticker} --daily-profiles --limit 10",
            language="powershell",
        )
        st.code(
            f"python main.py fetch-news --ticker {ticker} --all-profiles --limit 10",
            language="powershell",
        )
    _render_news_profile_quality_section(news_status)
    _render_daily_news_profile_recommendation_section(ticker, data, settings)
    source_profile_rows = news_status.get("source_profiles") or []
    if source_profile_rows:
        st.caption("Available news source credibility profiles")
        st.dataframe(pd.DataFrame(source_profile_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Source-type News Impact details are available in Evaluation > "
            "News Impact Validation > Source Types. Daily Brief shows a short summary only."
        )

    external_frame = _health_section_frame(report, "External data")
    if not external_frame.empty:
        st.caption("External data status")
        st.dataframe(external_frame, use_container_width=True, hide_index=True)

    model_frame = _health_section_frame(report, "Models")
    if not model_frame.empty:
        st.caption("Model file status")
        st.dataframe(model_frame, use_container_width=True, hide_index=True)

    st.subheader("daily_predict.log")
    log_tail = daily_details.get("log_tail") or _read_daily_log_tail(100)
    if log_tail:
        st.text_area(
            "Latest 100 log lines",
            value="\n".join(str(line) for line in log_tail[-100:]),
            height=420,
            disabled=True,
            key="operations_daily_predict_log_tail",
        )
    else:
        st.info("daily_predict.log is not available yet.")


def _render_health_warnings_summary(report: dict[str, object]) -> None:
    summary = report.get("warning_summary") or {}
    action_required = summary.get("action_required", []) or []
    known_missed = summary.get("known_missed", []) or []
    known_ok = summary.get("known_ok", []) or []
    known_missed_sections = safe_int(summary.get("known_missed_section_count"))
    if known_missed_sections is None:
        known_missed_sections = len(known_missed)
    known_missed_dates = safe_int(summary.get("known_missed_date_count"))
    if known_missed_dates is None:
        known_missed_dates = _known_missed_date_count_from_items(known_missed)

    st.subheader("Health Warnings Summary")
    cols = st.columns(6)
    cols[0].metric("Overall status", report.get("overall_status") or "-")
    cols[1].metric("Action required", _format_number(len(action_required), digits=0))
    cols[2].metric("Known missed dates", _format_number(known_missed_dates, digits=0))
    cols[3].metric("Known missed sections", _format_number(known_missed_sections, digits=0))
    cols[4].metric("Known OK", _format_number(len(known_ok), digits=0))
    cols[5].metric("Last health-check", _format_datetime(report.get("generated_at")))

    if action_required:
        st.warning("Some health-check items need review before relying on automation.")
        st.dataframe(
            _health_warning_display_frame(action_required),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success(
            "Action required: 0. Daily operation can continue if Known missed and Known OK items match expectations."
        )

    if known_missed:
        with st.expander("Known missed live dates", expanded=True):
            st.caption(
                "These live prediction gaps remain visible because no live prediction was actually saved. "
                "They are not counted as Action required because they have an explicit reason and action."
            )
            details = _known_missed_details_frame(known_missed)
            if not details.empty:
                st.dataframe(details, use_container_width=True, hide_index=True)
            st.dataframe(
                _health_warning_display_frame(known_missed),
                use_container_width=True,
                hide_index=True,
            )

    if known_ok:
        with st.expander("Known OK / operation can continue", expanded=True):
            st.caption(
                "These WARN items remain visible, but they have an accepted fallback or are outside "
                "the active operation window."
            )
            st.dataframe(
                _health_warning_display_frame(known_ok),
                use_container_width=True,
                hide_index=True,
            )


def _render_append_only_health_visibility(database_path, ticker: str) -> None:
    visibility = build_append_only_health_visibility(database_path, ticker=ticker)
    st.subheader("Append-only Health Visibility")
    st.caption(
        "append-only履歴は監査用です。現時点ではhealth-checkやPrediction History metricsの判定元ではありません。"
    )
    st.caption(
        "daily_predictへの自動dual-writeはまだ有効化していません。snapshot比較は監査目的です。"
        "mismatchが出た場合も、現段階では手動確認対象です。"
    )

    status = str(visibility.get("visibility_status") or "-")
    if status == "append_only_mismatch_detected_info":
        st.info("Append-only mismatch was detected as informational visibility. It is not Action required yet.")
    elif status == "append_only_empty":
        st.info("Append-only tables are initialized but empty. This is informational only.")
    elif status == "append_only_not_initialized":
        st.info("Append-only tables are not initialized. This is informational only.")
    elif visibility.get("append_only_vs_snapshot_matches"):
        st.success("Append-only latest official records match the current predictions snapshot.")
    else:
        st.info("Append-only visibility is available.")

    cols = st.columns(5)
    cols[0].metric("Visibility status", status)
    cols[1].metric("Runs", _format_number(visibility.get("prediction_runs_count"), digits=0))
    cols[2].metric("Records", _format_number(visibility.get("prediction_records_count"), digits=0))
    cols[3].metric("Latest official", _format_number(visibility.get("latest_official_records_count"), digits=0))
    cols[4].metric("Snapshot match", _format_enabled(visibility.get("append_only_vs_snapshot_matches")))

    latest_cols = st.columns(5)
    latest_cols[0].metric("Latest run type", visibility.get("latest_run_type") or "-")
    latest_cols[1].metric("Latest feature date", _format_date(visibility.get("latest_feature_date")))
    latest_cols[2].metric("Mismatch count", _format_number(visibility.get("mismatch_count"), digits=0))
    latest_cols[3].metric("Pending", _format_number(visibility.get("pending_count"), digits=0))
    latest_cols[4].metric("Unevaluated", _format_number(visibility.get("unevaluated_count"), digits=0))

    st.caption(f"Latest run id: {visibility.get('latest_run_id') or '-'}")
    st.caption(f"Latest run timestamp: {_format_datetime(visibility.get('latest_run_timestamp'))}")
    st.caption(
        "Visibility classifications: "
        + ", ".join(str(item) for item in visibility.get("visibility_classifications", []))
    )
    st.caption(
        "Current source of truth remains the existing predictions snapshot; append-only is a parallel audit log."
    )


def _health_warning_display_frame(items: list[dict[str, object]]) -> pd.DataFrame:
    rows = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "Section": item.get("section"),
                "Name": item.get("name"),
                "Status": item.get("status"),
                "Known missed details": _health_known_missed_summary(item),
                "Reason": item.get("reason"),
                "Message": item.get("message"),
            }
        )
    return pd.DataFrame(rows)


def _health_known_missed_summary(item: dict[str, object]) -> str:
    details = item.get("details")
    if not isinstance(details, dict):
        return ""
    rows = details.get("known_missed_live_dates") or []
    if not isinstance(rows, list) or not rows:
        return ""
    parts = []
    for row in rows[:3]:
        if not isinstance(row, dict):
            continue
        date = row.get("missing_date") or row.get("date") or "-"
        reason = row.get("known_reason_code") or row.get("inferred_reason") or "-"
        action = row.get("known_action") or "-"
        allow_backfill = row.get("known_allow_backfill")
        backfill = "yes" if allow_backfill is True else "no" if allow_backfill is False else "-"
        parts.append(f"{date}: {reason}; action={action}; backfill={backfill}")
    if len(rows) > 3:
        parts.append(f"+{len(rows) - 3} more")
    return " | ".join(parts)


def _known_missed_date_count_from_items(items: list[dict[str, object]]) -> int:
    details = _known_missed_details_frame(items)
    if not details.empty:
        return len(details)
    return len(items)


def _known_missed_details_frame(items: list[dict[str, object]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        details = item.get("details")
        if not isinstance(details, dict):
            continue
        missed_rows = details.get("known_missed_live_dates") or []
        if not isinstance(missed_rows, list):
            continue
        for row in missed_rows:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "Feature date": row.get("missing_date") or row.get("date") or "-",
                    "Reason code": row.get("known_reason_code") or row.get("reason_code") or "-",
                    "Reason": row.get("known_reason") or row.get("reason") or "-",
                    "Action": row.get("known_action") or row.get("action") or "-",
                    "Allow backfill": _format_enabled(
                        row.get("known_allow_backfill")
                        if "known_allow_backfill" in row
                        else row.get("allow_backfill")
                    ),
                    "Note": row.get("note") or row.get("recommended_action") or "-",
                }
            )
    return pd.DataFrame(rows)


def _render_data_collection_monitor(
    *,
    report: dict[str, object],
    news_status: dict[str, object],
    daily_item: dict[str, object] | None,
    daily_news_item: dict[str, object] | None,
    binary_item: dict[str, object] | None,
    multiclass_item: dict[str, object] | None,
) -> None:
    st.subheader("Data Collection Monitor")
    st.caption(
        "Use this while NewsLimit 3 is collecting history. It checks whether daily_predict, "
        "saved predictions, and news fetching are accumulating data without forcing a profile decision yet."
    )

    daily_status = _health_status(daily_item)
    daily_details = _health_details(daily_item)
    daily_news_details = _health_details(daily_news_item)
    news_health = news_status.get("health", {}) or {}
    history = _fetch_history_frame(news_status)
    daily_summary = _fetch_history_daily_summary(history)
    latest_batch = _latest_fetch_batch_summary(history)
    success_days = _daily_predict_success_streak_days()
    observation_days = _newslimit_observation_days(daily_summary)

    if daily_status == "FAIL":
        st.error("daily_predict latest run failed. Check daily_predict.log before relying on automation.")
    elif daily_status == "WARN":
        st.warning("daily_predict latest run is WARN. The app may still be usable, but review the latest log.")
    else:
        st.success("daily_predict latest run is OK.")

    prediction_cols = st.columns(4)
    prediction_cols[0].metric("Consecutive success days", _format_number(success_days, digits=0))
    prediction_cols[1].metric("Latest binary prediction", _health_detail(binary_item, "date"))
    prediction_cols[2].metric("Latest multiclass prediction", _health_detail(multiclass_item, "date"))
    prediction_cols[3].metric("fetch_history days", _format_number(len(daily_summary), digits=0))

    latest_cols = st.columns(4)
    latest_cols[0].metric("Latest fetched", _format_number(latest_batch.get("fetched"), digits=0))
    latest_cols[1].metric("Latest added", _format_number(latest_batch.get("added"), digits=0))
    latest_cols[2].metric("Latest duplicates", _format_number(latest_batch.get("duplicates"), digits=0))
    latest_cols[3].metric("Latest duplicate rate", _format_percent(latest_batch.get("duplicate_rate")))

    news_cols = st.columns(4)
    news_cols[0].metric("Unknown sources", _format_number(news_health.get("unknown_source_count"), digits=0))
    news_cols[1].metric("News rows", _format_number(news_status.get("count"), digits=0))
    news_cols[2].metric("Latest fetched_at", news_status.get("latest_fetched_at") or "-")
    news_cols[3].metric("Latest news date", _format_date(news_status.get("latest_date")))

    st.caption(f"Latest daily_predict success: {daily_details.get('latest_success') or '-'}")
    st.caption(
        "Latest NewsLimit 3 fetch batch: "
        f"fetched {_format_number(latest_batch.get('fetched'), digits=0)}, "
        f"added {_format_number(latest_batch.get('added'), digits=0)}, "
        f"duplicates {_format_number(latest_batch.get('duplicates'), digits=0)}."
    )
    st.caption(
        "Latest daily news mode: "
        f"{daily_news_details.get('news_profile_mode') or '-'} / "
        f"DailyNewsProfiles={_format_enabled(daily_news_details.get('daily_news_profiles'))}."
    )
    _render_daily_run_data_freshness(daily_details)
    if safe_int(news_health.get("unknown_source_count")):
        st.warning("Unknown news sources are present. Add source profiles before reading News Impact too strongly.")

    _render_news_quality_history(news_status)

    observation_status = "collecting" if observation_days < 3 else "reviewable"
    observation_message = (
        "NewsLimit 3 is running, but more daily history is needed before changing profile settings."
        if observation_days < 3
        else "NewsLimit 3 has enough initial observation days for a light review. Prefer 5+ days before final profile changes."
    )
    st.caption(
        f"Observation days: {observation_days}. Recommended minimum: 3-5 days. "
        f"Status: {observation_status}. {observation_message}"
    )

    if daily_summary.empty:
        st.info("No fetch_history daily summary is available yet.")
        return

    st.caption("fetch_history daily trend")
    display = _fetch_history_daily_summary_display_frame(daily_summary.tail(14))
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.plotly_chart(_build_fetch_history_daily_chart(daily_summary.tail(14)), use_container_width=True)



def _render_daily_run_data_freshness(daily_details: dict[str, object]) -> None:
    status = daily_details.get("data_freshness_status")
    has_freshness = any(
        daily_details.get(key) is not None
        for key in (
            "expected_latest_trading_date",
            "latest_raw_date_at_run",
            "latest_feature_date_at_run",
            "freshness_lag_trading_days",
            "stale_at_run_time",
        )
    )
    if not has_freshness:
        st.info(
            "Latest daily_predict log does not include data freshness details yet. "
            "They will appear after the next run with the updated script."
        )
        return

    st.caption("Daily run data freshness")
    cols = st.columns(5)
    cols[0].metric("Expected latest date", daily_details.get("expected_latest_trading_date") or "-")
    cols[1].metric("Raw date at run", daily_details.get("latest_raw_date_at_run") or "-")
    cols[2].metric("Feature date at run", daily_details.get("latest_feature_date_at_run") or "-")
    cols[3].metric(
        "Freshness lag",
        _format_number(daily_details.get("freshness_lag_trading_days"), digits=0),
    )
    cols[4].metric("Stale at run", _format_enabled(daily_details.get("stale_at_run_time")))

    message = str(daily_details.get("data_freshness_message") or "")
    if str(status or "").upper() in {"WARN", "MISSING"} or daily_details.get("stale_at_run_time") is True:
        st.warning(message or "Latest daily_predict run used stale feature data.")
    elif str(status or "").upper() == "OK":
        st.success(message or "Latest daily_predict run used current feature data.")
    else:
        st.info(message or "Data freshness status is not available for the latest run.")

    st.caption(
        "10:30 observation: Current daily run time is 10:30. Latest data freshness is "
        f"{status or '-'} and stale_at_run_time is {_format_enabled(daily_details.get('stale_at_run_time'))}. "
        "Keep observing a few runs before deciding whether 18:00 is necessary."
    )


def _render_news_quality_history(news_status: dict[str, object]) -> None:
    quality_history = news_status.get("quality_history") or {}
    summary = quality_history.get("summary") or {}
    batch_rows = quality_history.get("batches") or []
    if not summary and not batch_rows:
        return

    st.subheader("News Quality History")
    st.caption(
        "check-news results are summarized once per fetch batch. "
        "Older batches without saved quality fields are shown as not available."
    )

    latest_status = str(summary.get("latest_check_news_status") or "").upper()
    latest_unknown_count = summary.get("latest_unknown_source_count")
    if latest_status in {"WARN", "ERROR"}:
        st.error(
            f"Latest check-news status is {latest_status}: "
            f"{summary.get('latest_check_news_message') or 'Review the latest news quality result.'}"
        )
    elif latest_unknown_count is not None and safe_int(latest_unknown_count) > 0:
        st.warning(
            "Unknown news sources were detected in the latest checked batch. "
            "Review them before relying strongly on source-weighted interpretation."
        )
    elif latest_status == "OK":
        st.success("Latest check-news result is OK.")
    else:
        st.info("Latest check-news result is not available for the newest stored batch.")

    latest_cols = st.columns(4)
    latest_cols[0].metric("Latest check-news", latest_status or "not available")
    latest_cols[1].metric(
        "Latest unknown sources",
        _optional_count_text(latest_unknown_count),
    )
    latest_cols[2].metric(
        "Quality-observed batches",
        _format_number(summary.get("observed_batch_count"), digits=0),
    )
    latest_cols[3].metric(
        "Quality-unrecorded batches",
        _format_number(summary.get("unrecorded_batch_count"), digits=0),
    )

    status_cols = st.columns(3)
    status_cols[0].metric("check-news OK", _format_number(summary.get("ok_count"), digits=0))
    status_cols[1].metric("check-news WARN", _format_number(summary.get("warn_count"), digits=0))
    status_cols[2].metric("check-news ERROR", _format_number(summary.get("error_count"), digits=0))

    latest_sources = str(summary.get("latest_unknown_sources") or "").strip()
    if latest_unknown_count is None:
        latest_sources_label = "not available"
    elif safe_int(latest_unknown_count) == 0:
        latest_sources_label = "none"
    else:
        latest_sources_label = latest_sources or "not available"
    st.caption(
        f"Latest message: {summary.get('latest_check_news_message') or 'not available'}"
    )
    st.caption(f"Latest unknown source names: {latest_sources_label}")

    if batch_rows:
        st.caption("Recent fetch batches")
        batch_frame = pd.DataFrame(batch_rows).head(14)
        st.dataframe(
            _news_quality_history_display_frame(batch_frame),
            use_container_width=True,
            hide_index=True,
        )


def _news_quality_history_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "fetched_at" in display.columns:
        display["fetched_at"] = display["fetched_at"].map(_format_datetime)
    for column in ("fetched_count", "added_count", "duplicate_count"):
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: _format_number(value, digits=0)
            )
    if "unknown_source_count" in display.columns:
        display["unknown_source_count"] = display["unknown_source_count"].map(
            _optional_count_text
        )
    for column in ("check_news_status", "check_news_message", "unknown_sources"):
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: str(value).strip() if str(value).strip() else "not available"
            )
    columns = [
        "fetched_at",
        "query_profile",
        "fetched_count",
        "added_count",
        "duplicate_count",
        "check_news_status",
        "unknown_source_count",
        "unknown_sources",
        "check_news_message",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "fetched_at": "Fetched At",
            "query_profile": "Query Profile",
            "fetched_count": "Fetched",
            "added_count": "Added",
            "duplicate_count": "Duplicates",
            "check_news_status": "Check News Status",
            "unknown_source_count": "Unknown Source Count",
            "unknown_sources": "Unknown Sources",
            "check_news_message": "Check News Message",
        }
    )


def _render_task_scheduler_status(scheduler_item: dict[str, object] | None) -> None:
    st.subheader("Windows Task Scheduler")
    st.caption(
        "Use this helper when you want Windows to run NewsLimit 3 daily collection without opening the app."
    )
    details = _health_details(scheduler_item)
    status_cols = st.columns(5)
    status_cols[0].metric("Registered", _format_enabled(details.get("registered")))
    status_cols[1].metric("Task name", details.get("task_name") or "stock-ai daily predict")
    status_cols[2].metric("Next run", _format_datetime(details.get("next_run_time")))
    status_cols[3].metric("Last run", _format_datetime(details.get("last_run_time")))
    status_cols[4].metric("Last result", str(details.get("last_result") if details.get("last_result") is not None else "-"))
    st.caption(
        "Current daily run time: "
        f"{_scheduler_time_label(details.get('next_run_time'))}. "
        "10:30 is under observation; use the next few freshness results before considering 18:00."
    )

    if scheduler_item is None:
        st.warning("Task Scheduler status is unavailable. Run health-check or use show_daily_predict_task.ps1.")
    elif _health_status(scheduler_item) == "OK":
        st.success(_health_message(scheduler_item))
    else:
        st.warning(_health_message(scheduler_item))

    st.caption("Recommended registration command")
    st.code(
        '.\\scripts\\register_daily_predict_task.ps1 -Time "10:30" -Force',
        language="powershell",
    )
    st.caption("Check or remove the scheduled task")
    st.code(
        ".\\scripts\\show_daily_predict_task.ps1\n"
        ".\\scripts\\unregister_daily_predict_task.ps1",
        language="powershell",
    )


def _scheduler_time_label(value) -> str:
    if value is None or str(value).strip() == "":
        return "-"
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%H:%M")


def _fetch_history_frame(news_status: dict[str, object]) -> pd.DataFrame:
    rows = news_status.get("fetch_history") or news_status.get("recent_fetch_history") or []
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).copy()
    if frame.empty:
        return frame
    for column in ("fetched_count", "added_count", "duplicate_count"):
        if column not in frame.columns:
            frame[column] = 0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(int)
    if "fetched_at" not in frame.columns:
        frame["fetched_at"] = ""
    frame["fetched_at"] = pd.to_datetime(frame["fetched_at"], errors="coerce", format="mixed")
    frame = frame.dropna(subset=["fetched_at"]).copy()
    if frame.empty:
        return frame
    frame["date"] = frame["fetched_at"].dt.strftime("%Y-%m-%d")
    return frame


def _fetch_history_daily_summary(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "date" not in history.columns:
        return pd.DataFrame()
    profile_agg = ("profile", "nunique") if "profile" in history.columns else ("date", "count")
    grouped = history.groupby("date", dropna=False).agg(
        fetched=("fetched_count", "sum"),
        added=("added_count", "sum"),
        duplicates=("duplicate_count", "sum"),
        profiles_count=profile_agg,
        max_profile_limit=("fetched_count", "max"),
        latest_fetched_at=("fetched_at", "max"),
    )
    result = grouped.reset_index()
    result["duplicate_rate"] = result.apply(
        lambda row: row["duplicates"] / row["fetched"] if row["fetched"] else None,
        axis=1,
    )
    result = result.sort_values("date").reset_index(drop=True)
    result["latest_fetched_at"] = result["latest_fetched_at"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return result


def _latest_fetch_batch_summary(history: pd.DataFrame) -> dict[str, object]:
    if history.empty or "fetched_at" not in history.columns:
        return {"fetched": 0, "added": 0, "duplicates": 0, "duplicate_rate": None}
    latest_at = history["fetched_at"].max()
    if pd.isna(latest_at):
        return {"fetched": 0, "added": 0, "duplicates": 0, "duplicate_rate": None}
    batch = history[history["fetched_at"] >= latest_at - pd.Timedelta(minutes=10)].copy()
    fetched = int(batch["fetched_count"].sum())
    added = int(batch["added_count"].sum())
    duplicates = int(batch["duplicate_count"].sum())
    return {
        "fetched": fetched,
        "added": added,
        "duplicates": duplicates,
        "duplicate_rate": duplicates / fetched if fetched else None,
    }


def _newslimit_observation_days(daily_summary: pd.DataFrame) -> int:
    if daily_summary.empty or "max_profile_limit" not in daily_summary.columns:
        return 0
    observed = daily_summary[pd.to_numeric(daily_summary["max_profile_limit"], errors="coerce").eq(3)]
    return int(len(observed))


def _daily_predict_success_streak_days() -> int:
    log_path = PROJECT_ROOT / "logs" / "daily_predict.log"
    if not log_path.exists():
        return 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return 0
    dates: set[pd.Timestamp] = set()
    for line in text.splitlines():
        if "daily_predict finished successfully" not in line:
            continue
        if len(line) < 11 or not line.startswith("["):
            continue
        date = pd.to_datetime(line[1:11], errors="coerce")
        if pd.notna(date):
            dates.add(date.normalize())
    if not dates:
        return 0
    current = max(dates)
    streak = 0
    while current in dates:
        streak += 1
        current = current - pd.Timedelta(days=1)
    return streak


def _fetch_history_daily_summary_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "duplicate_rate" in display.columns:
        display["duplicate_rate"] = display["duplicate_rate"].map(_format_percent)
    for column in ("fetched", "added", "duplicates", "profiles_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    columns = [
        "date",
        "fetched",
        "added",
        "duplicates",
        "duplicate_rate",
        "profiles_count",
        "latest_fetched_at",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "date": "Date",
            "fetched": "Fetched",
            "added": "Added",
            "duplicates": "Duplicates",
            "duplicate_rate": "Duplicate Rate",
            "profiles_count": "Profiles",
            "latest_fetched_at": "Latest Fetched At",
        }
    )


def _build_fetch_history_daily_chart(frame: pd.DataFrame):
    figure = go.Figure()
    if frame.empty:
        return figure
    figure.add_trace(go.Bar(x=frame["date"], y=frame["fetched"], name="Fetched", marker_color="#4c78a8"))
    figure.add_trace(go.Bar(x=frame["date"], y=frame["added"], name="Added", marker_color="#54a24b"))
    figure.add_trace(go.Bar(x=frame["date"], y=frame["duplicates"], name="Duplicates", marker_color="#e45756"))
    figure.update_layout(
        barmode="group",
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Date",
        yaxis_title="Rows",
        legend_title_text="fetch_history",
    )
    return figure


def _render_news_profile_quality_section(news_status: dict[str, object]) -> None:
    quality_rows = news_status.get("profile_quality") or []
    recommendation_rows = news_status.get("profile_recommendations") or []
    history_rows = news_status.get("recent_fetch_history") or []
    if not quality_rows and not history_rows:
        return

    st.subheader("News Profile Quality")
    st.caption(
        "This checks which query profiles are adding useful saved news and which are mostly duplicates. "
        "Use it to keep daily news fetching light."
    )
    st.caption(
        "For price moves and Risk tendencies after each profile's news, see "
        "Evaluation > News Impact Validation > Profile News Impact."
    )
    if quality_rows:
        quality = pd.DataFrame(quality_rows)
        st.caption("Profile quality summary")
        st.dataframe(
            _news_profile_quality_display_frame(quality),
            use_container_width=True,
            hide_index=True,
        )
    if recommendation_rows:
        recommendations = pd.DataFrame(recommendation_rows)
        recommendation_series = recommendations.get("recommendation", pd.Series(dtype="object"))
        review = recommendations[recommendation_series.astype(str).eq("review_needed")].copy()
        if not review.empty:
            st.warning("Some news profiles should be reviewed before keeping them in daily operations.")
            for _, row in review.head(5).iterrows():
                st.write(f"- {row.get('profile')}: {row.get('reason')}")
        st.caption("Profile operation recommendation")
        st.dataframe(
            _news_profile_recommendation_display_frame(recommendations),
            use_container_width=True,
            hide_index=True,
        )
    if history_rows:
        st.caption("Recent fetch-news history")
        st.dataframe(
            _news_fetch_history_display_frame(pd.DataFrame(history_rows)),
            use_container_width=True,
            hide_index=True,
        )


def _render_daily_news_profile_recommendation_section(
    ticker: str,
    data: pd.DataFrame,
    settings,
) -> None:
    st.subheader("Daily News Profile Recommendation")
    st.caption(
        "News Profile Quality checks fetch quality and duplicate rate. Profile News Impact checks "
        "post-news price moves and Risk tendencies. This section combines both to decide which "
        "profiles are worth keeping in daily operation."
    )
    st.caption(
        "This report is read-only. It does not rewrite configs/news_queries.yaml. "
        "Low-sample profiles are provisional research hints, not statistical conclusions."
    )
    try:
        module = _news_impact_module()
        news_dir = _news_module().default_news_dir(settings.paths.raw_data_dir.parent)
        profile_impact = module.build_profile_news_impact(data, news_dir, ticker)
        recommendations = module.build_daily_news_profile_recommendations(profile_impact)
    except Exception as error:
        LOGGER.debug(
            "Could not build daily news profile recommendations: %s",
            error,
            exc_info=True,
        )
        st.caption("Daily News Profile Recommendation is not available yet.")
        return

    if recommendations.empty:
        st.info("No daily news profile recommendation rows are available yet.")
        return

    recommended_profiles = module.recommended_daily_profile_names(recommendations)
    candidate_profiles = module.daily_news_profile_candidate_names(recommendations)
    recommended_text = ", ".join(recommended_profiles) if recommended_profiles else "-"
    candidate_text = ", ".join(candidate_profiles) if candidate_profiles else "-"

    cols = st.columns(3)
    cols[0].metric("Recommended daily profiles", _format_number(len(recommended_profiles), digits=0))
    cols[1].metric("Daily candidates", _format_number(len(candidate_profiles), digits=0))
    cols[2].metric(
        "Review / reduce rows",
        _format_number(
            len(
                recommendations[
                    recommendations["daily_operation_recommendation"].astype(str).isin(
                        ("review_needed", "reduce_limit", "disable_candidate")
                    )
                ]
            ),
            digits=0,
        ),
    )

    st.caption(f"Recommended daily profile set: {recommended_text}")
    if candidate_profiles:
        st.caption(f"Profiles to consider adding after more samples: {candidate_text}")
    st.dataframe(
        _daily_news_profile_recommendation_display_frame(recommendations),
        use_container_width=True,
        hide_index=True,
    )

    diff_columns = [
        "query_profile",
        "enabled_for_daily",
        "daily_operation_recommendation",
        "suggested_enabled_for_daily",
        "config_diff",
    ]
    diff_frame = recommendations[[column for column in diff_columns if column in recommendations.columns]].copy()
    st.caption("Read-only config difference candidates")
    st.dataframe(
        _daily_news_profile_config_diff_display_frame(diff_frame),
        use_container_width=True,
        hide_index=True,
    )

    _render_news_limit_recommendation_section(module, recommendations)
    _render_newslimit3_evaluation_section(settings)

    st.caption("Suggested daily PowerShell command")
    st.code(
        ".\\scripts\\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 5",
        language="powershell",
    )
    st.caption(
        "For a single-profile manual check, use "
        ".\\scripts\\daily_predict.ps1 -FetchNews -NewsQueryProfile toyota_stock -NewsLimit 5."
    )


def _render_newslimit3_evaluation_section(settings) -> None:
    st.subheader("NewsLimit 3 Evaluation")
    st.caption(
        "This is an operational data-quality evaluation, not proof of market predictive power. "
        "It uses daily-profile fetch_history rows from 2026-06-05 onward."
    )
    try:
        profile_quality = _news_profile_quality_module()
        news_dir = _news_module().default_news_dir(settings.paths.raw_data_dir.parent)
        evaluation = profile_quality.build_newslimit3_evaluation(news_dir, start_date="2026-06-05")
        report_path = profile_quality.save_newslimit3_evaluation_report(
            evaluation,
            PROJECT_ROOT / "reports" / "news",
        )
        config_review = profile_quality.build_daily_profile_config_review(news_dir, start_date="2026-06-05")
        config_review_path = profile_quality.save_daily_profile_config_review_report(
            config_review,
            PROJECT_ROOT / "reports" / "news",
        )
        before_after = profile_quality.build_daily_profile_before_after_evaluation(
            news_dir,
            start_date="2026-06-05",
        )
        before_after_report_path = profile_quality.save_daily_profile_before_after_report(
            before_after,
            PROJECT_ROOT / "reports" / "news",
        )
    except Exception as error:
        LOGGER.debug("Could not build NewsLimit 3 evaluation: %s", error, exc_info=True)
        st.caption("NewsLimit 3 evaluation is not available yet.")
        return

    daily_summary = evaluation.get("daily_summary")
    profile_evaluation = evaluation.get("profile_evaluation")
    if not isinstance(daily_summary, pd.DataFrame) or not isinstance(profile_evaluation, pd.DataFrame):
        st.info("No NewsLimit 3 evaluation rows are available yet.")
        return
    if daily_summary.empty or profile_evaluation.empty:
        st.info("No NewsLimit 3 daily-profile history is available yet.")
        return

    cols = st.columns(5)
    cols[0].metric("Overall recommendation", str(evaluation.get("overall_recommendation") or "-"))
    cols[1].metric("Observation days", _format_number(evaluation.get("observation_days"), digits=0))
    cols[2].metric("Total fetched", _format_number(evaluation.get("total_fetched"), digits=0))
    cols[3].metric("Total added", _format_number(evaluation.get("total_added"), digits=0))
    cols[4].metric("Duplicate rate", _format_percent(evaluation.get("duplicate_rate")))
    st.caption(str(evaluation.get("overall_reason") or ""))
    st.caption(str(evaluation.get("caveats") or ""))
    if report_path is not None:
        st.caption(f"Report CSV: {report_path}")

    observed_daily_profiles = len(profile_evaluation)
    suggested_daily_profiles = _suggested_daily_profile_count(config_review)
    current_daily_profiles = _current_daily_profile_count()
    fetched_before = observed_daily_profiles * 3
    fetched_after = suggested_daily_profiles * 3
    profile_cols = st.columns(4)
    profile_cols[0].metric("Observed daily profiles", _format_number(observed_daily_profiles, digits=0))
    profile_cols[1].metric("Current daily profiles", _format_number(current_daily_profiles, digits=0))
    profile_cols[2].metric("Recommended daily profiles", _format_number(suggested_daily_profiles, digits=0))
    profile_cols[3].metric("Fetched/day", f"{fetched_before} -> {fetched_after}")

    _render_daily_profile_before_after(before_after, before_after_report_path)

    st.caption("Daily summary")
    st.dataframe(
        _newslimit3_daily_summary_display_frame(daily_summary),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Profile evaluation")
    st.dataframe(
        _newslimit3_profile_evaluation_display_frame(profile_evaluation),
        use_container_width=True,
        hide_index=True,
    )

    st.caption("Conservative daily profile config review")
    st.dataframe(
        _daily_profile_config_review_display_frame(config_review),
        use_container_width=True,
        hide_index=True,
    )
    if config_review_path is not None:
        st.caption(f"Config review CSV: {config_review_path}")

    st.caption("Recommended action")
    st.write(_newslimit3_action_text(str(evaluation.get("overall_recommendation") or "")))


def _render_daily_profile_before_after(
    evaluation: dict[str, object],
    report_path,
) -> None:
    summary = evaluation.get("summary")
    profiles = evaluation.get("profile_evaluation")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return

    st.subheader("Before / After Daily Profile Change")
    st.caption(
        "This is an operational data-quality evaluation, not proof of market predictive power. "
        "The split is detected from the last batch containing toyota_earnings."
    )
    cols = st.columns(4)
    cols[0].metric(
        "Overall recommendation",
        str(evaluation.get("overall_recommendation") or "-"),
    )
    cols[1].metric(
        "After observation days",
        _format_number(evaluation.get("after_observation_days"), digits=0),
    )
    cols[2].metric(
        "Duplicate rate change",
        str(evaluation.get("duplicate_rate_change") or "-"),
    )
    cols[3].metric(
        "Added/day change",
        str(evaluation.get("added_per_day_change") or "-"),
    )
    cols[4].metric(
        "Unknown source change",
        str(evaluation.get("unknown_source_change") or "-"),
    )
    st.caption(str(evaluation.get("overall_reason") or ""))
    st.caption(str(evaluation.get("removed_profile_effect") or ""))
    st.caption(f"Detected change boundary: {evaluation.get('change_boundary') or '-'}")
    st.dataframe(
        _daily_profile_before_after_summary_display_frame(summary),
        use_container_width=True,
        hide_index=True,
    )
    if _daily_profile_unknown_history_available(summary):
        before_row = summary[summary["period"].eq("before")]
        after_row = summary[summary["period"].eq("after")]
        unknown_cols = st.columns(3)
        unknown_cols[0].metric(
            "Before unknown sources",
            _unknown_history_metric_with_coverage(before_row),
        )
        unknown_cols[1].metric(
            "After unknown sources",
            _unknown_history_metric_with_coverage(after_row),
        )
        unknown_cols[2].metric(
            "Latest unknown sources",
            _latest_unknown_sources_label(after_row),
        )
    else:
        st.caption(
            "Unknown source history: not available for older fetch_history rows. "
            "New check-news runs will populate it."
        )

    st.caption("check-news status and quality coverage")
    st.dataframe(
        _daily_profile_quality_comparison_display_frame(summary),
        use_container_width=True,
        hide_index=True,
    )

    if isinstance(profiles, pd.DataFrame) and not profiles.empty:
        st.caption("Post-change 4-profile evaluation")
        st.dataframe(
            _daily_profile_after_evaluation_display_frame(profiles),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Profile-level unknown-source fields repeat the shared check-news batch snapshot; "
            "they show the quality state when each profile ran, not which profile produced the unknown source."
        )
    st.caption(str(evaluation.get("caveats") or ""))
    if report_path is not None:
        st.caption(f"Before / after report CSV: {report_path}")


def _daily_profile_before_after_summary_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("duplicate_rate",):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in ("added_per_day", "added_per_profile_run"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    for column in (
        "observation_days",
        "daily_profile_count",
        "theoretical_fetched_per_day",
        "total_fetched",
        "total_added",
        "total_duplicates",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    for column in (
        "unknown_source_count",
        "unknown_source_occurrences",
        "unknown_source_observed_batches",
        "fetch_batch_count",
        "unknown_source_unrecorded_batches",
        "check_news_ok_count",
        "check_news_warn_count",
        "check_news_error_count",
        "check_news_unrecorded_count",
    ):
        if column in display.columns:
            display[column] = display[column].map(_optional_count_text)
    return display


def _daily_profile_quality_comparison_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "period",
        "unknown_source_count",
        "unknown_source_occurrences",
        "unknown_source_observed_batches",
        "unknown_source_unrecorded_batches",
        "check_news_ok_count",
        "check_news_warn_count",
        "check_news_error_count",
        "check_news_unrecorded_count",
        "check_news_status",
        "latest_unknown_sources",
    ]
    display = frame[[column for column in columns if column in frame.columns]].copy()
    for column in (
        "unknown_source_count",
        "unknown_source_occurrences",
        "unknown_source_observed_batches",
        "unknown_source_unrecorded_batches",
        "check_news_ok_count",
        "check_news_warn_count",
        "check_news_error_count",
        "check_news_unrecorded_count",
    ):
        if column in display.columns:
            display[column] = display[column].map(_optional_count_text)
    for column in ("check_news_status", "latest_unknown_sources"):
        if column in display.columns:
            display[column] = display[column].map(
                lambda value: str(value).strip() if str(value).strip() else "not available"
            )
    return display.rename(
        columns={
            "period": "Period",
            "unknown_source_count": "Unknown Source Total",
            "unknown_source_occurrences": "Unknown Source Batches",
            "unknown_source_observed_batches": "Quality Observed Batches",
            "unknown_source_unrecorded_batches": "Quality Unrecorded Batches",
            "check_news_ok_count": "check-news OK",
            "check_news_warn_count": "check-news WARN",
            "check_news_error_count": "check-news ERROR",
            "check_news_unrecorded_count": "check-news Unrecorded",
            "check_news_status": "Latest check-news",
            "latest_unknown_sources": "Latest Unknown Sources",
        }
    )


def _daily_profile_unknown_history_available(frame: pd.DataFrame) -> bool:
    if frame.empty or "unknown_source_count" not in frame.columns:
        return False
    return pd.to_numeric(frame["unknown_source_count"], errors="coerce").notna().any()


def _unknown_history_metric(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return "not available"
    value = pd.to_numeric(frame.iloc[0].get(column), errors="coerce")
    if pd.isna(value):
        return "not available"
    return _format_number(value, digits=0)


def _unknown_history_metric_with_coverage(frame: pd.DataFrame) -> str:
    count = _unknown_history_metric(frame, "unknown_source_count")
    if count == "not available":
        return count
    observed = _unknown_history_metric(frame, "unknown_source_observed_batches")
    total = _unknown_history_metric(frame, "fetch_batch_count")
    if observed == "not available" or total == "not available":
        return count
    return f"{count} ({observed}/{total} batches)"


def _latest_unknown_sources_label(frame: pd.DataFrame) -> str:
    if frame.empty or "latest_unknown_sources" not in frame.columns:
        return "not available"
    value = str(frame.iloc[0].get("latest_unknown_sources") or "").strip()
    if value:
        return value
    unknown_count = _unknown_history_metric(frame, "unknown_source_count")
    return "none" if unknown_count == "0" else "not available"


def _optional_count_text(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "not available"
    return _format_number(numeric, digits=0)


def _daily_profile_after_evaluation_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "duplicate_rate" in display.columns:
        display["duplicate_rate"] = display["duplicate_rate"].map(_format_percent)
    if "added_per_run" in display.columns:
        display["added_per_run"] = display["added_per_run"].map(
            lambda value: _format_number(value, digits=2)
        )
    for column in (
        "run_count",
        "total_fetched",
        "total_added",
        "total_duplicates",
        "unknown_source_count",
        "unknown_source_occurrences",
        "unknown_source_observed_batches",
        "unknown_source_unrecorded_batches",
        "check_news_ok_count",
        "check_news_warn_count",
        "check_news_error_count",
    ):
        if column in display.columns:
            display[column] = display[column].map(_optional_count_text)
    return display


def _newslimit3_daily_summary_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    if "duplicate_rate" in display.columns:
        display["duplicate_rate"] = display["duplicate_rate"].map(_format_percent)
    for column in ("fetched", "added", "duplicates", "profiles_count", "unknown_source_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    return display


def _newslimit3_profile_evaluation_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in ("duplicate_rate",):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    if "added_per_run" in display.columns:
        display["added_per_run"] = display["added_per_run"].map(lambda value: _format_number(value, digits=2))
    for column in ("runs", "total_fetched", "total_added", "total_duplicates"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    return display


def _daily_profile_config_review_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "duplicate_rate" in display.columns:
        display["duplicate_rate"] = display["duplicate_rate"].map(_format_percent)
    for column in ("total_fetched", "total_added"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    columns = [
        "profile",
        "group",
        "current_enabled_for_daily",
        "current_decision",
        "total_fetched",
        "total_added",
        "duplicate_rate",
        "recommendation",
        "suggested_enabled_for_daily",
        "config_change",
        "reason",
    ]
    return display[[column for column in columns if column in display.columns]]


def _current_daily_profile_count() -> int:
    try:
        profiles = _news_query_profiles_module().news_query_profile_rows()
    except Exception as error:
        LOGGER.debug("Could not count current daily news profiles: %s", error, exc_info=True)
        return 0
    return sum(str(profile.get("enabled_for_daily")).lower() == "true" for profile in profiles)


def _suggested_daily_profile_count(frame: pd.DataFrame) -> int:
    if frame.empty or "suggested_enabled_for_daily" not in frame.columns:
        return 0
    return int(frame["suggested_enabled_for_daily"].astype(bool).sum())


def _newslimit3_action_text(recommendation: str) -> str:
    if recommendation == "mixed":
        return (
            "Keep NewsLimit 3 as the global default for now. Review duplicate-heavy profiles individually "
            "instead of increasing all profiles to 5."
        )
    if recommendation == "keep_newslimit_3":
        return "Keep NewsLimit 3. It is light enough for daily operation and still adding useful rows."
    if recommendation == "needs_more_data":
        return "Keep collecting. Avoid changing configs until more daily observations are available."
    return "Review the profile table before changing configs/news_queries.yaml."


def _render_news_limit_recommendation_section(module, recommendations: pd.DataFrame) -> None:
    st.subheader("NewsLimit Recommendation")
    st.caption(
        "Read-only comparison for daily NewsLimit candidates 3 / 5 / 10. "
        "This helps keep daily news fetching light without changing configs or scripts."
    )
    limit_frame = module.build_news_limit_recommendations(recommendations)
    if limit_frame.empty:
        st.info("No NewsLimit recommendation rows are available yet.")
        return

    suggested_limit = module.recommended_daily_news_limit(limit_frame)
    st.caption(
        "Suggested overall daily limit: "
        f"{suggested_limit if suggested_limit is not None else 'monitor_more'}"
    )
    st.dataframe(
        _news_limit_recommendation_display_frame(limit_frame),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("Command examples")
    st.code(
        ".\\scripts\\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3",
        language="powershell",
    )
    st.code(
        ".\\scripts\\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 5",
        language="powershell",
    )
    st.code(
        "python main.py fetch-news --all-profiles --limit 10",
        language="powershell",
    )


def _news_limit_recommendation_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "duplicate_rate" in display.columns:
        display["duplicate_rate"] = display["duplicate_rate"].map(_format_percent)
    for column in (
        "latest_fetched_count",
        "latest_added_count",
        "latest_duplicate_count",
        "total_saved_rows",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    columns = [
        "query_profile",
        "enabled_for_daily",
        "group",
        "latest_fetched_count",
        "latest_added_count",
        "latest_duplicate_count",
        "duplicate_rate",
        "total_saved_rows",
        "quality_status",
        "daily_operation_recommendation",
        "suggested_daily_news_limit",
        "suggested_research_news_limit",
        "limit_3_assessment",
        "limit_5_assessment",
        "limit_10_assessment",
        "reason",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "query_profile": "Query Profile",
            "enabled_for_daily": "Daily Now",
            "group": "Group",
            "latest_fetched_count": "Latest Fetched",
            "latest_added_count": "Latest Added",
            "latest_duplicate_count": "Latest Duplicates",
            "duplicate_rate": "Duplicate Rate",
            "total_saved_rows": "Saved Rows",
            "quality_status": "Quality",
            "daily_operation_recommendation": "Profile Recommendation",
            "suggested_daily_news_limit": "Daily NewsLimit",
            "suggested_research_news_limit": "Research NewsLimit",
            "limit_3_assessment": "Limit 3",
            "limit_5_assessment": "Limit 5",
            "limit_10_assessment": "Limit 10",
            "reason": "Reason",
        }
    )


def _daily_news_profile_recommendation_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in (
        "duplicate_rate",
        "average_return_5d",
        "average_return_20d",
        "average_max_up_20d",
        "average_max_down_20d",
        "large_up_rate",
        "large_down_rate",
    ):
        if column in display.columns:
            display[column] = display[column].map(_format_percent)
    for column in (
        "average_crash_risk",
        "average_surge_risk",
        "average_importance_score",
        "average_source_credibility_score",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    for column in (
        "total_saved_rows",
        "added_count",
        "duplicate_count",
        "news_count",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    columns = [
        "query_profile",
        "group",
        "enabled_for_daily",
        "language",
        "region",
        "daily_operation_recommendation",
        "priority",
        "total_saved_rows",
        "news_count",
        "added_count",
        "duplicate_count",
        "duplicate_rate",
        "quality_status",
        "current_recommendation",
        "average_return_5d",
        "average_return_20d",
        "average_max_down_20d",
        "large_down_rate",
        "average_crash_risk",
        "average_surge_risk",
        "average_importance_score",
        "average_source_credibility_score",
        "latest_fetched_at",
        "recommendation_reason",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "query_profile": "Query Profile",
            "group": "Group",
            "enabled_for_daily": "Daily Now",
            "language": "Language",
            "region": "Region",
            "daily_operation_recommendation": "Daily Recommendation",
            "priority": "Priority",
            "total_saved_rows": "Saved Rows",
            "news_count": "Impact News Count",
            "added_count": "Added",
            "duplicate_count": "Duplicates",
            "duplicate_rate": "Duplicate Rate",
            "quality_status": "Quality",
            "current_recommendation": "Quality Recommendation",
            "average_return_5d": "Avg Return 5D",
            "average_return_20d": "Avg Return 20D",
            "average_max_down_20d": "Avg Max Down 20D",
            "large_down_rate": "Large Down Rate",
            "average_crash_risk": "Avg Crash Risk",
            "average_surge_risk": "Avg Surge Risk",
            "average_importance_score": "Avg Importance",
            "average_source_credibility_score": "Avg Source Credibility",
            "latest_fetched_at": "Latest Fetched At",
            "recommendation_reason": "Reason",
        }
    )


def _daily_news_profile_config_diff_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "query_profile": "Query Profile",
            "enabled_for_daily": "Current Daily",
            "daily_operation_recommendation": "Recommendation",
            "suggested_enabled_for_daily": "Suggested Daily",
            "config_diff": "Config Diff Candidate",
        }
    )


def _news_profile_quality_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "duplicate_rate" in display.columns:
        display["duplicate_rate"] = display["duplicate_rate"].map(_format_percent)
    for column in (
        "average_sentiment_score",
        "average_importance_score",
        "average_source_credibility_score",
        "average_source_market_impact_weight",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=2))
    for column in (
        "saved_rows",
        "history_fetched_count",
        "history_added_count",
        "history_duplicate_count",
        "duplicate_count",
        "unique_title_count",
        "unique_url_count",
        "mojibake_suspect_count",
        "invalid_date_count",
        "unknown_source_count",
    ):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    columns = [
        "profile",
        "group",
        "enabled_for_daily",
        "language",
        "region",
        "saved_rows",
        "history_fetched_count",
        "history_added_count",
        "history_duplicate_count",
        "duplicate_rate",
        "latest_news_date",
        "latest_fetched_at",
        "average_importance_score",
        "average_source_credibility_score",
        "quality_status",
        "quality_message",
        "category_distribution",
        "source_type_distribution",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "profile": "Profile",
            "group": "Group",
            "enabled_for_daily": "Daily",
            "language": "Language",
            "region": "Region",
            "saved_rows": "Saved Rows",
            "history_fetched_count": "Fetched",
            "history_added_count": "Added",
            "history_duplicate_count": "Duplicates",
            "duplicate_rate": "Duplicate Rate",
            "latest_news_date": "Latest News Date",
            "latest_fetched_at": "Latest Fetched At",
            "average_importance_score": "Avg Importance",
            "average_source_credibility_score": "Avg Source Credibility",
            "quality_status": "Quality",
            "quality_message": "Message",
            "category_distribution": "Category Distribution",
            "source_type_distribution": "Source Type Distribution",
        }
    )


def _news_profile_recommendation_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.rename(
        columns={
            "profile": "Profile",
            "group": "Group",
            "enabled_for_daily": "Daily",
            "recommendation": "Recommendation",
            "priority": "Priority",
            "reason": "Reason",
        }
    )


def _news_fetch_history_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    for column in ("fetched_count", "added_count", "duplicate_count"):
        if column in display.columns:
            display[column] = display[column].map(lambda value: _format_number(value, digits=0))
    columns = [
        "fetched_at",
        "profile",
        "group",
        "enabled_for_daily",
        "language",
        "region",
        "fetched_count",
        "added_count",
        "duplicate_count",
        "warning",
        "error",
    ]
    display = display[[column for column in columns if column in display.columns]]
    return display.rename(
        columns={
            "fetched_at": "Fetched At",
            "profile": "Profile",
            "group": "Group",
            "enabled_for_daily": "Daily",
            "language": "Language",
            "region": "Region",
            "fetched_count": "Fetched",
            "added_count": "Added",
            "duplicate_count": "Duplicates",
            "warning": "Warning",
            "error": "Error",
        }
    )


def _render_today_signal_config_status() -> None:
    module = _today_signal_validation_module()
    description = module.describe_today_signal_config()
    status = str(description.get("status") or "unknown")
    message = str(description.get("message") or "")
    st.subheader("Today Signal Rule Config")
    if status == "configured":
        st.success("Today Signal config: configured")
    elif status == "default":
        st.info("Today Signal config: default")
    else:
        st.warning("Today Signal config: fallback defaults")
    st.caption(message)
    st.caption(f"Config path: {description.get('path') or '-'}")

    rows = module.today_signal_config_rows()
    if rows:
        st.dataframe(
            _today_signal_config_display_frame(pd.DataFrame(rows)),
            use_container_width=True,
            hide_index=True,
        )


def _today_signal_config_display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    display = frame.copy()
    if "value" in display.columns:
        display["value"] = display["value"].map(lambda value: _format_number(value, digits=3))
    return display.rename(
        columns={
            "config_key": "Config Key",
            "value": "Value",
            "description": "Description",
        }
    )


@st.cache_data(ttl=60, show_spinner=False)
def _load_health_report_for_ui(ticker: str) -> dict[str, object]:
    return run_health_check(ticker=ticker, log_tail=100)


def _news_fetch_status(ticker: str, settings) -> dict[str, object]:
    try:
        news = _news_module()
        fetcher = _news_fetcher_module()
        quality = _news_quality_module()
        query_profiles = _news_query_profiles_module()
        profile_quality = _news_profile_quality_module()
        source_profiles = _news_sources_module()
        news_dir = news.default_news_dir(settings.paths.raw_data_dir.parent)
        news_path = news.news_path_candidates(news_dir, ticker)[0]
        status = fetcher.latest_news_status(news_path)
        status["health"] = quality.check_news_csv(news_path)
        status["exists"] = Path(news_path).exists()
        status["query_profiles"] = query_profiles.news_query_profile_rows()
        profile_quality_frame = profile_quality.build_profile_quality_frame(news_dir, ticker)
        profile_recommendations = profile_quality.build_profile_recommendations(profile_quality_frame)
        status["profile_quality"] = profile_quality_frame.to_dict(orient="records")
        status["profile_recommendations"] = profile_recommendations.to_dict(orient="records")
        status["recent_fetch_history"] = profile_quality.recent_fetch_history(
            news_dir,
            limit=20,
        ).to_dict(orient="records")
        fetch_history = profile_quality.read_fetch_history(news_dir)
        status["fetch_history"] = fetch_history.to_dict(orient="records")
        quality_history = profile_quality.build_news_quality_history(fetch_history)
        status["quality_history"] = {
            "summary": quality_history["summary"],
            "batches": quality_history["batches"].to_dict(orient="records"),
        }
        status["source_profiles"] = source_profiles.news_source_profile_rows()
        status["unknown_sources"] = _unknown_news_source_rows(news_path)
        return status
    except Exception as error:
        LOGGER.debug("Could not build news fetch status: %s", error, exc_info=True)
        return {
            "path": "",
            "count": 0,
            "latest_date": None,
            "latest_fetched_at": None,
            "exists": False,
            "health": {},
            "query_profiles": [],
            "profile_quality": [],
            "profile_recommendations": [],
            "recent_fetch_history": [],
            "fetch_history": [],
            "quality_history": {},
            "source_profiles": [],
            "unknown_sources": [],
        }


def _unknown_news_source_rows(news_path: str | Path) -> list[dict[str, object]]:
    path = Path(news_path)
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError):
        return []
    if frame.empty or "source_type" not in frame.columns:
        return []

    source_type = frame["source_type"].fillna("").astype(str).str.strip().str.lower()
    unknown = frame[source_type.isin({"", "unknown"})].copy()
    if unknown.empty:
        return []

    for column in ("source", "date", "category", "title", "url"):
        if column not in unknown.columns:
            unknown[column] = ""
    unknown["parsed_date"] = pd.to_datetime(unknown["date"], errors="coerce", format="mixed")
    rows: list[dict[str, object]] = []
    for source, group in unknown.groupby(unknown["source"].fillna("").astype(str), dropna=False):
        latest = group["parsed_date"].dropna().max()
        sample = group.sort_values("parsed_date", ascending=False, na_position="last").iloc[0]
        categories = sorted(
            value
            for value in group["category"].fillna("").astype(str).unique()
            if value
        )
        rows.append(
            {
                "source": source or "(empty)",
                "count": int(len(group)),
                "latest_news_date": _format_date(latest) if pd.notna(latest) else "",
                "category": ", ".join(categories),
                "title_example": str(sample.get("title") or ""),
                "url_example": str(sample.get("url") or ""),
            }
        )
    return sorted(rows, key=lambda row: (-(safe_int(row.get("count")) or 0), str(row.get("source"))))


def _render_unknown_sources_section(news_status: dict[str, object]) -> None:
    rows = news_status.get("unknown_sources") or []
    st.caption("Unknown Sources")
    if not rows:
        st.caption("No unknown news sources are currently detected.")
        return
    st.warning("Unknown news sources are present. Review these before relying too strongly on News Impact.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _health_item(report: dict[str, object], section: str, name: str) -> dict[str, object] | None:
    for item in report.get("items", []) or []:
        if item.get("section") == section and item.get("name") == name:
            return item
    return None


def _health_items(report: dict[str, object], section: str) -> list[dict[str, object]]:
    return [
        item
        for item in report.get("items", []) or []
        if item.get("section") == section
    ]


def _health_status(item: dict[str, object] | None) -> str:
    if item is None:
        return "-"
    return str(item.get("status") or "-")


def _health_message(item: dict[str, object] | None) -> str:
    if item is None:
        return "-"
    return str(item.get("message") or "-")


def _health_details(item: dict[str, object] | None) -> dict[str, object]:
    if item is None:
        return {}
    details = item.get("details")
    return details if isinstance(details, dict) else {}


def _health_detail(item: dict[str, object] | None, key: str) -> str:
    value = _health_details(item).get(key)
    if value is None or value == "":
        return "-"
    return str(value)


def _health_section_status(report: dict[str, object], section: str) -> str:
    statuses = [_health_status(item) for item in _health_items(report, section)]
    if not statuses:
        return "-"
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "OK"


def _render_health_item_message(item: dict[str, object] | None) -> None:
    if item is None:
        st.info("Status item is not available.")
        return
    status = _health_status(item)
    message = _health_message(item)
    if status == "OK":
        st.success(message)
    elif status == "WARN":
        st.warning(message)
    else:
        st.error(message)


def _health_section_frame(report: dict[str, object], section: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in _health_items(report, section):
        details = _health_details(item)
        rows.append(
            {
                "Status": item.get("status"),
                "Name": item.get("name"),
                "Data status": details.get("data_status"),
                "Latest date": details.get("latest_date"),
                "Rows": details.get("rows"),
                "Modified at": details.get("modified_at"),
                "Message": item.get("message"),
            }
        )
    return pd.DataFrame(rows)


def _read_daily_log_tail(line_count: int) -> list[str]:
    log_path = PROJECT_ROOT / "logs" / "daily_predict.log"
    if not log_path.exists():
        return []
    try:
        return log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    except OSError as error:
        LOGGER.debug("Could not read daily_predict.log: %s", error)
        return []


def _format_number(value, digits: int = 2) -> str:
    value = safe_float(value)
    if value is None:
        return "-"
    return f"{value:,.{digits}f}"


def _format_date(value) -> str:
    if value is None:
        return "-"
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "-"
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _format_datetime(value) -> str:
    if value is None:
        return "-"
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "-"
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def _format_enabled(value) -> str:
    if value is True:
        return "Enabled"
    if value is False:
        return "Disabled"
    return "-"


def _format_percent(value) -> str:
    value = safe_float(value)
    if value is None:
        return "-"
    return f"{value * 100:.2f}%"


def _class_label_from_value(value) -> int | None:
    label = safe_int(value)
    if label is None:
        return None
    if label < 0:
        return None
    return label


def _prediction_label(row: pd.Series) -> str:
    return _prediction_class_name(row.get("predicted_class"), row.get("target"))


def _prediction_class_name(value, target) -> str:
    class_id = safe_int(value)
    if class_id is None:
        return "unknown"
    if target == "next_close_binary":
        return BINARY_CLASS_NAMES.get(class_id, "unknown")
    return CLASS_NAMES.get(class_id, "unknown")


def _parse_top_features(value) -> list[dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if not isinstance(value, str):
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            pass
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return parsed


def _load_binary_metrics(ticker: str) -> dict[str, object]:
    metrics_path = PROJECT_ROOT / "reports" / "metrics" / f"lightgbm_binary_{ticker}.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _load_multiclass_metrics(
    ticker: str,
    threshold_profile: str | None = None,
    weight_mode: str | None = None,
) -> dict[str, object]:
    labels = _label_distribution_module()
    artifact_stem = labels.multiclass_artifact_stem(ticker, threshold_profile, weight_mode)
    metrics_path = PROJECT_ROOT / "reports" / "metrics" / f"{artifact_stem}.json"
    if not metrics_path.exists():
        return {}
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _build_probability_chart(up_probability, down_probability) -> go.Figure:
    up = safe_float(up_probability)
    down = safe_float(down_probability)
    if up is None and down is None:
        up, down = 0.0, 0.0
    elif up is None:
        up = max(0.0, 1.0 - float(down))
    elif down is None:
        down = max(0.0, 1.0 - float(up))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=["Probability"],
            x=[down],
            name="Down",
            orientation="h",
            marker_color="#d95f5f",
            text=[_format_percent(down)],
            textposition="inside",
        )
    )
    fig.add_trace(
        go.Bar(
            y=["Probability"],
            x=[up],
            name="Up",
            orientation="h",
            marker_color="#3a8f6b",
            text=[_format_percent(up)],
            textposition="inside",
        )
    )
    fig.update_layout(
        barmode="stack",
        height=150,
        xaxis=dict(range=[0, 1], tickformat=".0%"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _multiclass_probability_map(row: pd.Series) -> dict[str, float | None]:
    return {
        label: safe_float(row.get(column))
        for _class_id, label, column, _color in MULTICLASS_PROBABILITY_FIELDS
    }


def _build_multiclass_probability_chart(probability_map: dict[str, float | None]) -> go.Figure:
    labels = [label for _class_id, label, _column, _color in MULTICLASS_PROBABILITY_FIELDS]
    values = [probability_map.get(label) or 0.0 for label in labels]
    colors = [color for _class_id, _label, _column, color in MULTICLASS_PROBABILITY_FIELDS]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[_format_percent(value) for value in values],
            textposition="outside",
            hovertemplate="Class=%{x}<br>Probability=%{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        yaxis=dict(range=[0, max(1.0, max(values, default=0.0) * 1.2)], tickformat=".0%"),
        xaxis_title="Class",
        yaxis_title="Probability",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _multiclass_tail_risk_label(probability_map: dict[str, float | None]) -> str:
    surge = probability_map.get("surge") or 0.0
    crash = probability_map.get("crash") or 0.0
    if crash >= 0.35 and surge >= 0.35:
        return "Both high"
    if crash >= 0.35:
        return "Crash high"
    if surge >= 0.35:
        return "Surge high"
    if max(surge, crash) >= 0.20:
        return "Elevated"
    return "Normal"


def _multiclass_prediction_rationale(
    latest_multiclass: pd.Series,
    top_features: list[dict[str, object]],
) -> str:
    probability_map = _multiclass_probability_map(latest_multiclass)
    prediction_label = _prediction_class_name(
        latest_multiclass.get("predicted_class"),
        latest_multiclass.get("target"),
    )
    confidence = safe_float(latest_multiclass.get("confidence"))
    top_probability = probability_map.get(prediction_label)
    feature_names = [
        str(item.get("feature"))
        for item in top_features[:3]
        if item.get("feature") is not None
    ]

    parts = [
        f"The 5-class model selects {prediction_label} with {_format_percent(confidence)} confidence."
    ]
    if top_probability is not None and confidence is not None and abs(top_probability - confidence) > 0.02:
        parts.append(
            f"The stored {prediction_label} probability is {_format_percent(top_probability)}."
        )
    crash = probability_map.get("crash")
    surge = probability_map.get("surge")
    if crash is not None and crash >= 0.20:
        parts.append("Crash probability is elevated relative to the MVP tail-risk threshold.")
    if surge is not None and surge >= 0.20:
        parts.append("Surge probability is elevated relative to the MVP tail-risk threshold.")
    if prediction_label == "flat" and max(crash or 0.0, surge or 0.0) >= 0.15:
        parts.append("Although flat is most likely, tail-class probability should not be ignored.")
    if feature_names:
        parts.append(f"Top training-level drivers are {', '.join(feature_names)}.")
    return " ".join(parts)


def _build_confidence_accuracy_chart(confidence_accuracy: pd.DataFrame) -> go.Figure:
    frame = confidence_accuracy.copy()
    frame["accuracy"] = frame["accuracy"].map(safe_float)
    frame = frame.dropna(subset=["accuracy"])
    if frame.empty:
        return go.Figure()

    fig = go.Figure(
        go.Bar(
            x=frame["confidence_bucket"].astype(str),
            y=frame["accuracy"],
            marker_color="#4f78a8",
            text=frame["accuracy"].map(_format_percent),
            textposition="outside",
            customdata=frame[["prediction_count", "correct_count"]],
            hovertemplate=(
                "Confidence=%{x}<br>"
                "Accuracy=%{y:.2%}<br>"
                "Predictions=%{customdata[0]}<br>"
                "Correct=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        height=280,
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        xaxis_title="Confidence bucket",
        yaxis_title="Accuracy",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _build_feature_contribution_chart(top_features: list[dict[str, object]]) -> go.Figure:
    frame = pd.DataFrame(top_features).head(10)
    if frame.empty or "feature" not in frame:
        return go.Figure()

    if "contribution" in frame.columns:
        frame["contribution"] = frame["contribution"].map(safe_float)
        frame = frame.dropna(subset=["contribution"])
        frame["abs_contribution"] = frame["contribution"].abs()
        frame = frame.sort_values("abs_contribution", ascending=True)
        colors = frame["contribution"].map(lambda value: "#3a8f6b" if value >= 0 else "#d95f5f")
        fig = go.Figure(
            go.Bar(
                x=frame["contribution"],
                y=frame["feature"],
                orientation="h",
                marker_color=colors,
                customdata=frame[["direction", "value"]] if {"direction", "value"}.issubset(frame.columns) else None,
                hovertemplate=(
                    "Feature=%{y}<br>"
                    "Contribution=%{x:.5f}<br>"
                    "Direction=%{customdata[0]}<br>"
                    "Value=%{customdata[1]}<extra></extra>"
                )
                if {"direction", "value"}.issubset(frame.columns)
                else "Feature=%{y}<br>Contribution=%{x:.5f}<extra></extra>",
            )
        )
        fig.add_vline(x=0, line_color="#555555", line_width=1)
        fig.update_layout(
            height=380,
            xaxis_title="SHAP contribution to Up class",
            yaxis_title="",
            margin=dict(l=20, r=20, t=10, b=30),
        )
        return fig

    if "importance" not in frame.columns:
        return go.Figure()

    frame["importance"] = frame["importance"].map(safe_float)
    frame = frame.dropna(subset=["importance"]).sort_values("importance", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=frame["importance"],
            y=frame["feature"],
            orientation="h",
            marker_color="#4f78a8",
        )
    )
    fig.update_layout(
        height=360,
        xaxis_title="Importance",
        yaxis_title="",
        margin=dict(l=20, r=20, t=10, b=30),
    )
    return fig


def _build_prediction_rationale(
    latest: pd.Series,
    prediction_label: str,
    up_probability,
    down_probability,
    top_features: list[dict[str, object]],
    metrics: dict[str, object],
) -> str:
    up = safe_float(up_probability)
    down = safe_float(down_probability)
    confidence = max([value for value in (up, down) if value is not None], default=None)
    up_drivers = _driver_names(top_features, direction="up")
    down_drivers = _driver_names(top_features, direction="down")
    important_names = [
        str(item.get("feature"))
        for item in top_features[:3]
        if item.get("feature") is not None
    ]

    parts: list[str] = []
    if prediction_label == "up":
        parts.append(f"Model leans upward with {_format_percent(up)} up probability.")
    elif prediction_label == "down":
        parts.append(f"Model leans downward with {_format_percent(down)} down probability.")
    else:
        parts.append("Model prediction is currently unknown because the saved class could not be read.")

    if confidence is not None and confidence < 0.55:
        parts.append("Confidence is weak, so treat this as a low-conviction signal.")
    elif confidence is not None and confidence >= 0.65:
        parts.append("Confidence is relatively strong for this MVP model.")

    if up_drivers:
        parts.append(f"SHAP upward drivers are {', '.join(up_drivers)}.")
    if down_drivers:
        parts.append(f"SHAP downward drivers are {', '.join(down_drivers)}.")
    if not up_drivers and not down_drivers and important_names:
        parts.append(f"Top drivers are {', '.join(important_names)}.")

    technical_reasons = build_basic_reasons(latest)
    if technical_reasons:
        parts.append(f"Rule checks also point to: {', '.join(technical_reasons)}.")

    auc = safe_float(metrics.get("auc"))
    accuracy = safe_float(metrics.get("accuracy"))
    if auc is not None and accuracy is not None:
        parts.append(
            f"Recent test metrics are Accuracy {_format_percent(accuracy)} and AUC {_format_percent(auc)}."
        )

    return " ".join(parts)


def _has_shap_contributions(top_features: list[dict[str, object]]) -> bool:
    return any("contribution" in item for item in top_features)


def _format_correctness(value, status) -> str:
    if status != "confirmed":
        return "Pending"
    try:
        if pd.isna(value):
            return "Unknown"
    except (TypeError, ValueError):
        pass
    return "Correct" if bool(value) else "Wrong"


def _driver_names(top_features: list[dict[str, object]], direction: str) -> list[str]:
    drivers: list[tuple[str, float]] = []
    for item in top_features:
        contribution = safe_float(item.get("contribution"))
        if contribution is None:
            continue
        if direction == "up" and contribution <= 0:
            continue
        if direction == "down" and contribution >= 0:
            continue
        feature = item.get("feature")
        if feature is None:
            continue
        drivers.append((str(feature), abs(contribution)))
    drivers.sort(key=lambda item: item[1], reverse=True)
    return [feature for feature, _ in drivers[:3]]


if __name__ == "__main__":
    main()


