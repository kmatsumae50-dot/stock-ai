from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.labels.targets import add_next_close_binary_target
from stock_ai.models.baseline import select_feature_columns
from stock_ai.models.predict import load_model_bundle
from stock_ai.utils.safe_cast import safe_float, safe_int
from stock_ai.validation.time_split import chronological_train_valid_test_split


SHADOW_COLUMNS = [
    "run_timestamp",
    "ticker",
    "feature_date",
    "target_close_date",
    "candidate_name",
    "predicted_direction",
    "probability_up",
    "probability_down",
    "confidence",
    "source",
    "model_note",
    "actual_return",
    "actual_direction",
    "result",
    "evaluated_at",
    "caveat",
]
SUMMARY_COLUMNS = [
    "candidate_name",
    "sample_count",
    "confirmed_count",
    "pending_count",
    "accuracy",
    "balanced_accuracy",
    "up_recall",
    "down_recall",
    "up_precision",
    "down_precision",
    "predicted_up_count",
    "predicted_down_count",
    "confidence_ge_55_count",
    "confidence_ge_55_hit_count",
    "confidence_ge_55_miss_count",
    "confidence_ge_55_hit_rate",
    "latest_feature_date",
    "caveat",
]


@dataclass(frozen=True)
class ShadowCandidate:
    name: str
    model_note: str
    source: str = "shadow"


@dataclass(frozen=True)
class BinaryShadowResult:
    predictions: pd.DataFrame
    saved_paths: dict[str, str]


@dataclass(frozen=True)
class BinaryShadowEvaluationResult:
    predictions: pd.DataFrame
    summary: pd.DataFrame
    comparison: pd.DataFrame
    saved_paths: dict[str, str]
    conclusions: list[str]


def default_shadow_candidates() -> list[ShadowCandidate]:
    return [
        ShadowCandidate(
            name="production_current",
            model_note="Current production binary model loaded from artifact; not saved to SQLite.",
        ),
        ShadowCandidate(
            name="class_weight_balanced",
            model_note="Temporary production-style binary model with class_weight='balanced'; not saved as production.",
        ),
    ]


def build_binary_shadow_predictions(
    data: pd.DataFrame,
    *,
    ticker: str,
    production_model_path: str | Path,
    candidates: list[ShadowCandidate] | None = None,
    run_timestamp: str | None = None,
) -> pd.DataFrame:
    candidates = candidates or default_shadow_candidates()
    frame = _prepare_feature_frame(data, ticker)
    if frame.empty:
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    latest = frame.sort_values("date").tail(1)
    if latest.empty:
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    timestamp = run_timestamp or pd.Timestamp.now().isoformat(timespec="seconds")
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.name == "production_current":
            prediction = _predict_production_current(
                latest,
                production_model_path=production_model_path,
            )
        elif candidate.name == "class_weight_balanced":
            prediction = _predict_class_weight_balanced(frame, latest)
        else:
            raise ValueError(f"Unsupported shadow candidate: {candidate.name}")
        if prediction is None:
            continue
        rows.append(
            _shadow_row(
                prediction,
                latest,
                frame,
                candidate=candidate,
                run_timestamp=timestamp,
            )
        )
    return pd.DataFrame(rows, columns=SHADOW_COLUMNS)


def save_shadow_predictions(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "binary_shadow_predictions.csv"
    json_path = output / "binary_shadow_predictions.json"
    existing = _read_shadow_predictions(csv_path)
    combined = _merge_shadow_predictions(existing, predictions)
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "rows": _json_records(combined),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "json": str(json_path)}


def evaluate_shadow_predictions(
    shadow_path: str | Path,
    feature_data: pd.DataFrame,
    *,
    ticker: str,
    output_dir: str | Path,
    live_predictions: pd.DataFrame | None = None,
    run_date: str | None = None,
) -> BinaryShadowEvaluationResult:
    predictions = _read_shadow_predictions(shadow_path)
    if not predictions.empty:
        predictions = predictions[predictions["ticker"].astype(str).eq(ticker)].copy()
    predictions = _evaluate_prediction_rows(predictions, feature_data)
    summary = summarize_shadow_predictions(predictions)
    comparison = compare_shadow_with_live(predictions, live_predictions)
    conclusions = classify_shadow_conclusions(summary)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "binary_shadow_predictions.csv", index=False, encoding="utf-8-sig")
    (output / "binary_shadow_predictions.json").write_text(
        json.dumps(
            {
                "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "rows": _json_records(predictions),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    stamp = run_date or date.today().isoformat()
    md_path = output / f"binary_shadow_summary_{stamp}.md"
    md_path.write_text(_summary_markdown(summary, comparison, predictions, conclusions), encoding="utf-8")
    return BinaryShadowEvaluationResult(
        predictions=predictions,
        summary=summary,
        comparison=comparison,
        saved_paths={
            "csv": str(output / "binary_shadow_predictions.csv"),
            "json": str(output / "binary_shadow_predictions.json"),
            "md": str(md_path),
        },
        conclusions=conclusions,
    )


def summarize_shadow_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    rows: list[dict[str, object]] = []
    for candidate, group in predictions.groupby("candidate_name", dropna=False):
        confirmed = group[group["result"].isin(["hit", "miss"])].copy()
        pending = group[group["result"].eq("pending")].copy()
        predicted = group["predicted_direction"].astype(str)
        metrics = _direction_metrics(confirmed)
        high_conf = confirmed[pd.to_numeric(confirmed.get("confidence"), errors="coerce") >= 0.55].copy()
        hit_count = int(high_conf["result"].eq("hit").sum())
        miss_count = int(high_conf["result"].eq("miss").sum())
        latest = pd.to_datetime(group.get("feature_date"), errors="coerce").dropna()
        caveats = []
        if len(confirmed) < 10:
            caveats.append("low confirmed sample count")
        rows.append(
            {
                "candidate_name": candidate,
                "sample_count": int(len(group)),
                "confirmed_count": int(len(confirmed)),
                "pending_count": int(len(pending)),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "up_recall": metrics["up_recall"],
                "down_recall": metrics["down_recall"],
                "up_precision": metrics["up_precision"],
                "down_precision": metrics["down_precision"],
                "predicted_up_count": int(predicted.eq("up").sum()),
                "predicted_down_count": int(predicted.eq("down").sum()),
                "confidence_ge_55_count": int(len(high_conf)),
                "confidence_ge_55_hit_count": hit_count,
                "confidence_ge_55_miss_count": miss_count,
                "confidence_ge_55_hit_rate": _safe_divide(hit_count, len(high_conf)),
                "latest_feature_date": (
                    pd.Timestamp(latest.max()).strftime("%Y-%m-%d") if not latest.empty else None
                ),
                "caveat": "; ".join(caveats),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def compare_shadow_with_live(
    predictions: pd.DataFrame,
    live_predictions: pd.DataFrame | None,
) -> pd.DataFrame:
    columns = [
        "feature_date",
        "candidate_name",
        "shadow_direction",
        "shadow_confidence",
        "production_live_direction",
        "production_live_confidence",
        "direction_matches_live",
        "result",
    ]
    if predictions.empty:
        return pd.DataFrame(columns=columns)
    live = _normalize_live_predictions(live_predictions)
    rows = []
    for row in predictions.to_dict(orient="records"):
        feature_date = _date_text(row.get("feature_date"))
        live_row = live[live["feature_date"].eq(feature_date)].head(1) if not live.empty else pd.DataFrame()
        live_direction = live_row["production_live_direction"].iloc[0] if not live_row.empty else None
        live_confidence = safe_float(live_row["production_live_confidence"].iloc[0]) if not live_row.empty else None
        shadow_direction = str(row.get("predicted_direction") or "")
        rows.append(
            {
                "feature_date": feature_date,
                "candidate_name": row.get("candidate_name"),
                "shadow_direction": shadow_direction,
                "shadow_confidence": safe_float(row.get("confidence")),
                "production_live_direction": live_direction,
                "production_live_confidence": live_confidence,
                "direction_matches_live": (
                    shadow_direction == live_direction if live_direction is not None else None
                ),
                "result": row.get("result"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def classify_shadow_conclusions(summary: pd.DataFrame) -> list[str]:
    conclusions = ["production_change_not_ready", "continue_shadow_monitoring"]
    if summary.empty:
        return conclusions
    balanced = _summary_candidate(summary, "class_weight_balanced")
    current = _summary_candidate(summary, "production_current")
    if balanced is not None and current is not None:
        balanced_confirmed = safe_int(balanced.get("confirmed_count")) or 0
        current_confirmed = safe_int(current.get("confirmed_count")) or 0
        if min(balanced_confirmed, current_confirmed) < 10:
            conclusions.insert(0, "shadow_sample_too_small")
        elif _gt(balanced.get("balanced_accuracy"), current.get("balanced_accuracy")):
            conclusions.insert(0, "class_weight_balanced_shadow_promising")
        else:
            conclusions.insert(0, "class_weight_balanced_not_yet_better")
    return list(dict.fromkeys(conclusions))


def _prepare_feature_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    frame = data.copy()
    if "target_next_close_binary" not in frame.columns:
        frame = add_next_close_binary_target(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    if "ticker" in frame.columns:
        frame = frame[frame["ticker"].astype(str).eq(ticker)].copy()
    return frame.reset_index(drop=True)


def _predict_production_current(
    latest: pd.DataFrame,
    *,
    production_model_path: str | Path,
) -> dict[str, object] | None:
    bundle = load_model_bundle(production_model_path)
    return _predict_with_model(bundle["model"], list(bundle["feature_columns"]), latest)


def _predict_class_weight_balanced(frame: pd.DataFrame, latest: pd.DataFrame) -> dict[str, object] | None:
    train, valid, _test = chronological_train_valid_test_split(frame)
    model, feature_columns = _train_balanced_binary_model(train, valid)
    return _predict_with_model(model, feature_columns, latest)


def _train_balanced_binary_model(train: pd.DataFrame, valid: pd.DataFrame):
    import lightgbm as lgb

    feature_columns = select_feature_columns(train, target_column="target_next_close_binary")
    train_data = train.dropna(subset=feature_columns + ["target_next_close_binary"])
    valid_data = valid.dropna(subset=feature_columns + ["target_next_close_binary"])
    train_data = train_data[train_data["target_next_close_binary"] >= 0]
    valid_data = valid_data[valid_data["target_next_close_binary"] >= 0]
    if train_data.empty or valid_data.empty:
        raise ValueError("Not enough rows to train class_weight_balanced shadow model.")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train_data[feature_columns],
        train_data["target_next_close_binary"].astype(int),
        eval_set=[(valid_data[feature_columns], valid_data["target_next_close_binary"].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model, feature_columns


def _predict_with_model(model, feature_columns: list[str], latest: pd.DataFrame) -> dict[str, object] | None:
    rows = latest.dropna(subset=feature_columns)
    if rows.empty:
        return None
    probabilities = np.asarray(model.predict_proba(rows[feature_columns]), dtype=float)
    probability_down = float(probabilities[0, 0])
    probability_up = float(probabilities[0, 1])
    predicted_class = int(probability_up >= 0.5)
    return {
        "predicted_class": predicted_class,
        "predicted_direction": "up" if predicted_class == 1 else "down",
        "probability_up": probability_up,
        "probability_down": probability_down,
        "confidence": max(probability_up, probability_down),
    }


def _shadow_row(
    prediction: dict[str, object],
    latest: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    candidate: ShadowCandidate,
    run_timestamp: str,
) -> dict[str, object]:
    latest_row = latest.iloc[0]
    feature_date = pd.Timestamp(latest_row["date"]).normalize()
    actual_return, actual_direction, result, target_close_date = _actual_for_feature_date(
        frame,
        feature_date,
        predicted_direction=str(prediction["predicted_direction"]),
    )
    return {
        "run_timestamp": run_timestamp,
        "ticker": str(latest_row.get("ticker")),
        "feature_date": feature_date.strftime("%Y-%m-%d"),
        "target_close_date": target_close_date,
        "candidate_name": candidate.name,
        "predicted_direction": prediction["predicted_direction"],
        "probability_up": prediction["probability_up"],
        "probability_down": prediction["probability_down"],
        "confidence": prediction["confidence"],
        "source": candidate.source,
        "model_note": candidate.model_note,
        "actual_return": actual_return,
        "actual_direction": actual_direction,
        "result": result,
        "evaluated_at": pd.Timestamp.now().isoformat(timespec="seconds") if result != "pending" else "",
        "caveat": "shadow only; not saved to SQLite; production_change_not_ready",
    }


def _evaluate_prediction_rows(predictions: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    frame = feature_data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    rows = predictions.copy()
    for column in SHADOW_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    for index, row in rows.iterrows():
        feature_date = pd.to_datetime(row.get("feature_date"), errors="coerce")
        if pd.isna(feature_date):
            rows.loc[index, "result"] = "pending"
            rows.loc[index, "caveat"] = _append_text(str(row.get("caveat") or ""), "invalid feature_date")
            continue
        ticker = str(row.get("ticker") or "")
        ticker_frame = frame[frame["ticker"].astype(str).eq(ticker)].copy() if "ticker" in frame.columns else frame
        actual_return, actual_direction, result, target_date = _actual_for_feature_date(
            ticker_frame,
            pd.Timestamp(feature_date).normalize(),
            predicted_direction=str(row.get("predicted_direction") or ""),
        )
        rows.loc[index, "target_close_date"] = target_date or row.get("target_close_date")
        rows.loc[index, "actual_return"] = actual_return
        rows.loc[index, "actual_direction"] = actual_direction
        rows.loc[index, "result"] = result
        if result != "pending":
            rows.loc[index, "evaluated_at"] = pd.Timestamp.now().isoformat(timespec="seconds")
    return rows[SHADOW_COLUMNS].reset_index(drop=True)


def _actual_for_feature_date(
    frame: pd.DataFrame,
    feature_date: pd.Timestamp,
    *,
    predicted_direction: str,
) -> tuple[float | None, str, str, str | None]:
    if frame.empty or "close" not in frame.columns:
        return None, "", "pending", _next_business_day(feature_date)
    rows = frame.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["date"]).sort_values("date")
    current = rows[rows["date"].eq(feature_date)]
    future = rows[rows["date"] > feature_date].head(1)
    if current.empty:
        return None, "", "pending", _next_business_day(feature_date)
    if future.empty:
        return None, "", "pending", _next_business_day(feature_date)
    close = safe_float(current.iloc[0].get("close"))
    next_close = safe_float(future.iloc[0].get("close"))
    target_date = pd.Timestamp(future.iloc[0]["date"]).strftime("%Y-%m-%d")
    if close is None or next_close is None or close == 0:
        return None, "", "pending", target_date
    actual_return = next_close / close - 1
    actual_direction = "up" if actual_return > 0 else "down"
    result = "hit" if actual_direction == predicted_direction else "miss"
    return float(actual_return), actual_direction, result, target_date


def _read_shadow_predictions(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=SHADOW_COLUMNS)
    rows = pd.read_csv(csv_path)
    for column in SHADOW_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    result = rows[SHADOW_COLUMNS].copy()
    for column in SHADOW_COLUMNS:
        if column not in {"probability_up", "probability_down", "confidence", "actual_return"}:
            result[column] = result[column].fillna("").astype(object)
    return result


def _merge_shadow_predictions(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new_rows[SHADOW_COLUMNS].copy() if not new_rows.empty else pd.DataFrame(columns=SHADOW_COLUMNS)
    if new_rows.empty:
        return existing[SHADOW_COLUMNS].copy()
    combined = pd.concat([existing[SHADOW_COLUMNS], new_rows[SHADOW_COLUMNS]], ignore_index=True)
    combined["_feature_date_sort"] = pd.to_datetime(combined["feature_date"], errors="coerce")
    combined["_run_sort"] = pd.to_datetime(combined["run_timestamp"], errors="coerce")
    combined = combined.sort_values(["_feature_date_sort", "_run_sort"])
    combined = combined.drop_duplicates(["ticker", "feature_date", "candidate_name"], keep="last")
    combined = combined.drop(columns=["_feature_date_sort", "_run_sort"])
    return combined.sort_values(["feature_date", "candidate_name"]).reset_index(drop=True)


def _normalize_live_predictions(live_predictions: pd.DataFrame | None) -> pd.DataFrame:
    columns = ["feature_date", "production_live_direction", "production_live_confidence"]
    if live_predictions is None or live_predictions.empty:
        return pd.DataFrame(columns=columns)
    rows = live_predictions.copy()
    if "target" in rows.columns:
        rows = rows[rows["target"].astype(str).eq("next_close_binary")]
    rows["feature_date"] = pd.to_datetime(rows.get("date"), errors="coerce").dt.strftime("%Y-%m-%d")
    rows["production_live_direction"] = rows.get("predicted_class", pd.Series(dtype=object)).map(
        lambda value: "up" if safe_int(value) == 1 else "down" if safe_int(value) == 0 else ""
    )
    rows["production_live_confidence"] = rows.get("confidence", pd.Series(dtype=object)).map(safe_float)
    return rows[columns].dropna(subset=["feature_date"]).reset_index(drop=True)


def _direction_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    if rows.empty:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "up_recall": None,
            "down_recall": None,
            "up_precision": None,
            "down_precision": None,
        }
    actual = rows["actual_direction"].map(_direction_to_int)
    predicted = rows["predicted_direction"].map(_direction_to_int)
    valid = actual.notna() & predicted.notna()
    if not bool(valid.any()):
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "up_recall": None,
            "down_recall": None,
            "up_precision": None,
            "down_precision": None,
        }
    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    up_recall = _safe_divide(tp, tp + fn)
    down_recall = _safe_divide(tn, tn + fp)
    return {
        "accuracy": _safe_divide(tp + tn, len(actual)),
        "balanced_accuracy": (
            (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None
        ),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "up_precision": _safe_divide(tp, tp + fp),
        "down_precision": _safe_divide(tn, tn + fn),
    }


def _summary_markdown(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    predictions: pd.DataFrame,
    conclusions: list[str],
) -> str:
    latest = predictions.sort_values("feature_date").tail(8) if not predictions.empty else pd.DataFrame()
    pending = predictions[predictions["result"].eq("pending")].copy() if not predictions.empty else pd.DataFrame()
    confirmed = predictions[predictions["result"].isin(["hit", "miss"])].copy() if not predictions.empty else pd.DataFrame()
    lines = [
        "# Binary Shadow Evaluation",
        "",
        "This report tracks shadow binary candidates outside production. It does not write to SQLite and does not change daily_predict or production model artifacts.",
        "",
        "## Conclusions",
        *[f"- {item}" for item in conclusions],
        "",
        "## Candidate metrics",
        _markdown_table(summary) if not summary.empty else "No shadow rows.",
        "",
        "## Latest shadow predictions",
        _markdown_table(latest[SHADOW_COLUMNS]) if not latest.empty else "No latest rows.",
        "",
        "## Pending predictions",
        _markdown_table(pending[SHADOW_COLUMNS]) if not pending.empty else "No pending rows.",
        "",
        "## Confirmed predictions",
        _markdown_table(confirmed[SHADOW_COLUMNS].tail(20)) if not confirmed.empty else "No confirmed rows.",
        "",
        "## Production live comparison",
        _markdown_table(comparison) if not comparison.empty else "No matching live predictions were available.",
        "",
        "## Caveats",
        "- Shadow rows are research records only.",
        "- production_current is included for comparison, but no SQLite live rows are created here.",
        "- class_weight_balanced is not a production model change.",
        "- Keep production_change_not_ready until enough confirmed shadow rows accumulate.",
    ]
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    normalized = frame.where(pd.notna(frame), "")
    headers = [str(column) for column in normalized.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in normalized.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
    return json.loads(normalized.where(pd.notna(normalized), None).to_json(orient="records"))


def _date_text(value) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _next_business_day(feature_date: pd.Timestamp) -> str:
    next_day = pd.bdate_range(feature_date + pd.Timedelta(days=1), periods=1)
    if len(next_day) == 0:
        return ""
    return pd.Timestamp(next_day[0]).strftime("%Y-%m-%d")


def _direction_to_int(value) -> int | None:
    text = str(value or "").strip().lower()
    if text == "up":
        return 1
    if text == "down":
        return 0
    return None


def _summary_candidate(summary: pd.DataFrame, candidate: str) -> pd.Series | None:
    match = summary[summary["candidate_name"].astype(str).eq(candidate)]
    if match.empty:
        return None
    return match.iloc[0]


def _gt(value, reference) -> bool:
    left = safe_float(value)
    right = safe_float(reference)
    return left is not None and right is not None and left > right


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _append_text(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"
