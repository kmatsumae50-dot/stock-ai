from __future__ import annotations

from dataclasses import dataclass
from math import inf

import pandas as pd

from stock_ai.labels.targets import CLASS_NAMES, classify_return

DEFAULT_THRESHOLD_PROFILE = "original_5_1"
DEFAULT_WEIGHT_MODE = "none"
MULTICLASS_MODEL_BASE_NAME = "lightgbm_multiclass"
MANUAL_TAIL_BOOST_WEIGHTS = {
    0: 3.0,  # crash
    1: 1.0,  # down
    2: 0.8,  # flat
    3: 1.0,  # up
    4: 3.0,  # surge
}


@dataclass(frozen=True)
class LabelScheme:
    key: str
    name: str
    description: str
    kind: str
    surge_threshold: float | None = None
    up_threshold: float | None = None
    down_threshold: float | None = None
    crash_threshold: float | None = None
    surge_quantile: float | None = None
    up_quantile: float | None = None
    down_quantile: float | None = None
    crash_quantile: float | None = None


def threshold_profile_options() -> list[str]:
    return [scheme.key for scheme in threshold_profile_schemes()]


def weight_mode_options() -> list[str]:
    return ["none", "balanced", "manual_tail_boost"]


def threshold_profile_schemes() -> list[LabelScheme]:
    return [
        fixed_label_scheme(
            key="original_5_1",
            name="original_5_1",
            surge_threshold=0.05,
            up_threshold=0.01,
            down_threshold=-0.01,
            crash_threshold=-0.05,
            description="Original: surge/crash at +/-5%, up/down at +/-1%.",
        ),
        fixed_label_scheme(
            key="practical_3_0_5",
            name="practical_3_0_5",
            surge_threshold=0.03,
            up_threshold=0.005,
            down_threshold=-0.005,
            crash_threshold=-0.03,
            description="Practical: surge/crash at +/-3%, up/down at +/-0.5%.",
        ),
        quantile_label_scheme(
            key="quantile",
            name="quantile",
            description=(
                "Quantile: bottom 5% crash, bottom 30% down, middle 40% flat, "
                "top 30% up, top 5% surge."
            ),
        ),
    ]


def normalize_threshold_profile(profile: str | None) -> str:
    if profile is None or not str(profile).strip():
        return DEFAULT_THRESHOLD_PROFILE
    value = str(profile).strip()
    aliases = {
        "fixed_5_1": "original_5_1",
        "original": "original_5_1",
        "current": "original_5_1",
        "fixed_3_0_5": "practical_3_0_5",
        "practical": "practical_3_0_5",
        "quantile_5_30_70_95": "quantile",
    }
    value = aliases.get(value, value)
    valid = set(threshold_profile_options())
    if value not in valid:
        raise ValueError(f"Unknown threshold profile: {profile}. Valid profiles: {sorted(valid)}")
    return value


def normalize_weight_mode(weight_mode: str | None) -> str:
    if weight_mode is None or not str(weight_mode).strip():
        return DEFAULT_WEIGHT_MODE
    value = str(weight_mode).strip()
    aliases = {
        "default": "none",
        "off": "none",
        "tailboost": "manual_tail_boost",
        "tail_boost": "manual_tail_boost",
        "manual": "manual_tail_boost",
    }
    value = aliases.get(value, value)
    valid = set(weight_mode_options())
    if value not in valid:
        raise ValueError(f"Unknown weight mode: {weight_mode}. Valid modes: {sorted(valid)}")
    return value


def get_threshold_profile(profile: str | None) -> LabelScheme:
    key = normalize_threshold_profile(profile)
    for scheme in threshold_profile_schemes():
        if scheme.key == key:
            return scheme
    raise ValueError(f"Unknown threshold profile: {profile}")


def multiclass_class_weight(weight_mode: str | None = None):
    mode = normalize_weight_mode(weight_mode)
    if mode == "none":
        return None
    if mode == "balanced":
        return "balanced"
    if mode == "manual_tail_boost":
        return MANUAL_TAIL_BOOST_WEIGHTS.copy()
    raise ValueError(f"Unsupported weight mode: {weight_mode}")


def weight_mode_suffix(weight_mode: str | None = None) -> str:
    mode = normalize_weight_mode(weight_mode)
    if mode == "none":
        return ""
    if mode == "balanced":
        return "balanced"
    if mode == "manual_tail_boost":
        return "tailboost"
    raise ValueError(f"Unsupported weight mode: {weight_mode}")


def multiclass_model_name(profile: str | None = None, weight_mode: str | None = None) -> str:
    key = normalize_threshold_profile(profile)
    mode_suffix = weight_mode_suffix(weight_mode)
    parts = [MULTICLASS_MODEL_BASE_NAME]
    if key != DEFAULT_THRESHOLD_PROFILE:
        parts.append(key)
    if mode_suffix:
        parts.append(mode_suffix)
    return "_".join(parts)


def multiclass_artifact_stem(
    ticker: str,
    profile: str | None = None,
    weight_mode: str | None = None,
) -> str:
    return f"{multiclass_model_name(profile, weight_mode)}_{ticker}"


def profile_from_model_name(model_name: str | None) -> str:
    text = str(model_name or "")
    prefix = f"{MULTICLASS_MODEL_BASE_NAME}_"
    if text == MULTICLASS_MODEL_BASE_NAME:
        return DEFAULT_THRESHOLD_PROFILE
    if text.startswith(prefix):
        candidate = text.removeprefix(prefix)
        for suffix in ("_balanced", "_tailboost"):
            if candidate.endswith(suffix):
                candidate = candidate[: -len(suffix)]
        try:
            return normalize_threshold_profile(candidate)
        except ValueError:
            return DEFAULT_THRESHOLD_PROFILE
    return DEFAULT_THRESHOLD_PROFILE


def weight_mode_from_model_name(model_name: str | None) -> str:
    text = str(model_name or "")
    if text.endswith("_balanced"):
        return "balanced"
    if text.endswith("_tailboost"):
        return "manual_tail_boost"
    return DEFAULT_WEIGHT_MODE


def fixed_label_scheme(
    *,
    key: str,
    name: str,
    surge_threshold: float,
    up_threshold: float,
    down_threshold: float,
    crash_threshold: float,
    description: str | None = None,
) -> LabelScheme:
    return LabelScheme(
        key=key,
        name=name,
        description=description
        or (
            f"surge >= {surge_threshold:.2%}, up >= {up_threshold:.2%}, "
            f"down <= {down_threshold:.2%}, crash <= {crash_threshold:.2%}"
        ),
        kind="fixed",
        surge_threshold=surge_threshold,
        up_threshold=up_threshold,
        down_threshold=down_threshold,
        crash_threshold=crash_threshold,
    )


def quantile_label_scheme(
    *,
    key: str = "quantile",
    name: str = "quantile",
    description: str = "bottom 5% crash, bottom 30% down, middle 40% flat, top 30% up, top 5% surge",
) -> LabelScheme:
    return LabelScheme(
        key=key,
        name=name,
        description=description,
        kind="quantile",
        crash_quantile=0.05,
        down_quantile=0.30,
        up_quantile=0.70,
        surge_quantile=0.95,
    )


def default_candidate_schemes() -> list[LabelScheme]:
    return [
        fixed_label_scheme(
            key="original_5_1",
            name="A: original_5_1",
            surge_threshold=0.05,
            up_threshold=0.01,
            down_threshold=-0.01,
            crash_threshold=-0.05,
        ),
        fixed_label_scheme(
            key="practical_3_0_5",
            name="B: practical_3_0_5",
            surge_threshold=0.03,
            up_threshold=0.005,
            down_threshold=-0.005,
            crash_threshold=-0.03,
        ),
        quantile_label_scheme(),
    ]


def label_scheme_from_settings(settings) -> LabelScheme:
    labels = settings.labels
    return fixed_label_scheme(
        key=DEFAULT_THRESHOLD_PROFILE,
        name=DEFAULT_THRESHOLD_PROFILE,
        surge_threshold=float(labels.surge_threshold),
        up_threshold=float(labels.up_threshold),
        down_threshold=float(labels.down_threshold),
        crash_threshold=float(labels.crash_threshold),
    )


def scheme_to_dict(scheme: LabelScheme) -> dict[str, object]:
    return {
        "key": scheme.key,
        "name": scheme.name,
        "description": scheme.description,
        "kind": scheme.kind,
        "surge_threshold": scheme.surge_threshold,
        "up_threshold": scheme.up_threshold,
        "down_threshold": scheme.down_threshold,
        "crash_threshold": scheme.crash_threshold,
        "surge_quantile": scheme.surge_quantile,
        "up_quantile": scheme.up_quantile,
        "down_quantile": scheme.down_quantile,
        "crash_quantile": scheme.crash_quantile,
    }


def ensure_next_close_return(
    data: pd.DataFrame,
    *,
    return_column: str = "next_close_return",
) -> pd.DataFrame:
    return _ensure_return_column(data, return_column=return_column)


def resolve_label_thresholds(
    data: pd.DataFrame,
    scheme: LabelScheme,
    *,
    return_column: str = "next_close_return",
) -> dict[str, float | None]:
    frame = _ensure_return_column(data, return_column=return_column)
    return _scheme_thresholds(pd.to_numeric(frame[return_column], errors="coerce"), scheme)


def add_target_for_scheme(
    data: pd.DataFrame,
    scheme: LabelScheme,
    *,
    target_column: str = "target_next_close_class",
    return_column: str = "next_close_return",
    thresholds: dict[str, float | None] | None = None,
) -> pd.DataFrame:
    frame = _ensure_return_column(data, return_column=return_column)
    active_thresholds = thresholds or resolve_label_thresholds(frame, scheme, return_column=return_column)
    frame[target_column] = pd.to_numeric(frame[return_column], errors="coerce").map(
        lambda value: _classify_with_thresholds(
            value,
            surge_threshold=active_thresholds["surge_threshold"],
            up_threshold=active_thresholds["up_threshold"],
            down_threshold=active_thresholds["down_threshold"],
            crash_threshold=active_thresholds["crash_threshold"],
        )
    )
    return frame


def analyze_label_distribution(
    data: pd.DataFrame,
    scheme: LabelScheme,
    *,
    return_column: str = "next_close_return",
) -> dict[str, object]:
    frame = _ensure_return_column(data, return_column=return_column)
    labels, thresholds = label_returns(frame[return_column], scheme)
    labeled = frame.copy()
    labeled["label_class"] = labels
    labeled = labeled[labeled["label_class"] >= 0].copy()
    labeled["label_name"] = labeled["label_class"].map(CLASS_NAMES)
    if "date" in labeled.columns:
        labeled["year"] = pd.to_datetime(labeled["date"], errors="coerce").dt.year

    distribution = class_distribution(labeled["label_class"])
    summary = summarize_distribution(distribution)
    return {
        "scheme": scheme,
        "thresholds": thresholds,
        "labeled_data": labeled,
        "distribution": distribution,
        "summary": summary,
    }


def label_returns(returns: pd.Series, scheme: LabelScheme) -> tuple[pd.Series, dict[str, float | None]]:
    clean_returns = pd.to_numeric(returns, errors="coerce")
    thresholds = _scheme_thresholds(clean_returns, scheme)
    labels = clean_returns.map(
        lambda value: _classify_with_thresholds(
            value,
            surge_threshold=thresholds["surge_threshold"],
            up_threshold=thresholds["up_threshold"],
            down_threshold=thresholds["down_threshold"],
            crash_threshold=thresholds["crash_threshold"],
        )
    )
    return labels.astype(int), thresholds


def class_distribution(labels: pd.Series) -> pd.DataFrame:
    valid = pd.to_numeric(labels, errors="coerce")
    valid = valid[valid >= 0].astype(int)
    total = int(len(valid))
    counts = valid.value_counts().to_dict()
    rows = []
    for class_id in sorted(CLASS_NAMES):
        count = int(counts.get(class_id, 0))
        rows.append(
            {
                "class_id": class_id,
                "class": CLASS_NAMES[class_id],
                "count": count,
                "share": count / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def summarize_distribution(distribution: pd.DataFrame) -> dict[str, object]:
    if distribution.empty:
        return {
            "total": 0,
            "min_class_count": 0,
            "max_class_count": 0,
            "imbalance_ratio": inf,
            "trainability": "Weak",
            "warnings": ["No labels are available."],
        }

    counts = pd.to_numeric(distribution["count"], errors="coerce").fillna(0)
    total = int(counts.sum())
    min_count = int(counts.min()) if len(counts) else 0
    max_count = int(counts.max()) if len(counts) else 0
    imbalance = (max_count / min_count) if min_count > 0 else inf
    shares = {
        str(row["class"]): float(row["share"])
        for _, row in distribution.iterrows()
    }

    warnings: list[str] = []
    for class_name in ("crash", "surge"):
        count = int(distribution.loc[distribution["class"] == class_name, "count"].iloc[0])
        share = float(distribution.loc[distribution["class"] == class_name, "share"].iloc[0])
        if count < 30:
            warnings.append(f"{class_name} has fewer than 30 samples.")
        if share < 0.01:
            warnings.append(f"{class_name} is below 1% of samples.")

    if min_count == 0:
        trainability = "Weak"
    elif min_count < 30 or imbalance >= 20:
        trainability = "Weak"
    elif min_count < 100 or imbalance >= 10:
        trainability = "Caution"
    else:
        trainability = "Usable"

    return {
        "total": total,
        "min_class_count": min_count,
        "max_class_count": max_count,
        "imbalance_ratio": imbalance,
        "trainability": trainability,
        "warnings": warnings,
        "shares": shares,
    }


def distribution_by_group(labeled_data: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if labeled_data.empty or group_column not in labeled_data.columns:
        return pd.DataFrame()
    grouped = (
        labeled_data.dropna(subset=[group_column])
        .groupby([group_column, "label_name"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    if grouped.empty:
        return pd.DataFrame()

    pivot = grouped.pivot(index=group_column, columns="label_name", values="count").fillna(0)
    ordered_columns = [CLASS_NAMES[class_id] for class_id in sorted(CLASS_NAMES)]
    for column in ordered_columns:
        if column not in pivot.columns:
            pivot[column] = 0
    pivot = pivot[ordered_columns].astype(int)
    pivot["total"] = pivot.sum(axis=1)
    for column in ordered_columns:
        pivot[f"{column}_share"] = pivot[column] / pivot["total"].replace(0, pd.NA)
    return pivot.reset_index()


def compare_label_schemes(
    data: pd.DataFrame,
    schemes: list[LabelScheme] | None = None,
    *,
    return_column: str = "next_close_return",
) -> pd.DataFrame:
    candidates = schemes or default_candidate_schemes()
    rows: list[dict[str, object]] = []
    for scheme in candidates:
        analysis = analyze_label_distribution(data, scheme, return_column=return_column)
        distribution = analysis["distribution"]
        summary = analysis["summary"]
        row: dict[str, object] = {
            "scheme": scheme.name,
            "description": scheme.description,
            "trainability": summary["trainability"],
            "total": summary["total"],
            "min_class_count": summary["min_class_count"],
            "imbalance_ratio": summary["imbalance_ratio"],
        }
        for _, item in distribution.iterrows():
            class_name = str(item["class"])
            row[f"{class_name}_count"] = int(item["count"])
            row[f"{class_name}_share"] = float(item["share"])
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_return_column(data: pd.DataFrame, *, return_column: str) -> pd.DataFrame:
    frame = data.copy()
    if return_column in frame.columns:
        return frame
    if "close" not in frame.columns:
        raise ValueError(f"{return_column} is missing and close is not available.")
    sort_columns = [column for column in ("ticker", "date") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns).copy()
    if "ticker" in frame.columns:
        grouped = frame.groupby("ticker", group_keys=False)
        next_close = grouped["close"].shift(-1)
    else:
        next_close = frame["close"].shift(-1)
    frame["next_close"] = next_close
    frame[return_column] = frame["next_close"] / frame["close"] - 1
    return frame


def _scheme_thresholds(returns: pd.Series, scheme: LabelScheme) -> dict[str, float | None]:
    if scheme.kind == "fixed":
        return {
            "surge_threshold": scheme.surge_threshold,
            "up_threshold": scheme.up_threshold,
            "down_threshold": scheme.down_threshold,
            "crash_threshold": scheme.crash_threshold,
        }
    if scheme.kind == "quantile":
        return {
            "surge_threshold": _quantile(returns, scheme.surge_quantile),
            "up_threshold": _quantile(returns, scheme.up_quantile),
            "down_threshold": _quantile(returns, scheme.down_quantile),
            "crash_threshold": _quantile(returns, scheme.crash_quantile),
        }
    raise ValueError(f"Unsupported label scheme kind: {scheme.kind}")


def _quantile(returns: pd.Series, value: float | None) -> float | None:
    if value is None:
        return None
    clean = pd.to_numeric(returns, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(value))


def _classify_with_thresholds(
    value: float,
    *,
    surge_threshold: float | None,
    up_threshold: float | None,
    down_threshold: float | None,
    crash_threshold: float | None,
) -> int:
    if any(
        threshold is None
        for threshold in (surge_threshold, up_threshold, down_threshold, crash_threshold)
    ):
        return -1
    return classify_return(
        value,
        surge_threshold=float(surge_threshold),
        up_threshold=float(up_threshold),
        down_threshold=float(down_threshold),
        crash_threshold=float(crash_threshold),
    )
