from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised when PyYAML is not installed.
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathSettings:
    raw_data_dir: Path
    processed_data_dir: Path
    feature_data_dir: Path
    prediction_data_dir: Path
    model_dir: Path
    database_path: Path


@dataclass(frozen=True)
class LabelSettings:
    surge_threshold: float
    up_threshold: float
    down_threshold: float
    crash_threshold: float


@dataclass(frozen=True)
class FeatureSettings:
    moving_average_windows: tuple[int, ...]
    rsi_window: int
    volatility_window: int
    bollinger_window: int


@dataclass(frozen=True)
class KnownMissedLiveDateSettings:
    date: str | None = None
    ticker: str | None = None
    reason_code: str | None = None
    reason: str | None = None
    action: str | None = None
    allow_backfill: bool = False


@dataclass(frozen=True)
class OperationsSettings:
    live_prediction_start_date: str | None = None
    known_missed_live_dates: tuple[KnownMissedLiveDateSettings, ...] = ()


@dataclass(frozen=True)
class Settings:
    paths: PathSettings
    labels: LabelSettings
    features: FeatureSettings
    operations: OperationsSettings
    data: dict[str, Any]


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_settings(path: str | Path | None = None) -> Settings:
    settings_path = Path(path) if path else PROJECT_ROOT / "configs" / "settings.yml"
    if yaml is not None:
        with settings_path.open("r", encoding="utf-8") as file:
            raw = yaml.safe_load(file)
    else:
        raw = _load_simple_yaml(settings_path)

    paths = raw["paths"]
    labels = raw["labels"]
    features = raw["features"]
    operations = raw.get("operations", {}) or {}

    return Settings(
        paths=PathSettings(
            raw_data_dir=_resolve_project_path(paths["raw_data_dir"]),
            processed_data_dir=_resolve_project_path(paths["processed_data_dir"]),
            feature_data_dir=_resolve_project_path(paths["feature_data_dir"]),
            prediction_data_dir=_resolve_project_path(paths["prediction_data_dir"]),
            model_dir=_resolve_project_path(paths["model_dir"]),
            database_path=_resolve_project_path(paths["database_path"]),
        ),
        labels=LabelSettings(
            surge_threshold=float(labels["surge_threshold"]),
            up_threshold=float(labels["up_threshold"]),
            down_threshold=float(labels["down_threshold"]),
            crash_threshold=float(labels["crash_threshold"]),
        ),
        features=FeatureSettings(
            moving_average_windows=tuple(int(v) for v in features["moving_average_windows"]),
            rsi_window=int(features["rsi_window"]),
            volatility_window=int(features["volatility_window"]),
            bollinger_window=int(features["bollinger_window"]),
        ),
        operations=OperationsSettings(
            live_prediction_start_date=operations.get("live_prediction_start_date"),
            known_missed_live_dates=_parse_known_missed_live_dates(
                operations.get("known_missed_live_dates")
            ),
        ),
        data=raw.get("data", {}),
    )


def _parse_known_missed_live_dates(value: Any) -> tuple[KnownMissedLiveDateSettings, ...]:
    if not isinstance(value, list):
        return ()
    rows: list[KnownMissedLiveDateSettings] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        rows.append(
            KnownMissedLiveDateSettings(
                date=_optional_str(item.get("date")),
                ticker=_optional_str(item.get("ticker")),
                reason_code=_optional_str(item.get("reason_code")),
                reason=_optional_str(item.get("reason")),
                action=_optional_str(item.get("action")),
                allow_backfill=bool(item.get("allow_backfill", False)),
            )
        )
    return tuple(rows)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def ensure_project_dirs(settings: Settings) -> None:
    for path in (
        settings.paths.raw_data_dir,
        settings.paths.processed_data_dir,
        settings.paths.feature_data_dir,
        settings.paths.prediction_data_dir,
        settings.paths.model_dir,
        settings.paths.database_path.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Load the small project settings file without requiring PyYAML."""
    result: dict[str, Any] = {}
    current_section: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
            continue

        if current_section is None or ":" not in line:
            continue

        key, value = line.strip().split(":", maxsplit=1)
        result[current_section][key.strip()] = _parse_scalar(value.strip())

    return result


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(item.strip()) for item in body.split(",")]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
