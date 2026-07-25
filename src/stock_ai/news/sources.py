from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - PyYAML is listed in requirements.
    yaml = None

from stock_ai.config import PROJECT_ROOT
from stock_ai.utils.safe_cast import safe_float


NEWS_SOURCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "news_sources.yaml"
SOURCE_TYPES = {
    "official",
    "wire",
    "financial_media",
    "general_news",
    "aggregator",
    "unknown",
}


@dataclass(frozen=True)
class NewsSourceProfile:
    name: str
    aliases: tuple[str, ...]
    credibility_score: float
    market_impact_weight: float
    source_type: str


DEFAULT_SOURCE_PROFILE = NewsSourceProfile(
    name="Unknown",
    aliases=("unknown", ""),
    credibility_score=0.30,
    market_impact_weight=0.25,
    source_type="unknown",
)


def load_news_source_profiles(path: str | Path = NEWS_SOURCE_CONFIG_PATH) -> list[NewsSourceProfile]:
    config_path = Path(path)
    if not config_path.exists():
        return [DEFAULT_SOURCE_PROFILE]
    raw = _load_yaml(config_path)
    rows = raw.get("sources", []) if isinstance(raw, dict) else []
    profiles: list[NewsSourceProfile] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        aliases = tuple(
            str(alias or "").strip().lower()
            for alias in row.get("aliases", [])
            if str(alias or "").strip() or alias == ""
        )
        source_type = _normalize_source_type(row.get("source_type"))
        profiles.append(
            NewsSourceProfile(
                name=name,
                aliases=aliases or (name.lower(),),
                credibility_score=_clamped_score(row.get("credibility_score"), DEFAULT_SOURCE_PROFILE.credibility_score),
                market_impact_weight=_clamped_score(
                    row.get("market_impact_weight"),
                    DEFAULT_SOURCE_PROFILE.market_impact_weight,
                ),
                source_type=source_type,
            )
        )
    return profiles or [DEFAULT_SOURCE_PROFILE]


def resolve_news_source_profile(
    source: str,
    path: str | Path = NEWS_SOURCE_CONFIG_PATH,
) -> NewsSourceProfile:
    source_text = str(source or "").strip().lower()
    profiles = load_news_source_profiles(path)
    unknown = next((profile for profile in profiles if profile.source_type == "unknown"), DEFAULT_SOURCE_PROFILE)
    if not source_text:
        return unknown
    for profile in profiles:
        tokens = {profile.name.lower(), *profile.aliases}
        if any(token and token in source_text for token in tokens):
            return profile
    return unknown


def news_source_profile_rows(path: str | Path = NEWS_SOURCE_CONFIG_PATH) -> list[dict[str, object]]:
    return [
        {
            "name": profile.name,
            "aliases": ", ".join(alias for alias in profile.aliases if alias),
            "credibility_score": profile.credibility_score,
            "market_impact_weight": profile.market_impact_weight,
            "source_type": profile.source_type,
        }
        for profile in load_news_source_profiles(path)
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}
    return _load_simple_sources_yaml(path)


def _load_simple_sources_yaml(path: Path) -> dict[str, Any]:
    sources: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        stripped = line.strip()
        if not stripped or stripped == "sources:":
            continue
        if stripped.startswith("- "):
            if current:
                sources.append(current)
            current = {}
            body = stripped[2:].strip()
            if ":" in body:
                key, value = body.split(":", maxsplit=1)
                current[key.strip()] = _strip_yaml_value(value.strip())
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", maxsplit=1)
            current[key.strip()] = _strip_yaml_value(value.strip())
    if current:
        sources.append(current)
    return {"sources": sources}


def _strip_yaml_value(value: str) -> object:
    clean = value.strip()
    if clean.startswith("[") and clean.endswith("]"):
        return [
            _strip_quotes(item.strip())
            for item in clean[1:-1].split(",")
            if item.strip() or item.strip() == '""'
        ]
    return _strip_quotes(clean)


def _strip_quotes(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _normalize_source_type(value: object) -> str:
    source_type = str(value or "unknown").strip()
    return source_type if source_type in SOURCE_TYPES else "unknown"


def _clamped_score(value: object, default: float) -> float:
    numeric = safe_float(value)
    if numeric is None:
        numeric = default
    return round(max(0.0, min(1.0, float(numeric))), 3)
