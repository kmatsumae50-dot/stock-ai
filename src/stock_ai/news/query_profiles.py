from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - PyYAML is listed in requirements.
    yaml = None

from stock_ai.config import PROJECT_ROOT


NEWS_QUERY_CONFIG_PATH = PROJECT_ROOT / "configs" / "news_queries.yaml"
DEFAULT_NEWS_STORAGE_TICKER = "7203.T"


@dataclass(frozen=True)
class NewsQueryProfile:
    name: str
    query: str
    ticker: str = ""
    category: str = "auto"
    language: str = ""
    region: str = ""
    enabled_for_daily: bool = False
    group: str = "research"

    def storage_ticker(self, default_ticker: str = DEFAULT_NEWS_STORAGE_TICKER) -> str:
        return self.ticker.strip() or default_ticker

    def row_ticker(self) -> str:
        return self.ticker.strip()


def load_news_query_profiles(path: str | Path = NEWS_QUERY_CONFIG_PATH) -> list[NewsQueryProfile]:
    config_path = Path(path)
    if not config_path.exists():
        return []
    raw = _load_yaml(config_path)
    rows = raw.get("profiles", []) if isinstance(raw, dict) else []
    profiles: list[NewsQueryProfile] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        query = str(row.get("query") or "").strip()
        if not name or not query or name in seen:
            continue
        seen.add(name)
        profiles.append(
            NewsQueryProfile(
                name=name,
                query=query,
                ticker=str(row.get("ticker") or "").strip(),
                category=str(row.get("category") or "auto").strip() or "auto",
                language=str(row.get("language") or "").strip(),
                region=str(row.get("region") or "").strip(),
                enabled_for_daily=_as_bool(row.get("enabled_for_daily"), default=False),
                group=str(row.get("group") or "research").strip() or "research",
            )
        )
    return profiles


def get_news_query_profile(
    name: str,
    path: str | Path = NEWS_QUERY_CONFIG_PATH,
) -> NewsQueryProfile:
    normalized = str(name or "").strip()
    for profile in load_news_query_profiles(path):
        if profile.name == normalized:
            return profile
    raise ValueError(f"Unknown news query profile: {name}")


def news_query_profile_rows(path: str | Path = NEWS_QUERY_CONFIG_PATH) -> list[dict[str, str]]:
    return [
        {
            "name": profile.name,
            "query": profile.query,
            "ticker": profile.ticker or "",
            "storage_ticker": profile.storage_ticker(),
            "category": profile.category,
            "language": profile.language,
            "region": profile.region,
            "enabled_for_daily": str(profile.enabled_for_daily),
            "group": profile.group,
        }
        for profile in load_news_query_profiles(path)
    ]


def daily_news_query_profiles(path: str | Path = NEWS_QUERY_CONFIG_PATH) -> list[NewsQueryProfile]:
    return [profile for profile in load_news_query_profiles(path) if profile.enabled_for_daily]


def grouped_news_query_profile_rows(path: str | Path = NEWS_QUERY_CONFIG_PATH) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for profile in load_news_query_profiles(path):
        rows.append(
            {
                "group": profile.group,
                "name": profile.name,
                "enabled_for_daily": profile.enabled_for_daily,
                "language": profile.language,
                "region": profile.region,
                "category": profile.category,
                "ticker": profile.ticker or "",
                "query": profile.query,
            }
        )
    return rows


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is not None:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return loaded if isinstance(loaded, dict) else {}
    return _load_simple_profiles_yaml(path)


def _load_simple_profiles_yaml(path: Path) -> dict[str, Any]:
    profiles: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].rstrip()
        stripped = line.strip()
        if not stripped or stripped == "profiles:":
            continue
        if stripped.startswith("- "):
            if current:
                profiles.append(current)
            current = {}
            body = stripped[2:].strip()
            if ":" in body:
                key, value = body.split(":", maxsplit=1)
                current[key.strip()] = _strip_quotes(value.strip())
            continue
        if current is not None and ":" in stripped:
            key, value = stripped.split(":", maxsplit=1)
            current[key.strip()] = _strip_quotes(value.strip())
    if current:
        profiles.append(current)
    return {"profiles": profiles}


def _strip_quotes(value: str) -> str:
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default
