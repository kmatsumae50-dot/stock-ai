from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

from stock_ai.config import PROJECT_ROOT


SECTOR_JA_MAP: dict[str, str] = {
    "Automotive": "自動車",
    "Semiconductors": "半導体",
    "Banking & Finance": "銀行・金融",
    "Wholesale & Trading": "総合商社",
    "Electronics": "電機・精密",
    "Machinery & Defense": "防衛・重工",
    "Telecommunications & Tech": "通信・AI投資",
    "Telecommunications": "通信・ディフェンシブ",
    "Entertainment & Gaming": "ゲーム・エンタメ",
    "Retail": "小売・内需",
    "Pharmaceuticals": "医薬品",
    "Chemicals & Materials": "化学・素材",
}


@dataclass(frozen=True)
class TickerInfo:
    code: str
    name: str = ""
    sector: str = ""

    @property
    def sector_ja(self) -> str:
        return SECTOR_JA_MAP.get(self.sector, self.sector)


def load_universe(path: str | Path | None = None) -> list[TickerInfo]:
    universe_path = Path(path) if path else PROJECT_ROOT / "configs" / "universe.yml"
    if yaml is not None:
        raw = yaml.safe_load(universe_path.read_text(encoding="utf-8"))
        return [
            TickerInfo(
                code=item["code"],
                name=item.get("name", ""),
                sector=item.get("sector", ""),
            )
            for item in raw["tickers"]
        ]
    return _load_simple_universe(universe_path)


def _load_simple_universe(path: Path) -> list[TickerInfo]:
    tickers: list[TickerInfo] = []
    current: dict[str, str] = {}

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line:
            continue

        if line.startswith("- "):
            if current:
                tickers.append(
                    TickerInfo(
                        code=current.get("code", ""),
                        name=current.get("name", ""),
                        sector=current.get("sector", ""),
                    )
                )
            current = {}
            line = line[2:].strip()

        if ":" in line:
            key, value = line.split(":", maxsplit=1)
            current[key.strip()] = value.strip().strip('"').strip("'")

    if current:
        tickers.append(
            TickerInfo(
                code=current.get("code", ""),
                name=current.get("name", ""),
                sector=current.get("sector", ""),
            )
        )

    return [ticker for ticker in tickers if ticker.code]
