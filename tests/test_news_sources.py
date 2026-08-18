from stock_ai.news.fetcher import score_news_importance_with_reason
from stock_ai.news.sources import (
    load_news_source_profiles,
    resolve_news_source_profile,
)


def test_resolve_news_source_profile_matches_alias():
    profile = resolve_news_source_profile("Reuters")

    assert profile.name == "Reuters"
    assert profile.source_type == "wire"
    assert profile.credibility_score >= 0.8
    assert profile.market_impact_weight >= 0.8


def test_resolve_news_source_profile_falls_back_to_unknown():
    profile = resolve_news_source_profile("Small Unknown Blog")

    assert profile.name == "Unknown"
    assert profile.source_type == "unknown"


def test_resolve_added_source_profiles():
    expected = {
        "TheStreet Pro": "financial_media",
        "Stock Titan": "financial_media",
        "simplywall.st": "financial_media",
        "CleanTechnica": "general_news",
        "Seeking Alpha": "financial_media",
        "Carscoops": "general_news",
        "Fathom Journal": "general_news",
        "TradingView": "financial_media",
        "Yahoo!ニュース": "aggregator",
        "netdenjd.com": "general_news",
        "supercarblondie.com": "general_news",
        "조선일보": "general_news",
        "Anadolu Ajansı": "wire",
        "Techzine Global": "general_news",
        "株探": "financial_media",
        "かぶたん": "financial_media",
        "日本投資機構株式会社": "financial_media",
        "ダイヤモンド・オンライン": "general_news",
        "ダイヤモンドオンライン": "general_news",
        "Diamond Online": "general_news",
        "DIAMOND online": "general_news",
        "The Telegraph": "general_news",
        "Telegraph": "general_news",
        "The Daily Telegraph": "general_news",
        "Daily Telegraph": "general_news",
        "telegraph.co.uk": "general_news",
        "Yahoo": "aggregator",
        "Frontstretch": "general_news",
        "marketscreener.com": "financial_media",
        "The Motley Fool": "financial_media",
        "トヨタ自動車株式会社 公式企業サイト": "official",
        "Schaeffer's Investment Research": "financial_media",
        "CNA": "general_news",
        "Channel NewsAsia": "general_news",
    }

    for source, source_type in expected.items():
        profile = resolve_news_source_profile(source)
        assert profile.name != "Unknown"
        assert profile.source_type == source_type


def test_recent_unknown_source_profiles_are_conservative():
    yahoo = resolve_news_source_profile("Yahoo")
    frontstretch = resolve_news_source_profile("Frontstretch")
    toyota_official = resolve_news_source_profile("トヨタ自動車株式会社 公式企業サイト")

    assert yahoo.source_type == "aggregator"
    assert yahoo.market_impact_weight <= 0.35
    assert frontstretch.source_type == "general_news"
    assert frontstretch.market_impact_weight <= 0.35
    assert toyota_official.source_type == "official"
    assert 0.35 <= toyota_official.market_impact_weight <= 0.55


def test_carexpert_source_profile_is_conservative():
    for source in ("CarExpert", "carexpert", "carexpert.com.au"):
        profile = resolve_news_source_profile(source)
        assert profile.name == "CarExpert"
        assert profile.source_type == "general_news"
        assert 0.50 <= profile.credibility_score <= 0.60
        assert 0.20 <= profile.market_impact_weight <= 0.35


def test_july_unknown_source_profiles_are_resolved_conservatively():
    expected = {
        "Benzinga": ("Benzinga", "financial_media", 0.50),
        "benzinga.com": ("Benzinga", "financial_media", 0.50),
        "Stocktwits": ("Stocktwits", "aggregator", 0.35),
        "stocktwits.com": ("Stocktwits", "aggregator", 0.35),
        "timothysykes.com": ("Timothy Sykes", "general_news", 0.30),
        "Tim Sykes": ("Timothy Sykes", "general_news", 0.30),
        "産経ニュース": ("Sankei News", "general_news", 0.40),
        "sankei.com": ("Sankei News", "general_news", 0.40),
    }

    for source, (name, source_type, maximum_impact) in expected.items():
        profile = resolve_news_source_profile(source)
        assert profile.name == name
        assert profile.source_type == source_type
        assert profile.market_impact_weight <= maximum_impact
        assert profile.credibility_score < 0.70


def test_stockstotrade_source_profile_is_resolved_conservatively():
    for source in ("StocksToTrade", "stockstotrade", "stockstotrade.com", "Stocks To Trade"):
        profile = resolve_news_source_profile(source)
        assert profile.name == "StocksToTrade"
        assert profile.source_type == "financial_media"
        assert profile.credibility_score == 0.48
        assert profile.market_impact_weight == 0.30


def test_late_july_unknown_source_profiles_are_resolved_conservatively():
    expected = {
        "CarbonCredits.com": ("CarbonCredits.com", "general_news", 0.30),
        "carbon credits": ("CarbonCredits.com", "general_news", 0.30),
        "Top Speed": ("Top Speed", "general_news", 0.35),
        "topspeed.com": ("Top Speed", "general_news", 0.35),
        "yourvalley.net": ("YourValley.net", "general_news", 0.20),
        "ロイター": ("Reuters", "wire", 0.85),
        "jp.reuters.com": ("Reuters", "wire", 0.85),
    }

    for source, (name, source_type, maximum_impact) in expected.items():
        profile = resolve_news_source_profile(source)
        assert profile.name == name
        assert profile.source_type == source_type
        assert profile.market_impact_weight <= maximum_impact

    for source in ("CarbonCredits.com", "Top Speed", "yourvalley.net"):
        assert resolve_news_source_profile(source).credibility_score < 0.60


def test_august_unknown_source_profiles_are_resolved_conservatively():
    expected = {
        "AOL.com": ("AOL", "aggregator", 0.30),
        "Car Watch": ("Car Watch", "general_news", 0.35),
        "car.watch.impress.co.jp": ("Car Watch", "general_news", 0.35),
        "CnEVPost": ("CnEVPost", "general_news", 0.35),
        "cnevpost.com": ("CnEVPost", "general_news", 0.35),
        "Sahm": ("Sahm", "financial_media", 0.40),
        "sahmcapital.com": ("Sahm", "financial_media", 0.40),
        "Stock Traders Daily": ("Stock Traders Daily", "financial_media", 0.30),
        "stocktradersdaily.com": ("Stock Traders Daily", "financial_media", 0.30),
        "qz.com": ("Quartz", "financial_media", 0.45),
        "中日BIZナビ": ("Chunichi BIZ Navi", "financial_media", 0.40),
        "biz.chunichi.co.jp": ("Chunichi BIZ Navi", "financial_media", 0.40),
    }

    for source, (name, source_type, maximum_impact) in expected.items():
        profile = resolve_news_source_profile(source)
        assert profile.name == name
        assert profile.source_type == source_type
        assert profile.market_impact_weight <= maximum_impact
        assert profile.credibility_score < 0.70


def test_load_news_source_profiles_from_yaml(tmp_path):
    path = tmp_path / "news_sources.yaml"
    path.write_text(
        """
sources:
  - name: Test Wire
    aliases: ["test wire"]
    credibility_score: 0.81
    market_impact_weight: 0.72
    source_type: wire
""",
        encoding="utf-8",
    )

    profiles = load_news_source_profiles(path)

    assert len(profiles) == 1
    assert profiles[0].name == "Test Wire"
    assert profiles[0].source_type == "wire"


def test_importance_reason_includes_source_impact():
    result = score_news_importance_with_reason(
        "Toyota earnings guidance China EV update",
        ticker="7203.T",
        title="Toyota earnings guidance China EV update",
        source="Reuters",
    )

    assert result.score >= 0.8
    assert "source impact" in result.reason
    assert "major source" in result.reason
