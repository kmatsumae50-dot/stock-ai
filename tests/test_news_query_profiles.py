from stock_ai.news.query_profiles import (
    daily_news_query_profiles,
    get_news_query_profile,
    grouped_news_query_profile_rows,
    load_news_query_profiles,
    news_query_profile_rows,
)


def test_load_news_query_profiles_from_yaml(tmp_path):
    config_path = tmp_path / "news_queries.yaml"
    config_path.write_text(
        """
profiles:
  - name: toyota_stock
    query: "Toyota stock"
    ticker: "7203.T"
    category: "auto"
    enabled_for_daily: true
    group: "core"
  - name: auto_industry
    query: "Toyota Honda Nissan Tesla BYD auto industry"
    ticker: ""
    category: "auto"
    enabled_for_daily: false
    group: "sector"
  - name: toyota_jp
    query: "トヨタ 株価 ニュース"
    ticker: "7203.T"
    category: "auto"
    language: "ja"
    region: "JP"
    enabled_for_daily: true
    group: "jp"
""",
        encoding="utf-8",
    )

    profiles = load_news_query_profiles(config_path)
    daily_profiles = daily_news_query_profiles(config_path)
    group_rows = grouped_news_query_profile_rows(config_path)

    assert [profile.name for profile in profiles] == ["toyota_stock", "auto_industry", "toyota_jp"]
    assert profiles[0].storage_ticker() == "7203.T"
    assert profiles[0].enabled_for_daily is True
    assert profiles[0].group == "core"
    assert profiles[1].row_ticker() == ""
    assert profiles[1].storage_ticker("7203.T") == "7203.T"
    assert profiles[1].enabled_for_daily is False
    assert profiles[2].language == "ja"
    assert profiles[2].region == "JP"
    assert [profile.name for profile in daily_profiles] == ["toyota_stock", "toyota_jp"]
    assert {row["group"] for row in group_rows} == {"core", "sector", "jp"}


def test_get_news_query_profile_and_rows(tmp_path):
    config_path = tmp_path / "news_queries.yaml"
    config_path.write_text(
        """
profiles:
  - name: macro_auto
    query: "USDJPY tariffs interest rates auto stocks Toyota"
    ticker: ""
    category: "macro"
    enabled_for_daily: false
    group: "macro"
""",
        encoding="utf-8",
    )

    profile = get_news_query_profile("macro_auto", config_path)
    rows = news_query_profile_rows(config_path)

    assert profile.query.startswith("USDJPY")
    assert rows == [
        {
            "name": "macro_auto",
            "query": "USDJPY tariffs interest rates auto stocks Toyota",
            "ticker": "",
            "storage_ticker": "7203.T",
            "category": "macro",
            "language": "",
            "region": "",
            "enabled_for_daily": "False",
            "group": "macro",
        }
    ]
