import pandas as pd

from stock_ai.labels.distribution import (
    analyze_label_distribution,
    compare_label_schemes,
    distribution_by_group,
    fixed_label_scheme,
    get_threshold_profile,
    multiclass_artifact_stem,
    multiclass_class_weight,
    multiclass_model_name,
    profile_from_model_name,
    quantile_label_scheme,
    threshold_profile_options,
    weight_mode_from_model_name,
    weight_mode_options,
)


def test_analyze_label_distribution_counts_all_classes():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "next_close_return": [-0.06, -0.02, 0.0, 0.02, 0.06],
            "regime": ["Panic", "Bear Trend", "Sideways", "Bull Trend", "Bull Trend"],
        }
    )
    scheme = fixed_label_scheme(
        key="test",
        name="test",
        surge_threshold=0.05,
        up_threshold=0.01,
        down_threshold=-0.01,
        crash_threshold=-0.05,
    )

    result = analyze_label_distribution(data, scheme)
    distribution = result["distribution"].set_index("class")

    assert distribution.loc["crash", "count"] == 1
    assert distribution.loc["down", "count"] == 1
    assert distribution.loc["flat", "count"] == 1
    assert distribution.loc["up", "count"] == 1
    assert distribution.loc["surge", "count"] == 1
    assert result["summary"]["trainability"] == "Weak"


def test_distribution_by_group_returns_counts_and_shares():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "next_close_return": [-0.06, 0.0, 0.02, 0.06],
            "regime": ["A", "A", "B", "B"],
        }
    )
    scheme = fixed_label_scheme(
        key="test",
        name="test",
        surge_threshold=0.05,
        up_threshold=0.01,
        down_threshold=-0.01,
        crash_threshold=-0.05,
    )
    result = analyze_label_distribution(data, scheme)

    grouped = distribution_by_group(result["labeled_data"], "regime")

    row_a = grouped[grouped["regime"] == "A"].iloc[0]
    row_b = grouped[grouped["regime"] == "B"].iloc[0]
    assert row_a["crash"] == 1
    assert row_a["flat"] == 1
    assert row_a["crash_share"] == 0.5
    assert row_b["up"] == 1
    assert row_b["surge"] == 1


def test_compare_label_schemes_includes_quantile_candidate():
    data = pd.DataFrame({"next_close_return": [-0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05]})

    comparison = compare_label_schemes(data, schemes=[quantile_label_scheme()])

    assert comparison.loc[0, "scheme"] == "quantile"
    assert comparison.loc[0, "total"] == 7
    assert comparison.loc[0, "min_class_count"] >= 1


def test_threshold_profiles_preserve_original_model_name():
    assert threshold_profile_options() == ["original_5_1", "practical_3_0_5", "quantile"]
    assert weight_mode_options() == ["none", "balanced", "manual_tail_boost"]
    assert get_threshold_profile("fixed_5_1").key == "original_5_1"
    assert multiclass_model_name("original_5_1") == "lightgbm_multiclass"
    assert multiclass_model_name("practical_3_0_5") == "lightgbm_multiclass_practical_3_0_5"
    assert multiclass_artifact_stem("7203.T", "original_5_1") == "lightgbm_multiclass_7203.T"
    assert (
        multiclass_artifact_stem("7203.T", "practical_3_0_5")
        == "lightgbm_multiclass_practical_3_0_5_7203.T"
    )


def test_multiclass_weight_modes_extend_model_name_without_breaking_default():
    assert multiclass_model_name("original_5_1", "none") == "lightgbm_multiclass"
    assert (
        multiclass_model_name("practical_3_0_5", "balanced")
        == "lightgbm_multiclass_practical_3_0_5_balanced"
    )
    assert (
        multiclass_model_name("practical_3_0_5", "manual_tail_boost")
        == "lightgbm_multiclass_practical_3_0_5_tailboost"
    )
    assert (
        multiclass_artifact_stem("7203.T", "practical_3_0_5", "balanced")
        == "lightgbm_multiclass_practical_3_0_5_balanced_7203.T"
    )
    assert multiclass_class_weight("none") is None
    assert multiclass_class_weight("balanced") == "balanced"
    assert multiclass_class_weight("manual_tail_boost")[0] == 3.0
    assert (
        profile_from_model_name("lightgbm_multiclass_practical_3_0_5_balanced")
        == "practical_3_0_5"
    )
    assert weight_mode_from_model_name("lightgbm_multiclass_practical_3_0_5_tailboost") == "manual_tail_boost"
