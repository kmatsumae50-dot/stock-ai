from stock_ai.evaluation.model_decision_notes import (
    build_model_decision_notes_summary,
    known_missed_live_date_details,
    model_decision_notes_markdown,
    write_model_decision_notes_report,
)


def _health_report():
    return {
        "overall_status": "WARN",
        "warning_summary": {
            "action_required_count": 0,
            "known_missed_date_count": 2,
            "known_ok_count": 1,
            "known_missed": [
                {
                    "details": {
                        "known_missed_live_dates": [
                            {
                                "missing_date": "2026-06-15",
                                "known_reason_code": "data_not_available_at_run_time",
                                "known_reason": "Data was not available at run time.",
                                "known_action": "do_not_create_live_prediction",
                                "known_allow_backfill": True,
                            },
                            {
                                "missing_date": "2026-06-23",
                                "known_reason_code": "data_not_available_at_run_time",
                                "known_reason": "Raw and feature data lagged at run time.",
                                "known_action": "do_not_create_live_prediction",
                                "known_allow_backfill": True,
                            },
                        ]
                    }
                }
            ],
        },
    }


def test_known_missed_live_date_details_flattens_dates():
    rows = known_missed_live_date_details(_health_report()["warning_summary"])

    assert [row["feature_date"] for row in rows] == ["2026-06-15", "2026-06-23"]
    assert rows[0]["reason_code"] == "data_not_available_at_run_time"
    assert rows[0]["allow_backfill"] is True


def test_model_decision_notes_summary_keeps_current_production_decision():
    summary = build_model_decision_notes_summary(
        health_report=_health_report(),
        report_date="2026-06-24",
    )

    assert summary["operational_health"]["action_required_count"] == 0
    assert summary["operational_health"]["known_missed_date_count"] == 2
    assert summary["binary_evaluation"]["role"] == "primary direction reference"
    assert summary["multiclass_evaluation"]["role"] == "experimental/reference only"
    assert summary["production_decision"]["binary_model_changed"] is False
    assert summary["production_decision"]["sqlite_history_changed"] is False


def test_model_decision_notes_markdown_contains_required_sections(tmp_path):
    summary = build_model_decision_notes_summary(
        health_report=_health_report(),
        report_date="2026-06-24",
    )
    markdown = model_decision_notes_markdown(summary)

    assert "Current operational health" in markdown
    assert "Known missed live dates" in markdown
    assert "2026-06-23" in markdown
    assert "Binary evaluation summary" in markdown
    assert "Multiclass flat bias summary" in markdown
    assert "no production model change has been applied" in markdown

    path = write_model_decision_notes_report(summary, tmp_path)
    assert path.name == "model_decision_notes_2026-06-24.md"
    assert "10:30 scheduler observation" in path.read_text(encoding="utf-8")
