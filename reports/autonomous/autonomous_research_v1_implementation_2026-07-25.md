# Autonomous Research v1 Implementation

## Outcome

Autonomous Research v1 is implemented as a gated, offline-only research workflow. It reads the latest healthy Autonomous Operations cycle, runs at most three existing binary walk-forward candidates, compares production, baselines, live history, and read-only shadow history, and writes state, append-only history, Markdown/JSON reports, and deduplicated human-review tasks.

No production change was applied.

## Added

- CLI: `python main.py autonomous-research --ticker 7203.T`
- Module: `src/stock_ai/operations/autonomous_research.py`
- Wrapper: `scripts/run_autonomous_research.ps1`
- Tests: `tests/test_autonomous_research.py`
- Documentation: `docs/autonomous_research.md`
- State: `reports/autonomous/research_state.json`
- History: `reports/autonomous/research_history.jsonl`
- Reports: `reports/autonomous/research_YYYY-MM-DD_HHMM.md` and `.json`
- Wrapper logs: `logs/autonomous/research_YYYY-MM-DD_HHMM.log`

The wrapper was not registered in Task Scheduler.

## First Heavy Cycle

- Report: `reports/autonomous/research_2026-07-25_1216.md`
- Fingerprint: `38a44da802307be5a4b14998848e428974954ec77027e7906b4dd86f0f91119a`
- Trigger: `first_run`
- Candidates: `production_current`, `class_weight_balanced`, `scale_pos_weight_medium`
- Classification: `research_candidate_requires_more_evidence`
- Recommendation: `production_change_not_ready; continue evidence collection`
- Heavy runtime: approximately one minute
- Live confirmed reference: 37
- Shadow confirmed after read-only in-memory evaluation: 2

Ranking:

| Rank | Candidate | Score | Historical balanced accuracy | Up recall | Down recall | Decision |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | scale_pos_weight_medium | 0.4072 | 0.5121 | 0.4422 | 0.5820 | no_material_change |
| 2 | class_weight_balanced | 0.4030 | 0.5201 | 0.4712 | 0.5689 | promising_historical_only |
| 3 | production_current | 0.3897 | 0.5074 | 0.3500 | 0.6647 | production_still_stronger |

`class_weight_balanced` improved balanced accuracy and up recall historically, but down recall deteriorated and confirmed shadow evidence was below the ten-row minimum. No proposal-ready candidate exists.

## Duplicate And Force Verification

- A second normal run used the same fingerprint and returned `research_skipped_duplicate_experiment` without running the heavy experiment.
- The final duplicate verification returned exit code 0 and created no duplicate task.
- `--force --skip-offline` verified the force/orchestration path without repeating the heavy experiment.
- `scripts/run_autonomous_research.ps1` completed with exit code 0 on the duplicate-skip path and removed its lock.
- The wrapper checks the latest date-stamped Operations report, uses the formal `.venv`, fixes the working directory, logs stdout/stderr, enforces a 120-minute maximum, and passes through exit codes.

During implementation validation, a report-discovery bug was found: `operations_*.json` also matched `operations_task_queue.json`. The Research loader and wrapper now accept only date-stamped Operations reports. The failed gate observation remains in append-only Research history for auditability.

## Task Queue

The first heavy cycle initially detected the already-known weak live model result. The existing Operations task `model_up_recall_low:7203.T` already covers that review, so Research now detects it and does not create a duplicate degradation task. The temporary duplicate Research task was moved to resolved/monitoring history; the existing Operations approval task remains the active review item.

No candidate proposal task was created because no candidate passed the production proposal gate.

## Safety Invariants

Before and after hashes/counts matched:

- Production DB SHA-256: `7CFA733FD5EFF9FEAF234C8E2E5AE4E67AFB5F097466E9BA0D61ED3ACC1F25AA`
- `predictions`: 81 -> 81
- `prediction_runs`: 1 -> 1
- `prediction_records`: 2 -> 2
- Binary artifact SHA-256: `EA947A34A711F560236F89411EA25B58E1F58337A6F0BBF497E9ED2B10D626FB`
- Multiclass artifact SHA-256: `CBA07F0A648884876EDF65CBA691EE54987B22A48E66AC138B91306AB02F300D`
- `configs/settings.yml`: unchanged
- `configs/news_queries.yaml`: unchanged
- `configs/news_sources.yaml`: unchanged
- `scripts/daily_predict.ps1`: unchanged
- Autonomous Operations module/wrapper: unchanged
- Daily predict Scheduler XML: unchanged
- Autonomous Operations Scheduler XML: unchanged

Explicitly:

- Production model changed: No
- Production artifact changed: No
- Production config changed: No
- Production SQLite changed: No
- Existing predictions changed: No
- Live/backfill prediction created: No
- `daily_predict.ps1` changed: No
- Task Scheduler changed: No
- Autonomous Operations changed: No
- Automatic merge performed: No

## Verification

- Research tests: 22 passed
- Full pytest: 371 passed
- Ruff: OK
- Compileall: OK
- Health-check: Overall WARN, Action required 0, Known missed dates 4, Known OK 1
- Unknown sources: 0
- News check: OK
- Autonomous Operations: `completed_read_only_cycle`, exit 0
- Streamlit: HTTP 200 on port 8501
- Research wrapper: exit 0, lock removed

## Next Human Actions

1. Keep production unchanged.
2. Continue collecting confirmed live and shadow observations.
3. Re-run Research after five new confirmations, seven days, or a material live metric change.
4. Review `class_weight_balanced` only after more shadow support and a closer look at its down-recall loss.
5. Do not register Autonomous Research in Task Scheduler until a separate deployment review explicitly approves it.
