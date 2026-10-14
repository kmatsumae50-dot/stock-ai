# stock-ai

Japanese stock analysis and daily prediction operations for Toyota Motor (`7203.T`).

Official working folder:

```powershell
C:\Users\kmats\Documents\ai-stock\stock-ai
```

Do not operate from the legacy folder under `C:\Users\kmats\Documents\ai 株` plus `\stock-ai`.
That path is mentioned only as a caution, not as a valid project root.

## Current Operation Snapshot

As of 2026-09-04, production operation is intentionally conservative:

- Production direction judgement mainly reads the binary model.
- The binary model is materially underperforming simple baselines in the latest reviewed live diagnostics, so it must not be used alone for investment decisions.
- The production binary model, multiclass model, `daily_predict.ps1`, Task Scheduler, SQLite history, and feature generation are not changed.
- Multiclass is reference/experiment only because the current 5-class model has strong flat bias.
- Known missed live dates are documented and must not be recreated as live predictions.
- Binary shadow output is manual/opt-in only. It has no automatic scheduled generation.
- Append-only tables are audit support only. The existing `predictions` snapshot remains the health and evaluation source.

Detailed rules are maintained in:

- [docs/current_user_manual.md](docs/current_user_manual.md)
- [docs/environment_handoff.md](docs/environment_handoff.md)
- [reports/evaluation/model_decision_notes_2026-06-24.md](reports/evaluation/model_decision_notes_2026-06-24.md)

## Daily Operation

Use the official folder and its `.venv` for all commands:

```powershell
cd "C:\Users\kmats\Documents\ai-stock\stock-ai"
```

Start the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\stock_ai\app\main.py
```

Current scheduled task:

- Task name: `stock-ai daily predict`
- Time: `10:30`
- Working directory: `C:\Users\kmats\Documents\ai-stock\stock-ai`
- Daily command: `.\scripts\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3`

Manual daily run, only when needed:

```powershell
.\scripts\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3
```

Health-check:

```powershell
.\.venv\Scripts\python.exe main.py health-check --ticker 7203.T
```

## Health-Check Reading

Current expected state:

- `Overall WARN`
- `Action required: 0`
- Known missed sections: `1`
- Known missed dates: `8`
- Known OK: `1`
- `unknown_source_count: 0`
- `check-news: OK`

`Overall WARN` is acceptable only when `Action required` is `0` and the WARN items are known:

- Known missed live dates: see `operations.known_missed_live_dates` in `configs/settings.yml` and the current operation manual.
- Known OK: TOPIX proxy / fallback, currently using `1306.T` when the primary TOPIX symbol is unavailable.

If `Action required` appears, restore operational health before reading model output.

## Prediction Date Semantics

In the existing SQLite snapshot, `predictions.date` is effectively the Feature date, not the save timestamp or target close date.

- Feature date: the market data date used by the model.
- Target close date: the next trading day's close after the Feature date.
- Saved at: the SQLite write timestamp.

For scheduled tasks, also distinguish `scheduled_for` from `actual_started_at`. A catch-up run started after 10:30 is not a prediction made at 10:30.

## Production Guardrails

Do not casually change these while 10:30 operation and offline experiments are still being monitored:

- production model config
- binary production model
- multiclass production model
- `daily_predict.ps1`
- Task Scheduler time
- SQLite schema
- prediction history
- feature generation
- `NewsLimit 3`
- daily 4 profile configuration
- `known_missed_live_dates`
- news source profiles
- earnings CSV files

This repository is an analysis and validation tool. It is not investment advice.
