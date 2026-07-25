# stock-ai

Japanese stock analysis and daily prediction operations for Toyota Motor (`7203.T`).

Official working folder:

```powershell
C:\Users\kmats\Documents\ai-stock\stock-ai
```

Do not operate from the legacy folder under `C:\Users\kmats\Documents\ai 株` plus `\stock-ai`.
That path is mentioned only as a caution, not as a valid project root.

## Current Operation Snapshot

As of 2026-06-24, production operation is intentionally conservative:

- Production direction judgement mainly reads the binary model.
- The binary edge over baseline is still weak, so it must not be used alone for investment decisions.
- The production binary model, multiclass model, `daily_predict.ps1`, Task Scheduler, SQLite history, and feature generation are not changed.
- Multiclass is reference/experiment only because the current 5-class model has strong flat bias.
- Known missed live dates are documented and must not be recreated as live predictions.

Detailed rules are maintained in:

- [docs/current_user_manual.md](docs/current_user_manual.md)
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
- Known missed dates: `2`
- Known OK: `1`
- `unknown_source_count: 0`
- `check-news: OK`

`Overall WARN` is acceptable only when `Action required` is `0` and the WARN items are known:

- Known missed live dates: `2026-06-15`, `2026-06-23`
- Known OK: TOPIX fallback

If `Action required` appears, restore operational health before reading model output.

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
