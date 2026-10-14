# Production Prediction Integrity Audit - 2026-09-02

## Scope and safety

- Repository: `C:\Users\kmats\Documents\ai-stock\stock-ai`
- Audit only: no production model, threshold, prediction history, scheduler, agent, or known-missed configuration change.
- No retraining, live/backfill creation, shadow generation, worktree, commit, merge, or push.
- The only configuration change is the reviewed addition of five news-source profiles. News scores and feature parquet were then regenerated as requested.

## Executive conclusion

- Primary diagnosis: `model_staleness_or_regime_failure_likely` (D).
- Secondary diagnosis: `model_discrimination_failure_confirmed` (E).
- No train/predict/save/evaluate polarity mismatch or off-by-one error was found.
- The current artifact was rebuilt on 2026-09-02, but its actual fit partition ends on 2021-08-23. The fixed chronological 70/15/15 split therefore keeps recent regimes out of model fitting.
- Recent production output has collapsed below the 0.50 Up threshold: all confirmed predictions since 2026-08-01 are Down, while actual Up rates remain near one half or higher.
- Inverting probabilities improves all-live ROC AUC from 0.3571 to 0.6429. This is a diagnostic signal, not permission to invert production output.
- `recent_regime_robustness` is the appropriate next research direction, specifically testing training-window recency and regime robustness without changing production in the same cycle.

## 1. Unknown source resolution

All five rows were delivered through Google News RSS, occur once locally, and have no duplicate title or URL in the CSV.

| Source | Seen | Date | Profile / category | Directness and content | Classification |
|---|---:|---|---|---|---|
| CryptoRank | 1 | 2026-08-19 | `toyota_stock` / `recall_safety` | Direct Toyota recall story; crypto-data publisher/aggregation-oriented platform, not primary Toyota disclosure | `CryptoRank`, aliases `cryptorank`, `cryptorank.io`; `general_news`; credibility 0.45; impact 0.22 |
| NHKニュース | 1 | 2026-08-04 | `toyota_earnings_jp` / `earnings` | Direct Toyota earnings outlook report; public-broadcast news | `NHK News`, aliases `NHKニュース`, `nhk news`, `nhk.or.jp`; `general_news`; credibility 0.82; impact 0.38 |
| Visual Capitalist | 1 | 2025-12-06 | `auto_industry` / `other` | Indirect global car-brand ranking; data-visualization publisher | `Visual Capitalist`, aliases `visualcapitalist`, `visualcapitalist.com`; `general_news`; credibility 0.62; impact 0.25 |
| みんかぶ | 1 | 2026-08-24 | `toyota_jp` / `other` | Direct Toyota share-price/analyst target story; Japanese investment-information medium | `Minkabu`, aliases `みんかぶ`, `minkabu`, `minkabu.jp`; `financial_media`; credibility 0.68; impact 0.46 |
| ニュースイッチ by 日刊工業新聞社 | 1 | 2026-08-19 | `toyota_earnings_jp` / `earnings` | Direct/sector earnings outlook; Nikkan Kogyo industrial-news site, exact item may be original or curated | `NewsSwitch by Nikkan Kogyo Shimbun`, aliases exact display name, `ニュースイッチ`, `newswitch`, `newswitch.jp`; `general_news`; credibility 0.70; impact 0.38 |

Representative titles are respectively: “Toyota Stock Falls as 508,000 Camry Hybrids Recalled Over Safety Defect”; “トヨタ 今年度の業績見通しを上方修正 営業利益は3兆4000億円”; “Ranked: The World's Best Selling Car Brands”; “トヨタは小幅に3日続伸、国内大手証券は目標株価3700円に引き上げ”; and “トヨタ・ホンダ上方、スズキは下方…完成車各社の通期見通し”.

Media-characterization references: [CryptoRank terms](https://cryptorank.io/terms-and-conditions), [NHK's public-welfare role in the Broadcasting Act](https://www.japaneselawtranslation.go.jp/en/laws/view/2954/en), [Visual Capitalist about](https://www.visualcapitalist.com/about/), [Minkabu](https://minkabu.jp/), and [Nikkan Kogyo media business](https://corp.nikkan.co.jp/p/media/index).

After rescoring: 256 news rows, `check-news: OK`, duplicates 0, mojibake suspects 0, and `unknown_source_count: 0`. Source-related Action required is resolved.

## 2. 2026-08-24 missing-live review

Conclusion: `task_scheduler_not_run`.

- The scheduled task time was 10:30.
- Windows System events show shutdown at 2026-08-24 00:39 and the next boot at 2026-08-25 01:58. The machine was therefore off at the scheduled run.
- A catch-up started at 2026-08-25 02:04 but stopped after Python initialization when the PC shut down at 02:05. No generation failure was logged because prediction generation was never reached.
- The successful 18:57 run fetched current data and saved feature date 2026-08-25, not 2026-08-24.
- Raw and feature rows for 2026-08-24 now exist, but neither binary nor multiclass prediction exists for that feature date.
- The 2026-08-21 live prediction targeting the 2026-08-24 close does exist. The missing item is a live prediction *using 2026-08-24 close data* to target the next trading close.

Human-reviewable configuration candidate, not applied:

```yaml
    - date: "2026-08-24"
      ticker: "7203.T"
      reason_code: "task_scheduler_not_run"
      reason: "Windows System logs show the PC shut down at 2026-08-24 00:39 and did not start again until 2026-08-25 01:58, so the 2026-08-24 10:30 scheduled daily_predict run could not occur. The 02:04 catch-up ended when the PC shut down at 02:05 before prediction generation."
      action: "do_not_create_live_prediction"
      allow_backfill: true
```

No live or backfill row was created.

## 3. Dirty main review

No unrelated or unexplained code change was found. Nothing was staged or committed.

| Group | Files | Classification | Suggested handling |
|---|---|---|---|
| News source profiles | `configs/news_sources.yaml`, `tests/test_news_sources.py` | C, F | Review and commit together. Includes prior TechStock2 work plus the five profiles in this audit. |
| Autonomous stale-lock/research guard maintenance | `scripts/run_autonomous_operations.ps1`, `scripts/run_autonomous_research.ps1`, `scripts/autonomous_lock_helpers.ps1`, scheduler/lock tests | A, F | Separate maintenance commit after human review. |
| Shadow lifecycle/status maintenance | `src/stock_ai/cli.py`, `src/stock_ai/evaluation/binary_shadow.py`, `tests/test_binary_shadow.py` | A, F | Separate shadow-maintenance commit after human review. |
| Recovery documentation | `reports/autonomous/post_unattended_audit_recovery_2026-08-19.md` | A, F (optional docs) | Commit separately or with stale-lock maintenance. |
| Generated shadow summary | `reports/evaluation/binary_shadow_summary_2026-08-19.md` | B, G | Runtime/evaluation output; do not commit unless reports are intentionally versioned. |

Recommended commit order: (1) source profiles/tests, (2) stale-lock and scheduler guard maintenance/tests, (3) shadow CLI/lifecycle/tests, (4) optional recovery documentation. Do not include generated state, runtime locks, DB files, or the generated shadow summary.

## 4. Target, class, and probability semantics

| Stage | Actual implementation | Result |
|---|---|---|
| Target creation | `next_close_return = next available close / current close - 1`; `> 0` => 1, otherwise 0; final row => -1 | Up=1, flat/down=0 |
| Training | LightGBM binary classifier uses `target_next_close_binary` | Consistent |
| Artifact | Actual `model.classes_ == [0, 1]`; `predict_proba` shape is `(n, 2)` | column 0=Down, column 1=Up |
| Live prediction | `probability_down = proba[:,0]`; `probability_up = proba[:,1]`; `p_up >= 0.5` => Up | Consistent |
| Confidence | `max(probability_down, probability_up)` | Consistent |
| SQLite | Saves predicted class and both probabilities unchanged | Consistent |
| Live evaluation | Next observed trading row; `actual_return > 0` => 1 | Consistent |
| Offline evaluation | Uses column 1 for ROC AUC and the same target | Consistent |
| Shadow | Uses `predict_proba[:,1]` and return >0 as Up | Consistent |
| Autonomous Research | Target validator recomputes `(next_close_return > 0)` and reports mismatch | Consistent |
| UI | Binary labels map 0=Down, 1=Up | Consistent |

The binary prediction helper hard-codes probability positions rather than looking up `classes_`; this is a future robustness opportunity, but the production artifact's observed `[0,1]` ordering makes the current path correct.

Training and evaluation use the next observed market row, so weekends and holidays do not create an off-by-one target. Some display-only helpers use business-day inference and can be holiday-naive, but that did not affect the evaluated labels or the ten inspected samples.

## 5. Current versus inverted diagnostic

| Period | n | Current AUC | Inverted AUC | Current accuracy / BA | Inverted accuracy / BA | Current Up/Down recall | Inverted Up/Down recall |
|---|---:|---:|---:|---|---|---|---|
| All confirmed | 60 | 0.3571 | 0.6429 | 0.4167 / 0.4043 | 0.5833 / 0.5957 | 0.0345 / 0.7742 | 0.9655 / 0.2258 |
| Since 2026-08-01 | 17 | 0.3194 | 0.6806 | 0.4706 / 0.5000 | 0.5294 / 0.5000 | 0 / 1 | 1 / 0 |
| Since 2026-08-20 | 8 | 0.1333 | 0.8667 | 0.3750 / 0.5000 | 0.6250 / 0.5000 | 0 / 1 | 1 / 0 |
| Latest 20 | 20 | 0.4343 | 0.5657 | 0.5500 / 0.5000 | 0.4500 / 0.5000 | 0 / 1 | 1 / 0 |
| Latest 10 | 10 | 0.2500 | 0.7500 | 0.4000 / 0.5000 | 0.6000 / 0.5000 | 0 / 1 | 1 / 0 |

The inversion result is not evidence of a code-level class swap: the latest artifact calculation exactly matches SQLite (`p_up=0.4778548634`, `p_down=0.5221451366`, predicted Down, absolute difference 0). It instead shows that recent ranking is anti-correlated on a small sample.

## 6. Probability distribution

| Period | n | mean / median | min / max / std | <0.4 | 0.4-0.5 | 0.5-0.6 | >0.6 | Pred Up | Actual Up |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|
| All confirmed | 60 | 0.4784 / 0.4796 | 0.4065 / 0.5448 / 0.0299 | 0 | 52 | 8 | 0 | 13.3% | 48.3% |
| Since 2026-08-01 | 17 | 0.4697 / 0.4795 | 0.4065 / 0.4919 / 0.0236 | 0 | 17 | 0 | 0 | 0% | 52.9% |
| Since 2026-08-20 | 8 | 0.4546 / 0.4617 | 0.4065 / 0.4836 / 0.0268 | 0 | 8 | 0 | 0 | 0% | 62.5% |
| Latest 20 | 20 | 0.4661 / 0.4776 | 0.4065 / 0.4919 / 0.0235 | 0 | 20 | 0 | 0 | 0% | 45.0% |
| Latest 10 | 10 | 0.4607 / 0.4698 | 0.4065 / 0.4905 / 0.0270 | 0 | 10 | 0 | 0 | 0% | 60.0% |

This is a narrow probability distribution shifted below 0.50, not a numerical collapse to a constant. The fixed threshold turns that moderate shift into an all-Down decision stream.

## 7. Feature and preprocessing integrity

- Artifact expects 163 features; all 163 are present, numeric, and in the exact stored/model order.
- `model.feature_name_` exactly equals the artifact feature list.
- Missing expected columns: 0. Extra dataset columns: 15, all ignored by explicit feature selection.
- Latest row missing expected values: 0. Latest row infinities: 0. No categorical feature configuration is used.
- No scaler or separate transform exists; LightGBM receives the selected numeric frame directly.
- Prediction drops a latest row if any expected feature is NaN. The current row passes, and freshness is OK with lag 0.
- Thirty-six features are constant in the ten-row recent window, mostly sparse earnings/news/narrative flags. That is not itself corruption, but it reduces available short-window variation.
- Drift estimates use only ten recent rows and are descriptive. Raw level features such as Toyota close/Bollinger levels and several external closes are outside the 2010-2021 fit range. Among nonzero-importance features, Nissan MA25 and Honda MA25 are outside the fit range; normalized indicators also show distribution shifts.

## 8. Date alignment samples

Ten consecutive confirmed samples from 2026-08-18 through 2026-09-01 were checked. Each prediction's feature date equals the source OHLC row; target close date is the next observed row; actual label equals the sign of that return. The sample includes four correct Down predictions, six incorrect Down predictions, six actual Up days, four actual Down days, and eight rows from 2026-08-20 onward. No off-by-one or holiday mismatch was found.

Examples: 2026-09-01 close 3243 -> 2026-09-02 close 3144, actual Down, predicted Down, `p_up=0.4585`; 2026-08-31 close 3156 -> 2026-09-01 close 3243, actual Up, predicted Down, `p_up=0.4649`; 2026-08-20 close 3066 -> 2026-08-21 close 3132, actual Up, predicted Down, `p_up=0.4065`.

## 9. Artifact freshness and regime drift

- Artifact modified/trained: 2026-09-02 10:30 JST; classes `[0,1]`; 163 features; best iteration 3.
- Fit partition: 2010-01-04 through 2021-08-23.
- Validation partition: 2021-08-24 through 2024-02-27.
- Test partition: 2024-02-28 through 2026-09-02.
- Artifact test metadata: accuracy 0.5130, F1 0.1075, AUC 0.5445 on 614 rows.
- The artifact is operationally fresh but its fitted observations are not recent. Daily retraining with a fixed fractional split only moves the old training boundary slowly.
- No config fingerprint or explicit split dates are embedded in artifact metadata. Current settings SHA-256 is recorded in the safety section; this traceability gap is not the cause of the polarity result.
- Training Up base rate is 48.1%; 2026-08-20 onward is 55.6% over nine confirmed returns.
- Existing regime rules classify the ten recent feature rows as six High Volatility, three Panic, and one Bull Trend, versus a historically mixed Bull/Bear/Sideways distribution.

These observations support stale-fit/regime mismatch. PSI magnitudes are intentionally not treated as inferential evidence because the recent sample has only ten rows.

## 10. Baseline integrity

The previous-day-direction baseline uses the return ending on the same feature date to forecast the next-row target. It uses the identical 60 confirmed dates, has no future value in its input, and maps through the same observed trading calendar. Accuracy is 0.5333, balanced accuracy 0.5328, Up recall 0.5172, and Down recall 0.5484. It is not receiving a more favorable target alignment.

## 11. Shadow and autonomous research

- Shadow remains manual-only: total 2, confirmed 2, hits 2, misses 0, no SQLite save, and no new shadow prediction generated.
- Agent preflight is unhealthy only because main is dirty; `due=false`, Codex invocation false, worktree creation false. No heavy cycle or research worktree was started.

## 12. Safety verification

Unchanged before/after:

- Production DB SHA-256: `5401D4A578534F999AC58821E83906ADA8885D85AEB029D9ADA6D10EB4F52196`
- DB counts: predictions 127; prediction_runs 1; prediction_records 2.
- Binary artifact: `BD2E27B6E7A2B196DE2E3B3EA2C5D35D7A76736FCA18D74D878EAAC8CD492F31`
- Multiclass artifact: `6901C5169BF8241B17D1CF663AD454C3CE290C1BBA624B53F1DA9B0696C4AB3B`
- Settings: `A3EE5F158C88DEA3F76E71C1973865CE6B32AA64E579C522702C373EF82ACCBE`
- News queries: `0C6805866806B0AD23ED21A1E8C575FB92D7E82E85C91B8D4506077A597E3682`
- daily_predict: `83F25744BC1409033EDF74E37C506BAF84BD6E2BD0315CF463407B74B659CBA2`
- Scheduler XML: daily `FD7EF4...F7DD7`; operations `45A078...F81A3`; research `CF96AA...92AE4B`.

Intentionally changed: `configs/news_sources.yaml` from `FB4CF0...D8944` to `F4CB11...90772`, plus regenerated news scores, fetch-history quality fields, and feature parquet.

## 13. Verification

- Source tests: 13 passed.
- Full pytest: 434 passed.
- Ruff: OK.
- Compileall: OK.
- Streamlit: HTTP 200 on port 8501, then stopped after verification.
- Health: `unknown_source_count=0`, `check-news=OK`, freshness OK. Action required remains 1 only for unreviewed 2026-08-24 missing feature-date prediction; TOPIX fallback remains Known OK.

## Recommendation

Do not invert probabilities or change the production threshold. After human review of the 2026-08-24 known-missed candidate and cleanup/commit of intended main changes, the next single research hypothesis should be `recent_regime_robustness`: compare time-aware rolling or expanding fit windows that include recent observations, preserve leakage controls, and evaluate stability across historical and recent periods. Production remains unchanged until such an offline result clears explicit guardrails.
