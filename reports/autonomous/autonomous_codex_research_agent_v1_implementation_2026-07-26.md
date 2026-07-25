# Autonomous Codex Research Agent v1 Implementation

## Outcome

The bounded Agent infrastructure is implemented and manually exercised. The first cycle stopped safely before hypothesis selection and worktree creation because the formal project repository has no Git `HEAD` and all project files are untracked. The installed Desktop Codex executable was found but could not be executed from PowerShell (`codex_help_failed`, WinError 5 Access denied). No CLI flags were guessed.

This is the required safe behavior for the current environment. No actual hypothesis implementation or offline experiment was run because doing so would require editing main or fabricating a non-Git worktree.

## Added Components

- Agent module: `src/stock_ai/operations/autonomous_codex_agent.py`
- CLI: `python main.py autonomous-codex-agent`
- Wrapper: `scripts/run_autonomous_codex_research_agent.ps1`
- Prompt contract: `prompts/autonomous_codex_research_agent.md`
- Tests: `tests/test_autonomous_codex_agent.py`
- Documentation: `docs/autonomous_codex_research_agent.md`
- Backlog: `reports/autonomous/codex_research_hypotheses.json`
- State: `reports/autonomous/codex_research_agent_state.json`
- Cycle reports: `reports/autonomous/codex_agent_YYYY-MM-DD_HHMM.md` and `.json`

No Agent Task Scheduler task was created.

## Gate

The gate reads the latest Operations and Autonomous Research JSON reports, production guard, SQLite integrity/counts, shared task queue, and Git state. It verifies:

- Operations `completed_read_only_cycle`
- Action required 0, unknown source 0, no unresolved missing-live finding
- fresh raw/features and readable production SQLite
- current Operations and Research report timestamps
- current Research classification, fingerprint, candidate ranking, and open `model_research` tasks
- production settings hash matches the latest Research guard
- valid main Git `HEAD` and a clean main worktree
- no concurrent Agent lock

The final manual cycle used fresh reports:

- Operations: `operations_2026-07-26_0020.json`
- Research: `research_2026-07-26_0021.json`
- Research fingerprint: `38a44da802307be5a4b14998848e428974954ec77027e7906b4dd86f0f91119a`

Final gate reasons were only:

- `main_git_head_missing`
- `main_worktree_dirty`

## Hypothesis Backlog

The deterministic initial backlog contains seven unique proposed hypotheses and no duplicate fingerprints:

1. train-only threshold guardrails for `class_weight_balanced`
2. period stability of `scale_pos_weight_medium`
3. production/balanced disagreement hold rule
4. confidence calibration diagnosis
5. low up-prediction decomposition
6. recent-regime robustness
7. bounded existing-feature-group ablation

When the gate becomes healthy, current evidence prioritizes `balanced_threshold_guardrail`: test whether a validation-only threshold can retain the balanced candidate's up-side benefit while limiting its down-recall loss. The actual cycle selected no hypothesis because selection intentionally occurs only after the gate.

## Worktree And Codex Safety

Eligible cycles create at most one branch/worktree:

- branch: `autonomous-research/YYYY-MM-DD/<hypothesis-id>`
- path: `C:\Users\kmats\Documents\ai-stock\stock-ai-worktrees\<hypothesis-id>`

The worktree helper rejects branch/path collisions and refuses to proceed if `database/market.db` is Git-tracked or appears in the worktree. Tests prove worktree isolation in a temporary clean repository. The Agent checks that worktree `HEAD` remains equal to the original main `HEAD` after Codex execution; a commit is an unsafe failure. Main `HEAD` and status are compared before/after.

The Codex probe first runs `codex --help` and `codex exec --help`. Invocation is constructed only if the help text confirms working-directory, workspace-write sandbox, approval, and final-output flags. In this environment:

- executable discovered: yes
- help executable: no
- reason: `codex_help_failed`
- detail: Access denied
- actual Codex invocation: none

## Evaluation Contract

The generated prompt limits work to one hypothesis and requires historical all, since 2024/2025/2026, recent six/three months, and live-like holdout windows. It requires production and five simple baselines, class metrics, confusion counts, leakage checks, tests, and explicit safety booleans.

Result validation rejects missing windows, missing baselines, failed tests, leakage, production/SQLite/prediction changes, commits, merges, or Scheduler changes. Human review is possible only after balanced accuracy and the primary metric improve, down recall stays within guardrail, simple baselines are beaten, multiple windows agree, the gain is not recent-only, leakage passes, and sample size is sufficient.

## Manual Cycle

- Wrapper dry-run: executed; worktree not created; exit 4 propagated correctly
- Normal manual cycle: executed once after fresh Operations/Research checks
- Final classification: `agent_skipped_gate`
- Stop reason: `main_git_head_missing`
- Exit code: 4
- Selected hypothesis: none, by gate policy
- Worktree: not created
- Codex task: not started
- Backlog/state: initialized
- Shared task: one deduplicated `codex_research_agent` gate-review task
- Locks remaining: none
- Commit/merge/push: none
- Task Scheduler registration: none

## Production Invariants

Before the Agent work:

- database SHA-256: `7CFA733FD5EFF9FEAF234C8E2E5AE4E67AFB5F097466E9BA0D61ED3ACC1F25AA`
- predictions: 81
- prediction_runs: 1
- prediction_records: 2
- binary artifact SHA-256: `EA947A34A711F560236F89411EA25B58E1F58337A6F0BBF497E9ED2B10D626FB`
- multiclass artifact SHA-256: `CBA07F0A648884876EDF65CBA691EE54987B22A48E66AC138B91306AB02F300D`
- settings SHA-256: `F9818647DC268C134227427E6376CAD235872B6A217E504BC29868950C583431`
- daily wrapper SHA-256: `83F25744BC1409033EDF74E37C506BAF84BD6E2BD0315CF463407B74B659CBA2`
- Scheduler XML aggregate SHA-256: `A693C8AE5E398387309A605F674942F1E7CFB6CDB393AFCFA90FE3F6A2E270B0`

The final verification section records the after-values. Operational Operations/Research/Agent reports and the shared task queue are expected monitoring outputs; they do not represent production changes.

Final comparison:

- all five production hashes above: unchanged
- predictions / prediction_runs / prediction_records: `81 / 1 / 2`, unchanged
- existing three-task Scheduler XML aggregate SHA-256: unchanged at `A693C8AE5E398387309A605F674942F1E7CFB6CDB393AFCFA90FE3F6A2E270B0`
- Agent Scheduler task registered: no
- Agent worktree created: no
- Agent/wrapper lock remaining: no
- Git HEAD: still absent; the Agent did not create a commit

## Verification

- Agent tests: 29 focused tests passed
- Full pytest: 410 passed
- Ruff: all checks passed
- Compileall: passed
- Health-check: Action required 0, Known missed dates 4, Known OK 1
- Autonomous Operations: `completed_read_only_cycle`
- Autonomous Research: duplicate-fingerprint safe skip, production change false
- Streamlit: formal `.venv` process on port 8501 returned HTTP 200

## Remaining Human Prerequisites

An actual worktree experiment is not yet eligible. A human must:

1. review `.gitignore` so production DB/artifacts/data are not tracked unintentionally
2. create the repository's initial reviewed commit
3. confirm `git status --short` is empty
4. provide a PowerShell-accessible Codex CLI and verify its help output
5. rerun the Agent dry-run

The Agent must not create the initial commit itself. Task Scheduler registration remains a separate future decision after multiple successful bounded manual cycles.
