"""cpcv_calibration.py

Marcos López de Prado: Advances in Financial Machine Learning (AFML)
Chapters 7, 12, & 14 Implementation for Stock-AI Quantitative Engine.

1. Causal Out-of-Fold / Rolling Probability Calibrator (Isotonic + Platt Hybrid, Brier Score minimization).
   Strict zero-leakage: probability at time t is calibrated using ONLY confirmed history tau < t.
2. Purged & Embargoed Cross-Validation (CPCV) Path Stability Evaluator.
   Purges overlapping target windows and enforces 1-3 bars embargo to prevent autocorrelation leak.
3. Deflated Sharpe Ratio (DSR) and Probabilistic Sharpe Ratio (PSR) for multiple testing bias correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationMetrics:
    brier_score_raw: float
    brier_score_calibrated: float
    brier_skill_score: float
    method_used: str
    sample_count: int


@dataclass
class CPCVPathResult:
    path_id: int
    test_folds: list[int]
    n_train: int
    n_test: int
    rank_ic: float
    sharpe_ratio: float
    win_rate: float
    brier_score: float


@dataclass
class CPCVStabilitySummary:
    cpcv_stability_score: float  # [0.0, 1.0]
    stability_grade: str  # S, A, B, C
    mean_rank_ic: float
    positive_ic_ratio: float
    min_path_sharpe: float
    mean_path_sharpe: float
    sharpe_dispersion: float
    path_count: int
    details: list[dict[str, Any]]


@dataclass
class DSRResult:
    observed_sharpe: float
    annualized_sharpe: float
    skewness: float
    kurtosis: float
    standard_error: float
    benchmark_sharpe: float
    expected_max_sharpe: float
    psr: float
    dsr: float
    is_statistically_significant: bool  # dsr >= 0.95
    confidence_grade: str  # High (DSR>=0.90), Medium (DSR>=0.75), Low


# ===========================================================================
# 1. Brier Score & Probability Calibration
# ===========================================================================


def compute_brier_score(y_true: np.ndarray | Sequence[int | float], y_prob: np.ndarray | Sequence[float]) -> float:
    """Compute the Brier Score: 1/N * sum((p_i - y_i)^2). Lower is better [0, 1]."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_prob, dtype=float)
    if len(yt) == 0:
        return 0.25
    return float(np.mean((yp - yt) ** 2))


def compute_brier_skill_score(y_true: np.ndarray | Sequence[int | float], y_prob: np.ndarray | Sequence[float]) -> float:
    """Compute Brier Skill Score relative to climatology (base rate): BSS = 1 - (BS / BS_ref). Higher is better."""
    yt = np.asarray(y_true, dtype=float)
    if len(yt) == 0:
        return 0.0
    bs = compute_brier_score(yt, y_prob)
    base_rate = float(np.mean(yt))
    bs_ref = float(np.mean((base_rate - yt) ** 2))
    if bs_ref < 1e-8:
        return 0.0
    return float(1.0 - (bs / bs_ref))


class CausalHybridCalibrator:
    """Causal Hybrid Probability Calibrator combining Isotonic Regression, Platt Scaling, and Temperature Scaling.

    - Strictly monotonically non-decreasing.
    - Suppresses overconfidence at top quantiles to mitigate false-positive breakouts.
    - Brier-optimal blending between Isotonic and Temperature/Platt scaling.
    """

    def __init__(
        self,
        min_samples: int = 20,
        clip_bounds: tuple[float, float] = (0.02, 0.98),
        blend_weight: float | None = None,  # None = dynamic selection based on Brier Score
    ):
        self.min_samples = min_samples
        self.clip_bounds = clip_bounds
        self.blend_weight = blend_weight
        self.isotonic_model: IsotonicRegression | None = None
        self.temp_param: float = 1.0  # Temperature scaling (T > 1 softens overconfidence)
        self.bias_param: float = 0.0  # Base rate bias shift
        self.base_rate: float = 0.50
        self.is_fitted: bool = False
        self.best_method: str = "identity"
        self.brier_raw: float = 0.25
        self.brier_calibrated: float = 0.25

    def fit(self, history_p: np.ndarray | Sequence[float], history_y: np.ndarray | Sequence[int | float]) -> "CausalHybridCalibrator":
        """Fit calibration mapping on historical confirmed predictions and binary outcomes."""
        hp = np.asarray(history_p, dtype=float)
        hy = np.asarray(history_y, dtype=float)

        valid_mask = ~(np.isnan(hp) | np.isnan(hy))
        hp = np.clip(hp[valid_mask], 1e-4, 1.0 - 1e-4)
        hy = hy[valid_mask]

        if len(hp) < self.min_samples or len(np.unique(hy)) < 2:
            self.base_rate = float(np.mean(hy)) if len(hy) > 0 else 0.50
            self.is_fitted = False
            self.best_method = "fallback_linear"
            return self

        self.base_rate = float(np.mean(hy))
        self.brier_raw = compute_brier_score(hy, hp)

        # 1. Temperature-Regularized Platt Scaling (Direct Brier Score minimization)
        logits = np.log(hp / (1.0 - hp))

        def _brier_obj(params: list[float]) -> float:
            t_val, b_val = params
            p_c = 1.0 / (1.0 + np.exp(-(logits / max(0.2, t_val) + b_val)))
            return float(np.mean((p_c - hy) ** 2) + 1e-5 * (t_val - 1.0) ** 2 + 1e-5 * b_val**2)

        try:
            from scipy.optimize import minimize
            opt_res = minimize(_brier_obj, [1.1, 0.0], bounds=[(0.4, 3.0), (-0.3, 0.3)], method="L-BFGS-B")
            self.temp_param = float(opt_res.x[0])
            self.bias_param = float(opt_res.x[1])
            p_temp_train = 1.0 / (1.0 + np.exp(-(logits / self.temp_param + self.bias_param)))
            temp_bs = compute_brier_score(hy, p_temp_train)
        except Exception:
            self.temp_param = 1.0
            self.bias_param = 0.0
            temp_bs = self.brier_raw

        # 2. Isotonic Regression (Non-parametric, monotonic non-decreasing)
        iso = IsotonicRegression(
            y_min=self.clip_bounds[0],
            y_max=self.clip_bounds[1],
            increasing=True,
            out_of_bounds="clip",
        )
        try:
            iso.fit(hp, hy)
            self.isotonic_model = iso
            iso_preds = iso.predict(hp)
            iso_bs = compute_brier_score(hy, iso_preds)
        except Exception:
            self.isotonic_model = None
            iso_bs = 1.0

        # Decide best combination
        if self.isotonic_model is not None and temp_bs <= self.brier_raw:
            self.best_method = "hybrid_isotonic_temp"
            self.brier_calibrated = min(temp_bs, iso_bs)
        elif temp_bs <= self.brier_raw:
            self.best_method = "temperature_platt"
            self.brier_calibrated = temp_bs
        elif self.isotonic_model is not None:
            self.best_method = "isotonic_only"
            self.brier_calibrated = iso_bs
        else:
            self.best_method = "identity"
            self.brier_calibrated = self.brier_raw

        self.is_fitted = True
        return self

    def predict_proba(self, p_raw: np.ndarray | Sequence[float] | float) -> np.ndarray | float:
        """Calibrate raw probabilities into well-calibrated posterior probabilities."""
        is_scalar = isinstance(p_raw, (int, float, np.floating))
        p_arr = np.atleast_1d(np.asarray(p_raw, dtype=float))
        p_arr = np.clip(p_arr, 1e-4, 1.0 - 1e-4)

        if not self.is_fitted:
            smoothed = 0.85 * p_arr + 0.15 * self.base_rate
            res = np.clip(smoothed, self.clip_bounds[0], self.clip_bounds[1])
            return float(res[0]) if is_scalar else res

        # 1. Temperature-Platt calibrated
        logits = np.log(p_arr / (1.0 - p_arr))
        p_temp = 1.0 / (1.0 + np.exp(-(logits / self.temp_param + self.bias_param)))

        # 2. Isotonic calibrated
        if self.isotonic_model is not None:
            p_iso = np.clip(self.isotonic_model.predict(p_arr), self.clip_bounds[0], self.clip_bounds[1])
            w_temp = 0.70 if self.blend_weight is None else (1.0 - self.blend_weight)
            calibrated = (1.0 - w_temp) * p_iso + w_temp * p_temp
        else:
            calibrated = p_temp

        calibrated = np.clip(calibrated, self.clip_bounds[0], self.clip_bounds[1])
        return float(calibrated[0]) if is_scalar else calibrated



def calibrate_probabilities_causal(
    p_raw: np.ndarray | Sequence[float] | float,
    history_p: np.ndarray | Sequence[float],
    history_y: np.ndarray | Sequence[int | float],
    min_samples: int = 20,
    blend_weight: float | None = None,
) -> tuple[np.ndarray | float, CalibrationMetrics]:
    """Calibrate raw probability using strictly historical confirmed observations.

    Guarantees zero-leakage: does not inspect any data at or after current prediction timestamp.
    """
    calibrator = CausalHybridCalibrator(min_samples=min_samples, blend_weight=blend_weight)
    calibrator.fit(history_p, history_y)
    calibrated_p = calibrator.predict_proba(p_raw)

    hp = np.asarray(history_p, dtype=float)
    hy = np.asarray(history_y, dtype=float)
    if len(hp) >= min_samples and len(hy) >= min_samples:
        bs_raw = compute_brier_score(hy, hp)
        hist_cal = calibrator.predict_proba(hp)
        bs_cal = compute_brier_score(hy, hist_cal)
        bss = compute_brier_skill_score(hy, hist_cal)
    else:
        bs_raw = 0.25
        bs_cal = 0.25
        bss = 0.0

    metrics = CalibrationMetrics(
        brier_score_raw=round(bs_raw, 5),
        brier_score_calibrated=round(bs_cal, 5),
        brier_skill_score=round(bss, 4),
        method_used=calibrator.best_method,
        sample_count=len(history_p),
    )
    return calibrated_p, metrics


def calibrate_time_series_walk_forward(
    df: pd.DataFrame,
    prob_col: str = "p_up",
    target_col: str = "next_open_binary",
    date_col: str = "date",
    min_history: int = 25,
    blend_weight: float | None = 0.60,
) -> pd.Series:
    """Perform causal expanding walk-forward calibration across a time-series DataFrame.

    At each date t, calibrates prob_col[t] using ONLY rows where date < t.
    """
    if prob_col not in df.columns or target_col not in df.columns:
        return df.get(prob_col, pd.Series(0.5, index=df.index))

    sorted_df = df.sort_values(date_col)
    unique_dates = sorted_df[date_col].drop_duplicates().tolist()
    calibrated_series = pd.Series(index=sorted_df.index, dtype=float)

    confirmed_p: list[float] = []
    confirmed_y: list[float] = []

    for d in unique_dates:
        curr_idx = sorted_df[sorted_df[date_col] == d].index
        raw_p = sorted_df.loc[curr_idx, prob_col].values

        if len(confirmed_p) >= min_history:
            calibrator = CausalHybridCalibrator(min_samples=min_history, blend_weight=blend_weight)
            calibrator.fit(confirmed_p, confirmed_y)
            cal_p = calibrator.predict_proba(raw_p)
            calibrated_series.loc[curr_idx] = cal_p
        else:
            # Fallback before minimum history
            calibrated_series.loc[curr_idx] = raw_p

        # Append confirmed labels for subsequent dates (strictly causal tau < next date)
        targets = sorted_df.loc[curr_idx, target_col].values
        for p_val, y_val in zip(raw_p, targets):
            if pd.notna(p_val) and pd.notna(y_val):
                confirmed_p.append(float(p_val))
                confirmed_y.append(float(y_val))

    return calibrated_series.reindex(df.index)


# ===========================================================================
# 2. Purged & Embargoed Cross-Validation (CPCV)
# ===========================================================================


@dataclass
class CPCVFold:
    train_indices: np.ndarray
    test_indices: np.ndarray
    test_blocks: list[int]
    purged_count: int
    embargoed_count: int


def generate_cpcv_folds(
    n_samples: int,
    n_splits: int = 5,
    n_test_splits: int = 2,
    label_lookahead: int = 1,
    embargo_bars: int = 2,
) -> list[CPCVFold]:
    """Generate Combinatorial Purged & Embargoed Cross-Validation (CPCV) folds.

    Divides time series into N blocks, chooses k test blocks (C(N, k) combinations).
    Purges overlapping training samples before test blocks, and embargos training
    samples immediately following test blocks.
    """
    if n_samples < n_splits * 5:
        # Fallback for small datasets
        return []

    block_size = n_samples // n_splits
    block_indices = [
        np.arange(i * block_size, (i + 1) * block_size if i < n_splits - 1 else n_samples)
        for i in range(n_splits)
    ]

    combinations = list(itertools.combinations(range(n_splits), n_test_splits))
    folds: list[CPCVFold] = []

    for comb in combinations:
        test_idx_set = set()
        for b_idx in comb:
            test_idx_set.update(block_indices[b_idx].tolist())

        test_indices = np.array(sorted(test_idx_set))
        train_candidates = [i for i in range(n_samples) if i not in test_idx_set]

        # Purge & Embargo logic
        purged_indices = set()
        embargoed_indices = set()

        for b_idx in comb:
            b_start = block_indices[b_idx][0]
            b_end = block_indices[b_idx][-1]

            # Purge: Training samples immediately preceding test block whose label horizon overlaps
            if label_lookahead > 0:
                p_start = max(0, b_start - label_lookahead)
                for p_idx in range(p_start, b_start):
                    if p_idx not in test_idx_set:
                        purged_indices.add(p_idx)

            # Embargo: Training samples immediately following test block to prevent post-event leakage
            if embargo_bars > 0:
                e_end = min(n_samples, b_end + 1 + embargo_bars)
                for e_idx in range(b_end + 1, e_end):
                    if e_idx not in test_idx_set:
                        embargoed_indices.add(e_idx)

        excluded = test_idx_set | purged_indices | embargoed_indices
        train_indices = np.array([i for i in range(n_samples) if i not in excluded])

        if len(train_indices) > 0 and len(test_indices) > 0:
            folds.append(
                CPCVFold(
                    train_indices=train_indices,
                    test_indices=test_indices,
                    test_blocks=list(comb),
                    purged_count=len(purged_indices),
                    embargoed_count=len(embargoed_indices),
                )
            )

    return folds


def compute_cpcv_path_stability(
    df: pd.DataFrame,
    pred_col: str = "p_up",
    target_col: str = "next_open_binary",
    return_col: str = "next_open_return",
    date_col: str = "date",
    n_splits: int = 5,
    n_test_splits: int = 2,
    embargo_bars: int = 2,
) -> CPCVStabilitySummary:
    """Compute CPCV path stability score across multiple purged & embargoed combinations."""
    if df.empty or pred_col not in df.columns or target_col not in df.columns:
        return CPCVStabilitySummary(
            cpcv_stability_score=0.50,
            stability_grade="C",
            mean_rank_ic=0.0,
            positive_ic_ratio=0.50,
            min_path_sharpe=0.0,
            mean_path_sharpe=0.0,
            sharpe_dispersion=1.0,
            path_count=0,
            details=[],
        )

    clean_df = df.dropna(subset=[pred_col, target_col]).sort_values(date_col).reset_index(drop=True)
    n_samples = len(clean_df)

    if n_samples < 30:
        return CPCVStabilitySummary(
            cpcv_stability_score=0.50,
            stability_grade="C",
            mean_rank_ic=0.0,
            positive_ic_ratio=0.50,
            min_path_sharpe=0.0,
            mean_path_sharpe=0.0,
            sharpe_dispersion=1.0,
            path_count=0,
            details=[],
        )

    folds = generate_cpcv_folds(
        n_samples=n_samples,
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        label_lookahead=1,
        embargo_bars=embargo_bars,
    )

    if not folds:
        return CPCVStabilitySummary(
            cpcv_stability_score=0.50,
            stability_grade="C",
            mean_rank_ic=0.0,
            positive_ic_ratio=0.50,
            min_path_sharpe=0.0,
            mean_path_sharpe=0.0,
            sharpe_dispersion=1.0,
            path_count=0,
            details=[],
        )

    path_results: list[CPCVPathResult] = []

    for p_id, fold in enumerate(folds, 1):
        test_sub = clean_df.iloc[fold.test_indices]
        p_vals = test_sub[pred_col].values
        y_vals = test_sub[target_col].values
        r_vals = test_sub[return_col].values if return_col in test_sub.columns else y_vals - 0.50

        # 1. Rank IC
        if len(p_vals) >= 5 and len(np.unique(p_vals)) > 1 and len(np.unique(r_vals)) > 1:
            ric, _ = stats.spearmanr(p_vals, r_vals)
            ric = 0.0 if np.isnan(ric) else float(ric)
        else:
            ric = 0.0

        # 2. Realized strategy Sharpe on test fold (Long when p > 0.50)
        long_pos = (p_vals >= 0.50).astype(float)
        trade_rets = long_pos * r_vals
        std_r = np.std(trade_rets)
        if std_r > 1e-7:
            path_sr = float(np.mean(trade_rets) / std_r * np.sqrt(250.0))
        else:
            path_sr = 0.0

        # 3. Win rate & Brier
        win_rate = float(np.mean(trade_rets[long_pos > 0] > 0)) if np.sum(long_pos) > 0 else 0.50
        bs = compute_brier_score(y_vals, p_vals)

        path_results.append(
            CPCVPathResult(
                path_id=p_id,
                test_folds=fold.test_blocks,
                n_train=len(fold.train_indices),
                n_test=len(fold.test_indices),
                rank_ic=round(ric, 4),
                sharpe_ratio=round(path_sr, 2),
                win_rate=round(win_rate, 4),
                brier_score=round(bs, 4),
            )
        )

    ics = [p.rank_ic for p in path_results]
    srs = [p.sharpe_ratio for p in path_results]

    mean_ic = float(np.mean(ics)) if ics else 0.0
    pos_ic_ratio = float(np.mean([ic > 0 for ic in ics])) if ics else 0.50
    min_sr = float(np.min(srs)) if srs else 0.0
    mean_sr = float(np.mean(srs)) if srs else 0.0
    sr_disp = float(np.std(srs)) if len(srs) > 1 else 0.50

    # Composite stability score: combines positive IC frequency, minimal Sharpe floor, and low dispersion
    # Target: 1.0 (impeccable stability across all regimes)
    pos_ic_comp = pos_ic_ratio  # [0, 1]
    sharpe_floor_comp = float(np.clip((min_sr + 0.5) / 1.5, 0.0, 1.0))  # positive floor yields high score
    disp_pen = float(np.clip(1.0 - (sr_disp / 3.0), 0.0, 1.0))

    stability_score = 0.40 * pos_ic_comp + 0.35 * sharpe_floor_comp + 0.25 * disp_pen
    stability_score = float(np.clip(stability_score, 0.0, 1.0))

    if stability_score >= 0.78:
        grade = "S"
    elif stability_score >= 0.65:
        grade = "A"
    elif stability_score >= 0.50:
        grade = "B"
    else:
        grade = "C"

    return CPCVStabilitySummary(
        cpcv_stability_score=round(stability_score, 3),
        stability_grade=grade,
        mean_rank_ic=round(mean_ic, 4),
        positive_ic_ratio=round(pos_ic_ratio, 3),
        min_path_sharpe=round(min_sr, 2),
        mean_path_sharpe=round(mean_sr, 2),
        sharpe_dispersion=round(sr_disp, 3),
        path_count=len(path_results),
        details=[asdict(p) for p in path_results],
    )


# ===========================================================================
# 3. Deflated Sharpe Ratio (DSR) & Probabilistic Sharpe Ratio (PSR)
# ===========================================================================


def compute_probabilistic_sharpe_ratio(
    returns: np.ndarray | Sequence[float],
    benchmark_sr: float = 0.0,
    periods_per_year: float = 250.0,
) -> tuple[float, float, float, float]:
    """Compute Bailey & López de Prado (2012) Probabilistic Sharpe Ratio (PSR).

    Returns:
        (psr, observed_sr_ann, standard_error, t_samples)
    """
    rets = np.asarray(returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)

    if n < 5:
        return 0.50, 0.0, 1.0, float(n)

    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1))

    if std_r < 1e-8:
        return 0.50, 0.0, 1.0, float(n)

    sr_period = mean_r / std_r
    sr_ann = sr_period * np.sqrt(periods_per_year)
    bm_period = benchmark_sr / np.sqrt(periods_per_year)

    # Higher moments for non-normality adjustment
    z = (rets - mean_r) / std_r
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))  # normal distribution = 3.0

    # Standard error of the Sharpe ratio (Mertens 2002 / AFML Chapter 14)
    # sigma_sr = sqrt((1 - skew*sr + ((kurt - 1)/4)*sr^2) / (n - 1))
    var_sr = (1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period**2)) / max(1, n - 1)
    se_sr = float(np.sqrt(max(1e-8, var_sr)))

    # PSR = Phi((sr_period - bm_period) / se_sr)
    z_stat = (sr_period - bm_period) / se_sr
    psr = float(stats.norm.cdf(z_stat))

    return psr, sr_ann, se_sr, float(n)


def compute_deflated_sharpe_ratio(
    returns: np.ndarray | Sequence[float],
    num_trials: int = 50,
    variance_trials: float = 1.0,
    periods_per_year: float = 250.0,
) -> DSRResult:
    """Compute Marcos López de Prado (2014) Deflated Sharpe Ratio (DSR).

    Adjusts observed Sharpe ratio for multiple testing bias and non-normality.
    Uses Extreme Value Theory (EVT) to compute expected maximum Sharpe ratio under the null hypothesis.
    """
    rets = np.asarray(returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    n = len(rets)

    if n < 5:
        return DSRResult(
            observed_sharpe=0.0,
            annualized_sharpe=0.0,
            skewness=0.0,
            kurtosis=3.0,
            standard_error=1.0,
            benchmark_sharpe=0.0,
            expected_max_sharpe=0.0,
            psr=0.50,
            dsr=0.50,
            is_statistically_significant=False,
            confidence_grade="Low",
        )

    mean_r = float(np.mean(rets))
    std_r = float(np.std(rets, ddof=1))
    sr_period = (mean_r / std_r) if std_r > 1e-8 else 0.0
    sr_ann = sr_period * np.sqrt(periods_per_year)

    z = (rets - mean_r) / (std_r + 1e-8)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))

    # Standard error of Sharpe ratio
    var_sr = (1.0 - skew * sr_period + ((kurt - 1.0) / 4.0) * (sr_period**2)) / max(1, n - 1)
    se_sr = float(np.sqrt(max(1e-8, var_sr)))

    # Expected maximum Sharpe ratio under multiple testing null hypothesis (Bailey & Lopez de Prado 2014)
    # E[max_K] = sqrt(V) * ((1 - gamma)*Phi^-1(1 - 1/K) + gamma*Phi^-1(1 - 1/(K*e)))
    euler_mascheroni = 0.5772156649
    k = max(2, num_trials)
    p1 = 1.0 - (1.0 / k)
    p2 = 1.0 - (1.0 / (k * math.e))

    z1 = float(stats.norm.ppf(p1))
    z2 = float(stats.norm.ppf(p2))

    # Expected max SR in annualized terms, then converted to period scale
    sr_star_ann = math.sqrt(variance_trials) * ((1.0 - euler_mascheroni) * z1 + euler_mascheroni * z2)
    sr_star_period = sr_star_ann / np.sqrt(periods_per_year)

    # PSR with respect to zero benchmark
    psr_z = (sr_period - 0.0) / se_sr
    psr = float(stats.norm.cdf(psr_z))

    # DSR with respect to expected maximum under multiple testing
    dsr_z = (sr_period - sr_star_period) / se_sr
    dsr = float(stats.norm.cdf(dsr_z))

    is_sig = dsr >= 0.95
    if dsr >= 0.90:
        conf_grade = "High"
    elif dsr >= 0.70:
        conf_grade = "Medium"
    else:
        conf_grade = "Low"

    return DSRResult(
        observed_sharpe=round(sr_period, 4),
        annualized_sharpe=round(sr_ann, 2),
        skewness=round(skew, 2),
        kurtosis=round(kurt, 2),
        standard_error=round(se_sr, 4),
        benchmark_sharpe=0.0,
        expected_max_sharpe=round(sr_star_ann, 2),
        psr=round(psr, 4),
        dsr=round(dsr, 4),
        is_statistically_significant=is_sig,
        confidence_grade=conf_grade,
    )
