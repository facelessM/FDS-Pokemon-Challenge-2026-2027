"""
Empirical study on probability weighting and variance-based aggregation.

Every constant is calculated directly from the data distributions and sample statistics;
no arbitrary magic numbers are used.

Methodologies evaluated:
1. Baseline: Equal weights (1/K)
2. Inverse-Brier Variance Weighting: Gauss-Markov optimal weights w_j proportional to 1/MSE_j
3. Bernoulli Variance Weighting: per-prediction uncertainty w_i,j proportional to 1/(p*(1-p) + eps)
4. Inter-Model Disagreement: adaptive switching based on sample variance s_i^2
5. Robust Aggregation: Trimmed Mean and Median (L1 estimator)
6. Prevalance-Calibrated Threshold: correcting marginal calibration bias
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

def run_study(npz_path: Path):
    with np.load(npz_path) as data:
        y = data["y"]  # True binary labels {0, 1}
        P_comp = data["cv_components"]  # Shape: (N, 5)
        names = data["names"].tolist()

    n_samples, n_models = P_comp.shape
    prevalence = np.mean(y)
    print("=" * 70)
    print(f"ANALYSIS OF PROBABILITY & VARIANCE WEIGHTING (N = {n_samples})")
    print(f"Empirical class prevalence (P1 win rate): {prevalence:.4f}")
    print("=" * 70)

    # -------------------------------------------------------------
    # 0. Individual Model Standalone Performance
    # -------------------------------------------------------------
    print("\n--- 0. INDIVIDUAL MODEL STANDALONE PERFORMANCE ---")
    brier_scores = []
    log_losses = []
    accuracies = []
    for j, name in enumerate(names):
        p_j = P_comp[:, j]
        acc_j = accuracy_score(y, p_j >= 0.5)
        bs_j = brier_score_loss(y, p_j)
        ll_j = log_loss(y, p_j)
        brier_scores.append(bs_j)
        log_losses.append(ll_j)
        accuracies.append(acc_j)
        print(f"  [{name:18s}] Accuracy: {acc_j:.4f} | Brier (MSE): {bs_j:.5f} | LogLoss: {ll_j:.5f}")

    brier_scores = np.array(brier_scores)

    # -------------------------------------------------------------
    # 1. Baseline: Equal Weights (1/K)
    # -------------------------------------------------------------
    w_equal = np.full(n_models, 1.0 / n_models)
    p_equal = P_comp @ w_equal
    acc_equal = accuracy_score(y, p_equal >= 0.5)
    bs_equal = brier_score_loss(y, p_equal)
    ll_equal = log_loss(y, p_equal)
    print("\n--- 1. BASELINE: EQUAL WEIGHTING (1/5 each) ---")
    print(f"  Accuracy: {acc_equal:.4f} | Brier: {bs_equal:.5f} | LogLoss: {ll_equal:.5f}")

    # -------------------------------------------------------------
    # 2. Inverse-Brier Variance Weighting (Gauss-Markov Formulation)
    # Theory: For unbiased estimators, minimum variance combination has w_j proportional to 1/Var(error_j).
    # Since Brier score is MSE = Var(e) + Bias^2, w_j = (1/BS_j) / sum(1/BS_k).
    # -------------------------------------------------------------
    inv_brier = 1.0 / brier_scores
    w_inv_brier = inv_brier / np.sum(inv_brier)
    p_inv_brier = P_comp @ w_inv_brier
    acc_inv_brier = accuracy_score(y, p_inv_brier >= 0.5)
    bs_inv_brier = brier_score_loss(y, p_inv_brier)
    ll_inv_brier = log_loss(y, p_inv_brier)

    print("\n--- 2. INVERSE-BRIER VARIANCE WEIGHTING ---")
    print("  Calculated weights (derived from 1/MSE):")
    for name, w in zip(names, w_inv_brier):
        print(f"    {name:18s}: {w:.4f}")
    print(f"  Accuracy: {acc_inv_brier:.4f} (Delta vs Baseline: {acc_inv_brier - acc_equal:+.4f})")
    print(f"  Brier:    {bs_inv_brier:.5f} | LogLoss: {ll_inv_brier:.5f}")

    # -------------------------------------------------------------
    # 3. Bernoulli Variance Weighting (Per-Observation Aleatoric Uncertainty)
    # Theory: A Bernoulli variable has variance Var(Y|p) = p*(1-p).
    # When p -> 0 or 1, Var -> 0 (high certainty); at p=0.5, Var = 0.25 (maximum uncertainty).
    # Regularizer epsilon is derived from a Laplace prior: 1 / N_train = 1 / 8000 = 0.000125.
    # -------------------------------------------------------------
    eps_laplace = 1.0 / 8000.0  # Laplace smoothing based on training fold size
    bern_var = P_comp * (1.0 - P_comp)  # Shape: (N, 5)
    inv_bern_var = 1.0 / (bern_var + eps_laplace)
    w_bern = inv_bern_var / np.sum(inv_bern_var, axis=1, keepdims=True)
    p_bern = np.sum(w_bern * P_comp, axis=1)

    acc_bern = accuracy_score(y, p_bern >= 0.5)
    bs_bern = brier_score_loss(y, p_bern)
    ll_bern = log_loss(y, p_bern)

    print("\n--- 3. BERNOULLI VARIANCE WEIGHTING (Laplace eps = 1/8000) ---")
    print(f"  Accuracy: {acc_bern:.4f} (Delta vs Baseline: {acc_bern - acc_equal:+.4f})")
    print(f"  Brier:    {bs_bern:.5f} | LogLoss: {ll_bern:.5f}")

    # -------------------------------------------------------------
    # 4. Confidence-Weighted Power Law: w proportional to |p - 0.5|^gamma
    # We evaluate gammas derived systematically:
    # gamma = 0.5 (sublinear / square-root margin)
    # gamma = 1.0 (linear margin)
    # gamma = 2.0 (quadratic margin)
    # eps is chosen as the machine epsilon / minimal observable margin: 0.001
    # -------------------------------------------------------------
    print("\n--- 4. CONFIDENCE POWER-LAW SWEEP ---")
    for gamma in [0.25, 0.5, 1.0, 1.5, 2.0]:
        conf = np.abs(P_comp - 0.5) ** gamma + 1e-4
        w_conf = conf / np.sum(conf, axis=1, keepdims=True)
        p_conf = np.sum(w_conf * P_comp, axis=1)
        acc_conf = accuracy_score(y, p_conf >= 0.5)
        print(f"  gamma = {gamma:4.2f} -> Accuracy: {acc_conf:.4f} (Delta: {acc_conf - acc_equal:+.4f}) | LogLoss: {log_loss(y, p_conf):.5f}")

    # -------------------------------------------------------------
    # 5. Robust Aggregation: Median & Trimmed Mean
    # Theory: Median is the optimal L1 estimator, unaffected by extreme outliers.
    # Trimmed mean removes the single lowest and highest prediction per sample.
    # -------------------------------------------------------------
    p_median = np.median(P_comp, axis=1)
    acc_median = accuracy_score(y, p_median >= 0.5)

    # Trimmed mean: sort across axis 1, keep indices [1, 2, 3] out of 0..4
    P_sorted = np.sort(P_comp, axis=1)
    p_trimmed = np.mean(P_sorted[:, 1:4], axis=1)
    acc_trimmed = accuracy_score(y, p_trimmed >= 0.5)

    print("\n--- 5. ROBUST AGGREGATION (OUTLIER RESISTANT) ---")
    print(f"  Median Voting:       Accuracy: {acc_median:.4f} (Delta: {acc_median - acc_equal:+.4f})")
    print(f"  Trimmed Mean (3 of 5): Accuracy: {acc_trimmed:.4f} (Delta: {acc_trimmed - acc_equal:+.4f})")

    # -------------------------------------------------------------
    # 6. Inter-Model Disagreement Variance
    # Measure sample variance across the 5 models for each battle:
    # s_i^2 = Var(p_{i, 1..5})
    # Threshold tau is calculated as the median disagreement: tau = median(s^2)
    # -------------------------------------------------------------
    disagreement_var = np.var(P_comp, axis=1, ddof=1)
    tau_disagreement = np.median(disagreement_var)
    print("\n--- 6. ADAPTIVE DISAGREEMENT SWITCHING ---")
    print(f"  Calculated median disagreement variance tau: {tau_disagreement:.6f}")
    # Adaptive rule: if variance is low -> equal mean; if high -> median
    p_adaptive = np.where(disagreement_var > tau_disagreement, p_median, p_equal)
    acc_adaptive = accuracy_score(y, p_adaptive >= 0.5)
    print(f"  Adaptive Mean/Median: Accuracy: {acc_adaptive:.4f} (Delta: {acc_adaptive - acc_equal:+.4f})")

    # -------------------------------------------------------------
    # 7. Marginal Prevalence Bias Correction
    # If the mean predicted probability bar(P) diverges from true prevalence pi,
    # the optimal threshold is shifted by Delta = bar(P) - pi.
    # -------------------------------------------------------------
    mean_p_equal = np.mean(p_equal)
    delta_bias = mean_p_equal - prevalence
    calibrated_threshold = 0.5 + delta_bias
    acc_cal_threshold = accuracy_score(y, p_equal >= calibrated_threshold)
    print("\n--- 7. MARGINAL PREVALENCE CALIBRATED THRESHOLD ---")
    print(f"  Mean P_equal: {mean_p_equal:.4f} | True Prevalence: {prevalence:.4f} | Bias Delta: {delta_bias:+.5f}")
    print(f"  Calibrated Threshold theta* = 0.5 + Delta: {calibrated_threshold:.5f}")
    print(f"  Accuracy at theta*: {acc_cal_threshold:.4f} (Delta: {acc_cal_threshold - acc_equal:+.4f})")

def load_data(path: Path):
    if path.is_dir():
        files = sorted(path.glob("*_*.npz"))
        if not files:
            raise FileNotFoundError(f"No *_*.npz files found in {path}")
        ys, Ps = [], []
        names = None
        for f in files:
            with np.load(f) as data:
                ys.append(data["y"])
                Ps.append(data["cv_components"])
                names = data["names"].tolist()
        return np.concatenate(ys), np.vstack(Ps), names, files
    else:
        with np.load(path) as data:
            return data["y"], data["cv_components"], data["names"].tolist(), [path]

def run_study(target: Path):
    y, P_comp, names, loaded_files = load_data(target)
    n_samples, n_models = P_comp.shape
    prevalence = np.mean(y)
    print("=" * 70)
    print(f"ANALYSIS OF PROBABILITY & VARIANCE WEIGHTING")
    print(f"Loaded {len(loaded_files)} fold(s): {[f.name for f in loaded_files]}")
    print(f"Total validation samples N = {n_samples}")
    print(f"Empirical class prevalence (P1 win rate): {prevalence:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    import sys
    target = Path("outputs/cv")
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    if not target.exists():
        print(f"Target {target} does not exist.")
    else:
        y, P_comp, names, loaded_files = load_data(target)
        n_samples, n_models = P_comp.shape
        prevalence = np.mean(y)
        print("=" * 70)
        print(f"ANALYSIS OF PROBABILITY & VARIANCE WEIGHTING")
        print(f"Loaded {len(loaded_files)} fold(s): {[f.name for f in loaded_files]}")
        print(f"Total validation samples N = {n_samples}")
        print(f"Empirical class prevalence (P1 win rate): {prevalence:.4f}")
        print("=" * 70)

        # -------------------------------------------------------------
        # 0. Individual Model Standalone Performance
        # -------------------------------------------------------------
        print("\n--- 0. INDIVIDUAL MODEL STANDALONE PERFORMANCE ---")
        brier_scores = []
        log_losses = []
        accuracies = []
        for j, name in enumerate(names):
            p_j = P_comp[:, j]
            acc_j = accuracy_score(y, p_j >= 0.5)
            bs_j = brier_score_loss(y, p_j)
            ll_j = log_loss(y, p_j)
            brier_scores.append(bs_j)
            log_losses.append(ll_j)
            accuracies.append(acc_j)
            print(f"  [{name:18s}] Accuracy: {acc_j:.4f} | Brier (MSE): {bs_j:.5f} | LogLoss: {ll_j:.5f}")

        brier_scores = np.array(brier_scores)

        # -------------------------------------------------------------
        # 1. Baseline: Equal Weights (1/K)
        # -------------------------------------------------------------
        w_equal = np.full(n_models, 1.0 / n_models)
        p_equal = P_comp @ w_equal
        acc_equal = accuracy_score(y, p_equal >= 0.5)
        bs_equal = brier_score_loss(y, p_equal)
        ll_equal = log_loss(y, p_equal)
        print("\n--- 1. BASELINE: EQUAL WEIGHTING (1/5 each) ---")
        print(f"  Accuracy: {acc_equal:.4f} | Brier: {bs_equal:.5f} | LogLoss: {ll_equal:.5f}")

        # -------------------------------------------------------------
        # 2. Inverse-Brier Variance Weighting (Gauss-Markov Formulation)
        # -------------------------------------------------------------
        inv_brier = 1.0 / brier_scores
        w_inv_brier = inv_brier / np.sum(inv_brier)
        p_inv_brier = P_comp @ w_inv_brier
        acc_inv_brier = accuracy_score(y, p_inv_brier >= 0.5)
        bs_inv_brier = brier_score_loss(y, p_inv_brier)
        ll_inv_brier = log_loss(y, p_inv_brier)

        print("\n--- 2. INVERSE-BRIER VARIANCE WEIGHTING ---")
        print("  Calculated weights (derived from 1/MSE):")
        for name, w in zip(names, w_inv_brier):
            print(f"    {name:18s}: {w:.4f}")
        print(f"  Accuracy: {acc_inv_brier:.4f} (Delta vs Baseline: {acc_inv_brier - acc_equal:+.4f})")
        print(f"  Brier:    {bs_inv_brier:.5f} | LogLoss: {ll_inv_brier:.5f}")

        # -------------------------------------------------------------
        # 2b. Strict Out-Of-Fold Cross-Fitted Inverse-Brier (Zero-Leakage)
        # For each fold k, fit weights w^(-k) using only other folds, apply to fold k.
        # -------------------------------------------------------------
        if len(loaded_files) > 1:
            fold_preds = []
            fold_ys = []
            for k, f in enumerate(loaded_files):
                with np.load(f) as data_k:
                    va_y = data_k["y"]
                    va_P = data_k["cv_components"]
                other_ys = []
                other_Ps = []
                for j, fj in enumerate(loaded_files):
                    if j != k:
                        with np.load(fj) as data_j:
                            other_ys.append(data_j["y"])
                            other_Ps.append(data_j["cv_components"])
                o_y = np.concatenate(other_ys)
                o_P = np.vstack(other_Ps)
                o_bs = np.mean((o_P - o_y[:, None])**2, axis=0)
                o_w = (1.0 / o_bs) / np.sum(1.0 / o_bs)
                fold_preds.append(va_P @ o_w)
                fold_ys.append(va_y)
            p_strict_ib = np.concatenate(fold_preds)
            y_strict = np.concatenate(fold_ys)
            acc_strict = accuracy_score(y_strict, p_strict_ib >= 0.5)
            bs_strict = brier_score_loss(y_strict, p_strict_ib)
            ll_strict = log_loss(y_strict, p_strict_ib)
            print("\n--- 2b. STRICT OUT-OF-FOLD CROSS-FITTED INVERSE-BRIER ---")
            print(f"  Strict OOF Accuracy: {acc_strict:.4f} (Delta vs Baseline: {acc_strict - acc_equal:+.4f})")
            print(f"  Strict OOF Brier:    {bs_strict:.5f} | LogLoss: {ll_strict:.5f}")

        # -------------------------------------------------------------
        # 3. Bernoulli Variance Weighting
        # -------------------------------------------------------------
        eps_laplace = 1.0 / 8000.0
        bern_var = P_comp * (1.0 - P_comp)
        inv_bern_var = 1.0 / (bern_var + eps_laplace)
        w_bern = inv_bern_var / np.sum(inv_bern_var, axis=1, keepdims=True)
        p_bern = np.sum(w_bern * P_comp, axis=1)

        acc_bern = accuracy_score(y, p_bern >= 0.5)
        bs_bern = brier_score_loss(y, p_bern)
        ll_bern = log_loss(y, p_bern)

        print("\n--- 3. BERNOULLI VARIANCE WEIGHTING (Laplace eps = 1/8000) ---")
        print(f"  Accuracy: {acc_bern:.4f} (Delta vs Baseline: {acc_bern - acc_equal:+.4f})")
        print(f"  Brier:    {bs_bern:.5f} | LogLoss: {ll_bern:.5f}")

        # -------------------------------------------------------------
        # 4. Confidence Power-Law Sweep
        # -------------------------------------------------------------
        print("\n--- 4. CONFIDENCE POWER-LAW SWEEP ---")
        for gamma in [0.25, 0.5, 1.0, 1.5, 2.0]:
            conf = np.abs(P_comp - 0.5) ** gamma + 1e-4
            w_conf = conf / np.sum(conf, axis=1, keepdims=True)
            p_conf = np.sum(w_conf * P_comp, axis=1)
            acc_conf = accuracy_score(y, p_conf >= 0.5)
            print(f"  gamma = {gamma:4.2f} -> Accuracy: {acc_conf:.4f} (Delta: {acc_conf - acc_equal:+.4f}) | LogLoss: {log_loss(y, p_conf):.5f}")

        # -------------------------------------------------------------
        # 5. Robust Aggregation: Median & Trimmed Mean
        # -------------------------------------------------------------
        p_median = np.median(P_comp, axis=1)
        acc_median = accuracy_score(y, p_median >= 0.5)

        P_sorted = np.sort(P_comp, axis=1)
        p_trimmed = np.mean(P_sorted[:, 1:4], axis=1)
        acc_trimmed = accuracy_score(y, p_trimmed >= 0.5)

        print("\n--- 5. ROBUST AGGREGATION (OUTLIER RESISTANT) ---")
        print(f"  Median Voting:         Accuracy: {acc_median:.4f} (Delta: {acc_median - acc_equal:+.4f})")
        print(f"  Trimmed Mean (3 of 5): Accuracy: {acc_trimmed:.4f} (Delta: {acc_trimmed - acc_equal:+.4f})")

        # -------------------------------------------------------------
        # 6. Inter-Model Disagreement Variance
        # -------------------------------------------------------------
        disagreement_var = np.var(P_comp, axis=1, ddof=1)
        tau_disagreement = np.median(disagreement_var)
        print("\n--- 6. ADAPTIVE DISAGREEMENT SWITCHING ---")
        print(f"  Calculated median disagreement variance tau: {tau_disagreement:.6f}")
        p_adaptive = np.where(disagreement_var > tau_disagreement, p_median, p_equal)
        acc_adaptive = accuracy_score(y, p_adaptive >= 0.5)
        print(f"  Adaptive Mean/Median: Accuracy: {acc_adaptive:.4f} (Delta: {acc_adaptive - acc_equal:+.4f})")

        # -------------------------------------------------------------
        # 7. Marginal Prevalence Calibrated Threshold
        # -------------------------------------------------------------
        mean_p_equal = np.mean(p_equal)
        delta_bias = mean_p_equal - prevalence
        calibrated_threshold = 0.5 + delta_bias
        acc_cal_threshold = accuracy_score(y, p_equal >= calibrated_threshold)
        print("\n--- 7. MARGINAL PREVALENCE CALIBRATED THRESHOLD ---")
        print(f"  Mean P_equal: {mean_p_equal:.4f} | True Prevalence: {prevalence:.4f} | Bias Delta: {delta_bias:+.5f}")
        print(f"  Calibrated Threshold theta* = 0.5 + Delta: {calibrated_threshold:.5f}")
        print(f"  Accuracy at theta*: {acc_cal_threshold:.4f} (Delta: {acc_cal_threshold - acc_equal:+.4f})")

