"""
Linear regression baseline for the FaaS surrogate experiment.

Same outer 5-fold as ga_mlp_runs.py / cmaes_mlp_runs.py (shuffle seed 42).
Same inner train / medval / val cut, so val_per_line rows match those runners
when PMS_SEED is the same integer.

What this script does NOT copy from the neural search:
  - no architecture search
  - no log1p target and no MinMax scaling (raw S, N, M and raw T, so the
    coefficients stay in milliseconds per unit of each knob)
  - the 20% medval check is recorded in passed_gate, and it does not throw
    the fit away. There is only one straight line per fold.

Fit uses the inner train rows only. Medval only answers "was this line under
20% there?". Val is the per-line diagnostic. The outer 126 rows are the score.
"""
import os
import math
import datetime
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold


def _env_int(name, default):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_seed(name, default):
    """Return None (stochastic inner split) unless the env var is an integer."""
    return _env_int(name, default)


# ── CONFIG ───────────────────────────────────────────────────────────
DATA_FILE   = "master_dataset_NSMT.csv"   # swap this to run calibration variants
RESULTS_DIR = "linreg_results"
MODEL_TAG   = "NSMT"


K_FOLDS    = _env_int("PMS_FOLDS", 5)
KFOLD_SEED = 42


VAL_RATIO_OF_80    = 0.30
MEDVAL_RATIO_OF_80 = 0.20


# Same knob as the MLP runners. Set PMS_SEED=<int> so this val slice is the
# same rows as GA / CMA-ES. Unset = a fresh inner draw each run.
SEED = _env_seed("PMS_SEED", None)


MAPE_GATE = 20.0


LOG_FILE = f"experiment_logs_{MODEL_TAG}_linreg.txt"
# ─────────────────────────────────────────────────────────────────────




def safe_mape(y_true, y_pred):
    """MAPE with a small floor on the denominator to avoid div-by-zero."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom  = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def _smape(a, p):
    # Fraction in 0..2. Percent fields multiply by 100 downstream.
    a, p = np.asarray(a, float), np.asarray(p, float)
    num  = np.abs(a - p)
    den  = np.abs(a) + np.abs(p)
    mask = den > 0
    return float(np.mean(2.0 * num[mask] / den[mask])) if np.any(mask) else 0.0


def _mae(a, p):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))


def _rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot




def inner_split(X, y, seed):
    """Same 50/20/30 cut as ga_mlp_runs.inner_split. Kept for a future m-out twin."""
    X_tm, X_val, y_tm, y_val = train_test_split(
        X, y, test_size=VAL_RATIO_OF_80, random_state=seed
    )
    medval_frac_of_rest = MEDVAL_RATIO_OF_80 / (1.0 - VAL_RATIO_OF_80)
    X_train, X_medval, y_train, y_medval = train_test_split(
        X_tm, y_tm, test_size=medval_frac_of_rest, random_state=seed
    )
    return X_train, y_train, X_medval, y_medval, X_val, y_val


def inner_split_indices(train_index, seed):
    """Same split as inner_split. Returns global row indices into the dataset."""
    train_index = np.asarray(train_index)
    idx_tm, idx_val = train_test_split(
        train_index, test_size=VAL_RATIO_OF_80, random_state=seed
    )
    medval_frac_of_rest = MEDVAL_RATIO_OF_80 / (1.0 - VAL_RATIO_OF_80)
    idx_train, idx_medval = train_test_split(
        idx_tm, test_size=medval_frac_of_rest, random_state=seed
    )
    return idx_train, idx_medval, idx_val


def val_per_line_block(df, idx_val, val_preds, fold_idx, passed_gate):
    """Per-row val diagnostics. Same columns as ga_mlp_runs / cmaes_mlp_runs."""
    idx_val = np.asarray(idx_val)
    block = df.iloc[idx_val].copy()
    val_preds = np.maximum(0.0, np.asarray(val_preds, float))
    t_true = block.iloc[:, -1].values.astype(float)
    block["prediction"] = val_preds
    abs_err = np.abs(t_true - val_preds)
    block["abs_err"] = abs_err
    block["abs_pct_err"] = abs_err / np.maximum(np.abs(t_true), 1e-8) * 100.0
    block["signed_pct_err"] = (
        (val_preds - t_true) / np.maximum(np.abs(t_true), 1e-8) * 100.0
    )
    s = block.iloc[:, 0].values.astype(float)
    n = block.iloc[:, 1].values.astype(float)
    m = block.iloc[:, 2].values.astype(float)
    k = np.ceil(n / np.maximum(s, 1e-8))
    block["K"] = k
    block["regime_sat"] = (k > m).astype(int)
    block["fold"] = int(fold_idx)
    block["row_id"] = idx_val.astype(int)
    block["passed_gate"] = int(passed_gate)
    return block


def input_names(num_inputs):
    names = []
    for i in range(num_inputs):
        names.append(["S", "N", "M"][i] if i < 3 else f"x{i}")
    return names


def val_per_line_column_names(num_inputs):
    return input_names(num_inputs) + [
        "T",
        "prediction",
        "abs_err",
        "abs_pct_err",
        "signed_pct_err",
        "K",
        "regime_sat",
        "fold",
        "row_id",
        "passed_gate",
    ]


def _require_disjoint(a, b, what):
    overlap = set(np.asarray(a).tolist()) & set(np.asarray(b).tolist())
    if overlap:
        raise RuntimeError(f"{what}: {len(overlap)} shared rows")




def run_linreg_kfold():
    if SEED is not None:
        np.random.seed(SEED)

    df = pd.read_csv(DATA_FILE, header=None)
    X  = df.iloc[:, :-1].values.astype(float)
    y  = df.iloc[:,  -1].values.astype(float)
    num_inputs = X.shape[1]
    names = input_names(num_inputs)

    print(f"Dataset  : {DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"K-folds  : {K_FOLDS}-fold CV (seed {KFOLD_SEED})  |  tag: {MODEL_TAG}")
    print("Search   : none (ordinary least squares, one line per fold)")
    print("Target   : raw last column, no log. Inputs raw, no MinMax.")
    print("Protocol : same outer folds and same inner train/medval/val cut as "
          "ga_mlp_runs.py. The line is fit on train only. Val is the per-line "
          "file. The held-out 20% is the reported MAPE.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=KFOLD_SEED)

    fold_results = []
    all_actuals = []
    all_preds = []
    heldout_parts = []
    val_parts = []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print("\n" + "=" * 70)
        print(f"  FOLD {fold_idx}/{K_FOLDS}")
        print("=" * 70)

        X_test_fold = X[test_index]
        y_test_fold = y[test_index]
        idx_train, idx_medval, idx_val = inner_split_indices(train_index, SEED)
        _require_disjoint(idx_train, test_index, "train/test")
        _require_disjoint(idx_medval, test_index, "medval/test")
        _require_disjoint(idx_val, test_index, "val/test")
        _require_disjoint(idx_train, idx_val, "train/val")
        _require_disjoint(idx_train, idx_medval, "train/medval")
        _require_disjoint(idx_medval, idx_val, "medval/val")

        X_train, y_train = X[idx_train], y[idx_train]
        X_medval, y_medval = X[idx_medval], y[idx_medval]
        X_val, y_val = X[idx_val], y[idx_val]

        print(f"  [LINREG] Fitting on {len(y_train)} train rows "
              f"(medval {len(y_medval)}, val {len(y_val)}, test {len(y_test_fold)})...")
        try:
            model = LinearRegression(fit_intercept=True)
            model.fit(X_train, y_train)
        except Exception as exc:
            print(f"  WARNING: fit failed this fold ({exc}); skipping.")
            continue

        coef_txt = ", ".join(
            f"{name}={coef:.6g}" for name, coef in zip(names, model.coef_)
        )
        print(f"  Line: T ≈ {model.intercept_:.6g} + {coef_txt}")

        med_mape = safe_mape(y_medval, model.predict(X_medval))
        passed_gate = int(med_mape < MAPE_GATE)
        print(f"  Medval MAPE (not used to fit): {med_mape:.2f}%  "
              f"passed_gate={passed_gate}")

        val_preds = model.predict(X_val)
        val_parts.append(
            val_per_line_block(
                df, idx_val, val_preds, fold_idx, passed_gate=passed_gate
            )
        )

        preds = np.maximum(0.0, model.predict(X_test_fold))
        mape_val  = safe_mape(y_test_fold, preds)
        smape_val = _smape(y_test_fold, preds)
        mae_val   = _mae(y_test_fold, preds)
        rmse_val  = _rmse(y_test_fold, preds)
        r2_val    = safe_r2(y_test_fold, preds)
        n_test    = len(y_test_fold)
        signed_pct = (preds - y_test_fold) / np.maximum(np.abs(y_test_fold), 1e-8) * 100
        bad_count  = int(np.sum(np.abs(signed_pct) > 200))

        print(f"\n  External MAPE  (fold {fold_idx}, {n_test} rows): {mape_val:.2f}%")
        print(f"  External SMAPE (fold {fold_idx}, {n_test} rows): {smape_val:.2%}")
        print(f"  External MAE   (fold {fold_idx}, {n_test} rows): {mae_val:.1f} ms")
        print(f"  External RMSE  (fold {fold_idx}, {n_test} rows): {rmse_val:.1f} ms")
        print(f"  External R2    (fold {fold_idx}, {n_test} rows): {r2_val:.4f}")
        print(f"  Bad cases (|err|>200%)         : {bad_count}")

        fold_results.append({
            "fold":        fold_idx,
            "intercept":   float(model.intercept_),
            "coefficients": coef_txt,
            "medval_mape": med_mape,
            "passed_gate": passed_gate,
            "mape":        mape_val,
            "smape":       smape_val,
            "mae":         mae_val,
            "rmse":        rmse_val,
            "r2":          r2_val,
            "n_test":      n_test,
            "bad_count":   bad_count,
        })
        all_actuals.extend(y_test_fold.tolist())
        all_preds.extend(preds.tolist())

        block = df.iloc[test_index].copy()
        block["prediction"] = preds
        block["fold"] = fold_idx
        heldout_parts.append(block)

    if not fold_results:
        print("No successful folds — nothing to report.")
        return

    T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
              5: 2.571,  7: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}

    def _t_crit(n):
        return T_CRIT.get(n - 1, 1.96)

    def _avg_std_ci(key):
        vals = [r[key] for r in fold_results]
        n    = len(vals)
        avg  = float(np.mean(vals))
        std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci   = _t_crit(n) * std / math.sqrt(n) if n > 1 else 0.0
        return avg, std, ci

    avg_mape,  std_mape,  ci_mape  = _avg_std_ci("mape")
    avg_smape, std_smape, ci_smape = _avg_std_ci("smape")
    avg_mae,   std_mae,   ci_mae   = _avg_std_ci("mae")
    avg_rmse,  std_rmse,  ci_rmse  = _avg_std_ci("rmse")
    avg_r2,    std_r2,    ci_r2    = _avg_std_ci("r2")

    all_actuals = np.asarray(all_actuals, dtype=float)
    all_preds   = np.asarray(all_preds,   dtype=float)
    n_pool      = len(all_actuals)
    pool_mape   = safe_mape(all_actuals, all_preds)
    pool_smape  = _smape(all_actuals, all_preds)
    pool_mae    = _mae(all_actuals, all_preds)
    pool_rmse   = _rmse(all_actuals, all_preds)
    pool_r2     = safe_r2(all_actuals, all_preds)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fold_lines = "\n".join(
        f"  Fold {r['fold']}: intercept={r['intercept']:.6g}  {r['coefficients']}  "
        f"medvalMAPE={r['medval_mape']:.2f}%  gate={r['passed_gate']}  "
        f"MAPE={r['mape']:.2f}%  SMAPE={r['smape']:.2%}  "
        f"MAE={r['mae']:.1f}ms  RMSE={r['rmse']:.1f}ms  R2={r['r2']:.4f}  "
        f"n={r['n_test']}  bad={r['bad_count']}"
        for r in fold_results
    )

    def _fmt(avg, std, ci, suffix=""):
        return (f"{avg:.2f}{suffix}  "
                f"(std ± {std:.2f}{suffix}  |  95% CI ± {ci:.2f}{suffix}  "
                f"→ [{avg - ci:.2f}{suffix}, {avg + ci:.2f}{suffix}])")

    log_text = (
        f"\n{'=' * 70}\n"
        f"RUN TIMESTAMP : {now}\n"
        f"DATA FILE     : {DATA_FILE}\n"
        f"MODEL         : Ordinary least squares (sklearn LinearRegression)\n"
        f"TRAINER       : one straight line, fit on inner train only\n"
        f"CONFIG        : {num_inputs} inputs -> 1 target | "
        f"{K_FOLDS}-fold CV (seed {KFOLD_SEED}) | tag: {MODEL_TAG}\n"
        f"SCALING       : none. Raw inputs, raw target. No log1p.\n"
        f"PROTOCOL      : same outer 5-fold (seed {KFOLD_SEED}) and same inner "
        f"train/medval/val cut as ga_mlp_runs.py. Fit on train. Medval only sets "
        f"passed_gate (<{MAPE_GATE:.0f}%), it does not drop the line. Val is the "
        f"per-line file. The held-out 20% is the reported MAPE.\n"
        f"SEED          : {SEED}\n"
        f"METRICS       : per-fold, POOLED over all rows, AND mean ± 95% t-CI across "
        f"folds | MAPE in %, SMAPE on 0..200% scale\n"
        f"{'─' * 70}\n"
        f"{fold_lines}\n"
        f"{'─' * 70}\n"
        f"POOLED over all {n_pool} rows:\n"
        f"  POOL MAPE : {pool_mape:.2f}%\n"
        f"  POOL SMAPE: {pool_smape * 100:.2f}%\n"
        f"  POOL MAE  : {pool_mae:.1f} ms\n"
        f"  POOL RMSE : {pool_rmse:.1f} ms\n"
        f"  POOL R2   : {pool_r2:.4f}\n"
        f"{'─' * 70}\n"
        f"MEAN ± 95% t-CI ACROSS {len(fold_results)} FOLDS:\n"
        f"AVG MAPE  : {_fmt(avg_mape,  std_mape,  ci_mape,  '%')}\n"
        f"AVG SMAPE : {_fmt(avg_smape * 100, std_smape * 100, ci_smape * 100, '%')}\n"
        f"AVG MAE   : {_fmt(avg_mae,   std_mae,   ci_mae,   ' ms')}\n"
        f"AVG RMSE  : {_fmt(avg_rmse,  std_rmse,  ci_rmse,  ' ms')}\n"
        f"AVG R2    : {avg_r2:.4f}  (std ± {std_r2:.4f}  |  95% CI ± {ci_r2:.4f}  "
        f"→ [{avg_r2 - ci_r2:.4f}, {avg_r2 + ci_r2:.4f}])\n"
        f"{'=' * 70}\n"
    )
    print(log_text)
    with open(LOG_FILE, "a") as f:
        f.write(log_text)

    pd.DataFrame(fold_results).to_csv(
        os.path.join(RESULTS_DIR, f"kfold_results_{MODEL_TAG}_linreg.csv"),
        index=False
    )
    if heldout_parts:
        held_path = os.path.join(
            RESULTS_DIR, f"heldout_predictions_{MODEL_TAG}_linreg.csv"
        )
        pd.concat(heldout_parts, ignore_index=True).to_csv(
            held_path, index=False, header=False
        )
        print(f"Held-out predictions : {held_path}")
    if val_parts:
        val_path = os.path.join(
            RESULTS_DIR, f"val_per_line_{MODEL_TAG}_linreg.csv"
        )
        val_df = pd.concat(val_parts, ignore_index=True)
        val_df.columns = val_per_line_column_names(num_inputs)
        val_df.to_csv(val_path, index=False)
        print(f"Val per-line errors   : {val_path}")
    print(f"Log : {LOG_FILE}")
    print(f"CSV : {os.path.join(RESULTS_DIR, f'kfold_results_{MODEL_TAG}_linreg.csv')}")


if __name__ == "__main__":
    run_linreg_kfold()
