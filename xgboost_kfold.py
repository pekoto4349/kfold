"""
XGBoost baseline for the FaaS surrogate experiment.

Same outer 5-fold as ga_mlp_runs.py / cmaes_mlp_runs.py (shuffle seed 42).
Same inner train / medval / val cut, so val_per_line rows match those runners
when PMS_SEED is the same integer.

What this script does NOT copy from the neural search:
  - no architecture search and no 900 network trains
  - fixed tree settings (the v1 defaults below), not an Optuna search
  - the 20% medval check is recorded in passed_gate, and it does not throw
    the model away. There is only one booster per fold.

What it copies from ga_mlp_runs.py, including the time preparation:
  - same outer folds and same inner train / medval / val cut
  - log1p on the time, then scale the inputs and that logged time to [-1, 1]
  - those scales are fit on the training rows only
  - guesses are turned back into milliseconds before any error number
  - negatives are clamped to 0

Fit uses the inner train rows. Early stopping watches RMSE on medval in that
compressed space and chooses how many trees to keep. Val is not used for
fitting or stopping. The outer 126 rows are the score. The reported number
is still MAPE on those rows, in real milliseconds.
"""
import os
import math
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise SystemExit(
        "xgboost is not installed. Install it with: pip install xgboost"
    ) from exc


def _env_int(name, default):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name, default):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_seed(name, default):
    """Return None (stochastic inner split) unless the env var is an integer."""
    return _env_int(name, default)


# ── CONFIG ───────────────────────────────────────────────────────────
DATA_FILE   = "master_dataset_NSMT.csv"   # swap this to run calibration variants
RESULTS_DIR = "xgb_results"
MODEL_TAG   = "NSMT"


K_FOLDS    = _env_int("PMS_FOLDS", 5)
KFOLD_SEED = 42


VAL_RATIO_OF_80    = 0.30
MEDVAL_RATIO_OF_80 = 0.20


SEED = _env_seed("PMS_SEED", None)


MAPE_GATE = 20.0


# Fixed v1 tree settings. Early stopping usually halts before the cap.
N_ESTIMATORS = _env_int("PMS_XGB_TREES", 500)
MAX_DEPTH    = _env_int("PMS_XGB_DEPTH", 3)
EARLY_STOP   = _env_int("PMS_XGB_ROUNDS", 30)
LEARNING_RATE = _env_float("PMS_XGB_LR", 0.05)
SUBSAMPLE     = 0.8
COLSAMPLE     = 1.0
REG_LAMBDA    = 1.0


LOG_FILE = f"experiment_logs_{MODEL_TAG}_xgb.txt"
# ─────────────────────────────────────────────────────────────────────




def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom  = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def _smape(a, p):
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


class SameTimePrep:
    """log1p then MinMax [-1, 1], fit on the training rows only.

    This is the same preparation as ga_mlp_runs.build_mlp. The trees learn
    the compressed time. inverse_y turns a guess back into milliseconds.
    """

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.x_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)
        self.y_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(
            np.log1p(y).reshape(-1, 1)
        )
        back = self.inverse_y(self.transform_y(y))
        if not np.allclose(back, y, rtol=1e-5, atol=1e-3):
            raise RuntimeError("time preparation did not round-trip on the training rows")
        return self

    def transform_X(self, X):
        return self.x_scaler.transform(np.asarray(X, float))

    def transform_y(self, y):
        logged = np.log1p(np.asarray(y, float)).reshape(-1, 1)
        return self.y_scaler.transform(logged).ravel()

    def inverse_y(self, y_scaled):
        logged = self.y_scaler.inverse_transform(
            np.asarray(y_scaled, float).reshape(-1, 1)
        ).ravel()
        return np.expm1(logged)


def predict_ms(model, prep, X):
    """Guess in the compressed space, then milliseconds, then clamp at 0."""
    return np.maximum(0.0, prep.inverse_y(model.predict(prep.transform_X(X))))


def add_group_averages(df):
    """Repeat the group average on every job row so the per-line file has it.

    avg_pct_err_* is the average percent off (MAPE of that group).
    avg_ms_err_* is the average miss in milliseconds, which is not a percent.
    """
    out = df.copy()
    regime = np.where(out["regime_sat"].to_numpy() == 1, "saturated", "unsaturated")
    out["avg_pct_err_all"] = float(out["abs_pct_err"].mean())
    out["avg_ms_err_all"] = float(out["abs_err"].mean())
    keys = {
        "S": out["S"],
        "N": out["N"],
        "M": out["M"],
        "regime": pd.Series(regime, index=out.index),
    }
    for name, key in keys.items():
        out[f"avg_pct_err_same_{name}"] = out.groupby(key, sort=False)["abs_pct_err"].transform("mean")
        out[f"avg_ms_err_same_{name}"] = out.groupby(key, sort=False)["abs_err"].transform("mean")
    return out


def error_table(block, num_inputs):
    """One row per job, with the error columns and the group averages."""
    names = input_names(num_inputs)
    data = {}
    for i, name in enumerate(names):
        data[name] = block.iloc[:, i].to_numpy(float)
    data["T"] = block.iloc[:, num_inputs].to_numpy(float)
    data["prediction"] = np.asarray(block["prediction"], float)
    out = pd.DataFrame(data)
    out["abs_err"] = np.abs(out["T"] - out["prediction"])
    out["abs_pct_err"] = out["abs_err"] / np.maximum(np.abs(out["T"]), 1e-8) * 100.0
    out["signed_pct_err"] = (out["prediction"] - out["T"]) / np.maximum(np.abs(out["T"]), 1e-8) * 100.0
    k = np.ceil(out["N"] / np.maximum(out["S"], 1e-8))
    out["K"] = k
    out["regime_sat"] = (k > out["M"]).astype(int)
    out["fold"] = np.asarray(block["fold"], int)
    return add_group_averages(out)


def summarize_errors(df_rows, set_name, note):
    """One row per group. mape_pct is the mean of abs_pct_err for that group."""
    work = df_rows.copy()
    work["regime"] = np.where(work["regime_sat"].to_numpy() == 1, "saturated", "unsaturated")
    rows = []

    def one(group, value, sub):
        rows.append({
            "set": set_name,
            "group": group,
            "value": value,
            "n_rows": int(len(sub)),
            "mape_pct": float(sub["abs_pct_err"].mean()),
            "median_abs_pct_err": float(sub["abs_pct_err"].median()),
            "mean_abs_err_ms": float(sub["abs_err"].mean()),
            "median_abs_err_ms": float(sub["abs_err"].median()),
            "mean_signed_pct_err": float(sub["signed_pct_err"].mean()),
            "median_signed_pct_err": float(sub["signed_pct_err"].median()),
            "note": note,
        })

    one("all", "all", work)
    for regime, sub in work.groupby("regime", sort=True):
        one("regime", regime, sub)
    for col in ("S", "N", "M"):
        for val, sub in work.groupby(col, sort=True):
            shown = int(val) if float(val) == int(float(val)) else val
            one(col, shown, sub)
    return pd.DataFrame(rows)


def format_avg_lines(avg_df, set_name):
    lines = []
    part = avg_df[avg_df["set"] == set_name]
    for rec in part.itertuples(index=False):
        lines.append(
            f"  {rec.group}={rec.value}  n={rec.n_rows}  "
            f"MAPE={rec.mape_pct:.2f}%  "
            f"median_pct={rec.median_abs_pct_err:.2f}%  "
            f"mean_ms={rec.mean_abs_err_ms:.1f}  "
            f"mean_signed_pct={rec.mean_signed_pct_err:.2f}"
        )
    return "\n".join(lines)


LEGEND_ROWS = [
    {
        "name": "mape_pct",
        "is_this_mape": "yes",
        "meaning": (
            "Average percent off for that group. Always positive. "
            "This is MAPE. 10 means the jobs in the group were 10% off on average."
        ),
    },
    {
        "name": "median_abs_pct_err",
        "is_this_mape": "no",
        "meaning": (
            "Percent off of the middle job. Half the jobs are closer than this, "
            "half are worse. One disaster moves MAPE a lot and barely moves this."
        ),
    },
    {
        "name": "mean_abs_err_ms",
        "is_this_mape": "no",
        "meaning": (
            "Average miss in milliseconds. A 1000 ms miss counts the same on a "
            "short job and a long job. MAPE would call the short job much worse."
        ),
    },
    {
        "name": "median_abs_err_ms",
        "is_this_mape": "no",
        "meaning": "Miss in milliseconds of the middle job. Not a percent.",
    },
    {
        "name": "mean_signed_pct_err",
        "is_this_mape": "no",
        "meaning": (
            "Average percent with a sign. Positive means the guesses were too high. "
            "Too-high and too-low cancel, so this can look small when MAPE is large."
        ),
    },
    {
        "name": "median_signed_pct_err",
        "is_this_mape": "no",
        "meaning": "Signed percent of the middle job. Positive means that guess was too high.",
    },
    {
        "name": "abs_err",
        "is_this_mape": "no",
        "meaning": "One job, milliseconds off. The average of this column is mean_abs_err_ms.",
    },
    {
        "name": "abs_pct_err",
        "is_this_mape": "yes_for_one_job",
        "meaning": (
            "One job, percent off, always positive. "
            "The average of this column is mape_pct."
        ),
    },
    {
        "name": "signed_pct_err",
        "is_this_mape": "no",
        "meaning": (
            "One job, percent with a sign. Positive means this guess was too high. "
            "MAPE uses the absolute value so the sign is dropped."
        ),
    },
    {
        "name": "avg_pct_err_same_S / _N / _M / _regime",
        "is_this_mape": "yes_for_that_group",
        "meaning": (
            "Written on every per-line row. MAPE of the jobs that share that S, "
            "or that N, or that M, or that saturated/unsaturated label."
        ),
    },
    {
        "name": "avg_ms_err_same_S / _N / _M / _regime",
        "is_this_mape": "no",
        "meaning": (
            "Written on every per-line row. Average millisecond miss of the jobs "
            "that share that knob. Not a percent."
        ),
    },
]


def fit_xgb(X_train, y_train, X_medval, y_medval):
    """Train on compressed time. Stop when medval RMSE there has stalled."""
    params = dict(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE,
        reg_lambda=REG_LAMBDA,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=1,
    )
    try:
        model = XGBRegressor(**params, early_stopping_rounds=EARLY_STOP)
        model.fit(
            X_train, y_train,
            eval_set=[(X_medval, y_medval)],
            verbose=False,
        )
    except TypeError:
        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_medval, y_medval)],
            early_stopping_rounds=EARLY_STOP,
            verbose=False,
        )
    return model


def run_xgb_kfold():
    if SEED is not None:
        np.random.seed(SEED)

    df = pd.read_csv(DATA_FILE, header=None)
    X  = df.iloc[:, :-1].values.astype(float)
    y  = df.iloc[:,  -1].values.astype(float)
    num_inputs = X.shape[1]
    names = input_names(num_inputs)

    print(f"Dataset  : {DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"K-folds  : {K_FOLDS}-fold CV (seed {KFOLD_SEED})  |  tag: {MODEL_TAG}")
    print(f"Search   : none. Trees cap={N_ESTIMATORS}, depth={MAX_DEPTH}, "
          f"lr={LEARNING_RATE}, early_stop={EARLY_STOP} on medval RMSE")
    print("Target   : same as the nets. log of the time, then scale to [-1, 1],")
    print("           fit on the training rows only. Errors are in milliseconds")
    print("           after the guess is turned back.")
    print("Protocol : same outer folds and same inner train/medval/val cut as "
          "ga_mlp_runs.py. Val is the per-line file. The held-out 20% is the "
          "reported MAPE.")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=KFOLD_SEED)

    fold_results = []
    importance_rows = []
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

        print(f"  [XGB] Fitting on {len(y_train)} train rows, "
              f"early stop on {len(y_medval)} medval rows...")
        print("  Time prep: log, then scale to [-1, 1], fit on these train rows only.")
        prep = SameTimePrep().fit(X_train, y_train)
        try:
            model = fit_xgb(
                prep.transform_X(X_train), prep.transform_y(y_train),
                prep.transform_X(X_medval), prep.transform_y(y_medval),
            )
        except Exception as exc:
            print(f"  WARNING: fit failed this fold ({exc}); skipping.")
            continue

        best_iter = int(getattr(model, "best_iteration", N_ESTIMATORS - 1)) + 1
        print(f"  Trees kept: {best_iter} (cap {N_ESTIMATORS})")

        importances = np.asarray(model.feature_importances_, float)
        imp_txt = ", ".join(
            f"{name}={imp:.4f}" for name, imp in zip(names, importances)
        )
        print(f"  Importance (gain, sums to 1): {imp_txt}")
        for name, imp in zip(names, importances):
            importance_rows.append({
                "fold": fold_idx,
                "feature": name,
                "importance": float(imp),
            })

        med_mape = safe_mape(y_medval, predict_ms(model, prep, X_medval))
        passed_gate = int(med_mape < MAPE_GATE)
        print(f"  Medval MAPE (after early stopping, back in milliseconds): {med_mape:.2f}%  "
              f"passed_gate={passed_gate}")

        val_preds = predict_ms(model, prep, X_val)
        val_parts.append(
            val_per_line_block(
                df, idx_val, val_preds, fold_idx, passed_gate=passed_gate
            )
        )

        preds = predict_ms(model, prep, X_test_fold)
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
            "fold":         fold_idx,
            "trees_kept":   best_iter,
            "max_depth":    MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "medval_mape":  med_mape,
            "passed_gate":  passed_gate,
            "mape":         mape_val,
            "smape":        smape_val,
            "mae":          mae_val,
            "rmse":         rmse_val,
            "r2":           r2_val,
            "n_test":       n_test,
            "bad_count":    bad_count,
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

    held_detail = None
    val_detail = None
    avg_df = None
    if heldout_parts:
        held_detail = error_table(
            pd.concat(heldout_parts, ignore_index=True), num_inputs
        )
    if val_parts:
        val_named = pd.concat(val_parts, ignore_index=True)
        val_named.columns = val_per_line_column_names(num_inputs)
        val_detail = add_group_averages(val_named)
    avg_parts = []
    if held_detail is not None:
        avg_parts.append(summarize_errors(
            held_detail,
            "heldout_exam",
            "Exam. Each job once. mape_pct is the average percent error of the group.",
        ))
    if val_detail is not None:
        avg_parts.append(summarize_errors(
            val_detail,
            "val",
            "Not the exam. Some jobs appear in more than one fold.",
        ))
    if avg_parts:
        avg_df = pd.concat(avg_parts, ignore_index=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fold_lines = "\n".join(
        f"  Fold {r['fold']}: trees={r['trees_kept']} depth={r['max_depth']}  "
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
        f"MODEL         : XGBoost (XGBRegressor)\n"
        f"TRAINER       : gradient boosted trees, early stop on medval RMSE "
        f"in the compressed time\n"
        f"CONFIG        : {num_inputs} inputs -> 1 target | "
        f"{K_FOLDS}-fold CV (seed {KFOLD_SEED}) | tag: {MODEL_TAG}\n"
        f"TREES         : cap={N_ESTIMATORS} depth={MAX_DEPTH} lr={LEARNING_RATE} "
        f"subsample={SUBSAMPLE} colsample={COLSAMPLE} lambda={REG_LAMBDA} "
        f"early_stop={EARLY_STOP}\n"
        f"SCALING       : same as the nets. log1p(time), then MinMax [-1, 1] "
        f"on inputs and on the logged time. Fit on the training rows only. "
        f"Guesses are turned back into milliseconds before MAPE, sMAPE, MAE, "
        f"RMSE, and the per-line errors. Negatives clamped to 0.\n"
        f"PROTOCOL      : same outer 5-fold (seed {KFOLD_SEED}) and same inner "
        f"train/medval/val cut as ga_mlp_runs.py. Fit on train. Early stopping "
        f"watches medval RMSE in the compressed space and does not see val or "
        f"the outer test. passed_gate (<{MAPE_GATE:.0f}% medval MAPE, in "
        f"milliseconds) is recorded and does not drop the model. Val is the "
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
        f"{'─' * 70}\n"
        f"GROUP AVERAGES, held-out exam (each job once).\n"
        f"MAPE is the average percent off. mean_ms is the average millisecond\n"
        f"miss, which is not a percent. mean_signed_pct is too-high minus too-low,\n"
        f"so it is not MAPE. median_pct is the middle job.\n"
        f"{format_avg_lines(avg_df, 'heldout_exam') if avg_df is not None else '  (none)'}\n"
        f"{'─' * 70}\n"
        f"GROUP AVERAGES, val (not the exam, some jobs repeat).\n"
        f"{format_avg_lines(avg_df, 'val') if avg_df is not None else '  (none)'}\n"
        f"{'=' * 70}\n"
    )
    print(log_text)
    with open(LOG_FILE, "a") as f:
        f.write(log_text)

    pd.DataFrame(fold_results).to_csv(
        os.path.join(RESULTS_DIR, f"kfold_results_{MODEL_TAG}_xgb.csv"),
        index=False
    )
    if importance_rows:
        imp_path = os.path.join(RESULTS_DIR, f"importance_{MODEL_TAG}.csv")
        pd.DataFrame(importance_rows).to_csv(imp_path, index=False)
        print(f"Feature importance    : {imp_path}")
    if heldout_parts:
        held_path = os.path.join(
            RESULTS_DIR, f"heldout_predictions_{MODEL_TAG}_xgb.csv"
        )
        pd.concat(heldout_parts, ignore_index=True).to_csv(
            held_path, index=False, header=False
        )
        print(f"Held-out predictions : {held_path}")
    if held_detail is not None:
        detail_path = os.path.join(
            RESULTS_DIR, f"heldout_per_line_{MODEL_TAG}_xgb.csv"
        )
        held_detail.to_csv(detail_path, index=False)
        print(f"Held-out per-line     : {detail_path}")
    if val_detail is not None:
        val_path = os.path.join(
            RESULTS_DIR, f"val_per_line_{MODEL_TAG}_xgb.csv"
        )
        val_detail.to_csv(val_path, index=False)
        print(f"Val per-line errors   : {val_path}")
    if avg_df is not None:
        avg_path = os.path.join(RESULTS_DIR, f"error_averages_{MODEL_TAG}_xgb.csv")
        avg_df.to_csv(avg_path, index=False)
        print(f"Error averages        : {avg_path}")
        legend_path = os.path.join(RESULTS_DIR, f"error_legend_{MODEL_TAG}_xgb.csv")
        pd.DataFrame(LEGEND_ROWS).to_csv(legend_path, index=False)
        print(f"Error legend          : {legend_path}")
    print(f"Log : {LOG_FILE}")
    print(f"CSV : {os.path.join(RESULTS_DIR, f'kfold_results_{MODEL_TAG}_xgb.csv')}")


if __name__ == "__main__":
    run_xgb_kfold()
