import os
import math
import datetime
import numpy as np
import pandas as pd
import cma
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# ── ENV OVERRIDES (optional) ─────────────────────────────────────────
# Small helpers so a fast SMOKE RUN is possible without editing this file.
# Every override defaults to the value baked in below, so leaving the env
# vars unset reproduces the original behavior exactly.
def _env_int(name, default):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_seed(name, default):
    """Return None (stochastic) unless PMS_SEED is set to an integer."""
    return _env_int(name, default)

# ── CONFIG ───────────────────────────────────────────────────────────
DATA_FILE  = "master_dataset.csv"   # swap this to run calibration variants
RESULTS_DIR = "cmaes_results"
MODEL_TAG  = "NSMT"                 # label for this run  e.g. NSMT, NSMKT, etc.

N_FOLDS     = _env_int("PMS_FOLDS", 5)
                     # TRUE k-fold cross-validation (KFOLD=5), like pms_kfold.py.
                     # The 630 rows are cut into 5 disjoint folds; each fold is
                     # the test set exactly once, so every row is tested once.
                     # (Overridable via PMS_FOLDS for a fast smoke run.)
KFOLD_SEED  = 42     # fixed seed → reproducible folds (matches pms_kfold.py)
VAL_RATIO   = 0.20   # 20% of the 4 training folds used as CMA-ES fitness
                     # signal (= PMS medval).  This is the inner model-selection
                     # split, nested inside each outer fold.

# Reproducibility seed for the STOCHASTIC parts (inner val split, MLP training,
# and CMA-ES sampling).  Default None → non-deterministic, like the Octave GA.
# Set PMS_SEED=<int> to make a full run reproducible.
SEED        = _env_seed("PMS_SEED", None)

# Architecture search space — same ceiling as PMS best (NLmax=5, NperLmax=55)
# CMA-ES searches TWO continuous variables in [0, 1]:
#   x[0] → number of hidden layers : decoded as round(1 + x[0] * 4)  → 1..5
#   x[1] → neurons per layer       : decoded as round(1 + x[1] * 54) → 1..55
# All layers share the same neuron count (simplification vs PMS per-layer encoding).
# Activation fixed to 'relu' (standard Python equivalent of PMS tansig/logsig mix).

CMA_MAX_EVAL = _env_int("PMS_MAXEVAL", 200)
                     # CMA-ES evaluation budget per run.
                     # PMS uses pop=30 × gen=30 = 900 evals, but CMA-ES is
                     # more sample-efficient — 200 is sufficient for 2D search.
                     # (Overridable via PMS_MAXEVAL for a fast smoke run.)

# Stopping threshold: same concept as PMS "medval MAPE < 20% gate"
# If the best MAPE found by CMA-ES never drops below this, note it in the log.
MAPE_GATE = 20.0     # percent

LOG_FILE = f"experiment_logs_{MODEL_TAG}_cmaes.txt"
# ─────────────────────────────────────────────────────────────────────


# ── METRICS  (identical to pms_runs.py) ──────────────────────────────

def safe_mape(y_true, y_pred):
    """MAPE with a small floor on the denominator to avoid div-by-zero."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom  = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)   # in percent

def _smape(a, p):
    # NOTE: this returns the fraction 2|a-p|/(|a|+|p|), i.e. a 0..2 range that
    # is reported as 0..200% (the "%" fields multiply by 100 downstream).
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
    """
    Coefficient of determination R^2 = 1 - SS_res / SS_tot.
    "Fraction of T's variance explained" (master doc Section 12), computed on
    the same clamped test predictions as the other metrics.
    - NOT clamped: a negative R^2 (worse than predicting the mean) is returned
      as-is, on purpose.
    - Guarded: if the test targets have zero variance (SS_tot == 0) the value is
      undefined, so we return 0.0 rather than dividing by zero.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot

# ─────────────────────────────────────────────────────────────────────


# ── ARCHITECTURE DECODE ───────────────────────────────────────────────
# CMA-ES works in continuous space.  We decode its output to valid integers.

def decode_architecture(x):
    """
    x is a 2-element list from CMA-ES, each in roughly [0, 1].
    Returns (n_layers, neurons_per_layer) as integers.
    """
    n_layers = int(np.clip(round(1 + float(x[0]) * 4), 1, 5))
    neurons  = int(np.clip(round(1 + float(x[1]) * 54), 1, 55))
    return n_layers, neurons

def build_mlp(n_layers, neurons):
    """
    Build a scaled MLP with the given architecture.

    Octave PMS normalizes every input AND the target to [-1, 1] via
    normalize.m before training, because neural nets train poorly on raw,
    differently-scaled features (here M~32 vs N~10000 vs T~52000).
    We reproduce that exactly:
      - MinMaxScaler([-1, 1]) on the inputs   (inside the pipeline)
      - MinMaxScaler([-1, 1]) on the target   (TransformedTargetRegressor)
    Predictions are automatically inverse-transformed back to real T units,
    so the rest of the script (clamp, metrics) is unchanged.
    """
    net = MLPRegressor(
        hidden_layer_sizes = (neurons,) * n_layers,  # e.g. (30, 30, 30) for 3 layers
        activation         = "relu",
        solver             = "adam",
        max_iter           = 2000,
        random_state       = SEED,    # None → non-deterministic (like PMS GA)
        early_stopping     = True,    # holds back 10% internally for early stop
        n_iter_no_change   = 20,
    )
    pipe = Pipeline([
        ("scale_x", MinMaxScaler(feature_range=(-1, 1))),
        ("mlp",     net),
    ])
    return TransformedTargetRegressor(
        regressor   = pipe,
        transformer = MinMaxScaler(feature_range=(-1, 1)),   # scales target T
    )

# ─────────────────────────────────────────────────────────────────────


# ── CMA-ES ARCHITECTURE SEARCH ────────────────────────────────────────

def cmaes_search(X_train, y_train):
    """
    Run CMA-ES to find the best (n_layers, neurons) architecture.

    Fitness function: MAPE on an inner validation split carved from X_train.
    This mirrors PMS's medval set — data the GA evaluates candidates on,
    but that is NOT the final test set.

    Returns (best_n_layers, best_neurons, best_val_mape).
    """
    # Inner split: 80% for fitting, 20% for CMA-ES fitness
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_RATIO, random_state=SEED
    )

    def fitness(x):
        n_layers, neurons = decode_architecture(x)
        model = build_mlp(n_layers, neurons)
        try:
            model.fit(X_fit, y_fit)
            preds = model.predict(X_val)
            return safe_mape(y_val, preds)   # CMA-ES minimises → MAPE is perfect
        except Exception:
            return 999.0   # penalise failed fits heavily

    # Start in the middle of the search space: 3 layers, 30 neurons
    x0     = [0.5, 0.5]
    sigma0 = 0.3          # initial step size in normalised [0,1] space

    opts = {
        "maxfevals":  CMA_MAX_EVAL,
        "bounds":     [[0.0, 0.0], [1.0, 1.0]],
        "verbose":    -9,   # silent
        "tolx":       1e-4,
        "tolfun":     1e-4,
    }
    if SEED is not None:
        opts["seed"] = SEED   # make CMA-ES sampling reproducible when seeded

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        solutions  = es.ask()
        fitnesses  = [fitness(x) for x in solutions]
        es.tell(solutions, fitnesses)

    best_x                = es.result.xbest
    best_n_layers, best_neurons = decode_architecture(best_x)
    best_val_mape         = es.result.fbest

    return best_n_layers, best_neurons, best_val_mape

# ─────────────────────────────────────────────────────────────────────


# ── MAIN LOOP ─────────────────────────────────────────────────────────

def run_cmaes_repeated():

    # Seed numpy for reproducibility of the stochastic parts when SEED is set;
    # otherwise stay non-deterministic like the Octave GA.
    if SEED is not None:
        np.random.seed(SEED)

    df = pd.read_csv(DATA_FILE, header=None)
    X  = df.iloc[:, :-1].values.astype(float)   # all columns except last = inputs
    y  = df.iloc[:,  -1].values.astype(float)   # last column = target T

    num_inputs = X.shape[1]
    print(f"Dataset  : {DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"Folds    : {N_FOLDS}-fold CV  |  tag: {MODEL_TAG}")
    print(f"Search   : CMA-ES  max_eval={CMA_MAX_EVAL} per fold")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_results = []
    all_actuals = []   # true T for every test row, pooled across folds
    all_preds   = []   # clamped prediction for every test row, pooled across folds

    # TRUE k-fold: split the rows into N_FOLDS disjoint folds.  Each fold is
    # the test set exactly once; the other 4 folds are training.  This covers
    # 100% of the rows with NO overlap between test sets (unlike a repeated
    # random holdout).  shuffle + fixed seed matches pms_kfold.py.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=KFOLD_SEED)

    for run_idx, (train_idx, test_idx) in enumerate(kf.split(X), start=1):
        print("\n" + "=" * 70)
        print(f"  FOLD {run_idx}/{N_FOLDS}")
        print("=" * 70)

        # ── outer fold: 4 folds train (80%) / 1 fold test (20%) ───────
        # The test fold is never seen during search, training, or selection
        # (= PMS finalval, but now guaranteed disjoint across folds).
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ── CMA-ES: search for best architecture on train portion ─────
        print("  [CMA-ES] Searching architecture...")
        best_layers, best_neurons, val_mape = cmaes_search(X_train, y_train)
        print(f"  Best architecture found: {best_layers} layers × {best_neurons} neurons")
        print(f"  Best val MAPE (inner)  : {val_mape:.2f}%")

        if val_mape > MAPE_GATE:
            print(f"  NOTE: val MAPE {val_mape:.2f}% exceeds gate {MAPE_GATE}% "
                  f"(PMS would not archive this model)")

        # ── Final model: retrain on FULL 80% train with best arch ─────
        # (PMS equivalent: pick best net from bestnets/, evaluate on finalval)
        print("  [MLP] Training final model on full train set...")
        final_model = build_mlp(best_layers, best_neurons)
        final_model.fit(X_train, y_train)

        # ── Evaluate on the 20% hold-out (= PMS finalval) ─────────────
        preds = final_model.predict(X_test)
        preds = np.maximum(0.0, preds)   # clamp negatives, same as pms_runs.py

        mape_val  = safe_mape(y_test, preds)
        smape_val = _smape(y_test, preds)
        mae_val   = _mae(y_test, preds)
        rmse_val  = _rmse(y_test, preds)
        r2_val    = safe_r2(y_test, preds)
        n_test    = len(y_test)

        # ── count "bad" cases (|signed%err| > 200%, same threshold as PMS) ─
        signed_pct = (preds - y_test) / np.maximum(np.abs(y_test), 1e-8) * 100
        bad_count  = int(np.sum(np.abs(signed_pct) > 200))

        print(f"\n  MAPE   (test, {n_test} rows): {mape_val:.2f}%")
        print(f"  SMAPE  (test, {n_test} rows): {smape_val:.2%}")
        print(f"  MAE    (test, {n_test} rows): {mae_val:.1f} ms")
        print(f"  RMSE   (test, {n_test} rows): {rmse_val:.1f} ms")
        print(f"  R2     (test, {n_test} rows): {r2_val:.4f}")
        print(f"  Bad cases (|err|>200%)      : {bad_count}")

        run_results.append({
            "run":          run_idx,
            "n_layers":     best_layers,
            "neurons":      best_neurons,
            "val_mape":     val_mape,
            "mape":         mape_val,
            "smape":        smape_val,
            "mae":          mae_val,
            "rmse":         rmse_val,
            "r2":           r2_val,
            "n_test":       n_test,
            "bad_count":    bad_count,
        })

        # accumulate raw test rows for the POOLED-over-all-rows metrics
        all_actuals.extend(y_test.tolist())
        all_preds.extend(preds.tolist())

    # ── per-fold mean ± 95% t-CI — identical CI logic to pms_runs.py ──
    T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
              5: 2.571,  7: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}

    def _t_crit(n):
        df = n - 1
        return T_CRIT.get(df, 1.96)

    def _avg_std_ci(key):
        vals = [r[key] for r in run_results]
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

    # ── POOLED over all rows (matches pms_kfold's aggregation) ──
    all_actuals = np.asarray(all_actuals, dtype=float)
    all_preds   = np.asarray(all_preds,   dtype=float)
    n_pool      = len(all_actuals)
    pool_mape   = safe_mape(all_actuals, all_preds)   # already in percent
    pool_smape  = _smape(all_actuals, all_preds)      # fraction (×100 to show %)
    pool_mae    = _mae(all_actuals, all_preds)
    pool_rmse   = _rmse(all_actuals, all_preds)
    pool_r2     = safe_r2(all_actuals, all_preds)

    # ── build log (same format as pms_runs.py) ────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_lines = "\n".join(
        f"  Fold {r['run']}: "
        f"arch={r['n_layers']}L×{r['neurons']}N  "
        f"val_MAPE={r['val_mape']:.2f}%  "
        f"MAPE={r['mape']:.2f}%  SMAPE={r['smape']:.2%}  "
        f"MAE={r['mae']:.1f}ms  RMSE={r['rmse']:.1f}ms  R2={r['r2']:.4f}  "
        f"n={r['n_test']}  bad={r['bad_count']}"
        for r in run_results
    )

    def _fmt(avg, std, ci, suffix=""):
        return (f"{avg:.2f}{suffix}  "
                f"(std ± {std:.2f}{suffix}  |  95% CI ± {ci:.2f}{suffix}  "
                f"→ [{avg - ci:.2f}{suffix}, {avg + ci:.2f}{suffix}])")

    log_text = (
        f"\n{'=' * 70}\n"
        f"RUN TIMESTAMP : {now}\n"
        f"DATA FILE     : {DATA_FILE}\n"
        f"MODEL         : CMA-ES + MLPRegressor (sklearn)\n"
        f"CONFIG        : {num_inputs} inputs -> 1 target | "
        f"{N_FOLDS}-fold cross-validation | tag: {MODEL_TAG}\n"
        f"SEARCH SPACE  : layers [1,5], neurons/layer [1,55], activation=relu\n"
        f"SCALING       : inputs + target MinMax[-1,1] (matches Octave normalize.m)\n"
        f"CMA-ES BUDGET : {CMA_MAX_EVAL} evaluations per fold\n"
        f"SPLIT         : {N_FOLDS} disjoint folds | per fold: train "
        f"(20% inner val for CMA-ES) / 1 fold test | fold seed={KFOLD_SEED} | "
        f"stochastic seed={SEED}\n"
        f"METRICS       : per-fold, mean ± 95% t-CI across folds, AND pooled "
        f"over all rows (matches pms_kfold) | MAPE in %, SMAPE on 0..200% scale\n"
        f"CONFOUNDS     : this run used {CMA_MAX_EVAL} evals/fold. At default "
        f"budgets CMA-ES uses 200 evals/fold vs the GA's 900 — an eval-budget "
        f"confound when attributing differences to the search paradigm. Also, "
        f"CMA-ES searches a single uniform width while the GA searches per-layer "
        f"widths (architecture-space asymmetry)\n"
        f"{'─' * 70}\n"
        f"{run_lines}\n"
        f"{'─' * 70}\n"
        f"AVG MAPE  : {_fmt(avg_mape,  std_mape,  ci_mape,  '%')}\n"
        f"AVG SMAPE : {_fmt(avg_smape * 100, std_smape * 100, ci_smape * 100, '%')}\n"
        f"AVG MAE   : {_fmt(avg_mae,   std_mae,   ci_mae,   ' ms')}\n"
        f"AVG RMSE  : {_fmt(avg_rmse,  std_rmse,  ci_rmse,  ' ms')}\n"
        f"AVG R2    : {avg_r2:.4f}  (std ± {std_r2:.4f}  |  95% CI ± {ci_r2:.4f}  "
        f"→ [{avg_r2 - ci_r2:.4f}, {avg_r2 + ci_r2:.4f}])\n"
        f"{'─' * 70}\n"
        f"POOLED over all {n_pool} rows (pms_kfold-style aggregation):\n"
        f"POOL MAPE : {pool_mape:.2f}%\n"
        f"POOL SMAPE: {pool_smape * 100:.2f}%\n"
        f"POOL MAE  : {pool_mae:.1f} ms\n"
        f"POOL RMSE : {pool_rmse:.1f} ms\n"
        f"POOL R2   : {pool_r2:.4f}\n"
        f"{'=' * 70}\n"
    )

    print(log_text)

    with open(LOG_FILE, "a") as f:
        f.write(log_text)

    pd.DataFrame(run_results).to_csv(
        os.path.join(RESULTS_DIR, f"run_results_{MODEL_TAG}_cmaes.csv"),
        index=False
    )

    print(f"Log : {LOG_FILE}")
    print(f"CSV : {os.path.join(RESULTS_DIR, f'run_results_{MODEL_TAG}_cmaes.csv')}")


# ── ENTRY POINT ───────────────────────────────────────────────────────
if __name__ == "__main__":
    run_cmaes_repeated()
