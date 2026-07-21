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
# vars unset reproduces the default behavior exactly.
def _env_int(name, default):
    val = os.environ.get(name)
    if val is None or val.strip() == "":
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_seed(name, default):
    """Return None (stochastic) unless the env var is set to an integer."""
    return _env_int(name, default)

# ── CONFIG ───────────────────────────────────────────────────────────
DATA_FILE  = "master_dataset.csv"   # swap this to run calibration variants
RESULTS_DIR = "cmaes_results"
MODEL_TAG  = "NSMT"                 # label for this run  e.g. NSMT, NSMKT, etc.

# K-FOLD cross-validation, exactly like pms_kfold.py. Octave's ONE internal
# 80-20 split (createmodel.m line 49) is what a fold IS; KFold just repeats it
# 5 times over different partitions, so every row is the held-out 20% exactly
# once. There is NO nested "80-20 inside the 80":
#   * the fold's 20% held-out = Octave's finalval = the reported MAPE
#     (the "bottom-right MAPE" in the PMS graph, createmodel.m line 539-540)
#   * the fold's 80% is split ONLY into the sets the search needs, never into
#     another finalval.
K_FOLDS    = _env_int("PMS_FOLDS", 5)
KFOLD_SEED = 42   # matches pms_kfold.py -> identical folds -> directly comparable

# ── How the fold's 80% is split for the search (createmodel.m line 57) ─
#   train  (~50% of the 80%) -> the network is actually trained on this
#   medval (~20% of the 80%) -> SAVE GATE: keep a candidate only if medval MAPE < 20%
#   val    (~30% of the 80%) -> SELECT the best saved candidate on this
# The selected net is then scored on the fold's held-out 20% (the reported MAPE).
VAL_RATIO_OF_80    = 0.30   # createmodel.m: subset(trainver2,...,0.3)
MEDVAL_RATIO_OF_80 = 0.20   # createmodel.m: subset(trainver2,...,0.2)
# train = the remaining ~0.50 of the 80%.

# Reproducibility seed for the STOCHASTIC parts (the split, MLP init, and
# CMA-ES sampling). Default None -> non-deterministic, like the Octave GA.
# Set PMS_SEED=<int> to make a full run reproducible (all runs become identical).
SEED = _env_seed("PMS_SEED", None)

# Architecture search space. CMA-ES searches THREE continuous variables in [0, 1]:
#   x[0] -> number of hidden layers : decoded to [3, 4]  (matches Octave NLmax)
#   x[1] -> neurons per layer       : decoded to [1, 55]
#   x[2] -> activation              : decoded to an index into ACTIVATIONS
# All layers share the same neuron count (simplification vs the GA's per-layer
# encoding).
N_LAYERS_MIN, N_LAYERS_MAX = 3, 4
NEURONS_MIN,  NEURONS_MAX  = 1, 55

# Activation is now SEARCHED too, like Octave (whose GA evolves the transfer
# function). scikit-learn applies ONE activation to all hidden layers, so the
# search picks a single network-wide activation from these analogues of
# Octave's set:
#   "tanh"     ~ tansig       "logistic" ~ logsig
#   "identity" ~ purelin      "relu"     (bonus; Octave has no relu)
ACTIVATIONS = ["tanh", "logistic", "relu", "identity"]

CMA_MAX_EVAL = _env_int("PMS_MAXEVAL", 200)
                     # CMA-ES evaluation budget per fold. PMS uses 900, but
                     # CMA-ES is more sample-efficient — 200 suffices for a 2D
                     # search. (Overridable via PMS_MAXEVAL for a fast smoke run.)

MAPE_GATE = 20.0     # Octave's medval save-gate (nnscript.m: <0.20)

LOG_FILE = f"experiment_logs_{MODEL_TAG}_cmaes.txt"
# ─────────────────────────────────────────────────────────────────────


# ── METRICS  (identical to pms_runs.py / ga_mlp_runs.py) ─────────────

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
    """Coefficient of determination R^2 = 1 - SS_res / SS_tot."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot

# ─────────────────────────────────────────────────────────────────────


# ── SPLIT THE FOLD'S 80% INTO train / medval / val (NO extra finalval) ─

def inner_split(X, y, seed):
    """
    Split a fold's 80% training data into train / medval / val (~50/20/30),
    matching Octave's train/medval/val proportions (createmodel.m line 57).
    NO finalval is carved here: the fold's held-out 20% is the reported set
    (Octave's finalval / bottom-right graph MAPE role), so we never do an
    "80-20 inside the 80".
    Returns X_train, y_train, X_medval, y_medval, X_val, y_val.
    """
    X_tm, X_val, y_tm, y_val = train_test_split(
        X, y, test_size=VAL_RATIO_OF_80, random_state=seed
    )
    medval_frac_of_rest = MEDVAL_RATIO_OF_80 / (1.0 - VAL_RATIO_OF_80)
    X_train, X_medval, y_train, y_medval = train_test_split(
        X_tm, y_tm, test_size=medval_frac_of_rest, random_state=seed
    )
    return X_train, y_train, X_medval, y_medval, X_val, y_val

# ─────────────────────────────────────────────────────────────────────


# ── ARCHITECTURE DECODE / BUILD ───────────────────────────────────────
# CMA-ES works in continuous space; decode its output to valid integers.

def decode_architecture(x):
    """x is a 3-element vector in ~[0, 1]. Returns (n_layers, neurons, activation)."""
    n_layers = int(np.clip(round(N_LAYERS_MIN + float(x[0]) * (N_LAYERS_MAX - N_LAYERS_MIN)),
                           N_LAYERS_MIN, N_LAYERS_MAX))
    neurons  = int(np.clip(round(NEURONS_MIN + float(x[1]) * (NEURONS_MAX - NEURONS_MIN)),
                           NEURONS_MIN, NEURONS_MAX))
    act_idx  = int(np.clip(round(float(x[2]) * (len(ACTIVATIONS) - 1)),
                           0, len(ACTIVATIONS) - 1))
    return n_layers, neurons, ACTIVATIONS[act_idx]

# Octave PMS normalizes every input AND the target to [-1, 1] via normalize.m;
# reproduced with MinMaxScaler([-1, 1]) on inputs (pipeline) and target
# (TransformedTargetRegressor). The trainer matches Octave's Levenberg-Marquardt
# (net.trainFcn='trainlm') as closely as sklearn allows:
#   solver="lbfgs"  -> quasi-Newton, the closest sklearn analogue to LM.
#   activation      -> SEARCHED (one of ACTIVATIONS), matching Octave's evolved
#                      tansig/logsig/purelin transfer functions (+ relu).
# early_stopping is intentionally OFF (Octave has none; lbfgs ignores it).

def build_mlp(n_layers, neurons, activation):
    net = MLPRegressor(
        hidden_layer_sizes = (neurons,) * n_layers,
        activation         = activation,
        solver             = "lbfgs",
        max_iter           = 2000,
        random_state       = SEED,   # None -> non-deterministic, like the Octave GA
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


# ── CMA-ES ARCHITECTURE SEARCH (Octave-faithful) ──────────────────────

def cmaes_search(X_train, y_train, X_medval, y_medval, X_val, y_val):
    """
    Run CMA-ES to find the best (n_layers, neurons) architecture, faithfully
    to Octave:
      - objective minimised by CMA-ES = TRAINING MSE on `train` (nnscript.m: perf=thismse)
      - a candidate is SAVED only if its medval MAPE < MAPE_GATE (nnscript.m: <0.20)
      - after search, the best SAVED candidate is SELECTED on `val` (createmodel.m)
    Returns (best_model, best_n_layers, best_neurons, best_activation, num_saved).
    """
    saved    = []                       # list of (model, medval_mape, n_layers, neurons, act)
    fallback = {"model": None, "medval": float("inf"), "n_layers": None,
                "neurons": None, "act": None}

    def objective(x):
        n_layers, neurons, activation = decode_architecture(x)
        model = build_mlp(n_layers, neurons, activation)
        try:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            train_mse  = float(np.mean((train_pred - y_train) ** 2))   # <- CMA-ES objective
            med_pred   = model.predict(X_medval)
            med_mape   = safe_mape(y_medval, med_pred)
        except Exception:
            return 1e18                 # penalise failed fits
        if med_mape < MAPE_GATE:        # Octave save-gate
            saved.append((model, med_mape, n_layers, neurons, activation))
        if med_mape < fallback["medval"]:
            fallback["model"]    = model
            fallback["medval"]   = med_mape
            fallback["n_layers"] = n_layers
            fallback["neurons"]  = neurons
            fallback["act"]      = activation
        return train_mse

    x0     = [0.5, 0.5, 0.5]   # mid layers, mid neurons, mid activation
    sigma0 = 0.3               # initial step size in normalised [0,1] space

    opts = {
        "maxfevals":  CMA_MAX_EVAL,
        "bounds":     [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        "verbose":    -9,   # silent
        "tolx":       1e-4,
        "tolfun":     1e-4,
    }
    if SEED is not None:
        opts["seed"] = SEED

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

    # ── model selection on `val` among SAVED candidates ───────────────
    pool = saved if saved else (
        [(fallback["model"], fallback["medval"], fallback["n_layers"],
          fallback["neurons"], fallback["act"])]
        if fallback["model"] is not None else []
    )
    if not pool:
        return None, 0, 0, None, 0

    def val_mape_of(entry):
        return safe_mape(y_val, entry[0].predict(X_val))

    best_model, _, best_layers, best_neurons, best_act = min(pool, key=val_mape_of)
    return best_model, best_layers, best_neurons, best_act, len(saved)

# ─────────────────────────────────────────────────────────────────────


# ── MAIN LOOP ─────────────────────────────────────────────────────────

def run_cmaes_kfold():

    if SEED is not None:
        np.random.seed(SEED)

    df = pd.read_csv(DATA_FILE, header=None)
    X  = df.iloc[:, :-1].values.astype(float)   # all columns except last = inputs
    y  = df.iloc[:,  -1].values.astype(float)   # last column = target T

    num_inputs = X.shape[1]
    print(f"Dataset  : {DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"K-folds  : {K_FOLDS}-fold CV (seed {KFOLD_SEED})  |  tag: {MODEL_TAG}")
    print(f"Search   : CMA-ES  max_eval={CMA_MAX_EVAL} per fold")
    print(f"Protocol : k-fold like pms_kfold.py — one Octave-style 80-20 per fold, "
          f"repeated {K_FOLDS}x. The 80% is split into train/medval/val; the held-out "
          f"20% is the reported MAPE (= Octave finalval / bottom-right graph).")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # TRUE k-fold: same splitter and seed as pms_kfold.py, so the Python and
    # Octave experiments see IDENTICAL folds and are directly comparable.
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=KFOLD_SEED)

    fold_results = []
    all_actuals  = []   # every held-out row's true T, pooled across folds
    all_preds    = []   # every held-out row's prediction, pooled across folds

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X), start=1):
        print("\n" + "=" * 70)
        print(f"  FOLD {fold_idx}/{K_FOLDS}")
        print("=" * 70)

        X_tr_fold, X_test_fold = X[train_index], X[test_index]
        y_tr_fold, y_test_fold = y[train_index], y[test_index]

        # ── split the fold's 80% into train/medval/val for the search ────
        # (NO extra finalval — the held-out 20% below is the reported set.)
        (X_train, y_train, X_medval, y_medval,
         X_val, y_val) = inner_split(X_tr_fold, y_tr_fold, SEED)

        print("  [CMA-ES] Searching architecture (objective = training MSE)...")
        best_model, best_layers, best_neurons, best_act, num_saved = cmaes_search(
            X_train, y_train, X_medval, y_medval, X_val, y_val
        )

        if best_model is None:
            print("  WARNING: no candidate could be trained this fold; skipping.")
            continue

        print(f"  Best architecture (selected on val): "
              f"{best_layers} layers × {best_neurons} neurons, act={best_act}")
        print(f"  Candidates saved (medval MAPE < {MAPE_GATE:.0f}%): {num_saved}")

        # ── evaluate on the fold's held-out 20% (the reported MAPE) ─────
        preds = best_model.predict(X_test_fold)
        preds = np.maximum(0.0, preds)   # clamp negatives, same as pms_kfold.py

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
            "fold":       fold_idx,
            "n_layers":   best_layers,
            "neurons":    best_neurons,
            "activation": best_act,
            "candidates": num_saved,
            "mape":       mape_val,
            "smape":      smape_val,
            "mae":        mae_val,
            "rmse":       rmse_val,
            "r2":         r2_val,
            "n_test":     n_test,
            "bad_count":  bad_count,
        })
        all_actuals.extend(y_test_fold.tolist())
        all_preds.extend(preds.tolist())

    if not fold_results:
        print("No successful folds — nothing to report.")
        return

    # ── per-fold mean ± 95% t-CI — identical CI logic to pms_kfold.py ──
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
    candidate_list                 = [r["candidates"] for r in fold_results]

    # ── POOLED over all rows (every row tested exactly once) — pms_kfold-style ─
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
        f"  Fold {r['fold']}: "
        f"arch={r['n_layers']}L×{r['neurons']}N act={r['activation']}  "
        f"MAPE={r['mape']:.2f}%  SMAPE={r['smape']:.2%}  "
        f"MAE={r['mae']:.1f}ms  RMSE={r['rmse']:.1f}ms  R2={r['r2']:.4f}  "
        f"n={r['n_test']}  candidates={r['candidates']}  bad={r['bad_count']}"
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
        f"MODEL         : CMA-ES + MLPRegressor (sklearn)\n"
        f"TRAINER       : solver=lbfgs, activation=SEARCHED (matches Octave trainlm)\n"
        f"CONFIG        : {num_inputs} inputs -> 1 target | "
        f"{K_FOLDS}-fold CV (seed {KFOLD_SEED}) | tag: {MODEL_TAG}\n"
        f"SEARCH SPACE  : layers [{N_LAYERS_MIN},{N_LAYERS_MAX}], "
        f"neurons/layer [{NEURONS_MIN},{NEURONS_MAX}], "
        f"activation in {ACTIVATIONS}\n"
        f"SCALING       : inputs + target MinMax[-1,1] (matches Octave normalize.m)\n"
        f"CMA-ES BUDGET : {CMA_MAX_EVAL} evaluations per fold | objective=training MSE\n"
        f"PROTOCOL      : k-fold like pms_kfold.py — ONE Octave-style 80-20 per "
        f"fold, repeated {K_FOLDS}x, every row held out exactly once. The 80% is "
        f"split into train (fits candidates) / medval (gate <{MAPE_GATE:.0f}%) / "
        f"val (selects best net). NO nested finalval; the held-out 20% is the "
        f"reported MAPE (= Octave finalval / bottom-right graph MAPE).\n"
        f"SEED          : {SEED}\n"
        f"METRICS       : per-fold, POOLED over all rows, AND mean ± 95% t-CI across "
        f"folds | MAPE in %, SMAPE on 0..200% scale\n"
        f"{'─' * 70}\n"
        f"{fold_lines}\n"
        f"{'─' * 70}\n"
        f"CANDIDATE COUNTS : {candidate_list}\n"
        f"POOLED over all {n_pool} rows:\n"
        f"  POOL MAPE : {pool_mape:.2f}%\n"
        f"  POOL SMAPE: {pool_smape * 100:.2f}%\n"
        f"  POOL MAE  : {pool_mae:.1f} ms\n"
        f"  POOL RMSE : {pool_rmse:.1f} ms\n"
        f"  POOL R2   : {pool_r2:.4f}\n"
        f"{'─' * 70}\n"
        f"MEAN ± 95% t-CI ACROSS {K_FOLDS} FOLDS:\n"
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
        os.path.join(RESULTS_DIR, f"kfold_results_{MODEL_TAG}_cmaes.csv"),
        index=False
    )

    print(f"Log : {LOG_FILE}")
    print(f"CSV : {os.path.join(RESULTS_DIR, f'kfold_results_{MODEL_TAG}_cmaes.csv')}")


# ── ENTRY POINT ───────────────────────────────────────────────────────
if __name__ == "__main__":
    run_cmaes_kfold()
