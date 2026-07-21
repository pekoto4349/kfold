import os
import math
import random
import datetime
import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
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
DATA_FILE   = "master_dataset.csv"   # swap this to run calibration variants
RESULTS_DIR = "ga_results"
MODEL_TAG   = "NSMT"                  # label for this run  e.g. NSMT, NSMKT, etc.

# Repeated INDEPENDENT runs (NOT k-fold), exactly like pms_runs.py: each run
# does one fresh Octave-style split and reports ONE finalval MAPE, then we
# aggregate mean ± 95% CI across runs.
NUM_RUNS = _env_int("PMS_RUNS", _env_int("PMS_FOLDS", 5))

# ── OCTAVE-FAITHFUL SPLIT (createmodel.m lines 49 + 57) ───────────────
# Octave PMS splits the data ONCE into four disjoint sets:
#   finalval  = 20% of ALL data        -> the single reported MAPE (never seen)
#   train / medval / val  from the 80% -> ~50% / ~20% / ~30% of that 80%
# and each set has a distinct job (there is NO "inner/outer" k-fold here):
#   train  -> the network is actually trained on this
#   medval -> SAVE GATE: keep a candidate only if its medval MAPE < 20%
#   val    -> SELECT the best saved candidate on this
#   finalval -> report one MAPE on this held-out set
FINALVAL_RATIO     = 0.20   # createmodel.m: subset(trainver1,1,1,0.2)
VAL_RATIO_OF_80    = 0.30   # createmodel.m: subset(trainver2,...,0.3)
MEDVAL_RATIO_OF_80 = 0.20   # createmodel.m: subset(trainver2,...,0.2)
# train = the remaining ~0.50 of the 80%.

# Reproducibility seed for the STOCHASTIC parts (the split, MLP init, and the
# GA's own RNG). Default None -> non-deterministic, exactly like the Octave GA.
# Set PMS_SEED=<int> to make a full run reproducible (all runs become identical).
SEED = _env_seed("PMS_SEED", None)

# GA settings — chosen to MATCH Octave PMS so the comparison is fair.
# Octave PMS uses PopulationSize = 30, Generations = 30  -> 900 evaluations.
POP_SIZE = _env_int("PMS_POP", 30)   # candidate architectures alive each generation
N_GEN    = _env_int("PMS_GEN", 30)   # number of generations
CXPB     = 0.8       # crossover probability (Octave default ~0.8)
MUTPB    = 0.2       # mutation probability  (Octave default ~0.2)
INDPB    = 0.2       # per-gene mutation probability when an individual mutates
TOURNSIZE = 3        # tournament selection size (Octave uses tournament selection)

# Architecture search space (UNCHANGED — the neuron/layer bounds are managed
# separately). Chromosome = [n_layers, n1, n2, n3, n4, n5]:
#   gene 0     : number of hidden layers, in [1, 5]
#   genes 1..5 : neurons for each layer,  in [1, 55]
# Only the first n_layers neuron genes are used to build the network.
N_LAYERS_MIN, N_LAYERS_MAX = 1, 5
NEURONS_MIN,  NEURONS_MAX  = 1, 55

MAPE_GATE = 20.0     # Octave's medval save-gate (nnscript.m: <0.20)

LOG_FILE = f"experiment_logs_{MODEL_TAG}_ga.txt"
# ─────────────────────────────────────────────────────────────────────


# ── METRICS  (identical to pms_runs.py / cmaes_mlp_runs.py) ──────────

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


# ── OCTAVE-FAITHFUL 4-WAY SPLIT ───────────────────────────────────────

def octave_split(X, y, seed):
    """
    Reproduce createmodel.m's split:
      finalval = 20% of all
      then split the 80% into train (~50%) / medval (~20%) / val (~30%).
    Returns X_train, y_train, X_medval, y_medval, X_val, y_val, X_final, y_final.
    """
    # 1) peel off finalval (20% of all)  -> createmodel.m line 49
    X_tv, X_final, y_tv, y_final = train_test_split(
        X, y, test_size=FINALVAL_RATIO, random_state=seed
    )
    # 2) peel off val (30% of the 80%)   -> createmodel.m line 57
    X_tm, X_val, y_tm, y_val = train_test_split(
        X_tv, y_tv, test_size=VAL_RATIO_OF_80, random_state=seed
    )
    # 3) split the rest into train / medval.  medval is 20% of the 80%,
    #    which is (20/70) of what remains after val was removed.
    medval_frac_of_rest = MEDVAL_RATIO_OF_80 / (1.0 - VAL_RATIO_OF_80)
    X_train, X_medval, y_train, y_medval = train_test_split(
        X_tm, y_tm, test_size=medval_frac_of_rest, random_state=seed
    )
    return X_train, y_train, X_medval, y_medval, X_val, y_val, X_final, y_final

# ─────────────────────────────────────────────────────────────────────


# ── ARCHITECTURE BUILD ────────────────────────────────────────────────
# Octave PMS normalizes every input AND the target to [-1, 1] via normalize.m
# before training. We reproduce that exactly with MinMaxScaler([-1, 1]) on the
# inputs (inside the pipeline) and on the target (TransformedTargetRegressor);
# predictions are inverse-transformed back to real T units automatically.
#
# The trainer is chosen to MATCH Octave's Levenberg-Marquardt (net.trainFcn=
# 'trainlm') as closely as scikit-learn allows:
#   solver     = "lbfgs"  -> quasi-Newton, the closest sklearn analogue to LM,
#                            and the recommended solver for small datasets.
#   activation = "tanh"   -> matches Octave's tansig/logsig transfer family.
# early_stopping is intentionally OFF (Octave has none; lbfgs ignores it).

def build_mlp(hidden_layer_sizes):
    net = MLPRegressor(
        hidden_layer_sizes = hidden_layer_sizes,
        activation         = "tanh",
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


# ── DEAP SETUP ────────────────────────────────────────────────────────
# Minimisation problem: the GA fitness is TRAINING MSE (nnscript.m: perf=thismse),
# exactly like Octave. An "Individual" is a list of genes carrying a fitness.
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))   # -1 -> minimise
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def make_individual():
    """Create one random chromosome within the architecture bounds."""
    genes = [random.randint(N_LAYERS_MIN, N_LAYERS_MAX)]          # gene 0 = layer count
    genes += [random.randint(NEURONS_MIN, NEURONS_MAX)           # genes 1..5 = widths
              for _ in range(N_LAYERS_MAX)]
    return creator.Individual(genes)


def genes_to_hidden(individual):
    """Decode a chromosome into the hidden-layer-sizes tuple actually used."""
    n_layers = int(individual[0])
    widths   = [int(individual[1 + i]) for i in range(n_layers)]
    return n_layers, widths

# ─────────────────────────────────────────────────────────────────────


# ── GA ARCHITECTURE SEARCH (Octave-faithful) ──────────────────────────

def ga_search(X_train, y_train, X_medval, y_medval, X_val, y_val):
    """
    Run a classic GA to find the best architecture, faithfully to Octave:
      - fitness of each candidate = TRAINING MSE on `train`  (nnscript.m: perf=thismse)
      - a candidate is SAVED only if its medval MAPE < MAPE_GATE  (nnscript.m: <0.20)
      - after the GA, the best SAVED candidate is SELECTED on `val`  (createmodel.m)
    Returns (best_model, best_arch, n_layers, num_saved).
    best_model is the trained model; best_arch is the list of layer widths.
    """
    # `saved` holds every candidate that passed the medval gate, as Octave does
    # by archiving nets into bestnets/. `fallback` keeps the single best-by-medval
    # candidate so a run never ends with nothing (Octave would simply save none).
    saved    = []                       # list of (model, medval_mape, widths)
    fallback = {"model": None, "medval": float("inf"), "widths": None}

    def evaluate(individual):
        n_layers, widths = genes_to_hidden(individual)
        model = build_mlp(tuple(widths))
        try:
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            train_mse  = float(np.mean((train_pred - y_train) ** 2))   # <- GA fitness
            med_pred   = model.predict(X_medval)
            med_mape   = safe_mape(y_medval, med_pred)
        except Exception:
            return (1e18,)              # penalise failed fits
        if med_mape < MAPE_GATE:        # Octave save-gate
            saved.append((model, med_mape, widths))
        if med_mape < fallback["medval"]:
            fallback["model"], fallback["medval"], fallback["widths"] = model, med_mape, widths
        return (train_mse,)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    low = [N_LAYERS_MIN] + [NEURONS_MIN] * N_LAYERS_MAX
    up  = [N_LAYERS_MAX] + [NEURONS_MAX] * N_LAYERS_MAX
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutUniformInt, low=low, up=up, indpb=INDPB)
    toolbox.register("select",   tools.selTournament, tournsize=TOURNSIZE)

    # ── initial population ────────────────────────────────────────────
    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(1)           # elitism on training-MSE fitness
    hof.update(pop)

    # ── generational loop ─────────────────────────────────────────────
    for _ in range(N_GEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = offspring
        pop[0] = toolbox.clone(hof[0])
        hof.update(pop)

    # ── model selection on `val` among SAVED candidates ───────────────
    pool = saved if saved else (
        [(fallback["model"], fallback["medval"], fallback["widths"])]
        if fallback["model"] is not None else []
    )
    if not pool:
        return None, None, 0, 0

    def val_mape_of(entry):
        return safe_mape(y_val, entry[0].predict(X_val))

    best_model, _, best_widths = min(pool, key=val_mape_of)
    return best_model, best_widths, len(best_widths), len(saved)

# ─────────────────────────────────────────────────────────────────────


# ── MAIN LOOP ─────────────────────────────────────────────────────────

def run_ga_repeated():

    if SEED is not None:
        random.seed(SEED)
        np.random.seed(SEED)

    df = pd.read_csv(DATA_FILE, header=None)
    X  = df.iloc[:, :-1].values.astype(float)   # all columns except last = inputs
    y  = df.iloc[:,  -1].values.astype(float)   # last column = target T

    num_inputs = X.shape[1]
    print(f"Dataset  : {DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"Runs     : {NUM_RUNS} repeated runs  |  tag: {MODEL_TAG}")
    print(f"Search   : GA (DEAP)  pop={POP_SIZE} x gen={N_GEN} = "
          f"{POP_SIZE * N_GEN} evals per run")
    print(f"Protocol : Octave-faithful single split "
          f"(finalval 20% | train/medval/val ~50/20/30 of 80%)")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_results = []

    for run_idx in range(1, NUM_RUNS + 1):
        print("\n" + "=" * 70)
        print(f"  RUN {run_idx}/{NUM_RUNS}")
        print("=" * 70)

        (X_train, y_train, X_medval, y_medval,
         X_val, y_val, X_final, y_final) = octave_split(X, y, SEED)

        print("  [GA] Evolving architecture (fitness = training MSE)...")
        best_model, best_widths, n_layers, num_saved = ga_search(
            X_train, y_train, X_medval, y_medval, X_val, y_val
        )

        if best_model is None:
            print("  WARNING: no candidate could be trained this run; skipping.")
            continue

        print(f"  Best architecture (selected on val): "
              f"{n_layers} layers, widths {best_widths}")
        print(f"  Candidates saved (medval MAPE < {MAPE_GATE:.0f}%): {num_saved}")

        # ── evaluate the selected model on finalval (the ONE reported MAPE) ─
        preds = best_model.predict(X_final)
        preds = np.maximum(0.0, preds)   # clamp negatives, same as pms_runs.py

        mape_val  = safe_mape(y_final, preds)
        smape_val = _smape(y_final, preds)
        mae_val   = _mae(y_final, preds)
        rmse_val  = _rmse(y_final, preds)
        r2_val    = safe_r2(y_final, preds)
        n_final   = len(y_final)

        signed_pct = (preds - y_final) / np.maximum(np.abs(y_final), 1e-8) * 100
        bad_count  = int(np.sum(np.abs(signed_pct) > 200))

        print(f"\n  MAPE   (finalval, {n_final} rows): {mape_val:.2f}%")
        print(f"  SMAPE  (finalval, {n_final} rows): {smape_val:.2%}")
        print(f"  MAE    (finalval, {n_final} rows): {mae_val:.1f} ms")
        print(f"  RMSE   (finalval, {n_final} rows): {rmse_val:.1f} ms")
        print(f"  R2     (finalval, {n_final} rows): {r2_val:.4f}")
        print(f"  Bad cases (|err|>200%)          : {bad_count}")

        run_results.append({
            "run":        run_idx,
            "n_layers":   n_layers,
            "widths":     "-".join(str(w) for w in best_widths),
            "candidates": num_saved,
            "mape":       mape_val,
            "smape":      smape_val,
            "mae":        mae_val,
            "rmse":       rmse_val,
            "r2":         r2_val,
            "n_final":    n_final,
            "bad_count":  bad_count,
        })

    if not run_results:
        print("No successful runs — nothing to report.")
        return

    # ── mean ± 95% t-CI across runs — identical CI logic to pms_runs.py ──
    T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
              5: 2.571,  7: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}

    def _t_crit(n):
        return T_CRIT.get(n - 1, 1.96)

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
    candidate_list                 = [r["candidates"] for r in run_results]

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_lines = "\n".join(
        f"  Run {r['run']}: "
        f"arch={r['n_layers']}L[{r['widths']}]  "
        f"MAPE={r['mape']:.2f}%  SMAPE={r['smape']:.2%}  "
        f"MAE={r['mae']:.1f}ms  RMSE={r['rmse']:.1f}ms  R2={r['r2']:.4f}  "
        f"n={r['n_final']}  candidates={r['candidates']}  bad={r['bad_count']}"
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
        f"MODEL         : Genetic Algorithm (DEAP) + MLPRegressor (sklearn)\n"
        f"TRAINER       : solver=lbfgs, activation=tanh (matches Octave trainlm/tansig)\n"
        f"CONFIG        : {num_inputs} inputs -> 1 target | "
        f"{NUM_RUNS} repeated runs | tag: {MODEL_TAG}\n"
        f"SEARCH SPACE  : layers [{N_LAYERS_MIN},{N_LAYERS_MAX}], "
        f"neurons/layer [{NEURONS_MIN},{NEURONS_MAX}]\n"
        f"SCALING       : inputs + target MinMax[-1,1] (matches Octave normalize.m)\n"
        f"GA SETTINGS   : pop={POP_SIZE} x gen={N_GEN} = {POP_SIZE * N_GEN} evals | "
        f"cxpb={CXPB} mutpb={MUTPB} tournsize={TOURNSIZE} | fitness=training MSE\n"
        f"PROTOCOL      : Octave-faithful single split per run — finalval 20%, "
        f"then train/medval/val ~50/20/30 of the 80%. medval is the save-gate "
        f"(<{MAPE_GATE:.0f}%), val selects the best saved net, finalval is the one "
        f"reported MAPE. No k-fold, no inner/outer.\n"
        f"SEED          : {SEED}\n"
        f"METRICS       : per-run on finalval + mean ± 95% t-CI across runs | "
        f"MAPE in %, SMAPE on 0..200% scale\n"
        f"{'─' * 70}\n"
        f"{run_lines}\n"
        f"{'─' * 70}\n"
        f"CANDIDATE COUNTS : {candidate_list}\n"
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

    pd.DataFrame(run_results).to_csv(
        os.path.join(RESULTS_DIR, f"run_results_{MODEL_TAG}_ga.csv"),
        index=False
    )

    print(f"Log : {LOG_FILE}")
    print(f"CSV : {os.path.join(RESULTS_DIR, f'run_results_{MODEL_TAG}_ga.csv')}")


# ── ENTRY POINT ───────────────────────────────────────────────────────
if __name__ == "__main__":
    run_ga_repeated()
