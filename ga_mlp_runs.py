import os
import math
import random
import datetime
import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# ── CONFIG ───────────────────────────────────────────────────────────
DATA_FILE   = "master_dataset.csv"   # swap this to run calibration variants
RESULTS_DIR = "ga_results"
MODEL_TAG   = "NSMT"                  # label for this run  e.g. NSMT, NSMKT, etc.

N_FOLDS     = 5      # TRUE k-fold cross-validation (KFOLD=5), like pms_kfold.py
                     # The 630 rows are cut into 5 disjoint folds; each fold is
                     # the test set exactly once, so every row is tested once.
KFOLD_SEED  = 42     # fixed seed → reproducible folds (matches pms_kfold.py)
VAL_RATIO   = 0.20   # 20% of the 4 training folds used as GA fitness signal
                     # (= PMS medval).  Inner model-selection split, nested
                     # inside each outer fold.

# GA settings — chosen to MATCH Octave PMS exactly so the comparison is fair.
# Octave PMS uses PopulationSize = 30, Generations = 30  → 900 evaluations.
POP_SIZE = 30        # number of candidate architectures alive each generation
N_GEN    = 30        # number of generations
CXPB     = 0.8       # crossover probability (Octave default ~0.8)
MUTPB    = 0.2       # mutation probability  (Octave default ~0.2)
INDPB    = 0.2       # per-gene mutation probability when an individual mutates
TOURNSIZE = 3        # tournament selection size (Octave uses tournament selection)

# Architecture search space — same ceiling as PMS best (NLmax=5, NperLmax=55)
# Chromosome = [n_layers, neurons_L1, neurons_L2, neurons_L3, neurons_L4, neurons_L5]
#   gene 0      : number of hidden layers, in [1, 5]
#   genes 1..5  : neurons for each layer,  in [1, 55]
# Only the first n_layers neuron genes are used to build the network.
# This mirrors Octave's per-layer encoding (variable depth + per-layer width).
N_LAYERS_MIN, N_LAYERS_MAX = 1, 5
NEURONS_MIN,  NEURONS_MAX  = 1, 55

MAPE_GATE = 20.0     # same concept as PMS "medval MAPE < 20% gate"

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
    a, p = np.asarray(a, float), np.asarray(p, float)
    num  = np.abs(a - p)
    den  = np.abs(a) + np.abs(p)
    mask = den > 0
    return float(np.mean(2.0 * num[mask] / den[mask])) if np.any(mask) else 0.0

def _mae(a, p):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float))))

def _rmse(a, p):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float)) ** 2)))

# ─────────────────────────────────────────────────────────────────────


# ── ARCHITECTURE BUILD ────────────────────────────────────────────────
# A chromosome is a list of 6 integers.  We read the first gene as the
# layer count and the next genes as per-layer neuron counts.

def build_mlp_from_genes(individual):
    """
    Turn a chromosome [n_layers, n1, n2, n3, n4, n5] into a scaled MLP.

    Octave PMS normalizes every input AND the target to [-1, 1] via
    normalize.m before training, because neural nets train poorly on raw,
    differently-scaled features (here M~32 vs N~10000 vs T~52000).
    We reproduce that exactly:
      - MinMaxScaler([-1, 1]) on the inputs   (inside the pipeline)
      - MinMaxScaler([-1, 1]) on the target   (TransformedTargetRegressor)
    Predictions are automatically inverse-transformed back to real T units.
    """
    n_layers = int(individual[0])
    neurons  = [int(individual[1 + i]) for i in range(n_layers)]  # first n_layers widths
    hidden   = tuple(neurons)
    net = MLPRegressor(
        hidden_layer_sizes = hidden,
        activation         = "relu",
        solver             = "adam",
        max_iter           = 2000,
        random_state       = None,    # non-deterministic, like the Octave GA
        early_stopping     = True,
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


# ── DEAP SETUP ────────────────────────────────────────────────────────
# We tell DEAP: this is a MINIMISATION problem (smaller MAPE is better),
# and an "Individual" is a list of genes carrying a fitness value.

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))   # -1 → minimise
creator.create("Individual", list, fitness=creator.FitnessMin)


def make_individual():
    """Create one random chromosome within the architecture bounds."""
    genes = [random.randint(N_LAYERS_MIN, N_LAYERS_MAX)]          # gene 0 = layer count
    genes += [random.randint(NEURONS_MIN, NEURONS_MAX)           # genes 1..5 = widths
              for _ in range(N_LAYERS_MAX)]
    return creator.Individual(genes)


def build_toolbox(X_fit, y_fit, X_val, y_val):
    """
    Assemble the GA operators.  This is where the 'genetics' live:
      - evaluate : fitness = validation MAPE of the decoded network
      - mate     : two-point crossover (swap gene segments between two parents)
      - mutate   : uniform-int reset of random genes within bounds
      - select   : tournament selection (pick best of TOURNSIZE random picks)
    """
    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        model = build_mlp_from_genes(individual)
        try:
            model.fit(X_fit, y_fit)
            preds = model.predict(X_val)
            return (safe_mape(y_val, preds),)   # DEAP expects a tuple
        except Exception:
            return (999.0,)                     # penalise failed fits

    # Bounds for mutation: gene 0 is layer count, genes 1..5 are widths.
    low = [N_LAYERS_MIN] + [NEURONS_MIN] * N_LAYERS_MAX
    up  = [N_LAYERS_MAX] + [NEURONS_MAX] * N_LAYERS_MAX

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate",     tools.cxTwoPoint)
    toolbox.register("mutate",   tools.mutUniformInt, low=low, up=up, indpb=INDPB)
    toolbox.register("select",   tools.selTournament, tournsize=TOURNSIZE)
    return toolbox

# ─────────────────────────────────────────────────────────────────────


# ── GA ARCHITECTURE SEARCH ────────────────────────────────────────────

def ga_search(X_train, y_train):
    """
    Run a classic single-objective GA to find the best architecture.

    Fitness = MAPE on an inner validation split carved from X_train,
    mirroring PMS's medval set (NOT the final test set).

    Returns (best_n_layers, best_neurons_list, best_val_mape).
    """
    # Inner split: 80% to fit candidates, 20% as the GA's fitness signal
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VAL_RATIO, random_state=None
    )

    toolbox = build_toolbox(X_fit, y_fit, X_val, y_val)

    # ── initial population ────────────────────────────────────────────
    pop = toolbox.population(n=POP_SIZE)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # Hall of Fame keeps the single best individual ever seen = ELITISM
    hof = tools.HallOfFame(1)
    hof.update(pop)

    # ── generational loop ─────────────────────────────────────────────
    for gen in range(N_GEN):
        # 1. SELECTION — choose parents via tournaments
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # 2. CROSSOVER — splice gene segments between paired parents
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values     # mark as needing re-evaluation
                del child2.fitness.values

        # 3. MUTATION — randomly reset some genes within bounds
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # 4. RE-EVALUATE only the changed individuals
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid:
            ind.fitness.values = toolbox.evaluate(ind)

        # 5. ELITISM — the new generation replaces the old, but we force
        #    the best-ever individual back in so a good solution is never lost
        pop[:] = offspring
        pop[0] = toolbox.clone(hof[0])
        hof.update(pop)

    best        = hof[0]
    best_layers = int(best[0])
    best_widths = [int(best[1 + i]) for i in range(best_layers)]
    best_mape   = best.fitness.values[0]
    return best_layers, best_widths, best_mape

# ─────────────────────────────────────────────────────────────────────


# ── MAIN LOOP ─────────────────────────────────────────────────────────

def run_ga_repeated():

    df = pd.read_csv(DATA_FILE, header=None)
    X  = df.iloc[:, :-1].values.astype(float)   # all columns except last = inputs
    y  = df.iloc[:,  -1].values.astype(float)   # last column = target T

    num_inputs = X.shape[1]
    print(f"Dataset  : {DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"Folds    : {N_FOLDS}-fold CV  |  tag: {MODEL_TAG}")
    print(f"Search   : GA (DEAP)  pop={POP_SIZE} x gen={N_GEN} = "
          f"{POP_SIZE * N_GEN} evals per fold")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_results = []

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

        # ── GA: search for best architecture on the train portion ─────
        print("  [GA] Evolving architecture...")
        best_layers, best_widths, val_mape = ga_search(X_train, y_train)
        print(f"  Best architecture found: {best_layers} layers, widths {best_widths}")
        print(f"  Best val MAPE (inner)  : {val_mape:.2f}%")

        if val_mape > MAPE_GATE:
            print(f"  NOTE: val MAPE {val_mape:.2f}% exceeds gate {MAPE_GATE}% "
                  f"(PMS would not archive this model)")

        # ── Final model: retrain best architecture on full 80% train ──
        print("  [MLP] Training final model on full train set...")
        final_individual = [best_layers] + best_widths + \
                           [1] * (N_LAYERS_MAX - best_layers)   # pad unused genes
        final_model = build_mlp_from_genes(final_individual)
        final_model.fit(X_train, y_train)

        # ── Evaluate on the 20% hold-out (= PMS finalval) ─────────────
        preds = final_model.predict(X_test)
        preds = np.maximum(0.0, preds)   # clamp negatives, same as pms_runs.py

        mape_val  = safe_mape(y_test, preds)
        smape_val = _smape(y_test, preds)
        mae_val   = _mae(y_test, preds)
        rmse_val  = _rmse(y_test, preds)
        n_test    = len(y_test)

        signed_pct = (preds - y_test) / np.maximum(np.abs(y_test), 1e-8) * 100
        bad_count  = int(np.sum(np.abs(signed_pct) > 200))

        print(f"\n  MAPE   (test, {n_test} rows): {mape_val:.2f}%")
        print(f"  SMAPE  (test, {n_test} rows): {smape_val:.2%}")
        print(f"  MAE    (test, {n_test} rows): {mae_val:.1f} ms")
        print(f"  RMSE   (test, {n_test} rows): {rmse_val:.1f} ms")
        print(f"  Bad cases (|err|>200%)      : {bad_count}")

        run_results.append({
            "run":       run_idx,
            "n_layers":  best_layers,
            "widths":    "-".join(str(w) for w in best_widths),
            "val_mape":  val_mape,
            "mape":      mape_val,
            "smape":     smape_val,
            "mae":       mae_val,
            "rmse":      rmse_val,
            "n_test":    n_test,
            "bad_count": bad_count,
        })

    # ── pooled statistics — identical CI logic to pms_runs.py ─────────
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

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    run_lines = "\n".join(
        f"  Fold {r['run']}: "
        f"arch={r['n_layers']}L[{r['widths']}]  "
        f"val_MAPE={r['val_mape']:.2f}%  "
        f"MAPE={r['mape']:.2f}%  SMAPE={r['smape']:.2%}  "
        f"MAE={r['mae']:.1f}ms  RMSE={r['rmse']:.1f}ms  "
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
        f"MODEL         : Genetic Algorithm (DEAP) + MLPRegressor (sklearn)\n"
        f"CONFIG        : {num_inputs} inputs -> 1 target | "
        f"{N_FOLDS}-fold cross-validation | tag: {MODEL_TAG}\n"
        f"SEARCH SPACE  : layers [1,5], neurons/layer [1,55], activation=relu\n"
        f"SCALING       : inputs + target MinMax[-1,1] (matches Octave normalize.m)\n"
        f"GA SETTINGS   : pop={POP_SIZE} x gen={N_GEN} = {POP_SIZE * N_GEN} evals | "
        f"cxpb={CXPB} mutpb={MUTPB} tournsize={TOURNSIZE} (matches Octave PMS)\n"
        f"SPLIT         : 5 disjoint folds | per fold: 4 folds train "
        f"(20% inner val for GA) / 1 fold test | seed={KFOLD_SEED}\n"
        f"METRICS       : mean ± 95% t-CI across the {N_FOLDS} fold scores\n"
        f"{'─' * 70}\n"
        f"{run_lines}\n"
        f"{'─' * 70}\n"
        f"AVG MAPE  : {_fmt(avg_mape,  std_mape,  ci_mape,  '%')}\n"
        f"AVG SMAPE : {_fmt(avg_smape * 100, std_smape * 100, ci_smape * 100, '%')}\n"
        f"AVG MAE   : {_fmt(avg_mae,   std_mae,   ci_mae,   ' ms')}\n"
        f"AVG RMSE  : {_fmt(avg_rmse,  std_rmse,  ci_rmse,  ' ms')}\n"
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
