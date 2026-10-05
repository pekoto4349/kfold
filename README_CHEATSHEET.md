# PMS / FaaS PREDICTION — PhD PROGRESS CHEAT SHEET

> Purpose: a one-stop, bullet-point briefing to walk my PhD supervisor through the
> **entire** experiment on a call: what we predict, why every input/metric/algorithm
> was chosen, how each error metric is computed internally, and what the results say
> so far. Every "why" is spelled out. Read top-to-bottom or jump by section.
>
> Sources this summarizes: `pms_research_master.md`, `README_fold_csvs.md`,
> `conversation_summary.txt`, `specs/faas-surrogate-comparison.md`,
> `specs/next-steps-and-interpretation.md`, `specs/build-review-log.md`, and the
> scripts `pms_kfold.py`, `pms_runs.py`, `ga_mlp_runs.py`, `cmaes_mlp_runs.py`.

---

## 1. THE 30-SECOND PITCH

- We predict **T = end-to-end response time** of a serverless (FaaS) **split-join**
  workflow running on **Node-RED (orchestrator) + Apache OpenWhisk (FaaS runtime)**.
- A job of size **N** is split into chunks of size **S**, chunks run in parallel
  across up to **M** containers; we measure **T**.
- Dataset: **630 controlled measurements** of `(S, N, M) -> T`.
- Legacy tool "**PMS**" (Octave, after Kousiouris et al.) trains a neural network
  whose architecture is chosen by a **genetic algorithm** to predict T.
- The contribution is **three-fold**:
  1. Make that Octave experiment **statistically sound** (true k-fold CV + CIs).
  2. Add **physics-motivated calibration inputs** derived from queueing theory.
  3. Run a **controlled comparison** of the legacy Octave method vs modern Python
     methods (**DEAP GA**, **CMA-ES**, **XGBoost**) under one identical, fair protocol.
- **The headline is NOT** "we predicted FaaS time with ML." It is: *"a controlled
  comparison isolating whether prediction gains come from the **optimizer**, the
  **search paradigm**, or the **model family** — plus queueing-theory calibration
  features — on a real Node-RED/OpenWhisk split-join workflow."*

---

## 2. THE SYSTEM & DATASET (the ground truth)

- **Orchestrator**: Node-RED — splits the workload, dispatches chunks to OpenWhisk,
  polls for results, joins them.
- **FaaS runtime**: Apache OpenWhisk (standalone/local mode).
- **Workload pattern**: split-join (SPMD-style parallelism).
- **Mental model of one run**:
  - Job of **N** tasks arrives.
  - Node-RED splits it into chunks of **S** tasks → there are **K = ceil(N/S)** chunks
    (each chunk = one OpenWhisk activation).
  - OpenWhisk runs up to **M** at once.
  - If `K > M`, extra chunks wait and run in later "waves"; wave count **B = ceil(K/M)**.
  - When all chunks finish, Node-RED joins → total wall-clock time = **T**.
- **Dataset**: `master_dataset.csv`, **630 rows, no header, 4 columns in order S, N, M, T**.
  - (⚠️ column order gotcha: first col is **S = split size**, second is **N = workload**.)
  - **S**: 20, 40, 60, 80, 100 (5 values)
  - **N**: 100, 600, 1100, … 10100 (step 500; 21 values)
  - **M**: 2, 8, 14, 20, 26, 32 (6 values)
  - 21 × 5 × 6 = **630 rows**.
- **T** is in milliseconds, **right-skewed**, and **grows non-linearly** — it *explodes*
  once the system saturates (demand exceeds capacity). This skew/blow-up is exactly why
  metric choice (below) matters and why calibration features help.

---

## 3. INPUT VARIABLES — AND WHY EXACTLY THESE

### 3.1 The three primary inputs: N, S, M (why only these)
- **N (workload size)** — total tasks; set by the caller; the *demand* signal.
- **S (split size)** — tasks per chunk; set by orchestration policy; controls
  *parallelism granularity* (small S → many fine chunks; large S → few coarse ones).
- **M (max containers)** — concurrency cap; set by operator; the *capacity*.
- **Why only these three**: they are **fully orthogonal** — any combination is settable
  independently. This is the **minimum sufficient independent representation** of the
  configuration.
- **Why not add derived quantities as primary inputs**: K, U, B, C are *read-outs* of
  (N, S, M) — putting them in as primaries **double-counts information** and **breaks
  provisioning use** (if you want the model to find the N,S,M that hits a target T, it
  must take only the knobs you can actually set).

### 3.2 The primary target: T
- **T** is the **SLO-relevant** quantity an operator commits to. D and W are
  sub-components reported by the platform; the deleted variable H is discussed in §5.

---

## 4. CALIBRATION INPUTS — DERIVED, PHYSICS-MOTIVATED HELPERS

### 4.1 Why add them at all
- The `NSM → T` surface has a sharp, **physically-real regime boundary at K = M**
  (the saturation point). Below it (K ≤ M) all chunks run at once, T ≈ one activation's
  time; above it (K > M) queuing begins and T grows with the number of waves.
- A plain neural net must **discover** this implicit N/S-vs-M interaction from only 630
  rows. Calibration inputs **hand the network the governing physics directly** — this is
  **domain-knowledge injection / physics-informed feature engineering**.
- Each one is **pure algebra of N, S, M** — no new measurements, **no data leakage**.
- The scientific question each answers: *"does explicitly encoding this piece of the
  physics improve the legacy surrogate?"*

### 4.2 The four calibration inputs (in order of physical directness)
- **K = ceil(N / S)** — *invocation count* (how many chunks/activations exist). Most
  direct demand signal; integer; discretizes the config (e.g. N=100,S=20 → K=5 vs
  N=100,S=100 → K=1).
- **C = K / B** — *average concurrent containers busy per wave*.
  - ⚠️ **Correction we caught**: an earlier guess `C = min(K, M)` is **WRONG** — it only
    holds for full waves, not the partial last wave.
  - Derivation: first `B-1` waves run M each, last wave runs `R = K - M*(B-1) < M`; total
    slots `= M*(B-1) + R = K`, so average over B waves `= K/B`.
  - Example: N=100,S=20,M=2 → K=5,B=3, waves [2,2,1], avg = 5/3 ≈ 1.67 (NOT 2).
  - Properties: `C ≤ M` always; `C = M` only when K is an exact multiple of M; `C = K`
    when `K < M`.
- **U = K / M** — *utilization* (demand-to-capacity ratio). This is the queueing-theory
  **ρ (rho)** from the M/M/c model. `U < 1` = undersaturated (no queue); `U ≥ 1` =
  saturated (queue forms). Continuous → captures the congestion *gradient*. It is the
  "before-the-ceiling" ratio (can exceed 1).
- **B = ceil(K / M)** — *wave count* (sequential rounds to finish all K). The most direct
  multiplier on time: `T ≈ B × (per-wave time)`. Small integer, {1..~5}.
- **Relationship**: `B = ceil(K/M) = ceil(U)` and `C = K/B`. So **C** is the
  "after-the-ceiling" average utilization; **U** is the "before-the-ceiling" ratio — they
  carry genuinely different info (equal only when K divides M evenly).

### 4.3 Why test them ONE AT A TIME (not combined)
- K, C, U, B are all derived from the same N, S, M and are **monotonically related
  (high collinearity)**.
- Combining them (e.g. `NSMKUT`) **collapses the experiment** — you can't attribute any
  improvement to a specific feature.
- With only 630 rows, extra correlated inputs also **risk overfitting**.
- So each is a separate **4-input model** vs the **NSMT baseline**; we see which helps most.

### 4.4 Parking-lot analogy (for the call)
- Lot has **M** spots; cars arrive in groups (**N** total cars, **S** per group):
  - **K** = number of groups that show up (= ceil(N/S)).
  - **B** = number of parking rounds needed (= ceil(K/M)).
  - **U** = how overbooked the lot is (= K/M; "2.5× more groups than spots").
  - **C** = average spots occupied per round (= K/B; below M when the last round doesn't fill).

---

## 5. THE DELETED VARIABLE H, AND ITS REPLACEMENT T_wave

- **Why H = T − D − W was DELETED**:
  - **T** is a **critical-path** time (bounded by the slowest chunk: max over K of D+W).
  - **D** (duration) and **W** (wait) are **per-activation averages**.
  - So `H = T − mean(D) − mean(W)` **mixes a maximum with two averages** — statistically
    incoherent. It also absorbs the fixed ~3-second Node-RED **polling delay** (a clock
    artifact) and cross-chunk variance. → It's a **measurement artifact, not a modelable
    quantity** → removed.
- **Its replacement: T_wave = T / B** (dataset tag `NSM_T-B`):
  - In the saturated regime `T ≈ B × (per-wave time)`, so `T_wave ≈ per-wave cost +
    overhead/B`, converging to a smooth per-activation cost as B grows.
  - Removes the discrete **staircase** (T jumps each time K crosses a multiple of M),
    leaving a **smoother surface** that should be easier to predict.
  - Needs **no new data** (B is derived, T is in the CSV). If `NSMTw` clearly beats
    `NSMT`, it validates the decomposition `T = T_wave × B` as the preferred strategy.

---

## 6. THE ERROR METRICS — HOW EACH IS COMPUTED INTERNALLY & WHY

> All metrics compare the **predicted T** to the **true T** on **held-out** rows.
> All are computed **identically across every method** so the comparison is fair. Code
> lives in `safe_mape`, `_smape`, `_mae`, `_rmse`, `safe_r2` (same formulas in all scripts).

- **MAPE — Mean Absolute Percentage Error** *(the headline number)*
  - Formula: `mean( |y_true − y_pred| / max(|y_true|, 1e-8) ) × 100` → reported in **%**.
  - Plain words: "on average, how many percent off am I?"
  - **Why the `1e-8` floor**: prevents divide-by-zero when a true value is ~0.
  - **Why it's the headline**: it's scale-free (comparable across T, D, W which differ by
    orders of magnitude) and it's the metric the legacy PMS reports, so it anchors the
    comparison to prior work.
  - **Weakness (state honestly)**: with a **right-skewed** target, a small absolute miss
    on a **tiny true T** looks like a huge percentage → MAPE can be inflated by the small-T
    regime. That's exactly why we also report SMAPE and R².

- **SMAPE — Symmetric MAPE**
  - Formula: `mean( 2·|y_true − y_pred| / (|y_true| + |y_pred|) )`, averaging only over
    rows where the denominator > 0.
  - Internally returns a **fraction in [0, 2]**, reported as a **0..200% scale** (×100).
  - Plain words: same idea as MAPE but the error is measured against the **average of true
    and predicted**, so over- and under-predictions are treated more evenly and it **can't
    blow up** when the true value is tiny.
  - **Why include it**: it's the robustness cross-check on MAPE's small-T inflation. ⚠️
    Always state the **0..200% convention** so the number isn't misread.

- **MAE — Mean Absolute Error**
  - Formula: `mean( |y_true − y_pred| )` — in **real milliseconds**.
  - Plain words: "on average, how many ms off am I?"
  - **Why include it**: percentage metrics hide the physical magnitude; MAE reports error
    in the units the operator cares about (ms). Every miss counts **linearly**.

- **RMSE — Root Mean Squared Error**
  - Formula: `sqrt( mean( (y_true − y_pred)^2 ) )` — in **ms**.
  - Plain words: like MAE but errors are **squared before averaging**, then square-rooted,
    so **big misses are punished much harder**.
  - **Why include it**: it's the "are there large, ugly mistakes?" detector. Because T
    explodes at saturation, RMSE surfaces whether a model fails badly in that tail.
  - **MAE vs RMSE**: if `RMSE ≫ MAE`, errors are dominated by a few large outliers (the
    saturated regime); if they're close, errors are evenly spread.

- **R² — Coefficient of Determination**
  - Formula: `1 − SS_res / SS_tot`, where `SS_res = Σ(y_true − y_pred)^2` and
    `SS_tot = Σ(y_true − mean(y_true))^2`.
  - Plain words: "what fraction of the ups-and-downs in T did I explain?"
    `1.0` = perfect; `0` = no better than always guessing the mean; **negative** = worse
    than guessing the mean.
  - **Why include it**: it's the **universal yardstick** — computable for *every* method
    (Octave-ANN, DEAP-GA, CMA-ES, XGBoost, LinReg) and a standard reviewer expectation.
  - **Deliberate design choices in `safe_r2`**:
    - **NOT clamped** — a negative R² is returned as-is on purpose (it's diagnostic).
    - **Zero-variance guard** — if a fold's true targets have zero variance (`SS_tot = 0`),
      returns `0.0` instead of dividing by zero (never happens with the shuffled 630 rows,
      but the code is safe).

- **"Bad cases" counter**: rows where `|signed % error| > 200%` are counted per fold.
  - **Why**: a ready-made shortlist of the ugliest outliers for regime/outlier analysis
    (matches the legacy PMS 200% threshold so counts are comparable).

- **Negative-prediction clamp**: predictions are clamped to `0` before metrics
  (`np.maximum(0, preds)`).
  - **Why**: denormalization floating-point edges can yield tiny negative times, which are
    physically impossible; clamping is applied identically everywhere for fairness.

---

## 7. AGGREGATION: TWO WAYS TO AVERAGE OVER THE 5 FOLDS (both reported)

- **POOLED** — concatenate all 630 predictions, compute **one** number over everything.
- **MEAN ± 95% CI** — average the **5 per-fold** numbers and attach a confidence range.
- **Why both**: for **MAPE/MAE/SMAPE** the two coincide (equal fold sizes); for
  **R²/RMSE** they differ slightly (non-linear math), so we always **state which one we
  quote**.
- **Confidence interval math**: `CI = t_(df=4, 0.975) · std / sqrt(n)` with the
  t-distribution across the 5 folds; for `n=5`, `df=4`, **t = 2.776** (hard-coded table,
  so no SciPy dependency).
- **Why a CI at all**: the GA/CMA-ES/MLP training are **stochastic**, so a single number
  is untrustworthy — multiple evaluations + a CI is the correct way to compare stochastic
  optimizers (cf. Bischl et al. 2023; Raschka 2018).

---

## 8. THE STATISTICAL PROTOCOL (identical for ALL methods → fairness)

- **TRUE 5-fold cross-validation (K=5)**: 630 rows cut into **5 disjoint folds**; each
  fold is the test set **exactly once**; every row is tested exactly once → **100%
  coverage, no overlap**.
  - **Why it replaced the old approach**: the earlier "repeated random 80/20 holdout"
    could test some rows twice and skip others — not true k-fold.
- **Fixed shuffle seed `random_state = 42`** in all three scripts → the **same 5 folds**
  (same 126 rows per fold) → results compared on **identical test sets**. (42 is arbitrary;
  what matters is it never changes.)
- **Nested structure (gold standard)**: inside each outer fold, an **inner 20% split**
  (`VAL_RATIO = 0.20`) of the 4 training folds scores candidate architectures during the
  search. The **outer test fold is never seen** during search/training/selection → an
  **honest** score. This is exactly the PMS "medval" idea, but now guaranteed disjoint.
- **Metrics**: MAPE (main) + SMAPE + MAE + RMSE + R², computed identically.
- **Uncertainty**: mean ± 95% t-CI across the 5 folds (df=4 → t=2.776).
- **Bad cases** logged when `|signed % error| > 200%`.

---

## 9. THE COMPARISON LADDER — CHANGE **ONE** THING AT A TIME

> The whole point is **clean attribution**: each step changes exactly one variable, so any
> difference in results has a **single, identifiable cause**.

```
Step 1 : Octave PMS = GA  + LM-trained MLP     (legacy baseline)
Step 2a: DEAP GA     = GA  + Adam-trained MLP   (same algorithm, modern Python)
Step 2b: CMA-ES      = ES  + Adam-trained MLP   (smarter search, same model)
Step 3 : XGBoost     = boosted trees            (different model family)
```

- **Step1 → Step2a** isolates the **implementation/platform** effect (same GA family).
- **Step1/2a → Step2b** isolates the **search-paradigm** effect (GA vs evolution strategy).
- **everything → Step3** isolates the **model-family** effect (neural net vs trees).

### 9.1 Vocabulary that must stay correct (these caused real confusion)
- **Evolutionary Computation** = the big family (nature-inspired population search).
- **Genetic Algorithm (GA)** = a member; improves by **selection + crossover + mutation**
  on chromosomes. **Octave PMS uses a GA.**
- **Evolution Strategy (ES)** = another member. **CMA-ES is an ES** — it has **NO
  crossover**; it samples from a probability "cloud" and adapts the cloud's **mean and
  covariance** toward better samples.
- **CMA-ES is therefore NOT a genetic algorithm** — it's a cousin (same family).
- **DEAP is NOT an algorithm** — it's a Python **framework/toolbox** for *building*
  evolutionary algorithms. We used DEAP to **build a GA**. Correct phrasing: *"a genetic
  algorithm implemented with the DEAP framework (Fortin et al., 2012)."*
- What we built in DEAP = a **generational GA with elitism** (tournament selection,
  two-point crossover, uniform-int mutation, Hall-of-Fame elitism) — essentially DEAP's
  `eaSimple` + elitism. Phrase it as a *"modern re-implementation of the same GA,"* not
  "a modern Octave."

### 9.2 The TWO levels inside a neural network (key conceptual point)
- The **GA / CMA-ES chooses the ARCHITECTURE** (how many layers, neurons, which
  activation) — the *size/shape of the brain*.
- The **TRAINER chooses the WEIGHTS** (Octave: **Levenberg-Marquardt**; Python: **Adam**)
  — the *wiring strengths*.
- **The GA does NOT pick the weights.** Architecture and weights are two separate jobs by
  two separate tools (a very common beginner confusion — state it explicitly).

### 9.3 Why we run BOTH DEAP GA and CMA-ES (and it's not bloat)
- **DEAP GA vs Octave** → answers *"was my Octave **implementation** the bottleneck?"*
- **CMA-ES vs the GAs** → answers *"is the GA **paradigm** itself the bottleneck?"*
- Two extra rows in the comparison table, clean attribution. We **explicitly rejected**
  multi-objective NSGA-II / Pareto fronts as over-complication for a thesis.

---

## 10. DEAP GA SCRIPT (`ga_mlp_runs.py`) — SELECTION LOGIC IN DETAIL

- **Role**: "the modern Octave" — a GA wired to **match Octave operator-for-operator**.
- **Chromosome** = `[n_layers, n1, n2, n3, n4, n5]` — gene 0 is layer count (1..5), genes
  1..5 are per-layer neuron counts (1..55). Only the first `n_layers` width genes are used.
  - **Why this encoding**: it mirrors Octave's **per-layer** encoding (variable depth +
    per-layer width).
- **Search space ceiling**: layers ∈ [1,5], neurons/layer ∈ [1,55] — the **"5-55"** ceiling
  (see §12 for why 55).
- **GA operators (the "genetics"), chosen to match Octave PMS**:
  - **Selection**: tournament, `tournsize = 3` (pick best of 3 random picks).
  - **Crossover**: two-point (`cxTwoPoint`), `CXPB = 0.8` — swap gene segments between two
    parents.
  - **Mutation**: uniform-int reset within bounds (`mutUniformInt`), `MUTPB = 0.2`
    (per-individual), `INDPB = 0.2` (per-gene).
  - **Elitism**: Hall-of-Fame(1) — the best-ever individual is forced back into each new
    generation so a good solution is **never lost**.
  - **Budget**: `POP = 30 × GEN = 30 = 900 evaluations/fold` (matches Octave's 30×30).
- **Fitness** = **validation MAPE** of the decoded network on the inner 20% split
  (minimisation; `weights = (-1.0,)`). Failed fits are penalised with `999.0`.
- **Final model**: retrain the best architecture on the **full** 80% training rows, then
  score on the untouched test fold.
- **Robustness fix we added**: `creator.create(...)` is **guarded** so re-importing the
  module in one process doesn't raise "class already exists."

---

## 11. CMA-ES SCRIPT (`cmaes_mlp_runs.py`) — SELECTION LOGIC IN DETAIL

- **Role**: the **smarter cousin** search (evolution strategy, not a GA).
- **Library**: `cma` (pycma).
- **"Ask/tell" loop**: `ask()` → get candidate architectures; **train + score each** on
  the inner validation split; `tell()` → CMA-ES **moves and reshapes its sampling cloud**
  (updates mean + covariance) toward the best. Repeat until budget/tolerances hit.
- **Continuous → integer decode trick**: CMA-ES works in continuous space, so it searches
  **two variables in [0,1]**:
  - `x[0] → n_layers = round(1 + x[0]·4)` → **1..5**
  - `x[1] → neurons = round(1 + x[1]·54)` → **1..55**
  - **All layers share the same neuron count** (a simplification vs the GA's per-layer
    encoding — this is a *documented, deliberate* architecture-space asymmetry, see §14).
- **Start point / step**: `x0 = [0.5, 0.5]` (≈ 3 layers, 30 neurons — middle of the space);
  initial step size `sigma0 = 0.3`; bounds `[0,1]²`; tolerances `tolx = tolfun = 1e-4`.
- **Budget**: `CMA_MAX_EVAL = 200 evaluations/fold`.
  - **Why 200 vs the GA's 900**: CMA-ES is **more sample-efficient** — 200 is enough for a
    2-D search. (This budget gap is a documented fairness caveat, §14.)
- **Fitness** = validation MAPE (CMA-ES **minimises**, so MAPE is a perfect objective).
- **Final model + scoring**: identical to the GA script (retrain best arch on full train,
  score on untouched test fold, same metrics/clamp).

---

## 12. THE MLP + TRAINER + ARCHITECTURE CEILING — WHY

- **Model**: `sklearn.MLPRegressor`, `activation = "relu"`, `solver = "adam"`,
  `max_iter = 2000`, `early_stopping = True`, `n_iter_no_change = 20`.
  - **Why early stopping**: holds back 10% internally to stop before overfitting.
  - **Why Adam**: the modern-Python trainer counterpart to Octave's Levenberg-Marquardt;
    the trainer difference is itself one of the things the comparison measures.
- **Why fixed `relu` in Python vs Octave's evolved activation**: Octave **evolves** the
  activation per layer (logsig/tansig/purelin as a gene); the Python scripts **fix relu**
  (standard, robust default). This difference is documented, not hidden.
- **Scaling to [-1, 1] (critical for fairness)**: inputs via `MinMaxScaler([-1,1])` inside
  a `Pipeline`, target via `TransformedTargetRegressor`.
  - **Why**: neural nets train badly on raw, wildly-scaled features (here **M~32 vs
    N~10000 vs T~52000**). Octave's `normalize.m` already scales inputs **and** target to
    [-1,1]; we reproduce it **exactly** so the Python ANNs aren't unfairly handicapped.
  - Predictions are auto-inverse-transformed back to real ms, so clamp/metrics are unchanged.
- **Architecture ceiling "5 layers, 55 neurons" (the "5-55")** — why 55:
  - Empirical neuron sweep: 30 → 17.4%, 38 → 17.98%, 45 → 16.11%, then 50, 55 improving
    with **shrinking gains**.
  - **55 is the "elbow"**: the point where the 95% CIs of consecutive runs first **overlap**
    → further growth is within noise.
  - **STOPPING RULE**: stop expanding neurons/layers once the new architecture's 95% CI
    overlaps the previous best's CI. Reporting *"5 layers, up to 55 neurons"* is more
    reproducible/defensible than chasing the single best observed number.
- **The MAPE < 20% "gate"**: PMS only *archives* a candidate net if validation MAPE < 20%.
  The Python scripts keep the same **20% threshold as a note** (log a warning if the best
  found exceeds it) — same concept, same comparability.

---

## 13. WHAT'S IDENTICAL vs DIFFERENT (Python scripts vs Octave)

- **IDENTICAL (deliberately, for fairness)**:
  - GA machinery in `ga_mlp_runs.py` (selection/crossover/mutation/elitism/budget).
  - TRUE 5-fold CV, metrics (MAPE/SMAPE/MAE/RMSE/R²), 95% t-CI.
  - Input + target scaling to [-1,1] (matches `normalize.m`).
  - Negative-prediction clamp; 200% bad-case threshold.
- **DIFFERENT (on purpose / unavoidable)** — the **fitness objective is the important one**:
  - **Octave**: the GA optimizes **training MSE** (`perf = thismse`) but **archives** nets
    by a **separate** gate (validation MAPE < 20%). *The thing it searches for and the
    thing it keeps are MISMATCHED.*
  - **Python**: the search is scored **directly on validation MAPE** — the same metric used
    to select and to report. One target, no contradiction. **ALIGNED.**
  - → This alignment is itself a **tested improvement**: *"did fixing PMS's objective
    mismatch help?"*
  - Trainer: **Adam** (Python) vs **Levenberg-Marquardt** (Octave).
  - Activation: fixed **relu** (Python) vs **evolved** logsig/tansig/purelin (Octave).

---

## 14. KNOWN, DELIBERATE CONFOUNDS (state these before the prof does)

- **Eval-budget confound**: GA **900** evals/fold vs CMA-ES **200** evals/fold. Real
  confound when attributing differences to the *search paradigm* — but an **intentional
  design choice** (CMA-ES is more sample-efficient). **Documented in the logs, not "fixed."**
- **Architecture-space asymmetry**: GA searches **per-layer** widths; CMA-ES searches a
  **single uniform** width. Deliberate; documented.
- **Trainer difference**: LM (Octave) vs Adam (Python) — one of the things being compared.
- **Scope honesty**: one workflow pattern, one platform, **n = 630** — manage with rigor
  and honest claims, don't overreach.

---

## 15. XGBoost — THE DIFFERENT MODEL FAMILY (Step 3, planned)

- **What it is**: XGBoost is **not** a search helper — it's a different **kind of
  predictor**: a team of small decision trees where each new tree corrects the previous
  trees' errors ("**gradient boosting**"). It **replaces the neural network**, not the GA.
- **Why it fits**: for small/medium **tabular** data (630 rows of S,N,M→T), boosted trees
  consistently **match or beat neural nets** and are the **expected reviewer baseline**
  (Grinsztajn et al. 2022; Holzmüller et al. 2024).
- **What it answers**: *"is the neural network even the right model, or do trees win?"*
- **Bonus**: XGBoost gives **feature importance** natively (neural nets don't) — perfect
  for showing which of S,N,M,K,C,U,B matter most.
- **Optional domain-aware extras (publishable angles)**: monotonic constraints (T rises
  with demand, falls with capacity), quantile/Tweedie objective for tail (p90/p99) error,
  leave-one-M-out generalization.
- **Optuna clarification (was a real confusion)**: Optuna is a **hyperparameter-optimization
  framework**, not a GA. Its default sampler (**TPE**) is **Bayesian**; it *also* offers a
  `CmaEsSampler` (ES) and `NSGAIISampler` (multi-objective GA). So "use Optuna" by default
  = **Bayesian**, a different paradigm. For a true single-objective GA that matches Octave,
  **DEAP** is the right tool. If a Bayesian leg is wanted, pair Optuna-TPE with XGBoost.

---

## 16. SUPPORTING STATISTICS TO STRENGTHEN RIGOR (planned)

- **Linear regression** (`T = a·S + b·N + c·M + intercept`): a dead-simple **"floor"
  baseline (Model 0)**. If the ANN/XGBoost can't beat it, complexity isn't earning its
  keep. Octave doesn't do this — add it as one extra table row.
- **R²**: the universal 0..1 yardstick, added as a column alongside MAPE/SMAPE/MAE/RMSE
  for every method (already implemented across all scripts).
- **p-value — two distinct roles**:
  1. **Justify calibration inputs**: fit a linear regression including K (or U/B/C); a tiny
     p-value on that coefficient is statistical **proof** the feature genuinely helps (not
     luck).
  2. **Compare models rigorously**: a **paired t-test on per-fold errors** gives a p-value
     → *"CMA-ES is significantly better than Octave (p=0.02)"*, stronger than "the CIs don't
     overlap."
- **CI overlap is a weak test** — the paired t-test is the real significance check.

---

## 17. RESULTS SO FAR (Octave PMS legacy baseline, 5-fold, 4-55 arch)

> Numbers below are from the current experiment logs (Octave PMS runner). Lower MAPE/SMAPE/
> MAE/RMSE = better; higher R² = better. MAPE/SMAPE quoted as **mean ± 95% CI** across folds;
> R²/MAE/RMSE from the **pooled** aggregation. (Python DEAP-GA / CMA-ES / XGBoost legs are the
> next runs to fill the same table.)

| Run (tag)     | What it adds / predicts        | MAPE (mean ±95% CI) | SMAPE (mean) | Pooled R² | Pooled MAE (ms) | Pooled RMSE (ms) |
|---------------|--------------------------------|---------------------|--------------|-----------|-----------------|------------------|
| **NSMT**      | baseline (S,N,M → T)           | 17.27% ± 3.58%      | 15.43%       | 0.844     | 3244.9          | 11659.3          |
| **NSMKT**     | + invocation count K=ceil(N/S) | **13.26% ± 3.33%**  | **12.03%**   | **0.931** | **2500.2**      | **7777.8**       |
| **NSMUT**     | + utilization U=K/M            | 16.60% ± 6.12%      | 14.31%       | 0.891     | 3253.3          | 9753.0           |
| **NSMBT**     | + wave count B=ceil(K/M)       | 15.17% ± 2.76%      | 13.71%       | 0.829     | 3052.1          | 12207.4          |
| **NSMD**      | predict duration D             | 13.95% ± 5.24%      | 12.68%       | 0.918     | 171.5           | 261.0            |
| **NSM_T-B**   | predict T_wave = T/B           | 14.07% ± 1.24%      | 13.83%       | 0.814     | 255.1           | 423.1            |

- **Take-aways to say on the call**:
  - **`NSMKT` (adding K) is the clear winner so far** — best MAPE (13.26%), best SMAPE,
    best R² (0.931), lowest MAE **and** lowest RMSE. Strong evidence that **invocation
    count is the most useful physics hint**.
  - `NSMUT` (utilization) and `NSMBT` (waves) help on some metrics but not consistently;
    `NSMBT` actually has the **worst RMSE**, i.e. it still misses badly in the tail.
  - `NSMD` (predicting the compute component directly) gets a high R² (0.918) and tiny MAE/
    RMSE **in D's units** — consistent with the hypothesis that **components are smoother/
    easier to predict** than the messy total T.
  - `NSM_T-B` (T_wave) has the **tightest MAPE CI (±1.24%)** — the smoothing helps
    stability, even if pooled R² is lower (different target scale).
  - ⚠️ Datasets `NSMCT` and `NSMW` exist but **haven't been run/logged yet**; MAE/RMSE for
    D/T_wave rows are in **different units** than T rows, so don't compare those columns
    across targets — compare **within** a target and use **R²/MAPE** across targets.

---

## 18. THE RUN TABLE / NAMING CONVENTION

- Naming = `[inputs][target]`, uppercase, in CSV column order. The scripts **always predict
  the last column** and use earlier columns as inputs → to test an idea you **swap the CSV**,
  you don't edit code.

| Run    | Inputs (CSV cols)   | Target       | Isolates                                  |
|--------|---------------------|--------------|-------------------------------------------|
| NSMT   | S, N, M             | T            | baseline                                  |
| NSMKT  | S, N, M, K          | T            | does invocation count help?               |
| NSMCT  | S, N, M, C          | T            | does avg concurrency beat raw count?      |
| NSMUT  | S, N, M, U          | T            | does utilization help?                    |
| NSMBT  | S, N, M, B          | T            | continuous U vs discrete wave structure   |
| NSMD   | S, N, M             | D            | is the compute component more predictable?|
| NSMW   | S, N, M             | W            | is the wait/queue component the learnable core? |
| NSMTw  | S, N, M             | T_wave = T/B | does wave structure explain most of T?    |

---

## 19. THREE FAMILIES OF EXPERIMENTS (the roadmap in one glance)

- **Family A — add a clue, still predict T**: `NSMKT / NSMUT / NSMCT / NSMBT`. One extra
  derived input at a time (§4.3). *Runnable now* via a dataset generator.
- **Family B — change the target**: `NSMTw` (predict T/B; no new data needed), plus `NSMD`
  (compute time) and `NSMW` (wait/queue time). **W is where the saturation blow-up lives**,
  so it likely carries the most insight. **Blocked** until D and W are exported into CSVs
  (current `master_dataset.csv` has only S,N,M,T).
  - ⚠️ **max-vs-average caution** (same trap that killed H): T is a **maximum** (critical
    path) while D/W are typically reported as **averages** across chunks — decide and
    **document** exactly which statistic you export (mean vs critical-path) and keep it
    consistent.
- **Family C — two-stage cascade**: `NSM → W^`, then `NSM, W^ → T^` (optionally with D^).
  - **Why it can win**: if Stage 1 predicts the wait well, Stage 2 gets the single most
    explanatory signal handed to it instead of rediscovering queueing from raw N,S,M; it
    also mirrors the real physics.
  - **The honesty rules (beginners get this wrong)**: **never feed the *true* W** into
    Stage 2 (that leaks the answer); feed **out-of-fold predicted W^**; keep the same
    5-fold protocol; **errors compound**, so only keep the cascade if it **beats direct
    NSM→T** head-to-head (ideally with a passing paired t-test).

---

## 20. IMMEDIATE NEXT STEPS (what I'll do after this call)

1. **Run the Python legs** (DEAP GA, CMA-ES) on the same CSVs to fill the comparison table
   next to the Octave numbers.
2. **Run `NSMCT`** (already have the dataset) and generate/run the rest of Family A.
3. **Export D and W** from OpenWhisk activation records (choosing & documenting mean vs
   critical-path) → unlock `NSMD` / `NSMW` / cascade.
4. Build `NSMTw = T/B` runs (no new data needed).
5. Add **XGBoost + feature importance** and a **linear-regression floor + R²** (Step 3 / §16).
6. Add the **paired t-test** on per-fold errors for real p-values.
7. For a stronger venue: **leave-one-M-out generalization** + a **provisioning demo**
   ("pick N,S,M to hit a target T, then verify").
8. Assemble the **master comparison table**: rows = methods (Octave-GA, DEAP-GA, CMA-ES,
   XGBoost, LinReg) × input/target families; columns = MAPE, SMAPE, MAE, RMSE, R² (each
   mean ± 95% CI).

---

## 21. LIKELY PROF QUESTIONS — QUICK ANSWERS

- **"Is CMA-ES a GA?"** → No. It's an **evolution strategy** (same family, **no
  crossover**; it adapts a sampling distribution).
- **"Is DEAP an algorithm?"** → No. It's a **framework**; we built a GA with it.
- **"Why 5 folds / seed 42?"** → True k-fold = disjoint folds, full coverage; seed 42 fixes
  the shuffle so all methods use identical folds. 42 is arbitrary but fixed.
- **"Why not combine all calibration inputs?"** → Collinearity + attribution + small n
  (overfitting risk).
- **"Why did you delete H?"** → It mixed a **max (T)** with **averages (D, W)** plus a fixed
  polling delay — a measurement artifact. Replaced by **T_wave = T/B**.
- **"Why is C = K/B and not min(K,M)?"** → min(K,M) overstates the **partial last wave**;
  the true average over all B waves is K/B.
- **"Is the comparison fair given GA=900 vs CMA-ES=200 evals?"** → It's a **documented
  confound** (CMA-ES is more sample-efficient by design), reported in the logs, not hidden.
- **"Why MLP + Adam if XGBoost may win?"** → The MLP keeps the comparison anchored to the
  legacy PMS neural net; XGBoost is Step 3 precisely to test whether **trees beat nets** on
  this tabular data.
- **"Where's the novelty?"** → The **controlled attribution** (optimizer vs search paradigm
  vs model family) + **queueing-theory calibration features** on a **real** Node-RED/
  OpenWhisk split-join workflow — an underfilled gap in the FaaS literature.

---

## 22. FILES MAP (where everything lives)

- `pms_research_master.md` — the full master research doc (source of truth).
- `README_fold_csvs.md` — what every per-fold CSV means (true_y / heldout / finalval / bad_mape).
- `conversation_summary.txt` — plain-words handoff (metrics, seed, how each script works).
- `pms_kfold.py` — legacy **Octave PMS** k-fold harness (needs Docker + Octave + pekoto paths).
- `pms_runs.py` — Octave PMS repeated-runs harness (5 repeats on the full dataset).
- `ga_mlp_runs.py` — **DEAP GA + MLP**, true 5-fold, scaled (§10).
- `cmaes_mlp_runs.py` — **CMA-ES + MLP**, true 5-fold, scaled (§11).
- `master_dataset.csv` + `master_dataset_<TAG>.csv` — the 630-row dataset and its variants.
- `experiment_logs_4-55_<TAG>.txt` — result logs per variant (source of §17 numbers).
- `specs/faas-surrogate-comparison.md` — the hardening spec for the two Python scripts.
- `specs/next-steps-and-interpretation.md` — Families A/B/C roadmap + how to read results.
- `specs/build-review-log.md` — the /spec → /build → /review loop record.

--- END OF CHEAT SHEET ---
