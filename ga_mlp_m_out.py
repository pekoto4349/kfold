#!/usr/bin/env python3
"""
Leave-one-M-out surrogate run (GA + MLP).

Train on all rows with M != M_hold; test on rows with M == M_hold.
Same inner train/medval/val search as ga_mlp_runs.py; only the outer split changes.

Edit DATA_FILE / MODEL_TAG in ga_mlp_runs.py (or set env) before running — this
module reuses that config via import.
"""
import math
import os
import random
import datetime

import numpy as np
import pandas as pd

import ga_mlp_runs as g

M_COL = 2
RUN_SUFFIX = "_m_out"


def run_ga_leave_one_m_out():
    if g.SEED is not None:
        random.seed(g.SEED)
        np.random.seed(g.SEED)

    df = pd.read_csv(g.DATA_FILE, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    m_values = sorted(df.iloc[:, M_COL].unique())
    n_splits = len(m_values)

    num_inputs = X.shape[1]
    log_file = f"experiment_logs_{g.MODEL_TAG}_ga{RUN_SUFFIX}.txt"
    results_csv = f"kfold_results_{g.MODEL_TAG}_ga{RUN_SUFFIX}.csv"
    heldout_csv = f"heldout_predictions_{g.MODEL_TAG}_ga{RUN_SUFFIX}.csv"

    print(f"Dataset  : {g.DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"Protocol : leave-one-M-out ({n_splits} holds)  |  tag: {g.MODEL_TAG}")
    print(f"Search   : GA pop={g.POP_SIZE} x gen={g.N_GEN} evals per hold")

    os.makedirs(g.RESULTS_DIR, exist_ok=True)

    fold_results = []
    all_actuals = []
    all_preds = []
    heldout_parts = []

    for fold_idx, m_hold in enumerate(m_values, start=1):
        print("\n" + "=" * 70)
        print(f"  M_HOLD {int(m_hold)}  ({fold_idx}/{n_splits})")
        print("=" * 70)

        train_index = np.where(df.iloc[:, M_COL].values != m_hold)[0]
        test_index = np.where(df.iloc[:, M_COL].values == m_hold)[0]

        X_tr_fold, X_test_fold = X[train_index], X[test_index]
        y_tr_fold, y_test_fold = y[train_index], y[test_index]

        (X_train, y_train, X_medval, y_medval, X_val, y_val) = g.inner_split(
            X_tr_fold, y_tr_fold, g.SEED
        )

        print("  [GA] Evolving architecture (fitness = training MSE)...")
        best_model, best_widths, n_layers, best_act, num_saved = g.ga_search(
            X_train, y_train, X_medval, y_medval, X_val, y_val
        )

        if best_model is None:
            print("  WARNING: no candidate could be trained this hold; skipping.")
            continue

        print(
            f"  Best architecture (selected on val): "
            f"{n_layers} layers, widths {best_widths}, act={best_act}"
        )
        print(f"  Candidates saved (medval MAPE < {g.MAPE_GATE:.0f}%): {num_saved}")

        preds = best_model.predict(X_test_fold)
        preds = np.maximum(0.0, preds)

        mape_val = g.safe_mape(y_test_fold, preds)
        smape_val = g._smape(y_test_fold, preds)
        mae_val = g._mae(y_test_fold, preds)
        rmse_val = g._rmse(y_test_fold, preds)
        r2_val = g.safe_r2(y_test_fold, preds)
        n_test = len(y_test_fold)

        signed_pct = (preds - y_test_fold) / np.maximum(np.abs(y_test_fold), 1e-8) * 100
        bad_count = int(np.sum(np.abs(signed_pct) > 200))

        print(f"\n  External MAPE  (M={int(m_hold)}, {n_test} rows): {mape_val:.2f}%")
        print(f"  External SMAPE : {smape_val:.2%}")
        print(f"  External MAE   : {mae_val:.1f} ms")
        print(f"  External RMSE  : {rmse_val:.1f} ms")
        print(f"  External R2    : {r2_val:.4f}")

        fold_results.append({
            "fold": fold_idx,
            "M_hold": int(m_hold),
            "n_layers": n_layers,
            "widths": "-".join(str(w) for w in best_widths),
            "activation": best_act,
            "candidates": num_saved,
            "mape": mape_val,
            "smape": smape_val,
            "mae": mae_val,
            "rmse": rmse_val,
            "r2": r2_val,
            "n_test": n_test,
            "bad_count": bad_count,
        })
        all_actuals.extend(y_test_fold.tolist())
        all_preds.extend(preds.tolist())

        block = df.iloc[test_index].copy()
        block["prediction"] = preds
        block["fold"] = int(m_hold)
        heldout_parts.append(block)

    if not fold_results:
        print("No successful holds — nothing to report.")
        return

    T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
              5: 2.571, 7: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}

    def _t_crit(n):
        return T_CRIT.get(n - 1, 1.96)

    def _avg_std_ci(key):
        vals = [r[key] for r in fold_results]
        n = len(vals)
        avg = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci = _t_crit(n) * std / math.sqrt(n) if n > 1 else 0.0
        return avg, std, ci

    avg_mape, std_mape, ci_mape = _avg_std_ci("mape")
    avg_smape, std_smape, ci_smape = _avg_std_ci("smape")
    avg_mae, std_mae, ci_mae = _avg_std_ci("mae")
    avg_rmse, std_rmse, ci_rmse = _avg_std_ci("rmse")
    avg_r2, std_r2, ci_r2 = _avg_std_ci("r2")

    all_actuals = np.asarray(all_actuals, dtype=float)
    all_preds = np.asarray(all_preds, dtype=float)
    n_pool = len(all_actuals)
    pool_mape = g.safe_mape(all_actuals, all_preds)
    pool_smape = g._smape(all_actuals, all_preds)
    pool_mae = g._mae(all_actuals, all_preds)
    pool_rmse = g._rmse(all_actuals, all_preds)
    pool_r2 = g.safe_r2(all_actuals, all_preds)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fold_lines = "\n".join(
        f"  M={r['M_hold']}: arch={r['n_layers']}L[{r['widths']}] act={r['activation']}  "
        f"MAPE={r['mape']:.2f}%  n={r['n_test']}  candidates={r['candidates']}"
        for r in fold_results
    )

    log_text = (
        f"\n{'=' * 70}\n"
        f"RUN TIMESTAMP : {now}\n"
        f"DATA FILE     : {g.DATA_FILE}\n"
        f"MODEL         : GA + MLP (leave-one-M-out)\n"
        f"CONFIG        : {num_inputs} inputs | {n_splits} M holds | tag: {g.MODEL_TAG}\n"
        f"{'─' * 70}\n"
        f"{fold_lines}\n"
        f"{'─' * 70}\n"
        f"POOLED over all {n_pool} rows:\n"
        f"  POOL MAPE : {pool_mape:.2f}%\n"
        f"  POOL R2   : {pool_r2:.4f}\n"
        f"MEAN ± 95% t-CI ACROSS {n_splits} M HOLDS:\n"
        f"AVG MAPE  : {avg_mape:.2f}% (std ± {std_mape:.2f}% | CI ± {ci_mape:.2f}%)\n"
        f"{'=' * 70}\n"
    )
    print(log_text)
    with open(log_file, "a") as f:
        f.write(log_text)

    pd.DataFrame(fold_results).to_csv(
        os.path.join(g.RESULTS_DIR, results_csv), index=False
    )
    held_path = os.path.join(g.RESULTS_DIR, heldout_csv)
    pd.concat(heldout_parts, ignore_index=True).to_csv(
        held_path, index=False, header=False
    )
    print(f"Held-out predictions : {held_path}")
    print(f"Log : {log_file}")
    print(f"CSV : {os.path.join(g.RESULTS_DIR, results_csv)}")


if __name__ == "__main__":
    run_ga_leave_one_m_out()
