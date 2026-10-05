#!/usr/bin/env python3
"""
Leave-one-M-out surrogate run (CMA-ES + MLP).

Train on all rows with M != M_hold; test on rows with M == M_hold.
Same inner train/medval/val search as cmaes_mlp_runs.py.
"""
import math
import os
import random
import datetime

import numpy as np
import pandas as pd

import cmaes_mlp_runs as c

M_COL = 2
RUN_SUFFIX = "_m_out"


def run_cmaes_leave_one_m_out():
    if c.SEED is not None:
        random.seed(c.SEED)
        np.random.seed(c.SEED)

    df = pd.read_csv(c.DATA_FILE, header=None)
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    m_values = sorted(df.iloc[:, M_COL].unique())
    n_splits = len(m_values)

    num_inputs = X.shape[1]
    log_file = f"experiment_logs_{c.MODEL_TAG}_cmaes{RUN_SUFFIX}.txt"
    results_csv = f"kfold_results_{c.MODEL_TAG}_cmaes{RUN_SUFFIX}.csv"
    heldout_csv = f"heldout_predictions_{c.MODEL_TAG}_cmaes{RUN_SUFFIX}.csv"

    print(f"Dataset  : {c.DATA_FILE}  ({len(df)} rows, {num_inputs} inputs)")
    print(f"Protocol : leave-one-M-out ({n_splits} holds)  |  tag: {c.MODEL_TAG}")
    print(f"Search   : CMA-ES {c.CMA_MAX_EVAL} evals per hold")

    os.makedirs(c.RESULTS_DIR, exist_ok=True)

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

        (X_train, y_train, X_medval, y_medval, X_val, y_val) = c.inner_split(
            X_tr_fold, y_tr_fold, c.SEED
        )

        print("  [CMA-ES] Searching architecture (objective = training MSE)...")
        best_model, best_layers, best_neurons, best_act, num_saved = c.cmaes_search(
            X_train, y_train, X_medval, y_medval, X_val, y_val
        )

        if best_model is None:
            print("  WARNING: no candidate could be trained this hold; skipping.")
            continue

        print(
            f"  Best architecture (selected on val): "
            f"{best_layers} layers × {best_neurons} neurons, act={best_act}"
        )
        print(f"  Candidates saved (medval MAPE < {c.MAPE_GATE:.0f}%): {num_saved}")

        preds = best_model.predict(X_test_fold)
        preds = np.maximum(0.0, preds)

        mape_val = c.safe_mape(y_test_fold, preds)
        smape_val = c._smape(y_test_fold, preds)
        mae_val = c._mae(y_test_fold, preds)
        rmse_val = c._rmse(y_test_fold, preds)
        r2_val = c.safe_r2(y_test_fold, preds)
        n_test = len(y_test_fold)

        signed_pct = (preds - y_test_fold) / np.maximum(np.abs(y_test_fold), 1e-8) * 100
        bad_count = int(np.sum(np.abs(signed_pct) > 200))

        print(f"\n  External MAPE  (M={int(m_hold)}, {n_test} rows): {mape_val:.2f}%")
        print(f"  External R2    : {r2_val:.4f}")

        fold_results.append({
            "fold": fold_idx,
            "M_hold": int(m_hold),
            "n_layers": best_layers,
            "neurons": best_neurons,
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

    all_actuals = np.asarray(all_actuals, dtype=float)
    all_preds = np.asarray(all_preds, dtype=float)
    pool_mape = c.safe_mape(all_actuals, all_preds)
    pool_r2 = c.safe_r2(all_actuals, all_preds)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_text = (
        f"\n{'=' * 70}\n"
        f"RUN TIMESTAMP : {now}\n"
        f"CMA-ES leave-one-M-out | tag: {c.MODEL_TAG}\n"
        f"POOLED MAPE : {pool_mape:.2f}%  R2: {pool_r2:.4f}\n"
        f"AVG MAPE across holds: {avg_mape:.2f}%\n"
        f"{'=' * 70}\n"
    )
    print(log_text)
    with open(log_file, "a") as f:
        f.write(log_text)

    pd.DataFrame(fold_results).to_csv(
        os.path.join(c.RESULTS_DIR, results_csv), index=False
    )
    held_path = os.path.join(c.RESULTS_DIR, heldout_csv)
    pd.concat(heldout_parts, ignore_index=True).to_csv(
        held_path, index=False, header=False
    )
    print(f"Held-out predictions : {held_path}")


if __name__ == "__main__":
    run_cmaes_leave_one_m_out()
