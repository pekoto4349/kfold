#!/usr/bin/env python3
"""
Sat/unsat regime analysis from row-level held-out prediction CSVs.

Saturated: K = ceil(N/S) > M.  Unsaturated otherwise.

Input CSVs have no header (same layout as pms_kfold heldout files):
  S, N, M, [extra inputs...], T, prediction, fold

Works on Octave heldouts, ga_results/heldout_predictions_*_ga.csv,
cmaes heldouts, or *_m_out heldout files — no rerun needed once preds exist.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_heldout(path):
    raw = pd.read_csv(path, header=None)
    ncol = raw.shape[1]
    if ncol < 6:
        raise ValueError(f"{path}: expected at least 6 columns, got {ncol}")
    # Last two columns are always prediction, fold (ga/cmaes/pms convention).
    t_col = ncol - 3
    names = ["S", "N", "M"]
    for i in range(3, t_col):
        names.append(f"x{i}")
    names += ["T_actual", "prediction", "fold"]
    raw.columns = names
    return raw


def load_octave_glob(base_models_dir, neurons_layers, model_tag):
    pattern = os.path.join(
        base_models_dir,
        f"model_Dynamic_Fold*_{neurons_layers}_{model_tag}",
        "heldout_predictions_FOLD*.csv",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        pattern2 = os.path.join(
            base_models_dir,
            f"model_Dynamic_Fold*_{neurons_layers}",
            "heldout_predictions_FOLD*.csv",
        )
        files = sorted(glob.glob(pattern2))
    if not files:
        raise FileNotFoundError(f"No Octave heldout files for {neurons_layers}")
    return pd.concat([load_heldout(f) for f in files], ignore_index=True)


def regime_label(s, n, m):
    k = np.ceil(np.asarray(n, float) / np.maximum(np.asarray(s, float), 1e-8))
    return k > np.asarray(m, float)


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred)
    mask = den > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * num[mask] / den[mask]) * 100)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))))


def rmse(y_true, y_pred):
    diff = np.asarray(y_true, float) - np.asarray(y_pred, float)
    return float(np.sqrt(np.mean(diff ** 2)))


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


def enrich(df):
    out = df.copy()
    out["K"] = np.ceil(out["N"] / np.maximum(out["S"], 1e-8))
    out["saturated"] = regime_label(out["S"], out["N"], out["M"])
    out["abs_err"] = np.abs(out["T_actual"] - out["prediction"])
    out["pct_err"] = (
        (out["prediction"] - out["T_actual"])
        / np.maximum(np.abs(out["T_actual"]), 1e-8)
        * 100
    )
    out["abs_pct_err"] = np.abs(out["pct_err"])
    return out


def metrics_block(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return {"n": 0, "mape_pct": float("nan"), "smape_pct": float("nan"),
                "mae": float("nan"), "rmse": float("nan"), "r2": float("nan")}
    return {
        "n": n,
        "mape_pct": safe_mape(y_true, y_pred),
        "smape_pct": smape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "r2": safe_r2(y_true, y_pred),
    }


def write_summary(path, label, overall, sat, unsat):
    lines = [
        f"Regime analysis: {label}",
        f"Rows total: {overall['n']}",
        "",
        "OVERALL",
        f"  MAPE  {overall['mape_pct']:.2f}%",
        f"  SMAPE {overall['smape_pct']:.2f}%",
        f"  MAE   {overall['mae']:.1f} ms",
        f"  RMSE  {overall['rmse']:.1f} ms",
        f"  R2    {overall['r2']:.4f}",
        "",
        f"SATURATED (K > M), n={sat['n']}",
        f"  MAPE  {sat['mape_pct']:.2f}%",
        f"  SMAPE {sat['smape_pct']:.2f}%",
        f"  MAE   {sat['mae']:.1f} ms",
        f"  RMSE  {sat['rmse']:.1f} ms",
        f"  R2    {sat['r2']:.4f}",
        "",
        f"UNSATURATED (K <= M), n={unsat['n']}",
        f"  MAPE  {unsat['mape_pct']:.2f}%",
        f"  SMAPE {unsat['smape_pct']:.2f}%",
        f"  MAE   {unsat['mae']:.1f} ms",
        f"  RMSE  {unsat['rmse']:.1f} ms",
        f"  R2    {unsat['r2']:.4f}",
        "",
    ]
    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser(description="Sat/unsat metrics from held-out preds")
    ap.add_argument("--heldout", help="Path to one held-out CSV (no header)")
    ap.add_argument("--octave", action="store_true", help="Load Octave fold heldouts")
    ap.add_argument("--ga", metavar="TAG", help="e.g. NSMT -> ga_results/heldout_*_TAG_ga.csv")
    ap.add_argument("--cmaes", metavar="TAG", help="Same for cmaes_results")
    ap.add_argument("--m-out", action="store_true", help="Use *_m_out.csv suffix with --ga/--cmaes")
    ap.add_argument("--neurons-layers", default="4-55")
    ap.add_argument("--model-tag", default="NSMT")
    ap.add_argument("--base-models-dir", default="/home/pekoto/models")
    ap.add_argument("--out-dir", default="regime_results")
    ap.add_argument("--label", default="heldout")
    args = ap.parse_args()

    if args.octave:
        df = load_octave_glob(args.base_models_dir, args.neurons_layers, args.model_tag)
        label = f"octave_{args.neurons_layers}_{args.model_tag}"
    elif args.ga:
        name = f"heldout_predictions_{args.ga}_ga_m_out.csv" if args.m_out else f"heldout_predictions_{args.ga}_ga.csv"
        path = os.path.join("ga_results", name)
        if not os.path.isfile(path) and args.m_out:
            path = os.path.join("ga_results", f"heldout_predictions_{args.ga}_ga.csv")
        df = load_heldout(path)
        label = args.label if args.label != "heldout" else f"ga_{args.ga}"
    elif args.cmaes:
        name = (
            f"heldout_predictions_{args.cmaes}_cmaes_m_out.csv"
            if args.m_out
            else f"heldout_predictions_{args.cmaes}_cmaes.csv"
        )
        path = os.path.join("cmaes_results", name)
        if not os.path.isfile(path) and args.m_out:
            path = os.path.join("cmaes_results", f"heldout_predictions_{args.cmaes}_cmaes.csv")
        df = load_heldout(path)
        label = args.label if args.label != "heldout" else f"cmaes_{args.cmaes}"
    elif args.heldout:
        df = load_heldout(args.heldout)
        label = args.label
    else:
        ap.error("Provide --heldout, --octave, --ga TAG, or --cmaes TAG")

    enriched = enrich(df)
    os.makedirs(args.out_dir, exist_ok=True)

    sat_df = enriched[enriched["saturated"]]
    unsat_df = enriched[~enriched["saturated"]]

    sat_path = os.path.join(args.out_dir, f"{label}_saturated.csv")
    unsat_path = os.path.join(args.out_dir, f"{label}_unsaturated.csv")
    all_path = os.path.join(args.out_dir, f"{label}_all_rows.csv")
    enriched.to_csv(all_path, index=False)
    sat_df.to_csv(sat_path, index=False)
    unsat_df.to_csv(unsat_path, index=False)

    ya, yp = enriched["T_actual"], enriched["prediction"]
    overall = metrics_block(ya, yp)
    sat_m = metrics_block(sat_df["T_actual"], sat_df["prediction"]) if len(sat_df) else metrics_block([], [])
    unsat_m = metrics_block(unsat_df["T_actual"], unsat_df["prediction"]) if len(unsat_df) else metrics_block([], [])

    summary_path = os.path.join(args.out_dir, f"{label}_summary.txt")
    write_summary(summary_path, label, overall, sat_m, unsat_m)
    print(f"Wrote {all_path}")
    print(f"Wrote {sat_path}")
    print(f"Wrote {unsat_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
