import os
import glob
import numpy as np
import pandas as pd


# ── CONFIG ──────────────────────────────────────────────────────────
BASE_MODELS_DIR = "/home/pekoto/models"
NEURONS_LAYERS  = "5-45"          # change to match the run you want
# ────────────────────────────────────────────────────────────────────




# ── STEP 1: find and load heldout files ─────────────────────────────
pattern = os.path.join(
    BASE_MODELS_DIR,
    f"model_Dynamic_Fold*_{NEURONS_LAYERS}",
    "heldout_predictions_FOLD*.csv"
)
files = sorted(glob.glob(pattern))
print(f"Files found: {len(files)}")
for f in files:
    print(f"  {f}")
if not files:
    raise FileNotFoundError(
        "No heldout files found. Check BASE_MODELS_DIR and NEURONS_LAYERS.")


# master_dataset.csv column order: S, N, M, T
frames = []
for f in files:
    df = pd.read_csv(f, header=None,
                     names=["S", "N", "M", "T_actual", "T_predicted", "fold"])
    frames.append(df)
all_held = pd.concat(frames, ignore_index=True)
print(f"Total held-out rows: {len(all_held)}\n")




# ── STEP 2: metric helpers ───────────────────────────────────────────
def mape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    return np.mean(np.abs(a - p) / np.maximum(np.abs(a), 1e-8))


def smape(a, p):
    a, p = np.asarray(a, float), np.asarray(p, float)
    num = np.abs(a - p)
    den = (np.abs(a) + np.abs(p)) / 2.0
    mask = den > 0
    return np.mean(num[mask] / den[mask]) if np.any(mask) else 0.0


def mae(a, p):
    return np.mean(np.abs(np.asarray(a, float) - np.asarray(p, float)))


def rmse(a, p):
    return np.sqrt(np.mean((np.asarray(a, float) - np.asarray(p, float))**2))




# ── STEP 3: overall ──────────────────────────────────────────────────
a_all = all_held["T_actual"].values
p_all = all_held["T_predicted"].values


print("=" * 70)
print(f"HELDOUT ANALYSIS  —  architecture: {NEURONS_LAYERS}")
print("=" * 70)
print(f"\n  OVERALL ({len(all_held)} rows)")
print(f"    MAPE  : {mape(a_all, p_all):.2%}")
print(f"    SMAPE : {smape(a_all, p_all):.2%}")
print(f"    MAE   : {mae(a_all, p_all):.1f} ms")
print(f"    RMSE  : {rmse(a_all, p_all):.1f} ms")




# ── STEP 4: per M (max containers — resource capacity dimension) ─────
print(f"\n{'─'*70}")
print("  PER M VALUE  (resource capacity)")
print(f"  {'M':>4}  {'rows':>5}  {'T median':>10}  {'MAPE':>8}  "
      f"{'SMAPE':>8}  {'MAE (ms)':>10}")
print(f"  {'─'*4}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
for m_val, grp in sorted(all_held.groupby("M")):
    a, p = grp["T_actual"].values, grp["T_predicted"].values
    print(f"  M={m_val:2d}  {len(grp):5d}  "
          f"{np.median(a):10.0f}  "
          f"{mape(a,p):8.2%}  "
          f"{smape(a,p):8.2%}  "
          f"{mae(a,p):10.1f}")




# ── STEP 5: per S (split size — granularity dimension) ──────────────
print(f"\n{'─'*70}")
print("  PER S VALUE  (split size / granularity)")
print(f"  {'S':>4}  {'rows':>5}  {'T median':>10}  {'MAPE':>8}  "
      f"{'SMAPE':>8}  {'MAE (ms)':>10}")
print(f"  {'─'*4}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
for s_val, grp in sorted(all_held.groupby("S")):
    a, p = grp["T_actual"].values, grp["T_predicted"].values
    print(f"  S={s_val:3d}  {len(grp):5d}  "
          f"{np.median(a):10.0f}  "
          f"{mape(a,p):8.2%}  "
          f"{smape(a,p):8.2%}  "
          f"{mae(a,p):10.1f}")




# ── STEP 6: per N range (workload size bands) ────────────────────────
print(f"\n{'─'*70}")
print("  PER N BAND  (workload size)")
bins   = [0, 1000, 3000, 6000, 10200]
labels = ["small (100-1000)", "medium (1001-3000)",
          "large (3001-6000)", "xlarge (6001-10100)"]
all_held["N_band"] = pd.cut(all_held["N"], bins=bins, labels=labels)
print(f"  {'N band':<22}  {'rows':>5}  {'T median':>10}  {'MAPE':>8}  "
      f"{'SMAPE':>8}  {'MAE (ms)':>10}")
print(f"  {'─'*22}  {'─'*5}  {'─'*10}  {'─'*8}  {'─'*8}  {'─'*10}")
for band, grp in all_held.groupby("N_band", observed=True):
    a, p = grp["T_actual"].values, grp["T_predicted"].values
    print(f"  {str(band):<22}  {len(grp):5d}  "
          f"{np.median(a):10.0f}  "
          f"{mape(a,p):8.2%}  "
          f"{smape(a,p):8.2%}  "
          f"{mae(a,p):10.1f}")




# ── STEP 7: worst 10 individual predictions ──────────────────────────
print(f"\n{'─'*70}")
print("  WORST 10 INDIVIDUAL PREDICTIONS (by abs % error)")
all_held["abs_pct_err"] = (
    np.abs(all_held["T_actual"] - all_held["T_predicted"])
    / np.maximum(np.abs(all_held["T_actual"]), 1e-8) * 100
)
worst = all_held.nlargest(10, "abs_pct_err")[
    ["S", "N", "M", "T_actual", "T_predicted", "abs_pct_err", "fold"]
].reset_index(drop=True)
print(worst.to_string(
    index=False,
    formatters={
        "T_actual":    "{:>10.1f}".format,
        "T_predicted": "{:>12.1f}".format,
        "abs_pct_err": "{:>10.1f}%".format,
    }
))
print("=" * 70)