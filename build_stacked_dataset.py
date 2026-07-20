import os
import glob
import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────
BASE_MODELS_DIR   = "/home/pekoto/models"

# Tag used in Run 1 (the SNM → W kfold run)
W_MODEL_TAG       = "SNM-W-5-45"

# Original full dataset: columns S, N, M, T  (used to recover T)
MASTER_DATASET    = "master_dataset.csv"

# Output file that will be used as DATA_FILE in Run 2
OUTPUT_FILE       = "snmwhat_t.csv"
# ─────────────────────────────────────────────────────────────────────


# ── STEP 1: load OOF W_hat from Run 1 heldout files ─────────────────
# heldout columns (written by pms_kfold.py, header=False, index=False):
#   col 0 = S
#   col 1 = N
#   col 2 = M
#   col 3 = W_actual   (the real measured W from snm_w.csv)
#   col 4 = W_hat_OOF  (what PMS predicted for that held-out row)
#   col 5 = fold_index

pattern = os.path.join(
    BASE_MODELS_DIR,
    f"model_Dynamic_Fold*_{W_MODEL_TAG}",
    "heldout_predictions_FOLD*.csv"
)
files = sorted(glob.glob(pattern))

print(f"Looking for Run 1 heldout files with tag: {W_MODEL_TAG}")
print(f"Found: {len(files)} files")
for f in files:
    print(f"  {f}")

if not files:
    raise FileNotFoundError(
        "No heldout files found for the W model run. "
        "Make sure Run 1 (SNM -> W) has completed and W_MODEL_TAG matches."
    )

frames = []
for f in files:
    df = pd.read_csv(f, header=None,
                     names=["S", "N", "M", "W_actual", "W_hat_OOF", "fold"])
    frames.append(df)

oof_df = pd.concat(frames, ignore_index=True)

print(f"\nTotal OOF rows collected: {len(oof_df)}")
print("Sample:")
print(oof_df.head())

# Sanity check: each (S, N, M) should appear exactly once across all folds
duplicates = oof_df.duplicated(subset=["S", "N", "M"])
if duplicates.any():
    raise RuntimeError(
        f"Duplicate (S,N,M) rows found in OOF data ({duplicates.sum()} duplicates). "
        "This should not happen in a proper k-fold run. "
        "Check that Run 1 completed all 5 folds without errors."
    )

print("\nSanity check passed: each (S,N,M) appears exactly once in OOF data.")


# ── STEP 2: load T from the original master dataset ──────────────────
# master_dataset.csv columns: S, N, M, T
master = pd.read_csv(MASTER_DATASET, header=None,
                     names=["S", "N", "M", "T"])

print(f"\nMaster dataset rows: {len(master)}")
print("Sample:")
print(master.head())


# ── STEP 3: join OOF W_hat with T on (S, N, M) ──────────────────────
# We keep only S, N, M, W_hat_OOF, T — that is the stacked training file.
merged = pd.merge(
    oof_df[["S", "N", "M", "W_hat_OOF"]],
    master[["S", "N", "M", "T"]],
    on=["S", "N", "M"],
    how="inner"
)

print(f"\nMerged rows: {len(merged)}")

if len(merged) != len(master):
    raise RuntimeError(
        f"Row count mismatch after merge: merged={len(merged)}, "
        f"master={len(master)}. "
        "Some (S,N,M) keys did not match between the OOF data and master_dataset. "
        "Check that snm_w.csv and master_dataset.csv share exactly the same (S,N,M) grid."
    )

print("Merge successful: all rows matched.")
print("Sample of merged result:")
print(merged.head())


# ── STEP 4: write output CSV ─────────────────────────────────────────
# Column order: S, N, M, W_hat_OOF, T
# No header, no index — same format as master_dataset.csv
merged[["S", "N", "M", "W_hat_OOF", "T"]].to_csv(
    OUTPUT_FILE, index=False, header=False
)

print(f"\nStacked dataset written to: {OUTPUT_FILE}")
print(f"Columns: S, N, M, W_hat_OOF, T")
print(f"Rows: {len(merged)}")
print(
    "\nNext step: set DATA_FILE = '"
    + OUTPUT_FILE
    + "' in pms_kfold.py and run again (Run 2)."
)
