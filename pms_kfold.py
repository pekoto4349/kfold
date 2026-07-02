import os
import shutil
import subprocess
import datetime


import numpy as np
import pandas as pd


from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error




# =====================================================================
# CONFIG
# =====================================================================


DATA_FILE = "master_dataset.csv"


BASE_MODELS_DIR = "/home/pekoto/models"
LOCAL_REPO_DIR = "/home/pekoto/repos/PMS-octave"


IMAGE_NAME = "gkousiou/laboctave"


K_FOLDS = 5
NEURONS_LAYERS = "4-45"


LOG_FILE = f"experiment_logs_{NEURONS_LAYERS}.txt"


# Keep True only if getPrediction.m still removes the first row with (2:end)
ADD_DUMMY_FIRST_ROW_FOR_PREDICTION = True


# If True, negative timing predictions are evaluated as 0.
# If False, raw PMS predictions are evaluated exactly as produced.
CLAMP_NEGATIVE_PREDICTIONS_TO_ZERO = True




# =====================================================================
# METRICS
# =====================================================================


def safe_mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)


    denominator = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)   # percent




def smape(y_true, y_pred):
    """
    Symmetric MAPE in [0, 1] scale (multiply by 100 for %).
    Per point: 2 * |y - yhat| / (|y| + |yhat|); skip points where y == yhat == 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    num = np.abs(y_true - y_pred)
    den = np.abs(y_true) + np.abs(y_pred)
    mask = den > 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(2.0 * num[mask] / den[mask]))




def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))




def safe_r2(y_true, y_pred):
    """
    Coefficient of determination R^2 = 1 - SS_res / SS_tot.
    Identical formula to ga_mlp_runs.py / cmaes_mlp_runs.py so R2 is comparable
    across methods. NOT clamped (a negative R2 passes through); guards a
    zero-variance fold (SS_tot == 0) by returning 0.0 instead of dividing by 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot == 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot




# =====================================================================
# DOCKER HELPERS
# =====================================================================


def build_mounts():
    if not os.path.exists(LOCAL_REPO_DIR):
        raise FileNotFoundError(f"Local PMS repo not found: {LOCAL_REPO_DIR}")


    mounts = []


    for filename in os.listdir(LOCAL_REPO_DIR):
        if filename.endswith(".m") or filename.endswith(".sh"):
            local_file = os.path.join(LOCAL_REPO_DIR, filename)
            container_file = f"/home/joe/{filename}"


            mounts.extend(["-v", f"{local_file}:{container_file}"])


            # PMS sometimes calls /home/joe/modelauncher without .m
            if filename == "modelauncher.m":
                mounts.extend(["-v", f"{local_file}:/home/joe/modelauncher"])


    return mounts




def run_command(command, title):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    print(" ".join(command))


    result = subprocess.run(command)


    if result.returncode != 0:
        print(f"WARNING: Docker returned exit code {result.returncode}")


    return result.returncode




def docker_base_args(model_id, num_inputs):
    return [
        "docker", "run", "--rm",
        "-w", "/home/joe",
        "-e", f"modelID={model_id}",
        "-e", f"NUMINPUTS={num_inputs}",
        "-e", "NUMOUTPUTS=1",
        "-v", f"{BASE_MODELS_DIR}:/models",
    ]




def run_pms_model(model_id, num_inputs, mounts):
    command = (
        docker_base_args(model_id, num_inputs)
        + mounts
        + [
            IMAGE_NAME,
            "bash", "/home/joe/dockerLAB-entrypoint-octave.sh", "model"
        ]
    )


    return run_command(command, f"{model_id} | PMS MODEL")




def run_pms_prediction(model_id, num_inputs, timestamp, mounts):
    command = (
        docker_base_args(model_id, num_inputs)
        + ["-e", f"TIMESTAMP={timestamp}"]
        + mounts
        + [
            IMAGE_NAME,
            "bash", "/home/joe/dockerLAB-entrypoint-octave.sh", "prediction"
        ]
    )


    return run_command(command, f"{model_id} | PMS PREDICTION")




# =====================================================================
# FILE HELPERS
# =====================================================================


def write_training_file(model_dir, train_df):
    train_path = os.path.join(model_dir, "inputfile.csv")
    train_df.to_csv(train_path, index=False, header=False)




def write_prediction_file(model_dir, test_inputs, num_inputs, timestamp):
    estimation_path = os.path.join(model_dir, f"estimation_{timestamp}.csv")


    prediction_inputs = test_inputs.copy()


    if ADD_DUMMY_FIRST_ROW_FOR_PREDICTION:
        dummy = pd.DataFrame([[0] * num_inputs])
        prediction_inputs = pd.concat([dummy, prediction_inputs], ignore_index=True)


    prediction_inputs.to_csv(estimation_path, index=False, header=False)




def write_true_y_file(model_dir, actuals, timestamp):
    true_y_path = os.path.join(model_dir, f"true_y_{timestamp}.csv")
    pd.DataFrame(actuals).to_csv(true_y_path, index=False, header=False)




def read_pms_internal_graph_mape(model_dir):
    """
    Reads the exact PMS internal graph MAPE exported from createmodel.m.


    createmodel.m must contain:
        csvwrite([workDir '/pms_internal_metrics.csv'], score);


    where score is the same variable used in:
        scorestring=['Final validation MAPE:',num2str(score),'%'];
    """


    metrics_path = os.path.join(model_dir, "pms_internal_metrics.csv")


    if not os.path.exists(metrics_path):
        return None


    values = pd.read_csv(metrics_path, header=None).values.flatten()
    return float(values[0])




# =====================================================================
# MAIN K-FOLD
# =====================================================================


def run_pms_kfold():
    df = pd.read_csv(DATA_FILE, header=None)


    total_columns = df.shape[1]
    input_columns = list(range(total_columns - 1))
    target_column = total_columns - 1
    num_inputs = len(input_columns)


    print(f"Dataset: {DATA_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Inputs: {num_inputs}")
    print("Target: last column")
    print(f"K-folds: {K_FOLDS}")
    print(f"Architecture tag: {NEURONS_LAYERS}")
    print(f"Clamp negative predictions: {CLAMP_NEGATIVE_PREDICTIONS_TO_ZERO}")


    mounts = build_mounts()


    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)


    fold_results = []
    all_actuals = []
    all_predictions = []


    for fold_index, (train_index, test_index) in enumerate(kf.split(df), start=1):
        print("\n" + "=" * 80)
        print(f"FOLD {fold_index}/{K_FOLDS}")
        print("=" * 80)


        model_id = f"model_Dynamic_Fold{fold_index}_{NEURONS_LAYERS}"
        timestamp = f"FOLD{fold_index}"
        model_dir = os.path.join(BASE_MODELS_DIR, model_id)


        # Clean fold folder
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)


        os.makedirs(model_dir, exist_ok=True)


        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()


        test_inputs = test_df[input_columns].copy()
        actuals = test_df[target_column].astype(float).values


        print(f"Train rows: {len(train_df)}")
        print(f"Test rows:  {len(test_df)}")


        # Write PMS files
        write_training_file(model_dir, train_df)
        write_prediction_file(model_dir, test_inputs, num_inputs, timestamp)
        write_true_y_file(model_dir, actuals, timestamp)


        # Run PMS model
        model_return_code = run_pms_model(model_id, num_inputs, mounts)


        bestmodel_path = os.path.join(model_dir, "bestmodel.mat")
        config_path = os.path.join(model_dir, "config.mat")


        if not os.path.exists(bestmodel_path):
            raise RuntimeError(f"Missing bestmodel.mat for fold {fold_index}")


        if not os.path.exists(config_path):
            raise RuntimeError(f"Missing config.mat for fold {fold_index}")


        # Read exact PMS graph MAPE
        pms_internal_graph_mape = read_pms_internal_graph_mape(model_dir)
        print(f"PMS internal graph MAPE: {pms_internal_graph_mape}")


        # Run PMS prediction
        prediction_return_code = run_pms_prediction(
            model_id=model_id,
            num_inputs=num_inputs,
            timestamp=timestamp,
            mounts=mounts
        )


        output_path = os.path.join(model_dir, f"out_{timestamp}.csv")


        if not os.path.exists(output_path):
            raise RuntimeError(f"Missing prediction output for fold {fold_index}")


        raw_predictions = pd.read_csv(output_path, header=None).values.flatten().astype(float)


        if CLAMP_NEGATIVE_PREDICTIONS_TO_ZERO:
            predictions = np.maximum(0, raw_predictions)
        else:
            predictions = raw_predictions


        if len(predictions) != len(actuals):
            raise RuntimeError(
                f"Fold {fold_index}: prediction length mismatch. "
                f"Predictions={len(predictions)}, Actuals={len(actuals)}"
            )


        # External held-out metrics
        external_mape = safe_mape(actuals, predictions)
        external_smape = smape(actuals, predictions)
        external_mae = mean_absolute_error(actuals, predictions)
        external_rmse = rmse(actuals, predictions)
        external_r2 = safe_r2(actuals, predictions)


        print(f"External fold MAPE:  {external_mape:.2f}%")
        print(f"External fold SMAPE: {external_smape:.2%}")
        print(f"External fold MAE:   {external_mae:.6f}")
        print(f"External fold RMSE:  {external_rmse:.6f}")
        print(f"External fold R2:    {external_r2:.4f}")


        fold_results.append({
            "fold": fold_index,


            # PMS model-command graph MAPE
            "pms_internal_graph_mape": pms_internal_graph_mape,


            # External prediction-command metrics
            "external_fold_mape": external_mape,
            "external_fold_smape": external_smape,
            "external_fold_mae": external_mae,
            "external_fold_rmse": external_rmse,
            "external_fold_r2": external_r2,


            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "model_return_code": model_return_code,
            "prediction_return_code": prediction_return_code,
        })


        all_actuals.extend(actuals)
        all_predictions.extend(predictions)


        # Save held-out predictions for inspection
        heldout = test_df.copy()
        heldout["prediction"] = predictions
        heldout["fold"] = fold_index


        heldout_path = os.path.join(model_dir, f"heldout_predictions_{timestamp}.csv")
        heldout.to_csv(heldout_path, index=False, header=False)


    # =================================================================
    # FINAL RESULTS
    # =================================================================


    all_actuals = np.asarray(all_actuals, dtype=float)
    all_predictions = np.asarray(all_predictions, dtype=float)


    final_external_mape = safe_mape(all_actuals, all_predictions)
    final_external_smape = smape(all_actuals, all_predictions)
    final_external_mae = mean_absolute_error(all_actuals, all_predictions)
    final_external_rmse = rmse(all_actuals, all_predictions)
    final_external_r2 = safe_r2(all_actuals, all_predictions)


    # ── mean ± 95% t-CI across the K folds (added: matches ga/cmaes) ──
    # POOLED (above) aggregates every row into one metric; this block instead
    # averages the per-fold metrics and adds a t-distribution CI (df = K-1).
    T_CRIT = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
              5: 2.571,  7: 2.306, 9: 2.262, 14: 2.145, 19: 2.093}

    def _t_crit(n):
        return T_CRIT.get(n - 1, 1.96)

    def _avg_std_ci(key):
        vals = [r[key] for r in fold_results]
        n    = len(vals)
        avg  = float(np.mean(vals))
        std  = float(np.std(vals, ddof=1)) if n > 1 else 0.0
        ci   = _t_crit(n) * std / np.sqrt(n) if n > 1 else 0.0
        return avg, std, ci

    avg_mape,  std_mape,  ci_mape  = _avg_std_ci("external_fold_mape")
    avg_smape, std_smape, ci_smape = _avg_std_ci("external_fold_smape")
    avg_mae,   std_mae,   ci_mae   = _avg_std_ci("external_fold_mae")
    avg_rmse,  std_rmse,  ci_rmse  = _avg_std_ci("external_fold_rmse")
    avg_r2,    std_r2,    ci_r2    = _avg_std_ci("external_fold_r2")


    results_df = pd.DataFrame(fold_results)


    result_dir = os.path.join(BASE_MODELS_DIR, "kfold_results")
    os.makedirs(result_dir, exist_ok=True)


    metrics_path = os.path.join(result_dir, f"kfold_metrics_{NEURONS_LAYERS}.csv")
    results_df.to_csv(metrics_path, index=False)


    external_mapes = [
        f"{r['external_fold_mape']:.2f}%"
        for r in fold_results
    ]


    external_smapes = [
        f"{r['external_fold_smape']:.2%}"
        for r in fold_results
    ]


    external_r2s = [
        f"{r['external_fold_r2']:.4f}"
        for r in fold_results
    ]


    pms_internal_mapes = [
        "None" if r["pms_internal_graph_mape"] is None
        else f"{r['pms_internal_graph_mape']:.2f}%"
        for r in fold_results
    ]


    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    log_text = (
        f"\n{'=' * 80}\n"
        f"RUN TIMESTAMP: {current_time}\n"
        f"DATA FILE: {DATA_FILE}\n"
        f"IMAGE: {IMAGE_NAME}\n"
        f"LOCAL PMS REPO: {LOCAL_REPO_DIR}\n"
        f"CONFIG: {num_inputs} Inputs -> 1 Target | {K_FOLDS}-Fold CV | {NEURONS_LAYERS}\n"
        f"CLAMP NEGATIVE PREDICTIONS: {CLAMP_NEGATIVE_PREDICTIONS_TO_ZERO}\n"
        f"METRICS: MAPE in %, SMAPE on 0..200% scale | reported per-fold, POOLED "
        f"over all rows, AND mean ± 95% t-CI across folds\n"
        f"PMS INTERNAL GRAPH MAPES: {pms_internal_mapes}\n"
        f"EXTERNAL FOLD MAPES (%): {external_mapes}\n"
        f"EXTERNAL FOLD SMAPEs (%): {external_smapes}\n"
        f"EXTERNAL FOLD R2s: {external_r2s}\n"
        f"{'-' * 80}\n"
        f"FINAL EXTERNAL POOLED MAPE:  {final_external_mape:.2f}%\n"
        f"FINAL EXTERNAL POOLED SMAPE: {final_external_smape:.2%}\n"
        f"FINAL EXTERNAL POOLED MAE:   {final_external_mae:.6f}\n"
        f"FINAL EXTERNAL POOLED RMSE:  {final_external_rmse:.6f}\n"
        f"FINAL EXTERNAL POOLED R2:    {final_external_r2:.4f}\n"
        f"{'-' * 80}\n"
        f"MEAN ± 95% t-CI ACROSS {K_FOLDS} FOLDS:\n"
        f"  MAPE : {avg_mape:.2f}% (std ± {std_mape:.2f}% | 95% CI ± {ci_mape:.2f}% "
        f"→ [{avg_mape - ci_mape:.2f}%, {avg_mape + ci_mape:.2f}%])\n"
        f"  SMAPE: {avg_smape * 100:.2f}% (std ± {std_smape * 100:.2f}% | 95% CI ± "
        f"{ci_smape * 100:.2f}% → [{(avg_smape - ci_smape) * 100:.2f}%, "
        f"{(avg_smape + ci_smape) * 100:.2f}%])\n"
        f"  MAE  : {avg_mae:.6f} (std ± {std_mae:.6f} | 95% CI ± {ci_mae:.6f})\n"
        f"  RMSE : {avg_rmse:.6f} (std ± {std_rmse:.6f} | 95% CI ± {ci_rmse:.6f})\n"
        f"  R2   : {avg_r2:.4f} (std ± {std_r2:.4f} | 95% CI ± {ci_r2:.4f} "
        f"→ [{avg_r2 - ci_r2:.4f}, {avg_r2 + ci_r2:.4f}])\n"
        f"{'=' * 80}\n"
    )


    print(log_text)


    with open(LOG_FILE, "a") as f:
        f.write(log_text)


    print(f"Saved metrics: {metrics_path}")
    print(f"Saved log:     {LOG_FILE}")




if __name__ == "__main__":
    run_pms_kfold()