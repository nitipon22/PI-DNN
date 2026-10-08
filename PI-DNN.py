# ================================
# -1. Fix seed
#
# ================================
import os
import random

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneGroupOut, GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import tracemalloc
import math
import matplotlib.pyplot as plt
import shap
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf


def reset_seeds(seed=SEED):
    """reset seed ของ python / numpy / tensorflow (เรียกก่อนสร้างโมเดลทุกครั้งเพื่อให้ผลซ้ำได้)"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

reset_seeds(SEED)
try:
    keras.utils.set_random_seed(SEED)
except Exception as e:
    print(f"[INFO] keras.utils.set_random_seed ใช้ไม่ได้: {e}")
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print(f"[INFO] enable_op_determinism ใช้ไม่ได้: {e}")

# ================================
# 0. Output dir
# ================================
output_dir = "imgRevision/outputs/pi-dnn64"
os.makedirs(output_dir, exist_ok=True)

# ================================
# 1. Load data
# ================================

df = pd.read_csv(
    "batteryNew/Battery_RUL_with_ID.csv"
)

print("First 5 records:", df.head())

# ================================
# 2.features / target / group
# ================================
features = [
    #'Cycle_Index',
    'Discharge Time (s)',
    'Decrement 3.6-3.4V (s)',
    'Max. Voltage Dischar. (V)',
    'Min. Voltage Charg. (V)',
    'Time at 4.15V (s)',
    'Time constant current (s)',
    'Charging time (s)',
    'Total time (s)',
]
target_col = 'RUL'
group_col = 'Battery_ID'

# Indices into `features` used by the monotonicity penalty in train_step().
# Discharge Time (s): RUL should be non-decreasing as this grows -> penalize negative gradient.
# Decrement 3.6-3.4V (s): RUL should be non-decreasing as this grows -> penalize negative gradient.
idx_DT = features.index('Discharge Time (s)')
idx_DEC = features.index('Decrement 3.6-3.4V (s)')

X_raw = df[features].copy()
y = df[target_col].copy()
groups = df[group_col].copy()

# ================================
# 3. NaN
# ================================
cols_must_be_positive = [c for c in
                          ['Discharge Time (s)', 'Charging time (s)', 'Total time (s)']
                          if c in X_raw.columns]


def fit_outlier_bounds(X_train_df):
    bounds = {}
    for col in X_train_df.select_dtypes(include=[np.number]).columns:
        series = X_train_df[col]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 2.0 * IQR, Q3 + 2.0 * IQR)
    return bounds


def apply_outlier_bounds(X_df, bounds, cols_must_be_positive):
    X_out = X_df.copy()
    for col in cols_must_be_positive:
        if col in X_out.columns:
            X_out.loc[X_out[col] < 0, col] = np.nan
    for col, (lower, upper) in bounds.items():
        if col in X_out.columns:
            X_out.loc[(X_out[col] < lower) | (X_out[col] > upper), col] = np.nan
    return X_out


# ================================
# 4. KMeans-based imputation
# ================================
def kmeans_impute_fit(X_train_df, n_clusters=5, random_state=SEED):
    X_train_filled_mean = X_train_df.fillna(X_train_df.mean())
    scaler_km = StandardScaler()
    X_train_scaled_km = scaler_km.fit_transform(X_train_filled_mean)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    train_clusters = kmeans.fit_predict(X_train_scaled_km)

    X_train_imputed = X_train_df.copy()
    cluster_means = {}
    global_means = X_train_df.mean()

    for col in X_train_df.columns:
        col_cluster_means = {}
        for cluster_id in np.unique(train_clusters):
            mask = train_clusters == cluster_id
            c_mean = X_train_df.loc[mask, col].mean()
            if np.isnan(c_mean):
                c_mean = global_means[col]
            col_cluster_means[cluster_id] = c_mean
            fill_mask = mask & X_train_df[col].isna()
            X_train_imputed.loc[fill_mask, col] = c_mean
        cluster_means[col] = col_cluster_means

    return X_train_imputed, kmeans, scaler_km, cluster_means, global_means


def kmeans_impute_transform(X_df, kmeans, scaler_km, cluster_means, global_means):
    X_filled_tmp = X_df.fillna(global_means)
    X_scaled_km = scaler_km.transform(X_filled_tmp)
    pred_clusters = kmeans.predict(X_scaled_km)

    X_imputed = X_df.copy()
    for col in X_df.columns:
        for cluster_id in np.unique(pred_clusters):
            mask = pred_clusters == cluster_id
            fill_value = cluster_means[col].get(cluster_id, global_means[col])
            fill_mask = mask & X_df[col].isna()
            X_imputed.loc[fill_mask, col] = fill_value
        X_imputed[col] = X_imputed[col].fillna(global_means[col])
    return X_imputed


# ================================
# 5. safe_predict
# ================================
def safe_predict(model, X_input, context_label=""):
    y_pred = model.predict(X_input, verbose=0).flatten()
    bad_mask = ~np.isfinite(y_pred)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        print(f"[WARNING] {context_label}: พบค่า NaN/Inf จำนวน {n_bad} จุดใน prediction "
              f"(โมเดลน่าจะ diverge) -> จะตัด combo นี้ทิ้งจากการพิจารณา best_params")
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    return y_pred, n_bad

# ================================
# 5b. Build Model Function
#     NOTE: parameter renamed to `input_shape` to match all call sites below.
# ================================
def build_model(optimizer='adam', learning_rate=0.001, neurons=64, input_shape=None):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1)
    ])

    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            opt = Adam(learning_rate=learning_rate)
    else:
        opt = optimizer

    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model


# ================================
# 5c. Physics-informed diagnostic metrics
#     (evaluation-time only -- these metrics do not affect training.
#      They measure how well the trained model satisfies the predefined
#      physics-based monotonicity constraints on a given dataset.)
#
#     1) Monotonicity Violation Rate (MVR)
#        Proportion of samples for which the gradient of the predicted RUL
#        with respect to a constrained feature has the wrong sign:
#
#          - Discharge Time (s):
#            RUL should increase or remain unchanged as Discharge Time increases.
#            The expected gradient is therefore >= 0.
#            A gradient < 0 is counted as a monotonicity violation.
#
#          - Decrement 3.6-3.4 V (s):
#            RUL should increase or remain unchanged as this feature increases.
#            The expected gradient is therefore >= 0.
#            A gradient < 0 is counted as a monotonicity violation.
#
#     2) Gradient Consistency Score (GCS)
#        Proportion of samples for which the gradients of both constrained
#        features simultaneously satisfy their expected monotonic directions.
#        A value closer to 1 indicates more consistent satisfaction of the
#        monotonicity constraints.
#
#     3) Physically Implausible Predictions (PIP)
#        Proportion of predictions that are physically implausible based
#        directly on the predicted RUL values, without using gradient information.
#        For example, a predicted RUL < 0 is considered physically implausible
#        because remaining useful life cannot be negative.
# ================================
def compute_gradient_metrics(model, X_scaled, idx_DT, idx_DEC, batch_size=256):
    X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
    n = X_tf.shape[0]
    grad_DT_all = np.zeros(n, dtype=np.float32)
    grad_DEC_all = np.zeros(n, dtype=np.float32)

    for start in range(0, n, batch_size):
        end = start + batch_size
        X_batch = X_tf[start:end]
        with tf.GradientTape() as tape:
            tape.watch(X_batch)
            preds = tf.squeeze(model(X_batch, training=False), axis=1)
        grads = tape.gradient(preds, X_batch)
        grads_np = grads.numpy() if grads is not None else np.zeros(
            (X_batch.shape[0], X_batch.shape[1]), dtype=np.float32
        )
        grad_DT_all[start:end] = grads_np[:, idx_DT]
        grad_DEC_all[start:end] = grads_np[:, idx_DEC]
        
    mono_violation_DT = grad_DT_all < 0.0      # ควรเป็น >= 0
    mono_violation_DEC = grad_DEC_all < 0.0    # ควรเป็น >= 0

    return {
        'grad_DT': grad_DT_all,
        'grad_DEC': grad_DEC_all,
        'mono_violation_DT': mono_violation_DT,
        'mono_violation_DEC': mono_violation_DEC,
    }


def compute_physics_informed_metrics(model, X_scaled, y_pred, idx_DT, idx_DEC, rul_lower_bound=0.0):
    grad_info = compute_gradient_metrics(model, X_scaled, idx_DT, idx_DEC)

    viol_DT = grad_info['mono_violation_DT']
    viol_DEC = grad_info['mono_violation_DEC']
    viol_any = viol_DT | viol_DEC
    consistent_both = (~viol_DT) & (~viol_DEC)

    n = len(y_pred)
    mvr_DT = float(np.mean(viol_DT))
    mvr_DEC = float(np.mean(viol_DEC))
    mvr_overall = float(np.mean(viol_any))
    gradient_consistency = float(np.mean(consistent_both))

    implausible_mask = y_pred < rul_lower_bound
    pip_rate = float(np.mean(implausible_mask))

    return {
        'MVR_DischargeTime': mvr_DT,
        'MVR_Decrement': mvr_DEC,
        'MVR_overall': mvr_overall,
        'Gradient_Consistency': gradient_consistency,
        'Physically_Implausible_Rate': pip_rate,
        'n_implausible': int(implausible_mask.sum()),
        'n_samples': n,
        'mean_grad_DT': float(np.mean(grad_info['grad_DT'])),
        'mean_grad_DEC': float(np.mean(grad_info['grad_DEC'])),
    }


# ================================
# 6. Training Step with Monotonicity
#    Custom physics-informed training step: MAE loss + a monotonicity
#    penalty on the gradient of the prediction w.r.t. two input features.
#    Called via train_physics_informed() below instead of model.fit().
#
#    Constraint (both features): RUL should be non-decreasing as the feature grows
#    -> penalize negative gradient: relu(-grad)
# ================================
lambda_mono = 0.001

def train_step(model, optimizer, X_batch, y_batch):
    if len(X_batch.shape) == 1:
        X_batch = tf.expand_dims(X_batch, axis=0)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(X_batch)
            preds = tf.squeeze(model(X_batch, training=True), axis=1)

        input_grads = inner_tape.gradient(preds, X_batch)

        mae = tf.reduce_mean(tf.abs(y_batch - preds))

        if input_grads is not None:
            penalty_DT = tf.reduce_mean(
                tf.nn.relu(-input_grads[:, idx_DT])
            )

            penalty_DEC = tf.reduce_mean(
                tf.nn.relu(-input_grads[:, idx_DEC])
            )
            mono = penalty_DT + penalty_DEC
        else:
            mono = 0.0

        loss = mae + lambda_mono * mono

    grads = outer_tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, mae, mono


def train_physics_informed(model, X, y, epochs=20, batch_size=8, seed=SEED):

    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    n = int(X_tf.shape[0])
    optimizer = model.optimizer
    rng = np.random.RandomState(seed)

    # Compile a NEW tf.function bound to this specific (model, optimizer) pair.
    # Reusing one global @tf.function across many different optimizer instances
    # (a new Adam/SGD is created per grid-search combo / per fold) makes TF think
    # it's retracing an already-built graph and it refuses to create the
    # optimizer's slot variables (momentum/velocity) a second time -> ValueError.
    # A fresh tf.function per training run still gets traced-once/reused-many-times
    # performance across the epoch/batch loop below.
    step_fn = tf.function(train_step)

    diverged = False
    for epoch in range(epochs):
        perm = tf.constant(rng.permutation(n), dtype=tf.int32)
        X_shuf = tf.gather(X_tf, perm)
        y_shuf = tf.gather(y_tf, perm)

        for start in range(0, n, batch_size):
            end = start + batch_size
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]
            loss, mae, mono = step_fn(model, optimizer, X_batch, y_batch)

            if not tf.math.is_finite(loss):
                diverged = True
                break
        if diverged:
            break

    return diverged

# ================================
# 7. Hyperparameter Grid
# ================================
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'neurons': [32, 64, 128]
}

param_options = [
    {'optimizer': opt, 'neurons': n}
    for opt in param_grid['optimizer']
    for n in param_grid['neurons']
]

# ================================
# 8. Outer CV: LeaveOneGroupOut ตาม Battery_ID
#    เท่ากับ "train N-1 ก้อน test 1 ก้อน" ตามจำนวน battery ทั้งหมด
# ================================
n_batteries = groups.nunique()
print(f"\nจำนวน Battery_ID ทั้งหมด: {n_batteries}")
outer_gkf = LeaveOneGroupOut()
n_outer_splits = n_batteries

fold_results = []
all_test_preds = []
all_test_true = []
dnn_best_model = None
last_fold_X_test_scaled = None

for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(X_raw, y, groups=groups), 1):
    print(f"\n===== Outer Fold {fold}/{n_outer_splits} =====")
    start_time = time.time()
    tracemalloc.start()

    X_train_df = X_raw.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_raw.iloc[test_idx].reset_index(drop=True)
    y_train_full = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train_full = groups.iloc[train_idx].reset_index(drop=True)

    # ---- 8.1 Outlier bounds ----
    outlier_bounds = fit_outlier_bounds(X_train_df)
    X_train_df = apply_outlier_bounds(X_train_df, outlier_bounds, cols_must_be_positive)
    X_test_df = apply_outlier_bounds(X_test_df, outlier_bounds, cols_must_be_positive)

    # ---- 8.2 KMeans imputation ----
    X_train_imputed, km_model, km_scaler, cluster_means, global_means = kmeans_impute_fit(X_train_df)
    X_test_imputed = kmeans_impute_transform(X_test_df, km_model, km_scaler, cluster_means, global_means)

    # ---- 8.3 Train fold (inner)----
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    inner_train_idx, inner_val_idx = next(
        gss.split(X_train_imputed, y_train_full, groups=groups_train_full)
    )

    X_inner_train_df = X_train_imputed.iloc[inner_train_idx]
    X_inner_val_df = X_train_imputed.iloc[inner_val_idx]
    y_inner_train = y_train_full.iloc[inner_train_idx]
    y_inner_val = y_train_full.iloc[inner_val_idx]

    # ---- 8.4 Scaler X fit ----
    inner_scaler = StandardScaler()
    X_inner_train_scaled = inner_scaler.fit_transform(X_inner_train_df)
    X_inner_val_scaled = inner_scaler.transform(X_inner_val_df)

    # y scaler
    y_scaler_inner = StandardScaler()
    y_inner_train_scaled = y_scaler_inner.fit_transform(
        y_inner_train.values.reshape(-1, 1)
    ).flatten()

    # ---- 8.5 Grid search on inner validation ----
    best_val_mae = float('inf')
    best_params = None
    batch_size = 64
    for params in param_options:
        reset_seeds(SEED)  # ให้ทุก combo เริ่มจาก seed เดียวกัน (weight init เหมือนกันทุกครั้งที่รัน)
        model_tmp = build_model(
            input_shape=X_inner_train_scaled.shape[1],
            optimizer=params['optimizer'],
            neurons=params['neurons'],
        )
        diverged_train = train_physics_informed(
            model_tmp, X_inner_train_scaled, y_inner_train_scaled,
            epochs=20, batch_size=batch_size, seed=SEED,
        )
        if diverged_train:
            print(f"  -> ตัด params={params} ทิ้ง เพราะโมเดล diverge ระหว่าง physics-informed training")
            continue
        y_val_pred_scaled, n_bad = safe_predict(
            model_tmp, X_inner_val_scaled,
            context_label=f"Fold {fold} grid-search params={params}"
        )
        y_val_pred = y_scaler_inner.inverse_transform(
            y_val_pred_scaled.reshape(-1, 1)
        ).flatten()
        val_mae = mean_absolute_error(y_inner_val, y_val_pred)

        if n_bad > 0:
            print(f"  -> ตัด params={params} ทิ้ง เพราะโมเดล diverge ระหว่าง grid search")
            continue

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_params = params

    if best_params is None:
        print(f"[WARNING] Fold {fold}: ทุก hyperparameter diverge ระหว่าง grid search, ใช้ค่า fallback (adam, 32)")
        best_params = {'optimizer': 'adam', 'neurons': 32}

    print(f"Fold {fold} best params (inner validation MAE={best_val_mae:.4f}): {best_params}")

    # ---- 8.6 Retrain final model on train fold ----
    final_scaler = StandardScaler()
    X_train_scaled = final_scaler.fit_transform(X_train_imputed)
    X_test_scaled = final_scaler.transform(X_test_imputed)

    y_scaler_final = StandardScaler()
    y_train_scaled = y_scaler_final.fit_transform(
        y_train_full.values.reshape(-1, 1)
    ).flatten()

    reset_seeds(SEED)
    best_model = build_model(
        input_shape=X_train_scaled.shape[1],
        optimizer=best_params['optimizer'],
        neurons=best_params['neurons'],
    )
    final_diverged_during_training = train_physics_informed(
        best_model, X_train_scaled, y_train_scaled,
        epochs=20, batch_size=batch_size, seed=SEED,
    )
    if final_diverged_during_training:
        print(f"[WARNING] Fold {fold}: final model diverged during physics-informed training "
              f"(params={best_params})")

    # ---- 8.7 metrics ----
    y_train_pred_scaled, n_bad_train = safe_predict(best_model, X_train_scaled, context_label=f"Fold {fold} train")
    y_train_pred = y_scaler_final.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    train_mae = mean_absolute_error(y_train_full, y_train_pred)
    train_rmse = math.sqrt(mean_squared_error(y_train_full, y_train_pred))
    train_r2 = r2_score(y_train_full, y_train_pred)
    y_train_nonzero = y_train_full[y_train_full != 0]
    y_train_pred_nonzero = y_train_pred[y_train_full != 0]
    train_mape = np.mean(np.abs((y_train_nonzero - y_train_pred_nonzero) / y_train_nonzero)) * 100

    y_pred_scaled, n_bad_test = safe_predict(best_model, X_test_scaled, context_label=f"Fold {fold} test")
    y_pred = y_scaler_final.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    y_true_nonzero = y_test[y_test != 0]
    y_pred_nonzero = y_pred[y_test != 0]
    mape = np.mean(np.abs((y_true_nonzero - y_pred_nonzero) / y_true_nonzero)) * 100

    # ---- 8.7b Physics-informed diagnostic metrics (evaluation-time, gradient-based) ----
    physics_train = compute_physics_informed_metrics(
        best_model, X_train_scaled, y_train_pred, idx_DT, idx_DEC
    )
    physics_test = compute_physics_informed_metrics(
        best_model, X_test_scaled, y_pred, idx_DT, idx_DEC
    )

    all_test_preds.append(pd.Series(y_pred, index=test_idx))
    all_test_true.append(pd.Series(y_test.values, index=test_idx))

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    fold_results.append({
        'fold': fold,
        'Train_MAE': train_mae,
        'Train_RMSE': train_rmse,
        'Train_R2': train_r2,
        'Train_MAPE': train_mape,
        'Test_MAE': mae,
        'Test_RMSE': rmse,
        'Test_R2': r2,
        'Test_MAPE': mape,
        # ---- physics-informed diagnostics (train) ----
        'Train_MVR_DischargeTime': physics_train['MVR_DischargeTime'],
        'Train_MVR_Decrement': physics_train['MVR_Decrement'],
        'Train_MVR_overall': physics_train['MVR_overall'],
        'Train_Gradient_Consistency': physics_train['Gradient_Consistency'],
        'Train_Physically_Implausible_Rate': physics_train['Physically_Implausible_Rate'],
        # ---- physics-informed diagnostics (test) ----
        'Test_MVR_DischargeTime': physics_test['MVR_DischargeTime'],
        'Test_MVR_Decrement': physics_test['MVR_Decrement'],
        'Test_MVR_overall': physics_test['MVR_overall'],
        'Test_Gradient_Consistency': physics_test['Gradient_Consistency'],
        'Test_Physically_Implausible_Rate': physics_test['Physically_Implausible_Rate'],
        'time_sec': elapsed_time,
        'memory_MB': peak / (1024 * 1024),
        'best_params': best_params,
        'diverged': (n_bad_train > 0 or n_bad_test > 0 or final_diverged_during_training),
    })

    print(f"Fold {fold} | Train: MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}, MAPE={train_mape:.4f}% "
          f"| Test: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f}% "
          f"| Time={elapsed_time:.4f}s, Memory={peak/(1024*1024):.4f}MB, Best Params={best_params}")
    print(f"Fold {fold} | Physics (Test): MVR_overall={physics_test['MVR_overall']*100:.2f}%  "
          f"GradConsistency={physics_test['Gradient_Consistency']*100:.2f}%  "
          f"ImplausibleRate={physics_test['Physically_Implausible_Rate']*100:.2f}% "
          f"({physics_test['n_implausible']}/{physics_test['n_samples']} samples)")

    dnn_best_model = best_model
    last_fold_X_test_scaled = X_test_scaled

# ================================
# 9. Summary fold
# ================================
fold_results_df = pd.DataFrame(fold_results)
print("\n===== Summary across folds =====")
print(fold_results_df)

diverged_folds = fold_results_df[fold_results_df['diverged']]
if len(diverged_folds) > 0:
    print(f"\n[WARNING] พบ {len(diverged_folds)} fold ที่โมเดล diverge (มี NaN/Inf ระหว่าง predict): "
          f"{diverged_folds['fold'].tolist()}")
    print("แนะนำ: ดู best_params ของ fold เหล่านี้ ถ้าเจอ 'sgd' บ่อย ให้พิจารณาตัด sgd ออกจาก grid "
          "หรือลด learning_rate ลงอีก")

# ใช้เฉพาะ fold ที่ไม่ diverge ในการสรุปสถิติ เพื่อไม่ให้ค่า NaN/Inf ที่ถูกเคลียร์เป็น 0
# ไปดึงค่าเฉลี่ยผิดเพี้ยน
clean_results_df = fold_results_df[~fold_results_df['diverged']]
if len(clean_results_df) < len(fold_results_df):
    print(f"[INFO] ใช้ {len(clean_results_df)}/{len(fold_results_df)} fold ที่ไม่ diverge ในการคำนวณสรุปสถิติ")

metric_cols = ['Train_MAE', 'Train_RMSE', 'Train_R2', 'Train_MAPE',
               'Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MAPE',
               'time_sec', 'memory_MB']

summary_stats = pd.DataFrame({
    'mean': clean_results_df[metric_cols].mean(),
    'std': clean_results_df[metric_cols].std(),
    'median': clean_results_df[metric_cols].median(),
    'min': clean_results_df[metric_cols].min(),
    'max': clean_results_df[metric_cols].max(),
})
summary_stats['mean ± std'] = summary_stats.apply(
    lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1
)

print("\n===== Mean ± Std ของทุกพารามิเตอร์ (เฉพาะ fold ที่ไม่ diverge) =====")
print(summary_stats[['mean ± std', 'median', 'min', 'max']].to_string())

# ---- 9b. สรุป Physics-informed diagnostics แยกต่างหาก (รายงานเป็น %) ----
physics_metric_cols = [
    'Train_MVR_DischargeTime', 'Train_MVR_Decrement', 'Train_MVR_overall',
    'Train_Gradient_Consistency', 'Train_Physically_Implausible_Rate',
    'Test_MVR_DischargeTime', 'Test_MVR_Decrement', 'Test_MVR_overall',
    'Test_Gradient_Consistency', 'Test_Physically_Implausible_Rate',
]
physics_summary = pd.DataFrame({
    'mean_%': clean_results_df[physics_metric_cols].mean() * 100,
    'std_%': clean_results_df[physics_metric_cols].std() * 100,
    'min_%': clean_results_df[physics_metric_cols].min() * 100,
    'max_%': clean_results_df[physics_metric_cols].max() * 100,
})
print("\n===== Physics-informed Diagnostics (Monotonicity Violation / Gradient Consistency / "
      "Physically Implausible) — % ของตัวอย่าง, เฉพาะ fold ที่ไม่ diverge =====")
print(physics_summary.to_string(float_format=lambda x: f"{x:.2f}"))

print("\n===== รายงานผลแยกแต่ละ Fold =====")
for _, row in fold_results_df.iterrows():
    print(f"\n--- Fold {int(row['fold'])} (Best Params: {row['best_params']}, "
          f"Diverged={row['diverged']}) ---")
    print(f"  Train : MAE={row['Train_MAE']:.4f}  RMSE={row['Train_RMSE']:.4f}  "
          f"R2={row['Train_R2']:.4f}  MAPE={row['Train_MAPE']:.4f}%")
    print(f"  Test  : MAE={row['Test_MAE']:.4f}  RMSE={row['Test_RMSE']:.4f}  "
          f"R2={row['Test_R2']:.4f}  MAPE={row['Test_MAPE']:.4f}%")
    print(f"  Physics (Test): MVR_overall={row['Test_MVR_overall']*100:.2f}%  "
          f"GradConsistency={row['Test_Gradient_Consistency']*100:.2f}%  "
          f"ImplausibleRate={row['Test_Physically_Implausible_Rate']*100:.2f}%")
    print(f"  Time  : {row['time_sec']:.4f}s   Memory: {row['memory_MB']:.4f}MB")

all_test_preds_df = pd.concat(all_test_preds).sort_index()
all_test_true_df = pd.concat(all_test_true).sort_index()

# ==============================
# 10. Plot predicted vs actual
# ==============================
plt.figure(figsize=(8, 6))
plt.scatter(all_test_true_df.values, all_test_preds_df.values, color='red', label='Predicted')
plt.plot([all_test_true_df.min(), all_test_true_df.max()],
         [all_test_true_df.min(), all_test_true_df.max()],
         color='blue', linestyle='--', label='Ideal')
plt.xlabel('Actual RUL (Cycle)')
plt.ylabel('Predicted RUL (Cycle)')
plt.title('PI-DNN Leave-One-Battery-Out CV Predictions')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'pred_vs_actual.png'), dpi=600, bbox_inches='tight')
plt.show()

# ==============================
# 10b. Plot: Physics-informed diagnostics per fold
# ==============================
plt.figure(figsize=(9, 5))
x_folds = fold_results_df['fold']
plt.plot(x_folds, fold_results_df['Test_MVR_overall'] * 100, marker='o', label='Monotonicity Violation Rate (%)')
plt.plot(x_folds, fold_results_df['Test_Gradient_Consistency'] * 100, marker='s', label='Gradient Consistency (%)')
plt.plot(x_folds, fold_results_df['Test_Physically_Implausible_Rate'] * 100, marker='^',
         label='Physically Implausible Rate (%)')
plt.xlabel('Outer Fold (Battery held out)')
plt.ylabel('%')
plt.title('Physics-informed Diagnostics per Fold (Test set)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'physics_informed_diagnostics_per_fold.png'), dpi=600, bbox_inches='tight')
plt.show()

# ================================
# 11. SHAP (ใช้โมเดล/ข้อมูลจาก fold สุดท้าย)
#     ข้อมูลตอนนี้เป็น 2D (samples, features) อยู่แล้ว ไม่ต้อง reshape
# ================================
img_revision_dir = os.path.join(output_dir, 'imgRevision')
os.makedirs(img_revision_dir, exist_ok=True)

background_data = last_fold_X_test_scaled[:min(100, last_fold_X_test_scaled.shape[0])]
num_samples = last_fold_X_test_scaled.shape[0] // 2
X_test_for_shap = last_fold_X_test_scaled[:max(num_samples, 1)]

reset_seeds(SEED)  # GradientExplainer สุ่ม interpolation/background -> reset seed ก่อน
explainer = shap.GradientExplainer(dnn_best_model, background_data)
shap_values = explainer.shap_values(X_test_for_shap)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

# ตัด dimension ท้ายที่เป็น output=1 ออก ถ้ามี (shape อาจเป็น (samples, features, 1))
if shap_values.ndim == 3:
    shap_values_2d = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])
else:
    shap_values_2d = shap_values

X_test_for_plot = X_test_for_shap

shap.summary_plot(shap_values_2d, X_test_for_plot, feature_names=features, show=False)
plt.savefig(os.path.join(img_revision_dir, 'shap_summary_plot.png'), dpi=600, bbox_inches='tight')
plt.show()
plt.close()

shap.dependence_plot("Max. Voltage Dischar. (V)", shap_values_2d, X_test_for_plot, feature_names=features, show=False)
plt.savefig(os.path.join(img_revision_dir, 'shap_dependence_plot.png'), dpi=600, bbox_inches='tight')
plt.show()
plt.close()

print(f"\n[INFO] บันทึกรูป SHAP ไว้ที่: {img_revision_dir}")

# Dependence plot สำหรับทุก feature
X_test_for_plot_df = pd.DataFrame(X_test_for_plot, columns=features)
for i, feature in enumerate(features):
    plt.figure(figsize=(6, 4))
    shap.dependence_plot(feature, shap_values_2d, X_test_for_plot_df, show=False)
    plt.savefig(os.path.join(img_revision_dir, f'shap_dependence_{i}_{feature.replace(" ", "_").replace(".", "")}.png'),
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()
