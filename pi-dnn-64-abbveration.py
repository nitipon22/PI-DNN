# ================================================================
# -1. Fix seed (ต้องตั้ง env ก่อน import tensorflow)
#     แนะนำให้รันด้วย: PYTHONHASHSEED=42 python pi_dnn_ablation.py
# ================================================================
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
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.constraints import NonNeg
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import wilcoxon
import time
import tracemalloc
import math
import matplotlib.pyplot as plt
import shap
import tensorflow as tf


def reset_seeds(seed=SEED):
    """reset seed ของ python / numpy / tensorflow
    (เรียกก่อนสร้างโมเดลทุกครั้ง เพื่อให้ทุก variant / combo / fold เริ่มจาก weight init เดียวกัน)"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


reset_seeds(SEED)
try:
    keras.utils.set_random_seed(SEED)  # TF >= 2.7
except Exception as e:
    print(f"[INFO] keras.utils.set_random_seed ใช้ไม่ได้: {e}")
try:
    tf.config.experimental.enable_op_determinism()  # TF >= 2.9
except Exception as e:
    print(f"[INFO] enable_op_determinism ใช้ไม่ได้: {e}")

# ================================================================
# 0. Output dir
# ================================================================
output_dir = "imgRevision/outputs/pi-dnn64-ablation"
os.makedirs(output_dir, exist_ok=True)

# QUICK_TEST_MODE lets you sanity-check the whole pipeline (fewer epochs,
# smaller grid, fewer variants) before committing to the full run, which is
# expensive: n_variants x n_grid_combos x n_batteries(outer folds).
QUICK_TEST_MODE = False

# ================================================================
# 1. Load data
# ================================================================
df = pd.read_csv("batteryNew/Battery_RUL_with_ID.csv")
print("First 5 records:", df.head())

# ================================================================
# 2. Features / target / group
# ================================================================
features = [
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

# Indices into `features` used by every monotonicity-based constraint below.
# Discharge Time (s):      RUL should be NON-DECREASING as this grows -> penalize negative gradient.
# Decrement 3.6-3.4V (s):  RUL should be NON-DECREASING as this grows -> penalize negative gradient.
idx_DT = features.index('Discharge Time (s)')
idx_DEC = features.index('Decrement 3.6-3.4V (s)')

X_raw = df[features].copy()
y = df[target_col].copy()
groups = df[group_col].copy()

# ================================================================
# 3. Outlier bounds (fit on train fold only)
# ================================================================
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


# ================================================================
# 4. KMeans-based imputation (fit on train fold only)
# ================================================================
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


# ================================================================
# 5. safe_predict
# ================================================================
def safe_predict(model, X_input, context_label=""):
    y_pred = model.predict(X_input, verbose=0).flatten()
    bad_mask = ~np.isfinite(y_pred)
    n_bad = int(bad_mask.sum())
    if n_bad > 0:
        print(f"[WARNING] {context_label}: {n_bad} NaN/Inf predictions "
              f"(model likely diverged) -> this combo will be excluded")
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    return y_pred, n_bad


def _make_optimizer(optimizer, learning_rate):
    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            return Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            return SGD(learning_rate=learning_rate)
        else:
            return Adam(learning_rate=learning_rate)
    return optimizer


# ================================================================
# 5a. Model architectures
#     'standard'  : plain MLP -- unconstrained, used for baseline DNN and
#                   for all soft (loss-based) physics-informed variants.
#     'monotonic' : architecture with a HARD monotonicity guarantee. The two
#                   physics-relevant features are routed through a branch
#                   with non-negative weights only and monotonic (ReLU)
#                   activations, so the branch output is provably
#                   NON-DECREASING in both inputs (Discharge Time and
#                   Decrement) regardless of what is learned -- no loss
#                   penalty is needed to obtain the constraint, unlike the
#                   soft-penalty variants.
# ================================================================
def build_model(optimizer='adam', learning_rate=0.001, neurons=64, input_shape=None):
    """Standard, unconstrained MLP (used for baseline + soft-penalty variants)."""
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1)
    ])
    opt = _make_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model


def build_model_monotonic(optimizer='adam', learning_rate=0.001, neurons=64, input_shape=None,
                           idx_DT=idx_DT, idx_DEC=idx_DEC):
    """
    Hard-constrained alternative architecture. Splits the input into:
      - a monotonic branch (Discharge Time and Decrement, both used AS-IS)
        passed through NonNeg()-constrained Dense layers with ReLU (a
        monotonic activation), which guarantees the branch is
        non-decreasing in both inputs by construction (RUL non-decreasing
        in both features, matching the soft-penalty variants);
      - a free branch for all remaining features, unconstrained.
    The two branch outputs are summed. Because the free branch never sees
    Discharge Time / Decrement, the sum is still provably monotonic in those
    two features no matter what the free branch learns.
    """
    all_idx = list(range(input_shape))
    mono_idx = [idx_DT, idx_DEC]
    free_idx = [i for i in all_idx if i not in mono_idx]

    inputs = layers.Input(shape=(input_shape,))

    def _mono_transform(x):
        dt = x[:, idx_DT:idx_DT + 1]
        dec = x[:, idx_DEC:idx_DEC + 1]
        # RUL must be non-decreasing in BOTH features -> feed them as-is
        # (no negation) into the NonNeg-weight branch.
        return tf.concat([dt, dec], axis=1)

    def _free_transform(x, free_idx=free_idx):
        idx_tensor = tf.constant(free_idx, dtype=tf.int32)
        return tf.gather(x, idx_tensor, axis=1)

    mono_input = layers.Lambda(_mono_transform, output_shape=(2,))(inputs)
    mono_h = layers.Dense(max(neurons // 2, 4), activation='relu', kernel_constraint=NonNeg())(mono_input)
    mono_h = layers.Dense(max(neurons // 4, 4), activation='relu', kernel_constraint=NonNeg())(mono_h)
    mono_out = layers.Dense(1, kernel_constraint=NonNeg())(mono_h)

    free_input = layers.Lambda(_free_transform, output_shape=(len(free_idx),))(inputs)
    free_h = layers.Dense(neurons, activation='relu')(free_input)
    free_out = layers.Dense(1)(free_h)

    combined = layers.Add()([mono_out, free_out])
    model = keras.Model(inputs=inputs, outputs=combined)

    opt = _make_optimizer(optimizer, learning_rate)
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    return model


ARCHITECTURE_BUILDERS = {
    'standard': build_model,
    'monotonic': build_model_monotonic,
}


# ================================================================
# 5b. Physics-informed diagnostic metrics (evaluation-time only; identical
#     for every variant so the comparison across variants is apples-to-apples)
#
#     Constraint (both features): RUL non-decreasing as the feature grows
#       -> correct gradient is >= 0 ; gradient < 0 counts as a violation.
# ================================================================
def compute_gradient_metrics(model, X_scaled, idx_DT, idx_DEC, batch_size=256):
    """
    คำนวณ dRUL_pred/dx สำหรับทุกตัวอย่างใน X_scaled (หลัง scale แล้ว) ด้วย automatic
    differentiation บนโมเดลที่เทรนเสร็จแล้ว (inference-time gradient, ไม่ได้เทรนเพิ่ม)
    """
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

    # ให้สอดคล้องกับ penalty ใน train step: ทั้งสอง feature ต้องมี gradient >= 0
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
    implausible_mask = y_pred < rul_lower_bound

    return {
        'MVR_DischargeTime': float(np.mean(viol_DT)),
        'MVR_Decrement': float(np.mean(viol_DEC)),
        'MVR_overall': float(np.mean(viol_any)),
        'Gradient_Consistency': float(np.mean(consistent_both)),
        'Physically_Implausible_Rate': float(np.mean(implausible_mask)),
        'n_implausible': int(implausible_mask.sum()),
        'n_samples': n,
        'mean_grad_DT': float(np.mean(grad_info['grad_DT'])),
        'mean_grad_DEC': float(np.mean(grad_info['grad_DEC'])),
    }


# ================================================================
# 6. Training step dispatcher -- one factory covers baseline + every
#    constraint formulation, so the rest of the pipeline (imputation,
#    scaling, grid search, outer CV, metrics) never has to change per
#    variant. `constraint` is one of:
#      'none'            -> plain MAE loss (baseline DNN; also used with the
#                            'monotonic' architecture, where the constraint
#                            is structural rather than in the loss)
#      'gradient_hinge'   -> proposed PI-DNN: linear hinge penalty
#                            relu(-d(pred)/d(feature)) for BOTH features
#      'gradient_l2'      -> alternative soft penalty: same hinge but
#                            squared, penalizing large violations more
#                            steeply than small ones
#      'pairwise_ranking' -> alternative enforcement mechanism that avoids
#                            gradients entirely: perturbs each training
#                            example upward along one monotonic feature at a
#                            time and penalizes the model if the *prediction*
#                            goes DOWN
#
#    Constraint (both features): RUL non-decreasing as the feature grows.
# ================================================================
def make_train_step(constraint, idx_DT, idx_DEC, lambda_mono, rank_delta=0.1):
    def train_step(model, optimizer, X_batch, y_batch):
        if len(X_batch.shape) == 1:
            X_batch = tf.expand_dims(X_batch, axis=0)

        with tf.GradientTape() as outer_tape:
            if constraint in ('gradient_hinge', 'gradient_l2'):
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(X_batch)
                    preds = tf.squeeze(model(X_batch, training=True), axis=1)
                input_grads = inner_tape.gradient(preds, X_batch)
                mae = tf.reduce_mean(tf.abs(y_batch - preds))
                if input_grads is not None:
                    # RUL non-decreasing in both features -> penalize negative gradient
                    viol_DT = tf.nn.relu(-input_grads[:, idx_DT])
                    viol_DEC = tf.nn.relu(-input_grads[:, idx_DEC])
                    if constraint == 'gradient_l2':
                        viol_DT = tf.square(viol_DT)
                        viol_DEC = tf.square(viol_DEC)
                    mono = tf.reduce_mean(viol_DT) + tf.reduce_mean(viol_DEC)
                else:
                    mono = 0.0
                loss = mae + lambda_mono * mono

            elif constraint == 'pairwise_ranking':
                preds = tf.squeeze(model(X_batch, training=True), axis=1)
                mae = tf.reduce_mean(tf.abs(y_batch - preds))

                n_features = X_batch.shape[1]
                delta_DT = tf.one_hot(idx_DT, n_features, dtype=X_batch.dtype) * rank_delta
                delta_DEC = tf.one_hot(idx_DEC, n_features, dtype=X_batch.dtype) * rank_delta

                preds_DT_pert = tf.squeeze(model(X_batch + delta_DT, training=True), axis=1)
                preds_DEC_pert = tf.squeeze(model(X_batch + delta_DEC, training=True), axis=1)

                # RUL must not decrease when Discharge Time increases
                rank_DT = tf.reduce_mean(tf.nn.relu(preds - preds_DT_pert))
                # RUL must not decrease when Decrement increases
                rank_DEC = tf.reduce_mean(tf.nn.relu(preds - preds_DEC_pert))
                mono = rank_DT + rank_DEC
                loss = mae + lambda_mono * mono

            else:  # constraint == 'none'
                preds = tf.squeeze(model(X_batch, training=True), axis=1)
                mae = tf.reduce_mean(tf.abs(y_batch - preds))
                mono = 0.0
                loss = mae

        grads = outer_tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, mae, mono

    return train_step


def train_variant(model, X, y, constraint, idx_DT, idx_DEC, lambda_mono,
                   epochs=20, batch_size=8, rank_delta=0.1, seed=SEED):
    """
    Generic training loop used by every ablation arm. A fresh tf.function is
    compiled per (model, optimizer) pair for the same reason as in the
    original script: reusing one global compiled step across many distinct
    optimizer instances stops TF from creating new slot variables.

    การ shuffle ใช้ np.random.RandomState(seed) ที่สร้างใหม่ทุกครั้งที่เรียกฟังก์ชันนี้
    -> ทุก variant / combo / fold ได้ลำดับ shuffle ชุดเดียวกัน (reproducible และเทียบกันได้ตรง ๆ)
    """
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y, dtype=tf.float32)
    n = int(X_tf.shape[0])
    optimizer = model.optimizer
    rng = np.random.RandomState(seed)

    step_fn = tf.function(make_train_step(constraint, idx_DT, idx_DEC, lambda_mono, rank_delta))

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


# ================================================================
# 7. Ablation variants
#    Each row isolates ONE change from the proposed PI-DNN so the source of
#    any performance/physics difference can be attributed:
#      - Baseline-DNN            : removes the physics prior entirely
#      - PI-DNN (hinge, proposed): the method being validated
#      - PI-DNN (L2 penalty)     : same idea, different penalty shape
#      - PI-DNN (pairwise rank)  : same idea, gradient-free enforcement
#      - Hard-Constrained Arch.  : constraint moved from loss to architecture
# ================================================================
VARIANTS = [
    {'name': 'Baseline-DNN',              'architecture': 'standard',  'constraint': 'none',             'lambda_mono': 0.0},
    {'name': 'PI-DNN (hinge, proposed)',  'architecture': 'standard',  'constraint': 'gradient_hinge',   'lambda_mono': 0.001},
    {'name': 'PI-DNN (L2 penalty)',       'architecture': 'standard',  'constraint': 'gradient_l2',      'lambda_mono': 0.001},
    {'name': 'PI-DNN (pairwise rank)',    'architecture': 'standard',  'constraint': 'pairwise_ranking', 'lambda_mono': 0.001},
    {'name': 'Hard-Constrained Arch.',    'architecture': 'monotonic', 'constraint': 'none',             'lambda_mono': 0.0},
]
PROPOSED_VARIANT_NAME = 'PI-DNN (hinge, proposed)'  # used later for SHAP

if QUICK_TEST_MODE:
    VARIANTS = VARIANTS[:2]

# ================================================================
# 8. Hyperparameter grid (shared across all variants)
# ================================================================
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'neurons': [32, 64, 128] if not QUICK_TEST_MODE else [32],
}
param_options = [
    {'optimizer': opt, 'neurons': n}
    for opt in param_grid['optimizer']
    for n in param_grid['neurons']
]

EPOCHS = 20 if not QUICK_TEST_MODE else 3
BATCH_SIZE = 64

# ================================================================
# 9. Nested CV runner -- identical protocol for every variant (same outer
#    LeaveOneGroupOut folds, same outlier handling / imputation / scaling,
#    same inner grid search objective) so that only the constraint
#    formulation differs between runs.
# ================================================================
def run_nested_cv(variant_cfg, X_raw, y, groups, idx_DT, idx_DEC, param_options,
                   epochs=20, batch_size=64):
    name = variant_cfg['name']
    arch = variant_cfg['architecture']
    constraint = variant_cfg['constraint']
    lambda_mono = variant_cfg['lambda_mono']
    build_fn = ARCHITECTURE_BUILDERS[arch]

    n_batteries = groups.nunique()
    outer_gkf = LeaveOneGroupOut()

    fold_results = []
    all_test_preds, all_test_true = [], []
    last_model, last_X_test_scaled = None, None

    for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(X_raw, y, groups=groups), 1):
        print(f"\n===== [{name}] Outer Fold {fold}/{n_batteries} =====")
        start_time = time.time()
        tracemalloc.start()

        X_train_df = X_raw.iloc[train_idx].reset_index(drop=True)
        X_test_df = X_raw.iloc[test_idx].reset_index(drop=True)
        y_train_full = y.iloc[train_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)
        groups_train_full = groups.iloc[train_idx].reset_index(drop=True)

        outlier_bounds = fit_outlier_bounds(X_train_df)
        X_train_df = apply_outlier_bounds(X_train_df, outlier_bounds, cols_must_be_positive)
        X_test_df = apply_outlier_bounds(X_test_df, outlier_bounds, cols_must_be_positive)

        X_train_imputed, km_model, km_scaler, cluster_means, global_means = kmeans_impute_fit(X_train_df)
        X_test_imputed = kmeans_impute_transform(X_test_df, km_model, km_scaler, cluster_means, global_means)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        inner_train_idx, inner_val_idx = next(gss.split(X_train_imputed, y_train_full, groups=groups_train_full))

        X_inner_train_df = X_train_imputed.iloc[inner_train_idx]
        X_inner_val_df = X_train_imputed.iloc[inner_val_idx]
        y_inner_train = y_train_full.iloc[inner_train_idx]
        y_inner_val = y_train_full.iloc[inner_val_idx]

        inner_scaler = StandardScaler()
        X_inner_train_scaled = inner_scaler.fit_transform(X_inner_train_df)
        X_inner_val_scaled = inner_scaler.transform(X_inner_val_df)

        y_scaler_inner = StandardScaler()
        y_inner_train_scaled = y_scaler_inner.fit_transform(y_inner_train.values.reshape(-1, 1)).flatten()

        # ---- Inner grid search ----
        best_val_mae = float('inf')
        best_params = None
        for params in param_options:
            reset_seeds(SEED)  # ทุก combo / variant เริ่มจาก weight init เดียวกัน
            model_tmp = build_fn(
                input_shape=X_inner_train_scaled.shape[1],
                optimizer=params['optimizer'],
                neurons=params['neurons'],
            )
            diverged_train = train_variant(
                model_tmp, X_inner_train_scaled, y_inner_train_scaled,
                constraint=constraint, idx_DT=idx_DT, idx_DEC=idx_DEC,
                lambda_mono=lambda_mono, epochs=epochs, batch_size=batch_size,
                seed=SEED,
            )
            if diverged_train:
                print(f"  -> [{name}] dropping params={params} (diverged during training)")
                continue
            y_val_pred_scaled, n_bad = safe_predict(
                model_tmp, X_inner_val_scaled, context_label=f"[{name}] Fold {fold} grid params={params}"
            )
            if n_bad > 0:
                print(f"  -> [{name}] dropping params={params} (NaN/Inf predictions)")
                continue
            y_val_pred = y_scaler_inner.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
            val_mae = mean_absolute_error(y_inner_val, y_val_pred)
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_params = params

        if best_params is None:
            print(f"[WARNING] [{name}] Fold {fold}: every hyperparameter combo diverged, "
                  f"falling back to (adam, 32)")
            best_params = {'optimizer': 'adam', 'neurons': 32}

        print(f"[{name}] Fold {fold} best params (inner val MAE={best_val_mae:.4f}): {best_params}")

        # ---- Retrain on full train fold ----
        final_scaler = StandardScaler()
        X_train_scaled = final_scaler.fit_transform(X_train_imputed)
        X_test_scaled = final_scaler.transform(X_test_imputed)

        y_scaler_final = StandardScaler()
        y_train_scaled = y_scaler_final.fit_transform(y_train_full.values.reshape(-1, 1)).flatten()

        reset_seeds(SEED)
        best_model = build_fn(
            input_shape=X_train_scaled.shape[1],
            optimizer=best_params['optimizer'],
            neurons=best_params['neurons'],
        )
        final_diverged = train_variant(
            best_model, X_train_scaled, y_train_scaled,
            constraint=constraint, idx_DT=idx_DT, idx_DEC=idx_DEC,
            lambda_mono=lambda_mono, epochs=epochs, batch_size=batch_size,
            seed=SEED,
        )
        if final_diverged:
            print(f"[WARNING] [{name}] Fold {fold}: final model diverged (params={best_params})")

        y_train_pred_scaled, n_bad_train = safe_predict(best_model, X_train_scaled, f"[{name}] Fold {fold} train")
        y_train_pred = y_scaler_final.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
        train_mae = mean_absolute_error(y_train_full, y_train_pred)
        train_rmse = math.sqrt(mean_squared_error(y_train_full, y_train_pred))
        train_r2 = r2_score(y_train_full, y_train_pred)
        y_train_nz = y_train_full[y_train_full != 0]
        y_train_pred_nz = y_train_pred[y_train_full != 0]
        train_mape = np.mean(np.abs((y_train_nz - y_train_pred_nz) / y_train_nz)) * 100

        y_pred_scaled, n_bad_test = safe_predict(best_model, X_test_scaled, f"[{name}] Fold {fold} test")
        y_pred = y_scaler_final.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        y_true_nz = y_test[y_test != 0]
        y_pred_nz = y_pred[y_test != 0]
        mape = np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100

        physics_train = compute_physics_informed_metrics(best_model, X_train_scaled, y_train_pred, idx_DT, idx_DEC)
        physics_test = compute_physics_informed_metrics(best_model, X_test_scaled, y_pred, idx_DT, idx_DEC)

        all_test_preds.append(pd.Series(y_pred, index=test_idx))
        all_test_true.append(pd.Series(y_test.values, index=test_idx))

        elapsed_time = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        fold_results.append({
            'variant': name, 'fold': fold,
            'Train_MAE': train_mae, 'Train_RMSE': train_rmse, 'Train_R2': train_r2, 'Train_MAPE': train_mape,
            'Test_MAE': mae, 'Test_RMSE': rmse, 'Test_R2': r2, 'Test_MAPE': mape,
            'Train_MVR_DischargeTime': physics_train['MVR_DischargeTime'],
            'Train_MVR_Decrement': physics_train['MVR_Decrement'],
            'Train_MVR_overall': physics_train['MVR_overall'],
            'Train_Gradient_Consistency': physics_train['Gradient_Consistency'],
            'Train_Physically_Implausible_Rate': physics_train['Physically_Implausible_Rate'],
            'Test_MVR_DischargeTime': physics_test['MVR_DischargeTime'],
            'Test_MVR_Decrement': physics_test['MVR_Decrement'],
            'Test_MVR_overall': physics_test['MVR_overall'],
            'Test_Gradient_Consistency': physics_test['Gradient_Consistency'],
            'Test_Physically_Implausible_Rate': physics_test['Physically_Implausible_Rate'],
            'time_sec': elapsed_time, 'memory_MB': peak / (1024 * 1024),
            'best_params': best_params,
            'diverged': (n_bad_train > 0 or n_bad_test > 0 or final_diverged),
        })

        print(f"[{name}] Fold {fold} | Test MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} MAPE={mape:.4f}% | "
              f"MVR={physics_test['MVR_overall']*100:.2f}% GradCons={physics_test['Gradient_Consistency']*100:.2f}% "
              f"PIP={physics_test['Physically_Implausible_Rate']*100:.2f}%")

        last_model, last_X_test_scaled = best_model, X_test_scaled

    fold_results_df = pd.DataFrame(fold_results)
    all_test_preds_df = pd.concat(all_test_preds).sort_index()
    all_test_true_df = pd.concat(all_test_true).sort_index()
    return fold_results_df, all_test_preds_df, all_test_true_df, last_model, last_X_test_scaled


# ================================================================
# 10. Run every ablation arm
# ================================================================
all_variant_fold_results = []
variant_artifacts = {}  # name -> (last_model, last_X_test_scaled) for SHAP later

for variant_cfg in VARIANTS:
    fold_results_df, preds_df, true_df, last_model, last_X_test_scaled = run_nested_cv(
        variant_cfg, X_raw, y, groups, idx_DT, idx_DEC, param_options,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
    )
    all_variant_fold_results.append(fold_results_df)
    variant_artifacts[variant_cfg['name']] = {
        'model': last_model, 'X_test_scaled': last_X_test_scaled,
        'preds': preds_df, 'true': true_df,
    }

master_results_df = pd.concat(all_variant_fold_results, ignore_index=True)
master_results_df.to_csv(os.path.join(output_dir, 'ablation_fold_results.csv'), index=False)

# Drop diverged folds per-variant before summarizing (mirrors original script's logic)
clean_df = master_results_df[~master_results_df['diverged']].copy()
dropped = master_results_df[master_results_df['diverged']]
if len(dropped) > 0:
    print(f"\n[WARNING] Excluding {len(dropped)} diverged (variant, fold) combinations from summary stats:")
    print(dropped[['variant', 'fold', 'best_params']].to_string(index=False))

# ================================================================
# 11. Ablation summary table (mean +/- std across folds, per variant)
# ================================================================
summary_cols = ['Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MAPE',
                 'Test_MVR_overall', 'Test_Gradient_Consistency', 'Test_Physically_Implausible_Rate']

ablation_summary = clean_df.groupby('variant')[summary_cols].agg(['mean', 'std'])
ablation_summary.columns = ['_'.join(c) for c in ablation_summary.columns]
# Order rows to match VARIANTS definition order, not alphabetically
variant_order = [v['name'] for v in VARIANTS]
ablation_summary = ablation_summary.reindex(variant_order)
ablation_summary.to_csv(os.path.join(output_dir, 'ablation_summary.csv'))

print("\n===== ABLATION SUMMARY (mean +/- std across outer folds) =====")
for v in variant_order:
    row = ablation_summary.loc[v]
    print(f"\n{v}")
    print(f"  Test MAE  : {row['Test_MAE_mean']:.4f} +/- {row['Test_MAE_std']:.4f}")
    print(f"  Test RMSE : {row['Test_RMSE_mean']:.4f} +/- {row['Test_RMSE_std']:.4f}")
    print(f"  Test R2   : {row['Test_R2_mean']:.4f} +/- {row['Test_R2_std']:.4f}")
    print(f"  Test MAPE : {row['Test_MAPE_mean']:.4f}% +/- {row['Test_MAPE_std']:.4f}%")
    print(f"  MVR (overall)         : {row['Test_MVR_overall_mean']*100:.2f}% +/- {row['Test_MVR_overall_std']*100:.2f}%")
    print(f"  Gradient Consistency  : {row['Test_Gradient_Consistency_mean']*100:.2f}% +/- {row['Test_Gradient_Consistency_std']*100:.2f}%")
    print(f"  Physically Implausible: {row['Test_Physically_Implausible_Rate_mean']*100:.2f}% +/- {row['Test_Physically_Implausible_Rate_std']*100:.2f}%")

# ================================================================
# 12. Paired significance tests: does the proposed PI-DNN differ from each
#     other variant on the SAME held-out batteries? (Wilcoxon signed-rank,
#     paired by fold/battery since every variant uses identical outer folds)
# ================================================================
def paired_test(df, variant_a, variant_b, metric):
    a = df[df['variant'] == variant_a].sort_values('fold')
    b = df[df['variant'] == variant_b].sort_values('fold')
    common_folds = sorted(set(a['fold']) & set(b['fold']))
    a = a[a['fold'].isin(common_folds)].sort_values('fold')[metric].values
    b = b[b['fold'].isin(common_folds)].sort_values('fold')[metric].values
    if len(a) < 2 or np.allclose(a, b):
        return None, len(a)
    try:
        stat, p = wilcoxon(a, b)
    except ValueError:
        return None, len(a)
    return p, len(a)


print("\n===== Paired Wilcoxon signed-rank tests vs. proposed PI-DNN (same held-out batteries) =====")
sig_rows = []
for v in variant_order:
    if v == PROPOSED_VARIANT_NAME:
        continue
    for metric, label in [('Test_MAE', 'Test MAE'), ('Test_MVR_overall', 'MVR (overall)')]:
        p, n = paired_test(clean_df, PROPOSED_VARIANT_NAME, v, metric)
        if p is None:
            print(f"  {PROPOSED_VARIANT_NAME} vs {v} [{label}]: not enough paired/clean folds (n={n}) to test")
        else:
            sig = "significant (p<0.05)" if p < 0.05 else "not significant"
            print(f"  {PROPOSED_VARIANT_NAME} vs {v} [{label}]: p={p:.4f} (n={n} paired folds) -> {sig}")
        sig_rows.append({'comparison': f"{PROPOSED_VARIANT_NAME} vs {v}", 'metric': label, 'p_value': p, 'n_folds': n})

pd.DataFrame(sig_rows).to_csv(os.path.join(output_dir, 'ablation_significance_tests.csv'), index=False)

# ================================================================
# 13. Comparison plots
# ================================================================
plot_metrics = [
    ('Test_MAE', 'Test MAE (lower is better)', False),
    ('Test_R2', 'Test R2 (higher is better)', False),
    ('Test_MVR_overall', 'Monotonicity Violation Rate (%) (lower is better)', True),
    ('Test_Gradient_Consistency', 'Gradient Consistency (%) (higher is better)', True),
    ('Test_Physically_Implausible_Rate', 'Physically Implausible Rate (%) (lower is better)', True),
]

fig, axes = plt.subplots(1, len(plot_metrics), figsize=(5 * len(plot_metrics), 5))
if len(plot_metrics) == 1:
    axes = [axes]

for ax, (col, title, as_pct) in zip(axes, plot_metrics):
    means = [ablation_summary.loc[v, f'{col}_mean'] * (100 if as_pct else 1) for v in variant_order]
    stds = [ablation_summary.loc[v, f'{col}_std'] * (100 if as_pct else 1) for v in variant_order]
    ax.bar(range(len(variant_order)), means, yerr=stds, capsize=4,
           color=['#888888' if v == 'Baseline-DNN' else
                  ('#2c7fb8' if v == PROPOSED_VARIANT_NAME else '#a1cfe8') for v in variant_order])
    ax.set_xticks(range(len(variant_order)))
    ax.set_xticklabels(variant_order, rotation=45, ha='right', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ablation_comparison.png'), dpi=600, bbox_inches='tight')
plt.show()

# Predicted vs actual, one panel per variant
fig, axes = plt.subplots(1, len(variant_order), figsize=(5 * len(variant_order), 5), sharex=True, sharey=True)
if len(variant_order) == 1:
    axes = [axes]
for ax, v in zip(axes, variant_order):
    true_v = variant_artifacts[v]['true']
    pred_v = variant_artifacts[v]['preds']
    ax.scatter(true_v.values, pred_v.values, color='red', s=10, alpha=0.6)
    lims = [min(true_v.min(), pred_v.min()), max(true_v.max(), pred_v.max())]
    ax.plot(lims, lims, color='blue', linestyle='--')
    ax.set_title(v, fontsize=9)
    ax.set_xlabel('Actual RUL')
ax_first = axes[0]
ax_first.set_ylabel('Predicted RUL')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ablation_pred_vs_actual.png'), dpi=600, bbox_inches='tight')
plt.show()

# ================================================================
# 14. SHAP -- only for the proposed PI-DNN (as in the original script), to
#     explain what the validated model relies on, using its last-fold model
# ================================================================
img_revision_dir = os.path.join(output_dir, 'imgRevision')
os.makedirs(img_revision_dir, exist_ok=True)

proposed_model = variant_artifacts[PROPOSED_VARIANT_NAME]['model']
proposed_X_test_scaled = variant_artifacts[PROPOSED_VARIANT_NAME]['X_test_scaled']

background_data = proposed_X_test_scaled[:min(100, proposed_X_test_scaled.shape[0])]
num_samples = proposed_X_test_scaled.shape[0] // 2
X_test_for_shap = proposed_X_test_scaled[:max(num_samples, 1)]

reset_seeds(SEED)  # GradientExplainer สุ่ม interpolation/background -> reset seed ก่อน
explainer = shap.GradientExplainer(proposed_model, background_data)
shap_values = explainer.shap_values(X_test_for_shap)

if isinstance(shap_values, list):
    shap_values = shap_values[0]
if shap_values.ndim == 3:
    shap_values_2d = shap_values.reshape(shap_values.shape[0], shap_values.shape[1])
else:
    shap_values_2d = shap_values

shap.summary_plot(shap_values_2d, X_test_for_shap, feature_names=features, show=False)
plt.savefig(os.path.join(img_revision_dir, 'shap_summary_plot_proposed.png'), dpi=600, bbox_inches='tight')
plt.show()
plt.close()

X_test_for_plot_df = pd.DataFrame(X_test_for_shap, columns=features)
for i, feature in enumerate(features):
    plt.figure(figsize=(6, 4))
    shap.dependence_plot(feature, shap_values_2d, X_test_for_plot_df, show=False)
    plt.savefig(os.path.join(img_revision_dir,
                              f'shap_dependence_{i}_{feature.replace(" ", "_").replace(".", "")}.png'),
                dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()

print(f"\n[INFO] Ablation results, plots, and SHAP figures saved under: {output_dir}")