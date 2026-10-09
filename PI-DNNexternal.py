# ================================
# -1. Fix seed (ต้องตั้ง env ก่อน import tensorflow)
#     แนะนำให้รันด้วย: PYTHONHASHSEED=42 python pi_dnn_external_rul.py
# ================================
import os
import random

SEED = 42
SUBSET_SEED = 40  # คงเลข 40 เดิมไว้ เพื่อให้ได้ 10 cells ชุดเดิมที่ใช้ใน paper

os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import time
import math
import tracemalloc  # สำหรับ memory usage

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD


def reset_seeds(seed=SEED):
    """reset seed ของ python / numpy / tensorflow (เรียกก่อนสร้างโมเดลทุก fold)"""
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

# ================================
# 0. Output dir (ใช้เก็บรูปท้ายสคริปต์)
# ================================
output_dir = "imgRevision/external/outputs/pi-dnn"
os.makedirs(output_dir, exist_ok=True)

# ================================
# 1. Load .mat -> bat_dict (เหมือนเดิม)
# ================================
matFilename = '2018-02-20_batchdata_updated_struct_errorcorrect.mat'
f = h5py.File(matFilename, 'r')
batch = f['batch']  # ชื่อ group ของ dataset อาจต้องตรวจสอบ f.keys()

bat_dict = {}
num_cells = batch['summary'].shape[0]
for i in range(num_cells):
    cl = f[batch['cycle_life'][i, 0]][:]
    policy = f[batch['policy_readable'][i, 0]][:].tobytes()[::2].decode()
    summary_IR = np.hstack(f[batch['summary'][i, 0]]['IR'][0, :].tolist())
    summary_QC = np.hstack(f[batch['summary'][i, 0]]['QCharge'][0, :].tolist())
    summary_QD = np.hstack(f[batch['summary'][i, 0]]['QDischarge'][0, :].tolist())
    summary_TA = np.hstack(f[batch['summary'][i, 0]]['Tavg'][0, :].tolist())
    summary_TM = np.hstack(f[batch['summary'][i, 0]]['Tmin'][0, :].tolist())
    summary_TX = np.hstack(f[batch['summary'][i, 0]]['Tmax'][0, :].tolist())
    summary_CT = np.hstack(f[batch['summary'][i, 0]]['chargetime'][0, :].tolist())
    summary_CY = np.hstack(f[batch['summary'][i, 0]]['cycle'][0, :].tolist())
    summary = {'IR': summary_IR, 'QC': summary_QC, 'QD': summary_QD, 'Tavg':
               summary_TA, 'Tmin': summary_TM, 'Tmax': summary_TX, 'chargetime': summary_CT,
               'cycle': summary_CY}
    cycles = f[batch['cycles'][i, 0]]
    cycle_dict = {}
    for j in range(cycles['I'].shape[0]):
        I = np.hstack((f[cycles['I'][j, 0]][:]))
        Qc = np.hstack((f[cycles['Qc'][j, 0]][:]))
        Qd = np.hstack((f[cycles['Qd'][j, 0]][:]))
        Qdlin = np.hstack((f[cycles['Qdlin'][j, 0]][:]))
        T = np.hstack((f[cycles['T'][j, 0]][:]))
        Tdlin = np.hstack((f[cycles['Tdlin'][j, 0]][:]))
        V = np.hstack((f[cycles['V'][j, 0]][:]))
        dQdV = np.hstack((f[cycles['discharge_dQdV'][j, 0]][:]))
        t = np.hstack((f[cycles['t'][j, 0]][:]))
        cd = {'I': I, 'Qc': Qc, 'Qd': Qd, 'Qdlin': Qdlin, 'T': T, 'Tdlin': Tdlin, 'V': V, 'dQdV': dQdV, 't': t}
        cycle_dict[str(j)] = cd

    cell_dict = {'cycle_life': cl, 'charge_policy': policy, 'summary': summary, 'cycles': cycle_dict}
    key = 'b1c' + str(i)
    bat_dict[key] = cell_dict


# ================================
# 2. สร้าง DataFrame ระดับ per-cycle (เหมือนเดิม)
# ================================
rows = []
for cell_key, cell_data in bat_dict.items():
    _cl_arr = np.asarray(cell_data['cycle_life']).flatten()
    cycle_life = float(_cl_arr[0]) if _cl_arr.size > 0 else np.nan
    summary = cell_data['summary']
    num_cycles = len(summary['cycle'])

    for i in range(num_cycles):
        row = {
            'cell': cell_key,
            'cycle_index': summary['cycle'][i],
            'RUL': cycle_life - summary['cycle'][i],  # Remaining Useful Life
        }
        row['IR'] = summary['IR'][i]
        row['QC'] = summary['QC'][i]
        row['QD'] = summary['QD'][i]
        row['Tavg'] = summary['Tavg'][i]
        row['Tmin'] = summary['Tmin'][i]
        row['Tmax'] = summary['Tmax'][i]
        row['chargetime'] = summary['chargetime'][i]

        rows.append(row)

df = pd.DataFrame(rows)

# ================================
# 3. กำหนด features / target / group
#    หมายเหตุ: target แก้จาก 'cycle_index' -> 'RUL' เพราะทั้งกราฟ/ชื่อแกน/ชื่อ
#    ไฟล์ท้ายสคริปต์อ้างถึง "RUL" ทั้งหมด (ของเดิม y = df['cycle_index'] ซึ่ง
#    ไม่ตรงกับ label ที่เหลือ) ถ้าตั้งใจจะ predict cycle_index จริงๆ แจ้งได้เลย
# ================================
features = ['IR', 'QC', 'QD', 'Tavg', 'Tmin', 'Tmax', 'chargetime']
target_col = 'RUL'
group_col = 'cell'

X_raw = df[features].copy()
y_raw = df[target_col].copy()
groups = df[group_col].copy()

# ---- ตัดแถวที่ target (RUL) เป็น NaN/Inf ออกก่อนเข้า CV ----
y_raw_numeric = pd.to_numeric(y_raw, errors='coerce')
valid_mask = np.isfinite(y_raw_numeric.values)
n_dropped = int((~valid_mask).sum())
if n_dropped > 0:
    print(f"[WARNING] พบ RUL เป็น NaN/Inf จำนวน {n_dropped} แถว (จากทั้งหมด {len(y_raw_numeric)}) "
          f"-> ตัดออกก่อนเข้า CV")

X_raw = X_raw.loc[valid_mask].reset_index(drop=True)
y_raw = y_raw_numeric.loc[valid_mask].reset_index(drop=True)
groups = groups.loc[valid_mask].reset_index(drop=True)

# ---- สุ่มเลือก 10 cells สำหรับทำ additional/external validation dataset ใน paper ----
rng_subset = np.random.RandomState(SUBSET_SEED)
all_cells = groups.unique()
n_cells_subset = 10
if len(all_cells) > n_cells_subset:
    selected_cells = rng_subset.choice(all_cells, size=n_cells_subset, replace=False)
    print(f"\n[INFO] สุ่มเลือก {n_cells_subset} cells จากทั้งหมด {len(all_cells)} cells: {sorted(selected_cells)}")
    subset_mask = groups.isin(selected_cells)
    X_raw = X_raw.loc[subset_mask].reset_index(drop=True)
    y_raw = y_raw.loc[subset_mask].reset_index(drop=True)
    groups = groups.loc[subset_mask].reset_index(drop=True)
else:
    print(f"\n[INFO] จำนวน cells ({len(all_cells)}) น้อยกว่าหรือเท่ากับ {n_cells_subset} อยู่แล้ว ใช้ทั้งหมด")

feature_names = list(X_raw.columns)
idx_QC = feature_names.index('QC')
idx_QD = feature_names.index('QD')
idx_Tmax = feature_names.index('Tmax')

# Features ที่ใช้ใน monotonicity penalty
# *** ทิศทาง (sign) ไม่ถูก hard-code: ถูกกำหนดด้วย Pearson + voting จาก training cells
#     ในแต่ละ LOCO fold (ดู vote_monotonic_signs) ลำดับต้องตรงกับ mono_idx ด้านล่าง ***
mono_feature_names = ['QC', 'QD', 'Tmax']
mono_idx = [idx_QC, idx_QD, idx_Tmax]

input_shape = len(features)

# ================================
# 4. Preprocessing helpers: fit เฉพาะ train fold แล้ว apply ไปที่ทั้ง train/test
# ================================
def replace_negative_with_nan(X_df, cols):
    """แทนค่าติดลบด้วย NaN แบบ row-wise ไม่ต้อง fit จึงใช้กับ train หรือ test ตรงๆ ได้"""
    X_out = X_df.copy()
    for col in cols:
        if col in X_out.columns:
            X_out.loc[X_out[col] < 0, col] = np.nan
    return X_out


def fit_outlier_bounds(X_train_df):
    """คำนวณ IQR bounds จาก train fold เท่านั้น คืนค่า dict ของ (lower, upper) ต่อคอลัมน์"""
    bounds = {}
    for col in X_train_df.select_dtypes(include=[np.number]).columns:
        series = X_train_df[col]
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 2 * IQR, Q3 + 2 * IQR)
    return bounds


def apply_outlier_bounds(X_df, bounds):
    """ใช้ bounds ที่ fit จาก train fold มา mask ค่าผิดปกติเป็น NaN ให้กับ df ใดก็ได้ (train หรือ test)"""
    X_out = X_df.copy()
    for col, (lower, upper) in bounds.items():
        if col in X_out.columns:
            X_out.loc[(X_out[col] < lower) | (X_out[col] > upper), col] = np.nan
    return X_out


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
# 4b. Monotonic direction: Pearson (ต่อ cell) + majority voting
#     *** ใช้ข้อมูลจาก training cells เท่านั้น (cell ที่ held-out ไม่ถูกแตะเลย) ***
#
#       1) แต่ละ cell ใน training folds คำนวณ Pearson r(feature, RUL)
#       2) แต่ละ cell โหวต: r > 0 -> +1 , r < 0 -> -1 (ข้ามถ้า r เป็น NaN / |r| < min_abs_r)
#       3) sign สุดท้าย = เสียงข้างมาก (เสมอกัน -> ใช้ Pearson แบบ pooled บน training cells)
#
#     sign = +1 : RUL ควรไม่ลดลงเมื่อ feature เพิ่ม  -> penalize relu(-grad)
#     sign = -1 : RUL ควรไม่เพิ่มขึ้นเมื่อ feature เพิ่ม -> penalize relu(+grad)
#     (StandardScaler ใช้ std > 0 จึงไม่เปลี่ยนเครื่องหมายของ gradient)
#
#     แทนที่การ hard-code QC(+), QD(+), Tmax(-) ที่เคยอ้างอิง Spearman จากทั้ง 39 cells
#     (ซึ่งรวม cell ที่ถูก held-out ใน LOCO ด้วย)
# ================================
def vote_monotonic_signs(X_train_df, y_train, groups_train, feature_names_to_vote, min_abs_r=0.0):
    y_arr = np.asarray(y_train, dtype=float)
    g_arr = np.asarray(groups_train)
    result = {}

    for feat in feature_names_to_vote:
        x_arr = np.asarray(X_train_df[feat], dtype=float)

        per_cell_r = {}
        n_pos = n_neg = n_skipped = 0
        for g in np.unique(g_arr):
            m = g_arr == g
            xg, yg = x_arr[m], y_arr[m]
            if len(xg) < 3 or np.std(xg) == 0 or np.std(yg) == 0:
                n_skipped += 1
                per_cell_r[g] = np.nan
                continue
            r = np.corrcoef(xg, yg)[0, 1]
            per_cell_r[g] = r
            if not np.isfinite(r) or abs(r) < min_abs_r:
                n_skipped += 1
            elif r > 0:
                n_pos += 1
            else:
                n_neg += 1

        if np.std(x_arr) > 0 and np.std(y_arr) > 0:
            pooled_r = float(np.corrcoef(x_arr, y_arr)[0, 1])
        else:
            pooled_r = 0.0

        if n_pos > n_neg:
            sign = 1.0
        elif n_neg > n_pos:
            sign = -1.0
        else:
            sign = 1.0 if pooled_r >= 0 else -1.0  # tie-break

        result[feat] = {
            'sign': sign, 'n_pos': n_pos, 'n_neg': n_neg,
            'n_skipped': n_skipped, 'pooled_r': pooled_r,
            'per_cell_r': per_cell_r,
        }
    return result


# ================================
# 5. Physics-Informed DNN model + sign-aware monotonicity penalty
#    penalty = sum_k relu(-s_k * dRUL/dx_k), k ∈ {QC, QD, Tmax}
#    โดย s_k ∈ {+1, -1} มาจาก Pearson + voting บน training cells (ส่งเข้ามาเป็น tensor
#    เพื่อไม่ให้ retrace/ผูกกับค่าตายตัว)
# ================================
def build_model(optimizer='adam', learning_rate=0.001, neurons=64):
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_shape,)))
    model.add(layers.Dense(neurons, activation='relu'))
    model.add(layers.Dense(1))

    if isinstance(optimizer, str):
        if optimizer.lower() == 'adam':
            opt = Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = SGD(learning_rate=learning_rate)
        else:
            opt = Adam(learning_rate=learning_rate)
    else:
        opt = optimizer

    model.compile(
        optimizer=opt,
        loss='mae',
        metrics=['mae'],
    )
    return model


lambda_mono = 0.001


def train_step(model, optimizer, X_batch, y_batch, mono_signs):
    """mono_signs: tensor shape (3,) = [s_QC, s_QD, s_Tmax]"""
    if len(X_batch.shape) == 1:
        X_batch = tf.expand_dims(X_batch, axis=0)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(X_batch)
            preds = tf.squeeze(model(X_batch, training=True), axis=1)

        input_grads = inner_tape.gradient(preds, X_batch)

        mae = tf.reduce_mean(tf.abs(y_batch - preds))

        if input_grads is not None:
            penalty_QC = tf.reduce_mean(tf.nn.relu(-mono_signs[0] * input_grads[:, idx_QC]))
            penalty_QD = tf.reduce_mean(tf.nn.relu(-mono_signs[1] * input_grads[:, idx_QD]))
            penalty_Tmax = tf.reduce_mean(tf.nn.relu(-mono_signs[2] * input_grads[:, idx_Tmax]))
            mono = penalty_QC + penalty_QD + penalty_Tmax
        else:
            mono = tf.constant(0.0, dtype=tf.float32)

        loss = mae + lambda_mono * mono

    grads = outer_tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss, mae, mono


# ================================
# 6. Fixed hyperparameter (ไม่ทำ grid search เพื่อความเร็ว
#    ใช้ optimizer=adam, neurons=64 ตายตัวสำหรับทุก fold)
#    *** ตรวจสอบให้ตรงกับค่าที่ระบุใน paper ***
# ================================
fixed_params = {'optimizer': 'adam', 'neurons': 64}

batch_size = 64
epochs = 20

# ================================
# 7. Outer CV: LeaveOneGroupOut ตาม cell (leave-one-cell-out บน 10 cells ที่สุ่มมา)
# ================================
n_cells = groups.nunique()
print(f"\nจำนวน cell ทั้งหมด: {n_cells}")
outer_gkf = LeaveOneGroupOut()
n_outer_splits = n_cells

fold_results = []
all_test_preds = []
all_test_true = []
dnn_best_model = None

for fold, (train_idx, test_idx) in enumerate(outer_gkf.split(X_raw, y_raw, groups=groups), 1):
    held_out_cell = groups.iloc[test_idx].iloc[0]
    print(f"\n===== Outer Fold {fold}/{n_outer_splits} (held-out cell: {held_out_cell}) =====")
    start_time = time.time()
    tracemalloc.start()

    X_train_df = X_raw.iloc[train_idx].reset_index(drop=True)
    X_test_df = X_raw.iloc[test_idx].reset_index(drop=True)
    y_train_full = y_raw.iloc[train_idx].reset_index(drop=True)
    y_test_full = y_raw.iloc[test_idx].reset_index(drop=True)
    groups_train_full = groups.iloc[train_idx].reset_index(drop=True)

    # ---- 7.1 แทนค่าติดลบด้วย NaN (ไม่ต้อง fit ใช้กับ train/test ตรงๆ) ----
    X_train_df = replace_negative_with_nan(X_train_df, features)
    X_test_df = replace_negative_with_nan(X_test_df, features)

    # ---- 7.2 Outlier bounds: fit บน train fold เท่านั้น แล้ว apply กับทั้ง train/test ----
    outlier_bounds = fit_outlier_bounds(X_train_df)
    X_train_df = apply_outlier_bounds(X_train_df, outlier_bounds)
    X_test_df = apply_outlier_bounds(X_test_df, outlier_bounds)

    # ---- 7.3 KMeans imputation: fit บน train fold เท่านั้น ----
    X_train_imputed, km_model, km_scaler, cluster_means, global_means = kmeans_impute_fit(X_train_df)
    X_test_imputed = kmeans_impute_transform(X_test_df, km_model, km_scaler, cluster_means, global_means)

    X_train_imputed = X_train_imputed.fillna(0)
    X_test_imputed = X_test_imputed.fillna(0)

    # ---- 7.3b Monotonic signs: Pearson ต่อ cell + voting (training cells เท่านั้น!) ----
    #      ส่งเฉพาะ X_train_imputed / y_train_full / groups_train_full -> ไม่มี test data เข้าไปเลย
    sign_info = vote_monotonic_signs(
        X_train_imputed, y_train_full, groups_train_full, mono_feature_names
    )
    mono_signs = [sign_info[feat]['sign'] for feat in mono_feature_names]
    for feat in mono_feature_names:
        si = sign_info[feat]
        print(f"Fold {fold} monotonic sign [{feat}]: {'+' if si['sign'] > 0 else '-'}1 "
              f"(votes +:{si['n_pos']} / -:{si['n_neg']} / skipped:{si['n_skipped']}, "
              f"pooled r={si['pooled_r']:.4f})")
    mono_signs_tf = tf.constant(mono_signs, dtype=tf.float32)

    # ---- 7.4 StandardScaler สำหรับ input X ของ DNN: fit บน train fold เท่านั้น ----
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_imputed[features])
    X_test = x_scaler.transform(X_test_imputed[features])

    # ---- 7.5 StandardScaler สำหรับ y: fit บน train fold เท่านั้น ----
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train_full.values.reshape(-1, 1)).flatten()
    y_train = pd.Series(y_train_scaled, index=y_train_full.index)

    # ---- 7.6 เทรนโมเดลด้วย fixed hyperparameter ----
    reset_seeds(SEED)  # ทุก fold เริ่มจาก seed เดียวกัน (weight init + shuffle ซ้ำได้)
    best_model_keras = build_model(
        optimizer=fixed_params['optimizer'],
        neurons=fixed_params['neurons'],
    )
    optimizer = best_model_keras.optimizer

    X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tf = tf.convert_to_tensor(y_train.values, dtype=tf.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X_train_tf, y_train_tf))
    dataset = dataset.shuffle(buffer_size=1024, seed=SEED,
                              reshuffle_each_iteration=True).batch(batch_size)

    for epoch in range(epochs):
        for X_b, y_b in dataset:
            loss, mae_b, mono = train_step(best_model_keras, optimizer, X_b, y_b, mono_signs_tf)

    best_model = best_model_keras
    best_params = fixed_params
    # --- End Training ---

    # ================================
    # 7.7 คำนวณ Metrics หลัง inverse scaling
    # ================================
    y_train_pred_scaled = best_model.predict(X_train, verbose=0).reshape(-1, 1)
    y_test_pred_scaled = best_model.predict(X_test, verbose=0).reshape(-1, 1)

    y_train_true = y_train_full.values
    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).flatten()

    y_test_true = y_test_full.values
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled).flatten()

    train_mae = mean_absolute_error(y_train_true, y_train_pred)
    train_rmse = math.sqrt(mean_squared_error(y_train_true, y_train_pred))
    train_r2 = r2_score(y_train_true, y_train_pred)
    y_train_nonzero = y_train_true[y_train_true != 0]
    y_train_pred_nonzero = y_train_pred[y_train_true != 0]
    train_mape = np.mean(np.abs((y_train_nonzero - y_train_pred_nonzero) / y_train_nonzero)) * 100

    mae = mean_absolute_error(y_test_true, y_test_pred)
    rmse = math.sqrt(mean_squared_error(y_test_true, y_test_pred))
    r2 = r2_score(y_test_true, y_test_pred)
    y_test_nonzero = y_test_true[y_test_true != 0]
    y_test_pred_nonzero = y_test_pred[y_test_true != 0]
    mape = np.mean(np.abs((y_test_nonzero - y_test_pred_nonzero) / y_test_nonzero)) * 100

    all_test_preds.append(pd.Series(y_test_pred, index=test_idx))
    all_test_true.append(pd.Series(y_test_true, index=test_idx))

    elapsed_time = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    fold_results.append({
        'fold': fold,
        'held_out_cell': held_out_cell,
        'Train_MAE': train_mae,
        'Train_RMSE': train_rmse,
        'Train_R2': train_r2,
        'Train_MAPE': train_mape,
        'Test_MAE': mae,
        'Test_RMSE': rmse,
        'Test_R2': r2,
        'Test_MAPE': mape,
        # ---- monotonic sign ที่ได้จาก Pearson + voting (training cells เท่านั้น) ----
        'sign_QC': mono_signs[0],
        'sign_QD': mono_signs[1],
        'sign_Tmax': mono_signs[2],
        'votes_QC': f"+{sign_info['QC']['n_pos']}/-{sign_info['QC']['n_neg']}",
        'votes_QD': f"+{sign_info['QD']['n_pos']}/-{sign_info['QD']['n_neg']}",
        'votes_Tmax': f"+{sign_info['Tmax']['n_pos']}/-{sign_info['Tmax']['n_neg']}",
        'time_sec': elapsed_time,
        'memory_MB': peak / (1024 * 1024),
        'best_params': best_params,
    })

    print(f"Fold {fold} | Train: MAE={train_mae:.4f}, RMSE={train_rmse:.4f}, R2={train_r2:.4f}, MAPE={train_mape:.4f}% "
          f"| Test: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}, MAPE={mape:.4f}% "
          f"| Time={elapsed_time:.4f}s, Memory={peak/(1024*1024):.4f}MB, Best Params={best_params}")

    dnn_best_model = best_model

# ================================
# 8. สรุปผลลัพธ์ทุก fold
# ================================
fold_results_df = pd.DataFrame(fold_results)
print("\n===== Summary across folds =====")
print(fold_results_df)

print("\n===== Monotonic sign (Pearson + voting) ข้ามทุก fold =====")
for feat in mono_feature_names:
    col = f"sign_{feat}"
    print(f"{feat:5s}: +1 = {(fold_results_df[col] > 0).sum()} folds, "
          f"-1 = {(fold_results_df[col] < 0).sum()} folds")

all_test_preds_df = pd.concat(all_test_preds).sort_index()
all_test_true_df = pd.concat(all_test_true).sort_index()

# ==============================
# 9. Plot predicted vs actual (บันทึกที่ 600dpi)
# ==============================
plt.figure(figsize=(8, 6))
plt.scatter(all_test_true_df.values, all_test_preds_df.values, color='red', label='Predicted')
plt.plot([all_test_true_df.min(), all_test_true_df.max()],
         [all_test_true_df.min(), all_test_true_df.max()],
         color='blue', linestyle='--', label='Ideal')
plt.xlabel('Actual RUL (Cycle)')
plt.ylabel('Predicted RUL (Cycle)')
plt.title('PI-DNN Leave-One-Cell-Out CV Predictions (Original Scale)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'pred_vs_actual.png'), dpi=600, bbox_inches='tight')
plt.show()

print(f"\n[INFO] บันทึกรูปไว้ที่: {output_dir}")