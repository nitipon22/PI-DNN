"""
Nested Leave-One-Cell-Out CV  (XGBoost / RandomForest / DNN / PI-DNN)

โครงสร้าง (nested CV ที่ถูกหลัก):
  OUTER : LeaveOneGroupOut ตาม cell  -> ใช้ประเมินผลสุดท้ายเท่านั้น
    INNER : GroupKFold ตาม cell บน outer-train เท่านั้น -> ใช้ "คัด hyperparameter" อย่างเดียว
            (preprocessing + sign voting ถูก fit ใหม่บน inner-train ของแต่ละ inner fold,
             inner-val เป็นแค่ transform + วัด MAE ไม่ถูกใช้คำนวณสถิติใดๆ)
    REFIT : preprocessing + โมเดลถูก fit ใหม่บน outer-train ทั้งหมดด้วย best hyperparameter
            แล้ววัดผลบน held-out cell (ไม่เคยถูกแตะเลย)

วิธีรัน (แนะนำตั้ง PYTHONHASHSEED):
  PYTHONHASHSEED=42 python rul_nested_cv_all_models.py                # รันทั้ง 4 โมเดล
  PYTHONHASHSEED=42 python rul_nested_cv_all_models.py xgboost rf     # เลือกบางโมเดล
"""
import os
import sys
import time
import math
import random
import tracemalloc

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ================================
# 0. Config
# ================================
SEED = 42
SUBSET_SEED = 40            # คง 40 เดิมเพื่อให้ได้ 10 cells ชุดเดียวกับ paper
N_CELLS_SUBSET = 10
INNER_SPLITS = 3            # จำนวน inner fold (ตาม cell) สำหรับคัด hyperparameter
MAT_FILENAME = '2018-02-20_batchdata_updated_struct_errorcorrect.mat'
OUTPUT_ROOT = "imgRevision/external/outputs"

FEATURES = ['IR', 'QC', 'QD', 'Tavg', 'Tmin', 'Tmax', 'chargetime']
TARGET_COL = 'RUL'
GROUP_COL = 'cell'
MONO_FEATURES = ['QC', 'QD', 'Tmax']   # ใช้เฉพาะ PI-DNN (sign มาจาก Pearson + voting)

BATCH_SIZE = 64
EPOCHS = 20

# Hyperparameter grid ต่อโมเดล (ปรับขนาดได้ตามเวลา: ยิ่งใหญ่ยิ่งช้า)
#   ค่าเริ่มต้น = ค่าคงที่เดิมของแต่ละสคริปต์ (1 config ต่อโมเดล -> ข้ามขั้น inner search)
#   ถ้าอยากคัด hyperparameter จริง ให้เพิ่มค่าใน list เช่น 'max_depth': [3, 5, 7]
GRIDS = {
    'xgboost': {'max_depth': [5], 'n_estimators': [100]},
    'rf':      {'max_depth': [5], 'n_estimators': [100]},
    'dnn':     {'optimizer': ['adam'], 'neurons': [64], 'learning_rate': [1e-3]},
    'pidnn':   {'optimizer': ['adam'], 'neurons': [64], 'learning_rate': [1e-3],
                'lambda_mono': [1e-3]},
}
MODEL_TITLES = {'xgboost': 'XGBoost', 'rf': 'Random Forest', 'dnn': 'DNN', 'pidnn': 'PI-DNN'}

selected_models = [m for m in sys.argv[1:] if m in GRIDS] or list(GRIDS.keys())
use_tf = any(m in ('dnn', 'pidnn') for m in selected_models)

os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

if use_tf:
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam, SGD
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print(f"[INFO] enable_op_determinism ใช้ไม่ได้: {e}")

if 'xgboost' in selected_models:
    from xgboost import XGBRegressor


def reset_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    if use_tf:
        keras.utils.set_random_seed(seed)


# ================================
# 1. Load .mat -> DataFrame ระดับ per-cycle
#    (โหลดเฉพาะ summary ที่ใช้จริง; ส่วน cycles รายรอบไม่ได้ถูกใช้ใน pipeline เลยตัดออกเพื่อความเร็ว)
# ================================
def load_dataframe(mat_filename):
    rows = []
    with h5py.File(mat_filename, 'r') as f:
        batch = f['batch']
        num_cells = batch['summary'].shape[0]
        for i in range(num_cells):
            cl = np.asarray(f[batch['cycle_life'][i, 0]][:]).flatten()
            cycle_life = float(cl[0]) if cl.size > 0 else np.nan
            s = f[batch['summary'][i, 0]]

            def get(name):
                return np.hstack(s[name][0, :].tolist())

            summ = {'IR': get('IR'), 'QC': get('QCharge'), 'QD': get('QDischarge'),
                    'Tavg': get('Tavg'), 'Tmin': get('Tmin'), 'Tmax': get('Tmax'),
                    'chargetime': get('chargetime'), 'cycle': get('cycle')}
            for k in range(len(summ['cycle'])):
                row = {'cell': 'b1c' + str(i),
                       'cycle_index': summ['cycle'][k],
                       'RUL': cycle_life - summ['cycle'][k]}
                for col in FEATURES:
                    row[col] = summ[col][k]
                rows.append(row)
    return pd.DataFrame(rows)


df = load_dataframe(MAT_FILENAME)

X_raw = df[FEATURES].copy()
y_numeric = pd.to_numeric(df[TARGET_COL], errors='coerce')
groups = df[GROUP_COL].copy()

valid_mask = np.isfinite(y_numeric.values)
n_dropped = int((~valid_mask).sum())
if n_dropped > 0:
    print(f"[WARNING] พบ RUL เป็น NaN/Inf {n_dropped} แถว (จาก {len(y_numeric)}) -> ตัดออกก่อนเข้า CV")
X_raw = X_raw.loc[valid_mask].reset_index(drop=True)
y_raw = y_numeric.loc[valid_mask].reset_index(drop=True)
groups = groups.loc[valid_mask].reset_index(drop=True)

# สุ่ม 10 cells (legacy RandomState(40) -> ได้ชุดเดียวกับ paper ทุกโมเดล)
all_cells = groups.unique()
if len(all_cells) > N_CELLS_SUBSET:
    selected_cells = np.random.RandomState(SUBSET_SEED).choice(all_cells, size=N_CELLS_SUBSET, replace=False)
    print(f"\n[INFO] สุ่มเลือก {N_CELLS_SUBSET} cells จาก {len(all_cells)}: {sorted(selected_cells)}")
    m = groups.isin(selected_cells)
    X_raw = X_raw.loc[m].reset_index(drop=True)
    y_raw = y_raw.loc[m].reset_index(drop=True)
    groups = groups.loc[m].reset_index(drop=True)
else:
    print(f"\n[INFO] cells ({len(all_cells)}) <= {N_CELLS_SUBSET} ใช้ทั้งหมด")

feature_names = list(X_raw.columns)
idx_of = {n: feature_names.index(n) for n in feature_names}
mono_idx = [idx_of[n] for n in MONO_FEATURES]
input_shape = len(FEATURES)


# ================================
# 2. Preprocessing: fit บนชุดที่ส่งเข้ามาเท่านั้น แล้ว transform ชุดอื่น
# ================================
def replace_negative_with_nan(X_df, cols):
    X_out = X_df.copy()
    for col in cols:
        if col in X_out.columns:
            X_out.loc[X_out[col] < 0, col] = np.nan
    return X_out


def fit_outlier_bounds(X_df):
    bounds = {}
    for col in X_df.select_dtypes(include=[np.number]).columns:
        Q1, Q3 = X_df[col].quantile(0.25), X_df[col].quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 2 * IQR, Q3 + 2 * IQR)
    return bounds


def apply_outlier_bounds(X_df, bounds):
    X_out = X_df.copy()
    for col, (lo, hi) in bounds.items():
        if col in X_out.columns:
            X_out.loc[(X_out[col] < lo) | (X_out[col] > hi), col] = np.nan
    return X_out


def kmeans_impute_fit(X_df, n_clusters=5, random_state=SEED):
    X_filled = X_df.fillna(X_df.mean())
    scaler_km = StandardScaler()
    X_scaled = scaler_km.fit_transform(X_filled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    X_imp = X_df.copy()
    cluster_means = {}
    global_means = X_df.mean()
    for col in X_df.columns:
        col_means = {}
        for cid in np.unique(clusters):
            mask = clusters == cid
            c_mean = X_df.loc[mask, col].mean()
            if np.isnan(c_mean):
                c_mean = global_means[col]
            col_means[cid] = c_mean
            X_imp.loc[mask & X_df[col].isna(), col] = c_mean
        cluster_means[col] = col_means
    return X_imp, kmeans, scaler_km, cluster_means, global_means


def kmeans_impute_transform(X_df, kmeans, scaler_km, cluster_means, global_means):
    X_scaled = scaler_km.transform(X_df.fillna(global_means))
    pred_clusters = kmeans.predict(X_scaled)
    X_imp = X_df.copy()
    for col in X_df.columns:
        for cid in np.unique(pred_clusters):
            mask = pred_clusters == cid
            fill_value = cluster_means[col].get(cid, global_means[col])
            X_imp.loc[mask & X_df[col].isna(), col] = fill_value
        X_imp[col] = X_imp[col].fillna(global_means[col])
    return X_imp


def vote_monotonic_signs(X_df, y, groups_, feats, min_abs_r=0.0):
    """Pearson ต่อ cell + majority voting (ใช้เฉพาะข้อมูลที่ส่งเข้ามา = ชุด fit)"""
    y_arr = np.asarray(y, dtype=float)
    g_arr = np.asarray(groups_)
    result = {}
    for feat in feats:
        x_arr = np.asarray(X_df[feat], dtype=float)
        n_pos = n_neg = n_skip = 0
        for g in np.unique(g_arr):
            msk = g_arr == g
            xg, yg = x_arr[msk], y_arr[msk]
            if len(xg) < 3 or np.std(xg) == 0 or np.std(yg) == 0:
                n_skip += 1
                continue
            r = np.corrcoef(xg, yg)[0, 1]
            if not np.isfinite(r) or abs(r) < min_abs_r:
                n_skip += 1
            elif r > 0:
                n_pos += 1
            else:
                n_neg += 1
        pooled_r = float(np.corrcoef(x_arr, y_arr)[0, 1]) if np.std(x_arr) > 0 and np.std(y_arr) > 0 else 0.0
        if n_pos > n_neg:
            sign = 1.0
        elif n_neg > n_pos:
            sign = -1.0
        else:
            sign = 1.0 if pooled_r >= 0 else -1.0
        result[feat] = {'sign': sign, 'n_pos': n_pos, 'n_neg': n_neg,
                        'n_skipped': n_skip, 'pooled_r': pooled_r}
    return result


def fit_preprocessor(X_fit_raw, y_fit, groups_fit, need_signs=False):
    X_fit_raw = X_fit_raw.reset_index(drop=True)
    y_fit = pd.Series(np.asarray(y_fit, dtype=float)).reset_index(drop=True)
    groups_fit = pd.Series(np.asarray(groups_fit)).reset_index(drop=True)

    X_neg = replace_negative_with_nan(X_fit_raw, FEATURES)
    bounds = fit_outlier_bounds(X_neg)
    X_clean = apply_outlier_bounds(X_neg, bounds)
    X_imp, km, km_scaler, cluster_means, global_means = kmeans_impute_fit(X_clean)
    X_imp = X_imp.fillna(0)

    sign_info = vote_monotonic_signs(X_imp, y_fit, groups_fit, MONO_FEATURES) if need_signs else None
    x_scaler = StandardScaler().fit(X_imp[FEATURES])
    return {'bounds': bounds, 'km': km, 'km_scaler': km_scaler,
            'cluster_means': cluster_means, 'global_means': global_means,
            'x_scaler': x_scaler, 'sign_info': sign_info,
            'X_fit_scaled': x_scaler.transform(X_imp[FEATURES])}


def transform_with_preprocessor(state, X_new_raw):
    X_df = replace_negative_with_nan(X_new_raw.reset_index(drop=True), FEATURES)
    X_df = apply_outlier_bounds(X_df, state['bounds'])
    X_imp = kmeans_impute_transform(X_df, state['km'], state['km_scaler'],
                                    state['cluster_means'], state['global_means']).fillna(0)
    return state['x_scaler'].transform(X_imp[FEATURES])


# ================================
# 3. Models: fit_model(...) -> predict_fn (คืนค่าเป็น scale เดิมของ RUL เสมอ)
# ================================
def build_nn(neurons):
    model = keras.Sequential([
        layers.Input(shape=(input_shape,)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1),
    ])
    return model


def make_optimizer(name, lr):
    return SGD(learning_rate=lr) if str(name).lower() == 'sgd' else Adam(learning_rate=lr)


def nn_train_step(model, optimizer, X_b, y_b, mono_signs=None, lambda_mono=0.0):
    """mono_signs=None -> plain DNN (MAE); ไม่ None -> PI-DNN (MAE + sign-aware monotonic penalty)"""
    if len(X_b.shape) == 1:
        X_b = tf.expand_dims(X_b, axis=0)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(X_b)
            preds = tf.squeeze(model(X_b, training=True), axis=1)
        mae = tf.reduce_mean(tf.abs(y_b - preds))
        loss = mae
        if mono_signs is not None:
            grads_in = inner_tape.gradient(preds, X_b)
            if grads_in is not None:
                mono = tf.constant(0.0, dtype=tf.float32)
                for k, col_idx in enumerate(mono_idx):
                    mono += tf.reduce_mean(tf.nn.relu(-mono_signs[k] * grads_in[:, col_idx]))
                loss = mae + lambda_mono * mono

    grads = outer_tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def fit_model(name, params, X_fit, y_fit, mono_signs=None):
    if name == 'xgboost':
        mdl = XGBRegressor(max_depth=params['max_depth'], n_estimators=params['n_estimators'],
                           random_state=SEED, objective='reg:squarederror', n_jobs=-1)
        mdl.fit(X_fit, y_fit)
        return lambda Xn: mdl.predict(Xn)

    if name == 'rf':
        mdl = RandomForestRegressor(max_depth=params['max_depth'], n_estimators=params['n_estimators'],
                                    random_state=SEED, n_jobs=-1)
        mdl.fit(X_fit, y_fit)
        return lambda Xn: mdl.predict(Xn)

    # ---- DNN / PI-DNN ----
    reset_seeds(SEED)
    y_scaler = StandardScaler().fit(np.asarray(y_fit, dtype=float).reshape(-1, 1))
    y_scaled = y_scaler.transform(np.asarray(y_fit, dtype=float).reshape(-1, 1)).flatten()

    model = build_nn(params['neurons'])
    optimizer = make_optimizer(params['optimizer'], params['learning_rate'])
    signs_tf = tf.constant(mono_signs, dtype=tf.float32) if name == 'pidnn' else None
    lam = params.get('lambda_mono', 0.0)

    ds = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(X_fit, dtype=tf.float32), tf.convert_to_tensor(y_scaled, dtype=tf.float32)))
    ds = ds.shuffle(buffer_size=1024, seed=SEED, reshuffle_each_iteration=True).batch(BATCH_SIZE)
    for _ in range(EPOCHS):
        for X_b, y_b in ds:
            nn_train_step(model, optimizer, X_b, y_b, signs_tf, lam)

    def predict(Xn):
        p = model.predict(Xn, verbose=0).reshape(-1, 1)
        return y_scaler.inverse_transform(p).flatten()
    return predict


def signs_from_state(name, state):
    if name != 'pidnn':
        return None
    return [state['sign_info'][f]['sign'] for f in MONO_FEATURES]


# ================================
# 4. INNER: คัด hyperparameter (val ใช้ตัดสินใจอย่างเดียว ไม่ถูกใช้ fit อะไรทั้งสิ้น)
# ================================
def select_hyperparams(name, X_tr_raw, y_tr, g_tr):
    grid = list(ParameterGrid(GRIDS[name]))
    if len(grid) == 1:   # ค่าคงที่ตัวเดียว ไม่มีอะไรให้เลือก -> ข้าม inner ไปเลย
        return grid[0], float('nan')
    X_tr_raw = X_tr_raw.reset_index(drop=True)
    y_tr = pd.Series(np.asarray(y_tr, dtype=float)).reset_index(drop=True)
    g_tr = pd.Series(np.asarray(g_tr)).reset_index(drop=True)

    inner = GroupKFold(n_splits=min(INNER_SPLITS, g_tr.nunique()))
    scores = np.zeros(len(grid))
    n_inner = 0
    for itr, iva in inner.split(X_tr_raw, y_tr, groups=g_tr):
        # preprocessing fit บน inner-train เท่านั้น (ใช้ร่วมกันทุก config ใน fold นี้)
        st = fit_preprocessor(X_tr_raw.iloc[itr], y_tr.iloc[itr], g_tr.iloc[itr], need_signs=(name == 'pidnn'))
        X_in = st['X_fit_scaled']
        X_val = transform_with_preprocessor(st, X_tr_raw.iloc[iva])   # transform only
        y_in, y_val = y_tr.iloc[itr].values, y_tr.iloc[iva].values
        signs = signs_from_state(name, st)
        for k, p in enumerate(grid):
            pred = fit_model(name, p, X_in, y_in, signs)(X_val)
            scores[k] += mean_absolute_error(y_val, pred)
        n_inner += 1

    scores /= max(n_inner, 1)
    best_k = int(np.argmin(scores))
    return grid[best_k], float(scores[best_k])


# ================================
# 5. Metrics helper
# ================================
def calc_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    nz = y_true != 0
    mape = np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100
    return mae, rmse, r2, mape


# ================================
# 6. OUTER: Leave-One-Cell-Out + inner search + refit
# ================================
def run_nested_cv(name):
    title = MODEL_TITLES[name]
    out_dir = os.path.join(OUTPUT_ROOT, {'xgboost': 'xgboost', 'rf': 'randomforest',
                                         'dnn': 'dnn', 'pidnn': 'pi-dnn'}[name])
    os.makedirs(out_dir, exist_ok=True)

    n_cells = groups.nunique()
    print(f"\n{'=' * 70}\n[{title}] nested CV | outer folds = {n_cells} | "
          f"grid size = {len(list(ParameterGrid(GRIDS[name])))} | inner splits = {INNER_SPLITS}\n{'=' * 70}")

    fold_results, all_pred, all_true = [], [], []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(X_raw, y_raw, groups=groups), 1):
        held_out = groups.iloc[test_idx].iloc[0]
        print(f"\n--- [{title}] Outer Fold {fold}/{n_cells} (held-out: {held_out}) ---")
        t0 = time.time()
        tracemalloc.start()

        X_tr = X_raw.iloc[train_idx].reset_index(drop=True)
        X_te = X_raw.iloc[test_idx].reset_index(drop=True)
        y_tr = y_raw.iloc[train_idx].reset_index(drop=True)
        y_te = y_raw.iloc[test_idx].reset_index(drop=True)
        g_tr = groups.iloc[train_idx].reset_index(drop=True)

        # (a) INNER: เลือก hyperparameter จาก outer-train เท่านั้น
        best_params, inner_mae = select_hyperparams(name, X_tr, y_tr, g_tr)
        print(f"[{title}] fold {fold} best params = {best_params} (inner val MAE = {inner_mae:.4f})")

        # (b) REFIT: preprocessing + model บน outer-train ทั้งหมด
        state = fit_preprocessor(X_tr, y_tr, g_tr, need_signs=(name == 'pidnn'))
        X_train = state['X_fit_scaled']
        X_test = transform_with_preprocessor(state, X_te)
        signs = signs_from_state(name, state)
        predict = fit_model(name, best_params, X_train, y_tr.values, signs)

        # (c) ประเมินผลบน held-out cell
        y_train_pred, y_test_pred = predict(X_train), predict(X_test)
        tr_mae, tr_rmse, tr_r2, tr_mape = calc_metrics(y_tr.values, y_train_pred)
        te_mae, te_rmse, te_r2, te_mape = calc_metrics(y_te.values, y_test_pred)

        all_pred.append(pd.Series(y_test_pred, index=test_idx))
        all_true.append(pd.Series(y_te.values, index=test_idx))

        elapsed = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rec = {'fold': fold, 'held_out_cell': held_out, 'inner_val_MAE': inner_mae,
               'Train_MAE': tr_mae, 'Train_RMSE': tr_rmse, 'Train_R2': tr_r2, 'Train_MAPE': tr_mape,
               'Test_MAE': te_mae, 'Test_RMSE': te_rmse, 'Test_R2': te_r2, 'Test_MAPE': te_mape,
               'time_sec': elapsed, 'memory_MB': peak / (1024 * 1024), 'best_params': best_params}
        if name == 'pidnn':
            for feat, s in zip(MONO_FEATURES, signs):
                si = state['sign_info'][feat]
                rec[f'sign_{feat}'] = s
                rec[f'votes_{feat}'] = f"+{si['n_pos']}/-{si['n_neg']}"
                print(f"[{title}] fold {fold} sign[{feat}] = {'+' if s > 0 else '-'}1 "
                      f"(votes +:{si['n_pos']} / -:{si['n_neg']} / skipped:{si['n_skipped']}, "
                      f"pooled r={si['pooled_r']:.4f})")
        fold_results.append(rec)

        print(f"[{title}] Fold {fold} | Train: MAE={tr_mae:.4f}, RMSE={tr_rmse:.4f}, R2={tr_r2:.4f}, "
              f"MAPE={tr_mape:.4f}% | Test: MAE={te_mae:.4f}, RMSE={te_rmse:.4f}, R2={te_r2:.4f}, "
              f"MAPE={te_mape:.4f}% | Time={elapsed:.2f}s, Mem={peak / (1024 * 1024):.2f}MB")

    res_df = pd.DataFrame(fold_results)
    print(f"\n===== [{title}] Summary across folds =====")
    print(res_df)
    res_df.to_csv(os.path.join(out_dir, 'fold_results.csv'), index=False)

    pred_s = pd.concat(all_pred).sort_index()
    true_s = pd.concat(all_true).sort_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(true_s.values, pred_s.values, color='red', label='Predicted')
    plt.plot([true_s.min(), true_s.max()], [true_s.min(), true_s.max()],
             color='blue', linestyle='--', label='Ideal')
    plt.xlabel('Actual RUL (Cycle)')
    plt.ylabel('Predicted RUL (Cycle)')
    plt.title(f'{title} Nested Leave-One-Cell-Out CV Predictions (Original Scale)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'pred_vs_actual.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"[INFO] [{title}] บันทึกผลไว้ที่: {out_dir}")

    return res_df


# ================================
# 7. Run + ตารางเปรียบเทียบทุกโมเดล
# ================================
if __name__ == '__main__':
    summaries = []
    for model_name in selected_models:
        reset_seeds(SEED)
        res = run_nested_cv(model_name)
        summaries.append({
            'model': MODEL_TITLES[model_name],
            'Test_MAE_mean': res['Test_MAE'].mean(), 'Test_MAE_std': res['Test_MAE'].std(),
            'Test_RMSE_mean': res['Test_RMSE'].mean(), 'Test_R2_mean': res['Test_R2'].mean(),
            'Test_MAPE_mean': res['Test_MAPE'].mean(),
            'time_sec_mean': res['time_sec'].mean(), 'memory_MB_mean': res['memory_MB'].mean(),
        })

    cmp_df = pd.DataFrame(summaries)
    print("\n===== Model comparison (mean over outer folds) =====")
    print(cmp_df.to_string(index=False))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    cmp_df.to_csv(os.path.join(OUTPUT_ROOT, 'model_comparison.csv'), index=False)