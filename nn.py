# =====================================================================
# Battery RUL - รวมทุกโมเดลในสคริปต์เดียว (merged version)
#   LSTM | BiLSTM | GRU | DNN | PI-DNN | CNN-LSTM | CNN-XGBoost | Attention-LSTM
#
# Pipeline หลัก = สคริปต์รวมทุกโมเดล (ไฟล์ที่ 1)
#   Outer  : Leave-One-Battery-Out (test battery ไม่ถูกแตะจนถึงตอนประเมินผล)
#   Inner  : แบ่ง "ข้อมูลดิบ" ของ outer-train เป็น inner-train / inner-val ก่อน
#            preprocessing ทั้งหมด (outlier bounds -> KMeans impute -> scaler X)
#            fit บน inner-train เท่านั้น; inner-val แค่ transform ใช้เลือก hyperparameter
#   Final  : preprocessing ชุดใหม่ fit บน outer-train ทั้งก้อน -> retrain -> test
#
# ส่วนที่ย้ายมาจากไฟล์ PI-DNN / BiLSTM แยก:
#   - PI-DNN: guard กรณี gradient เป็น None / batch 1 มิติ, เก็บ votes ต่อ fold,
#             n_implausible, mean_adj_grad, Summary across folds, รายงานแยกแต่ละ fold,
#             สรุปความเสถียรของเครื่องหมาย monotonic ข้าม fold
#   - BiLSTM: SHAP squeeze รูปทรง 3D/4D อย่างปลอดภัย
#   - ทั้งสอง: shap_dependence_plot.png (Max. Voltage Dischar.) + คำแนะนำเมื่อ fold diverge
# =====================================================================
import os

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import random
import time
import math
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import TerminateOnNaN

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ================================
# CONFIG
# ================================
DATA_PATH = "batteryNew/Battery_RUL_with_ID.csv"
OUTPUT_ROOT = "imgRevision/outputs/nn"      # แต่ละโมเดลได้โฟลเดอร์ย่อยของตัวเอง
DPI = 600
SHOW_PLOTS = False                       # True = plt.show() ด้วย (ใน notebook)
MODELS_TO_RUN = ["LSTM", "BiLSTM", "GRU", "DNN", "PI-DNN", "CNN-LSTM", "CNN-XGBoost", "Attention-LSTM"]

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


set_seed()
try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("[WARNING] enable_op_determinism ไม่สำเร็จ:", e)

# ================================
# 1. Load data
# ================================
df = pd.read_csv(DATA_PATH)
print("First 5 records:\n", df.head())

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

X_raw = df[features].copy()
y = df[target_col].copy()
groups = df[group_col].copy()
N_FEATURES = len(features)

cols_must_be_positive = [c for c in
                         ['Discharge Time (s)', 'Charging time (s)', 'Total time (s)']
                         if c in X_raw.columns]


# ================================
# 2. Preprocessing (fit เฉพาะข้อมูลที่ส่งเข้า fit_transform เท่านั้น)
# ================================
# --- PI-DNN: feature ที่ใช้ใน monotonicity penalty (ทิศทางหาจาก Pearson+voting บน train เท่านั้น) ---
mono_feature_names = ['Discharge Time (s)', 'Decrement 3.6-3.4V (s)']
idx_DT = features.index('Discharge Time (s)')
idx_DEC = features.index('Decrement 3.6-3.4V (s)')


def vote_monotonic_signs(X_train_df, y_train, groups_train, feature_names, min_abs_r=0.0):
    """Pearson ต่อ battery -> โหวต (+/-) -> เสียงข้างมาก (เสมอ: ใช้ pooled r ตัดสิน)
    ใช้เฉพาะข้อมูลที่ส่งเข้ามา (ต้องเป็นข้อมูล train เท่านั้น)"""
    y_arr = np.asarray(y_train, dtype=float)
    g_arr = np.asarray(groups_train)
    result = {}
    for feat in feature_names:
        x_arr = np.asarray(X_train_df[feat], dtype=float)
        n_pos = n_neg = n_skipped = 0
        for g in np.unique(g_arr):
            m = g_arr == g
            xg, yg = x_arr[m], y_arr[m]
            if len(xg) < 3 or np.std(xg) == 0 or np.std(yg) == 0:
                n_skipped += 1
                continue
            r = np.corrcoef(xg, yg)[0, 1]
            if not np.isfinite(r) or abs(r) < min_abs_r:
                n_skipped += 1
            elif r > 0:
                n_pos += 1
            else:
                n_neg += 1
        pooled_r = (float(np.corrcoef(x_arr, y_arr)[0, 1])
                    if np.std(x_arr) > 0 and np.std(y_arr) > 0 else 0.0)
        if n_pos > n_neg:
            sign = 1.0
        elif n_neg > n_pos:
            sign = -1.0
        else:
            sign = 1.0 if pooled_r >= 0 else -1.0
        result[feat] = dict(sign=sign, n_pos=n_pos, n_neg=n_neg,
                            n_skipped=n_skipped, pooled_r=pooled_r)
    return result


def fit_outlier_bounds(X_train_df):
    bounds = {}
    for col in X_train_df.select_dtypes(include=[np.number]).columns:
        s = X_train_df[col]
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 2.0 * IQR, Q3 + 2.0 * IQR)
    return bounds


def apply_outlier_bounds(X_df, bounds, positive_cols):
    X_out = X_df.copy()
    for col in positive_cols:
        if col in X_out.columns:
            X_out.loc[X_out[col] < 0, col] = np.nan
    for col, (lo, hi) in bounds.items():
        if col in X_out.columns:
            X_out.loc[(X_out[col] < lo) | (X_out[col] > hi), col] = np.nan
    return X_out


def kmeans_impute_fit(X_train_df, n_clusters=5, random_state=SEED):
    X_filled = X_train_df.fillna(X_train_df.mean())
    scaler_km = StandardScaler()
    X_scaled_km = scaler_km.fit_transform(X_filled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    train_clusters = kmeans.fit_predict(X_scaled_km)

    X_imputed = X_train_df.copy()
    cluster_means = {}
    global_means = X_train_df.mean()

    for col in X_train_df.columns:
        col_means = {}
        for cid in np.unique(train_clusters):
            mask = train_clusters == cid
            c_mean = X_train_df.loc[mask, col].mean()
            if np.isnan(c_mean):
                c_mean = global_means[col]
            col_means[cid] = c_mean
            fill_mask = mask & X_train_df[col].isna().values
            X_imputed.loc[fill_mask, col] = c_mean
        cluster_means[col] = col_means
    return X_imputed, kmeans, scaler_km, cluster_means, global_means


def kmeans_impute_transform(X_df, kmeans, scaler_km, cluster_means, global_means):
    X_tmp = X_df.fillna(global_means)
    pred_clusters = kmeans.predict(scaler_km.transform(X_tmp))

    X_imputed = X_df.copy()
    for col in X_df.columns:
        for cid in np.unique(pred_clusters):
            mask = pred_clusters == cid
            fill_value = cluster_means[col].get(cid, global_means[col])
            fill_mask = mask & X_df[col].isna().values
            X_imputed.loc[fill_mask, col] = fill_value
        X_imputed[col] = X_imputed[col].fillna(global_means[col])
    return X_imputed


class Preprocessor:
    """outlier mask -> KMeans impute -> StandardScaler
    fit_transform() เรียนรู้ทุกอย่างจากข้อมูลที่ให้มา, transform() ใช้ค่าที่เรียนรู้แล้วเท่านั้น"""

    def fit_transform(self, X_df, y=None, groups_=None):
        X_df = X_df.reset_index(drop=True)
        self.bounds = fit_outlier_bounds(X_df)
        X_m = apply_outlier_bounds(X_df, self.bounds, cols_must_be_positive)
        (X_imp, self.km, self.km_scaler,
         self.cluster_means, self.global_means) = kmeans_impute_fit(X_m)
        # sign voting สำหรับ PI-DNN: ใช้เฉพาะข้อมูลที่ fit อยู่นี้
        self.mono_info, self.mono_signs = None, None
        if y is not None and groups_ is not None:
            self.mono_info = vote_monotonic_signs(X_imp, y, groups_, mono_feature_names)
            self.mono_signs = [self.mono_info[f]['sign'] for f in mono_feature_names]
        self.scaler = StandardScaler()
        return self.scaler.fit_transform(X_imp)

    def transform(self, X_df):
        X_df = X_df.reset_index(drop=True)
        X_m = apply_outlier_bounds(X_df, self.bounds, cols_must_be_positive)
        X_imp = kmeans_impute_transform(X_m, self.km, self.km_scaler,
                                        self.cluster_means, self.global_means)
        return self.scaler.transform(X_imp)


# ================================
# 3. สร้าง fold ทั้งหมดครั้งเดียว (ทุกโมเดลใช้ split/preprocess ชุดเดียวกัน)
# ================================
def build_fold_cache():
    folds = []
    outer = LeaveOneGroupOut()
    n_batt = groups.nunique()
    print(f"\nจำนวน Battery_ID ทั้งหมด: {n_batt}")

    for fold, (tr_idx, te_idx) in enumerate(outer.split(X_raw, y, groups=groups), 1):
        t0 = time.time()
        X_tr_raw = X_raw.iloc[tr_idx].reset_index(drop=True)
        X_te_raw = X_raw.iloc[te_idx].reset_index(drop=True)
        y_tr = y.iloc[tr_idx].reset_index(drop=True).values
        y_te = y.iloc[te_idx].reset_index(drop=True).values
        g_tr = groups.iloc[tr_idx].reset_index(drop=True)

        # ---- (A) Final pipeline: fit บน outer-train ทั้งก้อน ----
        prep_outer = Preprocessor()
        X_tr_s = prep_outer.fit_transform(X_tr_raw, y_tr, g_tr)
        X_te_s = prep_outer.transform(X_te_raw)

        # ---- (B) Inner pipeline: split ข้อมูล "ดิบ" ก่อน แล้ว fit บน inner-train เท่านั้น ----
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        i_tr, i_val = next(gss.split(X_tr_raw, y_tr, groups=g_tr))

        X_in_tr_raw = X_tr_raw.iloc[i_tr]
        X_in_val_raw = X_tr_raw.iloc[i_val]

        prep_inner = Preprocessor()
        X_in_tr_s = prep_inner.fit_transform(X_in_tr_raw, y_tr[i_tr], g_tr.iloc[i_tr])  # fit บน inner-train
        X_in_val_s = prep_inner.transform(X_in_val_raw)     # val แค่ transform

        folds.append(dict(
            fold=fold, test_idx=te_idx,
            X_train=X_tr_s, y_train=y_tr, X_test=X_te_s, y_test=y_te,
            X_in_tr=X_in_tr_s, y_in_tr=y_tr[i_tr],
            X_in_val=X_in_val_s, y_in_val=y_tr[i_val],
            mono_train=prep_outer.mono_signs, mono_info_train=prep_outer.mono_info,
            mono_in=prep_inner.mono_signs,
            prep_time=time.time() - t0,
        ))
        print(f"  [cache] fold {fold}/{n_batt} preprocessed ({time.time() - t0:.1f}s)")
    return folds


# ================================
# 4. Model builders
# ================================
def to_sequence(X_2d):
    X = np.asarray(X_2d)
    return X.reshape((X.shape[0], 1, X.shape[1]))


def make_optimizer(name, clip=False):
    kw = {'clipnorm': 1.0} if clip else {}
    if str(name).lower() == 'sgd':
        return SGD(learning_rate=LEARNING_RATE, momentum=0.9 if clip else 0.0, **kw)
    return Adam(learning_rate=LEARNING_RATE, **kw)


def _compile(model, optimizer, clip=False, loss='mse'):
    model.compile(optimizer=make_optimizer(optimizer, clip), loss=loss, metrics=['mae'])
    return model


def _reset():
    keras.backend.clear_session()
    set_seed(SEED)


def build_lstm(n_features, optimizer, neurons):
    _reset()
    m = keras.Sequential([
        layers.Input(shape=(1, n_features)),
        layers.LSTM(neurons, return_sequences=False),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1),
    ])
    return _compile(m, optimizer)


def build_gru(n_features, optimizer, neurons):
    _reset()
    m = keras.Sequential([
        layers.Input(shape=(1, n_features)),
        layers.GRU(neurons, return_sequences=False),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1),
    ])
    return _compile(m, optimizer)


def build_dnn(n_features, optimizer, neurons):
    _reset()
    m = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1),
    ])
    return _compile(m, optimizer)


def build_cnn_lstm(n_features, optimizer, neurons):
    _reset()
    inp = Input(shape=(1, n_features))
    x = layers.Permute((2, 1))(inp)                              # (f, 1)
    x = layers.Conv1D(32, kernel_size=1, activation='relu')(x)
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(x)   # (f, 64)
    x = layers.Permute((2, 1))(x)                                # (64, f)
    x = layers.Reshape((1, -1))(x)                               # (1, 64*f)
    x = layers.LSTM(neurons, return_sequences=False)(x)
    x = layers.Dense(neurons, activation='relu')(x)
    out = layers.Dense(1)(x)
    return _compile(Model(inp, out), optimizer)


def build_attention_lstm(n_features, optimizer, neurons):
    _reset()
    inp = Input(shape=(1, n_features))
    lstm_out = layers.LSTM(neurons, return_sequences=True)(inp)

    att = layers.Dense(1, activation='tanh')(lstm_out)
    att = layers.Flatten()(att)
    att = layers.Activation('softmax')(att)
    att = layers.RepeatVector(neurons)(att)
    att = layers.Permute([2, 1])(att)

    attended = layers.Multiply()([lstm_out, att])
    vec = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(attended)
    x = layers.Dense(neurons, activation='relu')(vec)
    out = layers.Dense(1)(x)
    return _compile(Model(inp, out), optimizer, clip=True)   # clipnorm + momentum(SGD)


def build_bilstm(n_features, optimizer, neurons):
    _reset()
    m = keras.Sequential([
        layers.Input(shape=(1, n_features)),
        layers.Bidirectional(layers.LSTM(neurons, return_sequences=False)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1),
    ])
    return _compile(m, optimizer)


def build_pi_dnn(n_features, optimizer, neurons):
    _reset()
    m = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(neurons, activation='relu'),
        layers.Dense(1),
    ])
    return _compile(m, optimizer, loss='mae')


def build_cnn_extractor(n_features, neurons=64):
    """CNN -> embedding (Dense 'feature_vector') -> Dense(1) (ใช้เทรนเท่านั้น)"""
    _reset()
    inp = Input(shape=(1, n_features))
    x = layers.Permute((2, 1))(inp)
    x = layers.Conv1D(32, kernel_size=1, activation='relu')(x)
    x = layers.Conv1D(64, kernel_size=1, activation='relu')(x)
    x = layers.Flatten()(x)
    feat = layers.Dense(neurons, activation='relu', name='feature_vector')(x)
    out = layers.Dense(1)(feat)
    train_model = Model(inp, out)
    train_model.compile(optimizer=Adam(), loss='mse', metrics=['mae'])
    return train_model, Model(inp, feat)


MODEL_CFG = {
    'LSTM':           dict(kind='keras', builder=build_lstm,           fmt='seq'),
    'BiLSTM':         dict(kind='keras', builder=build_bilstm,         fmt='seq'),
    'GRU':            dict(kind='keras', builder=build_gru,            fmt='seq'),
    'PI-DNN':         dict(kind='pi_dnn', builder=build_pi_dnn,        fmt='flat'),
    'DNN':            dict(kind='keras', builder=build_dnn,            fmt='flat'),
    'CNN-LSTM':       dict(kind='keras', builder=build_cnn_lstm,       fmt='seq'),
    'Attention-LSTM': dict(kind='keras', builder=build_attention_lstm, fmt='seq'),
    'CNN-XGBoost':    dict(kind='cnn_xgb'),
}

GRID_KERAS = [{'optimizer': o, 'neurons': n}
              for o in ['adam', 'sgd'] for n in [32, 64, 128]]
GRID_XGB = list(ParameterGrid({'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]}))
FALLBACK_KERAS = {'optimizer': 'adam', 'neurons': 32}
FALLBACK_XGB = {'n_estimators': 100, 'max_depth': 5}


# ================================
# 5. Predictor wrappers (คืนค่า RUL สเกลจริง + จำนวน NaN/Inf)
# ================================
class KerasPredictor:
    def __init__(self, model, y_scaler, fmt):
        self.model, self.y_scaler, self.fmt = model, y_scaler, fmt

    def model_input(self, X_2d):
        return to_sequence(X_2d) if self.fmt == 'seq' else np.asarray(X_2d)

    def predict(self, X_2d, label=""):
        raw = self.model.predict(self.model_input(X_2d), verbose=0).flatten()
        n_bad = int((~np.isfinite(raw)).sum())
        if n_bad > 0:
            print(f"[WARNING] {label}: พบ NaN/Inf {n_bad} จุด (โมเดลน่าจะ diverge)")
            raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        return self.y_scaler.inverse_transform(raw.reshape(-1, 1)).flatten(), n_bad


def embed_safe(extractor, X_2d, label=""):
    feats = extractor.predict(to_sequence(X_2d), verbose=0)
    n_bad = int((~np.isfinite(feats)).sum())
    if n_bad > 0:
        print(f"[WARNING] {label}: embedding มี NaN/Inf {n_bad} ค่า -> nan_to_num")
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, n_bad


class CnnXgbPredictor:
    def __init__(self, extractor, xgb):
        self.extractor, self.xgb = extractor, xgb

    def embed(self, X_2d, label=""):
        return embed_safe(self.extractor, X_2d, label)

    def predict(self, X_2d, label=""):
        feats, n_bad = self.embed(X_2d, label)
        return self.xgb.predict(feats), n_bad


# ---------- PI-DNN: loss = MAE + lambda * monotonicity penalty (sign-aware) ----------
LAMBDA_MONO = 0.001


def pi_train_step(model, optimizer, X_batch, y_batch, mono_signs):
    if len(X_batch.shape) == 1:                      # (จาก PI-DNN แยก) กัน batch 1 มิติ
        X_batch = tf.expand_dims(X_batch, axis=0)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(X_batch)
            preds = tf.squeeze(model(X_batch, training=True), axis=1)
        input_grads = inner_tape.gradient(preds, X_batch)
        mae = tf.reduce_mean(tf.abs(y_batch - preds))
        if input_grads is not None:                  # (จาก PI-DNN แยก) กัน gradient เป็น None
            penalty_DT = tf.reduce_mean(tf.nn.relu(-mono_signs[0] * input_grads[:, idx_DT]))
            penalty_DEC = tf.reduce_mean(tf.nn.relu(-mono_signs[1] * input_grads[:, idx_DEC]))
            mono = penalty_DT + penalty_DEC
        else:
            mono = 0.0
        loss = mae + LAMBDA_MONO * mono
    grads = outer_tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def train_physics_informed(model, X, y_scaled, mono_signs, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """คืน True ถ้า diverge (loss ไม่ finite)"""
    X_tf = tf.convert_to_tensor(X, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_scaled, dtype=tf.float32)
    signs_tf = tf.constant(mono_signs, dtype=tf.float32)
    n = int(X_tf.shape[0])
    optimizer = model.optimizer
    try:
        optimizer.build(model.trainable_variables)
    except Exception:
        pass
    rng = np.random.RandomState(SEED)
    step_fn = tf.function(pi_train_step)

    for _ in range(epochs):
        perm = tf.constant(rng.permutation(n), dtype=tf.int32)
        X_shuf, y_shuf = tf.gather(X_tf, perm), tf.gather(y_tf, perm)
        for start in range(0, n, batch_size):
            loss = step_fn(model, optimizer, X_shuf[start:start + batch_size],
                           y_shuf[start:start + batch_size], signs_tf)
            if not tf.math.is_finite(loss):
                return True
    return False


def compute_physics_informed_metrics(model, X_scaled, y_pred, mono_signs,
                                     batch_size=256, rul_lower_bound=0.0):
    """MVR (Monotonicity Violation Rate), Gradient Consistency, Physically Implausible Rate"""
    X_tf = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
    n = int(X_tf.shape[0])
    g_dt, g_dec = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for s0 in range(0, n, batch_size):
        xb = X_tf[s0:s0 + batch_size]
        with tf.GradientTape() as tape:
            tape.watch(xb)
            pr = tf.squeeze(model(xb, training=False), axis=1)
        gr = tape.gradient(pr, xb)
        gr = gr.numpy() if gr is not None else np.zeros((xb.shape[0], xb.shape[1]), np.float32)
        g_dt[s0:s0 + batch_size] = gr[:, idx_DT]
        g_dec[s0:s0 + batch_size] = gr[:, idx_DEC]
    adj_dt, adj_dec = mono_signs[0] * g_dt, mono_signs[1] * g_dec
    v_dt, v_dec = adj_dt < 0.0, adj_dec < 0.0
    implausible = np.asarray(y_pred) < rul_lower_bound
    return {
        'MVR_DischargeTime': float(np.mean(v_dt)),
        'MVR_Decrement': float(np.mean(v_dec)),
        'MVR_overall': float(np.mean(v_dt | v_dec)),
        'Gradient_Consistency': float(np.mean(~v_dt & ~v_dec)),
        'Physically_Implausible_Rate': float(np.mean(implausible)),
        'n_implausible': int(implausible.sum()),
        'n_samples': int(n),
        'mean_adj_grad_DT': float(np.mean(adj_dt)),
        'mean_adj_grad_DEC': float(np.mean(adj_dec)),
    }


class PiDnnPredictor(KerasPredictor):
    def __init__(self, model, y_scaler, mono_signs, train_diverged):
        super().__init__(model, y_scaler, 'flat')
        self.mono_signs, self.train_diverged = mono_signs, train_diverged


def fit_scaled_y(y_raw):
    ys = StandardScaler()
    return ys, ys.fit_transform(np.asarray(y_raw).reshape(-1, 1)).flatten()


def train_cnn_extractor(X_s, y_raw):
    _, y_s = fit_scaled_y(y_raw)
    train_model, extractor = build_cnn_extractor(X_s.shape[1])
    train_model.fit(to_sequence(X_s), y_s, epochs=EPOCHS, batch_size=BATCH_SIZE,
                    verbose=0, callbacks=[TerminateOnNaN()])
    return extractor


def fit_predictor(name, params, X_s, y_raw, mono_signs=None):
    cfg = MODEL_CFG[name]
    if cfg['kind'] == 'pi_dnn':
        ys, y_s = fit_scaled_y(y_raw)
        model = cfg['builder'](X_s.shape[1], params['optimizer'], params['neurons'])
        div = train_physics_informed(model, X_s, y_s, mono_signs)
        return PiDnnPredictor(model, ys, mono_signs, div)
    if cfg['kind'] == 'keras':
        ys, y_s = fit_scaled_y(y_raw)
        model = cfg['builder'](X_s.shape[1], params['optimizer'], params['neurons'])
        Xin = to_sequence(X_s) if cfg['fmt'] == 'seq' else np.asarray(X_s)
        model.fit(Xin, y_s, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
                  callbacks=[TerminateOnNaN()])
        return KerasPredictor(model, ys, cfg['fmt'])
    # CNN -> XGBoost
    extractor = train_cnn_extractor(X_s, y_raw)
    feats, _ = embed_safe(extractor, X_s, "final train embedding")
    xgb = XGBRegressor(random_state=SEED, **params)
    xgb.fit(feats, y_raw)
    return CnnXgbPredictor(extractor, xgb)


# ================================
# 6. เลือก hyperparameter (ใช้ inner-val เพื่อเลือกอย่างเดียว)
# ================================
def select_hyperparams(name, fc):
    best_mae, best_params = float('inf'), None
    X_tr, y_tr = fc['X_in_tr'], fc['y_in_tr']
    X_val, y_val = fc['X_in_val'], fc['y_in_val']

    if MODEL_CFG[name]['kind'] == 'cnn_xgb':
        extractor = train_cnn_extractor(X_tr, y_tr)           # เทรน CNN ครั้งเดียวบน inner-train
        f_tr, _ = embed_safe(extractor, X_tr, "inner-train embedding")
        f_val, _ = embed_safe(extractor, X_val, "inner-val embedding")
        for p in GRID_XGB:
            xgb = XGBRegressor(random_state=SEED, **p).fit(f_tr, y_tr)
            mae = mean_absolute_error(y_val, xgb.predict(f_val))
            if mae < best_mae:
                best_mae, best_params = mae, p
        return (best_params or FALLBACK_XGB), best_mae

    for p in GRID_KERAS:
        pred = fit_predictor(name, p, X_tr, y_tr, fc.get('mono_in'))
        if getattr(pred, 'train_diverged', False):
            print(f"  -> ตัด {p} ทิ้ง เพราะ diverge ระหว่าง physics-informed training")
            continue
        y_hat, n_bad = pred.predict(X_val, f"Fold {fc['fold']} grid {p}")
        if n_bad > 0:
            print(f"  -> ตัด {p} ทิ้ง เพราะ diverge")
            continue
        mae = mean_absolute_error(y_val, y_hat)
        if mae < best_mae:
            best_mae, best_params = mae, p
    if best_params is None:
        print(f"[WARNING] Fold {fc['fold']}: ทุก combo diverge -> fallback {FALLBACK_KERAS}")
        best_params = FALLBACK_KERAS
    return best_params, best_mae


# ================================
# 7. Metrics / plotting helpers
# ================================
def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nz = y_true != 0
    return dict(
        MAE=mean_absolute_error(y_true, y_pred),
        RMSE=math.sqrt(mean_squared_error(y_true, y_pred)),
        R2=r2_score(y_true, y_pred),
        MAPE=np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100,
    )


def save_fig(path):
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close('all')


def safe_name(s):
    return s.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "").replace("/", "_")


# ================================
# 8. SHAP (fold สุดท้าย) -> รูปแยกโฟลเดอร์ต่อโมเดล
# ================================
def squeeze_shap(sv, n_samples, n_features):
    """(จาก BiLSTM แยก) แปลง shap values ทุกรูปทรง ((n,1,f), (n,1,f,1), list) -> (n, f)"""
    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.ndim > 2:
        axes = tuple(ax for ax in range(1, sv.ndim - 1) if sv.shape[ax] == 1)
        if axes:
            sv = np.squeeze(sv, axis=axes)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv.reshape(sv.shape[0], sv.shape[1])
    sv = sv.reshape(sv.shape[0], -1)
    assert sv.shape == (n_samples, n_features), \
        f"SHAP shape mismatch: {sv.shape} vs {(n_samples, n_features)}"
    return sv


def run_shap(name, predictor, fc_last, out_dir):
    n_test = fc_last['X_test'].shape[0]
    te2d = fc_last['X_test'][:max(n_test // 2, 1)]

    set_seed(SEED)
    if MODEL_CFG[name]['kind'] != 'cnn_xgb':
        bg2d = fc_last['X_test'][:min(100, n_test)]
        explainer = shap.GradientExplainer(predictor.model, predictor.model_input(bg2d))
        set_seed(SEED)
        sv = explainer.shap_values(predictor.model_input(te2d))
        X_plot, feat_names = te2d, features
    else:
        bg, _ = predictor.embed(fc_last['X_train'][:100], "shap background")
        X_plot, _ = predictor.embed(te2d, "shap test")
        explainer = shap.TreeExplainer(predictor.xgb, bg)
        sv = explainer.shap_values(X_plot)
        feat_names = [f"CNN_feature_{i}" for i in range(X_plot.shape[1])]

    sv = squeeze_shap(sv, X_plot.shape[0], X_plot.shape[1])

    plt.figure()
    shap.summary_plot(sv, X_plot, feature_names=feat_names, show=False)
    save_fig(os.path.join(out_dir, 'shap_summary_plot.png'))

    X_df = pd.DataFrame(X_plot, columns=feat_names)
    if MODEL_CFG[name]['kind'] != 'cnn_xgb':
        dep_features = list(enumerate(feat_names))
        # (จากไฟล์แยก) dependence plot ของ Max. Voltage Dischar. ชื่อไฟล์เดิม
        plt.figure(figsize=(6, 4))
        shap.dependence_plot("Max. Voltage Dischar. (V)", sv, X_df, show=False)
        save_fig(os.path.join(out_dir, 'shap_dependence_plot.png'))
    else:  # embedding มีหลายมิติ -> เอา top-5 ตาม mean|SHAP|
        top = np.argsort(-np.abs(sv).mean(0))[:5]
        dep_features = [(int(i), feat_names[int(i)]) for i in top]

    for i, fname in dep_features:
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(fname, sv, X_df, show=False)
        save_fig(os.path.join(out_dir, f'shap_dependence_{i}_{safe_name(fname)}.png'))

    print(f"[INFO] {name}: บันทึกรูป SHAP ไว้ที่ {out_dir}")


# ================================
# 9. รันหนึ่งโมเดล
# ================================
METRIC_COLS = ['Train_MAE', 'Train_RMSE', 'Train_R2', 'Train_MAPE',
               'Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MAPE',
               'time_sec', 'memory_MB']


def run_model(name, folds):
    print(f"\n{'#' * 70}\n# MODEL: {name}\n{'#' * 70}")
    out_dir = os.path.join(OUTPUT_ROOT, safe_name(name.lower()))
    os.makedirs(out_dir, exist_ok=True)
    set_seed(SEED)

    results, all_pred, all_true = [], [], []
    last_pred, last_fc = None, None

    for fc in folds:
        fold = fc['fold']
        print(f"\n===== [{name}] Outer Fold {fold}/{len(folds)} =====")
        t0 = time.time()
        tracemalloc.start()

        best_params, best_val = select_hyperparams(name, fc)
        print(f"Fold {fold} best params (inner val MAE={best_val:.4f}): {best_params}")

        predictor = fit_predictor(name, best_params, fc['X_train'], fc['y_train'], fc.get('mono_train'))

        y_tr_pred, bad_tr = predictor.predict(fc['X_train'], f"Fold {fold} train")
        y_te_pred, bad_te = predictor.predict(fc['X_test'], f"Fold {fold} test")
        tr = compute_metrics(fc['y_train'], y_tr_pred)
        te = compute_metrics(fc['y_test'], y_te_pred)

        extra = {}
        if isinstance(predictor, PiDnnPredictor):
            ph_tr = compute_physics_informed_metrics(predictor.model, fc['X_train'], y_tr_pred, predictor.mono_signs)
            ph_te = compute_physics_informed_metrics(predictor.model, fc['X_test'], y_te_pred, predictor.mono_signs)
            extra.update({f'Train_{k}': v for k, v in ph_tr.items()})
            extra.update({f'Test_{k}': v for k, v in ph_te.items()})
            extra['sign_DT'], extra['sign_DEC'] = predictor.mono_signs
            mi = fc['mono_info_train']
            f_dt, f_dec = mono_feature_names
            extra['votes_DT'] = f"+{mi[f_dt]['n_pos']}/-{mi[f_dt]['n_neg']}"
            extra['votes_DEC'] = f"+{mi[f_dec]['n_pos']}/-{mi[f_dec]['n_neg']}"
            print(f"Fold {fold} mono signs (outer-train): DT={extra['sign_DT']:+.0f} "
                  f"(votes {extra['votes_DT']}, skipped {mi[f_dt]['n_skipped']}, pooled r={mi[f_dt]['pooled_r']:.4f}), "
                  f"DEC={extra['sign_DEC']:+.0f} "
                  f"(votes {extra['votes_DEC']}, skipped {mi[f_dec]['n_skipped']}, pooled r={mi[f_dec]['pooled_r']:.4f})")
            print(f"Fold {fold} | Physics (Test): MVR_overall={ph_te['MVR_overall']*100:.2f}%  "
                  f"GradConsistency={ph_te['Gradient_Consistency']*100:.2f}%  "
                  f"ImplausibleRate={ph_te['Physically_Implausible_Rate']*100:.2f}% "
                  f"({ph_te['n_implausible']}/{ph_te['n_samples']} samples)")

        all_pred.append(pd.Series(y_te_pred, index=fc['test_idx']))
        all_true.append(pd.Series(fc['y_test'], index=fc['test_idx']))

        elapsed = time.time() - t0 + fc['prep_time']
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append({
            'fold': fold,
            **{f'Train_{k}': v for k, v in tr.items()},
            **{f'Test_{k}': v for k, v in te.items()},
            'time_sec': elapsed, 'memory_MB': peak / (1024 * 1024),
            'best_params': best_params,
            'diverged': (bad_tr > 0 or bad_te > 0 or getattr(predictor, 'train_diverged', False)),
            **extra,
        })
        print(f"Fold {fold} | Train: MAE={tr['MAE']:.4f}, RMSE={tr['RMSE']:.4f}, "
              f"R2={tr['R2']:.4f}, MAPE={tr['MAPE']:.4f}% "
              f"| Test: MAE={te['MAE']:.4f}, RMSE={te['RMSE']:.4f}, "
              f"R2={te['R2']:.4f}, MAPE={te['MAPE']:.4f}% "
              f"| Time={elapsed:.2f}s, Memory={peak / (1024 * 1024):.2f}MB")

        last_pred, last_fc = predictor, fc

    # ---- สรุปผล ----
    res_df = pd.DataFrame(results)
    print(f"\n===== [{name}] Summary across folds =====")
    print(res_df.to_string())

    div = res_df[res_df['diverged']]
    if len(div) > 0:
        print(f"\n[WARNING] [{name}] พบ {len(div)} fold ที่ diverge: {div['fold'].tolist()}")
        print("แนะนำ: ดู best_params ของ fold เหล่านี้ ถ้าเจอ 'sgd' บ่อย ให้พิจารณาตัด sgd ออกจาก grid "
              "หรือลด learning_rate ลงอีก")
    clean = res_df[~res_df['diverged']]
    if len(clean) < len(res_df):
        print(f"[INFO] ใช้ {len(clean)}/{len(res_df)} fold ที่ไม่ diverge ในการคำนวณสรุปสถิติ")
    if len(clean) == 0:
        clean = res_df

    stats = pd.DataFrame({
        'mean': clean[METRIC_COLS].mean(), 'std': clean[METRIC_COLS].std(),
        'median': clean[METRIC_COLS].median(),
        'min': clean[METRIC_COLS].min(), 'max': clean[METRIC_COLS].max(),
    })
    stats['mean ± std'] = stats.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    print(f"\n===== [{name}] Mean ± Std (fold ที่ไม่ diverge: {len(clean)}/{len(res_df)}) =====")
    print(stats[['mean ± std', 'median', 'min', 'max']].to_string())

    # ---- รายงานแยกแต่ละ fold ----
    print(f"\n===== [{name}] รายงานผลแยกแต่ละ Fold =====")
    for _, row in res_df.iterrows():
        print(f"\n--- Fold {int(row['fold'])} (Best Params: {row['best_params']}, "
              f"Diverged={row['diverged']}) ---")
        if 'sign_DT' in row.index:
            print(f"  Mono sign : DT={int(row['sign_DT']):+d} (votes {row['votes_DT']})  "
                  f"DEC={int(row['sign_DEC']):+d} (votes {row['votes_DEC']})")
        print(f"  Train : MAE={row['Train_MAE']:.4f}  RMSE={row['Train_RMSE']:.4f}  "
              f"R2={row['Train_R2']:.4f}  MAPE={row['Train_MAPE']:.4f}%")
        print(f"  Test  : MAE={row['Test_MAE']:.4f}  RMSE={row['Test_RMSE']:.4f}  "
              f"R2={row['Test_R2']:.4f}  MAPE={row['Test_MAPE']:.4f}%")
        if 'Test_MVR_overall' in row.index:
            print(f"  Physics (Test): MVR_overall={row['Test_MVR_overall']*100:.2f}%  "
                  f"GradConsistency={row['Test_Gradient_Consistency']*100:.2f}%  "
                  f"ImplausibleRate={row['Test_Physically_Implausible_Rate']*100:.2f}%")
        print(f"  Time  : {row['time_sec']:.2f}s   Memory: {row['memory_MB']:.2f}MB")

    res_df.assign(best_params=res_df['best_params'].astype(str)).to_csv(
        os.path.join(out_dir, 'fold_results.csv'), index=False)
    stats.to_csv(os.path.join(out_dir, 'summary_stats.csv'))

    # ---- PI-DNN: physics diagnostics ----
    if isinstance(last_pred, PiDnnPredictor):
        phys_cols = [c for c in clean.columns
                     if any(k in c for k in ('MVR', 'Gradient_Consistency', 'Physically_Implausible'))]
        phys = pd.DataFrame({'mean_%': clean[phys_cols].mean() * 100,
                             'std_%': clean[phys_cols].std() * 100,
                             'min_%': clean[phys_cols].min() * 100,
                             'max_%': clean[phys_cols].max() * 100})
        print(f"\n===== [{name}] Physics-informed diagnostics (%) =====")
        print(phys.to_string(float_format=lambda v: f"{v:.2f}"))
        print("\n===== Monotonic sign (Pearson + voting) ข้ามทุก fold =====")
        print(f"Discharge Time (s)     : +1 = {(res_df['sign_DT'] > 0).sum()} folds, "
              f"-1 = {(res_df['sign_DT'] < 0).sum()} folds")
        print(f"Decrement 3.6-3.4V (s) : +1 = {(res_df['sign_DEC'] > 0).sum()} folds, "
              f"-1 = {(res_df['sign_DEC'] < 0).sum()} folds")
        phys.to_csv(os.path.join(out_dir, 'physics_summary.csv'))

        plt.figure(figsize=(9, 5))
        plt.plot(res_df['fold'], res_df['Test_MVR_overall'] * 100, marker='o', label='Monotonicity Violation Rate (%)')
        plt.plot(res_df['fold'], res_df['Test_Gradient_Consistency'] * 100, marker='s', label='Gradient Consistency (%)')
        plt.plot(res_df['fold'], res_df['Test_Physically_Implausible_Rate'] * 100, marker='^',
                 label='Physically Implausible Rate (%)')
        plt.xlabel('Outer Fold (Battery held out)')
        plt.ylabel('%')
        plt.title('Physics-informed Diagnostics per Fold (Test set)')
        plt.legend()
        plt.grid(True)
        save_fig(os.path.join(out_dir, 'physics_informed_diagnostics_per_fold.png'))

    # ---- Predicted vs Actual ----
    pred_all = pd.concat(all_pred).sort_index()
    true_all = pd.concat(all_true).sort_index()
    plt.figure(figsize=(8, 6))
    plt.scatter(true_all.values, pred_all.values, color='red', label='Predicted')
    lim = [true_all.min(), true_all.max()]
    plt.plot(lim, lim, color='blue', linestyle='--', label='Ideal')
    plt.xlabel('Actual RUL (Cycle)')
    plt.ylabel('Predicted RUL (Cycle)')
    plt.title(f'{name} Leave-One-Battery-Out CV Predictions')
    plt.legend()
    plt.grid(True)
    save_fig(os.path.join(out_dir, 'pred_vs_actual.png'))

    # ---- SHAP ----
    run_shap(name, last_pred, last_fc, out_dir)

    return name, stats, res_df


# ================================
# 10. Main
# ================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    fold_cache = build_fold_cache()

    summary_rows = []
    for model_name in MODELS_TO_RUN:
        name, stats, res_df = run_model(model_name, fold_cache)
        row = {'model': name}
        for m in ['Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MAPE', 'time_sec']:
            row[m] = stats.loc[m, 'mean ± std']
        row['n_diverged_folds'] = int(res_df['diverged'].sum())
        summary_rows.append(row)

    comparison = pd.DataFrame(summary_rows)
    print("\n" + "=" * 70 + "\nCOMPARISON (mean ± std across LOBO folds)\n" + "=" * 70)
    print(comparison.to_string(index=False))
    comparison.to_csv(os.path.join(OUTPUT_ROOT, 'model_comparison.csv'), index=False)