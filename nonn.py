import os
import random

# --- SEED: ต้องตั้งค่า env ก่อน import tensorflow ---
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import time
import math
import tracemalloc

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import LeaveOneGroupOut, GroupShuffleSplit, ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, SGD

random.seed(SEED)
np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism()

# ================================
# CONFIG
# ================================
DATA_PATH = "batteryNew/Battery_RUL_with_ID.csv"
OUTPUT_ROOT = "imgRevision/outputs/all_models"
DPI = 600
SHOW_PLOTS = False          # True = plt.show() ทุกภาพ (จะบล็อกสคริปต์ทีละภาพ)
MODELS_TO_RUN = None        # None = รันทุกโมเดล หรือระบุ เช่น ['SVR', 'XGBoost']

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ================================
# 1. Load data
# ================================
df = pd.read_csv(DATA_PATH)
print("First 5 records:\n", df.head())

target_col = 'RUL'
group_col = 'Battery_ID'

FEATURES_BASE = [
    'Discharge Time (s)',
    'Decrement 3.6-3.4V (s)',
    'Max. Voltage Dischar. (V)',
    'Min. Voltage Charg. (V)',
    'Time at 4.15V (s)',
    'Time constant current (s)',
    'Charging time (s)',
    'Total time (s)',
]
# ตามโค้ดเดิม: OLS ตัด 'Discharge Time (s)' ออก, LightGBM ใส่ 'Cycle_Index' เพิ่ม
FEATURES_OLS = [f for f in FEATURES_BASE if f != 'Discharge Time (s)']
FEATURES_LGB = FEATURES_BASE

y_all = df[target_col].copy()
groups_all = df[group_col].copy()

cols_must_be_positive = ['Discharge Time (s)', 'Charging time (s)', 'Total time (s)']


# ================================
# 2. Preprocessing building blocks
# ================================
def fit_outlier_bounds(X_df):
    bounds = {}
    for col in X_df.select_dtypes(include=[np.number]).columns:
        s = X_df[col]
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        bounds[col] = (Q1 - 2.0 * IQR, Q3 + 2.0 * IQR)
    return bounds


def apply_outlier_bounds(X_df, bounds, positive_cols):
    X_out = X_df.copy()
    for col in positive_cols:
        if col in X_out.columns:
            X_out.loc[X_out[col] < 0, col] = np.nan
    for col, (lower, upper) in bounds.items():
        if col in X_out.columns:
            X_out.loc[(X_out[col] < lower) | (X_out[col] > upper), col] = np.nan
    return X_out


def kmeans_impute_fit(X_df, n_clusters=5, random_state=SEED):
    X_filled = X_df.fillna(X_df.mean())
    scaler_km = StandardScaler()
    X_scaled_km = scaler_km.fit_transform(X_filled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled_km)

    X_imputed = X_df.copy()
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
            X_imputed.loc[mask & X_df[col].isna(), col] = c_mean
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
            X_imputed.loc[mask & X_df[col].isna(), col] = fill_value
        X_imputed[col] = X_imputed[col].fillna(global_means[col])
    return X_imputed


def fit_preprocessor(X_fit_df, y_fit, scale_X=True, scale_y=True):
    """
    fit ทุกขั้น (outlier bounds -> KMeans impute -> scaler X/y) จากข้อมูลที่ส่งเข้ามาเท่านั้น
    คืนค่า (prep, X_fit_out, y_fit_out)
    """
    X_fit_df = X_fit_df.reset_index(drop=True)

    bounds = fit_outlier_bounds(X_fit_df)
    X_masked = apply_outlier_bounds(X_fit_df, bounds, cols_must_be_positive)
    X_imp, km, km_scaler, cluster_means, global_means = kmeans_impute_fit(X_masked)

    x_scaler = StandardScaler().fit(X_imp) if scale_X else None
    X_out = x_scaler.transform(X_imp) if scale_X else X_imp.values.astype(float)

    y_arr = np.asarray(y_fit, dtype=float)
    y_scaler = StandardScaler().fit(y_arr.reshape(-1, 1)) if scale_y else None
    y_out = y_scaler.transform(y_arr.reshape(-1, 1)).flatten() if scale_y else y_arr

    prep = {
        'bounds': bounds, 'km': km, 'km_scaler': km_scaler,
        'cluster_means': cluster_means, 'global_means': global_means,
        'x_scaler': x_scaler, 'y_scaler': y_scaler,
    }
    return prep, X_out, y_out


def transform_X(X_df, prep):
    """ใช้ preprocessor ที่ fit แล้วกับข้อมูลใหม่ (val/test) โดยไม่ fit อะไรเพิ่ม"""
    X_df = X_df.reset_index(drop=True)
    X_masked = apply_outlier_bounds(X_df, prep['bounds'], cols_must_be_positive)
    X_imp = kmeans_impute_transform(
        X_masked, prep['km'], prep['km_scaler'],
        prep['cluster_means'], prep['global_means']
    )
    if prep['x_scaler'] is not None:
        return prep['x_scaler'].transform(X_imp)
    return X_imp.values.astype(float)


def inverse_y(prep, y_scaled):
    if prep['y_scaler'] is None:
        return np.asarray(y_scaled, dtype=float)
    return prep['y_scaler'].inverse_transform(np.asarray(y_scaled).reshape(-1, 1)).flatten()


# ================================
# 3. Utilities
# ================================
def safe_predict(model, X_input, context_label=""):
    y_pred = np.asarray(model.predict(X_input)).flatten()
    n_bad = int((~np.isfinite(y_pred)).sum())
    if n_bad > 0:
        print(f"[WARNING] {context_label}: พบค่า NaN/Inf จำนวน {n_bad} จุดใน prediction")
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    return y_pred, n_bad


def mape_fn(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = y_true != 0
    return np.mean(np.abs((y_true[m] - y_pred[m]) / y_true[m])) * 100


def calc_metrics(y_true, y_pred):
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': math.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mape_fn(y_true, y_pred),
    }


def save_fig(path):
    plt.savefig(path, dpi=DPI, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    plt.close('all')


def safe_name(s):
    return s.replace(" ", "_").replace(".", "").replace("(", "").replace(")", "").replace("/", "_")


# ================================
# 4. Model wrappers (ให้ทุกโมเดลมี fit/predict แบบเดียวกัน)
# ================================


class OLSRegressor:
    """OLS (LinearRegression) fit บนค่าจริง ไม่ scale เพื่อให้ตีความ coefficient ตามหน่วยจริงได้"""

    def __init__(self, feature_names=None):
        self.feature_names = feature_names
        self.model_ = LinearRegression()
        self.params_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.model_.fit(X, np.asarray(y, dtype=float))
        names = self.feature_names or [f"x{i}" for i in range(X.shape[1])]
        self.params_ = {'intercept': float(self.model_.intercept_)}
        self.params_.update({n: float(c) for n, c in zip(names, self.model_.coef_)})
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


# ================================
# 5. Model specs
# ================================
logspace_vals = [float(v) for v in np.logspace(-6, 1, 4)]

MODEL_SPECS = [
    dict(
        name='SVR', features=FEATURES_BASE,
        make=lambda p, feats: SVR(**p),
        param_options=list(ParameterGrid({'kernel': ['linear'], 'C': [0.1, 1, 10], 'gamma': ['scale']})),
        fallback={'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        scale_X=True, scale_y=True, explainer='kernel',
    ),
    dict(
        name='OLS', features=FEATURES_OLS,
        make=lambda p, feats: OLSRegressor(feature_names=feats),
        param_options=[{}],                      # ไม่มี hyperparameter
        fallback={},
        scale_X=False, scale_y=False, explainer='linear',
    ),
    dict(
        name='XGBoost', features=FEATURES_BASE,
        make=lambda p, feats: XGBRegressor(random_state=SEED, verbosity=0, **p),
        param_options=list(ParameterGrid({'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, 0]})),
        fallback={'n_estimators': 100, 'max_depth': 5},
        scale_X=True, scale_y=True, explainer='tree',
    ),
    dict(
        name='RandomForest', features=FEATURES_BASE,
        make=lambda p, feats: RandomForestRegressor(random_state=SEED, **p),
        param_options=list(ParameterGrid({'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, None]})),
        fallback={'n_estimators': 100, 'max_depth': 5},
        scale_X=True, scale_y=True, explainer='tree',
    ),
    dict(
        name='LightGBM', features=FEATURES_LGB,
        make=lambda p, feats: LGBMRegressor(random_state=SEED, verbose=-1, **p),
        # LightGBM ใช้ max_depth=-1 แทน "ไม่จำกัดความลึก"
        param_options=list(ParameterGrid({'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10, -1]})),
        fallback={'n_estimators': 100, 'max_depth': 5},
        scale_X=True, scale_y=True, explainer='tree',
    ),
    dict(
        name='BayesianRidge', features=FEATURES_BASE,
        make=lambda p, feats: BayesianRidge(**p),
        param_options=list(ParameterGrid({
            'alpha_1': logspace_vals, 'alpha_2': logspace_vals,
            'lambda_1': logspace_vals, 'lambda_2': logspace_vals,
        })),
        fallback={'alpha_1': 1e-6, 'alpha_2': 1e-6, 'lambda_1': 1e-6, 'lambda_2': 1e-6},
        scale_X=True, scale_y=True, explainer='linear',
    ),
]

# ================================
# 6. Outer CV (LeaveOneGroupOut) + inner split (คำนวณครั้งเดียว ใช้ร่วมกันทุกโมเดล)
#    inner split ทำบน index ของ RAW data ก่อนทำ preprocessing ใด ๆ
# ================================
n_batteries = groups_all.nunique()
print(f"\nจำนวน Battery_ID ทั้งหมด: {n_batteries}")

outer_splits = list(LeaveOneGroupOut().split(df, y_all, groups=groups_all))
inner_splits = []
for train_idx, _ in outer_splits:
    g_train = groups_all.iloc[train_idx].reset_index(drop=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    inner_splits.append(next(gss.split(np.zeros(len(train_idx)), groups=g_train)))


# ================================
# 7. Hyperparameter selection: inner val ใช้เลือก hyperparameter เท่านั้น
# ================================
def select_hyperparams(spec, X_train_df, y_train, inner_tr_idx, inner_val_idx, fold):
    feats = spec['features']
    X_in_tr_raw = X_train_df.iloc[inner_tr_idx].reset_index(drop=True)
    X_in_val_raw = X_train_df.iloc[inner_val_idx].reset_index(drop=True)
    y_in_tr = y_train.iloc[inner_tr_idx].reset_index(drop=True)
    y_in_val = y_train.iloc[inner_val_idx].reset_index(drop=True)

    # preprocessor fit บน inner train เท่านั้น / val แค่ transform
    inner_prep, X_in_tr, y_in_tr_out = fit_preprocessor(
        X_in_tr_raw, y_in_tr, spec['scale_X'], spec['scale_y']
    )
    X_in_val = transform_X(X_in_val_raw, inner_prep)

    best_val_mae, best_params = float('inf'), None
    for params in spec['param_options']:
        model_tmp = spec['make'](params, feats)
        model_tmp.fit(X_in_tr, y_in_tr_out)
        pred_s, n_bad = safe_predict(
            model_tmp, X_in_val, context_label=f"[{spec['name']}] Fold {fold} params={params}"
        )
        if n_bad > 0:
            print(f"  -> ตัด params={params} ทิ้ง (prediction มี NaN/Inf)")
            continue
        val_mae = mean_absolute_error(y_in_val, inverse_y(inner_prep, pred_s))
        if val_mae < best_val_mae:
            best_val_mae, best_params = val_mae, params

    if best_params is None:
        print(f"[WARNING] [{spec['name']}] Fold {fold}: ทุก combination ล้มเหลว ใช้ fallback {spec['fallback']}")
        best_params = spec['fallback']
        print(f"Fold {fold} best params (fallback): {best_params}")
    else:
        print(f"Fold {fold} best params (inner val MAE={best_val_mae:.4f}): {best_params}")
    return best_params


# ================================
# 8. Run one model through all outer folds
# ================================
def run_model(spec):
    name, feats = spec['name'], spec['features']
    print(f"\n{'#' * 70}\n# MODEL: {name}\n{'#' * 70}")

    fold_results, all_preds, all_true = [], [], []
    last_model, last_X_test = None, None

    for fold, ((train_idx, test_idx), (in_tr, in_val)) in enumerate(zip(outer_splits, inner_splits), 1):
        print(f"\n===== [{name}] Outer Fold {fold}/{len(outer_splits)} =====")
        start_time = time.time()
        tracemalloc.start()

        X_train_df = df.iloc[train_idx][feats].reset_index(drop=True)
        X_test_df = df.iloc[test_idx][feats].reset_index(drop=True)
        y_train = y_all.iloc[train_idx].reset_index(drop=True)
        y_test = y_all.iloc[test_idx].reset_index(drop=True)

        # ---- เลือก hyperparameter (inner แยกอิสระ) ----
        best_params = select_hyperparams(spec, X_train_df, y_train, in_tr, in_val, fold)

        # ---- Final: fit preprocessor ใหม่บน train fold ทั้งก้อน แล้ว transform ให้ test ----
        final_prep, X_train_f, y_train_f = fit_preprocessor(
            X_train_df, y_train, spec['scale_X'], spec['scale_y']
        )
        X_test_f = transform_X(X_test_df, final_prep)

        best_model = spec['make'](best_params, feats)
        best_model.fit(X_train_f, y_train_f)

        # ---- Metrics ----
        tr_s, n_bad_tr = safe_predict(best_model, X_train_f, f"[{name}] Fold {fold} train")
        tr_pred = inverse_y(final_prep, tr_s)
        tr_m = calc_metrics(y_train, tr_pred)

        te_s, n_bad_te = safe_predict(best_model, X_test_f, f"[{name}] Fold {fold} test")
        te_pred = inverse_y(final_prep, te_s)
        te_m = calc_metrics(y_test, te_pred)

        all_preds.append(pd.Series(te_pred, index=test_idx))
        all_true.append(pd.Series(y_test.values, index=test_idx))

        elapsed = time.time() - start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        report_params = getattr(best_model, 'params_', None) or best_params  # OLS: เก็บ coefficient
        fold_results.append({
            'fold': fold,
            'Train_MAE': tr_m['MAE'], 'Train_RMSE': tr_m['RMSE'], 'Train_R2': tr_m['R2'], 'Train_MAPE': tr_m['MAPE'],
            'Test_MAE': te_m['MAE'], 'Test_RMSE': te_m['RMSE'], 'Test_R2': te_m['R2'], 'Test_MAPE': te_m['MAPE'],
            'time_sec': elapsed, 'memory_MB': peak / (1024 * 1024),
            'best_params': report_params,
            'diverged': (n_bad_tr > 0 or n_bad_te > 0),
        })

        print(f"Fold {fold} | Train: MAE={tr_m['MAE']:.4f}, RMSE={tr_m['RMSE']:.4f}, R2={tr_m['R2']:.4f}, MAPE={tr_m['MAPE']:.4f}% "
              f"| Test: MAE={te_m['MAE']:.4f}, RMSE={te_m['RMSE']:.4f}, R2={te_m['R2']:.4f}, MAPE={te_m['MAPE']:.4f}% "
              f"| Time={elapsed:.2f}s, Memory={peak / (1024 * 1024):.2f}MB")

        last_model, last_X_test = best_model, X_test_f

    return {
        'fold_df': pd.DataFrame(fold_results),
        'preds': pd.concat(all_preds).sort_index(),
        'true': pd.concat(all_true).sort_index(),
        'last_model': last_model,
        'last_X_test': last_X_test,
    }


# ================================
# 9. Summary / plots / SHAP (แยกไฟล์ ภาพละโมเดล dpi 600)
# ================================
METRIC_COLS = ['Train_MAE', 'Train_RMSE', 'Train_R2', 'Train_MAPE',
               'Test_MAE', 'Test_RMSE', 'Test_R2', 'Test_MAPE',
               'time_sec', 'memory_MB']


def summarize(name, fold_df, out_dir):
    print(f"\n===== [{name}] Summary across folds =====")
    print(fold_df)

    div = fold_df[fold_df['diverged']]
    if len(div) > 0:
        print(f"[WARNING] [{name}] {len(div)} fold มี NaN/Inf: {div['fold'].tolist()}")
    clean = fold_df[~fold_df['diverged']]
    if len(clean) < len(fold_df):
        print(f"[INFO] [{name}] ใช้ {len(clean)}/{len(fold_df)} fold ที่ไม่ diverge ในการสรุปสถิติ")

    stats = pd.DataFrame({
        'mean': clean[METRIC_COLS].mean(), 'std': clean[METRIC_COLS].std(),
        'median': clean[METRIC_COLS].median(), 'min': clean[METRIC_COLS].min(), 'max': clean[METRIC_COLS].max(),
    })
    stats['mean ± std'] = stats.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1)
    print(f"\n===== [{name}] Mean ± Std (เฉพาะ fold ที่ไม่ diverge) =====")
    print(stats[['mean ± std', 'median', 'min', 'max']].to_string())

    fold_df.assign(best_params=fold_df['best_params'].astype(str)).to_csv(
        os.path.join(out_dir, 'fold_results.csv'), index=False)
    stats.to_csv(os.path.join(out_dir, 'summary_stats.csv'))
    return stats


def plot_pred_vs_actual(name, y_true, y_pred, out_dir):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true.values, y_pred.values, color='red', label='Predicted')
    lo, hi = y_true.min(), y_true.max()
    plt.plot([lo, hi], [lo, hi], color='blue', linestyle='--', label='Ideal')
    plt.xlabel('Actual RUL (Cycle)')
    plt.ylabel('Predicted RUL (Cycle)')
    plt.title(f'{name} Leave-One-Battery-Out CV Predictions')
    plt.legend()
    plt.grid(True)
    save_fig(os.path.join(out_dir, 'pred_vs_actual.png'))


def compute_shap_values(spec, model, X_last):
    kind = spec['explainer']
    n_rows = X_last.shape[0]
    X_explain = X_last[:max(n_rows // 2, 1)]

    if kind == 'tree':
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_explain)

    elif kind == 'kernel':
        background = X_last[:min(50, n_rows)]
        X_explain = X_explain[:min(50, X_explain.shape[0])]      # KernelExplainer ช้า
        explainer = shap.KernelExplainer(model.predict, background)
        sv = explainer.shap_values(X_explain)

    elif kind == 'linear':
        background = X_last[:min(100, n_rows)]
        explainer = shap.LinearExplainer(getattr(model, 'model_', model), background)
        sv = explainer.shap_values(X_explain)
    else:
        raise ValueError(kind)

    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)

    if sv.ndim > 2:   # เช่น (samples, 1, features) หรือ (samples, 1, features, 1)
        squeeze_axes = tuple(ax for ax in range(1, sv.ndim - 1) if sv.shape[ax] == 1)
        if squeeze_axes:
            sv = np.squeeze(sv, axis=squeeze_axes)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv.reshape(sv.shape[0], sv.shape[1])
    return sv, X_explain


def run_shap(spec, model, X_last, out_dir):
    feats = spec['features']
    shap_dir = os.path.join(out_dir, 'shap')
    os.makedirs(shap_dir, exist_ok=True)

    sv, X_plot = compute_shap_values(spec, model, X_last)
    if sv.shape[1] != len(feats):
        print(f"[WARNING] [{spec['name']}] shape ของ SHAP {sv.shape} ไม่ตรงกับจำนวน feature {len(feats)}")

    shap.summary_plot(sv, X_plot, feature_names=feats, show=False)
    save_fig(os.path.join(shap_dir, 'shap_summary_plot.png'))

    X_plot_df = pd.DataFrame(X_plot, columns=feats)
    main_feat = "Max. Voltage Dischar. (V)"
    if main_feat in feats:
        shap.dependence_plot(main_feat, sv, X_plot_df, show=False)
        save_fig(os.path.join(shap_dir, 'shap_dependence_plot.png'))

    for i, feature in enumerate(feats):
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(feature, sv, X_plot_df, show=False)
        save_fig(os.path.join(shap_dir, f'shap_dependence_{i}_{safe_name(feature)}.png'))

    print(f"[INFO] [{spec['name']}] บันทึกรูป SHAP ไว้ที่: {shap_dir}")


# ================================
# 10. Main
# ================================
combined_rows = []

for spec in MODEL_SPECS:
    if MODELS_TO_RUN is not None and spec['name'] not in MODELS_TO_RUN:
        continue

    model_dir = os.path.join(OUTPUT_ROOT, spec['name'])
    os.makedirs(model_dir, exist_ok=True)

    res = run_model(spec)
    stats = summarize(spec['name'], res['fold_df'], model_dir)
    plot_pred_vs_actual(spec['name'], res['true'], res['preds'], model_dir)

    try:
        run_shap(spec, res['last_model'], res['last_X_test'], model_dir)
    except Exception as e:   # ให้โมเดลอื่นรันต่อได้ถ้า SHAP ของโมเดลนี้พัง
        print(f"[ERROR] [{spec['name']}] SHAP ล้มเหลว: {type(e).__name__}: {e}")

    row = {'Model': spec['name']}
    for m in METRIC_COLS:
        row[f'{m}_mean'] = stats.loc[m, 'mean']
        row[f'{m}_std'] = stats.loc[m, 'std']
    combined_rows.append(row)

if combined_rows:
    combined_df = pd.DataFrame(combined_rows)
    combined_df.to_csv(os.path.join(OUTPUT_ROOT, 'all_models_summary.csv'), index=False)
    print("\n===== เปรียบเทียบทุกโมเดล (Test metrics, mean ± std) =====")
    for _, r in combined_df.iterrows():
        print(f"{r['Model']:<14} "
              f"MAE={r['Test_MAE_mean']:.4f}±{r['Test_MAE_std']:.4f}  "
              f"RMSE={r['Test_RMSE_mean']:.4f}±{r['Test_RMSE_std']:.4f}  "
              f"R2={r['Test_R2_mean']:.4f}±{r['Test_R2_std']:.4f}  "
              f"MAPE={r['Test_MAPE_mean']:.2f}%±{r['Test_MAPE_std']:.2f}")
    print(f"\n[INFO] ผลทั้งหมดอยู่ที่: {OUTPUT_ROOT}")