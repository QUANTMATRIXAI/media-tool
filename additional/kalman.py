import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import date
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

# =============================
# Helpers (metrics & transforms)
# =============================

def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1.0, cap: float | None = 500.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) >= eps
    if not np.any(mask):
        return np.nan
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    if cap is not None:
        return float(min(mape, cap))
    return float(mape)


def add_intercept(X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1), dtype=float)
    return np.hstack([ones, X])


@dataclass
class Standardization:
    scaler: Optional[object]
    feature_names: List[str]

    @staticmethod
    def fit(df: pd.DataFrame, cols: List[str], use: bool) -> "Standardization":
        if not use:
            return Standardization(None, cols)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler(with_mean=True, with_std=True)
        sc.fit(df[cols].values)
        return Standardization(sc, cols)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_names].values.astype(float)
        if self.scaler is None:
            return X
        return self.scaler.transform(X)

    def invert_betas(self, beta_std: np.ndarray, intercept_std: float) -> Tuple[np.ndarray, float]:
        if self.scaler is None:
            return beta_std.copy(), float(intercept_std)
        sd = self.scaler.scale_.copy()
        mu = self.scaler.mean_.copy()
        sd = np.where(sd == 0.0, 1.0, sd)
        betas_orig = beta_std / sd
        intercept_orig = intercept_std - float(np.dot(betas_orig, mu))
        return betas_orig, float(intercept_orig)


# =============================
# Structured config/results
# =============================


@dataclass
class KalmanConfig:
    group_cols: List[str] = field(default_factory=list)
    order_col: Optional[str] = None
    week_col: Optional[str] = None
    year_col: Optional[str] = None
    holdout_weeks: Optional[int] = None
    standardize: bool = True
    drop_constant: bool = True
    use_ols_init: bool = True
    update_test_with_truth: bool = False
    q: float = 1e-4
    r: float = 1.0
    target: str = ""
    predictors: List[str] = field(default_factory=list)


@dataclass
class KalmanResults:
    preds: pd.DataFrame
    metrics: pd.DataFrame
    coefs: pd.DataFrame


# =============================
# Date/Week utilities
# =============================

def build_order_from_week(df: pd.DataFrame, week_col: str, year_col: Optional[str]) -> pd.Series:
    out = []
    for idx, val in df[week_col].items():
        if pd.isna(val):
            out.append(pd.NaT)
            continue
        s = str(val).strip()
        up = s.upper().replace('_', '').strip()
        if 'W' in up:
            parts = up.split('W')
            try:
                yr = int(parts[0]); wk = int(parts[1])
            except Exception:
                raise ValueError(f"Cannot parse ISO week string: {s!r}")
        else:
            if not year_col:
                raise ValueError("Numeric week values require Year column.")
            try:
                wk = int(float(up)); yr = int(df.at[idx, year_col])
            except Exception:
                raise ValueError(f"Missing/invalid week/year at row {idx} for columns {week_col!r}/{year_col!r}")
        wk = max(1, min(int(wk), 53))
        try:
            ts = pd.Timestamp(date.fromisocalendar(int(yr), int(wk), 1))
        except Exception:
            ts = pd.to_datetime(f"{int(yr)}-01-01") + pd.to_timedelta((int(wk) - 1) * 7, unit='D')
        out.append(ts)
    return pd.Series(out, index=df.index, name='_ORDER_')


# =============================
# Kalman model (robust init)
# =============================
class TVLinearKalman:
    """Time‑varying linear regression with Kalman updates.
    Observation: y_t = x_t' beta_t + v_t,  v_t ~ N(0, R)
    State:       beta_t = beta_{t-1} + w_t, w_t ~ N(0, Q)
    We set Q = q * I, R = r (scalar).
    """
    def __init__(self, n_features: int, q: float = 1e-4, r: float = 1.0, init_cov: float = 1e3):
        self.n = int(n_features)
        self.q = float(q)
        self.r = float(r)
        self.init_cov = float(init_cov)
        self.Q = np.eye(self.n) * self.q
        self.R = self.r
        self.I = np.eye(self.n)
        self.beta0 = np.zeros(self.n)
        self.P0 = np.eye(self.n) * self.init_cov
        self.beta_last_: Optional[np.ndarray] = None
        self.P_last_: Optional[np.ndarray] = None

    def _step(self, x_t: np.ndarray, y_t: Optional[float], beta_prev: np.ndarray, P_prev: np.ndarray, update: bool = True):
        x_t = x_t.reshape(-1, 1)
        beta_pred = beta_prev
        P_pred = P_prev + self.Q
        y_pred = float(np.dot(beta_pred, x_t).item())
        if update and y_t is not None and np.isfinite(y_t):
            S = float((x_t.T @ P_pred @ x_t).item() + self.R)
            K = (P_pred @ x_t) / S
            resid = y_t - y_pred
            beta_upd = beta_pred + (K.flatten() * resid)
            P_upd = (self.I - (K @ x_t.T)) @ P_pred
        else:
            beta_upd = beta_pred
            P_upd = P_pred
        return beta_pred, P_pred, y_pred, beta_upd, P_upd

    def _ridge_init(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        T, p = X.shape
        n0 = max(min( max(p + 5, 10), T), 2)  # use up to T, at least 2
        X0, y0 = X[:n0], y[:n0]
        XtX = X0.T @ X0
        # scale‑aware ridge strength
        lam = 1e-2 * (np.trace(XtX) / max(p, 1)) + 1e-6
        beta = np.linalg.solve(XtX + lam * np.eye(p), X0.T @ y0)
        # safety: keep intercept reasonable
        if not np.isfinite(beta).all():
            beta = self.beta0.copy()
        if abs(beta[0]) < 1e-8:
            beta[0] = float(np.mean(y))
        return beta

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, use_ols_init: bool = True) -> Dict[str, np.ndarray]:
        T, p = X_train.shape
        assert p == self.n
        betas_prior = np.zeros((T, p))
        betas_filt = np.zeros((T, p))
        y_pred = np.zeros(T)

        # Robust init: ridge (works even when T < p). Also set intercept to mean(y) if needed.
        beta = self._ridge_init(X_train, y_train) if use_ols_init and T >= 2 else self.beta0.copy()
        if T >= 1 and abs(beta[0]) < 1e-8:
            beta[0] = float(np.mean(y_train))

        P = self.P0.copy()
        for t in range(T):
            beta_pred, P_pred, yhat, beta_upd, P_upd = self._step(X_train[t], float(y_train[t]), beta, P, update=True)
            betas_prior[t] = beta_pred; betas_filt[t] = beta_upd; y_pred[t] = yhat
            beta, P = beta_upd, P_upd
        self.beta_last_ = beta; self.P_last_ = P
        return {'betas_prior': betas_prior, 'betas_filt': betas_filt, 'y_pred_train': y_pred}

    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None, update: bool = False) -> Dict[str, np.ndarray]:
        assert self.beta_last_ is not None and self.P_last_ is not None, "Call fit() first."
        T, p = X.shape
        betas_prior = np.zeros((T, p)); betas_filt = np.zeros((T, p)); y_pred = np.zeros(T)
        beta = self.beta_last_.copy(); P = self.P_last_.copy()
        for t in range(T):
            y_t = float(y[t]) if (y is not None and update and np.isfinite(y[t])) else None
            beta_pred, P_pred, yhat, beta_upd, P_upd = self._step(X[t], y_t, beta, P, update=update)
            betas_prior[t] = beta_pred; betas_filt[t] = beta_upd; y_pred[t] = yhat
            beta, P = beta_upd, P_upd
        self.beta_last_ = beta; self.P_last_ = P
        return {'betas_prior': betas_prior, 'betas_filt': betas_filt, 'y_pred': y_pred}


# =============================
# Grouped pipeline (forecast‑safe test)
# =============================
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def fit_grouped_kalman(df: pd.DataFrame, config: KalmanConfig) -> KalmanResults:
    """Fits one Kalman model per group using the provided configuration."""
    cfg = config
    df = df.copy()

    group_cols = list(cfg.group_cols)
    order_col = cfg.order_col
    week_col = cfg.week_col
    year_col = cfg.year_col
    holdout_weeks = cfg.holdout_weeks
    standardize = cfg.standardize
    drop_constant = cfg.drop_constant
    use_ols_init = cfg.use_ols_init
    update_test_with_truth = cfg.update_test_with_truth
    q = cfg.q
    r = cfg.r
    target = cfg.target
    predictors = cfg.predictors

    # Build effective order column
    effective_order_col = None
    if week_col and week_col in df.columns:
        df['_ORDER_'] = build_order_from_week(df, week_col, year_col)
        effective_order_col = '_ORDER_'
    elif order_col and order_col in df.columns:
        effective_order_col = order_col
        try:
            df[effective_order_col] = pd.to_datetime(df[effective_order_col])
        except Exception:
            pass

    preds_rows, metrics_rows, coefs_rows = [], [], []

    # Grouping
    if group_cols:
        group_iter = df.groupby(group_cols, sort=False)
        group_cols_internal = group_cols
    else:
        df['_ALL_'] = 'ALL'
        group_cols_internal = ['_ALL_']
        group_iter = df.groupby(group_cols_internal, sort=False)

    for combo_vals, g in group_iter:
        g = g.copy()
        if effective_order_col and effective_order_col in g.columns:
            g = g.sort_values(effective_order_col)
        needed = predictors + [target]
        g = g.dropna(subset=needed)
        if g.empty:
            continue

        # Split: use last N unique periods as holdout if asked
        if holdout_weeks is None or holdout_weeks == 0:
            insample_mask = pd.Series(True, index=g.index)
        else:
            if effective_order_col and effective_order_col in g.columns:
                uniq = g[effective_order_col].drop_duplicates().sort_values()
                if len(uniq) > holdout_weeks:
                    cutoff = uniq.iloc[-holdout_weeks - 1]
                    insample_mask = g[effective_order_col] <= cutoff
                else:
                    insample_mask = pd.Series(True, index=g.index)
            else:
                n_holdout = min(holdout_weeks, len(g) - 2)
                insample_mask = pd.Series(True, index=g.index)
                if n_holdout > 0:
                    insample_mask.iloc[-n_holdout:] = False

        # Feature filtering: drop constants within train window
        preds_use = predictors
        if drop_constant:
            preds_use = [c for c in predictors if g.loc[insample_mask, c].nunique(dropna=True) > 1]
        if len(preds_use) == 0:
            continue

        # Standardize on train only
        std = Standardization.fit(g.loc[insample_mask], preds_use, use=standardize)

        # Design matrices
        Xg = std.transform(g)
        Xg = add_intercept(Xg)  # intercept first
        yg = g[target].values.astype(float)

        X_in = Xg[insample_mask.values]; y_in = yg[insample_mask.values]
        X_out = Xg[~insample_mask.values]; y_out = yg[~insample_mask.values]
        if X_in.shape[0] < 2:
            continue

        # Fit and predict
        kf = TVLinearKalman(n_features=X_in.shape[1], q=q, r=r)
        train_out = kf.fit(X_in, y_in, use_ols_init=use_ols_init)
        has_out = X_out.shape[0] > 0
        if has_out:
            test_out = kf.predict(X_out, y=y_out, update=update_test_with_truth)
            y_pred_out = test_out['y_pred']
        else:
            test_out = {'betas_prior': np.zeros((0, X_in.shape[1])), 'betas_filt': np.zeros((0, X_in.shape[1]))}
            y_pred_out = np.array([])

        # In-sample diagnostics: one-step-ahead vs filtered fit
        y_pred_in_one_step = train_out['y_pred_train']
        betas_filt_in = train_out['betas_filt']
        y_pred_in_filtered = np.einsum('ij,ij->i', betas_filt_in, X_in)

        # Metrics (use out-of-sample if present, else in-sample 1-step ahead)
        if has_out:
            r2 = r2_score(y_out, y_pred_out) if y_out.size > 1 else np.nan
            mape = safe_mape(y_out, y_pred_out)
            mae = mean_absolute_error(y_out, y_pred_out)
            mse = mean_squared_error(y_out, y_pred_out)
            rmse = float(np.sqrt(mse))
            naive_mean = np.full_like(y_out, y_in.mean())
            naive_last = np.full_like(y_out, y_in[-1])
            r2_mean = r2_score(y_out, naive_mean) if y_out.size > 1 else np.nan
            r2_last = r2_score(y_out, naive_last) if y_out.size > 1 else np.nan
            r2_train_one_step = r2_score(y_in, y_pred_in_one_step) if y_in.size > 1 else np.nan
            r2_train_filtered = r2_score(y_in, y_pred_in_filtered) if y_in.size > 1 else np.nan
        else:
            r2 = r2_score(y_in, y_pred_in_one_step) if y_in.size > 1 else np.nan
            mape = safe_mape(y_in, y_pred_in_one_step)
            mae = mean_absolute_error(y_in, y_pred_in_one_step)
            mse = mean_squared_error(y_in, y_pred_in_one_step)
            rmse = float(np.sqrt(mse))
            r2_mean = r2_last = np.nan
            r2_train_one_step = r2
            r2_train_filtered = r2_score(y_in, y_pred_in_filtered) if y_in.size > 1 else np.nan

        # Convert betas back to original feature scale (time-varying)
        def convert_series(beta_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            if beta_mat.size == 0: return beta_mat, beta_mat
            b_no_int = beta_mat[:, 1:]
            intercept_std = beta_mat[:, 0]
            b_orig_list, b0_list = [], []
            for t in range(beta_mat.shape[0]):
                b_orig, b0 = std.invert_betas(b_no_int[t], intercept_std[t])
                b_orig_list.append(b_orig); b0_list.append(b0)
            return np.vstack(b_orig_list), np.asarray(b0_list)

        b_prior_in_orig, b0_prior_in_orig = convert_series(train_out['betas_prior'])
        b_filt_in_orig,  b0_filt_in_orig  = convert_series(train_out['betas_filt'])
        b_prior_out_orig, b0_prior_out_orig = convert_series(test_out['betas_prior'])
        b_filt_out_orig,  b0_filt_out_orig  = convert_series(test_out.get('betas_filt', np.zeros_like(test_out['betas_prior'])))

        # Build per-row predictions
        g_in = g.loc[insample_mask]
        for i, idx in enumerate(g_in.index):
            row = {**({c: g_in.loc[idx, c] for c in group_cols_internal} if group_cols_internal else {}),
                   'Phase': 'in-sample', 'Index': idx,
                   'y_true': float(g_in.loc[idx, target]), 'y_pred': float(train_out['y_pred_train'][i])}
            if effective_order_col: row['Order'] = g_in.loc[idx, effective_order_col]
            row['Intercept_prior'] = float(b0_prior_in_orig[i]); row['Intercept_filt'] = float(b0_filt_in_orig[i])
            for j, name in enumerate(preds_use):
                row[f'Beta_{name}_prior'] = float(b_prior_in_orig[i, j]); row[f'Beta_{name}_filt'] = float(b_filt_in_orig[i, j])
            preds_rows.append(row)

        if has_out:
            g_out = g.loc[~insample_mask]
            for i, idx in enumerate(g_out.index):
                row = {**({c: g_out.loc[idx, c] for c in group_cols_internal} if group_cols_internal else {}),
                       'Phase': 'out-of-sample', 'Index': idx,
                       'y_true': float(g_out.loc[idx, target]), 'y_pred': float(y_pred_out[i])}
                if effective_order_col: row['Order'] = g_out.loc[idx, effective_order_col]
                row['Intercept_prior'] = float(b0_prior_out_orig[i]) if b0_prior_out_orig.size else np.nan
                row['Intercept_filt']  = float(b0_filt_out_orig[i])  if b0_filt_out_orig.size  else np.nan
                for j, name in enumerate(preds_use):
                    row[f'Beta_{name}_prior'] = float(b_prior_out_orig[i, j]) if b_prior_out_orig.size else np.nan
                    row[f'Beta_{name}_filt']  = float(b_filt_out_orig[i, j])  if b_filt_out_orig.size  else np.nan
                preds_rows.append(row)

        # Metrics row
        metrics_rows.append({
            **({c: g.iloc[0][c] for c in group_cols_internal} if group_cols_internal else {}),
            'q': float(q), 'r': float(r),
            'k_features': int(len(preds_use)),
            'n_in': int(insample_mask.sum()), 'n_out': int((~insample_mask).sum()),
            'R2': float(r2) if np.isfinite(r2) else np.nan,
            'MAPE': float(mape) if np.isfinite(mape) else np.nan,
            'MAE': float(mae) if np.isfinite(mae) else np.nan,
            'MSE': float(mse) if np.isfinite(mse) else np.nan,
            'RMSE': float(rmse) if np.isfinite(rmse) else np.nan,
            'R2_naive_mean': float(r2_mean) if np.isfinite(r2_mean) else np.nan,
            'R2_naive_last': float(r2_last) if np.isfinite(r2_last) else np.nan,
            'R2_train_one_step': float(r2_train_one_step) if np.isfinite(r2_train_one_step) else np.nan,
            'R2_train_filtered': float(r2_train_filtered) if np.isfinite(r2_train_filtered) else np.nan,
        })

        # Coefficients tidy table (filtered betas on original scale)
        def rows_from_betas(gpart: pd.DataFrame, betas_orig: np.ndarray, b0_orig: np.ndarray, phase: str):
            for i, idx in enumerate(gpart.index):
                row = {**({c: gpart.loc[idx, c] for c in group_cols_internal} if group_cols_internal else {}), 'Phase': phase, 'Index': idx,
                       'Intercept': float(b0_orig[i]) if b0_orig.size else np.nan}
                if effective_order_col: row['Order'] = gpart.loc[idx, effective_order_col]
                for j, name in enumerate(preds_use):
                    row[f'Beta_{name}'] = float(betas_orig[i, j]) if betas_orig.size else np.nan
                coefs_rows.append(row)
        rows_from_betas(g_in, b_filt_in_orig, b0_filt_in_orig, 'in-sample')
        if has_out:
            rows_from_betas(g.loc[~insample_mask], b_filt_out_orig, b0_filt_out_orig, 'out-of-sample')

    preds_df = pd.DataFrame(preds_rows)
    metrics_df = pd.DataFrame(metrics_rows)
    coefs_df = pd.DataFrame(coefs_rows)

    # Overall metrics if we have out-of-sample rows
    if not preds_df.empty and (preds_df['Phase'] == 'out-of-sample').any():
        mask_out = preds_df['Phase'] == 'out-of-sample'
        y_true_all = preds_df.loc[mask_out, 'y_true'].values
        y_pred_all = preds_df.loc[mask_out, 'y_pred'].values
        if y_true_all.size > 0:
            overall = {**{c: '[ALL]' for c in group_cols_internal}, 'q': np.nan, 'r': np.nan,
                       'n_in': int((preds_df['Phase'] == 'in-sample').sum()), 'n_out': int(mask_out.sum()),
                       'R2': r2_score(y_true_all, y_pred_all) if y_true_all.size > 1 else np.nan,
                       'MAPE': safe_mape(y_true_all, y_pred_all),
                       'MAE': mean_absolute_error(y_true_all, y_pred_all),
                       'MSE': mean_squared_error(y_true_all, y_pred_all),
                       'RMSE': float(np.sqrt(mean_squared_error(y_true_all, y_pred_all)))}
            metrics_df = pd.concat([metrics_df, pd.DataFrame([overall])], ignore_index=True)

    return KalmanResults(preds=preds_df, metrics=metrics_df, coefs=coefs_df)


# =============================
# Streamlit UI (no CLI)
# =============================
st.set_page_config(page_title="Kalman / Time‑Varying Linear Regression", layout="wide")
st.title("Kalman / Time‑Varying Linear Regression (State‑Space)")
st.caption("Upload CSV/Excel/Parquet, pick grouping/target/predictors, and get time‑varying regression with Kalman updates.")

file = st.file_uploader("Upload data file", type=["csv", "xlsx", "xls", "parquet"]) 

@st.cache_data(show_spinner=False)
def load_any(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith('.csv'):
        return pd.read_csv(uploaded)
    if name.endswith('.xlsx') or name.endswith('.xls'):
        return pd.read_excel(uploaded)
    if name.endswith('.parquet'):
        return pd.read_parquet(uploaded)
    # Fallbacks
    try:
        uploaded.seek(0); return pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0); return pd.read_excel(uploaded)

if not file:
    st.info("📂 Upload a CSV / Excel / Parquet to start.")
    st.stop()

try:
    df = load_any(file)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

cols = list(df.columns)
st.success(f"Loaded {len(df):,} rows × {len(cols)} columns")

# Sidebar
st.sidebar.header("Configuration")

# Grouping
group_cols = st.sidebar.multiselect("Grouping columns (optional)", cols)

# Time columns
order_col = st.sidebar.selectbox("Date column (optional)", [""] + cols)
week_col  = st.sidebar.selectbox("ISO Week column (optional)", [""] + cols)
year_col  = st.sidebar.selectbox("Year column (if week is numeric)", [""] + cols)
if order_col and week_col:
    st.sidebar.warning("Using Date column; ignoring Week column.")
    week_col = ""

# Split
use_holdout = st.sidebar.checkbox("Use holdout (last N periods)", value=False)
holdout_weeks = st.sidebar.number_input("Holdout periods (N)", min_value=1, max_value=52, value=12) if use_holdout else None

# Model options
standardize = st.sidebar.checkbox("Standardize predictors", True)
drop_constant = st.sidebar.checkbox("Drop constant features within each group", True)
use_ols_init = st.sidebar.checkbox("Ridge init (recommended)", True)
update_test_with_truth = st.sidebar.checkbox("Update test with truth (1‑step ahead)", False, help="Off = pure forecast. On = rolling 1‑step ahead with updates.")

q = st.sidebar.select_slider("q (process noise)", options=[1e-6,1e-5,1e-4,1e-3,1e-2,5e-2,1e-1], value=1e-4)
r = st.sidebar.number_input("r (observation noise)", value=1.0, min_value=1e-4, max_value=1e6, format="%f")

# Target & predictors
numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
if not numeric_cols:
    st.error("No numeric columns found. Target must be numeric.")
    st.stop()

target = st.selectbox("Target", numeric_cols)
predictor_candidates = [c for c in cols if c != target]
predictors = st.multiselect("Predictors", predictor_candidates, default=[c for c in numeric_cols if c != target][:6])

if not predictors:
    st.warning("Select at least one predictor.")

run = st.button("Run Kalman model", type="primary")
if not run:
    st.stop()

with st.spinner("Fitting time‑varying Kalman model..."):
    cfg = KalmanConfig(
        group_cols=group_cols,
        order_col=order_col or None,
        week_col=week_col or None,
        year_col=year_col or None,
        holdout_weeks=holdout_weeks,
        standardize=standardize,
        drop_constant=drop_constant,
        use_ols_init=use_ols_init,
        update_test_with_truth=update_test_with_truth,
        q=float(q),
        r=float(r),
        target=target,
        predictors=predictors,
    )
    results = fit_grouped_kalman(df=df, config=cfg)
    preds_df = results.preds
    metrics_df = results.metrics
    coefs_df = results.coefs

st.success("Done!")

# Metrics
st.subheader("Metrics")
if metrics_df.empty:
    st.warning("No metrics to show.")
else:
    warn_mask = metrics_df['R2'] < 0
    if warn_mask.any():
        st.warning("Some R2 scores are negative. This occurs when predictions underperform a naive mean baseline. Check `R2_train_one_step` for calibration and consider adjusting predictors or tuning q/r.")
    st.dataframe(metrics_df, use_container_width=True)

# Predictions
st.subheader("Predictions (sample)")
if preds_df.empty:
    st.info("No predictions to show.")
else:
    st.dataframe(preds_df.head(500), use_container_width=True)

# === NEW: First 10 predictions per group combination ===
st.subheader("First 10 predictions for each group combination")
if preds_df.empty:
    st.info("No predictions available.")
else:
    if group_cols:
        # ensure time order if available
        if 'Order' in preds_df.columns:
            preds_sorted = preds_df.sort_values(group_cols + ['Order'])
        else:
            preds_sorted = preds_df.sort_values(group_cols + ['Index'])
        # show up to first 10 groups to avoid huge UI
        max_groups_show = 10
        for gi, (keys, gsub) in enumerate(preds_sorted.groupby(group_cols)):
            if gi >= max_groups_show:
                st.caption(f"(Showing only first {max_groups_show} groups)")
                break
            label = ", ".join([f"{col}={val}" for col, val in zip(group_cols, (keys if isinstance(keys, tuple) else (keys,)))])
            with st.expander(f"Group: {label}"):
                cols_to_show = ['Phase']
                if 'Order' in gsub.columns:
                    cols_to_show.append('Order')
                if 'Index' in gsub.columns:
                    cols_to_show.append('Index')
                cols_to_show += ['y_true', 'y_pred']
                st.dataframe(gsub.head(10)[cols_to_show], use_container_width=True)
    else:
        preds_sorted = preds_df.sort_values('Order' if 'Order' in preds_df.columns else 'Index')
        cols_to_show = ['Phase']
        if 'Order' in preds_sorted.columns:
            cols_to_show.append('Order')
        if 'Index' in preds_sorted.columns:
            cols_to_show.append('Index')
        cols_to_show += ['y_true', 'y_pred']
        st.dataframe(preds_sorted.head(10)[cols_to_show], use_container_width=True)

# Optional chart
if not preds_df.empty and ('Order' in preds_df.columns):
    st.subheader("Time series — y_true vs y_pred (mean across groups)")
    plot_df = preds_df.copy()
    plot_df['__Order__'] = pd.to_datetime(plot_df['Order'])
    if group_cols:
        # allow simple filter on the first grouping col if many groups
        g0 = group_cols[0]
        vals = ["[ALL]"] + sorted(plot_df[g0].astype(str).unique().tolist())
        sel = st.selectbox(f"Filter {g0}", vals)
        if sel != "[ALL]":
            plot_df = plot_df[plot_df[g0].astype(str) == sel]
    agg = plot_df.groupby('__Order__', as_index=False)[['y_true','y_pred']].mean().sort_values('__Order__').set_index('__Order__')
    st.line_chart(agg)

# Downloads
st.subheader("Downloads")
col1, col2, col3 = st.columns(3)

def df_to_csv_bytes(d: pd.DataFrame) -> bytes:
    buf = BytesIO(); d.to_csv(buf, index=False); return buf.getvalue()

with col1:
    st.download_button("metrics.csv", df_to_csv_bytes(metrics_df), "metrics.csv", "text/csv")
with col2:
    st.download_button("predictions.csv", df_to_csv_bytes(preds_df), "predictions.csv", "text/csv")
with col3:
    st.download_button("coefficients.csv", df_to_csv_bytes(coefs_df), "coefficients.csv", "text/csv")
