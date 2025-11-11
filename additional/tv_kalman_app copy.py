import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from itertools import product
import io
from scipy import stats


class CustomConstrainedRidge:
    """Ridge regression with optional coefficient sign constraints."""

    def __init__(self, l2_penalty=0.1, learning_rate=0.001, iterations=10000,
                 adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 non_positive_features=None, non_negative_features=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.l2_penalty = l2_penalty
        self.adam = adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.non_positive_features = tuple(non_positive_features) if non_positive_features else ()
        self.non_negative_features = tuple(non_negative_features) if non_negative_features else ()

    def fit(self, X, y, feature_names):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0.0
        self.X = X
        self.Y = y
        self.feature_names = feature_names

        configured_non_positive = set(self.non_positive_features)
        configured_non_negative = set(self.non_negative_features)

        self._non_positive_indices = [i for i, name in enumerate(feature_names)
                                      if name in configured_non_positive]
        self._non_negative_indices = [i for i, name in enumerate(feature_names)
                                      if name in configured_non_negative]

        if self.adam:
            self.m_W = np.zeros(self.n)
            self.v_W = np.zeros(self.n)
            self.m_b = 0.0
            self.v_b = 0.0
            self.t = 0

        for _ in range(self.iterations):
            self._update_weights()

        self.intercept_ = self.b
        self.coef_ = self.W.copy()
        return self

    def _update_weights(self):
        Y_pred = self.predict(self.X)
        grad_w = (-(2 * self.X.T.dot(self.Y - Y_pred)) + 2 * self.l2_penalty * self.W) / self.m
        grad_b = -(2.0 / self.m) * np.sum(self.Y - Y_pred)

        if self.adam:
            self.t += 1
            self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * grad_w
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
            self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (grad_w ** 2)
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)

            m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

            self.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            self.W -= self.learning_rate * grad_w
            self.b -= self.learning_rate * grad_b

        if self._non_positive_indices:
            self.W[self._non_positive_indices] = np.minimum(self.W[self._non_positive_indices], 0.0)
        if self._non_negative_indices:
            self.W[self._non_negative_indices] = np.maximum(self.W[self._non_negative_indices], 0.0)

    def predict(self, X):
        return X.dot(self.W) + self.b

# Set page config
st.set_page_config(
    page_title="Kalman Filter Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)


class ConstrainedTVLinearKalman:
    def __init__(self, n_features: int, q: float = 1e-4, r: float = 1.0, init_cov: float = 1e3,
                 min_pred=0, max_pred=None, use_log=False, adaptive=True,
                 q_alpha: float = 0.05, r_alpha: float = 0.05,
                 use_ridge_init: bool = True, ridge_alpha: float = 1.0,
                 non_negative_features=None, non_positive_features=None):
        self.n = n_features
        self.q_init = q
        self.r_init = r
        self.q = q
        self.r = r
        self.init_cov = init_cov
        self.Q = np.eye(self.n) * self.q
        self.R = self.r
        self.I = np.eye(self.n)
        self.beta0 = np.zeros(self.n)
        self.P0 = np.eye(self.n) * self.init_cov
        self.min_pred = min_pred
        self.max_pred = max_pred
        self.use_log = use_log
        self.adaptive = adaptive
        self.q_alpha = float(q_alpha)
        self.r_alpha = float(r_alpha)
        self.use_ridge_init = bool(use_ridge_init)
        self.ridge_alpha = float(ridge_alpha)
        self.non_negative_features = set(non_negative_features or [])
        self.non_positive_features = set(non_positive_features or [])
        self.feature_names = None
        self._nonneg_idx = []
        self._nonpos_idx = []
        
        # Adaptive bounds and tracking
        self.q_min = q * 0.1
        self.q_max = q * 10
        self.r_min = r * 0.1
        self.r_max = r * 10
        self.last_beta_upd = None
        self.innovations = []
    
    def _step(self, x_t, y_t, beta_prev, P_prev, update=True):
        x_t = x_t.reshape(-1, 1)
        beta_pred = beta_prev.copy()
        P_pred = P_prev + self.Q
        y_pred_raw = (beta_pred @ x_t).item()

        # Prediction used for Kalman update (log domain if use_log)
        pred_for_update = y_pred_raw
        if self.min_pred is not None or self.max_pred is not None:
            if self.use_log:
                lower = np.log1p(self.min_pred) if self.min_pred is not None else None
                upper = np.log1p(self.max_pred) if self.max_pred is not None else None
            else:
                lower = self.min_pred
                upper = self.max_pred
            if lower is not None and pred_for_update < lower:
                pred_for_update = lower
            if upper is not None and pred_for_update > upper:
                pred_for_update = upper

        # Prediction shown in original space (after optional clamping)
        if self.use_log:
            # Inverse of log1p transform used during training
            y_pred = np.expm1(pred_for_update)
        else:
            y_pred = pred_for_update

        if self.min_pred is not None:
            y_pred = max(y_pred, self.min_pred)
        if self.max_pred is not None:
            y_pred = min(y_pred, self.max_pred)

        innovation = None
        if update and np.isfinite(y_t):
            resid = y_t - pred_for_update
            # Innovation covariance and gain using current R
            S = (x_t.T @ P_pred @ x_t).item() + self.R
            # Numerical safety
            if not np.isfinite(S) or S <= 1e-12:
                S = 1e-12
            K = (P_pred @ x_t) / S
            beta_upd = beta_pred + K.flatten() * resid
            temp = self.I - K @ x_t.T
            P_upd = temp @ P_pred @ temp.T + (K @ K.T) * self.R

            if self._nonneg_idx or self._nonpos_idx:
                self._project_state(beta_upd, P_upd)
            innovation = float(resid)

            # Principled adaptive updates (innovation-based) for R and state-increment for Q
            if self.adaptive:
                # Update R from innovation variance: E[nu^2] = x' P_pred x + R
                s_state = float((x_t.T @ P_pred @ x_t).item())
                s_state = max(0.0, s_state)
                innovation_var = resid * resid
                r_sample = innovation_var - s_state
                if not np.isfinite(r_sample) or r_sample <= 0:
                    r_sample = max(self.r_min, innovation_var)
                r_sample = min(max(r_sample, self.r_min), self.r_max)
                self.r = (1.0 - self.r_alpha) * self.r + self.r_alpha * r_sample
                self.R = self.r

                # Update Q from state increments (random-walk: Q ≈ cov(Δbeta))
                if self.last_beta_upd is not None:
                    delta = beta_upd - self.last_beta_upd
                    q_sample = float(np.mean(delta * delta))
                    q_sample = max(1e-16, q_sample)
                    q_sample = min(max(q_sample, self.q_min), self.q_max)
                    self.q = (1.0 - self.q_alpha) * self.q + self.q_alpha * q_sample
                    self.Q = np.eye(self.n) * self.q
                self.last_beta_upd = beta_upd.copy()
        else:
            beta_upd = beta_pred.copy()
            P_upd = P_pred.copy()
            if self.last_beta_upd is None:
                self.last_beta_upd = beta_upd.copy()

        if update and np.isfinite(y_t):
            if innovation is None:
                innovation = float(y_t - pred_for_update)
            self.innovations.append(float(innovation))

        return beta_pred, P_pred, y_pred, beta_upd, P_upd

    def _project_state(self, beta_vec, cov_mat):
        """Project state to satisfy sign constraints using Mahalanobis metric."""
        active = []
        seen = set()
        for idx in self._nonneg_idx:
            if beta_vec[idx] < 0 and idx not in seen:
                active.append(idx)
                seen.add(idx)
        for idx in self._nonpos_idx:
            if beta_vec[idx] > 0 and idx not in seen:
                active.append(idx)
                seen.add(idx)
        if not active:
            return
        active = np.array(active, dtype=int)
        k = active.shape[0]
        if k == 0:
            return

        P_cc = cov_mat[np.ix_(active, active)]
        jitter = 1e-9
        I_k = np.eye(k)
        for _ in range(6):
            try:
                reg = P_cc + jitter * I_k
                solve_beta = np.linalg.solve(reg, beta_vec[active])
                solve_cov = np.linalg.solve(reg, cov_mat[active, :])
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            reg = P_cc + jitter * I_k
            solve_beta = np.linalg.pinv(reg) @ beta_vec[active]
            solve_cov = np.linalg.pinv(reg) @ cov_mat[active, :]

        beta_vec -= cov_mat[:, active] @ solve_beta
        beta_vec[active] = 0.0

        cov_mat -= cov_mat[:, active] @ solve_cov
        cov_mat[:] = 0.5 * (cov_mat + cov_mat.T)

    def fit(self, X_train, y_train, feature_names=None):
        T, p = X_train.shape
        betas_filt = np.zeros((T, p))
        y_pred = np.zeros(T)
        
        # Track Q and R over time if adaptive
        if self.adaptive:
            self.q_history = []
            self.r_history = []
        
        if self.use_log:
            # Use log1p for numerical stability and match inverse expm1
            y_train_transformed = np.log1p(y_train)
        else:
            y_train_transformed = y_train
        
        self.last_beta_upd = None
        self.innovations = []
        beta = self.beta0.copy()

        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = getattr(self, "feature_names", [f"feature_{i}" for i in range(p)])
        self._nonneg_idx = [i for i, name in enumerate(self.feature_names) if name in self.non_negative_features]
        self._nonpos_idx = [i for i, name in enumerate(self.feature_names) if name in self.non_positive_features]

        has_constraints = bool(self.non_negative_features or self.non_positive_features)

        if self.use_ridge_init:
            try:
                if has_constraints:
                    names = self.feature_names
                    intercept_first = bool(names) and names[0].lower() == 'intercept'
                    if intercept_first:
                        X_solver = X_train[:, 1:]
                        solver_names = names[1:]
                    else:
                        X_solver = X_train
                        solver_names = names

                    beta = self.beta0.copy()
                    if X_solver.shape[1] > 0:
                        solver = CustomConstrainedRidge(
                            l2_penalty=self.ridge_alpha,
                            non_negative_features=self.non_negative_features,
                            non_positive_features=self.non_positive_features
                        )
                        solver.fit(X_solver, y_train_transformed, solver_names)
                        if intercept_first:
                            beta[0] = solver.intercept_
                            beta[1:1 + len(solver.coef_)] = solver.coef_
                        else:
                            beta[:len(solver.coef_)] = solver.coef_
                            beta[0] = solver.intercept_
                    else:
                        beta[0] = float(np.mean(y_train_transformed))
                else:
                    ridge = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
                    ridge.fit(X_train, y_train_transformed)
                    coef = ridge.coef_.astype(float)
                    if coef.shape[0] == self.n:
                        beta = coef
                    else:
                        beta[: len(coef)] = coef
            except Exception:
                beta = self.beta0.copy()

        if not np.isfinite(beta).all():
            beta = self.beta0.copy()
        P = self.P0.copy()
        
        for t in range(T):
            _, _, yhat, beta_upd, P_upd = self._step(
                X_train[t], y_train_transformed[t], beta, P, update=True
            )
            betas_filt[t] = beta_upd
            y_pred[t] = yhat
            beta = beta_upd
            P = P_upd
            
            # Track Q and R values
            if self.adaptive:
                self.q_history.append(self.q)
                self.r_history.append(self.r)
        
        return betas_filt, y_pred


def _update_constraint_selection(key, values, allowed):
    """Helper to overwrite constraint session state safely."""
    allowed_set = set(allowed or [])
    filtered = [col for col in values if col in allowed_set]
    st.session_state[key] = sorted(filtered)


def _add_constraint_matches(key, matches, allowed):
    if not matches:
        return
    allowed_set = set(allowed or [])
    current = [col for col in st.session_state.get(key, []) if col in allowed_set]
    updated = sorted(set(current).union([m for m in matches if m in allowed_set]))
    st.session_state[key] = updated


def _remove_constraint_matches(key, matches, allowed):
    if not matches:
        return
    allowed_set = set(allowed or [])
    current = [col for col in st.session_state.get(key, []) if col in allowed_set]
    updated = [col for col in current if col not in matches]
    st.session_state[key] = updated


@st.cache_data
def load_data(uploaded_file):
    """Load Excel file."""
    return pd.read_excel(uploaded_file)


def build_design_matrix(df, feature_cols, standardize=True):
    """Build design matrix with intercept. Intercept is NOT standardized."""
    Xp = df[feature_cols].values.astype(float)
    scaler = None
    if standardize:
        scaler = StandardScaler()
        Xp = scaler.fit_transform(Xp)
    ones = np.ones((Xp.shape[0], 1), dtype=float)  # unscaled intercept
    X = np.hstack([ones, Xp])
    cols_with_intercept = ['Intercept'] + feature_cols
    return X, scaler, cols_with_intercept


def forward_selection(data, all_vars, y_var, max_vars, mandatory_vars, q, r, init_cov, use_log,
                      non_negative_constraints=None, non_positive_constraints=None):
    """Forward stepwise selection."""
    # Sort by date if available
    if 'Date' in data.columns:
        data = data.copy()
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data = data.sort_values('Date')
    
    mandatory_vars_filtered = [v for v in mandatory_vars if v in data.columns and v in all_vars]
    
    selected_vars = mandatory_vars_filtered.copy()
    remaining_vars = [v for v in all_vars if v not in mandatory_vars_filtered]
    num_to_select = max_vars - len(mandatory_vars_filtered)
    
    selection_history = []
    y = data[y_var].values
    y_max = np.max(y)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for step in range(num_to_select):
        status_text.text(f"Step {step + 1}/{num_to_select}: Testing {len(remaining_vars)} variables...")
        
        best_var = None
        best_r2 = -np.inf
        best_mae = np.inf
        
        for var in remaining_vars:
            test_vars = selected_vars + [var]
            
            try:
                X_test, _, cols_with_intercept = build_design_matrix(data, test_vars, standardize=True)
                
                kf = ConstrainedTVLinearKalman(
                    n_features=X_test.shape[1], q=q, r=r, init_cov=init_cov,
                    min_pred=0, max_pred=y_max*3, use_log=use_log,
                    non_negative_features=non_negative_constraints,
                    non_positive_features=non_positive_constraints
                )
                _, y_pred = kf.fit(X_test, y, feature_names=cols_with_intercept)
                
                r2 = r2_score(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_mae = mae
                    best_var = var
            except:
                continue
        
        if best_var is None:
            break
        
        selected_vars.append(best_var)
        remaining_vars.remove(best_var)
        
        selection_history.append({
            'Step': step + 1,
            'Variable': best_var,
            'R2': best_r2,
            'MAE': best_mae
        })
        
        progress_bar.progress((step + 1) / num_to_select)
    
    status_text.text("Selection complete!")
    progress_bar.empty()
    status_text.empty()
    
    return selected_vars, pd.DataFrame(selection_history)


def select_ridge_alpha(X, y, feature_names, alphas=None,
                       non_negative_features=None, non_positive_features=None):
    """Select ridge alpha using quick in-sample evaluation."""
    if alphas is None:
        alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_alpha = alphas[0]
    best_r2 = -np.inf
    non_negative_features = set(non_negative_features or [])
    non_positive_features = set(non_positive_features or [])
    feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
    intercept_first = bool(feature_names) and feature_names[0].lower() == 'intercept'

    for alpha in alphas:
        try:
            if non_negative_features or non_positive_features:
                if intercept_first:
                    X_solver = X[:, 1:]
                    solver_names = feature_names[1:]
                else:
                    X_solver = X
                    solver_names = feature_names

                if X_solver.shape[1] == 0:
                    preds = np.full_like(y, np.mean(y), dtype=float)
                else:
                    solver = CustomConstrainedRidge(
                        l2_penalty=alpha,
                        non_negative_features=non_negative_features,
                        non_positive_features=non_positive_features
                    )
                    solver.fit(X_solver, y, solver_names)
                    beta = np.zeros(X.shape[1])
                    if intercept_first:
                        beta[0] = solver.intercept_
                        beta[1:1 + len(solver.coef_)] = solver.coef_
                    else:
                        beta[:len(solver.coef_)] = solver.coef_
                        beta[0] = solver.intercept_
                    preds = X @ beta
            else:
                ridge = Ridge(alpha=alpha, fit_intercept=False)
                ridge.fit(X, y)
                preds = ridge.predict(X)
        except Exception:
            continue

        r2 = r2_score(y, preds)
        if r2 > best_r2:
            best_r2 = r2
            best_alpha = alpha

    return best_alpha


def hyperparameter_tuning(X, y, n_features, use_log, ridge_alpha,
                          feature_names=None, non_negative_features=None, non_positive_features=None):
    """Quick hyperparameter search."""
    q_values = [1e-5, 1e-4, 1e-3, 1e-2]
    r_values = [0.1, 1.0, 10.0, 100.0]
    init_cov_values = [1e2, 1e3]
    
    best_r2 = -np.inf
    best_params = None
    y_max = np.max(y)
    
    progress_bar = st.progress(0)
    total = len(q_values) * len(r_values) * len(init_cov_values)
    current = 0
    
    for q, r, init_cov in product(q_values, r_values, init_cov_values):
        current += 1
        try:
            kf = ConstrainedTVLinearKalman(
                n_features=n_features, q=q, r=r, init_cov=init_cov,
                min_pred=0, max_pred=y_max*3, use_log=use_log,
                ridge_alpha=ridge_alpha,
                non_negative_features=non_negative_features,
                non_positive_features=non_positive_features
            )
            _, y_pred = kf.fit(X, y, feature_names=feature_names)
            r2 = r2_score(y, y_pred)
            
            if r2 > best_r2:
                best_r2 = r2
                best_params = (q, r, init_cov)
        except:
            continue
        
        progress_bar.progress(current / total)
    
    progress_bar.empty()
    return best_params


def run_model_for_product(df, product_name, target_var, all_vars, mandatory_vars, 
                          max_vars, standardize, auto_tune, adaptive_qr=True, q_val=1e-4, r_val=1.0,
                          ridge_alpha_val=1.0, non_negative_constraints=None, non_positive_constraints=None,
                          forward_selection_enabled=True, manual_vars=None):
    """Run model for a single product and return results."""
    try:
        # Filter data
        data = df[df['Product title'] == product_name].copy()
        
        if len(data) < 5:  # Need minimum data points
            return None
        
        # Sort by date
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.sort_values('Date')

        manual_vars = list(manual_vars or [])
        non_negative_constraints = set(non_negative_constraints or [])
        non_positive_constraints = set(non_positive_constraints or [])
        candidate_order = list(dict.fromkeys(mandatory_vars + all_vars + manual_vars))
        available_columns = set(data.columns)
        candidate_vars = [col for col in candidate_order if col in available_columns]

        # Clean data
        data = data.dropna(subset=[target_var] + candidate_vars)
        
        if len(data) < 5:
            return None

        y = data[target_var].values
        y_max = np.max(y)

        if forward_selection_enabled:
            # Forward selection (without progress bars for batch mode)
            selected_vars = mandatory_vars.copy()
            remaining_vars = [v for v in all_vars if v not in selected_vars]
            num_to_select = max_vars - len(selected_vars)

            for _ in range(num_to_select):
                best_var = None
                best_r2 = -np.inf

                for var in remaining_vars:
                    test_vars = selected_vars + [var]

                    try:
                        X_test, _, cols_with_intercept = build_design_matrix(data, test_vars, standardize=True)

                        kf = ConstrainedTVLinearKalman(
                            n_features=X_test.shape[1], q=1e-4, r=1.0, init_cov=1e3,
                            min_pred=0, max_pred=y_max * 3, use_log=False,
                            non_negative_features=non_negative_constraints,
                            non_positive_features=non_positive_constraints
                        )
                        _, y_pred = kf.fit(X_test, y, feature_names=cols_with_intercept)

                        r2 = r2_score(y, y_pred)

                        if r2 > best_r2:
                            best_r2 = r2
                            best_var = var
                    except Exception:
                        continue

                if best_var is None:
                    break

                selected_vars.append(best_var)
                remaining_vars.remove(best_var)
        else:
            # Manual path: use user-provided predictors plus mandatory ones
            selected_vars = [v for v in manual_vars if v in data.columns]
            for mandatory in mandatory_vars:
                if mandatory in data.columns and mandatory not in selected_vars:
                    selected_vars.append(mandatory)

            # Ensure at least one predictor is available
            if not selected_vars:
                return None
        
        # Prepare final data with intercept
        X, scaler, cols_with_intercept = build_design_matrix(data, selected_vars, standardize=standardize)
        y_for_ridge = y  # use_log is False in batch mode
        ridge_alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
        ridge_alpha_auto = select_ridge_alpha(
            X, y, cols_with_intercept,
            alphas=ridge_alpha_candidates,
            non_negative_features=non_negative_constraints,
            non_positive_features=non_positive_constraints
        )
        
        # Hyperparameter tuning
        if auto_tune:
            ridge_alpha_opt = ridge_alpha_auto
            best_params = hyperparameter_tuning(
                X, y, X.shape[1], False, ridge_alpha_opt,
                feature_names=cols_with_intercept,
                non_negative_features=non_negative_constraints,
                non_positive_features=non_positive_constraints
            )
            if best_params:
                q_opt, r_opt, init_cov_opt = best_params
            else:
                q_opt, r_opt, init_cov_opt = 1e-4, 1.0, 1e3
        else:
            ridge_alpha_opt = ridge_alpha_val
            q_opt, r_opt, init_cov_opt = q_val, r_val, 1e3
        
        # Run final model
        kf = ConstrainedTVLinearKalman(
            n_features=X.shape[1], q=q_opt, r=r_opt, init_cov=init_cov_opt,
            min_pred=0, max_pred=np.max(y)*3, use_log=False, adaptive=adaptive_qr,
            ridge_alpha=ridge_alpha_opt,
            non_negative_features=non_negative_constraints,
            non_positive_features=non_positive_constraints
        )
        betas, y_pred = kf.fit(X, y, feature_names=cols_with_intercept)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mask = y != 0
        mape = np.mean(np.abs((y[mask] - y_pred[mask]) / y[mask])) * 100 if mask.sum() > 0 else np.nan
        
        # Get dates
        dates = pd.to_datetime(data['Date']) if 'Date' in data.columns else np.arange(len(data))
        
        return {
            'Product': product_name,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Variables': len(selected_vars),
            'Data Points': len(data),
            'Selected Variables': selected_vars,
            # Detailed data for tabs
            'data': data,
            'X': X,
            'y': y,
            'y_pred': y_pred,
            'betas': betas,
            'dates': dates,
            'q_opt': q_opt,
            'r_opt': r_opt,
            'ridge_alpha_opt': ridge_alpha_opt,
            'non_negative_constraints': sorted(non_negative_constraints),
            'non_positive_constraints': sorted(non_positive_constraints),
            'mandatory_vars': mandatory_vars
        }
    
    except Exception as e:
        return None


def build_coefficients_table(results_subset):
    """Return a wide dataframe with metrics + final betas per product."""
    if not results_subset:
        return None

    try:
        union_vars = sorted(
            list(set().union(*[set(['Intercept'] + r.get('Selected Variables', [])) for r in results_subset]))
        )
    except Exception:
        return None

    rows = []
    for r in results_subset:
        try:
            names = ['Intercept'] + r.get('Selected Variables', [])
            final_betas = r['betas'][-1, :]
            usable = min(len(names), final_betas.shape[0])
            name_to_beta = {names[i]: float(final_betas[i]) for i in range(usable)}
        except Exception:
            continue

        row = {
            'Product': r.get('Product'),
            'R2': float(r.get('R2', np.nan)),
            'MAE': float(r.get('MAE', np.nan)),
            'RMSE': float(r.get('RMSE', np.nan)),
            'MAPE': float(r.get('MAPE', np.nan)) if r.get('MAPE') is not None else float('nan'),
        }
        for v in union_vars:
            row[v] = name_to_beta.get(v, 0.0)
        rows.append(row)

    if not rows:
        return None

    coeffs_df = pd.DataFrame(rows, columns=['Product', 'R2', 'MAE', 'RMSE', 'MAPE'] + union_vars)
    coeffs_df['R2'] = coeffs_df['R2'].round(4)
    coeffs_df['MAE'] = coeffs_df['MAE'].round(2)
    coeffs_df['RMSE'] = coeffs_df['RMSE'].round(2)
    coeffs_df['MAPE'] = coeffs_df['MAPE'].round(2)
    return coeffs_df


def main():
    # Header
    st.markdown('<p class="main-header">📊 Kalman Filter Time-Series Analysis</p>', unsafe_allow_html=True)
    
    # Configuration on main view
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1rem !important; padding-bottom: 1rem !important;}
            section + section {margin-top: 0.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "non_neg_multiselect" not in st.session_state:
        st.session_state["non_neg_multiselect"] = []
    if "non_pos_multiselect" not in st.session_state:
        st.session_state["non_pos_multiselect"] = []
    if "constraint_filter" not in st.session_state:
        st.session_state["constraint_filter"] = ""
    if "manual_predictor_multiselect" not in st.session_state:
        st.session_state["manual_predictor_multiselect"] = []

    with st.container():
        st.markdown("#### Step 1 · Upload & inspect data")
        upload_col, summary_col = st.columns([3, 2])
        with upload_col:
            uploaded_file = st.file_uploader("📁 Upload Excel File", type=['xlsx', 'xls'], key="main_uploader")
        if uploaded_file is None:
            st.info("👆 Please upload an Excel file to begin")
            return

        with st.spinner("Loading data..."):
            df = load_data(uploaded_file)
        with summary_col:
            st.success("✓ File uploaded!")
            st.metric("Rows", f"{df.shape[0]:,}")
            st.metric("Columns", f"{df.shape[1]:,}")

        if 'Product title' not in df.columns:
            st.error("'Product title' column not found!")
            return
        products = df['Product title'].unique()

    st.divider()
    scope_col, variants_col = st.columns([1.25, 0.75], gap="large")
    with scope_col:
        st.markdown("#### Step 2 · Scope & target")
        run_all_products = st.checkbox("🔄 Run for All Products", value=False,
                                       help="Run model for all products and show summary", key="run_all_products")
        if run_all_products:
            selected_product = None
            st.info(f"Will run for {len(products)} products")
        else:
            selected_product = st.selectbox("🎯 Select Product", products, key="single_product_selector")
        target_var = st.selectbox("📈 Target Variable",
                                  ['Net items sold', 'Net Items Sold'] + list(df.columns),
                                  index=0, key="target_variable_selector")
        max_vars = st.slider("Max Variables", 5, 20, 10, key="max_variables_slider")
        if run_all_products:
            r2_threshold = st.slider("R² Threshold (%)", 0, 100, 70, 5,
                                     help="Show products with R² above this threshold",
                                     key="r2_threshold_slider") / 100
        else:
            r2_threshold = 0.70

    partial_end_date = None
    with variants_col:
        st.markdown("#### Step 3 · Model variants")
        run_dual_model = st.checkbox("📊 Run Dual Models", value=False,
                                     help="Run both full period model AND partial period model for comparison",
                                     key="dual_model_checkbox")
        if run_dual_model and not run_all_products and 'Date' in df.columns and selected_product:
            product_data = df[df['Product title'] == selected_product].copy()
            if len(product_data) > 0:
                dates_temp = pd.to_datetime(product_data['Date'])
                min_date = dates_temp.min().date()
                max_date = dates_temp.max().date()
                st.caption(f"Partial model window: {min_date} → {max_date}")
                partial_end_date = st.date_input(
                    "Partial Model End Date",
                    value=min_date + (max_date - min_date) * 0.7,
                    min_value=min_date,
                    max_value=max_date,
                    help="Select the end date for the partial model",
                    key="partial_model_end"
                )

    st.divider()
    features_col, constraints_col = st.columns([1.05, 0.95], gap="large")
    with features_col:
        st.markdown("#### Step 4 · Feature controls")
        mandatory_cols = st.columns(3)
        with mandatory_cols[0]:
            use_google_trends = st.checkbox("google_trends", value=True, key="mandatory_google")
        with mandatory_cols[1]:
            use_price = st.checkbox("Product variant price", value=True, key="mandatory_price")
        with mandatory_cols[2]:
            use_discount = st.checkbox("Category Discount", value=True, key="mandatory_discount")

        mandatory_vars = []
        if use_google_trends:
            mandatory_vars.append("google_trends")
        if use_price:
            mandatory_vars.append("Product variant price")
        if use_discount:
            mandatory_vars.append("Category Discount")

        run_forward_selection = st.checkbox(
            "Use forward selection (slower)",
            value=True,
            help="Disable to skip automatic forward selection and choose predictors manually.",
            key="forward_selection_toggle"
        )
        manual_predictor_choices = []
        if not run_forward_selection:
            feature_candidates = sorted(
                [
                    col for col in df.columns
                    if col not in {target_var, "Date", "Product title"}
                ]
            )
            st.session_state["manual_predictor_multiselect"] = [
                col for col in st.session_state.get("manual_predictor_multiselect", []) if col in feature_candidates
            ]
            if not st.session_state["manual_predictor_multiselect"]:
                default_manual = [col for col in mandatory_vars if col in feature_candidates]
                st.session_state["manual_predictor_multiselect"] = default_manual
            manual_predictor_choices = st.multiselect(
                "Select predictors when forward selection is disabled",
                options=feature_candidates,
                default=None,
                key="manual_predictor_multiselect",
                help="These columns feed directly into the Kalman model. Mandatory variables are always included."
            )
            if not manual_predictor_choices:
                st.info("No manual predictors selected. Mandatory variables will still be used.")

    with constraints_col:
        st.markdown("#### Step 5 · Coefficient sign constraints")
        exclude_for_constraints = {target_var, 'Date', 'Product title'}
        available_constraint_vars = sorted([col for col in df.columns if col not in exclude_for_constraints])
        _update_constraint_selection(
            "non_neg_multiselect",
            st.session_state.get("non_neg_multiselect", []),
            available_constraint_vars,
        )
        _update_constraint_selection(
            "non_pos_multiselect",
            st.session_state.get("non_pos_multiselect", []),
            available_constraint_vars,
        )

        constraint_filter = st.text_input(
            "Filter columns (case-insensitive)",
            value=st.session_state.get("constraint_filter", ""),
            key="constraint_filter",
            placeholder="Type e.g. impression"
        )
        matching_cols = []
        if constraint_filter:
            key_lower = constraint_filter.lower()
            matching_cols = [col for col in available_constraint_vars if key_lower in col.lower()]
            st.caption(f"{len(matching_cols)} columns match '{constraint_filter}'")

        constraint_cols = st.columns(2)
        with constraint_cols[0]:
            st.write("Force coefficients ≥ 0")
            st.multiselect(
                "Select columns for non-negative constraint",
                options=available_constraint_vars,
                default=None,
                key="non_neg_multiselect",
                label_visibility="collapsed"
            )
            if matching_cols:
                add_col, remove_col = st.columns(2)
                add_col.button(
                    "Add matches",
                    key="add_nonneg_matches",
                    on_click=_add_constraint_matches,
                    args=("non_neg_multiselect", tuple(matching_cols), tuple(available_constraint_vars)),
                )
                remove_col.button(
                    "Remove matches",
                    key="remove_nonneg_matches",
                    on_click=_remove_constraint_matches,
                    args=("non_neg_multiselect", tuple(matching_cols), tuple(available_constraint_vars)),
                )
            st.caption(f"Selected: {len(st.session_state['non_neg_multiselect'])}")

        with constraint_cols[1]:
            st.write("Force coefficients ≤ 0")
            st.multiselect(
                "Select columns for non-positive constraint",
                options=available_constraint_vars,
                default=None,
                key="non_pos_multiselect",
                label_visibility="collapsed"
            )
            if matching_cols:
                add_col, remove_col = st.columns(2)
                add_col.button(
                    "Add matches",
                    key="add_nonpos_matches",
                    on_click=_add_constraint_matches,
                    args=("non_pos_multiselect", tuple(matching_cols), tuple(available_constraint_vars)),
                )
                remove_col.button(
                    "Remove matches",
                    key="remove_nonpos_matches",
                    on_click=_remove_constraint_matches,
                    args=("non_pos_multiselect", tuple(matching_cols), tuple(available_constraint_vars)),
                )
            st.caption(f"Selected: {len(st.session_state['non_pos_multiselect'])}")

        non_negative_constraints = set(st.session_state["non_neg_multiselect"])
        non_positive_constraints = set(st.session_state["non_pos_multiselect"])
        overlap_constraints = non_negative_constraints & non_positive_constraints
        if overlap_constraints:
            st.warning(f"Constraints overlap for: {', '.join(sorted(overlap_constraints))}. Removing them from the non-positive list.")
            non_positive_constraints -= overlap_constraints
            st.session_state["non_pos_multiselect"] = sorted(non_positive_constraints)

    st.divider()
    with st.container():
        st.markdown("#### Step 6 · Hyperparameters & execution")
        hyper_left, hyper_right = st.columns(2)
        use_log = False
        with hyper_left:
            standardize = st.checkbox("Standardize Predictors", value=True, key="standardize_checkbox")
            adaptive_qr = st.checkbox("🔄 Adaptive Q & R", value=True,
                                      help="Q and R adapt over time based on model performance",
                                      key="adaptive_qr_checkbox")
            auto_tune = st.checkbox("Auto-tune Hyperparameters", value=True,
                                    help="Find optimal initial Q, R, and Ridge alpha values",
                                    key="auto_tune_checkbox")
        with hyper_right:
            ridge_alpha_val = 1.0
            if not auto_tune:
                st.write("Manual Hyperparameters")
                q_val = st.select_slider("Q (Process Noise)",
                                         options=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
                                         value=1e-4,
                                         key="manual_q_slider")
                r_val = st.select_slider("R (Observation Noise)",
                                         options=[0.1, 1.0, 10.0, 100.0],
                                         value=1.0,
                                         key="manual_r_slider")
                ridge_alpha_val = st.select_slider("Ridge Alpha",
                                                   options=[0.01, 0.1, 1.0, 10.0, 100.0],
                                                   value=1.0,
                                                   key="manual_ridge_slider")
            else:
                q_val = 1e-4
                r_val = 1.0

        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    # Main area
    if uploaded_file is not None:
        
        # Initialize session state for results
        if 'all_products_results' not in st.session_state:
            st.session_state.all_products_results = None
        if 'single_product_results' not in st.session_state:
            st.session_state.single_product_results = None
        
        # Run analysis when button clicked
        if run_analysis:
            
            # Check if running for all products
            if run_all_products:
                st.header("🔄 Running Analysis for All Products")
                
                # Get all products
                products = df['Product title'].unique()
                
                # Get variables
                all_impression_cols = [col for col in df.columns if 'impression' in col.lower()]
                mandatory_available = [v for v in mandatory_vars if v in df.columns]
                if run_forward_selection:
                    feature_pool = all_impression_cols
                else:
                    feature_pool = [col for col in manual_predictor_choices if col in df.columns]
                all_vars = list(dict.fromkeys(feature_pool + mandatory_available))
                
                # Run for each product
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, product in enumerate(products):
                    status_text.text(f"Processing {i+1}/{len(products)}: {product[:50]}...")
                    
                    result = run_model_for_product(
                        df, product, target_var, all_vars, mandatory_available,
                        max_vars, standardize, auto_tune,
                        adaptive_qr=adaptive_qr,
                        q_val=q_val if not auto_tune else 1e-4,
                        r_val=r_val if not auto_tune else 1.0,
                        ridge_alpha_val=ridge_alpha_val if not auto_tune else 1.0,
                        non_negative_constraints=non_negative_constraints,
                        non_positive_constraints=non_positive_constraints,
                        forward_selection_enabled=run_forward_selection,
                        manual_vars=manual_predictor_choices
                    )
                    
                    if result:
                        results.append(result)
                    
                    progress_bar.progress((i + 1) / len(products))
                
                progress_bar.empty()
                status_text.empty()
                
                # Store results in session state
                st.session_state.all_products_results = results
                
                # Show summary
                st.success(f"✓ Completed analysis for {len(results)} products")
        
        # Display results (whether just run or from session state)
        if run_all_products and st.session_state.all_products_results is not None:
            results = st.session_state.all_products_results
            
            # Filter high R2 products
            high_r2_results = [r for r in results if r['R2'] > r2_threshold]
            
            st.header(f"🎯 Products with R² > {r2_threshold:.2f} ({len(high_r2_results)} found)")
            
            if len(high_r2_results) > 0:
                # Create summary dataframe
                summary_data = [{
                    'Product': r['Product'],
                    'R2': r['R2'],
                    'MAE': r['MAE'],
                    'RMSE': r['RMSE'],
                    'MAPE': r['MAPE'],
                    'Variables': r['Variables'],
                    'Data Points': r['Data Points']
                } for r in high_r2_results]
                
                summary_df = pd.DataFrame(summary_data)
                summary_df['R2'] = summary_df['R2'].round(4)
                summary_df['MAE'] = summary_df['MAE'].round(2)
                summary_df['RMSE'] = summary_df['RMSE'].round(2)
                summary_df['MAPE'] = summary_df['MAPE'].round(2)
                
                st.dataframe(summary_df, use_container_width=True, height=400)
                
                # Show distribution chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=summary_df['Product'],
                    y=summary_df['R2'],
                    marker=dict(color=summary_df['R2'], colorscale='Viridis', showscale=True)
                ))
                fig.update_layout(
                    title="R² Scores for High-Performing Products",
                    xaxis_title="Product",
                    yaxis_title="R² Score",
                    height=500,
                    xaxis_tickangle=-45
                )
                st.plotly_chart(fig, use_container_width=True, key="batch_r2_bar_chart")
                # Final coefficients (last period) for high R� products
                coeffs_df = build_coefficients_table(high_r2_results)
                if coeffs_df is not None:
                    st.subheader("Final Coefficients (last period)")
                    st.dataframe(coeffs_df, use_container_width=True, height=400)
                    # Provide downloadable summary so each row = product and columns = metrics + betas
                    coeffs_output = io.BytesIO()
                    with pd.ExcelWriter(coeffs_output, engine='openpyxl') as writer:
                        coeffs_df.to_excel(writer, sheet_name='Coefficients & Metrics', index=False)
                    coeffs_output.seek(0)
                    st.download_button(
                        label="📥 Download Metrics + Final Betas",
                        data=coeffs_output,
                        file_name="kalman_product_coefficients.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True,
                        help="Each row corresponds to a product; columns contain metrics and final-period betas (0 if feature not used)."
                    )
                
                st.divider()
            
            all_coeffs_df = build_coefficients_table(results)
            if all_coeffs_df is not None:
                st.subheader("All Products · Metrics + Final Betas")
                st.dataframe(all_coeffs_df, use_container_width=True, height=400)
                all_coeffs_output = io.BytesIO()
                with pd.ExcelWriter(all_coeffs_output, engine='openpyxl') as writer:
                    all_coeffs_df.to_excel(writer, sheet_name='All Products', index=False)
                all_coeffs_output.seek(0)
                st.download_button(
                    label="📥 Download All Products Betas",
                    data=all_coeffs_output,
                    file_name="kalman_all_product_coefficients.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    help="Wide format table covering every product that was modeled."
                )
                
                st.divider()
                
                # Product selector for detailed view
                st.header("📊 Detailed Analysis by Product")
                
                product_names = [r['Product'] for r in high_r2_results]
                selected_detail_product = st.selectbox(
                    "Select a product to view detailed analysis:",
                    product_names,
                    key="detail_product_selector"
                )
                
                # Get selected product data
                selected_result = next(r for r in high_r2_results if r['Product'] == selected_detail_product)
                
                # Extract data
                y = selected_result['y']
                y_pred = selected_result['y_pred']
                betas = selected_result['betas']
                dates = selected_result['dates']
                selected_vars = selected_result['Selected Variables']
                mandatory_vars = selected_result['mandatory_vars']
                r2 = selected_result['R2']
                mae = selected_result['MAE']
                rmse = selected_result['RMSE']
                mape = selected_result['MAPE']
                residuals = y - y_pred
                
                # Show detailed tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📈 Summary", "📉 Predictions", "🎯 Scatter", "📊 Coefficients", "📄 Results"
                ])
                
                with tab1:
                    st.subheader(f"Performance Metrics - {selected_detail_product}")
                    
                    latest_actual = float(y[-1]) if len(y) else float("nan")
                    latest_pred = float(y_pred[-1]) if len(y_pred) else float("nan")
                    delta_latest = (
                        latest_pred - latest_actual
                        if np.isfinite(latest_actual) and np.isfinite(latest_pred)
                        else float("nan")
                    )
                    latest_cols = st.columns(3)
                    with latest_cols[0]:
                        actual_display = f"{latest_actual:,.2f}" if np.isfinite(latest_actual) else "—"
                        st.metric("Latest Actual", actual_display)
                    with latest_cols[1]:
                        pred_display = f"{latest_pred:,.2f}" if np.isfinite(latest_pred) else "—"
                        delta_display = f"{delta_latest:+.2f}" if np.isfinite(delta_latest) else "—"
                        st.metric("Latest Prediction", pred_display, delta=delta_display)
                    with latest_cols[2]:
                        st.metric("Data Points", len(y))
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("MAE", f"{mae:.2f}")
                    with col3:
                        st.metric("RMSE", f"{rmse:.2f}")
                    with col4:
                        st.metric("MAPE", f"{mape:.2f}%")
                    
                    st.divider()
                    
                    # Variable importance
                    st.subheader("📋 Selected Variables")
                    
                    # Add intercept to variable list
                    vars_with_intercept = ['Intercept'] + selected_vars
                    
                    var_df = pd.DataFrame({
                        'Variable': vars_with_intercept,
                        'Final Coefficient': betas[-1, :],
                        'Type': ['📍 Intercept' if v == 'Intercept' else ('🔒 Mandatory' if v in mandatory_vars else '✓ Selected') 
                                for v in vars_with_intercept]
                    })
                    st.dataframe(var_df, use_container_width=True)
                
                with tab2:
                    st.subheader("Time Series: Actual vs Predicted")
                    
                    fig1 = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                                        subplot_titles=('Predictions', 'Residuals'))
                    
                    fig1.add_trace(go.Scatter(x=dates, y=y, name='Actual', 
                                             mode='lines+markers',
                                             line=dict(color='#2E86AB', width=2),
                                             marker=dict(size=8)), row=1, col=1)
                    fig1.add_trace(go.Scatter(x=dates, y=y_pred, name='Predicted', 
                                             mode='lines+markers',
                                             line=dict(color='#A23B72', width=2),
                                             marker=dict(size=8)), row=1, col=1)
                    fig1.add_trace(go.Scatter(x=dates, y=[np.mean(y)]*len(dates), name='Mean', 
                                             mode='lines',
                                             line=dict(color='orange', width=2, dash='dash')), row=1, col=1)
                    
                    fig1.add_trace(go.Scatter(x=dates, y=residuals, mode='markers',
                                             marker=dict(color=['red' if r<0 else 'green' for r in residuals]),
                                             name='Residuals'), row=2, col=1)
                    fig1.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
                    
                    fig1.update_xaxes(title_text="Date", row=2, col=1)
                    fig1.update_yaxes(title_text=target_var, row=1, col=1)
                    fig1.update_yaxes(title_text="Residual", row=2, col=1)
                    fig1.update_layout(height=800, showlegend=True)
                    
                    st.plotly_chart(fig1, use_container_width=True, key="batch_predictions_chart")
                
                with tab3:
                    st.subheader("Scatter: Actual vs Predicted")
                    
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(x=y, y=y_pred, mode='markers',
                                             marker=dict(size=10, color=y, colorscale='Viridis',
                                                       showscale=True)))
                    
                    min_val, max_val = 0, max(y.max(), y_pred.max())
                    fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                             mode='lines', line=dict(color='red', dash='dash', width=2),
                                             name='Perfect'))
                    
                    fig2.update_layout(
                        title=f"Actual vs Predicted (R² = {r2:.4f})",
                        xaxis_title=f"Actual {target_var}",
                        yaxis_title=f"Predicted {target_var}",
                        height=600
                    )
                    
                    st.plotly_chart(fig2, use_container_width=True, key="batch_scatter_chart")
                
                with tab4:
                    st.subheader("Coefficient Evolution Over Time")
                    
                    # Include intercept in display - show ALL variables
                    vars_with_intercept = ['Intercept'] + selected_vars
                    n_vars = len(vars_with_intercept)
                    
                    fig3 = go.Figure()
                    
                    for var in vars_with_intercept:
                        idx = vars_with_intercept.index(var)
                        fig3.add_trace(go.Scatter(
                            y=betas[:, idx], 
                            mode='lines+markers',
                            line=dict(width=2),
                            marker=dict(size=6),
                            name=var[:40] + "..." if len(var) > 40 else var
                        ))
                    
                    fig3.add_hline(y=0, line_dash="dash", line_color="gray")
                    fig3.update_layout(
                        title=f"Coefficient Evolution Over Time ({n_vars} variables)",
                        xaxis_title="Time Index",
                        yaxis_title="Coefficient Value",
                        height=600,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=1.01
                        )
                    )
                    
                    st.plotly_chart(fig3, use_container_width=True, key="batch_coefficients_chart")
                
                with tab5:
                    st.subheader("📄 Results Table & Download")

                    # Create results dataframe for this product
                    product_data = selected_result['data'].copy()
                    product_data['Predicted'] = y_pred
                    product_data['Residuals'] = residuals

                    # Add coefficients (including intercept)
                    vars_with_intercept = ['Intercept'] + selected_vars
                    for i, var in enumerate(vars_with_intercept):
                        product_data[f'Beta_{var}'] = betas[:, i]

                    metrics_summary = {
                        'R2': float(r2),
                        'MAE': float(mae),
                        'RMSE': float(rmse),
                        'MAPE': float(mape) if mape is not None else float('nan')
                    }
                    for k, v in metrics_summary.items():
                        product_data[k] = v

                    beta_columns = [f'Beta_{var}' for var in vars_with_intercept]
                    static_cols = []
                    if 'Date' in product_data.columns:
                        static_cols.append('Date')
                    if 'Product title' in product_data.columns:
                        static_cols.append('Product title')
                    ordered_cols = static_cols + beta_columns + list(metrics_summary.keys())
                    remaining_cols = [col for col in product_data.columns if col not in ordered_cols]
                    product_data = product_data[ordered_cols + remaining_cols]

                    st.dataframe(product_data, use_container_width=True)

                    # Convert to Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        product_data.to_excel(writer, sheet_name='Results', index=False)
                        
                        metrics_df = pd.DataFrame({
                            'Metric': ['R²', 'MAE', 'RMSE', 'MAPE'],
                            'Value': [r2, mae, rmse, mape]
                        })
                        metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                        
                        var_df.to_excel(writer, sheet_name='Variables', index=False)
                    
                    output.seek(0)
                    
                    st.download_button(
                        label=f"📥 Download Results for {selected_detail_product}",
                        data=output,
                        file_name=f"kalman_{selected_detail_product.replace(' ', '_')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )
                
                st.divider()
                
                # Download all results
                st.subheader("💾 Download All Products Summary")
                output_all = io.BytesIO()
                with pd.ExcelWriter(output_all, engine='openpyxl') as writer:
                    # All products summary
                    all_summary = pd.DataFrame([{
                        'Product': r['Product'],
                        'R2': r['R2'],
                        'MAE': r['MAE'],
                        'RMSE': r['RMSE'],
                        'MAPE': r['MAPE'],
                        'Variables': r['Variables'],
                        'Data Points': r['Data Points']
                    } for r in results])
                    all_summary = all_summary.sort_values('R2', ascending=False)
                    all_summary.to_excel(writer, sheet_name='All Products', index=False)
                    
                    # High R2 products
                    summary_df.to_excel(writer, sheet_name=f'High R2 (>{r2_threshold:.2f})', index=False)
                
                output_all.seek(0)
                st.download_button(
                    label="📥 Download All Products Summary",
                    data=output_all,
                    file_name="kalman_all_products_summary.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            else:
                st.warning("No products found with R² > 0.70")
                
                # Show all results anyway
                st.subheader("All Products Results")
                summary_data = [{
                    'Product': r['Product'],
                    'R2': r['R2'],
                    'MAE': r['MAE'],
                    'RMSE': r['RMSE'],
                    'Variables': r['Variables'],
                    'Data Points': r['Data Points']
                } for r in results]
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
        
        elif run_analysis and not run_all_products:
            # Single product mode
            # Filter data
            data = df[df['Product title'] == selected_product].copy()
            
            if len(data) == 0:

                st.error(f"No data found for product: {selected_product}")
                return
            
            st.success(f"✓ Filtered to {len(data)} rows for '{selected_product}'")
            
            # Get variables
            all_impression_cols = [col for col in data.columns if "impression" in col.lower()]
            mandatory_available = [v for v in mandatory_vars if v in data.columns]
            if run_forward_selection:
                feature_pool = all_impression_cols
            else:
                feature_pool = [col for col in manual_predictor_choices if col in data.columns]
            all_vars = list(dict.fromkeys(feature_pool + mandatory_available))
            
            # Clean data
            data = data.dropna(subset=[target_var] + all_vars)
            
            st.info(f"{len(data)} rows after cleaning | {len(all_vars)} candidate variables")
            
            # Calculate partial_idx from the date selected in sidebar
            if run_dual_model and partial_end_date is not None and "Date" in data.columns:
                dates_temp = pd.to_datetime(data["Date"])
                partial_end_date_dt = pd.to_datetime(partial_end_date)
                partial_idx = (dates_temp <= partial_end_date_dt).sum()
                partial_percentage = int((partial_idx / len(data)) * 100)
                
                st.info(f"Partial model will use {partial_idx} rows ({partial_percentage}%) up to {partial_end_date}")
            else:
                partial_idx = None
                partial_percentage = 100
            
            # Feature selection
            st.header("Forward Selection Process")
            
            if run_forward_selection:
                with st.spinner("Running forward selection..."):
                    selected_vars, selection_history = forward_selection(
                        data, all_vars, target_var, max_vars, mandatory_vars,
                        q=1e-4, r=1.0, init_cov=1e3, use_log=use_log,
                        non_negative_constraints=non_negative_constraints,
                        non_positive_constraints=non_positive_constraints
                    )
                st.success(f"Selected {len(selected_vars)} variables")
            else:
                selection_history = pd.DataFrame(columns=["Step", "Variable", "R2", "MAE"])
                selected_vars = [v for v in manual_predictor_choices if v in data.columns]
                for mandatory in mandatory_available:
                    if mandatory in data.columns and mandatory not in selected_vars:
                        selected_vars.append(mandatory)
                selected_vars = list(dict.fromkeys(selected_vars))
                if not selected_vars:
                    st.error("No available predictors to run the model.")
                    return
                st.success(f"Using {len(selected_vars)} variables (manual selection)")
            # Sort by date
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
                data = data.sort_values('Date')
            
            # Prepare data with intercept
            dates = pd.to_datetime(data['Date']) if 'Date' in data.columns else np.arange(len(data))
            X, scaler, selected_vars_with_intercept = build_design_matrix(data, selected_vars, standardize=standardize)
            y = data[target_var].values
            y_for_ridge = np.log1p(y) if use_log else y
            ridge_alpha_candidates = [0.01, 0.1, 1.0, 10.0, 100.0]
            ridge_alpha_auto = select_ridge_alpha(
                X, y_for_ridge, selected_vars_with_intercept,
                alphas=ridge_alpha_candidates,
                non_negative_features=non_negative_constraints,
                non_positive_features=non_positive_constraints
            )

            # Hyperparameter tuning on full data
            if auto_tune:
                st.header("Hyperparameter Tuning")
                with st.spinner("Finding optimal parameters..."):
                    ridge_alpha_opt = ridge_alpha_auto
                    best_params = hyperparameter_tuning(
                        X, y, X.shape[1], use_log, ridge_alpha_opt,
                        feature_names=selected_vars_with_intercept,
                        non_negative_features=non_negative_constraints,
                        non_positive_features=non_positive_constraints
                    )
                    if best_params:
                        q_opt, r_opt, init_cov_opt = best_params
                    else:
                        q_opt, r_opt, init_cov_opt = 1e-4, 1.0, 1e3
                st.success(f"Best: Q={q_opt}, R={r_opt}, Ridge alpha={ridge_alpha_opt}")
            else:
                ridge_alpha_opt = ridge_alpha_val
                q_opt, r_opt, init_cov_opt = q_val, r_val, 1e3
            st.header("⚙️ Running Full Period Model")
            
            with st.spinner("Training Kalman Filter on full data..."):
                kf_full = ConstrainedTVLinearKalman(
                    n_features=X.shape[1], q=q_opt, r=r_opt, init_cov=init_cov_opt,
                    min_pred=0, max_pred=np.max(y)*3, use_log=use_log, adaptive=adaptive_qr,
                    ridge_alpha=ridge_alpha_opt,
                    non_negative_features=non_negative_constraints,
                    non_positive_features=non_positive_constraints
                )
                betas_full, y_pred_full = kf_full.fit(X, y, feature_names=selected_vars_with_intercept)
            
            # Calculate metrics for full model
            mae_full = mean_absolute_error(y, y_pred_full)
            rmse_full = np.sqrt(mean_squared_error(y, y_pred_full))
            r2_full = r2_score(y, y_pred_full)
            mask_full = y != 0
            mape_full = np.mean(np.abs((y[mask_full] - y_pred_full[mask_full]) / y[mask_full])) * 100 if mask_full.sum() > 0 else np.nan
            
            # Run PARTIAL period model if requested
            if run_dual_model and partial_idx is not None:
                st.header("⚙️ Running Partial Period Model")
                
                X_partial = X[:partial_idx]
                y_partial = y[:partial_idx]
                dates_partial = dates[:partial_idx]
                
                st.info(f"📊 Full Model: {len(X)} rows | Partial Model: {len(X_partial)} rows ({partial_percentage}%) up to {partial_end_date}")
                
                with st.spinner(f"Training Kalman Filter on {partial_percentage}% of data..."):
                    kf_partial = ConstrainedTVLinearKalman(
                        n_features=X_partial.shape[1], q=q_opt, r=r_opt, init_cov=init_cov_opt,
                        min_pred=0, max_pred=np.max(y_partial)*3, use_log=use_log, adaptive=adaptive_qr,
                        ridge_alpha=ridge_alpha_opt,
                        non_negative_features=non_negative_constraints,
                        non_positive_features=non_positive_constraints
                    )
                    betas_partial, y_pred_partial = kf_partial.fit(
                        X_partial, y_partial, feature_names=selected_vars_with_intercept
                    )
                
                # Calculate metrics for partial model
                mae_partial = mean_absolute_error(y_partial, y_pred_partial)
                rmse_partial = np.sqrt(mean_squared_error(y_partial, y_pred_partial))
                r2_partial = r2_score(y_partial, y_pred_partial)
                mask_partial = y_partial != 0
                mape_partial = np.mean(np.abs((y_partial[mask_partial] - y_pred_partial[mask_partial]) / y_partial[mask_partial])) * 100 if mask_partial.sum() > 0 else np.nan
            else:
                mae_partial = rmse_partial = r2_partial = mape_partial = None
                betas_partial = y_pred_partial = None
                partial_idx = None
            
            # Use full model metrics as primary
            mae, rmse, r2, mape = mae_full, rmse_full, r2_full, mape_full
            betas, y_pred = betas_full, y_pred_full
            
            # Store results in session state
            st.session_state.single_product_results = {
                'data': data, 'X': X, 'y': y, 'y_pred': y_pred, 'betas': betas,
                'dates': dates, 'residuals': y - y_pred, 'r2': r2, 'mae': mae,
                'rmse': rmse, 'mape': mape, 'selected_vars_with_intercept': selected_vars_with_intercept,
                'kf_full': kf_full, 'adaptive_qr': adaptive_qr, 'q_opt': q_opt, 'r_opt': r_opt,
                'init_cov_opt': init_cov_opt, 'selection_history': selection_history,
                'y_pred_full': y_pred_full, 'y_pred_partial': y_pred_partial,
                'partial_idx': partial_idx, 'partial_percentage': partial_percentage if run_dual_model else None,
                'run_dual_model': run_dual_model, 'mae_full': mae_full, 'rmse_full': rmse_full,
                'r2_full': r2_full, 'mape_full': mape_full, 'mae_partial': mae_partial,
                'rmse_partial': rmse_partial, 'r2_partial': r2_partial, 'mape_partial': mape_partial
            }
        
        # Display results from session state
        if st.session_state.single_product_results is not None and not run_all_products:
            results = st.session_state.single_product_results
            data = results['data']
            X = results['X']
            y = results['y']
            y_pred = results['y_pred']
            betas = results['betas']
            dates = results['dates']
            residuals = results['residuals']
            r2 = results['r2']
            mae = results['mae']
            rmse = results['rmse']
            mape = results['mape']
            selected_vars_with_intercept = results['selected_vars_with_intercept']
            kf_full = results['kf_full']
            adaptive_qr = results['adaptive_qr']
            q_opt = results['q_opt']
            r_opt = results['r_opt']
            init_cov_opt = results['init_cov_opt']
            selection_history = results['selection_history']
            y_pred_full = results['y_pred_full']
            y_pred_partial = results['y_pred_partial']
            partial_idx = results['partial_idx']
            partial_percentage = results['partial_percentage']
            run_dual_model = results['run_dual_model']
            mae_full = results['mae_full']
            rmse_full = results['rmse_full']
            r2_full = results['r2_full']
            mape_full = results['mape_full']
            mae_partial = results['mae_partial']
            rmse_partial = results['rmse_partial']
            r2_partial = results['r2_partial']
            mape_partial = results['mape_partial']
            
            # Results in tabs
            st.header("📊 Results")
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
                "📈 Summary", "📉 Predictions", "🎯 Scatter", "📊 Coefficients", 
                "🔄 Q & R Evolution", "🔍 Selection Process", "📄 Results", "🧪 Validation Tests"
            ])
            
            with tab1:
                st.subheader("Performance Metrics - Full Period Model")
                
                latest_actual = float(y[-1]) if len(y) else float("nan")
                latest_pred = float(y_pred[-1]) if len(y_pred) else float("nan")
                delta_latest = (
                    latest_pred - latest_actual
                    if np.isfinite(latest_actual) and np.isfinite(latest_pred)
                    else float("nan")
                )
                latest_cols = st.columns(3)
                with latest_cols[0]:
                    actual_display = f"{latest_actual:,.2f}" if np.isfinite(latest_actual) else "—"
                    st.metric("Latest Actual", actual_display)
                with latest_cols[1]:
                    pred_display = f"{latest_pred:,.2f}" if np.isfinite(latest_pred) else "—"
                    delta_display = f"{delta_latest:+.2f}" if np.isfinite(delta_latest) else "—"
                    st.metric("Latest Prediction", pred_display, delta=delta_display)
                with latest_cols[2]:
                    st.metric("Data Points", len(y))
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("R² Score", f"{r2:.4f}", 
                             delta="Good" if r2 > 0.5 else ("Moderate" if r2 > 0.3 else "Weak"))
                with col2:
                    st.metric("MAE", f"{mae:.2f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col4:
                    st.metric("MAPE", f"{mape:.2f}%")
                
                # Show partial model metrics if dual mode
                if run_dual_model and mae_partial is not None:
                    st.divider()
                    st.subheader(f"Performance Metrics - Partial Period Model ({partial_percentage}% of data)")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("R² Score", f"{r2_partial:.4f}",
                                 delta=f"{r2_partial - r2:.4f}",
                                 delta_color="normal" if r2_partial >= r2 else "inverse",
                                 help="Comparison with full model")
                    with col2:
                        st.metric("MAE", f"{mae_partial:.2f}",
                                 delta=f"{mae_partial - mae:.2f}",
                                 delta_color="inverse" if mae_partial > mae else "normal")
                    with col3:
                        st.metric("RMSE", f"{rmse_partial:.2f}",
                                 delta=f"{rmse_partial - rmse:.2f}",
                                 delta_color="inverse" if rmse_partial > rmse else "normal")
                    with col4:
                        st.metric("MAPE", f"{mape_partial:.2f}%",
                                 delta=f"{mape_partial - mape:.2f}%",
                                 delta_color="inverse" if mape_partial > mape else "normal")
                    
                    st.info(f"💡 Partial model R² is {'BETTER' if r2_partial > r2 else 'WORSE'} than full model - "
                           f"{'Model performs better with less data (possible noise in later periods)' if r2_partial > r2 else 'Model benefits from more data'}")
                
                st.divider()
            
            # Variable importance
            st.subheader("📋 Selected Variables")
            
            var_df = pd.DataFrame({
                'Variable': selected_vars_with_intercept,
                'Final Coefficient': betas[-1, :],
                'Type': ['📍 Intercept' if v == 'Intercept' else ('🔒 Mandatory' if v in mandatory_vars else '✓ Selected') 
                        for v in selected_vars_with_intercept]
            })
            st.dataframe(var_df, use_container_width=True)
        
        # Guard: ensure predictions are available before rendering detail tabs
        if (not run_all_products) and 'y_pred_full' not in locals():
            st.stop()

        with tab2:
            st.subheader("Time Series: Actual vs Predicted")
            
            fig1 = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                                subplot_titles=('Predictions', 'Residuals'))
            
            # Actual values
            fig1.add_trace(go.Scatter(x=dates, y=y, name='Actual', 
                                     mode='lines+markers',
                                     line=dict(color='#2E86AB', width=2),
                                     marker=dict(size=8)), row=1, col=1)
            
            # Full model predictions
            fig1.add_trace(go.Scatter(x=dates, y=y_pred_full, name='Full Model', 
                                     mode='lines+markers',
                                     line=dict(color='#A23B72', width=2),
                                     marker=dict(size=8)), row=1, col=1)
            
            # Partial model predictions if dual mode
            if run_dual_model and y_pred_partial is not None:
                fig1.add_trace(go.Scatter(x=dates[:partial_idx], y=y_pred_partial, 
                                         name=f'Partial Model ({partial_percentage}%)', 
                                         mode='lines+markers',
                                         line=dict(color='#F18F01', width=2, dash='dash'),
                                         marker=dict(size=8)), row=1, col=1)
                
                # Add vertical line at partial cutoff
                cutoff_date = pd.Timestamp(dates.iloc[partial_idx-1])
                fig1.add_shape(type="line",
                              x0=cutoff_date, x1=cutoff_date,
                              y0=0, y1=1,
                              yref="paper",
                              line=dict(color="green", width=2, dash="dot"),
                              row=1, col=1)
                fig1.add_annotation(x=cutoff_date, y=1, yref="paper",
                                   text=f"Partial Model End ({partial_percentage}%)",
                                   showarrow=False, yshift=10,
                                   row=1, col=1)
            
            fig1.add_trace(go.Scatter(x=dates, y=[np.mean(y)]*len(dates), name='Mean', 
                                     mode='lines',
                                     line=dict(color='orange', width=2, dash='dash')), row=1, col=1)
            
            # Residuals for full model
            residuals = y - y_pred_full
            fig1.add_trace(go.Scatter(x=dates, y=residuals, mode='markers',
                                     marker=dict(color=['red' if r<0 else 'green' for r in residuals]),
                                     name='Residuals (Full)'), row=2, col=1)
            
            # Residuals for partial model if dual mode
            if run_dual_model and y_pred_partial is not None:
                residuals_partial = y[:partial_idx] - y_pred_partial
                fig1.add_trace(go.Scatter(x=dates[:partial_idx], y=residuals_partial, mode='markers',
                                         marker=dict(color=['orange' if r<0 else 'yellow' for r in residuals_partial],
                                                   symbol='diamond', size=8),
                                         name='Residuals (Partial)'), row=2, col=1)
            
            fig1.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)
            
            fig1.update_xaxes(title_text="Date", row=2, col=1)
            fig1.update_yaxes(title_text=target_var, row=1, col=1)
            fig1.update_yaxes(title_text="Residual", row=2, col=1)
            fig1.update_layout(height=800, showlegend=True)
            
            st.plotly_chart(fig1, use_container_width=True, key="single_predictions_chart")
        
        with tab3:
            st.subheader("Scatter: Actual vs Predicted")
            
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y, y=y_pred, mode='markers',
                                     marker=dict(size=10, color=y, colorscale='Viridis',
                                               showscale=True)))
            
            min_val, max_val = 0, max(y.max(), y_pred.max())
            fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', line=dict(color='red', dash='dash', width=2),
                                     name='Perfect'))
            
            fig2.update_layout(
                title=f"Actual vs Predicted (R² = {r2:.4f})",
                xaxis_title=f"Actual {target_var}",
                yaxis_title=f"Predicted {target_var}",
                height=600
            )
            
            st.plotly_chart(fig2, use_container_width=True, key="single_scatter_chart")
        
        with tab4:
            st.subheader("Coefficient Evolution Over Time")
            
            # Include intercept in the plot (matches betas columns exactly)
            vars_with_intercept = ['Intercept'] + selected_vars
            n_vars = len(vars_with_intercept)
            
            # Show ALL variables (no limit)
            fig3 = go.Figure()
            
            for var in vars_with_intercept:
                idx = vars_with_intercept.index(var)  # matches betas columns exactly
                fig3.add_trace(go.Scatter(
                    y=betas[:, idx], 
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=6),
                    name=var[:40] + "..." if len(var) > 40 else var
                ))
            
            fig3.add_hline(y=0, line_dash="dash", line_color="gray")
            fig3.update_layout(
                title=f"Coefficient Evolution Over Time ({n_vars} variables)",
                xaxis_title="Time Index",
                yaxis_title="Coefficient Value",
                height=600,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01
                )
            )
            
            st.plotly_chart(fig3, use_container_width=True, key="single_coefficients_chart")
        
        with tab5:
            st.subheader("Q & R Evolution Over Time")
            
            if adaptive_qr and 'kf_full' in locals() and hasattr(kf_full, 'q_history') and len(kf_full.q_history) > 0:
                    fig_qr = make_subplots(rows=2, cols=1, 
                                          subplot_titles=('Q (Process Noise) Evolution', 
                                                         'R (Observation Noise) Evolution'),
                                          vertical_spacing=0.12)
                    
                    # Q evolution
                    fig_qr.add_trace(go.Scatter(
                        x=list(range(len(kf_full.q_history))),
                        y=kf_full.q_history,
                        mode='lines+markers',
                        line=dict(color='#2E86AB', width=2),
                        marker=dict(size=4),
                        name='Q'
                    ), row=1, col=1)
                    
                    fig_qr.add_hline(y=q_opt, line_dash="dash", line_color="gray", 
                                    annotation_text="Initial Q", row=1, col=1)
                    
                    # R evolution
                    fig_qr.add_trace(go.Scatter(
                        x=list(range(len(kf_full.r_history))),
                        y=kf_full.r_history,
                        mode='lines+markers',
                        line=dict(color='#A23B72', width=2),
                        marker=dict(size=4),
                        name='R'
                    ), row=2, col=1)
                    
                    fig_qr.add_hline(y=r_opt, line_dash="dash", line_color="gray",
                                    annotation_text="Initial R", row=2, col=1)
                    
                    fig_qr.update_xaxes(title_text="Week", row=2, col=1)
                    fig_qr.update_yaxes(title_text="Q Value", type="log", row=1, col=1)
                    fig_qr.update_yaxes(title_text="R Value", type="log", row=2, col=1)
                    fig_qr.update_layout(height=700, showlegend=False)
                    
                    st.plotly_chart(fig_qr, use_container_width=True, key="single_qr_evolution_chart")
                    
                    # Show statistics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Q Range", 
                                 f"{min(kf_full.q_history):.2e} - {max(kf_full.q_history):.2e}",
                                 delta=f"{(max(kf_full.q_history)/q_opt - 1)*100:.1f}% max change")
                    with col2:
                        st.metric("R Range",
                                 f"{min(kf_full.r_history):.2e} - {max(kf_full.r_history):.2e}",
                                 delta=f"{(max(kf_full.r_history)/r_opt - 1)*100:.1f}% max change")
                    
                    st.info("""
                    **How to interpret:**
                    - **Q increasing** = Model detects more volatility, allows coefficients to change faster
                    - **Q decreasing** = Model detects stability, keeps coefficients more stable
                    - **R increasing** = Model detects noisy data, trusts observations less
                    - **R decreasing** = Model detects clean data, trusts observations more
                    """)
            else:
                st.info("Adaptive Q & R is disabled. Enable it in the sidebar to see evolution over time.")
        
        with tab6:
            st.subheader("Forward Selection History")
            
            if not selection_history.empty:
                st.dataframe(selection_history, use_container_width=True)
                
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(
                    x=selection_history['Step'],
                    y=selection_history['R2'],
                    mode='lines+markers',
                    line=dict(color='#2E86AB', width=3),
                    marker=dict(size=12)
                ))
                fig4.add_hline(y=0, line_dash="dash", line_color="red")
                fig4.update_layout(
                    title="R² Improvement During Selection",
                    xaxis_title="Variables Added",
                    yaxis_title="R² Score",
                    height=400
                )
                
                st.plotly_chart(fig4, use_container_width=True, key="single_selection_chart")
            else:
                st.info("Only mandatory variables were used")
        
        with tab7:
            st.subheader("📄 Results Table & Download")

            # Create results dataframe
            results_df = data.copy()
            results_df['Predicted'] = y_pred
            results_df['Residuals'] = residuals

            # Add coefficients (including intercept)
            for i, var in enumerate(selected_vars_with_intercept):
                results_df[f'Beta_{var}'] = betas[:, i]

            metrics_summary = {
                'R2': float(r2),
                'MAE': float(mae),
                'RMSE': float(rmse),
                'MAPE': float(mape) if mape is not None else float('nan')
            }
            for k, v in metrics_summary.items():
                results_df[k] = v

            beta_columns = [f'Beta_{var}' for var in selected_vars_with_intercept]
            static_cols = []
            if 'Date' in results_df.columns:
                static_cols.append('Date')
            if 'Product title' in results_df.columns:
                static_cols.append('Product title')
            ordered_cols = static_cols + beta_columns + list(metrics_summary.keys())
            remaining_cols = [col for col in results_df.columns if col not in ordered_cols]
            results_df = results_df[ordered_cols + remaining_cols]

            st.dataframe(results_df, use_container_width=True)

            # Convert to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name='Results', index=False)
                
                metrics_df = pd.DataFrame({
                    'Metric': ['R²', 'MAE', 'RMSE', 'MAPE'],
                    'Value': [r2, mae, rmse, mape]
                })
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                var_df.to_excel(writer, sheet_name='Variables', index=False)
                
                if not selection_history.empty:
                    selection_history.to_excel(writer, sheet_name='Selection', index=False)
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Results",
                data=output,
                file_name=f"kalman_results_{selected_product.replace(' ', '_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
            st.success("✓ Results ready for download!")
        
        with tab8:
            st.subheader("🧪 Validation Tests")
            
            # Safety check - only run if adaptive is enabled
            if not adaptive_qr or not hasattr(kf_full, 'q_history') or len(kf_full.q_history) == 0:
                st.warning("⚠️ Validation tests require Adaptive Q & R to be enabled.")
                st.info("Enable 'Adaptive Q & R' in the sidebar and re-run the analysis.")
            else:
                st.markdown("**Check if your adaptive Q & R mechanism is working correctly**")
                
                # Test 1: Innovation Whiteness (Autocorrelation)
                st.markdown("### Test 1: Innovation Whiteness")
            st.markdown("*Innovations should be uncorrelated if filter is optimal*")
            
            innovations = np.array(kf_full.innovations, dtype=float) if getattr(kf_full, "innovations", None) is not None else residuals
            if kf_full.use_log and len(innovations) == 0:
                st.warning("⚠️ Innovations not captured for log-space analysis.")
                innovations = residuals
            if kf_full.use_log:
                st.caption("Innovations analysed in log-space because the model uses log-transform.")

            n = len(innovations)
            mean_res = float(np.mean(innovations)) if n > 0 else 0.0
            var_res = float(np.var(innovations)) if n > 0 else 0.0
            autocorr_data = []
            significant_lags = 0
            confidence = float('nan')
            if n >= 2 and var_res >= 1e-12:
                autocorr = np.correlate(innovations - mean_res, innovations - mean_res, mode='full')
                autocorr = autocorr[n-1:] / (var_res * n)
                lags_to_test = min(5, len(autocorr) - 1)
                confidence = 1.96 / np.sqrt(n)
                for lag in range(1, lags_to_test + 1):
                    is_sig = abs(autocorr[lag]) > confidence
                    significant_lags += int(is_sig)
                    autocorr_data.append({
                        'Lag': lag,
                        'Autocorrelation': autocorr[lag],
                        'Significant': '❌ YES' if is_sig else '✅ NO'
                    })
            else:
                st.warning("⚠️ Not enough variance to evaluate innovation autocorrelation.")
                lags_to_test = 0

            st.dataframe(pd.DataFrame(autocorr_data), use_container_width=True)
            st.info(f"95% Confidence Interval: ±{confidence:.3f}")
            
            if significant_lags == 0 and lags_to_test > 0:
                st.success("✅ PASS: Innovations appear to be white noise")
            elif lags_to_test == 0:
                st.info("ℹ️ Skipped autocorrelation verdict due to insufficient data.")
            else:
                st.warning(f"⚠️ WARNING: {significant_lags}/{lags_to_test} lags are significant. Filter may not be optimal.")
            
            # Test 2: Innovation Normality
            st.markdown("### Test 2: Innovation Normality")
            st.markdown("*Innovations should be normally distributed*")
            
            stat, p_value = stats.shapiro(innovations)
            kurt = stats.kurtosis(innovations)
            skew = stats.skew(innovations)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Shapiro-Wilk p-value", f"{p_value:.4f}")
            with col2:
                st.metric("Kurtosis", f"{kurt:.3f}", help="0 = normal, >0 = heavy tails")
            with col3:
                st.metric("Skewness", f"{skew:.3f}", help="0 = symmetric")
            with col4:
                if p_value > 0.05:
                    st.success("✅ Normal")
                else:
                    st.warning("⚠️ Non-normal")
            
            if p_value > 0.05 and abs(skew) < 0.5 and abs(kurt) < 1:
                st.success("✅ PASS: Innovations appear normally distributed")
            else:
                st.warning("⚠️ WARNING: Innovations show non-normality")
            
            # Test 3: Innovation Variance Consistency
            st.markdown("### Test 3: Innovation Variance vs R")
            st.markdown("*For well-tuned filter, innovation variance ≈ R*")
            
            innovation_var = np.var(innovations)
            mean_R = np.mean(kf_full.r_history)
            ratio = innovation_var / mean_R
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Innovation Variance", f"{innovation_var:.4f}")
            with col2:
                st.metric("Mean R Value", f"{mean_R:.4f}")
            with col3:
                st.metric("Ratio (Var/R)", f"{ratio:.4f}")
            
            if 0.5 < ratio < 2.0:
                st.success("✅ PASS: Innovation variance is consistent with R")
            elif 0.3 < ratio < 3.0:
                st.warning("⚠️ WARNING: Some mismatch, but within acceptable range")
            else:
                st.error("❌ FAIL: Large mismatch - R may be poorly tuned")
                if ratio > 3.0:
                    st.info("💡 Suggestion: Increase R or increase Q")
                else:
                    st.info("💡 Suggestion: Decrease R")
            
            # Test 4: Q & R Stability
            st.markdown("### Test 4: Q & R Stability")
            st.markdown("*Q and R should not oscillate wildly*")
            
            Q_history = np.array(kf_full.q_history)
            R_history = np.array(kf_full.r_history)
            
            Q_range_ratio = np.max(Q_history) / np.min(Q_history)
            R_range_ratio = np.max(R_history) / np.min(R_history)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Q Statistics:**")
                st.write(f"Initial: {Q_history[0]:.2e}")
                st.write(f"Final: {Q_history[-1]:.2e}")
                st.write(f"Min: {np.min(Q_history):.2e}")
                st.write(f"Max: {np.max(Q_history):.2e}")
                st.write(f"Range Ratio: {Q_range_ratio:.2f}")
                
                if Q_range_ratio < 3:
                    st.success(f"✅ Stable")
                elif Q_range_ratio < 10:
                    st.warning(f"⚠️ Moderate variation")
                else:
                    st.error(f"❌ High variation")
            
            with col2:
                st.markdown("**R Statistics:**")
                st.write(f"Initial: {R_history[0]:.2e}")
                st.write(f"Final: {R_history[-1]:.2e}")
                st.write(f"Min: {np.min(R_history):.2e}")
                st.write(f"Max: {np.max(R_history):.2e}")
                st.write(f"Range Ratio: {R_range_ratio:.2f}")
                
                if R_range_ratio < 3:
                    st.success(f"✅ Stable")
                elif R_range_ratio < 10:
                    st.warning(f"⚠️ Moderate variation")
                else:
                    st.error(f"❌ High variation")
            
            # Test 5: Adaptive vs Fixed Comparison
            st.markdown("### Test 5: Adaptive vs Fixed Comparison")
            st.markdown("*Does adaptive mechanism actually help?*")
            
            if st.button("🔄 Run Comparison with Fixed Q & R", use_container_width=True):
                with st.spinner("Running fixed Q & R model..."):
                    # Run fixed model
                    kf_fixed = ConstrainedTVLinearKalman(
                        n_features=X.shape[1], 
                        q=q_opt, 
                        r=r_opt, 
                        init_cov=init_cov_opt,
                        min_pred=0, 
                        max_pred=np.max(y)*3, 
                        use_log=False, 
                        adaptive=False,
                        non_negative_features=non_negative_constraints,
                        non_positive_features=non_positive_constraints,
                        ridge_alpha=ridge_alpha_opt
                    )
                    _, y_pred_fixed = kf_fixed.fit(X, y, feature_names=selected_vars_with_intercept)
                    
                    r2_fixed = r2_score(y, y_pred_fixed)
                    mae_fixed = mean_absolute_error(y, y_pred_fixed)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**R² Score:**")
                        st.write(f"Adaptive: {r2:.4f}")
                        st.write(f"Fixed: {r2_fixed:.4f}")
                        st.write(f"Difference: {r2 - r2_fixed:+.4f}")
                        
                        if r2 > r2_fixed + 0.01:
                            st.success("✅ Adaptive improves R²")
                        elif r2 > r2_fixed - 0.01:
                            st.warning("⚠️ Similar R² (no clear benefit)")
                        else:
                            st.error("❌ Adaptive WORSENS R²")
                    
                    with col2:
                        st.markdown("**MAE:**")
                        st.write(f"Adaptive: {mae:.2f}")
                        st.write(f"Fixed: {mae_fixed:.2f}")
                        st.write(f"Difference: {mae - mae_fixed:+.2f}")
                        
                        if mae < mae_fixed * 0.95:
                            st.success("✅ Adaptive improves MAE")
                        elif mae < mae_fixed * 1.05:
                            st.warning("⚠️ Similar MAE (no clear benefit)")
                        else:
                            st.error("❌ Adaptive WORSENS MAE")
                    
                    # Overall verdict
                    st.markdown("---")
                    st.markdown("### 🎯 Overall Verdict")
                    
                    if r2 > r2_fixed + 0.01 and mae < mae_fixed * 0.95:
                        st.success("✅✅✅ Adaptive mechanism HELPS - keep it!")
                    elif r2 > r2_fixed - 0.01 and mae < mae_fixed * 1.05:
                        st.warning("⚠️⚠️ Adaptive mechanism has MARGINAL benefit. May not be worth the complexity.")
                    else:
                        st.error("❌❌❌ Adaptive mechanism HURTS performance! Use fixed Q & R, or fix your adaptive logic.")


if __name__ == "__main__":
    main()




