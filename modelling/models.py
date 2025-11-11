"""
Custom Regression Models
Contains all custom model classes for elasticity modeling
"""

import numpy as np
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression


class CustomConstrainedRidge(BaseEstimator, RegressorMixin):
    """Ridge regression with coefficient sign constraints"""

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
        # Store as tuples for sklearn compatibility (immutable)
        self.non_positive_features = tuple(non_positive_features) if non_positive_features else ()
        self.non_negative_features = tuple(non_negative_features) if non_negative_features else ()

    def fit(self, X, Y, feature_names):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.feature_names = feature_names

        configured_non_positive = set(self.non_positive_features) if self.non_positive_features else set()
        configured_non_negative = set(self.non_negative_features) if self.non_negative_features else set()

        self._non_positive_feature_names = {name for name in feature_names if name in configured_non_positive}
        self._non_negative_feature_names = {name for name in feature_names if name in configured_non_negative}
        self._non_positive_indices = [i for i, name in enumerate(feature_names) if name in self._non_positive_feature_names]
        self._non_negative_indices = [i for i, name in enumerate(feature_names) if name in self._non_negative_feature_names]
        # Don't modify constructor parameters - use internal attributes only

        if self.adam:
            self.m_W = np.zeros(self.n)
            self.v_W = np.zeros(self.n)
            self.m_b = 0
            self.v_b = 0
            self.t = 0

        for _ in range(self.iterations):
            self.update_weights()

        self.intercept_ = self.b
        self.coef_ = self.W
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        grad_w = (-(2 * (self.X.T).dot(self.Y - Y_pred)) + 2 * self.l2_penalty * self.W) / self.m
        grad_b = -(2 / self.m) * np.sum(self.Y - Y_pred)

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

        if getattr(self, '_non_positive_indices', []):
            self.W[self._non_positive_indices] = np.minimum(self.W[self._non_positive_indices], 0)
        if getattr(self, '_non_negative_indices', []):
            self.W[self._non_negative_indices] = np.maximum(self.W[self._non_negative_indices], 0)

    def predict(self, X):
        return X.dot(self.W) + self.b


class ConstrainedLinearRegression(BaseEstimator, RegressorMixin):
    """Linear regression with coefficient sign constraints"""

    def __init__(self, learning_rate=0.001, iterations=10000,
                adam=False, beta1=0.9, beta2=0.999, epsilon=1e-8,
                non_positive_features=None, non_negative_features=None):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.adam = adam
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Store as tuples for sklearn compatibility (immutable)
        self.non_positive_features = tuple(non_positive_features) if non_positive_features else ()
        self.non_negative_features = tuple(non_negative_features) if non_negative_features else ()

    def fit(self, X, Y, feature_names):
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        self.b = 0
        self.X = X
        self.Y = Y
        self.feature_names = feature_names

        configured_non_positive = set(self.non_positive_features) if self.non_positive_features else set()
        configured_non_negative = set(self.non_negative_features) if self.non_negative_features else set()

        self._non_positive_feature_names = {name for name in feature_names if name in configured_non_positive}
        self._non_negative_feature_names = {name for name in feature_names if name in configured_non_negative}
        self._non_positive_indices = [i for i, name in enumerate(feature_names) if name in self._non_positive_feature_names]
        self._non_negative_indices = [i for i, name in enumerate(feature_names) if name in self._non_negative_feature_names]
        # Don't modify constructor parameters - use internal attributes only

        if self.adam:
            self.m_W = np.zeros(self.n)
            self.v_W = np.zeros(self.n)
            self.m_b = 0
            self.v_b = 0
            self.t = 0

        for _ in range(self.iterations):
            self.update_weights()

        self.intercept_ = self.b
        self.coef_ = self.W
        return self

    def update_weights(self):
        Y_pred = self.predict(self.X)
        dW = -(2 * self.X.T.dot(self.Y - Y_pred)) / self.m
        db = -2 * np.sum(self.Y - Y_pred) / self.m

        if self.adam:
            self.t += 1
            self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * dW
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
            self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (dW ** 2)
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)

            m_W_hat = self.m_W / (1 - self.beta1 ** self.t)
            m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
            v_W_hat = self.v_W / (1 - self.beta2 ** self.t)
            v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

            self.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            self.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            self.W -= self.learning_rate * dW
            self.b -= self.learning_rate * db

        if getattr(self, '_non_positive_indices', []):
            self.W[self._non_positive_indices] = np.minimum(self.W[self._non_positive_indices], 0)
        if getattr(self, '_non_negative_indices', []):
            self.W[self._non_negative_indices] = np.maximum(self.W[self._non_negative_indices], 0)

    def predict(self, X):
        return X.dot(self.W) + self.b


import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class RecursiveLeastSquaresRegressor(BaseEstimator, RegressorMixin):
    """Recursive Least Squares estimator with optional forgetting factor.

    Parameters
    ----------
    forgetting_factor : float, default=1.0
        RLS forgetting factor λ in (0, 1]. Use <1.0 for exponential forgetting.
    initial_covariance : float, default=1e3
        Initial diagonal value for the covariance matrix P0 = initial_covariance * I.
    fit_intercept : bool, default=True
        If True, model includes an intercept term.
    store_history : bool, default=False
        If True, stores coefficient and intercept history after each update.
    epsilon : float, default=1e-6
        Small number to guard against division by zero in the gain computation.
    """

    def __init__(
        self,
        forgetting_factor: float = 1.0,
        initial_covariance: float = 1e3,
        fit_intercept: bool = True,
        store_history: bool = False,
        epsilon: float = 1e-6,
    ):
        if not 0 < forgetting_factor <= 1:
            raise ValueError("forgetting_factor must be in (0, 1]")
        if initial_covariance <= 0:
            raise ValueError("initial_covariance must be positive")

        self.forgetting_factor = forgetting_factor
        self.initial_covariance = initial_covariance
        self.fit_intercept = fit_intercept
        self.store_history = store_history
        self.epsilon = epsilon

    # --------------------------- public API ---------------------------

    def fit(self, X, y):
        """Fit the RLS model to training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        X_array = self._prepare_features(X)
        y_array = np.asarray(y, dtype=float).reshape(-1)

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must have the same number of samples")

        n_features = X_array.shape[1]
        self._theta = np.zeros(n_features, dtype=float)
        self._covariance = np.eye(n_features, dtype=float) * self.initial_covariance

        if self.store_history:
            self.coef_history_ = []
            self.intercept_history_ = []

        for i in range(X_array.shape[0]):
            self._update_single(X_array[i], y_array[i])

        self._synchronize_public_coefficients()
        # sklearn nicety
        self.n_features_in_ = n_features - (1 if self.fit_intercept else 0)
        return self

    def update(self, X_new, y_new):
        """Online/batch update with new observations.
        
        Parameters
        ----------
        X_new : array-like of shape (n_samples, n_features) or (n_samples,)
            New training data.
        y_new : array-like of shape (n_samples,)
            New target values.
            
        Returns
        -------
        self : object
            Updated estimator.
        """
        if not hasattr(self, '_theta'):
            raise RuntimeError("Model must be fitted before calling update().")

        X_array = self._prepare_features(X_new)
        y_array = np.asarray(y_new, dtype=float).reshape(-1)

        if X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X_new and y_new must have the same number of samples")

        for i in range(X_array.shape[0]):
            self._update_single(X_array[i], y_array[i])

        self._synchronize_public_coefficients()
        return self

    def partial_fit(self, X, y):
        """Incremental fit on a batch of samples.
        
        This method is expected to be called several times consecutively
        on different chunks of a dataset. If the model is not fitted yet,
        it initializes the model first.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : object
            Updated estimator.
        """
        if not hasattr(self, '_theta'):
            # First call - initialize the model
            return self.fit(X, y)
        else:
            # Already fitted - update
            return self.update(X, y)

    def predict(self, X):
        """Predict using the RLS model.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples,)
            Samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not hasattr(self, 'coef_'):
            raise RuntimeError("Model must be fitted before calling predict().")

        X_array = np.asarray(X, dtype=float)
        if X_array.ndim == 1:
            # Be consistent with fit(): treat 1D as (n_samples, 1)
            X_array = X_array.reshape(-1, 1)

        expected = self.coef_.shape[0]
        if X_array.shape[1] != expected:
            raise ValueError(f"Expected {expected} features, got {X_array.shape[1]}")

        if self.fit_intercept:
            return X_array.dot(self.coef_) + self.intercept_
        return X_array.dot(self.coef_)

    # --------------------------- internals ---------------------------

    def _prepare_features(self, X):
        """Prepare feature matrix, optionally adding intercept column."""
        X_array = np.asarray(X, dtype=float)

        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)

        if self.fit_intercept:
            ones = np.ones((X_array.shape[0], 1), dtype=float)
            X_array = np.hstack([ones, X_array])

        if hasattr(self, '_theta') and X_array.shape[1] != len(self._theta):
            raise ValueError(
                "Incoming feature dimension does not match fitted model parameters"
            )

        return X_array

    def _update_single(self, x_vec, y_val):
        """Perform a single RLS update step."""
        # Gain
        px = self._covariance @ x_vec
        denom = self.forgetting_factor + float(x_vec.dot(px))
        if denom < self.epsilon:
            denom = self.epsilon
        gain = px / denom

        # Update theta
        residual = y_val - float(x_vec.dot(self._theta))
        self._theta = self._theta + gain * residual

        # Update covariance (Joseph form simplified for RLS)
        self._covariance = (self._covariance - np.outer(gain, px)) / self.forgetting_factor
        # Keep covariance symmetric (numerical hygiene)
        self._covariance = 0.5 * (self._covariance + self._covariance.T)

        # History (store post-update)
        if self.store_history:
            if self.fit_intercept:
                self.intercept_history_.append(float(self._theta[0]))
                self.coef_history_.append(self._theta[1:].copy())
            else:
                self.intercept_history_.append(0.0)
                self.coef_history_.append(self._theta.copy())

    def _synchronize_public_coefficients(self):
        """Sync internal _theta to public coef_ and intercept_ attributes."""
        if self.fit_intercept:
            self.intercept_ = float(self._theta[0])
            self.coef_ = self._theta[1:].astype(float, copy=True)
        else:
            self.intercept_ = 0.0
            self.coef_ = self._theta.astype(float, copy=True)

class StackedInteractionModel(BaseEstimator, RegressorMixin):
    """Stacked model with interaction terms for group-specific coefficients"""

    def __init__(self, base_model, group_keys, enforce_combined_constraints=False):
        self.base_model = base_model
        self.group_keys = group_keys
        self.enforce_combined_constraints = enforce_combined_constraints
        self.group_mapping = None
        self.feature_names = None
        self.fitted_model = None
        self.base_features_count = None

    def fit(self, X, y, feature_names=None, groups_df=None):
        self.feature_names = feature_names if feature_names is not None else (
            list(X.columns) if hasattr(X, 'columns') else [f"X{i}" for i in range(X.shape[1])]
        )
        self.base_features_count = len(self.feature_names)

        if groups_df is None:
            raise ValueError("groups_df is required for stacked models")
        
        # Validate dimensions
        X_array = X.values if hasattr(X, 'values') else X
        if len(X_array) != len(groups_df):
            raise ValueError(f"X and groups_df must have same length: X has {len(X_array)} rows, groups_df has {len(groups_df)} rows")

        if not self.group_keys:
            self.fitted_model = clone(self.base_model)
            if isinstance(self.fitted_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                self.fitted_model.fit(X, y, self.feature_names)
            else:
                self.fitted_model.fit(X, y)
            self.group_mapping = {}
            return self

        missing_keys = [k for k in self.group_keys if k not in groups_df.columns]
        if missing_keys:
            raise ValueError(f"Group keys {missing_keys} not found in groups_df")

        if len(self.group_keys) == 1:
            group_combinations = groups_df[self.group_keys[0]].astype(str)
        else:
            group_data = groups_df[self.group_keys].astype(str)
            group_combinations = group_data.apply(lambda row: "_".join(row), axis=1)

        unique_groups = sorted(group_combinations.unique())

        if len(unique_groups) == 1:
            self.fitted_model = clone(self.base_model)
            if isinstance(self.fitted_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                self.fitted_model.fit(X, y, self.feature_names)
            else:
                self.fitted_model.fit(X, y)
            self.group_mapping = {unique_groups[0]: 0}
            self.reference_group = unique_groups[0]
            return self

        self.group_mapping = {group: idx for idx, group in enumerate(unique_groups)}
        self.reference_group = unique_groups[0]

        dummy_matrix = np.zeros((len(X), len(unique_groups) - 1))
        for i, group in enumerate(group_combinations):
            group_idx = self.group_mapping[group]
            if group_idx > 0:
                dummy_matrix[i, group_idx - 1] = 1

        X_array = X.values if hasattr(X, 'values') else X
        interaction_features = []
        interaction_names = []

        for j, feat_name in enumerate(self.feature_names):
            for k in range(len(unique_groups) - 1):
                interaction = X_array[:, j] * dummy_matrix[:, k]
                interaction_features.append(interaction)
                group_name = unique_groups[k + 1]
                interaction_names.append(f"{feat_name}*{group_name}")

        X_stacked = np.hstack([
            X_array,
            dummy_matrix,
            np.column_stack(interaction_features) if interaction_features else np.empty((len(X), 0))
        ])

        dummy_names = [f"dummy_{unique_groups[i+1]}" for i in range(len(unique_groups) - 1)]
        self.all_feature_names = self.feature_names + dummy_names + interaction_names

        if isinstance(self.base_model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
            self.fitted_model = self._fit_with_combined_constraints(X_stacked, y, unique_groups)
        else:
            self.fitted_model = clone(self.base_model)
            self.fitted_model.fit(X_stacked, y)

        return self

    def _fit_with_combined_constraints(self, X_stacked, y, unique_groups):
        model = clone(self.base_model)

        if self.enforce_combined_constraints:
            model._parent_stacked = self
            model._unique_groups = unique_groups

        if isinstance(model, (CustomConstrainedRidge, ConstrainedLinearRegression)):
            model.fit(X_stacked, y, self.all_feature_names)
        else:
            model.fit(X_stacked, y)

        try:
            if self.enforce_combined_constraints and isinstance(model, CustomConstrainedRidge):
                parent = self
                n_base = parent.base_features_count
                n_groups = len(unique_groups)

                negative_feature_names = set(getattr(model, '_non_positive_feature_names', set()))
                positive_feature_names = set(getattr(model, '_non_negative_feature_names', set()))

                for g_idx in range(n_groups):
                    for f_idx, feat_name in enumerate(parent.feature_names[:n_base]):
                        combined_coef = model.W[f_idx]
                        interaction_idx = None

                        if g_idx > 0:
                            interaction_idx = n_base + (n_groups - 1) + (g_idx - 1) * n_base + f_idx
                            if interaction_idx < len(model.W):
                                combined_coef += model.W[interaction_idx]

                        if feat_name in negative_feature_names:
                            if combined_coef > 0:
                                if g_idx == 0:
                                    model.W[f_idx] = 0
                                else:
                                    correction = -combined_coef / 2
                                    model.W[f_idx] += correction
                                    if interaction_idx is not None and interaction_idx < len(model.W):
                                        model.W[interaction_idx] += correction

                        elif feat_name in positive_feature_names:
                            if combined_coef < 0:
                                if g_idx == 0:
                                    model.W[f_idx] = 0
                                else:
                                    correction = -combined_coef / 2
                                    model.W[f_idx] += correction
                                    if interaction_idx is not None and interaction_idx < len(model.W):
                                        model.W[interaction_idx] += correction
        except (AttributeError, IndexError, KeyError) as e:
            # Constraint enforcement failed - log but continue with unconstrained model
            import warnings
            warnings.warn(f"Combined constraint enforcement failed: {str(e)}. Using unconstrained model.")

        return model

    def predict(self, X, groups_df=None):
        if groups_df is None:
            raise ValueError("groups_df is required for prediction")
        
        # Validate dimensions
        X_array = X.values if hasattr(X, 'values') else X
        if len(X_array) != len(groups_df):
            raise ValueError(f"X and groups_df must have same length: X has {len(X_array)} rows, groups_df has {len(groups_df)} rows")

        if not self.group_keys or not self.group_mapping:
            return self.fitted_model.predict(X)

        if len(self.group_mapping) == 1:
            return self.fitted_model.predict(X)

        if len(self.group_keys) == 1:
            group_combinations = groups_df[self.group_keys[0]].astype(str)
        else:
            group_data = groups_df[self.group_keys].astype(str)
            group_combinations = group_data.apply(lambda row: "_".join(row), axis=1)

        dummy_matrix = np.zeros((len(X), len(self.group_mapping) - 1))
        for i, group in enumerate(group_combinations):
            if group in self.group_mapping:
                group_idx = self.group_mapping[group]
                if group_idx > 0:
                    dummy_matrix[i, group_idx - 1] = 1

        X_array = X.values if hasattr(X, 'values') else X
        interaction_features = []

        for j in range(X_array.shape[1]):
            for k in range(len(self.group_mapping) - 1):
                interaction = X_array[:, j] * dummy_matrix[:, k]
                interaction_features.append(interaction)

        X_stacked = np.hstack([
            X_array,
            dummy_matrix,
            np.column_stack(interaction_features) if interaction_features else np.empty((len(X), 0))
        ])

        return self.fitted_model.predict(X_stacked)

    def get_group_coefficients(self):
        if not hasattr(self.fitted_model, 'coef_'):
            return None

        if not self.group_keys or not self.group_mapping:
            return {
                'base': {
                    'intercept': self.fitted_model.intercept_,
                    'coefficients': dict(zip(self.feature_names, self.fitted_model.coef_[:self.base_features_count]))
                }
            }

        if len(self.group_mapping) == 1:
            group_name = list(self.group_mapping.keys())[0]
            return {
                group_name: {
                    'intercept': self.fitted_model.intercept_,
                    'coefficients': dict(zip(self.feature_names, self.fitted_model.coef_[:self.base_features_count]))
                }
            }

        coef_dict = {}
        n_features = len(self.feature_names)
        n_groups = len(self.group_mapping)

        sorted_groups = sorted(self.group_mapping.keys(), key=lambda x: self.group_mapping[x])

        for group_idx, group_name in enumerate(sorted_groups):
            combined_coefs = {}
            combined_intercept = self.fitted_model.intercept_

            if group_idx > 0:
                dummy_idx = n_features + group_idx - 1
                if dummy_idx < len(self.fitted_model.coef_):
                    combined_intercept += self.fitted_model.coef_[dummy_idx]

            for j, feat_name in enumerate(self.feature_names):
                base_coef = self.fitted_model.coef_[j]

                if group_idx > 0:
                    interaction_idx = n_features + (n_groups - 1) + (group_idx - 1) * n_features + j
                    if interaction_idx < len(self.fitted_model.coef_):
                        interaction_coef = self.fitted_model.coef_[interaction_idx]
                        combined_coefs[feat_name] = base_coef + interaction_coef
                    else:
                        combined_coefs[feat_name] = base_coef
                else:
                    combined_coefs[feat_name] = base_coef

            coef_dict[group_name] = {
                'intercept': combined_intercept,
                'coefficients': combined_coefs
            }

        return coef_dict


class StatsMixedEffectsModel(BaseEstimator, RegressorMixin):
    """Wrapper for statsmodels MixedLM"""

    def __init__(self, group_col='Brand', min_group_size=3):
        self.group_col = group_col
        self.min_group_size = min_group_size
        self._model_result = None
        self.fixed_coef_ = None
        self.intercept_ = 0.0
        self.random_effects_dict_ = {}
        self.feature_names_ = None
        self._fallback_model = None
        self._use_fallback = False

    def fit(self, X, y, groups):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=[f"X{i}" for i in range(X.shape[1])])

        self.feature_names_ = list(X.columns)

        groups_series = pd.Series(groups.values if hasattr(groups, 'values') else groups)
        group_counts = groups_series.value_counts()

        valid_groups = group_counts[group_counts >= self.min_group_size].index
        valid_mask = groups_series.isin(valid_groups)

        n_filtered = len(group_counts) - len(valid_groups)
        if n_filtered > 0:
            st.caption(f"Filtered {n_filtered} groups with < {self.min_group_size} observations")

        if valid_mask.sum() < len(X) * 0.5:
            self._use_fallback = True

        try:
            if not self._use_fallback and len(valid_groups) > 1:
                X_valid = X[valid_mask]
                y_valid = y[valid_mask] if hasattr(y, '__getitem__') else y[valid_mask]
                groups_valid = groups_series[valid_mask]

                X_with_const = sm.add_constant(X_valid, has_constant='add')

                mixed_model = sm.MixedLM(
                    endog=y_valid,
                    exog=X_with_const,
                    groups=groups_valid.values,
                    exog_re=None
                )

                try:
                    self._model_result = mixed_model.fit(method='lbfgs', reml=False)
                except (np.linalg.LinAlgError, ValueError):
                    try:
                        self._model_result = mixed_model.fit(method='bfgs', reml=False)
                    except (np.linalg.LinAlgError, ValueError):
                        self._model_result = mixed_model.fit(method='powell', reml=False)

                params = self._model_result.params
                self.intercept_ = params['const'] if 'const' in params else 0.0
                self.fixed_coef_ = params.drop('const').values

                self.random_effects_dict_ = {
                    group: effects.values[0]
                    for group, effects in self._model_result.random_effects.items()
                }

                for group in group_counts.index:
                    if group not in self.random_effects_dict_:
                        self.random_effects_dict_[group] = 0.0

            else:
                self._use_fallback = True

        except Exception as e:
            st.warning(f"Mixed effects failed: {str(e)}. Using fallback.")
            self._use_fallback = True

        if self._use_fallback:
            self._fallback_model = LinearRegression()
            self._fallback_model.fit(X, y)
            self.intercept_ = self._fallback_model.intercept_
            self.fixed_coef_ = self._fallback_model.coef_
            self.random_effects_dict_ = {group: 0.0 for group in group_counts.index}

        self.coef_ = self.fixed_coef_

        return self

    def predict(self, X, groups=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)

        if self._use_fallback and self._fallback_model is not None:
            return self._fallback_model.predict(X)
        else:
            X_with_const = sm.add_constant(X, has_constant='add')
            y_pred_fixed = X_with_const.values @ np.concatenate([[self.intercept_], self.fixed_coef_])

            if groups is None:
                return y_pred_fixed
            else:
                y_pred = y_pred_fixed.copy()
                groups_array = groups.values if hasattr(groups, 'values') else groups

                for i, group in enumerate(groups_array):
                    if group in self.random_effects_dict_:
                        y_pred[i] += self.random_effects_dict_[group]

                return y_pred
