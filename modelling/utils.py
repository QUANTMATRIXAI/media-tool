"""
Utility Functions for Modelling Module

This module provides helper functions for model validation and ensemble creation.

Key Functions:
--------------
- safe_mape: Calculate MAPE safely with automatic protection against small values
- build_weighted_ensemble_model: Build weighted ensemble by averaging coefficients
- create_ensemble_model_from_results: Create ensemble models from CV results
"""

import numpy as np
import pandas as pd
import streamlit as st


def safe_mape(y_true, y_pred):
    """
    Calculate MAPE safely with automatic protection against small values
    - Excludes values where |y_true| < 1.0 (handles count data with 0s)
    - Caps individual errors at 500% to prevent outlier explosion
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Only include values where |y_true| >= 1.0
    valid_mask = (np.abs(y_true) >= 1.0)

    if not valid_mask.any():
        return float("nan")

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # Calculate percentage errors
    percent_errors = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100

    # Cap extreme percentage errors at 500% to prevent small values from dominating
    percent_errors = np.minimum(percent_errors, 500.0)

    return np.mean(percent_errors)


# ═══════════════════════════════════════════════════════════════════════════
# ENSEMBLE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_weighted_ensemble_model(models_df, weight_metric='MAPE Test', grouping_keys=None):
    """
    Build a weighted ensemble model by averaging coefficients across models.
    Uses MAPE-based exponential weighting (lower MAPE = higher weight).

    Parameters:
    -----------
    models_df : pd.DataFrame
        DataFrame with model results (must have Fold='Avg' rows)
    weight_metric : str
        Column name to use for weighting (default: 'MAPE Test')
    grouping_keys : list
        Keys that define unique combinations

    Returns:
    --------
    dict with ensemble coefficients and metadata
    """
    if weight_metric not in models_df.columns:
        st.warning(f"Metric '{weight_metric}' not found. Using equal weights.")
        weights = np.ones(len(models_df))
    else:
        metric_values = pd.to_numeric(models_df[weight_metric], errors='coerce')

        if metric_values.isna().all():
            weights = np.ones(len(models_df))
        else:
            # Exponential weighting: lower metric = higher weight
            best_value = metric_values.min()
            weights = np.exp(-0.5 * (metric_values - best_value))
            weights = np.nan_to_num(weights, nan=0.0)

    # Normalize weights
    weight_sum = weights.sum()
    if weight_sum == 0 or np.isnan(weight_sum):
        weights = np.ones(len(models_df))
        weight_sum = weights.sum()

    weights = weights / weight_sum

    # Extract beta columns
    beta_cols = [c for c in models_df.columns if c.startswith('Beta_')]

    # Weighted average of betas
    ensemble_betas = {}
    for beta_col in beta_cols:
        if beta_col in models_df.columns:
            values = pd.to_numeric(models_df[beta_col], errors='coerce').fillna(0)
            ensemble_betas[beta_col] = np.average(values, weights=weights)

    # Weighted average of intercept
    ensemble_intercept = 0.0
    if 'B0 (Original)' in models_df.columns:
        b0_values = pd.to_numeric(models_df['B0 (Original)'], errors='coerce').fillna(0)
        ensemble_intercept = np.average(b0_values, weights=weights)

    # Metadata
    result = {
        'ensemble_betas': ensemble_betas,
        'ensemble_intercept': ensemble_intercept,
        'num_models': len(models_df),
        'weights': weights,
        'model_names': models_df['Model'].tolist() if 'Model' in models_df.columns else [],
        'best_model_idx': int(np.argmax(weights)),
        'weight_concentration': float(weights.max())
    }

    # Add weighted metrics
    for metric_col in ['R2 Test', 'R2 Train', 'MAPE Test', 'MAPE Train', 'MAE Test', 'MAE Train']:
        if metric_col in models_df.columns:
            metric_values = pd.to_numeric(models_df[metric_col], errors='coerce').fillna(0)
            result[f'ensemble_{metric_col.lower().replace(" ", "_")}'] = np.average(metric_values, weights=weights)

    return result


def create_ensemble_model_from_results(results_df, grouping_keys, feature_names,
                                       weight_metric='MAPE Test',
                                       filter_r2_min=None, filter_mape_max=None, filter_mae_max=None,
                                       filter_positive_features=None, filter_negative_features=None):
    """
    Create ensemble models for each unique combination from CV results.

    Parameters:
    -----------
    results_df : pd.DataFrame
        Results from cross-validation (must include Fold='Avg' rows)
    grouping_keys : list
        Keys that define unique combinations
    feature_names : list
        List of feature/predictor names
    weight_metric : str
        Metric to use for weighting ('MAPE Test', 'MAE Test', etc.)
    filter_r2_min : float, optional
        Minimum R² Test threshold to include a model
    filter_mape_max : float, optional
        Maximum MAPE Test threshold to include a model
    filter_mae_max : float, optional
        Maximum MAE Test threshold to include a model
    filter_positive_features : list, optional
        Features that must have positive coefficients (≥0) in models to be included
    filter_negative_features : list, optional
        Features that must have negative coefficients (≤0) in models to be included

    Returns:
    --------
    dict: {combination_key: ensemble_model_dict}
    """
    # Filter to only averaged results
    avg_results = results_df[results_df['Fold'] == 'Avg'].copy()

    if avg_results.empty:
        return {}

    # Apply optional filters
    if filter_r2_min is not None and 'R2 Test' in avg_results.columns:
        r2_values = pd.to_numeric(avg_results['R2 Test'], errors='coerce')
        avg_results = avg_results[r2_values >= filter_r2_min]

    if filter_mape_max is not None and 'MAPE Test' in avg_results.columns:
        mape_values = pd.to_numeric(avg_results['MAPE Test'], errors='coerce')
        avg_results = avg_results[mape_values <= filter_mape_max]

    if filter_mae_max is not None and 'MAE Test' in avg_results.columns:
        mae_values = pd.to_numeric(avg_results['MAE Test'], errors='coerce')
        avg_results = avg_results[mae_values <= filter_mae_max]

    # Apply sign-based filtering
    if filter_positive_features or filter_negative_features:
        initial_count = len(avg_results)
        mask = pd.Series([True] * len(avg_results), index=avg_results.index)

        # Check positive constraints
        if filter_positive_features:
            for feature in filter_positive_features:
                beta_col = f"Beta_{feature}"
                if beta_col in avg_results.columns:
                    beta_values = pd.to_numeric(avg_results[beta_col], errors='coerce')
                    mask &= (beta_values >= -1e-6)

        # Check negative constraints
        if filter_negative_features:
            for feature in filter_negative_features:
                beta_col = f"Beta_{feature}"
                if beta_col in avg_results.columns:
                    beta_values = pd.to_numeric(avg_results[beta_col], errors='coerce')
                    mask &= (beta_values <= 1e-6)

        avg_results = avg_results[mask]
        filtered_count = initial_count - len(avg_results)

        if filtered_count > 0:
            st.info(f"🔍 Sign filtering: Excluded {filtered_count} model(s) with incorrect coefficient signs")

    if avg_results.empty:
        st.warning("⚠️ No models passed the ensemble filters. Relax filter thresholds.")
        return {}

    # Build ensemble for each unique combination
    ensembles = {}

    if not grouping_keys:
        ensemble = build_weighted_ensemble_model(avg_results, weight_metric, grouping_keys)
        ensembles['ALL'] = ensemble
    else:
        for combo_vals, group_df in avg_results.groupby(grouping_keys, dropna=False):
            if len(group_df) == 0:
                continue

            combo_key = " | ".join([f"{k}={v}" for k, v in zip(grouping_keys, combo_vals)]) if isinstance(combo_vals, tuple) else f"{grouping_keys[0]}={combo_vals}"
            ensemble = build_weighted_ensemble_model(group_df, weight_metric, grouping_keys)
            ensembles[combo_key] = ensemble

    return ensembles
