"""
Main Modeling Pipeline

This module contains the core modeling pipeline function that orchestrates the entire
cross-validation and model training process. It supports various model types including
regular models, stacked interaction models, and ensemble models.

Key features:
- K-fold cross-validation with adaptive fold selection
- Support for multiple model types (Ridge, Lasso, ElasticNet, etc.)
- Stacked interaction models with group-specific coefficients
- Weighted ensemble model creation from CV results
- Comprehensive metrics tracking (R2, MAPE, MAE, MSE, RMSE)
"""

import numpy as np
import pandas as pd
import streamlit as st
import time

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from models import (
    CustomConstrainedRidge,
    ConstrainedLinearRegression,
    RecursiveLeastSquaresRegressor,
    StackedInteractionModel,
    StatsMixedEffectsModel
)

from utils.safe_mape import safe_mape, create_ensemble_model_from_results


def run_model_pipeline(
    df,
    grouping_keys,
    X_columns,
    target_col,
    k_folds,
    std_cols,
    models_dict,
    use_stacked=False,
    stacking_keys=None,
    filter_keys_for_stacking=None,
    log_transform_y=False,
    min_y_share_pct=1.0,
    # Ensemble parameters:
    enable_ensemble=False,
    ensemble_weight_metric='MAPE Test',
    ensemble_filter_r2_min=None,
    ensemble_filter_mape_max=None,
    ensemble_filter_mae_max=None,
    ensemble_filter_positive_features=None,
    ensemble_filter_negative_features=None
):
    """
    Run modeling pipeline
    Returns aggregated results (one row per group-model) and predictions
    """
    rows = []
    preds_records = []

    # Separate stacked and non-stacked models
    stacked_models = {k: v for k, v in models_dict.items() if isinstance(v, StackedInteractionModel)}
    regular_models = {k: v for k, v in models_dict.items() if not isinstance(v, StackedInteractionModel)}

    # For regular models: use ALL grouping_keys
    if grouping_keys and regular_models:
        grouped_regular = df.groupby(grouping_keys)
        group_list_regular = list(grouped_regular)
    else:
        group_list_regular = [((None,), df)] if regular_models else []

    # For stacked models: use ONLY filter_keys_for_stacking
    if filter_keys_for_stacking and stacked_models:
        grouped_stacked = df.groupby(filter_keys_for_stacking)
        group_list_stacked = list(grouped_stacked)
    else:
        group_list_stacked = [((None,), df)] if stacked_models else []

    # ═════════════════════════════════════════════════════════════════════════
    # FILTER GROUPS BY Y VARIABLE SHARE (min_y_share_pct)
    # ═════════════════════════════════════════════════════════════════════════
    total_y = df[target_col].sum()

    if min_y_share_pct > 0 and total_y > 0:
        # Filter regular models groups
        filtered_regular = []
        skipped_regular = []
        for gvals, gdf in group_list_regular:
            group_y_sum = gdf[target_col].sum()
            group_y_share = (group_y_sum / total_y) * 100
            if group_y_share >= min_y_share_pct:
                filtered_regular.append((gvals, gdf))
            else:
                gvals_tuple = (gvals,) if not isinstance(gvals, tuple) else gvals
                group_name = " | ".join([f"{k}={v}" for k, v in zip(grouping_keys, gvals_tuple)]) if grouping_keys else "All"
                skipped_regular.append(f"{group_name} ({group_y_share:.2f}%)")

        # Filter stacked models groups
        filtered_stacked = []
        skipped_stacked = []
        for gvals, gdf in group_list_stacked:
            group_y_sum = gdf[target_col].sum()
            group_y_share = (group_y_sum / total_y) * 100
            if group_y_share >= min_y_share_pct:
                filtered_stacked.append((gvals, gdf))
            else:
                gvals_tuple = (gvals,) if not isinstance(gvals, tuple) else gvals
                group_name = " | ".join([f"{k}={v}" for k, v in zip(filter_keys_for_stacking if filter_keys_for_stacking else [], gvals_tuple)]) if filter_keys_for_stacking else "All"
                skipped_stacked.append(f"{group_name} ({group_y_share:.2f}%)")

        # Update group lists
        group_list_regular = filtered_regular
        group_list_stacked = filtered_stacked

        # Show info about filtered groups
        if skipped_regular:
            st.info(f"🔍 Filtered {len(skipped_regular)} regular model group(s) with <{min_y_share_pct}% Y share:\n" +
                   "\n".join([f"• {name}" for name in skipped_regular[:10]]) +
                   (f"\n• ... and {len(skipped_regular) - 10} more" if len(skipped_regular) > 10 else ""))

        if skipped_stacked:
            st.info(f"🔍 Filtered {len(skipped_stacked)} stacked model group(s) with <{min_y_share_pct}% Y share:\n" +
                   "\n".join([f"• {name}" for name in skipped_stacked[:10]]) +
                   (f"\n• ... and {len(skipped_stacked) - 10} more" if len(skipped_stacked) > 10 else ""))

    n_regular_ops = len(group_list_regular) * len(regular_models) * k_folds
    n_stacked_ops = len(group_list_stacked) * len(stacked_models) * k_folds
    total_operations = n_regular_ops + n_stacked_ops

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    start_time = time.time()
    operation_count = 0

    def update_progress(group_name, model_name, fold_num):
        nonlocal operation_count
        operation_count += 1
        progress = operation_count / total_operations if total_operations > 0 else 1
        progress_bar.progress(progress)

        elapsed_time = time.time() - start_time
        if operation_count > 0:
            avg_time_per_op = elapsed_time / operation_count
            remaining_ops = total_operations - operation_count
            estimated_remaining = avg_time_per_op * remaining_ops

            status_text.text(f"Processing: {model_name} | {group_name} | Fold {fold_num}/{k_folds} | ~{estimated_remaining:.0f}s remaining")

    # ═════════════════════════════════════════════════════════════════════════
    # PROCESS REGULAR MODELS (grouped by ALL keys)
    # ═════════════════════════════════════════════════════════════════════════
    for gvals, gdf in group_list_regular:
        gvals = (gvals,) if not isinstance(gvals, tuple) else gvals
        group_display_name = " | ".join([f"{k}={v}" for k, v in zip(grouping_keys, gvals)]) if grouping_keys else "All"

        present_cols = [c for c in X_columns if c in gdf.columns]
        if len(present_cols) < len(X_columns):
            for mname in regular_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        X_full = gdf[present_cols].fillna(0).copy()
        y_full = gdf[target_col].copy()

        n_samples = len(X_full)
        n_features = len(present_cols)

        # CRITICAL FIX: Check for overfitting conditions
        min_samples_needed = max(k_folds, n_features * 2)  # Need at least 2x features for stable modeling

        if n_samples < min_samples_needed:
            st.warning(f"⚠️ Skipping {group_display_name}: Only {n_samples} samples but need {min_samples_needed} (have {n_features} features)")
            for mname in regular_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        # ADAPTIVE K-FOLD: Reduce folds for smaller groups
        adaptive_k = k_folds
        if n_samples < 20:
            adaptive_k = 2  # Use 2-fold for very small groups
        elif n_samples < 50:
            adaptive_k = min(3, k_folds)  # Use 3-fold for small groups

        kf = KFold(n_splits=adaptive_k, shuffle=True, random_state=42)

        if adaptive_k != k_folds:
            st.info(f"ℹ️ {group_display_name}: Using {adaptive_k}-fold CV (only {n_samples} samples)")

        for mname, mdl in regular_models.items():
            fold_results = []

            train_df = gdf
            display_name = mname

            # Use TRAIN data for CV (not full data)
            X_full = train_df[present_cols].fillna(0).copy()
            y_full = train_df[target_col].copy()


            # Regular model processing
            for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_full, y_full), 1):
                update_progress(group_display_name, mname, fold_id)

                X_tr, X_te = X_full.iloc[tr_idx].copy(), X_full.iloc[te_idx].copy()
                y_tr, y_te = y_full.iloc[tr_idx].copy(), y_full.iloc[te_idx].copy()

                # Store original y values for metrics calculation
                y_tr_original = y_tr.copy()
                y_te_original = y_te.copy()

                # Apply log transformation if requested
                if log_transform_y:
                    y_tr = np.log1p(y_tr)  # log(1 + y)
                    y_te = np.log1p(y_te)

                # Standardization
                scaler = {}
                if std_cols:
                    cols_to_scale = [c for c in std_cols if c in X_tr.columns]
                else:
                    cols_to_scale = []

                if cols_to_scale:
                    sc = StandardScaler().fit(X_tr[cols_to_scale])
                    X_tr[cols_to_scale] = sc.transform(X_tr[cols_to_scale])
                    X_te[cols_to_scale] = sc.transform(X_te[cols_to_scale])
                    scaler = {c: (m, s) for c, m, s in zip(cols_to_scale, sc.mean_, sc.scale_)}

                # Train model
                model_copy = clone(mdl)

                # Fit based on model type
                if isinstance(model_copy, (CustomConstrainedRidge, ConstrainedLinearRegression)):
                    model_copy.fit(X_tr.values, y_tr.values, X_tr.columns.tolist())
                    y_tr_pred = model_copy.predict(X_tr.values)
                    y_te_pred = model_copy.predict(X_te.values)
                    B0_std, B1_std = model_copy.intercept_, model_copy.coef_
                elif isinstance(model_copy, StatsMixedEffectsModel):
                    tr_orig_idx = X_tr.index
                    te_orig_idx = X_te.index

                    grp_col = model_copy.group_col

                    if '_' in grp_col and grp_col not in gdf.columns:
                        component_keys = grp_col.split('_')
                        if all(k in gdf.columns for k in component_keys):
                            groups_tr_values = gdf.loc[tr_orig_idx, component_keys].astype(str).apply(
                                lambda row: "_".join(row), axis=1
                            )
                            groups_te_values = gdf.loc[te_orig_idx, component_keys].astype(str).apply(
                                lambda row: "_".join(row), axis=1
                            )
                        else:
                            groups_tr_values = gdf.loc[tr_orig_idx, grouping_keys[0]]
                            groups_te_values = gdf.loc[te_orig_idx, grouping_keys[0]]
                    else:
                        if grp_col in gdf.columns:
                            groups_tr_values = gdf.loc[tr_orig_idx, grp_col]
                            groups_te_values = gdf.loc[te_orig_idx, grp_col]
                        else:
                            groups_tr_values = gdf.loc[tr_orig_idx, grouping_keys[0]]
                            groups_te_values = gdf.loc[te_orig_idx, grouping_keys[0]]

                    model_copy.fit(X_tr, y_tr, groups_tr_values)
                    y_tr_pred = model_copy.predict(X_tr, groups_tr_values)
                    y_te_pred = model_copy.predict(X_te, groups_te_values)
                    B0_std, B1_std = model_copy.intercept_, model_copy.coef_
                else:
                    model_copy.fit(X_tr, y_tr)
                    y_tr_pred = model_copy.predict(X_tr)
                    y_te_pred = model_copy.predict(X_te)
                    B0_std, B1_std = model_copy.intercept_, model_copy.coef_


                # Reverse transform predictions if log was applied
                if log_transform_y:
                    y_tr_pred = np.expm1(y_tr_pred)  # exp(pred) - 1
                    y_te_pred = np.expm1(y_te_pred)
                    # Ensure non-negative predictions
                    y_tr_pred = np.maximum(y_tr_pred, 0)
                    y_te_pred = np.maximum(y_te_pred, 0)

                # Metrics (calculated on original scale)
                r2_tr = r2_score(y_tr_original, y_tr_pred)
                r2_te = r2_score(y_te_original, y_te_pred)
                mape_tr = safe_mape(y_tr_original, y_tr_pred)
                mape_te = safe_mape(y_te_original, y_te_pred)
                mae_tr = np.mean(np.abs(y_tr_original - y_tr_pred))
                mae_te = np.mean(np.abs(y_te_original - y_te_pred))
                mse_tr = np.mean((y_tr_original - y_tr_pred)**2)
                mse_te = np.mean((y_te_original - y_te_pred)**2)
                rmse_tr = np.sqrt(mse_tr)
                rmse_te = np.sqrt(mse_te)

                # Reverse standardization
                raw_int, raw_coefs = B0_std, B1_std.copy()
                for i, col in enumerate(present_cols):
                    if col in scaler:
                        mu, sd = scaler[col]
                        raw_coefs[i] = raw_coefs[i] / sd
                        raw_int -= raw_coefs[i] * mu

                # FIX: Mean X - calculate ONLY on TRAINING data to avoid leakage
                mean_x = X_tr.mean(numeric_only=True).to_dict()

                # Create fold result
                d = {k: v for k, v in zip(grouping_keys, gvals)}
                d.update({
                    "Model": display_name,
                    "Fold": fold_id,
                    "B0 (Original)": raw_int,
                    "R2 Train": r2_tr,
                    "R2 Test": r2_te,
                    "MAPE Train": mape_tr,
                    "MAPE Test": mape_te,
                    "MAE Train": mae_tr,
                    "MAE Test": mae_te,
                    "MSE Train": mse_tr,
                    "MSE Test": mse_te,
                    "RMSE Train": rmse_tr,
                    "RMSE Test": rmse_te,
                })

                # Add mean X
                for c, v in mean_x.items():
                    d[c] = v

                # Add betas
                for i, c in enumerate(present_cols):
                    d[f"Beta_{c}"] = raw_coefs[i]

                fold_results.append(d)

                # Predictions
                pr = gdf.loc[X_te.index].copy()
                pr["Actual"] = y_te.values
                pr["Predicted"] = y_te_pred
                pr["Model"] = display_name
                pr["Fold"] = fold_id
                preds_records.append(pr)

            # Report fold results: individual folds + aggregated
            if fold_results:
                fold_df = pd.DataFrame(fold_results)

                # Add individual fold rows
                rows.append(fold_df)

                # Create aggregated row (average across folds)
                # Identify keys
                key_cols = [col for col in fold_df.columns if col in grouping_keys + list(getattr(mdl, 'group_keys', [])) or col == 'Model']

                # Numeric columns to average
                numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != 'Fold']

                # String columns to take first
                string_cols = [col for col in fold_df.columns if col not in numeric_cols and col not in key_cols and col != 'Fold']

                # Aggregate
                agg_dict = {}
                for col in numeric_cols:
                    agg_dict[col] = 'mean'
                for col in string_cols:
                    agg_dict[col] = 'first'

                aggregated = fold_df.groupby(key_cols).agg(agg_dict).reset_index()
                aggregated['Fold'] = 'Avg'  # Mark as average row
                rows.append(aggregated)

    # ═════════════════════════════════════════════════════════════════════════
    # PROCESS STACKED MODELS (grouped by FILTER keys only, interaction on STACKING keys)
    # ═════════════════════════════════════════════════════════════════════════
    for gvals, gdf in group_list_stacked:
        gvals = (gvals,) if not isinstance(gvals, tuple) else gvals
        group_display_name = " | ".join([f"{k}={v}" for k, v in zip(filter_keys_for_stacking, gvals)]) if filter_keys_for_stacking else "All"

        present_cols = [c for c in X_columns if c in gdf.columns]
        if len(present_cols) < len(X_columns):
            for mname in stacked_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        X_full = gdf[present_cols].fillna(0).copy()
        y_full = gdf[target_col].copy()

        n_samples = len(X_full)
        n_features = len(present_cols)

        # Count number of unique groups in stacking keys
        if stacked_models:
            first_stacked_model = list(stacked_models.values())[0]
            n_groups = gdf[first_stacked_model.group_keys].drop_duplicates().shape[0]
            # Each group needs enough samples
            min_samples_per_group = max(3, n_features // 2)
            min_samples_needed = max(k_folds, n_groups * min_samples_per_group)
        else:
            min_samples_needed = max(k_folds, n_features * 2)

        if n_samples < min_samples_needed:
            st.warning(f"⚠️ Skipping stacked model for {group_display_name}: Only {n_samples} samples but need {min_samples_needed}")
            for mname in stacked_models.keys():
                for fold in range(k_folds):
                    update_progress(group_display_name, mname, fold + 1)
            continue

        # ADAPTIVE K-FOLD: Reduce folds for smaller groups
        adaptive_k = k_folds
        if n_samples < 30:
            adaptive_k = 2  # Use 2-fold for very small groups (stacked needs more samples)
        elif n_samples < 60:
            adaptive_k = min(3, k_folds)  # Use 3-fold for small groups

        kf = KFold(n_splits=adaptive_k, shuffle=True, random_state=42)

        if adaptive_k != k_folds:
            st.info(f"ℹ️ {group_display_name} (stacked): Using {adaptive_k}-fold CV (only {n_samples} samples)")

        for mname, mdl in stacked_models.items():
            fold_results = []

            for fold_id, (tr_idx, te_idx) in enumerate(kf.split(X_full, y_full), 1):
                update_progress(group_display_name, mname, fold_id)

                # CRITICAL FIX: Use .iloc to get by position, keep track of original indices
                # tr_idx and te_idx are POSITIONS in X_full
                X_tr_orig = X_full.iloc[tr_idx].copy()
                X_te_orig = X_full.iloc[te_idx].copy()
                y_tr_orig = y_full.iloc[tr_idx].copy()
                y_te_orig = y_full.iloc[te_idx].copy()

                # Store ORIGINAL y values for metrics calculation (before any transformation)
                y_tr_original_scale = y_tr_orig.copy()
                y_te_original_scale = y_te_orig.copy()

                # Apply log transformation if requested
                if log_transform_y:
                    y_tr_orig = np.log1p(y_tr_orig)  # log(1 + y)
                    y_te_orig = np.log1p(y_te_orig)

                # Store original indices for later use
                tr_index_map = X_tr_orig.index.tolist()
                te_index_map = X_te_orig.index.tolist()

                # Get groups using the SAME POSITIONAL indices
                # This ensures perfect alignment: row i of groups corresponds to row i of X
                groups_tr = gdf.iloc[tr_idx][mdl.group_keys].copy()
                groups_te = gdf.iloc[te_idx][mdl.group_keys].copy()

                # Standardization BEFORE resetting indices
                scaler = {}
                if std_cols:
                    cols_to_scale = [c for c in std_cols if c in X_tr_orig.columns]
                else:
                    cols_to_scale = []

                if cols_to_scale:
                        sc = StandardScaler().fit(X_tr_orig[cols_to_scale])
                        X_tr_orig[cols_to_scale] = sc.transform(X_tr_orig[cols_to_scale])
                        X_te_orig[cols_to_scale] = sc.transform(X_te_orig[cols_to_scale])
                        scaler = {c: (m, s) for c, m, s in zip(cols_to_scale, sc.mean_, sc.scale_)}

                # NOW reset all indices in perfect synchronization
                X_tr_reset = X_tr_orig.reset_index(drop=True)
                X_te_reset = X_te_orig.reset_index(drop=True)
                y_tr_reset = y_tr_orig.reset_index(drop=True)
                y_te_reset = y_te_orig.reset_index(drop=True)
                y_tr_original_scale_reset = y_tr_original_scale.reset_index(drop=True)
                y_te_original_scale_reset = y_te_original_scale.reset_index(drop=True)
                groups_tr = groups_tr.reset_index(drop=True)
                groups_te = groups_te.reset_index(drop=True)

                model_copy = clone(mdl)
                model_copy.fit(X_tr_reset, y_tr_reset, feature_names=present_cols, groups_df=groups_tr)

                y_tr_pred = model_copy.predict(X_tr_reset, groups_df=groups_tr)
                y_te_pred = model_copy.predict(X_te_reset, groups_df=groups_te)

                # Reverse transform predictions if log was applied
                if log_transform_y:
                    y_tr_pred = np.expm1(y_tr_pred)  # exp(pred) - 1
                    y_te_pred = np.expm1(y_te_pred)
                    # Ensure non-negative predictions
                    y_tr_pred = np.maximum(y_tr_pred, 0)
                    y_te_pred = np.maximum(y_te_pred, 0)

                # Get group coefficients
                group_coefs = model_copy.get_group_coefficients()

                # Create result rows for each stacking group
                if len(mdl.group_keys) == 1:
                    test_groups = groups_te[mdl.group_keys[0]].astype(str)
                    train_groups = groups_tr[mdl.group_keys[0]].astype(str)
                else:
                    test_groups = groups_te[mdl.group_keys].astype(str).apply(lambda row: "_".join(row), axis=1)
                    train_groups = groups_tr[mdl.group_keys].astype(str).apply(lambda row: "_".join(row), axis=1)

                unique_test_groups = test_groups.unique()

                for group in unique_test_groups:
                    group_mask_te = (test_groups == group).values

                    if not group_mask_te.any():
                        continue

                    # Get original scale y values for this group
                    y_te_group_original = y_te_original_scale_reset[group_mask_te]
                    y_pred_te_group = y_te_pred[group_mask_te]

                    # Metrics on original scale
                    r2_te = r2_score(y_te_group_original, y_pred_te_group) if len(y_te_group_original) > 1 else np.nan
                    mape_te = safe_mape(y_te_group_original, y_pred_te_group)
                    mae_te = np.mean(np.abs(y_te_group_original - y_pred_te_group))
                    mse_te = np.mean((y_te_group_original - y_pred_te_group)**2)
                    rmse_te = np.sqrt(mse_te)

                    group_mask_tr = (train_groups == group).values

                    if group_mask_tr.any():
                        y_tr_group_original = y_tr_original_scale_reset[group_mask_tr]
                        y_pred_tr_group = y_tr_pred[group_mask_tr]
                        r2_tr = r2_score(y_tr_group_original, y_pred_tr_group) if len(y_tr_group_original) > 1 else np.nan
                        mape_tr = safe_mape(y_tr_group_original, y_pred_tr_group)
                        mae_tr = np.mean(np.abs(y_tr_group_original - y_pred_tr_group))
                        mse_tr = np.mean((y_tr_group_original - y_pred_tr_group)**2)
                        rmse_tr = np.sqrt(mse_tr)
                    else:
                        r2_tr = mape_tr = mae_tr = mse_tr = rmse_tr = np.nan

                    if group in group_coefs:
                        raw_int = group_coefs[group]['intercept']
                        raw_coefs_dict = group_coefs[group]['coefficients']

                        # Reverse standardization
                        if scaler and std_cols:
                            for col in std_cols:
                                if col in raw_coefs_dict and col in scaler:
                                    mu, sd = scaler[col]
                                    raw_coefs_dict[col] = raw_coefs_dict[col] / sd
                                    raw_int -= raw_coefs_dict[col] * mu

                        # FIX: Calculate mean X ONLY from TRAINING data to avoid leakage
                        if group_mask_tr.any():
                            train_indices_for_group = np.where(group_mask_tr)[0]
                            train_original_indices = [tr_index_map[i] for i in train_indices_for_group]
                            group_train_data = gdf.loc[train_original_indices]
                            mean_x = group_train_data[present_cols].mean(numeric_only=True).to_dict()
                        else:
                            # Fallback if group not in training (shouldn't happen in CV)
                            mean_x = {c: np.nan for c in present_cols}

                        # Create fold result
                        group_parts = group.split('_')
                        d = {}

                        # Add filter grouping keys
                        for idx, key in enumerate(filter_keys_for_stacking):
                            d[key] = gvals[idx]

                        # Add stacking keys
                        for idx, key in enumerate(mdl.group_keys):
                            d[key] = group_parts[idx] if idx < len(group_parts) else ''

                        d.update({
                            "Model": mname,
                            "Fold": fold_id,
                            "B0 (Original)": raw_int,
                            "R2 Train": r2_tr,
                            "R2 Test": r2_te,
                            "MAPE Train": mape_tr,
                            "MAPE Test": mape_te,
                            "MAE Train": mae_tr,
                            "MAE Test": mae_te,
                            "MSE Train": mse_tr,
                            "MSE Test": mse_te,
                            "RMSE Train": rmse_tr,
                            "RMSE Test": rmse_te,
                        })

                        # Add mean X
                        for c, v in mean_x.items():
                            d[c] = v

                        # Add betas
                        for feat_name in present_cols:
                            d[f"Beta_{feat_name}"] = raw_coefs_dict.get(feat_name, 0)

                        fold_results.append(d)

                        # FIX: Store predictions with consistent indices (original scale)
                        test_indices_for_group = np.where(group_mask_te)[0]
                        test_original_indices = [te_index_map[i] for i in test_indices_for_group]
                        pr = gdf.loc[test_original_indices].copy()
                        pr["Actual"] = y_te_group_original.values
                        pr["Predicted"] = y_pred_te_group
                        pr["Model"] = mname
                        pr["Fold"] = fold_id
                        preds_records.append(pr)

            # Report fold results: individual folds + aggregated
            if fold_results:
                fold_df = pd.DataFrame(fold_results)

                # Add individual fold rows
                rows.append(fold_df)

                # Create aggregated row (average across folds)
                # Identify keys
                key_cols = [col for col in fold_df.columns if col in (filter_keys_for_stacking + mdl.group_keys) or col == 'Model']

                # Numeric columns to average
                numeric_cols = fold_df.select_dtypes(include=[np.number]).columns.tolist()
                numeric_cols = [col for col in numeric_cols if col != 'Fold']

                # String columns to take first
                string_cols = [col for col in fold_df.columns if col not in numeric_cols and col not in key_cols and col != 'Fold']

                # Aggregate
                agg_dict = {}
                for col in numeric_cols:
                    agg_dict[col] = 'mean'
                for col in string_cols:
                    agg_dict[col] = 'first'

                aggregated = fold_df.groupby(key_cols).agg(agg_dict).reset_index()
                aggregated['Fold'] = 'Avg'  # Mark as average row
                rows.append(aggregated)

    # Clear progress
    progress_bar.empty()
    status_text.empty()

    total_time = time.time() - start_time
    st.success(f"✅ Completed in {total_time:.1f} seconds")

    if not rows:
        return None, None, None

    # Combine results
    results_df = pd.concat(rows, ignore_index=True)

    # Order columns (include Fold right after Model)
    front = grouping_keys + ["Model", "Fold"]
    metric_block = ["B0 (Original)",
                    "R2 Train", "R2 Test", "R2 Holdout",
                    "MAPE Train", "MAPE Test", "MAPE Holdout",
                    "MAE Train", "MAE Test", "MAE Holdout",
                    "MSE Train", "MSE Test", "MSE Holdout",
                    "RMSE Train", "RMSE Test", "RMSE Holdout"]

    mean_x_cols = [c for c in results_df.columns if c not in front + metric_block and not c.startswith("Beta_") and not c.startswith("Mean_")]
    beta_cols = [c for c in results_df.columns if c.startswith("Beta_")]
    mean_cols = [c for c in results_df.columns if c.startswith("Mean_")]

    existing_cols = []
    for col_group in [front, metric_block, mean_cols, beta_cols, mean_x_cols]:
        existing_cols.extend([c for c in col_group if c in results_df.columns])

    # Ensure we include any remaining columns
    for col in results_df.columns:
        if col not in existing_cols:
            existing_cols.append(col)

    results_df = results_df[existing_cols]

    preds_df = pd.concat(preds_records, ignore_index=True) if preds_records else None
    optimized_lambda_df = None

    # ═══════════════════════════════════════════════════════════════════════════
    # ENSEMBLE MODEL CREATION (if enabled)
    # ═══════════════════════════════════════════════════════════════════════════
    ensemble_df = None
    if enable_ensemble:
        st.info("🔄 Creating ensemble models from CV results (averaging coefficients across all models)...")

        ensembles = create_ensemble_model_from_results(
            results_df,
            grouping_keys,
            X_columns,
            weight_metric=ensemble_weight_metric,
            filter_r2_min=ensemble_filter_r2_min,
            filter_mape_max=ensemble_filter_mape_max,
            filter_mae_max=ensemble_filter_mae_max,
            filter_positive_features=ensemble_filter_positive_features,
            filter_negative_features=ensemble_filter_negative_features
        )

        if ensembles:
            # Convert ensemble dict to DataFrame
            ensemble_rows = []
            for combo_key, ensemble_data in ensembles.items():
                row = {}

                # Parse combination key back to individual keys
                if grouping_keys and " | " in combo_key:
                    parts = combo_key.split(" | ")
                    for part in parts:
                        if "=" in part:
                            k, v = part.split("=", 1)
                            row[k] = v
                elif grouping_keys and len(grouping_keys) == 1:
                    row[grouping_keys[0]] = combo_key.split("=")[1] if "=" in combo_key else combo_key

                row['Model'] = 'Weighted Ensemble'
                row['Fold'] = 'Ensemble'
                row['B0 (Original)'] = ensemble_data['ensemble_intercept']

                # Add ensemble betas
                for beta_name, beta_value in ensemble_data['ensemble_betas'].items():
                    row[beta_name] = beta_value

                # Add ensemble metrics
                for metric_key, metric_value in ensemble_data.items():
                    if metric_key.startswith('ensemble_'):
                        # Convert ensemble_r2_test -> R2 Test format
                        metric_name = metric_key.replace('ensemble_', '').replace('_', ' ').title()
                        row[metric_name] = metric_value

                # Add metadata
                row['Num_Models'] = ensemble_data['num_models']
                row['Best_Model'] = ensemble_data['model_names'][ensemble_data['best_model_idx']] if ensemble_data['model_names'] else ''
                row['Weight_Concentration'] = ensemble_data['weight_concentration']

                ensemble_rows.append(row)

            ensemble_df = pd.DataFrame(ensemble_rows)

            # Append ensemble results to main results
            results_df = pd.concat([results_df, ensemble_df], ignore_index=True)

            st.success(f"✅ Created {len(ensembles)} ensemble models")

    return results_df, preds_df, optimized_lambda_df, ensemble_df
