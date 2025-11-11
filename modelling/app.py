"""
Streamlit UI Application for Regression Modeling

This module provides the user interface for the regression modeling application.
It allows users to upload data, configure models, apply constraints, and visualize results.

The application supports:
- Multiple regression models (Linear, Ridge, Lasso, ElasticNet, Bayesian, Custom Constrained)
- Group-specific modeling with stacking
- Coefficient constraints (positive/negative)
- Model ensembling with weighted averaging
- Auto-residualization for handling multicollinearity
- Time series visualization and comparison
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.metrics import r2_score

from models import (
    CustomConstrainedRidge,
    ConstrainedLinearRegression,
    RecursiveLeastSquaresRegressor,
    StackedInteractionModel,
    StatsMixedEffectsModel
)
from pipeline import run_model_pipeline
from utils import safe_mape

warnings.filterwarnings('ignore')


def main():
    st.set_page_config(page_title="Modeling App", layout="wide")

    st.title("🎯 Regression Modeling App")

    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Sidebar
    with st.sidebar:
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.session_state.data = df
                st.success(f"✅ {len(df)} rows × {len(df.columns)} cols")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                return

        if st.session_state.data is not None:
            st.markdown("---")
            st.metric("Total Rows", len(st.session_state.data))
            st.metric("Total Columns", len(st.session_state.data.columns))

    if st.session_state.data is None:
        st.info("👆 Upload a file to begin")
        return

    # Always work off the most recent dataframe stored in session state
    df = st.session_state.data

    st.markdown("---")

    # Configuration
    st.header("Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1️⃣ Grouping Keys")
        available_cols = list(df.columns)
        selected_grouping_keys = st.multiselect(
            "Select grouping columns:",
            options=available_cols,
            default=[],
            help="Each unique combination gets its own model"
        )

        if selected_grouping_keys:
            combo_counts = df.groupby(selected_grouping_keys).size().reset_index(name='Count')
            st.caption(f"📌 {len(combo_counts)} unique combinations")

    with col2:
        st.subheader("2️⃣ Target & Predictors")
        numeric_cols = list(df.select_dtypes(include=[np.number]).columns)

        target_col = st.selectbox(
            "🎯 Target Variable:",
            options=numeric_cols,
            help="What to predict"
        )

        available_predictors = [c for c in numeric_cols if c != target_col and c not in selected_grouping_keys]
        default_predictors = st.session_state.get(
            'selected_predictors',
            available_predictors[:min(5, len(available_predictors))]
        )
        # Ensure defaults exist in current options
        default_predictors = [p for p in default_predictors if p in available_predictors]
        selected_predictors = st.multiselect(
            "📊 Predictors:",
            options=available_predictors,
            default=default_predictors,
            help="Features for the model"
        )
        st.session_state['selected_predictors'] = selected_predictors

    # ═══════════════════════════════════════════════════════════════════════════
    # RESIDUALIZATION FEATURE (Automatic per product/brand)
    # ═══════════════════════════════════════════════════════════════════════════
    residualization_mapping = {}
    with st.expander("🔧 Advanced: Auto-Residualization (Remove Multicollinearity)", expanded=False):
        st.markdown("""
        **Automatically remove correlation** by finding product-specific primary variables.

        **How it works:**
        - For each product (e.g., "paw diamond necklace"), finds its specific column (e.g., "paw diamond necklace_meta_impression")
        - Uses that as primary variable for that product
        - Residualizes all other predictors against it
        - Falls back to a general column (e.g., "impressions") if product-specific column doesn't exist
        """)

        enable_auto_residualization = st.checkbox(
            "Enable Auto-Residualization",
            value=False,
            help="Automatically detect primary variable per product/brand and residualize others"
        )

        # Clear residualisation state if checkbox is unchecked
        if not enable_auto_residualization and st.session_state.get('residualization_applied', False):
            st.session_state['residualization_applied'] = False
            st.session_state.pop('df_residualized', None)
            st.session_state.pop('selected_predictors_residualized', None)
            st.session_state.pop('residualization_mapping', None)
            st.info("🔄 Residualisation disabled - using original data")

        if enable_auto_residualization and selected_grouping_keys and len(selected_predictors) > 1:
            # Identify fallback primary variable - prefer general "impressions" column if present
            default_fallback = next(
                (
                    col for col in numeric_cols
                    if col.lower().strip() == "impressions"
                ),
                None
            )
            fallback_primary = st.selectbox(
                "📌 Fallback Primary Variable (if no product-specific column found):",
                options=numeric_cols,
                index=numeric_cols.index(default_fallback) if default_fallback and default_fallback in numeric_cols else 0,
                help="Column to use when no product-specific match exists (defaults to 'impressions' when available)"
            )

            # Auto-detect and show mapping
            st.markdown("**🔍 Detected Primary Variables:**")

            # Get unique values from first grouping key
            if selected_grouping_keys:
                first_group_key = selected_grouping_keys[0]
                unique_groups = df[first_group_key].unique()

                detected_mappings = {}

                # Candidate columns for impressions
                def _normalize(text: str) -> str:
                    text = text.lower().strip()
                    text = text.replace('-', ' ')
                    text = text.replace('/', ' ')
                    text = text.replace('&', ' and ')
                    text = text.replace("'", '')
                    text = "".join(ch if ch.isalnum() or ch == ' ' else ' ' for ch in text)
                    return "_".join(text.split())

                impression_cols = [
                    col for col in numeric_cols
                    if '_meta_impression' in col.lower()
                ]
                st.caption(f"**Matched {len(impression_cols)} impression column candidates**")

                for group_val in unique_groups:
                    if pd.notna(group_val):
                        group_key = str(group_val)
                        group_norm = _normalize(group_key)

                        # Try to find product-specific column
                        primary_found = None
                        best_match_score = 0

                        for col in impression_cols:
                            col_norm = _normalize(col.replace('_meta_impression', ''))
                            if not col_norm:
                                continue
                            if col_norm == group_norm:
                                primary_found = col
                                best_match_score = len(group_norm)
                                break
                            if group_norm in col_norm or col_norm in group_norm:
                                score = min(len(group_norm), len(col_norm))
                                if score > best_match_score:
                                    primary_found = col
                                    best_match_score = score

                        # Use fallback if not found (e.g., "impressions")
                        if primary_found is None:
                            primary_found = fallback_primary

                        detected_mappings[group_key] = primary_found

                # Show detected mappings in table format
                matched_count = sum(1 for v in detected_mappings.values() if v != fallback_primary)
                st.caption(f"**Matched {matched_count} of {len(detected_mappings)} products to specific impression columns**")

                # Display mappings in a dataframe table instead of list
                mapping_data = []
                for group, primary in detected_mappings.items():
                    status = "✅ Matched" if primary != fallback_primary else "⚠️ Fallback"
                    mapping_data.append({
                        'Product': group,
                        'Primary Variable': primary,
                        'Status': status
                    })

                mapping_df = pd.DataFrame(mapping_data)
                st.dataframe(mapping_df, use_container_width=True, height=400)

                # Apply residualization per group
                if st.button("✅ Apply Auto-Residualization", key="apply_residualization"):
                    df_residual = df.copy()
                    residualization_stats = []

                    for group_val, primary_var in detected_mappings.items():
                        # Filter to this group
                        group_mask = df_residual[first_group_key].astype(str) == str(group_val)
                        group_df = df_residual[group_mask]

                        if len(group_df) < 10:
                            continue

                        # Variables to residualize (all except primary and target)
                        vars_to_residualize = [
                            p for p in selected_predictors
                            if p != primary_var and p != target_col
                        ]

                        for var in vars_to_residualize:
                            if var in group_df.columns and primary_var in group_df.columns:
                                valid_mask = group_df[[primary_var, var]].notna().all(axis=1)

                                if valid_mask.sum() > 5:
                                    X_primary = group_df.loc[valid_mask, primary_var].values.reshape(-1, 1)
                                    y_secondary = group_df.loc[valid_mask, var].values

                                    # Fit regression
                                    lr = LinearRegression()
                                    lr.fit(X_primary, y_secondary)

                                    # Calculate residuals for this group
                                    X_all = group_df[primary_var].fillna(0).values.reshape(-1, 1)
                                    predicted = lr.predict(X_all)
                                    residuals = group_df[var].fillna(0).values - predicted

                                    # Store in original dataframe with product-specific name
                                    residual_col_name = f"{var}_residual"
                                    df_residual.loc[group_mask, residual_col_name] = residuals

                                    residualization_stats.append({
                                        'Group': group_val,
                                        'Primary': primary_var,
                                        'Residualized': var,
                                        'New Column': residual_col_name,
                                        'R²': lr.score(X_primary, y_secondary)
                                    })

                    if residualization_stats:
                        # Store the residualised dataframe in session state
                        st.session_state['df_residualized'] = df_residual

                        # Update predictor list - replace original with residual versions
                        residualized_var_names = set([s['Residualized'] for s in residualization_stats])

                        # Remove original variables and add their residual versions
                        new_selected_predictors = [p for p in selected_predictors
                                            if p not in residualized_var_names]

                        # Add unique residual columns
                        residual_cols = list(set([s['New Column'] for s in residualization_stats]))
                        new_selected_predictors.extend(residual_cols)

                        # Store both the new predictors and a flag in session state
                        st.session_state['selected_predictors_residualized'] = new_selected_predictors
                        st.session_state['residualization_applied'] = True
                        st.session_state['residualization_mapping'] = residualization_mapping

                        st.success(f"✅ Applied residualization to {len(residualized_var_names)} variable(s) across {len(detected_mappings)} groups")

                        # Show detailed stats table
                        stats_df = pd.DataFrame(residualization_stats)
                        st.dataframe(stats_df, use_container_width=True, height=300)

                        # Show residual counts per group
                        group_counts = (
                            stats_df.groupby('Group')['Residualized']
                            .count()
                            .reset_index()
                            .rename(columns={'Residualized': 'Residualized Features'})
                            .sort_values('Residualized Features', ascending=False)
                        )
                        st.caption("**Residualized features per product**")
                        st.dataframe(group_counts, use_container_width=True, height=250)

                        st.info(f"""
                        **Next Steps:**
                        - Created {len(residual_cols)} residual column(s)
                        - Original variables replaced with their residualized versions
                        - Constraints and standardization will now use residualized columns
                        - Products with specific columns: Use their primary + residuals
                        - Products using fallback ({fallback_primary}): All use same primary + residuals
                        """)

    # Check if residualization was applied and update data and predictors accordingly
    if st.session_state.get('residualization_applied', False):
        df_working = st.session_state.get('df_residualized', df)
        selected_predictors_working = st.session_state.get('selected_predictors_residualized', selected_predictors)

        # Show a notice that we're using residualized data
        st.success("📊 Using residualized data for all subsequent operations")
    else:
        df_working = df
        selected_predictors_working = selected_predictors

    st.markdown("---")





    col3, col4 = st.columns(2)

    with col3:
        st.subheader("3️⃣ Constraints")
        positive_constraints = st.multiselect(
            "Force ≥ 0:",
            options=selected_predictors_working,
            default=[],
            help="Variables that must have positive coefficients"
        )

        available_for_negative = [p for p in selected_predictors_working if p not in positive_constraints]
        negative_constraints = st.multiselect(
            "Force ≤ 0:",
            options=available_for_negative,
            default=[],
            help="Variables that must have negative coefficients"
        )

    with col4:
        st.subheader("4️⃣ Model Settings")
        k_folds = st.number_input(
            "CV Folds:",
            min_value=2,
            max_value=20,
            value=5
        )

        standardize_cols = st.multiselect(
            "Standardize:",
            options=selected_predictors_working,
            default=[]
        )

        log_transform_y = st.checkbox(
            "🔄 Log Transform Y Variable: log(y+1)",
            value=False,
            help="Apply log(y+1) transformation to target variable. Helps with:\n• Zero values in count data\n• Reducing impact of outliers\n• Stabilizing variance\n• Improving model fit for small discrete values"
        )

        min_y_share_pct = st.number_input(
            "Min Y Share % (Filter Groups):",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.1,
            help="Only train models on groups where Y variable sum is at least this % of total Y.\n"
                 "• Focuses on significant segments\n"
                 "• Improves model quality by excluding tiny groups\n"
                 "• Speeds up computation\n"
                 "Example: 1% means group must contribute ≥1% of total Y"
        )

    st.markdown("---")

    # Stacking configuration
    use_stacked = st.checkbox(
        "Enable Stacking (Group-Specific Coefficients)",
        value=False,
        help="Creates models with interaction terms for selected keys"
    )

    if use_stacked and selected_grouping_keys:
        st.info("📌 **Stacking Strategy:** Select which keys to filter by vs which to use for interactions")

        col_stack1, col_stack2 = st.columns(2)

        with col_stack1:
            filter_keys_for_stacking = st.multiselect(
                "🔍 Filter By (separate models):",
                options=selected_grouping_keys,
                default=[selected_grouping_keys[0]] if selected_grouping_keys else [],
                help="Create separate models for each unique value of these keys"
            )

        with col_stack2:
            remaining_keys = [k for k in selected_grouping_keys if k not in filter_keys_for_stacking]
            if remaining_keys:
                st.multiselect(
                    "🔄 Interaction Keys (within models):",
                    options=remaining_keys,
                    default=remaining_keys,
                    disabled=True,
                    help="These keys will create interaction terms within each filtered model"
                )
                stacking_keys = remaining_keys
            else:
                st.warning("⚠️ No keys left for interactions. Select fewer filter keys.")
                stacking_keys = []
    else:
        filter_keys_for_stacking = []
        stacking_keys = []

    st.markdown("---")

    # RLS configuration
    with st.expander("🔄 Recursive Least Squares Settings", expanded=False):
        st.caption("Configure how the RLS model updates coefficients over time.")
        col_rls1, col_rls2 = st.columns(2)

        with col_rls1:
            rls_forgetting = st.slider(
                "Forgetting Factor (λ)",
                min_value=0.80,
                max_value=1.00,
                value=0.99,
                step=0.01,
                help="Controls how quickly older observations are discounted. 1.0 = no forgetting"
            )

        with col_rls2:
            rls_initial_cov = st.number_input(
                "Initial Covariance",
                min_value=1.0,
                max_value=1e6,
                value=1000.0,
                step=100.0,
                help="Larger values allow larger coefficient adjustments during early updates"
            )

        rls_store_history = st.checkbox(
            "Store coefficient history",
            value=False,
            help="Track coefficient updates after each observation (useful for diagnostics)"
        )

    # Model selection
    st.subheader("5️⃣ Select Models")

    base_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5),
        "Bayesian Ridge": BayesianRidge(),
        "Recursive Least Squares": RecursiveLeastSquaresRegressor(
            forgetting_factor=rls_forgetting,
            initial_covariance=rls_initial_cov,
            store_history=rls_store_history
        ),
        "Custom Constrained Ridge": CustomConstrainedRidge(
            l2_penalty=0.1,
            learning_rate=0.001,
            iterations=10000,
            non_negative_features=positive_constraints,
            non_positive_features=negative_constraints
        ),
        "Constrained Linear Regression": ConstrainedLinearRegression(
            learning_rate=0.001,
            iterations=10000,
            non_negative_features=positive_constraints,
            non_positive_features=negative_constraints
        )
    }

    col_models = st.columns(4)
    selected_models = []

    for idx, model_name in enumerate(base_models.keys()):
        with col_models[idx % 4]:
            if st.checkbox(model_name, value=(idx < 3), key=f"model_{idx}"):
                selected_models.append(model_name)

    # Collect selected models
    models_to_run = {model_name: base_models[model_name] for model_name in selected_models}

    # Add stacked versions
    if use_stacked and stacking_keys:
        stacked_models = {}
        for name, model in models_to_run.items():
            # Add stacked versions for non-stacked models
            if not isinstance(model, StackedInteractionModel):
                is_constrained = isinstance(model, (CustomConstrainedRidge, ConstrainedLinearRegression))
                stacked_models[f"Stacked {name}"] = StackedInteractionModel(
                    base_model=model,
                    group_keys=stacking_keys,
                    enforce_combined_constraints=is_constrained
                )

        # Update models_to_run with stacked versions
        models_to_run.update(stacked_models)


    # Display summary
    base_count = len(selected_models)
    stacked_count = sum(1 for k in models_to_run.keys() if k.startswith('Stacked'))

    st.info(
        f"📊 Will run **{len(models_to_run)}** model variants: "
        f"{base_count} base models" +
        (f" + {stacked_count} stacked" if stacked_count > 0 else "")
    )

    st.markdown("---")

    # Ensemble Settings
    with st.expander("🎯 Ensemble Settings (Model Averaging)", expanded=False):
        st.caption("Combine multiple models per combination using MAPE-weighted averaging")

        col_ens1, col_ens2 = st.columns(2)

        with col_ens1:
            enable_ensemble = st.checkbox(
                "Enable Ensemble",
                value=False,
                help="Create weighted ensemble models by averaging coefficients across all models"
            )

            ensemble_weight_metric = st.selectbox(
                "Weighting Metric:",
                options=['MAPE Test', 'MAE Test'],
                index=0,
                help="Metric used for weighting models (lower MAPE/MAE = higher weight)"
            )

        with col_ens2:
            st.caption("Optional: Filter models before ensemble")

            use_r2_filter = st.checkbox("Filter by R² Test ≥", value=False)
            ensemble_r2_min = st.number_input(
                "Min R² Test:",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                disabled=not use_r2_filter
            ) if use_r2_filter else None

            use_mape_filter = st.checkbox("Filter by MAPE Test ≤", value=False)
            ensemble_mape_max = st.number_input(
                "Max MAPE Test (%):",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=5.0,
                disabled=not use_mape_filter
            ) if use_mape_filter else None

            use_mae_filter = st.checkbox("Filter by MAE Test ≤", value=False)
            ensemble_mae_max = st.number_input(
                "Max MAE Test:",
                min_value=0.0,
                value=100.0,
                step=10.0,
                disabled=not use_mae_filter
            ) if use_mae_filter else None

        # Sign-based filtering for ensemble
        st.markdown("---")
        st.caption("**🔍 Sign-Based Filtering**: Only include models where coefficients match expected signs")

        use_sign_filter = st.checkbox(
            "Enable Sign Filtering",
            value=False,
            help="Filter out models where coefficients don't match the expected positive/negative constraints"
        )

        if use_sign_filter:
            col_sign1, col_sign2 = st.columns(2)

            with col_sign1:
                ensemble_positive_features = st.multiselect(
                    "Must be Positive (≥ 0):",
                    options=selected_predictors_working,
                    default=positive_constraints if positive_constraints else [],
                    help="Only include models where these features have positive coefficients"
                )

            with col_sign2:
                available_for_negative_ensemble = [p for p in selected_predictors_working if p not in ensemble_positive_features]
                ensemble_negative_features = st.multiselect(
                    "Must be Negative (≤ 0):",
                    options=available_for_negative_ensemble,
                    default=[c for c in negative_constraints if c in available_for_negative_ensemble] if negative_constraints else [],
                    help="Only include models where these features have negative coefficients"
                )

            if ensemble_positive_features or ensemble_negative_features:
                st.info(f"📌 Will filter models: {len(ensemble_positive_features)} features must be ≥0, {len(ensemble_negative_features)} features must be ≤0")
        else:
            ensemble_positive_features = None
            ensemble_negative_features = None

        if enable_ensemble:
            st.info("📌 Ensemble will create one weighted model per combination by averaging all individual model coefficients")

    st.markdown("---")

    # Run button
    if not selected_predictors_working:
        st.error("❌ Please select at least one predictor")
        return

    if st.button("▶️ RUN MODELS", type="primary", use_container_width=True):
        # ═══════════════════════════════════════════════════════════════
        # Initialize stores (clear old data from previous runs)
        # ═══════════════════════════════════════════════════════════════
        st.session_state.beta_history_store = {}

        # Store flags in session state so they're available in display section
        st.session_state.enable_ensemble_flag = enable_ensemble

        # Use the working dataframe (either residualized or original)
        df_to_use = df_working
        predictors_to_use = selected_predictors_working

        # Show workflow info
        if enable_ensemble:
            st.info("🔄 **Workflow**: CV on all models → Create weighted ensemble")

        with st.spinner("Running models..."):
            st.session_state.optimized_lambdas = None
            results_df, predictions_df, lambda_df, ensemble_df = run_model_pipeline(
                df=df_to_use,
                grouping_keys=selected_grouping_keys,
                X_columns=predictors_to_use,
                target_col=target_col,
                k_folds=k_folds,
                std_cols=standardize_cols,
                models_dict=models_to_run,
                use_stacked=use_stacked,
                stacking_keys=stacking_keys,
                filter_keys_for_stacking=filter_keys_for_stacking,
                log_transform_y=log_transform_y,
                min_y_share_pct=min_y_share_pct,
                enable_ensemble=enable_ensemble,
                ensemble_weight_metric=ensemble_weight_metric,
                ensemble_filter_r2_min=ensemble_r2_min if use_r2_filter else None,
                ensemble_filter_mape_max=ensemble_mape_max if use_mape_filter else None,
                ensemble_filter_mae_max=ensemble_mae_max if use_mae_filter else None,
                ensemble_filter_positive_features=ensemble_positive_features if use_sign_filter else None,
                ensemble_filter_negative_features=ensemble_negative_features if use_sign_filter else None
            )

            if results_df is not None:
                st.session_state.results = results_df
                st.session_state.predictions = predictions_df
                st.session_state.optimized_lambdas = lambda_df
                st.session_state.ensemble_results = ensemble_df


    # Display results
    if st.session_state.results is not None:
        st.markdown("---")
        st.header("📈 Results")

        results_df = st.session_state.results

        st.subheader("📋 Detailed Results (Folds & Aggregates)")
        st.caption("Includes per-fold rows, holdout metrics, coefficients, and feature means")
        st.dataframe(results_df, use_container_width=True, height=min(600, 100 + len(results_df) * 20))

        csv_results = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_results,
            file_name="modeling_results.csv",
            mime="text/csv",
            key='download_model_results'
        )


        # Display ensemble results
        ensemble_df = st.session_state.get('ensemble_results')
        if ensemble_df is not None and not ensemble_df.empty:
            with st.expander("🎯 Ensemble Models (Weighted Average)", expanded=True):
                st.caption("Weighted ensemble models created by averaging coefficients across individual models")

                # Show key metrics
                st.markdown("### 📊 Ensemble Summary")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Combinations", len(ensemble_df))

                with col2:
                    avg_models = ensemble_df['Num_Models'].mean() if 'Num_Models' in ensemble_df.columns else 0
                    st.metric("Avg Models per Combination", f"{avg_models:.1f}")

                with col3:
                    avg_concentration = ensemble_df['Weight_Concentration'].mean() if 'Weight_Concentration' in ensemble_df.columns else 0
                    st.metric("Avg Weight Concentration", f"{avg_concentration:.2%}")

                st.markdown("---")

                # Display full ensemble data
                st.dataframe(ensemble_df, use_container_width=True, height=min(400, 100 + len(ensemble_df) * 22))

                csv_ensemble = ensemble_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Ensemble Results",
                    data=csv_ensemble,
                    file_name="ensemble_models.csv",
                    mime="text/csv",
                    key='download_ensemble_results'
                )

        # Time series visualization of predictions
        if st.session_state.predictions is not None:
            st.subheader("📈 Predicted vs Actual Over Time")

            predictions_df = st.session_state.predictions.copy()

            # Check if date column exists
            date_columns = [col for col in predictions_df.columns if 'date' in col.lower() or 'time' in col.lower() or col.lower() in ['week', 'month', 'year', 'period']]

            if date_columns:
                col1, col2 = st.columns(2)

                with col1:
                    # Date column selection
                    selected_date_col = st.selectbox(
                        "Select Date Column:",
                        options=date_columns,
                        index=0,
                        key='date_col_selector'
                    )

                with col2:
                    # Get grouping columns (exclude prediction-specific columns and numeric/data columns)
                    grouping_cols = [col for col in predictions_df.columns
                                if col not in ['Actual', 'Predicted', 'Model', 'Fold', selected_date_col]]

                    # Filter to keep only product/category identifier columns (not numeric data)
                    product_cols = []
                    for col in grouping_cols:
                        # Keep only columns that look like identifiers (not pure numeric data)
                        if predictions_df[col].dtype == 'object' or predictions_df[col].nunique() < 100:
                            product_cols.append(col)

                    # If we have product columns, use them; otherwise use first grouping column
                    if product_cols:
                        # Use only the first column (usually product name)
                        predictions_df['_group_id'] = predictions_df[product_cols[0]].astype(str)
                        unique_groups = sorted(predictions_df['_group_id'].unique())

                        selected_group = st.selectbox(
                            "Select Product:",
                            options=unique_groups,
                            index=0,
                            key='group_selector'
                        )
                    elif grouping_cols:
                        predictions_df['_group_id'] = predictions_df[grouping_cols[0]].astype(str)
                        unique_groups = sorted(predictions_df['_group_id'].unique())

                        selected_group = st.selectbox(
                            "Select Product:",
                            options=unique_groups,
                            index=0,
                            key='group_selector'
                        )
                    else:
                        selected_group = None

                # Convert date column
                try:
                    predictions_df[selected_date_col] = pd.to_datetime(predictions_df[selected_date_col])
                except:
                    pass  # Keep as is if conversion fails

                # Filter by selected group
                if selected_group is not None:
                    group_data = predictions_df[predictions_df['_group_id'] == selected_group].copy()
                else:
                    group_data = predictions_df.copy()

                # Aggregate predictions by date and model (average across folds)
                agg_data = group_data.groupby([selected_date_col, 'Model']).agg({
                    'Actual': 'mean',
                    'Predicted': 'mean'
                }).reset_index()

                agg_data = agg_data.sort_values(selected_date_col)

                # Get unique models
                unique_models = sorted(agg_data['Model'].unique())

                # Create single plot with all models
                fig = go.Figure()

                # Color palette for models
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

                # Add actual values (only once - all models have same actuals)
                actual_data = agg_data.drop_duplicates(subset=[selected_date_col])[
                    [selected_date_col, 'Actual']
                ].sort_values(selected_date_col)

                fig.add_trace(go.Scatter(
                    x=actual_data[selected_date_col],
                    y=actual_data['Actual'],
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='black', width=3),
                    marker=dict(size=8, symbol='circle')
                ))

                # Add predicted values for each model
                metrics_list = []
                for idx, model in enumerate(unique_models):
                    model_data = agg_data[agg_data['Model'] == model].copy()
                    model_data = model_data.sort_values(selected_date_col)

                    # Calculate metrics
                    r2 = r2_score(model_data['Actual'], model_data['Predicted'])
                    mae = np.mean(np.abs(model_data['Actual'] - model_data['Predicted']))
                    rmse = np.sqrt(np.mean((model_data['Actual'] - model_data['Predicted'])**2))
                    mape = safe_mape(model_data['Actual'], model_data['Predicted'])

                    metrics_list.append({
                        'Model': model,
                        'R²': r2,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape
                    })

                    # Get color
                    color = colors[idx % len(colors)]

                    fig.add_trace(go.Scatter(
                        x=model_data[selected_date_col],
                        y=model_data['Predicted'],
                        mode='lines+markers',
                        name=f'{model} (R²={r2:.3f})',
                        line=dict(color=color, width=2, dash='dash'),
                        marker=dict(size=6, symbol='circle'),
                        hovertemplate=f'<b>{model}</b><br>Date: %{{x}}<br>Predicted: %{{y:.2f}}<extra></extra>'
                    ))


                # Update layout
                title_text = f"<b>{selected_group if selected_group else 'All Data'}</b>"

                fig.update_layout(
                    title=title_text,
                    xaxis_title=selected_date_col,
                    yaxis_title="Value",
                    hovermode='x unified',
                    height=600,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor="rgba(255,255,255,0.9)"
                    ),
                    margin=dict(t=80, b=60, l=60, r=20)
                )

                st.plotly_chart(fig, use_container_width=True)

                # Show summary table
                st.subheader("📊 Model Performance Summary")
                if metrics_list:
                    summary_df = pd.DataFrame(metrics_list)
                    summary_df['R²'] = summary_df['R²'].apply(lambda x: f"{x:.4f}")
                    summary_df['MAE'] = summary_df['MAE'].apply(lambda x: f"{x:.2f}")
                    summary_df['RMSE'] = summary_df['RMSE'].apply(lambda x: f"{x:.2f}")
                    summary_df['MAPE'] = summary_df['MAPE'].apply(lambda x: f"{x:.2f}%")

                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

            else:
                st.info("💡 No date/time column found in predictions. Please ensure your data has a date column for time series visualization.")

        # ═══════════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    main()
