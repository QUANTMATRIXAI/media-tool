# Constrained Time‑Varying Linear Kalman Filter — A Practical Guide

## 1. Variance, Covariance, Covariance Matrix
- Variance Var(X) measures spread around the mean; larger variance → more dispersion.
- Covariance Cov(X,Y) measures joint linear variation; positive (move together), negative (opposite), near 0 (weak linear relation).
- Covariance matrix Σ = E[(x−μ)(x−μ)ᵀ] is symmetric PSD; its inverse is the precision matrix. Σ defines the geometry of uncertainty and the Mahalanobis distance.

## 2. Regression and Ridge
- OLS: β minimizes ∑(y−Xβ)²; unbiased (under assumptions) but can be high‑variance when X is ill‑conditioned.
- Ridge: add λ‖β‖² for stability. Bayesian view: Gaussian prior on β with covariance ∝ I/λ. Improves conditioning, shrinks coefficients.

In this project we use a ridge (or constrained ridge) warm start to initialize β before the Kalman recursion.

## 3. State‑Space Model and Kalman Filter
- Observation: y_t = x_tᵀ β_t + v_t, v_t ~ N(0, R)
- State (random walk): β_t = β_{t−1} + w_t, w_t ~ N(0, QI)

Kalman step at time t
- Predict: β_pred = β_prev; P_pred = P_prev + Q
- Innovation: y_pred = x_tᵀβ_pred; S = x_tᵀP_pred x_t + R; K = P_pred x_t / S; ν_t = y_t − y_pred
- Update (unconstrained): β_upd = β_pred + K ν_t; P_upd uses a numerically stable form

Interpretation
- Q (process noise) controls how quickly coefficients can change. Higher Q tracks fast dynamics but risks noise‑chasing.
- R (observation noise) controls how much we trust a new observation. Higher R smooths more.

## 4. Our Implementation (What, Where)
- Class: `ConstrainedTVLinearKalman` builds a time‑varying linear model with optional adaptive Q/R.
- Warm start: Ridge or custom constrained‑ridge initializes β (respects sign constraints at t=0).
- Per‑step filtering: Standard Kalman update followed by optional constraint projection.
- File: `tv_kalman_app.py` and `tv_kalman_app copy.py` (Streamlit apps).

Key points in code
- Kalman step: `_step(...)` updates β and P each time point.
- Constraint projection: `_project_state(beta_upd, P_upd)` enforces sign constraints after each update.
- Index mapping: feature names → constrained indices is computed in `fit(...)`.

## 5. Sign Constraints — Principled Projection (KKT/MAP)
Goal: enforce β_i ≥ 0 (for some i) and/or β_j ≤ 0 (for some j) at every time step using the correct uncertainty geometry.

Active set
- For non‑neg indices, mark i active if β[i] < 0.
- For non‑pos indices, mark j active if β[j] > 0.
- Build selector A for these active rows (subset of identity).

Closed‑form projection (Gaussian conditioning)
- β* = β − P Aᵀ (A P Aᵀ)⁻¹ (Aβ − 0)
- P* = P − P Aᵀ (A P Aᵀ)⁻¹ A P
- We add a small diagonal jitter to A P Aᵀ if needed for numerical stability.

Why it’s correct
- This is the exact MAP solution for a Gaussian posterior under linear equality constraints (the active inequalities). It projects in the Mahalanobis metric (defined by P⁻¹), not naïve Euclidean clamping.

## 6. Adaptive Q & R (Data‑Driven Tuning)
- R from innovations: roughly R ← (1−α_R)R + α_R(ν_t² − x_tᵀP_pred x_t), bounded to [r_min, r_max].
- Q from state increments: estimate mean square of Δβ_t and smooth into q, bounded to [q_min, q_max].

Tuning intuition
- Predictions lag changes → increase Q modestly.
- Predictions too jumpy → decrease Q and/or increase R.

## 7. Validation & Diagnostics
- Whiteness: small‑lag autocorrelations within ±1.96/√N.
- Normality: Shapiro p>0.05, mild skew/kurtosis.
- Innovation variance vs R: Var(ν) ≈ mean(R). Ratio near 1 indicates R is well‑scaled.
- Q & R stability: large oscillations suggest retuning α or bounds.

## 8. Practical Workflow
1) Select predictors (forward or manual). Standardize predictors; keep intercept unscaled.
2) Set constraints (≥0, ≤0) for selected predictors (avoid constraining intercept).
3) Use auto‑tune or start with Q≈1e−4, R≈1.0; run model.
4) Check validation: whiteness, normality, Var/R ratio, stability of Q/R.
5) Iterate: adjust Q/R or constraints where needed.

## 9. Common Pitfalls
- Overly large Q → coefficients chase noise.
- Overly small Q → lagging coefficients.
- Large mismatch Var(ν)/R → mis‑scaled R or Q.
- Constraining intercept or overlapping sign constraints (UI prevents overlap).

## 10. Where to Look in Code
- Kalman step: `ConstrainedTVLinearKalman._step`
- Projection: `ConstrainedTVLinearKalman._project_state`
- Ridge warm start: `CustomConstrainedRidge` and its `fit`
- Adaptive Q/R: inside `_step` after update
- Streamlit validation tests: Validation tab in the app

## 11. Glossary
- Innovation ν_t: difference between observation and prediction.
- Kalman gain K: fraction of the residual used to update state.
- Mahalanobis metric: distance weighted by Σ⁻¹; governs the geometry used in projection.

## 12. Summary
- You are running a time‑varying linear regression via a Kalman filter.
- Coefficients evolve with process noise Q; observations carry noise R.
- You initialize with (possibly constrained) ridge and enforce constraints at every step using a principled projection that respects covariance.
- Adaptive Q/R and validation complete the loop for robust performance.
