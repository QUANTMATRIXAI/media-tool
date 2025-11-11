import sys
import types

import numpy as np
import pandas as pd


def _install_streamlit_stub() -> None:
    """Provide a minimal Streamlit stub so algorithmic code can be imported in tests."""
    if "streamlit" in sys.modules:
        return

    class _StreamlitStub(types.SimpleNamespace):
        def set_page_config(self, *args, **kwargs):
            return None

        def markdown(self, *args, **kwargs):
            return None

        def cache_data(self, func=None, **kwargs):
            def decorator(fn):
                return fn

            if callable(func):
                return decorator(func)
            return decorator

        def __getattr__(self, name):
            def _no_op(*args, **kwargs):
                return None

            return _no_op

    sys.modules["streamlit"] = _StreamlitStub()


_install_streamlit_stub()

from tv_kalman_app import (  # noqa: E402  pylint: disable=wrong-import-position
    ConstrainedTVLinearKalman,
    CustomConstrainedRidge,
    run_model_for_product,
)


def test_constrained_ridge_enforces_sign_constraints():
    rng = np.random.default_rng(0)
    x_pos = np.linspace(0, 2, 120)
    x_neg = np.linspace(1, -1, 120)
    X = np.column_stack([x_pos, x_neg])
    y = 3.0 * x_pos - 2.0 * x_neg + rng.normal(0, 0.05, size=x_pos.shape[0])

    model = CustomConstrainedRidge(
        l2_penalty=0.01,
        learning_rate=0.01,
        iterations=4000,
        non_negative_features={"pos"},
        non_positive_features={"neg"},
    )
    model.fit(X, y, feature_names=["pos", "neg"])
    preds = model.predict(X)

    assert model.coef_[0] >= -1e-6
    assert model.coef_[1] <= 1e-6
    assert np.mean((preds - y) ** 2) < 0.2


def test_kalman_filter_tracks_dynamic_coefficients():
    rng = np.random.default_rng(42)
    n = 80
    trend = np.linspace(0, 3, n)
    time = np.linspace(0, 4 * np.pi, n)
    intercept = 10 + 0.5 * np.sin(time)
    slope = 1.5 + 0.3 * np.cos(time)

    X = np.column_stack([np.ones(n), trend])
    y = intercept + slope * trend + rng.normal(0, 0.2, size=n)

    kf = ConstrainedTVLinearKalman(
        n_features=2,
        q=1e-3,
        r=0.5,
        init_cov=1e2,
        adaptive=True,
        min_pred=None,
        max_pred=None,
        use_log=False,
    )
    _, y_pred = kf.fit(X, y, feature_names=["Intercept", "Trend"])

    from sklearn.metrics import r2_score

    assert r2_score(y, y_pred) > 0.9
    assert len(kf.q_history) == n
    assert len(kf.r_history) == n


def test_run_model_manual_selection_flow():
    rng = np.random.default_rng(7)
    n = 24
    dates = pd.date_range("2022-01-01", periods=n, freq="W")
    df = pd.DataFrame(
        {
            "Product title": ["Widget"] * n,
            "Date": dates,
            "google_trends": np.linspace(0, 3, n) + rng.normal(0, 0.2, size=n),
            "Product variant price": 50 + rng.normal(0, 0.5, size=n),
            "Category Discount": rng.uniform(0, 0.3, size=n),
            "promo": rng.binomial(1, 0.4, size=n),
        }
    )
    df["target"] = (
        25
        + 2.5 * df["google_trends"]
        - 1.2 * df["Category Discount"]
        + 2.0 * df["promo"]
        + rng.normal(0, 0.3, size=n)
    )

    mandatory_vars = ["google_trends", "Product variant price", "Category Discount"]
    all_vars = ["promo"] + mandatory_vars

    result = run_model_for_product(
        df=df,
        product_name="Widget",
        target_var="target",
        all_vars=all_vars,
        mandatory_vars=mandatory_vars,
        max_vars=5,
        standardize=True,
        auto_tune=False,
        adaptive_qr=False,
        q_val=1e-4,
        r_val=1.0,
        ridge_alpha_val=1.0,
        non_negative_constraints=None,
        non_positive_constraints=None,
        forward_selection_enabled=False,
        manual_vars=["promo"],
    )

    assert result is not None
    assert set(result["Selected Variables"]) == set(all_vars)
    assert result["R2"] > 0.8
