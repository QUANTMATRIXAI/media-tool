"""Safe MAPE function for modelling"""
import numpy as np

def safe_mape(y_true, y_pred):
    """Calculate MAPE safely with automatic protection against small values"""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    valid_mask = (np.abs(y_true) >= 1.0)
    if not valid_mask.any():
        return float("nan")
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    percent_errors = np.abs((y_true_valid - y_pred_valid) / y_true_valid) * 100
    percent_errors = np.minimum(percent_errors, 500.0)
    return np.mean(percent_errors)

def create_ensemble_model_from_results(*args, **kwargs):
    """Stub function - full implementation in modelling/utils.py"""
    return {}
