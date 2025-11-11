"""
Beta Format Converter
Converts Kalman modeling output format to standard optimizer beta format
"""

import pandas as pd
import numpy as np


def detect_beta_format(df):
    """
    Detect which beta format the file is in.
    
    Returns:
        'standard' - Standard optimizer format with Beta_ columns
        'kalman' - Kalman modeling output format with Intercept column
        'unknown' - Cannot determine format
    """
    columns = df.columns.tolist()
    
    # Check for Kalman format indicators
    has_intercept = 'Intercept' in columns
    has_metrics = any(col in columns for col in ['R2', 'MAE', 'RMSE', 'MAPE'])
    has_product = 'Product' in columns
    
    # Check for standard format indicators
    has_b0 = any(col.startswith('B0') for col in columns)
    has_beta_cols = any(col.startswith('Beta_') for col in columns)
    has_product_title = any('product' in col.lower() and 'title' in col.lower() for col in columns)
    
    if has_intercept and has_product and has_metrics:
        return 'kalman'
    elif (has_b0 or has_beta_cols) and has_product_title:
        return 'standard'
    else:
        return 'unknown'


def convert_kalman_to_standard(kalman_df):
    """
    Convert Kalman modeling output format to standard optimizer format.
    
    Kalman format:
        Product | R2 | MAE | RMSE | MAPE | Intercept | var1 | var2 | ...
    
    Standard format:
        Product title | B0 | Beta_var1 | Beta_var2 | ...
    
    Args:
        kalman_df: DataFrame in Kalman format
    
    Returns:
        DataFrame in standard optimizer format
    """
    # Create new dataframe
    standard_df = pd.DataFrame()
    
    # Copy Product column as "Product title"
    standard_df['Product title'] = kalman_df['Product']
    
    # Copy Intercept as B0
    if 'Intercept' in kalman_df.columns:
        standard_df['B0'] = kalman_df['Intercept']
    else:
        standard_df['B0'] = 0.0
    
    # Get all variable columns (exclude Product, metrics, and Intercept)
    exclude_cols = ['Product', 'R2', 'MAE', 'RMSE', 'MAPE', 'Intercept']
    variable_cols = [col for col in kalman_df.columns if col not in exclude_cols]
    
    # Convert variable columns to Beta_ format
    for col in variable_cols:
        # Clean up column name and add Beta_ prefix
        clean_name = col.strip()
        
        # Special handling for common patterns
        if 'impression' in clean_name.lower():
            # Keep impression columns as-is with Beta_ prefix
            if not clean_name.startswith('Beta_'):
                beta_col_name = f'Beta_{clean_name}'
            else:
                beta_col_name = clean_name
        else:
            # For other variables, add Beta_ prefix if not present
            if not clean_name.startswith('Beta_'):
                beta_col_name = f'Beta_{clean_name}'
            else:
                beta_col_name = clean_name
        
        standard_df[beta_col_name] = kalman_df[col]
    
    return standard_df


def auto_convert_beta_file(df):
    """
    Automatically detect and convert beta file format if needed.
    
    Args:
        df: Input DataFrame (any format)
    
    Returns:
        tuple: (converted_df, format_type, was_converted)
            - converted_df: DataFrame in standard format
            - format_type: 'standard', 'kalman', or 'unknown'
            - was_converted: Boolean indicating if conversion was performed
    """
    format_type = detect_beta_format(df)
    
    if format_type == 'kalman':
        converted_df = convert_kalman_to_standard(df)
        return converted_df, format_type, True
    elif format_type == 'standard':
        return df, format_type, False
    else:
        # Unknown format, return as-is
        return df, format_type, False


def get_format_info(format_type):
    """Get human-readable information about the detected format."""
    if format_type == 'kalman':
        return {
            'name': 'Kalman Modeling Output',
            'description': 'Time-varying coefficient model output with metrics and final period betas',
            'icon': '📈'
        }
    elif format_type == 'standard':
        return {
            'name': 'Standard Optimizer Format',
            'description': 'Standard beta coefficients with Beta_ prefix columns',
            'icon': '📊'
        }
    else:
        return {
            'name': 'Unknown Format',
            'description': 'Format could not be determined',
            'icon': '❓'
        }
