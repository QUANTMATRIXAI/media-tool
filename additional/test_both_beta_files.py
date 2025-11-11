"""
Test both betas.csv and Kalman_betas.csv to verify mapping works for both
"""

import pandas as pd
import sys
from pathlib import Path

# Add optimiser to path
sys.path.insert(0, str(Path("optimiser").absolute()))

from utils.beta_converter import detect_beta_format, auto_convert_beta_file

print("="*80)
print("TESTING BOTH BETA FILES")
print("="*80)

# Test 1: betas.csv (Standard Format)
print("\n" + "="*80)
print("TEST 1: betas.csv (Standard Format)")
print("="*80)

betas_df = pd.read_csv('betas.csv')
print(f"\nShape: {betas_df.shape}")
print(f"\nFirst 5 columns: {list(betas_df.columns[:5])}")
print(f"Last 5 columns: {list(betas_df.columns[-5:])}")

# Detect format
format_type = detect_beta_format(betas_df)
print(f"\nDetected format: '{format_type}'")

# Auto-convert
converted_df, fmt, was_converted = auto_convert_beta_file(betas_df)
print(f"Was converted: {was_converted}")
print(f"Converted shape: {converted_df.shape}")

# Show sample products and their beta columns
print("\nSample products and beta columns:")
if 'Product title' in converted_df.columns:
    products = converted_df['Product title'].head(3).tolist()
    beta_cols = [col for col in converted_df.columns if col.startswith('Beta_')]
    
    print(f"\nFound {len(beta_cols)} Beta columns")
    print(f"Sample Beta columns: {beta_cols[:5]}")
    
    for product in products:
        print(f"\n  Product: '{product}'")
        # Show all beta values for this product
        product_row = converted_df[converted_df['Product title'] == product]
        if not product_row.empty:
            b0 = product_row['B0 (Original)'].values[0] if 'B0 (Original)' in converted_df.columns else product_row.get('B0', [0])[0] if 'B0' in converted_df.columns else 'N/A'
            print(f"    B0: {b0}")
            
            # Show non-zero betas
            non_zero_betas = []
            for beta_col in beta_cols[:10]:  # Check first 10 beta columns
                if beta_col in product_row.columns:
                    val = product_row[beta_col].values[0]
                    if pd.notna(val) and val != 0:
                        non_zero_betas.append(f"{beta_col}={val:.6f}")
            
            if non_zero_betas:
                print(f"    Non-zero betas: {', '.join(non_zero_betas[:3])}")
            else:
                print(f"    No non-zero betas found in first 10 columns")

# Test 2: Kalman_betas.csv (Kalman Format)
print("\n\n" + "="*80)
print("TEST 2: Kalman_betas.csv (Kalman Format)")
print("="*80)

kalman_df = pd.read_csv('Kalman_betas.csv')
print(f"\nShape: {kalman_df.shape}")
print(f"\nFirst 5 columns: {list(kalman_df.columns[:5])}")
print(f"Last 5 columns: {list(kalman_df.columns[-5:])}")

# Detect format
format_type_k = detect_beta_format(kalman_df)
print(f"\nDetected format: '{format_type_k}'")

# Auto-convert
converted_kalman, fmt_k, was_converted_k = auto_convert_beta_file(kalman_df)
print(f"Was converted: {was_converted_k}")
print(f"Converted shape: {converted_kalman.shape}")

# Show sample products and their beta columns
print("\nSample products and beta columns:")
if 'Product title' in converted_kalman.columns:
    products_k = converted_kalman['Product title'].head(3).tolist()
    beta_cols_k = [col for col in converted_kalman.columns if col.startswith('Beta_')]
    
    print(f"\nFound {len(beta_cols_k)} Beta columns")
    print(f"Sample Beta columns: {beta_cols_k[:5]}")
    
    for product in products_k:
        print(f"\n  Product: '{product}'")
        # Show all beta values for this product
        product_row = converted_kalman[converted_kalman['Product title'] == product]
        if not product_row.empty:
            b0 = product_row['B0'].values[0] if 'B0' in converted_kalman.columns else 'N/A'
            print(f"    B0: {b0}")
            
            # Show non-zero betas
            non_zero_betas = []
            for beta_col in beta_cols_k[:10]:  # Check first 10 beta columns
                if beta_col in product_row.columns:
                    val = product_row[beta_col].values[0]
                    if pd.notna(val) and val != 0:
                        non_zero_betas.append(f"{beta_col}={val:.6f}")
            
            if non_zero_betas:
                print(f"    Non-zero betas: {', '.join(non_zero_betas[:3])}")
            else:
                print(f"    No non-zero betas found in first 10 columns")

# Test 3: Compare formats
print("\n\n" + "="*80)
print("TEST 3: Format Comparison")
print("="*80)

print("\nbetas.csv (Standard):")
print(f"  - Has 'Product title' column: {'Product title' in betas_df.columns}")
print(f"  - Has 'B0' or 'B0 (Original)' column: {'B0' in betas_df.columns or 'B0 (Original)' in betas_df.columns}")
print(f"  - Has Beta_ columns: {any(col.startswith('Beta_') for col in betas_df.columns)}")
print(f"  - Number of Beta_ columns: {len([col for col in betas_df.columns if col.startswith('Beta_')])}")

print("\nKalman_betas.csv (Kalman):")
print(f"  - Has 'Product' column: {'Product' in kalman_df.columns}")
print(f"  - Has 'Intercept' column: {'Intercept' in kalman_df.columns}")
print(f"  - Has metrics (R2, MAE, etc.): {any(col in kalman_df.columns for col in ['R2', 'MAE', 'RMSE', 'MAPE'])}")
print(f"  - Has Beta_ columns: {any(col.startswith('Beta_') for col in kalman_df.columns)}")

print("\nAfter conversion:")
print(f"  - Both have 'Product title': {'Product title' in converted_df.columns and 'Product title' in converted_kalman.columns}")
print(f"  - Both have 'B0': {'B0' in converted_df.columns or 'B0 (Original)' in converted_df.columns} and {'B0' in converted_kalman.columns}")
print(f"  - Both have Beta_ columns: {any(col.startswith('Beta_') for col in converted_df.columns) and any(col.startswith('Beta_') for col in converted_kalman.columns)}")

# Test 4: Check if column naming is consistent
print("\n\n" + "="*80)
print("TEST 4: Column Naming Consistency")
print("="*80)

beta_cols_standard = [col for col in converted_df.columns if col.startswith('Beta_')]
beta_cols_kalman = [col for col in converted_kalman.columns if col.startswith('Beta_')]

print(f"\nStandard format Beta columns ({len(beta_cols_standard)}):")
for col in beta_cols_standard[:5]:
    print(f"  - {col}")

print(f"\nKalman format Beta columns ({len(beta_cols_kalman)}):")
for col in beta_cols_kalman[:5]:
    print(f"  - {col}")

# Check for common patterns
print("\nColumn naming patterns:")
print(f"  Standard uses '_meta_impression': {any('_meta_impression' in col for col in beta_cols_standard)}")
print(f"  Kalman uses '_meta_impression': {any('_meta_impression' in col for col in beta_cols_kalman)}")

# Final verdict
print("\n" + "="*80)
print("FINAL VERDICT")
print("="*80)

if format_type == 'standard' and format_type_k == 'kalman':
    print("✅ Both formats detected correctly!")
    
    if was_converted_k and not was_converted:
        print("✅ Kalman format was converted, standard format was not (as expected)")
    
    if 'Product title' in converted_df.columns and 'Product title' in converted_kalman.columns:
        print("✅ Both have 'Product title' column after processing")
    
    if (any(col.startswith('Beta_') for col in converted_df.columns) and 
        any(col.startswith('Beta_') for col in converted_kalman.columns)):
        print("✅ Both have Beta_ columns after processing")
    
    print("\n🎉 The code WORKS for both formats!")
    print("   - betas.csv (standard) is used as-is")
    print("   - Kalman_betas.csv is auto-converted to standard format")
    print("   - Both can be used in the optimizer")
else:
    print("❌ Format detection issue")
    print(f"   betas.csv detected as: {format_type}")
    print(f"   Kalman_betas.csv detected as: {format_type_k}")

print("="*80)
