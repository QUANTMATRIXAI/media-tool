"""
Test that traffic ads budget maps to the correct beta column in Kalman format
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path("optimiser").absolute()))

from utils.beta_converter import auto_convert_beta_file
from utils.data_utils import get_channel_beta_mapping_with_fallback

print("="*80)
print("TRAFFIC ADS MAPPING TEST")
print("="*80)

# Load and convert Kalman betas
kalman_df = pd.read_csv('Kalman_betas.csv')
print("\n1. Original Kalman_betas.csv columns:")
print([col for col in kalman_df.columns if 'daily' in col.lower() or 'link' in col.lower()])

# Convert to standard format
converted_df, fmt, was_converted = auto_convert_beta_file(kalman_df)
print(f"\n2. After conversion (format: {fmt}, converted: {was_converted}):")
print([col for col in converted_df.columns if 'daily' in col.lower() or 'link' in col.lower()])

# Get smart mapping
mapping = get_channel_beta_mapping_with_fallback(converted_df)

print("\n3. Channel Mapping:")
traffic_channel = "traffic(all sku's)"
print(f"   '{traffic_channel}' → {mapping[traffic_channel]}")

# Check if column exists
traffic_beta_col = mapping["traffic(all sku's)"]
if traffic_beta_col and traffic_beta_col in converted_df.columns:
    print(f"\n✅ SUCCESS: Column '{traffic_beta_col}' exists in converted Kalman betas!")
    print(f"   Budget for 'traffic(all sku's)' will be allocated to this beta column.")
else:
    print(f"\n❌ ERROR: Column '{traffic_beta_col}' NOT found in converted Kalman betas!")

print("\n4. Sample beta values from this column:")
if traffic_beta_col and traffic_beta_col in converted_df.columns:
    sample_values = converted_df[traffic_beta_col].head(5)
    for idx, val in enumerate(sample_values):
        product = converted_df['Product title'].iloc[idx] if 'Product title' in converted_df.columns else f"Product {idx}"
        print(f"   {product}: {val}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("When using Kalman_betas.csv:")
print(f"  - Traffic ads budget → {traffic_beta_col}")
print(f"  - This beta column exists: {traffic_beta_col in converted_df.columns if traffic_beta_col else False}")
print("="*80)
