"""
Test channel mapping with both standard and Kalman beta formats
"""

import pandas as pd
import sys
from pathlib import Path

# Add optimiser to path
sys.path.insert(0, str(Path("optimiser").absolute()))

from utils.data_utils import get_channel_beta_mapping, get_channel_beta_mapping_with_fallback

print("="*80)
print("CHANNEL MAPPING TEST")
print("="*80)

# Load both beta files
betas_standard = pd.read_csv('betas.csv')
betas_kalman = pd.read_csv('Kalman_betas.csv')

print("\n1. Standard Beta File (betas.csv)")
print("-"*80)
print(f"Columns with 'Google': {[col for col in betas_standard.columns if 'google' in col.lower()]}")
print(f"Columns with 'Daily': {[col for col in betas_standard.columns if 'daily' in col.lower()]}")

print("\n2. Kalman Beta File (Kalman_betas.csv)")
print("-"*80)
print(f"Columns with 'Google': {[col for col in betas_kalman.columns if 'google' in col.lower()]}")
print(f"Columns with 'Daily': {[col for col in betas_kalman.columns if 'daily' in col.lower()]}")

print("\n3. Basic Channel Mapping (Old)")
print("-"*80)
basic_mapping = get_channel_beta_mapping()
for channel, beta_col in basic_mapping.items():
    print(f"  '{channel}' → {beta_col}")

print("\n4. Smart Channel Mapping with Fallback (New)")
print("-"*80)

print("\nFor Standard Beta File:")
smart_mapping_standard = get_channel_beta_mapping_with_fallback(betas_standard)
for channel, beta_col in smart_mapping_standard.items():
    exists = "✅" if beta_col is None or beta_col in betas_standard.columns else "❌"
    print(f"  {exists} '{channel}' → {beta_col}")

print("\nFor Kalman Beta File:")
smart_mapping_kalman = get_channel_beta_mapping_with_fallback(betas_kalman)
for channel, beta_col in smart_mapping_kalman.items():
    exists = "✅" if beta_col is None or beta_col in betas_kalman.columns else "❌"
    print(f"  {exists} '{channel}' → {beta_col}")

print("\n5. Verification")
print("-"*80)

# Check if mappings work for both files
all_good = True

for channel, beta_col in smart_mapping_standard.items():
    if beta_col is not None and beta_col not in betas_standard.columns:
        print(f"❌ Standard: '{channel}' maps to '{beta_col}' but column doesn't exist!")
        all_good = False

for channel, beta_col in smart_mapping_kalman.items():
    if beta_col is not None and beta_col not in betas_kalman.columns:
        print(f"❌ Kalman: '{channel}' maps to '{beta_col}' but column doesn't exist!")
        all_good = False

if all_good:
    print("✅ All channel mappings are valid for both file formats!")
    print("\n🎉 SUCCESS: The smart mapping works for both standard and Kalman formats!")
else:
    print("\n⚠️ Some mappings are invalid. Please review the output above.")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("The smart mapping function automatically detects which column format")
print("is available in the beta file and uses the correct one:")
print("  - Standard format: Beta_Daily_Impressions_OUTCOME_ENGAGEMENT")
print("  - Kalman format: Beta_Daily_Impressions_LINK_CLICKS")
print("="*80)
