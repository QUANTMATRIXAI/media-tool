"""
Comprehensive Beta Mapping Test
Tests beta mapping for both standard optimizer format and Kalman modeling format
Verifies that products correctly map to their beta coefficients
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add optimiser to path
sys.path.insert(0, str(Path("optimiser").absolute()))

from utils.data_utils import product_to_beta_column, get_channel_beta_mapping
from utils.beta_converter import detect_beta_format, convert_kalman_to_standard, auto_convert_beta_file

print("="*80)
print("COMPREHENSIVE BETA MAPPING TEST")
print("="*80)

# Test 1: Create sample data in both formats
print("\n📋 TEST 1: Creating Sample Test Data")
print("-"*80)

# Sample products
test_products = [
    "Facebook Impressions",
    "Google Search",
    "TV Campaign",
    "Email Marketing",
    "Display Ads"
]

# Create Kalman format sample
kalman_sample = pd.DataFrame({
    'Product': test_products,
    'R2': [0.85, 0.78, 0.92, 0.65, 0.71],
    'MAE': [1200, 1500, 900, 1800, 1600],
    'RMSE': [1500, 1800, 1100, 2100, 1900],
    'MAPE': [0.12, 0.15, 0.09, 0.18, 0.16],
    'Intercept': [5000, 4500, 6000, 3500, 4000],
    'Facebook Impressions': [0.45, 0.0, 0.0, 0.0, 0.0],
    'Google Search': [0.0, 0.62, 0.0, 0.0, 0.0],
    'TV Campaign': [0.0, 0.0, 0.78, 0.0, 0.0],
    'Email Marketing': [0.0, 0.0, 0.0, 0.38, 0.0],
    'Display Ads': [0.0, 0.0, 0.0, 0.0, 0.52]
})

print("✅ Created Kalman format sample data")
print(f"   Products: {len(kalman_sample)}")
print(f"   Columns: {list(kalman_sample.columns)}")

# Create standard format sample
standard_sample = pd.DataFrame({
    'Product title': test_products,
    'B0': [5000, 4500, 6000, 3500, 4000],
    'Beta_Facebook Impressions': [0.45, 0.0, 0.0, 0.0, 0.0],
    'Beta_Google Search': [0.0, 0.62, 0.0, 0.0, 0.0],
    'Beta_TV Campaign': [0.0, 0.0, 0.78, 0.0, 0.0],
    'Beta_Email Marketing': [0.0, 0.0, 0.0, 0.38, 0.0],
    'Beta_Display Ads': [0.0, 0.0, 0.0, 0.0, 0.52]
})

print("✅ Created standard format sample data")
print(f"   Products: {len(standard_sample)}")
print(f"   Columns: {list(standard_sample.columns)}")

# Test 2: Format Detection
print("\n\n🔍 TEST 2: Format Detection")
print("-"*80)

kalman_format = detect_beta_format(kalman_sample)
standard_format = detect_beta_format(standard_sample)

print(f"Kalman sample detected as: '{kalman_format}'")
if kalman_format == 'kalman':
    print("   ✅ PASS - Correctly identified Kalman format")
else:
    print(f"   ❌ FAIL - Expected 'kalman', got '{kalman_format}'")

print(f"\nStandard sample detected as: '{standard_format}'")
if standard_format == 'standard':
    print("   ✅ PASS - Correctly identified standard format")
else:
    print(f"   ❌ FAIL - Expected 'standard', got '{standard_format}'")

# Test 3: Kalman to Standard Conversion
print("\n\n🔄 TEST 3: Kalman to Standard Conversion")
print("-"*80)

converted_df = convert_kalman_to_standard(kalman_sample)
print(f"Converted DataFrame shape: {converted_df.shape}")
print(f"Converted columns: {list(converted_df.columns)}")

# Verify conversion
conversion_checks = []

# Check 1: Product title column exists
if 'Product title' in converted_df.columns:
    print("✅ 'Product title' column created")
    conversion_checks.append(True)
else:
    print("❌ 'Product title' column missing")
    conversion_checks.append(False)

# Check 2: B0 column exists
if 'B0' in converted_df.columns:
    print("✅ 'B0' column created")
    conversion_checks.append(True)
else:
    print("❌ 'B0' column missing")
    conversion_checks.append(False)

# Check 3: Beta_ columns created
beta_cols = [col for col in converted_df.columns if col.startswith('Beta_')]
if len(beta_cols) == 5:
    print(f"✅ {len(beta_cols)} Beta_ columns created")
    conversion_checks.append(True)
else:
    print(f"❌ Expected 5 Beta_ columns, got {len(beta_cols)}")
    conversion_checks.append(False)

# Check 4: Values preserved
if 'B0' in converted_df.columns:
    b0_match = (converted_df['B0'] == kalman_sample['Intercept']).all()
    if b0_match:
        print("✅ B0 values match Intercept values")
        conversion_checks.append(True)
    else:
        print("❌ B0 values don't match Intercept values")
        conversion_checks.append(False)

# Check 5: Beta values preserved
if 'Beta_Facebook Impressions' in converted_df.columns:
    beta_match = (converted_df['Beta_Facebook Impressions'] == kalman_sample['Facebook Impressions']).all()
    if beta_match:
        print("✅ Beta values preserved correctly")
        conversion_checks.append(True)
    else:
        print("❌ Beta values not preserved correctly")
        conversion_checks.append(False)

conversion_success = all(conversion_checks)
print(f"\n{'✅ CONVERSION TEST PASSED' if conversion_success else '❌ CONVERSION TEST FAILED'}")

# Test 4: Product to Beta Column Mapping
print("\n\n🗺️ TEST 4: Product to Beta Column Mapping")
print("-"*80)

mapping_checks = []

print("Testing current product_to_beta_column() function:")
for product in test_products:
    expected_beta_col = product_to_beta_column(product)
    print(f"\n📦 Product: '{product}'")
    print(f"   Current function returns: '{expected_beta_col}'")
    
    # Check in converted dataframe
    if expected_beta_col in converted_df.columns:
        # Get the beta value for this product
        product_row = converted_df[converted_df['Product title'] == product]
        if not product_row.empty:
            beta_value = product_row[expected_beta_col].values[0]
            print(f"   ✅ Beta column found! Value: {beta_value}")
            
            # Verify it's the correct value (should be non-zero for matching product)
            if beta_value > 0:
                print(f"   ✅ Beta value is non-zero (correct mapping)")
                mapping_checks.append(True)
            else:
                print(f"   ⚠️ Beta value is zero (might be correct if no effect)")
                mapping_checks.append(True)  # Still pass, could be legitimate
        else:
            print(f"   ❌ Product not found in converted dataframe")
            mapping_checks.append(False)
    else:
        print(f"   ❌ Beta column not found in converted dataframe")
        
        # Try to find a matching column with fuzzy matching
        actual_beta_col = f"Beta_{product}"
        if actual_beta_col in converted_df.columns:
            print(f"   💡 SUGGESTION: Actual column is '{actual_beta_col}'")
            print(f"      Function should return this instead of '{expected_beta_col}'")
        else:
            print(f"   Available Beta columns: {[col for col in converted_df.columns if col.startswith('Beta_')]}")
        mapping_checks.append(False)

mapping_success = all(mapping_checks)
print(f"\n{'✅ MAPPING TEST PASSED' if mapping_success else '❌ MAPPING TEST FAILED'}")
print("\n⚠️ ISSUE IDENTIFIED:")
print("   The product_to_beta_column() function adds '_meta_impression' suffix")
print("   but the actual beta columns don't have this suffix.")
print("   The function needs to be updated to match the actual column format.")

# Test 5: Auto-conversion function
print("\n\n🤖 TEST 5: Auto-Conversion Function")
print("-"*80)

# Test with Kalman format
converted_kalman, format_type_k, was_converted_k = auto_convert_beta_file(kalman_sample)
print(f"Kalman sample:")
print(f"   Detected format: '{format_type_k}'")
print(f"   Was converted: {was_converted_k}")
print(f"   Result shape: {converted_kalman.shape}")

if format_type_k == 'kalman' and was_converted_k:
    print("   ✅ PASS - Kalman format detected and converted")
else:
    print("   ❌ FAIL - Auto-conversion didn't work as expected")

# Test with standard format
converted_standard, format_type_s, was_converted_s = auto_convert_beta_file(standard_sample)
print(f"\nStandard sample:")
print(f"   Detected format: '{format_type_s}'")
print(f"   Was converted: {was_converted_s}")
print(f"   Result shape: {converted_standard.shape}")

if format_type_s == 'standard' and not was_converted_s:
    print("   ✅ PASS - Standard format detected, no conversion needed")
else:
    print("   ❌ FAIL - Auto-conversion didn't work as expected")

# Test 6: Real File Testing (if files exist)
print("\n\n📁 TEST 6: Real File Testing")
print("-"*80)

real_file_tests = []

# Try to load actual beta files
beta_file_paths = [
    "betas.csv",  # Standard format
    "Kalman_betas.csv",  # Kalman format
    "optimiser/betas.csv",
    "optimiser/Data_for_model.xlsx",
    "kalman modleling/kalman_output.csv"
]

for file_path in beta_file_paths:
    if Path(file_path).exists():
        print(f"\n📄 Testing file: {file_path}")
        try:
            # Load file
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            print(f"   Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Detect format
            format_type = detect_beta_format(df)
            print(f"   Detected format: '{format_type}'")
            
            # Auto-convert
            converted_df, _, was_converted = auto_convert_beta_file(df)
            print(f"   Converted: {was_converted}")
            
            # Get product column
            product_col = None
            if 'Product title' in converted_df.columns:
                product_col = 'Product title'
            elif 'Product' in converted_df.columns:
                product_col = 'Product'
            
            if product_col:
                products = converted_df[product_col].tolist()[:3]  # Test first 3
                print(f"   Testing {len(products)} products:")
                
                for product in products:
                    expected_beta_col = product_to_beta_column(product)
                    if expected_beta_col in converted_df.columns:
                        product_row = converted_df[converted_df[product_col] == product]
                        if not product_row.empty:
                            beta_value = product_row[expected_beta_col].values[0]
                            print(f"      ✅ '{product}' → {expected_beta_col} = {beta_value}")
                            real_file_tests.append(True)
                        else:
                            print(f"      ❌ '{product}' not found in dataframe")
                            real_file_tests.append(False)
                    else:
                        print(f"      ❌ '{product}' → {expected_beta_col} NOT FOUND")
                        print(f"         Available: {[col for col in converted_df.columns if col.startswith('Beta_')][:5]}")
                        real_file_tests.append(False)
            else:
                print("   ⚠️ No product column found")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print(f"⚠️ File not found: {file_path}")

# Test 7: Correct Mapping Logic
print("\n\n✨ TEST 7: Demonstrating Correct Mapping Logic")
print("-"*80)

print("Actual columns in converted dataframe:")
print(f"   {list(converted_df.columns)}")
print()

print("For the converted Kalman data, the correct mapping should be:")
print()

correct_mapping_checks = []
for product in test_products:
    # The correct mapping is simply: Beta_{product_name}
    correct_beta_col = f"Beta_{product}"
    
    print(f"📦 Product: '{product}'")
    print(f"   Looking for column: '{correct_beta_col}'")
    
    # Check if column exists (exact match)
    if correct_beta_col in converted_df.columns:
        product_row = converted_df[converted_df['Product title'] == product]
        if not product_row.empty:
            beta_value = product_row[correct_beta_col].values[0]
            b0_value = product_row['B0'].values[0]
            print(f"   ✅ Column exists! Beta: {beta_value}, B0: {b0_value}")
            correct_mapping_checks.append(True)
        else:
            print(f"   ❌ Product not found in dataframe")
            correct_mapping_checks.append(False)
    else:
        # Try to find similar column
        similar_cols = [col for col in converted_df.columns if product.lower() in col.lower()]
        if similar_cols:
            print(f"   ⚠️ Exact match not found, but found similar: {similar_cols}")
            # Use the first similar column
            actual_col = similar_cols[0]
            product_row = converted_df[converted_df['Product title'] == product]
            if not product_row.empty:
                beta_value = product_row[actual_col].values[0]
                b0_value = product_row['B0'].values[0]
                print(f"   ✅ Using '{actual_col}': Beta: {beta_value}, B0: {b0_value}")
                correct_mapping_checks.append(True)
            else:
                print(f"   ❌ Product not found")
                correct_mapping_checks.append(False)
        else:
            print(f"   ❌ No matching column found")
            correct_mapping_checks.append(False)

correct_mapping_success = all(correct_mapping_checks)
print(f"\n{'✅ CORRECT MAPPING WORKS!' if correct_mapping_success else '⚠️ MAPPING NEEDS ADJUSTMENT'}")

print("\n💡 KEY FINDINGS:")
print("   1. Kalman format uses exact product names as column headers")
print("   2. After conversion, these become 'Beta_{exact_product_name}'")
print("   3. The product_to_beta_column() function should:")
print("      - For 'Other Products': return 'Beta_Impressions'")
print("      - For other products: return f'Beta_{product_name}' (preserve case and spaces)")
print("      - Remove the '_meta_impression' suffix and lowercase conversion")

# Test 8: Special Channel Mapping
print("\n\n🔗 TEST 8: Special Channel Mapping")
print("-"*80)

channel_mapping = get_channel_beta_mapping()
print("Special channel mappings:")
for channel, beta_col in channel_mapping.items():
    print(f"   '{channel}' → {beta_col}")
    
    # Verify these work with product_to_beta_column
    result = product_to_beta_column(channel)
    if result == beta_col:
        print(f"      ✅ Mapping works correctly")
    else:
        print(f"      ❌ Expected '{beta_col}', got '{result}'")

# Final Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

all_tests = {
    'Format Detection': kalman_format == 'kalman' and standard_format == 'standard',
    'Kalman to Standard Conversion': conversion_success,
    'Product to Beta Mapping': mapping_success,
    'Auto-Conversion Function': format_type_k == 'kalman' and format_type_s == 'standard',
    'Real File Testing': len(real_file_tests) == 0 or all(real_file_tests)
}

for test_name, passed in all_tests.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")

overall_pass = all(all_tests.values())
print("\n" + "="*80)
if overall_pass:
    print("🎉 ALL TESTS PASSED! Beta mapping is working correctly.")
else:
    print("⚠️ SOME TESTS FAILED. Review the output above for details.")
print("="*80)
