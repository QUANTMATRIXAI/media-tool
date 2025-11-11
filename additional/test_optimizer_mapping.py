"""
Test script to verify optimizer data mapping
Checks if products are correctly mapped to their betas, prices, and budgets
"""

import pandas as pd
import sys
from pathlib import Path

# Add optimiser to path
sys.path.insert(0, str(Path("optimiser").absolute()))

from utils.data_utils import product_to_beta_column, get_channel_beta_mapping

print("="*80)
print("OPTIMIZER MAPPING VERIFICATION TEST")
print("="*80)

# Test 1: Load sample files
print("\n📁 TEST 1: Loading Sample Files")
print("-"*80)

try:
    # Try multiple beta file locations
    beta_files = ["optimiser/betas.csv", "betas.csv", "optimiser/Data_for_model.xlsx"]
    beta_df = None
    
    for beta_file in beta_files:
        if Path(beta_file).exists():
            try:
                if beta_file.endswith('.csv'):
                    beta_df = pd.read_csv(beta_file)
                else:
                    beta_df = pd.read_excel(beta_file)
                print(f"✅ Beta file loaded from: {beta_file}")
                print(f"   Products: {len(beta_df)}")
                print(f"   Columns: {list(beta_df.columns)[:10]}")
                break
            except Exception as e:
                print(f"⚠️ Could not load {beta_file}: {e}")
    
    if beta_df is None:
        print(f"❌ No beta file found. Tried: {beta_files}")
    
    # Load price file
    price_file = "optimiser/product_prices.csv"
    if Path(price_file).exists():
        price_df = pd.read_csv(price_file)
        print(f"✅ Price file loaded: {len(price_df)} products")
        print(f"   Columns: {list(price_df.columns)}")
    else:
        print(f"❌ Price file not found at {price_file}")
        price_df = None
    
    # Load CPM file
    cpm_file = "optimiser/CPM.xlsx"
    if Path(cpm_file).exists():
        cpm_df = pd.read_excel(cpm_file)
        print(f"✅ CPM file loaded: {len(cpm_df)} items")
        print(f"   Columns: {list(cpm_df.columns)}")
    else:
        print(f"❌ CPM file not found at {cpm_file}")
        cpm_df = None

except Exception as e:
    print(f"❌ Error loading files: {e}")
    sys.exit(1)

# Test 2: Check product mapping
print("\n🔍 TEST 2: Product to Beta Column Mapping")
print("-"*80)

if beta_df is not None:
    # Get product title column
    product_col = next((col for col in beta_df.columns if 'product' in col.lower() and 'title' in col.lower()), None)
    
    if product_col:
        products = beta_df[product_col].tolist()
        print(f"Found {len(products)} products in beta file\n")
        
        # Test first 3 products
        for i, product in enumerate(products[:3]):
            print(f"\n📦 Product {i+1}: '{product}'")
            print(f"   └─ Expected beta column: {product_to_beta_column(product)}")
            
            # Check if beta column exists
            expected_beta_col = product_to_beta_column(product)
            if expected_beta_col in beta_df.columns:
                beta_value = beta_df.loc[beta_df[product_col] == product, expected_beta_col].values[0]
                print(f"   └─ ✅ Beta column found! Value: {beta_value}")
            else:
                print(f"   └─ ❌ Beta column NOT found in file")
                print(f"   └─ Available Beta columns: {[col for col in beta_df.columns if col.startswith('Beta_')][:5]}")
            
            # Check B0 (intercept)
            b0_col = next((col for col in beta_df.columns if col.startswith('B0')), None)
            if b0_col:
                b0_value = beta_df.loc[beta_df[product_col] == product, b0_col].values[0]
                print(f"   └─ B0 (Intercept): {b0_value}")

# Test 3: Check price mapping
print("\n\n💰 TEST 3: Product to Price Mapping")
print("-"*80)

if price_df is not None and beta_df is not None:
    # Get product columns
    price_product_col = next((col for col in price_df.columns if 'product' in col.lower()), None)
    beta_product_col = next((col for col in beta_df.columns if 'product' in col.lower() and 'title' in col.lower()), None)
    price_col = next((col for col in price_df.columns if 'price' in col.lower()), None)
    
    if price_product_col and beta_product_col and price_col:
        # Check first 3 products from beta file
        for i, product in enumerate(beta_df[beta_product_col].tolist()[:3]):
            print(f"\n📦 Product: '{product}'")
            
            # Find price
            price_match = price_df[price_df[price_product_col].str.lower() == product.lower()]
            if not price_match.empty:
                price_value = price_match[price_col].values[0]
                print(f"   └─ ✅ Price found: ${price_value}")
            else:
                print(f"   └─ ❌ Price NOT found")
                print(f"   └─ Available products in price file: {price_df[price_product_col].tolist()[:5]}")

# Test 4: Check CPM mapping
print("\n\n📊 TEST 4: Product to CPM Mapping")
print("-"*80)

if cpm_df is not None and beta_df is not None:
    # Get item name column from CPM
    cpm_item_col = next((col for col in cpm_df.columns if 'item' in col.lower() and 'name' in col.lower()), None)
    cpm_col = next((col for col in cpm_df.columns if 'cpm' in col.lower()), None)
    beta_product_col = next((col for col in beta_df.columns if 'product' in col.lower() and 'title' in col.lower()), None)
    
    if cpm_item_col and cpm_col and beta_product_col:
        # Check first 3 products
        for i, product in enumerate(beta_df[beta_product_col].tolist()[:3]):
            print(f"\n📦 Product: '{product}'")
            
            # Find CPM
            cpm_match = cpm_df[cpm_df[cpm_item_col].str.lower() == product.lower()]
            if not cpm_match.empty:
                cpm_value = cpm_match[cpm_col].values[0]
                print(f"   └─ ✅ CPM found: ${cpm_value}")
            else:
                print(f"   └─ ❌ CPM NOT found")

# Test 5: Special channel mapping
print("\n\n🔗 TEST 5: Special Channel Mapping")
print("-"*80)

channel_mapping = get_channel_beta_mapping()
print("Special channel mappings:")
for channel, beta_col in channel_mapping.items():
    print(f"   '{channel}' → {beta_col}")

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print("✅ Mapping verification complete!")
print("📝 Review the output above to ensure:")
print("   1. Products map to correct Beta_ columns")
print("   2. Prices are found for each product")
print("   3. CPMs are found for each product")
print("   4. B0 (intercept) values are present")
print("="*80)
