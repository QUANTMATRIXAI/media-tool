"""
Filter and rename data for presentation
"""

import pandas as pd

# Load the data
df = pd.read_excel('optimiser/Data_for_model.xlsx')

print(f"Original data shape: {df.shape}")
print(f"Original columns: {list(df.columns)}")

# Product mapping - 8 products
product_mapping = {
    'Other Products': 'PowerGel',
    'minimal diamond climber earrings': 'Max',
    'pave diamond paperclip bracelet': 'FlexiCool',
    'pave diamond paperclip ring': 'NitroShot',
    'paw diamond necklace': 'ReLiefX',
    'textured diamond band': 'HydraUp',
    'twirl diamond hoops': 'Revive',
    'two-tone interlocking diamond band': 'HeatBoost'
}

# Filter rows - keep only specified products
if 'Product title' in df.columns:
    df_filtered = df[df['Product title'].isin(product_mapping.keys())].copy()
    
    # Rename products
    df_filtered['Product title'] = df_filtered['Product title'].map(product_mapping)
    
    print(f"\nFiltered to {len(df_filtered)} rows")
    print(f"Products kept: {df_filtered['Product title'].unique().tolist()}")
else:
    print("ERROR: 'Product title' column not found!")
    df_filtered = df.copy()

# Filter columns - keep only relevant impression columns
columns_to_keep = []

# Always keep these base columns
base_columns = ['Date', 'Product title', 'Amount spent (USD)', 'Gross profit', 'Gross sales', 
                'Net sales', 'Discounts', 'Net items sold', 'Gross margin', 'Week Number', 
                'Week End', 'Product variant price']

for col in base_columns:
    if col in df_filtered.columns:
        columns_to_keep.append(col)

# Rename meta impression columns for the specified products
rename_dict = {}
for old_name, new_name in product_mapping.items():
    # Look for meta impression column
    meta_col = f"{old_name}_meta_impression"
    if meta_col in df_filtered.columns:
        new_col_name = f"{new_name}_meta_impression"
        rename_dict[meta_col] = new_col_name
        columns_to_keep.append(new_col_name)

# Apply all renames at once
df_filtered = df_filtered.rename(columns=rename_dict)

# Rename "Impressions" column to "PowerGel_meta_impression" (since PowerGel is Other Products)
if 'Impressions' in df_filtered.columns:
    df_filtered = df_filtered.rename(columns={'Impressions': 'PowerGel_meta_impression'})
    columns_to_keep.append('PowerGel_meta_impression')

# Keep other general impression columns
impression_cols = ['Google_Impression', 'Daily_Impressions_OUTCOME_ENGAGEMENT', 
                   'Daily_Impressions_LINK_CLICKS']
for col in impression_cols:
    if col in df_filtered.columns:
        columns_to_keep.append(col)

# Keep other important columns
other_cols = ['google_trends', 'Category Discount']
for col in other_cols:
    if col in df_filtered.columns:
        columns_to_keep.append(col)

# Filter to only keep specified columns
df_final = df_filtered[columns_to_keep].copy()

print(f"\nFinal data shape: {df_final.shape}")
print(f"Final columns: {list(df_final.columns)}")
print(f"\nProducts in final data:")
print(df_final['Product title'].value_counts())

# Save to presentation folder
output_path = 'presentation_files/Data_for_model.xlsx'
df_final.to_excel(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")

# Show sample
print("\nSample data:")
print(df_final.head())
