import pandas as pd

# Product mapping - reverse PowerGel back to Other Products
product_mapping = {
    'PowerGel': 'Other Products',
    'Max': 'Max',
    'FlexiCool': 'FlexiCool',
    'NitroShot': 'NitroShot',
    'ReLiefX': 'ReLiefX',
    'HydraUp': 'HydraUp',
    'Revive': 'Revive',
    'HeatBoost': 'HeatBoost'
}

print('=== REVERTING PowerGel to Other Products ===\n')

# 1. Data_for_model.xlsx
print('1. Updating Data_for_model.xlsx...')
df = pd.read_excel('presentation_files/Data_for_model.xlsx')
df['Product title'] = df['Product title'].map(product_mapping)
df.to_excel('presentation_files/Data_for_model_fixed.xlsx', index=False)
print(f'   Products: {df["Product title"].unique().tolist()}\n')

# 2. budget_allocation.xlsx
print('2. Updating budget_allocation.xlsx...')
df = pd.read_excel('presentation_files/budget_allocation.xlsx')
# Find the product/item column
item_col = None
for col in df.columns:
    if 'item' in col.lower() or 'product' in col.lower():
        item_col = col
        break
if item_col:
    df[item_col] = df[item_col].map(lambda x: product_mapping.get(x, x))
    print(f'   Column: {item_col}')
    print(f'   Items: {df[item_col].unique().tolist()}\n')
else:
    print('   ⚠️ No item/product column found\n')
df.to_excel('presentation_files/budget_allocation_fixed.xlsx', index=False)

# 3. CPM.csv
print('3. Updating CPM.csv...')
df = pd.read_csv('presentation_files/CPM.csv')
item_col = None
for col in df.columns:
    if 'item' in col.lower() or 'product' in col.lower():
        item_col = col
        break
if item_col:
    df[item_col] = df[item_col].map(lambda x: product_mapping.get(x, x))
    print(f'   Column: {item_col}')
    print(f'   Items: {df[item_col].unique().tolist()}\n')
else:
    print('   ⚠️ No item/product column found\n')
df.to_csv('presentation_files/CPM_fixed.csv', index=False)

# 4. product_prices.csv
print('4. Updating product_prices.csv...')
df = pd.read_csv('presentation_files/product_prices.csv')
item_col = None
for col in df.columns:
    if 'item' in col.lower() or 'product' in col.lower():
        item_col = col
        break
if item_col:
    df[item_col] = df[item_col].map(lambda x: product_mapping.get(x, x))
    print(f'   Column: {item_col}')
    print(f'   Items: {df[item_col].unique().tolist()}\n')
else:
    print('   ⚠️ No item/product column found\n')
df.to_csv('presentation_files/product_prices_fixed.csv', index=False)

# 5. betas.csv (update the original one with Product title column)
print('5. Updating betas.csv...')
df = pd.read_csv('presentation_files/betas.csv')
# Rename Product to Product title if needed
if 'Product' in df.columns and 'Product title' not in df.columns:
    df = df.rename(columns={'Product': 'Product title'})
df['Product title'] = df['Product title'].map(product_mapping)
df.to_csv('presentation_files/betas_final.csv', index=False)
print(f'   Products: {df["Product title"].unique().tolist()}\n')

print('✅ ALL FILES UPDATED - PowerGel is now Other Products')
