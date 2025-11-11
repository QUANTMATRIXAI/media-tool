"""
Create budget allocation file for presentation with renamed products
"""

import pandas as pd
import numpy as np

# Load original budget file
df_original = pd.read_excel('optimiser/budget_allocation.xlsx')

print("Original budget file:")
print(df_original)

# Product mapping
product_mapping = {
    'Other Products': 'PowerGel',
    'Minimal Diamond Climber Earrings': 'Max',
    'Pave Diamond Paperclip Bracelet': 'FlexiCool',
    'Pave Diamond Paperclip Ring': 'NitroShot',
    'Paw Diamond Necklace': 'ReLiefX',
    'Textured Diamond Band': 'HydraUp',
    'Twirl Diamond Hoops': 'Revive',
    'Two-Tone Interlocking Diamond Band': 'HeatBoost'
}

# Create new budget data with only the 8 products + traffic + google
channels = [
    'PowerGel',
    'Max',
    'FlexiCool',
    'NitroShot',
    'ReLiefX',
    'HydraUp',
    'Revive',
    'HeatBoost',
    "Traffic(All Sku's)",
    'Google Campaigns'
]

# Create budget data (using sample values)
budget_data = {
    'Product Name': channels,
    '6th-12th': [15000, 8000, 7500, 6000, 9000, 5500, 6500, 7000, 12000, 10000],
    '13th-19th': [16000, 8500, 8000, 6500, 9500, 6000, 7000, 7500, 13000, 11000],
    '20th-26th': [15500, 8200, 7800, 6200, 9200, 5800, 6800, 7200, 12500, 10500]
}

# Try to get actual values from original file if they exist
for i, channel in enumerate(channels):
    # Find original name
    original_name = None
    for orig, new in product_mapping.items():
        if new == channel:
            original_name = orig
            break
    
    if original_name is None:
        original_name = channel  # For traffic and google
    
    # Find in original data
    matching_row = df_original[df_original['Product Name'].str.lower() == original_name.lower()]
    
    if len(matching_row) > 0:
        # Use actual values from original file
        for week_col in ['6th-12th', '13th-19th', '20th-26th']:
            if week_col in df_original.columns:
                budget_data[week_col][i] = matching_row[week_col].values[0]

# Create DataFrame
df_budget = pd.DataFrame(budget_data)

print("\n\nNew budget file:")
print(df_budget)

# Save to presentation folder
output_path = 'presentation_files/budget_allocation.xlsx'
df_budget.to_excel(output_path, index=False)
print(f"\n✅ Saved to: {output_path}")
