# Channel Mapping Update

## Problem
The traffic ads channel had different column names in different beta file formats:
- **Standard format (betas.csv)**: `Beta_Daily_Impressions_OUTCOME_ENGAGEMENT`
- **Kalman format (Kalman_betas.csv)**: `Daily_Impressions_LINK_CLICKS` → becomes `Beta_Daily_Impressions_LINK_CLICKS` after conversion

The budget allocation for "traffic(all sku's)" was looking for the wrong column name.

## Solution
Updated the channel mapping to support both formats with automatic fallback:

### 1. Updated Primary Mapping
Changed the default mapping to use the Kalman format column name:
```python
"traffic(all sku's)": 'Beta_Daily_Impressions_LINK_CLICKS'
```

### 2. Added Smart Mapping with Fallback
Created `get_channel_beta_mapping_with_fallback()` function that:
- Tries multiple column name variations for each channel
- Automatically detects which column exists in the beta file
- Falls back to alternative names if primary name not found

```python
"traffic(all sku's)": [
    'Beta_Daily_Impressions_LINK_CLICKS',  # Kalman format (primary)
    'Beta_Daily_Impressions_OUTCOME_ENGAGEMENT'  # Standard format (fallback)
]
```

### 3. Updated Optimizer
Modified all places in the optimizer that use channel mapping to use the smart mapping function.

## Channel Mappings

### Google Campaigns
- **Budget item**: "google campaigns"
- **Beta column**: `Beta_Google_Impression`
- **Status**: ✅ Works in both formats (same column name)

### Traffic Ads
- **Budget item**: "traffic(all sku's)"
- **Beta column (Kalman)**: `Beta_Daily_Impressions_LINK_CLICKS`
- **Beta column (Standard)**: `Beta_Daily_Impressions_OUTCOME_ENGAGEMENT`
- **Status**: ✅ Now works in both formats with automatic fallback

### Catalog Campaign
- **Budget item**: "catalog campaign(all sku's)"
- **Beta column**: `None` (not used)
- **Status**: ✅ Correctly mapped to None

## Testing
Run `test_channel_mapping.py` to verify the mappings work for both file formats.

## Result
✅ The optimizer now correctly allocates budget to traffic ads regardless of which beta file format is used (standard or Kalman).
