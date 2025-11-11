# Beta Mapping Test Results

## Test Summary

✅ **PASSED**: Format Detection  
✅ **PASSED**: Kalman to Standard Conversion  
✅ **PASSED**: Auto-Conversion Function  
❌ **FAILED**: Product to Beta Column Mapping  
❌ **FAILED**: Real File Testing  

---

## Key Findings

### 1. Format Detection Works Perfectly ✅
- Correctly identifies Kalman format (with `Product`, `Intercept`, metrics columns)
- Correctly identifies standard format (with `Product title`, `B0`, `Beta_*` columns)
- Auto-detection and conversion work as expected

### 2. Kalman to Standard Conversion Works Perfectly ✅
- `Product` column → `Product title` ✅
- `Intercept` column → `B0` ✅
- Variable columns → `Beta_{variable_name}` ✅
- All values preserved correctly ✅

### 3. Product to Beta Column Mapping Has Issues ❌

**Current Behavior:**
```python
product_to_beta_column("Facebook Impressions")
# Returns: "Beta_facebook impressions_meta_impression"
```

**Actual Column Names in Files:**
- For products: `{product_name}_meta_impression` (no Beta_ prefix, lowercase)
- For special channels: `Google_Impression`, `Daily_Impressions_OUTCOME_ENGAGEMENT`, `Impressions`
- After Kalman conversion: `Beta_{exact_product_name}` (preserves case and spaces)

**The Problem:**
1. Function adds `_meta_impression` suffix - but not all columns have this
2. Function converts to lowercase - but actual columns preserve case
3. Function doesn't handle special channel mappings correctly

---

## Real File Analysis

### File: `betas.csv` (Standard Format)
- **Status**: ✅ Detected correctly as standard format
- **Products**: 1365 rows, 76 columns
- **Mapping**: Works for "Other Products" → `Beta_Impressions`

### File: `Data_for_model.xlsx` (Unknown Format)
- **Status**: ❌ Detected as unknown format
- **Products**: 1167 rows, 38 columns
- **Columns Found**:
  - Product columns: `3 line diamond hoops_meta_impression`, `adjustable bar diamond lariat necklace_meta_impression`, etc.
  - Channel columns: `Google_Impression`, `Daily_Impressions_OUTCOME_ENGAGEMENT`, `Impressions`
  - Other columns: `Date`, `Product title`, `Amount spent (USD)`, `Gross profit`, etc.
- **Issue**: This appears to be a data file, not a beta coefficients file

---

## Recommendations

### Fix 1: Update `product_to_beta_column()` Function

The function needs to handle multiple scenarios:

```python
def product_to_beta_column(product_name, beta_df=None):
    """
    Convert a product name to its corresponding beta column format.
    
    Args:
        product_name: Name of the product/channel
        beta_df: Optional beta DataFrame to check actual column names
    
    Returns:
        Beta column name that matches the actual file format
    """
    normalized = product_name.lower()
    
    # Special case: "Other Products" uses "Beta_Impressions"
    if normalized == 'other products':
        return 'Beta_Impressions'
    
    # If beta_df provided, try to find exact match
    if beta_df is not None:
        # Try exact match with Beta_ prefix
        exact_match = f"Beta_{product_name}"
        if exact_match in beta_df.columns:
            return exact_match
        
        # Try with _meta_impression suffix (no Beta_ prefix)
        meta_match = f"{product_name}_meta_impression"
        if meta_match in beta_df.columns:
            return meta_match
        
        # Try lowercase with _meta_impression
        lower_meta = f"{normalized}_meta_impression"
        if lower_meta in beta_df.columns:
            return lower_meta
    
    # Default: try Beta_ prefix with exact name
    return f"Beta_{product_name}"
```

### Fix 2: Improve Format Detection for Data Files

The `Data_for_model.xlsx` file is not a beta coefficients file - it's a data file with:
- Time series data (Date, Week Number, Week End)
- Sales metrics (Gross profit, Net sales, etc.)
- Impression variables (product_meta_impression columns)

The format detector should distinguish between:
1. **Beta coefficient files** (coefficients for modeling)
2. **Data files** (actual time series data for modeling)

### Fix 3: Add Flexible Column Matching

Instead of hardcoding the mapping logic, implement fuzzy matching:

```python
def find_beta_column(product_name, beta_df):
    """Find the actual beta column for a product using fuzzy matching."""
    # Try multiple patterns
    patterns = [
        f"Beta_{product_name}",  # Kalman converted format
        f"{product_name}_meta_impression",  # Standard product format
        f"{product_name.lower()}_meta_impression",  # Lowercase variant
        product_name,  # Exact match
    ]
    
    for pattern in patterns:
        if pattern in beta_df.columns:
            return pattern
    
    # Try partial match
    for col in beta_df.columns:
        if product_name.lower() in col.lower():
            return col
    
    return None
```

---

## Test Coverage

The test successfully validates:
- ✅ Format detection for both Kalman and standard formats
- ✅ Conversion from Kalman to standard format
- ✅ Preservation of values during conversion
- ✅ Auto-conversion function behavior
- ❌ Product to beta column mapping (identified the bug)
- ❌ Real file compatibility (identified format issues)

---

## Next Steps

1. **Update `product_to_beta_column()` function** in `optimiser/utils/data_utils.py`
2. **Add flexible column matching** to handle different file formats
3. **Improve format detection** to distinguish beta files from data files
4. **Re-run tests** to verify fixes
5. **Add integration tests** with real optimizer workflow

---

## Conclusion

The beta mapping test successfully identified that:
1. **Format conversion works correctly** - Kalman format is properly converted to standard format
2. **The mapping function has bugs** - It doesn't match actual column naming conventions
3. **File format detection needs improvement** - Some files are misclassified

The core conversion logic is solid, but the product-to-column mapping needs to be more flexible to handle the various naming conventions used in different files.
