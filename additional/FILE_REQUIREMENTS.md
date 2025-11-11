# File Requirements for Complete App

## Overview
This document lists all the data files required to run each feature of the analytics app.

---

## 1. EDA Module (Exploratory Data Analysis)

### Required Files:
1. **Data file** (CSV or Excel)
   - Contains: Date, Product columns, Sales metrics, Impression variables
   - Example columns: Date, Product title, Gross sales, Net sales, Impressions, etc.
   - Used for: All EDA tabs (Overview, Explore, Correlation, Cardinality, Clustering)

### Features Available:
- ✅ Overview tab - Summary statistics
- ✅ Explore tab - Interactive charts and filters
- ✅ Correlation tab - Correlation heatmap
- ✅ Cardinality tab - Unique value analysis
- ✅ Clustering tab - K-Means, DBSCAN, Hierarchical clustering

---

## 2. Modeling Module

### Required Files:
1. **Data file** (CSV or Excel)
   - Same as EDA data file
   - Contains: Time series data with features and target variable

### Features Available:
- ✅ Regression models
- ✅ Variation models
- ✅ Ensemble/Classification models

---

## 3. Kalman Modeling Module

### Required Files:
1. **Data file** (CSV or Excel)
   - Time series data with product-level information
   - Contains: Date, Product, Sales, Impressions, other variables

### Features Available:
- ✅ Time-varying coefficient modeling
- ✅ Kalman filter estimation
- ✅ Product-specific beta coefficients over time

### Output Files Generated:
- `Kalman_betas.csv` - Final period beta coefficients for each product

---

## 4. Optimizer Module

### Required Files (4 mandatory + 2 optional):

#### **Mandatory Files:**

1. **Budget Allocation File** (`budget_allocation.csv` or `.xlsx`)
   - **Columns**: 
     - Column 1: Product/Channel name (e.g., "google campaigns", "traffic(all sku's)", "Other Products")
     - Remaining columns: Week names (e.g., "6th-12th", "13th-19th")
   - **Purpose**: Current budget allocation for each channel per week
   - **Example**:
     ```
     Channel Name       | 6th-12th | 13th-19th | 20th-26th
     google campaigns   | 10000    | 12000     | 11000
     traffic(all sku's) | 8000     | 9000      | 8500
     Other Products     | 15000    | 16000     | 15500
     ```

2. **CPM File** (`cpm.csv` or `CPM.xlsx`)
   - **Columns**: 
     - Item Name (matches budget allocation names)
     - CPM (Cost Per Thousand impressions)
   - **Purpose**: Cost per thousand impressions for each channel
   - **Example**:
     ```
     Item Name          | CPM
     google campaigns   | 5.50
     traffic(all sku's) | 4.20
     Other Products     | 3.80
     ```

3. **Beta Coefficients File** (`betas.csv` or `Kalman_betas.csv`)
   - **Two formats supported:**
   
   **A. Standard Format:**
   - **Columns**:
     - Product title
     - B0 (Intercept)
     - Beta_* columns (e.g., Beta_Google_Impression, Beta_Impressions, Beta_google_trends)
   - **Example**:
     ```
     Product title | B0    | Beta_Google_Impression | Beta_Impressions | Beta_google_trends
     Product A     | 100.5 | 0.00045               | 0.00032          | 0.15
     Product B     | 85.2  | 0.00038               | 0.00028          | 0.12
     ```
   
   **B. Kalman Format** (auto-converted):
   - **Columns**:
     - Product
     - R2, MAE, RMSE, MAPE (metrics)
     - Intercept
     - Variable columns (e.g., Google_Impression, Impressions, google_trends)
   - **Example**:
     ```
     Product   | R2   | MAE  | Intercept | Google_Impression | Impressions | google_trends
     Product A | 0.85 | 1200 | 100.5     | 0.00045          | 0.00032     | 0.15
     Product B | 0.78 | 1500 | 85.2      | 0.00038          | 0.00028     | 0.12
     ```

4. **Product Prices File** (`product_prices.csv`)
   - **Columns**:
     - Product name
     - Price
   - **Purpose**: Price for each product (used for revenue calculation)
   - **Example**:
     ```
     Product           | Price
     Product A         | 150.00
     Product B         | 200.00
     Other Products    | 0.00
     ```

#### **Optional Files:**

5. **Google Trends File** (`Seasonality_google_trend_extended.csv`) - OPTIONAL
   - **Columns**:
     - Week (date format: DD-MM-YYYY)
     - Multiple trend columns for different product categories
   - **Purpose**: Seasonality adjustment based on Google Trends data
   - **Example**:
     ```
     Week       | Jewelry: (Worldwide) | Necklaces: (Worldwide) | Rings: (Worldwide)
     01-01-2024 | 75                   | 68                     | 82
     08-01-2024 | 78                   | 70                     | 85
     ```

6. **Modeling Data File** (`Data_for_model.xlsx`) - OPTIONAL
   - **Columns**:
     - Date
     - Product title
     - Sales metrics (Gross sales, Net sales, etc.)
     - Impression variables
     - Other model variables (Category Discount, Product variant price, google_trends)
   - **Purpose**: Historical data for calculating variable means (used in contribution analysis)
   - **Example**:
     ```
     Date       | Product title | Gross sales | Impressions | google_trends | Category Discount
     2024-01-01 | Product A     | 5000        | 100000      | 75            | 10
     2024-01-08 | Product A     | 5500        | 110000      | 78            | 12
     ```

---

## 5. Optimizer Features & File Requirements

### **Configuration Tab** (Tab 1)
- **Required**: Budget, CPM, Beta, Price files
- **Optional**: Google Trends
- **Features**:
  - Week selection
  - Budget constraint settings (percentage or absolute)
  - Budget editor (edit base budgets)
  - CPM editor (edit CPM values)

### **Results Tab** (Tab 2)
- **Required**: All mandatory files + optimization must be run
- **Features**:
  - Budget comparison table (base vs optimized)
  - Product-specific beta & budget analysis (expander)
    - Shows all variables affecting selected product
    - Displays beta coefficients and budget allocations

### **Contribution Analysis Tab** (Tab 3)
- **Required**: All mandatory files + Modeling Data file
- **Features**:
  - Portfolio-level contribution chart
  - Shows only channels with budget allocation:
    - Google Ads Impressions
    - Traffic Ads Impressions
    - Other Products (Meta Ads)
  - Excludes fixed variables (Google Trends, Category Discount, Product Variant Price)

### **Pricing Strategy Tab** (Tab 4) - NEW!
- **Required**: Beta file, Price file
- **Features**:
  - Price elasticity calculation
  - Demand curve visualization
  - Revenue curve visualization
  - Optimal price recommendation
  - Price scenario analysis table

---

## 6. Channel Mapping

### Budget Item Names → Beta Columns

The optimizer maps budget items to beta coefficients using these rules:

1. **Google Campaigns**
   - Budget item: `"google campaigns"` (case-insensitive)
   - Maps to: `Beta_Google_Impression`

2. **Traffic Ads**
   - Budget item: `"traffic(all sku's)"` (case-insensitive)
   - Maps to: 
     - `Beta_Daily_Impressions_LINK_CLICKS` (Kalman format)
     - `Beta_Daily_Impressions_OUTCOME_ENGAGEMENT` (Standard format)
   - Auto-detects which column exists

3. **Other Products / Meta Ads**
   - Budget item: `"Other Products"` (case-insensitive)
   - Maps to: `Beta_Impressions`

4. **Individual Products**
   - Budget item: Any other product name
   - Maps to: `Beta_{product_name}_meta_impression`
   - Note: These are typically not included in budget allocation

---

## 7. File Format Support

### Supported Formats:
- ✅ CSV (`.csv`)
- ✅ Excel (`.xlsx`, `.xls`)

### Auto-Detection:
- ✅ Standard beta format (with `Beta_` prefix)
- ✅ Kalman beta format (auto-converts to standard)
- ✅ Column name variations (case-insensitive matching)

---

## 8. Complete File Checklist

### For Full App Functionality:

#### **Minimum Required (Core Features):**
- [ ] Data file for EDA/Modeling
- [ ] Budget allocation file
- [ ] CPM file
- [ ] Beta coefficients file (standard or Kalman format)
- [ ] Product prices file

#### **Recommended (Enhanced Features):**
- [ ] Google Trends file (for seasonality adjustment)
- [ ] Modeling data file (for contribution analysis)

#### **Generated Files (Outputs):**
- [ ] Kalman_betas.csv (from Kalman modeling)
- [ ] Optimized budget recommendations (from optimizer)

---

## 9. File Location

### Auto-Load Locations:
The app automatically looks for files in the root directory with these names:
- `budget_allocation.csv` or `budget.csv`
- `cpm.csv` or `CPM.xlsx`
- `betas.csv` or `Kalman_betas.csv`
- `product_prices.csv` or `prices.csv`
- `Seasonality_google_trend_extended.csv` or `google_trends.csv`
- `Data_for_model.xlsx` or `modeling_data.xlsx`

### Manual Upload:
If files are not found automatically, you can upload them via the sidebar in the Optimizer module.

---

## 10. Summary

### Files by Priority:

**Priority 1 (Essential):**
1. Budget allocation file
2. CPM file
3. Beta coefficients file
4. Product prices file

**Priority 2 (Important):**
5. Modeling data file (for contribution analysis)

**Priority 3 (Optional):**
6. Google Trends file (for seasonality)

**Total Files Needed for Complete Functionality: 4-6 files**

---

## Notes:
- All files support both CSV and Excel formats
- Column names are case-insensitive
- The app auto-detects and converts Kalman beta format
- Missing optional files will disable specific features but won't break the app
- The app shows clear error messages if required files are missing or invalid
