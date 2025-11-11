# Code Files Structure - Complete App

## Overview
This document lists all the Python code files required to run the complete analytics app.

---

## Root Directory Files

### 1. **app_structure.py** (Main Entry Point)
- **Purpose**: Main application file that loads all modules
- **Features**:
  - Project selector
  - Module navigation (EDA, Modeling, Optimization, Kalman)
  - User authentication
  - Database integration

### 2. **EDA.py**
- **Purpose**: Exploratory Data Analysis module
- **Features**:
  - Overview tab (summary statistics)
  - Explore tab (interactive charts)
  - Correlation tab (heatmap)
  - Cardinality tab (unique values)
  - Clustering tab (K-Means, DBSCAN, Hierarchical)
  - Data filters

### 3. **database.py**
- **Purpose**: Database management for user data and projects
- **Features**:
  - SQLite database operations
  - User management
  - Project management
  - File storage per user/project
  - Progress tracking

### 4. **file_manager.py**
- **Purpose**: File upload and management
- **Features**:
  - File upload handling
  - File storage in database
  - File retrieval
  - File type validation

### 5. **project_selector_page.py**
- **Purpose**: Project selection interface
- **Features**:
  - Project list display
  - Project creation
  - Project search and filtering
  - Grid/List view modes
  - Sorting options

### 6. **comment_manager.py**
- **Purpose**: Comment system for collaboration
- **Features**:
  - Add comments
  - View comments
  - Comment storage

---

## Optimizer Module (`optimiser/`)

### 7. **optimiser/app.py** (Main Optimizer)
- **Purpose**: Budget optimization module
- **Features**:
  - **Tab 1 - Configuration**:
    - Week selection
    - Budget constraint settings
    - Budget editor
    - CPM editor
  - **Tab 2 - Results**:
    - Budget comparison table
    - Product-specific beta & budget analysis
  - **Tab 3 - Contribution Analysis**:
    - Portfolio contribution chart
    - Variable filtering
  - **Tab 4 - Pricing Strategy**:
    - Price elasticity analysis
    - Demand curve
    - Revenue curve
    - Optimal pricing recommendation

### 8. **optimiser/utils/data_utils.py**
- **Purpose**: Data loading and validation utilities
- **Functions**:
  - `load_file()` - Load CSV/Excel files
  - `validate_budget_file()` - Validate budget file structure
  - `validate_cpm_file()` - Validate CPM file
  - `validate_beta_file()` - Validate beta coefficients file
  - `validate_price_file()` - Validate price file
  - `merge_data()` - Merge all data sources
  - `product_to_beta_column()` - Map product names to beta columns
  - `get_channel_beta_mapping()` - Get channel to beta mapping
  - `get_channel_beta_mapping_with_fallback()` - Smart mapping with fallback

### 9. **optimiser/utils/beta_converter.py**
- **Purpose**: Convert between beta file formats
- **Functions**:
  - `detect_beta_format()` - Detect standard vs Kalman format
  - `convert_kalman_to_standard()` - Convert Kalman to standard format
  - `auto_convert_beta_file()` - Auto-detect and convert
  - `get_format_info()` - Get format information

### 10. **optimiser/utils/optimization_utils.py**
- **Purpose**: Optimization algorithms and calculations
- **Functions**:
  - `calculate_impressions()` - Calculate impressions from budget
  - `create_impression_dict()` - Create impression dictionary
  - `predict_all_volumes()` - Predict volumes for all products
  - `calculate_revenue()` - Calculate revenue
  - `create_objective_function()` - Create optimization objective
  - `create_bounds()` - Create budget bounds
  - `optimize_budgets()` - Run optimization algorithm

### 11. **optimiser/utils/results_display.py**
- **Purpose**: Results formatting and display
- **Functions**:
  - `create_comparison_table()` - Create budget comparison table
  - `format_currency()` - Format currency values

---

## Modeling Module (`modelling/`)

### 12. **modelling/app.py**
- **Purpose**: Machine learning modeling module
- **Features**:
  - Regression models
  - Variation models
  - Ensemble/Classification models
  - Model training and evaluation

---

## Kalman Modeling Module (`kalman modleling/`)

### 13. **kalman modleling/kalman.py**
- **Purpose**: Time-varying coefficient modeling using Kalman filter
- **Features**:
  - Kalman filter implementation
  - Product-specific beta estimation
  - Time-varying coefficient tracking
  - Model evaluation metrics
  - Export to Kalman_betas.csv format

---

## Supporting Files

### 14. **custom.py** (Optional)
- **Purpose**: Custom styling and configurations
- **Features**:
  - Custom CSS
  - Theme settings
  - UI customizations

### 15. **tv_kalman_app.py** (Alternative Entry Point)
- **Purpose**: Alternative main app file
- **Features**: Similar to app_structure.py

---

## Complete File Tree

```
project_root/
│
├── app_structure.py              # Main entry point
├── EDA.py                         # EDA module
├── database.py                    # Database management
├── file_manager.py                # File management
├── project_selector_page.py       # Project selector
├── comment_manager.py             # Comment system
├── custom.py                      # Custom styling (optional)
│
├── optimiser/                     # Optimizer module
│   ├── app.py                     # Main optimizer app
│   └── utils/
│       ├── data_utils.py          # Data utilities
│       ├── beta_converter.py      # Beta format converter
│       ├── optimization_utils.py  # Optimization algorithms
│       └── results_display.py     # Results formatting
│
├── modelling/                     # Modeling module
│   └── app.py                     # Modeling app
│
└── kalman modleling/              # Kalman modeling module
    └── kalman.py                  # Kalman filter implementation
```

---

## Dependencies by Module

### **EDA Module**
- `EDA.py`
- `database.py` (for file storage)
- `file_manager.py` (for file uploads)

### **Modeling Module**
- `modelling/app.py`
- `database.py`
- `file_manager.py`

### **Kalman Modeling Module**
- `kalman modleling/kalman.py`
- `database.py`
- `file_manager.py`

### **Optimizer Module**
- `optimiser/app.py`
- `optimiser/utils/data_utils.py`
- `optimiser/utils/beta_converter.py`
- `optimiser/utils/optimization_utils.py`
- `optimiser/utils/results_display.py`
- `database.py`
- `file_manager.py`

### **Main App**
- `app_structure.py`
- `project_selector_page.py`
- `database.py`
- `file_manager.py`
- `comment_manager.py`
- All module files (EDA, Modeling, Optimizer, Kalman)

---

## Required Python Packages

```python
# Core
streamlit
pandas
numpy

# Visualization
plotly
matplotlib
seaborn

# Optimization
scipy

# Machine Learning
scikit-learn

# Database
sqlite3 (built-in)

# File Handling
openpyxl  # For Excel files
```

---

## Minimum Files to Run

### **Just EDA:**
- `app_structure.py` (or run `EDA.py` directly)
- `EDA.py`
- `database.py`
- `file_manager.py`

### **Just Optimizer:**
- `app_structure.py`
- `optimiser/app.py`
- `optimiser/utils/data_utils.py`
- `optimiser/utils/beta_converter.py`
- `optimiser/utils/optimization_utils.py`
- `optimiser/utils/results_display.py`
- `database.py`
- `file_manager.py`

### **Complete App:**
All 15 files listed above

---

## Total Code Files: 15

### Core Files: 7
1. app_structure.py
2. EDA.py
3. database.py
4. file_manager.py
5. project_selector_page.py
6. comment_manager.py
7. custom.py (optional)

### Optimizer Files: 5
8. optimiser/app.py
9. optimiser/utils/data_utils.py
10. optimiser/utils/beta_converter.py
11. optimiser/utils/optimization_utils.py
12. optimiser/utils/results_display.py

### Modeling Files: 1
13. modelling/app.py

### Kalman Files: 1
14. kalman modleling/kalman.py

### Alternative Entry: 1
15. tv_kalman_app.py (optional)

---

## How to Run

### Option 1: Run Complete App
```bash
streamlit run app_structure.py
```

### Option 2: Run Individual Modules
```bash
# EDA only
streamlit run EDA.py

# Optimizer only
cd optimiser
streamlit run app.py

# Kalman modeling only
cd "kalman modleling"
streamlit run kalman.py
```

---

## Notes:
- All files use relative imports
- Database file (`user_data.db`) is created automatically
- File structure must be maintained for imports to work
- Optional files can be omitted without breaking core functionality
