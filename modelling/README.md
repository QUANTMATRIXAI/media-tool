# Regression Modeling Application

## 📁 Project Structure

This application has been refactored into modular components for better maintainability:

```
modelling/
├── app.py                      # Streamlit UI (1,279 lines)
├── models.py                   # Custom model classes (753 lines)
├── utils.py                    # Helper & ensemble functions (396 lines)
├── pipeline.py                 # Main modeling pipeline (1,454 lines)
└── app_original_backup.py      # Original monolithic file (3,813 lines)
```

## 🔧 Module Descriptions

### `app.py` - Streamlit User Interface
The main entry point for the application. Contains:
- Data upload and configuration UI
- Model selection interface
- Results visualization and dashboards
- Download functionality

**Key Function:** `main()` - Streamlit application orchestrator

---

### `models.py` - Custom Regression Models
Contains all custom model implementations:

1. **`CustomConstrainedRidge`** - Ridge regression with coefficient sign constraints
   - Supports L2 regularization
   - Adam optimizer option
   - Non-negative/non-positive feature constraints

2. **`ConstrainedLinearRegression`** - Linear regression with sign constraints
   - No L2 penalty
   - Gradient descent with constraints
   - Adam optimizer support

3. **`RecursiveLeastSquaresRegressor`** - Sequential coefficient updating
   - Standard RLS with configurable forgetting factor
   - Supports optional coefficient history tracking
   - Provides incremental `update()` method for streaming data

4. **`StackedInteractionModel`** - Group-specific coefficient modeling
   - Creates interaction terms for each group
   - Dummy variable encoding
   - Extracts per-group coefficients

5. **`StatsMixedEffectsModel`** - Wrapper for statsmodels MixedLM
   - Random effects per group
   - Fallback to LinearRegression if mixed effects fail
   - Minimum group size filtering

---

### `utils.py` - Helper & Ensemble Functions

**Helper Functions:**
- `safe_mape()` - MAPE calculation with outlier protection

**Ensemble Functions:**
- `build_weighted_ensemble_model()` - Exponential weighting based on MAPE
- `create_ensemble_model_from_results()` - Creates ensembles from CV results with filtering

---

### `pipeline.py` - Main Modeling Pipeline

**Key Function:** `run_model_pipeline()`

Orchestrates the entire modeling workflow:

1. **Data Splitting**
   - Groups data by specified keys
   - Filters groups by Y-variable share
   - Creates train/test splits for evaluation

2. **Cross-Validation**
   - K-fold CV with adaptive fold selection (2-5 folds based on sample size)
   - Separate processing for regular vs. stacked models
   - Standardization and log transformation support

3. **Model Training**
   - Trains multiple model types per group
   - Handles custom constrained models
   - Tracks coefficients and metrics

4. **Ensemble Creation** (Optional)
   - Filters models by R², MAPE, MAE, sign constraints
   - Weighted averaging of coefficients

**Returns:**
- `results_df` - Cross-validation results with metrics
- `preds_df` - Predictions for all folds
- `ensemble_df` - Ensemble model results (if enabled)

---

## 🚀 Usage

### Running the Application

```bash
streamlit run app.py
```

### Importing Modules

```python
# Import model classes
from models import CustomConstrainedRidge

# Import utilities
from utils import safe_mape

# Import pipeline
from pipeline import run_model_pipeline
```

---

## 📊 Key Features

### 1. **Multi-Model Comparison**
- Linear Regression
- Ridge, Lasso, ElasticNet
- Bayesian Ridge
- Recursive Least Squares (configurable λ and covariance)
- Custom Constrained Ridge
- Constrained Linear Regression
- Mixed Effects Models

### 2. **Stacked Interaction Models**
- Group-specific coefficients via interaction terms
- Separate filtering vs. interaction keys
- Automatic dummy variable creation

### 3. **Ensembles & Diagnostics**
- Weighted averaging of eligible base models
- Sign-based model filtering
- Detailed metrics, coefficients, and download options

### 4. **Ensemble Modeling**
- Weighted averaging across models
- MAPE-based exponential weighting
- Optional filtering by:
  - R² threshold
  - MAPE threshold
  - MAE threshold
  - Coefficient sign correctness

### 5. **Coefficient Constraints**
- Positive constraints (≥0)
- Negative constraints (≤0)
- Applied during gradient descent
- Enforced throughout model training

### 6. **Auto-Residualization**
- Removes multicollinearity from product-specific variables
- Residualizes against primary variable
- Preserves interpretability

---

## 🔍 Code Quality Improvements

This refactoring addresses several issues from the original monolithic file:

### Fixed Issues:
✅ **Removed Duplicates**
- Eliminated duplicate `warnings.filterwarnings('ignore')`
- Removed duplicate module docstring
- Consolidated `safe_mape` to single location

✅ **Improved Exception Handling**
- Changed bare `except:` to specific exception types in models.py
- Added proper exception handling in StatsMixedEffectsModel

✅ **Better Organization**
- Separated concerns: UI, models, pipeline, utilities
- Each file has clear responsibility
- Easier to test and maintain

### Remaining Recommendations:
- Consider extracting configuration constants to `config.py`
- Add unit tests for model classes
- Add integration tests for pipeline
- Consider further splitting `pipeline.py` (still 1,454 lines)

---

## 📈 File Size Comparison

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| **Original** | 3,813 | 184 KB | Monolithic file |
| **app.py** | 1,279 | 58 KB | UI only |
| **models.py** | 753 | 30 KB | Model classes |
| **pipeline.py** | 1,454 | 74 KB | Pipeline logic |
| **utils.py** | 396 | 18 KB | Utilities |
| **Total New** | 3,882 | 180 KB | Modular structure |

---

## 🧪 Testing

Verify all modules have valid syntax:
```bash
python3 -m py_compile models.py utils.py pipeline.py app.py
```

---

## 📝 Dependencies

- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `streamlit` - Web UI framework
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning models and utilities
- `statsmodels` - Statistical models (Mixed Effects)

---

## 👥 Contributing

When modifying the code:
1. Keep modules focused on their specific responsibility
2. Add docstrings to all public functions
3. Use type hints where applicable
4. Test imports between modules
5. Update this README if structure changes

---

## 📜 Version History

- **v2.0** (2024-11-03) - Refactored into 4 modular files
- **v1.0** (2024-10-XX) - Original monolithic application
