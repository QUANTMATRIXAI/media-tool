# Utils Folder Usage - CORRECTED

## Summary

After investigation, here's the **correct** answer:

### ❌ **Root `utils/` folder** 
- **Status**: **NOT USED BY ANYONE**
- **Reason**: Old/outdated files that are no longer needed
- **Action**: ✅ Moved to `additional/` folder

---

## Actual Utils Folders in Use:

### 1. **`optimiser/utils/`** - Used by Optimizer
- **Location**: `/optimiser/utils/`
- **Used by**: **Optimizer module** (`optimiser/app.py`)
- **Files**:
  - `beta_converter.py` - Convert Kalman to standard beta format
  - `data_utils.py` - Data loading and validation
  - `optimization_utils.py` - Optimization algorithms
  - `results_display.py` - Results formatting

### 2. **`modelling/utils.py`** - Used by Modelling
- **Location**: `/modelling/utils.py` (single file, not a folder)
- **Used by**: **Modelling module** (`modelling/app.py`, `modelling/pipeline.py`)
- **Functions**:
  - `safe_mape()` - Calculate MAPE safely
  - `create_ensemble_model_from_results()` - Create ensemble models
  - `build_weighted_ensemble_model()` - Build weighted ensembles

---

## Which App Uses Which Utils?

### ✅ **Main App** (`app_structure.py`)
- Uses: **None** - No utils folder
- Just loads modules

### ✅ **EDA Module** (`EDA.py`)
- Uses: **None** - No utils folder
- Self-contained

### ✅ **Modelling Module** (`modelling/`)
- Uses: **`modelling/utils.py`** (its own utils file)
- Imports:
  ```python
  from utils import safe_mape
  from utils import create_ensemble_model_from_results
  ```
  (These import from `modelling/utils.py`, not root utils!)

### ✅ **Optimizer Module** (`optimiser/`)
- Uses: **`optimiser/utils/`** (its own utils folder)
- Imports:
  ```python
  from utils.data_utils import ...
  from utils.beta_converter import ...
  from utils.optimization_utils import ...
  from utils.results_display import ...
  ```

### ✅ **Kalman Modeling** (`kalman modleling/`)
- Uses: **None** - No utils folder
- Self-contained

---

## Why Was Root `utils/` Confusing?

1. **It looked like it was needed** because modelling imports `from utils`
2. **But Python imports are relative** - when modelling does `from utils import`, it imports from `modelling/utils.py`, NOT from root `utils/`
3. **Root `utils/` was never actually imported** by any file
4. **It was just old code** left over from earlier development

---

## Conclusion

### Answer to Your Question:

**The root `utils/` folder was NOT used by anyone!** ❌

It contained old/outdated copies of optimizer utils files that were never actually imported. The confusion came from:
- Modelling imports `from utils` → but this imports from `modelling/utils.py`
- Optimizer imports `from utils` → but this imports from `optimiser/utils/`

**Each module has its own utils:**
- Modelling: `modelling/utils.py` ✅
- Optimizer: `optimiser/utils/` ✅
- Root utils: Not used ❌ (moved to additional/)

---

## Current Clean Structure:

```
project_root/
│
├── app_structure.py
├── EDA.py
├── database.py
├── file_manager.py
├── project_selector_page.py
├── comment_manager.py
│
├── optimiser/
│   ├── app.py
│   └── utils/                    ← Optimizer's utils
│       ├── data_utils.py
│       ├── beta_converter.py
│       ├── optimization_utils.py
│       └── results_display.py
│
├── modelling/
│   ├── app.py
│   ├── pipeline.py
│   ├── models.py
│   └── utils.py                  ← Modelling's utils
│
├── kalman modleling/
│   └── kalman.py
│
└── additional/                   ← Old/unused files
    └── utils/                    ← Old root utils (not used)
```

**Clean and organized!** ✨
