# Utils Folder Usage

## Summary

There are **TWO** separate `utils/` folders in the project:

### 1. **Root `utils/` folder**
- **Location**: `/utils/`
- **Used by**: **Modelling module** (`modelling/app.py`, `modelling/pipeline.py`)
- **Files**:
  - `beta_mapper.py`
  - `data_utils.py`
  - `optimization_utils.py`
  - `results_display.py`

### 2. **Optimizer `utils/` folder**
- **Location**: `/optimiser/utils/`
- **Used by**: **Optimizer module** (`optimiser/app.py`)
- **Files**:
  - `beta_converter.py` (unique to optimizer)
  - `data_utils.py` (different from root utils)
  - `optimization_utils.py` (different from root utils)
  - `results_display.py` (similar to root utils)

---

## Which App Uses Which Utils?

### ✅ **Main App** (`app_structure.py`)
- Uses: **Root `utils/`** (only for modelling module loading)
- Import: `import utils as modelling_utils`

### ✅ **EDA Module** (`EDA.py`)
- Uses: **None** - No utils folder
- Self-contained module

### ✅ **Modelling Module** (`modelling/app.py`, `modelling/pipeline.py`)
- Uses: **Root `utils/`**
- Imports:
  - `from utils import safe_mape`
  - `from utils import create_ensemble_model_from_results`

### ✅ **Optimizer Module** (`optimiser/app.py`)
- Uses: **`optimiser/utils/`** (its own utils folder)
- Imports:
  - `from utils.data_utils import ...`
  - `from utils.beta_converter import ...`
  - `from utils.optimization_utils import ...`
  - `from utils.results_display import ...`

### ✅ **Kalman Modeling** (`kalman modleling/kalman.py`)
- Uses: **None** - No utils folder
- Self-contained module

---

## File Comparison

### Files in Both Locations:

| File | Root utils/ | optimiser/utils/ | Status |
|------|-------------|------------------|--------|
| `data_utils.py` | 10,130 bytes | 12,559 bytes | **Different** - Optimizer version is newer |
| `optimization_utils.py` | 13,137 bytes | 18,811 bytes | **Different** - Optimizer version is newer |
| `results_display.py` | 2,617 bytes | 2,615 bytes | **Similar** - Almost identical |
| `beta_mapper.py` | 2,801 bytes | 2,799 bytes | **Similar** - Almost identical |

### Unique Files:

| File | Location | Used By |
|------|----------|---------|
| `beta_converter.py` | `optimiser/utils/` only | Optimizer (for Kalman format conversion) |

---

## Recommendation

### ⚠️ Root `utils/` folder is REQUIRED for:
- **Modelling module** to work

### ✅ `optimiser/utils/` folder is REQUIRED for:
- **Optimizer module** to work

### 🔍 Status:
- Root `utils/` appears to be an **older version** of the optimizer utils
- The files are **outdated** compared to `optimiser/utils/`
- They are kept because **modelling module depends on them**

---

## Conclusion

**Answer: The root `utils/` folder is used by the MODELLING module.**

- Main app: Uses root utils (indirectly through modelling)
- EDA: No utils
- Modelling: Uses root utils ✅
- Optimizer: Uses its own optimiser/utils ✅
- Kalman: No utils

**Both utils folders are needed** - they serve different modules and cannot be removed without breaking functionality.
