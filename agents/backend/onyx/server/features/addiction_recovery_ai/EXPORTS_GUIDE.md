# Exports Guide - Addiction Recovery AI

## ✅ Recommended Export Files

### Export Files Overview

The project uses export files (`_*_exports.py`) to facilitate convenient imports:

```
addiction_recovery_ai/
├── _core_exports.py              # ✅ Core component exports
├── _core_modules_exports.py      # ✅ Core module exports
├── _core_system_exports.py       # ✅ Core system exports
├── _layers_exports.py             # ✅ Layer component exports
├── _models_exports.py             # ✅ Model exports
├── _training_exports.py            # ✅ Training component exports
└── _utils_exports.py              # ✅ Utility exports
```

## 📦 Export Files

### `_core_exports.py` - Core Components
- **Status**: ✅ Active
- **Purpose**: Exports core components
- **Exports**:
  - `AddictionAnalyzer`
  - `RecoveryPlanner`
  - `ProgressTracker`
  - `RelapsePrevention`
  - Base classes (`BaseModel`, `BasePredictor`, etc.)

**Usage:**
```python
from _core_exports import (
    AddictionAnalyzer,
    RecoveryPlanner,
    ProgressTracker,
    RelapsePrevention
)
```

### `_core_modules_exports.py` - Core Modules
- **Status**: ✅ Active
- **Purpose**: Exports core module system components
- **Exports**: Module loader, registry, etc.

### `_core_system_exports.py` - Core System
- **Status**: ✅ Active
- **Purpose**: Exports core system components
- **Exports**: System-level components

### `_layers_exports.py` - Layer Components
- **Status**: ✅ Active
- **Purpose**: Exports layer architecture components
- **Exports**: Service layer, model layer, etc.

### `_models_exports.py` - Model Exports
- **Status**: ✅ Active
- **Purpose**: Exports ML model components
- **Exports**: Model classes, model factories, etc.

### `_training_exports.py` - Training Components
- **Status**: ✅ Active
- **Purpose**: Exports training components
- **Exports**: Trainers, callbacks, etc.

### `_utils_exports.py` - Utility Exports
- **Status**: ✅ Active
- **Purpose**: Exports utility components
- **Exports**: Utility functions, helpers, etc.

## 📝 Usage Examples

### Using Core Exports
```python
from _core_exports import (
    AddictionAnalyzer,
    RecoveryPlanner,
    ProgressTracker,
    RelapsePrevention
)

analyzer = AddictionAnalyzer()
planner = RecoveryPlanner()
```

### Using Layer Exports
```python
from _layers_exports import ServiceLayer, ModelLayer

service_layer = ServiceLayer()
model_layer = ModelLayer()
```

### Using Model Exports
```python
from _models_exports import BaseModel, ModelFactory

model = ModelFactory.create_model(...)
```

## 🎯 Quick Reference

| Export File | Purpose | Status | When to Use |
|-------------|---------|--------|-------------|
| `_core_exports.py` | Core components | ✅ Active | Core functionality |
| `_core_modules_exports.py` | Core modules | ✅ Active | Module system |
| `_core_system_exports.py` | Core system | ✅ Active | System components |
| `_layers_exports.py` | Layer components | ✅ Active | Layered architecture |
| `_models_exports.py` | Model exports | ✅ Active | ML models |
| `_training_exports.py` | Training exports | ✅ Active | Training components |
| `_utils_exports.py` | Utility exports | ✅ Active | Utilities |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `CORE_GUIDE.md` for core components
- See `SERVICES_GUIDE.md` for services
- See `UTILITIES_GUIDE.md` for utilities






