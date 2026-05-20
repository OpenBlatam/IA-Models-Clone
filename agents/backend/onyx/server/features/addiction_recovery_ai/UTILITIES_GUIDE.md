# Utilities Guide - Addiction Recovery AI

## ✅ Recommended Utilities Structure

### `utils/` Directory - **USE THIS**

The utilities are organized in a categorized structure:

```
utils/
├── categories/              # ✅ Categorized utilities
│   ├── api/                 # API utilities
│   ├── async/               # Async utilities
│   ├── data/                # Data utilities
│   ├── functional/          # Functional programming utilities
│   ├── helpers/             # Helper utilities
│   ├── logging/             # Logging utilities
│   ├── ml/                  # Machine learning utilities
│   ├── monitoring/          # Monitoring utilities
│   ├── performance/         # Performance utilities
│   ├── rate_limiting/       # Rate limiting utilities
│   ├── resilience/          # Resilience utilities
│   ├── scheduling/          # Scheduling utilities
│   ├── security/            # Security utilities
│   ├── serialization/       # Serialization utilities
│   └── validation/          # Validation utilities
├── utility_factory.py       # ✅ Utility factory for centralized access
└── [specific utilities]     # Domain-specific utilities
```

## 📦 Utility Categories

### Core Utilities

#### `utils/helpers.py` - General Helpers
- **Status**: ✅ Active
- **Purpose**: General helper functions for recovery system
- **Features**: Days sober calculation, recovery stage calculation, etc.

#### `utils/utility_factory.py` - Utility Factory
- **Status**: ✅ Active
- **Purpose**: Centralized utility creation and management
- **Usage**:
```python
from utils.utility_factory import UtilityFactory

factory = UtilityFactory()
utility = factory.get_utility('data', 'data_processor')
```

### Categorized Utilities

#### `utils/categories/` - Organized by Category
- **Status**: ✅ Active
- **Purpose**: Utilities organized by functional category
- **Structure**: Each category has its own `__init__.py` with exports

**Available Categories:**
- `api/` - API-related utilities
- `async/` - Async/await utilities
- `data/` - Data processing utilities
- `functional/` - Functional programming utilities
- `helpers/` - Helper functions
- `logging/` - Logging utilities
- `ml/` - Machine learning utilities
- `monitoring/` - Monitoring utilities
- `performance/` - Performance optimization utilities
- `rate_limiting/` - Rate limiting utilities
- `resilience/` - Resilience patterns (circuit breakers, retries)
- `scheduling/` - Scheduling utilities
- `security/` - Security utilities
- `serialization/` - Serialization utilities
- `validation/` - Validation utilities

### Specialized Utilities

#### Helper Files
- `utils/string_helpers.py` - String manipulation
- `utils/date_helpers.py` - Date/time utilities
- `utils/math_helpers.py` - Math utilities
- `utils/collection_helpers.py` - Collection utilities
- `utils/performance_helpers.py` - Performance helpers
- `utils/async_helpers.py` - Async helpers
- `utils/functional_helpers.py` - Functional programming helpers
- `utils/pydantic_helpers.py` - Pydantic helpers
- `utils/testing_helpers.py` - Testing helpers

#### Utility Files
- `utils/data_utils.py` - Data utilities
- `utils/file_utils.py` - File utilities
- `utils/time_utils.py` - Time utilities
- `utils/logging_utils.py` - Logging utilities
- `utils/config_utils.py` - Configuration utilities
- `utils/queue_utils.py` - Queue utilities
- `utils/middleware_utils.py` - Middleware utilities
- `utils/model_utils.py` - Model utilities

## 📝 Usage Examples

### Using Utility Factory
```python
from utils.utility_factory import UtilityFactory

factory = UtilityFactory()

# Get a utility by category and name
data_processor = factory.get_utility('data', 'data_processor')
validator = factory.get_utility('validation', 'schema_validator')
```

### Direct Import from Categories
```python
from utils.categories.data import DataProcessor
from utils.categories.validation import SchemaValidator
from utils.categories.async import AsyncHelper
```

### Using Specific Helpers
```python
from utils.helpers import calculate_days_sober, calculate_recovery_stage
from utils.string_helpers import sanitize_string
from utils.date_helpers import format_date
```

### Using Core Utilities
```python
from utils.data_utils import process_data
from utils.file_utils import read_file
from utils.logging_utils import setup_logger
```

## 🏗️ Utilities Structure

```
utils/
├── categories/              # ✅ Categorized utilities (recommended)
│   ├── api/
│   ├── async/
│   ├── data/
│   └── ...
├── utility_factory.py       # ✅ Utility factory
├── helpers.py               # ✅ General helpers
├── [category]_helpers.py    # ✅ Category-specific helpers
└── [domain]_utils.py        # ✅ Domain-specific utilities
```

## 🎯 Quick Reference

### For General Helpers
```python
from utils.helpers import calculate_days_sober
```

### For Categorized Utilities
```python
from utils.categories.validation import validate_schema
from utils.categories.data import process_data
```

### For Utility Factory
```python
from utils.utility_factory import UtilityFactory
factory = UtilityFactory()
utility = factory.get_utility('category', 'utility_name')
```

### For Specific Utilities
```python
from utils.string_helpers import sanitize
from utils.date_helpers import format_date
from utils.file_utils import read_file
```

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `DOCUMENTATION_INDEX.md` for complete documentation index
- See utility documentation in `utils/categories/` for category-specific docs






