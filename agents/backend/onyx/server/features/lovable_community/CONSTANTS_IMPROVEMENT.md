# Constants Module Improvement

## Overview

Created a centralized `constants.py` module to eliminate magic numbers and hardcoded values throughout the codebase. This improves maintainability, makes values easier to change, and provides better documentation.

## Changes Made

### 1. Created `constants.py` Module
- **Purpose**: Centralize all application constants
- **Categories**:
  - Ranking Algorithm Constants
  - Pagination Constants
  - Tag Constants
  - Time Constants
  - Validation Constants

### 2. Updated `RankingService`
- **Replaced magic numbers**:
  - `3600` → `SECONDS_PER_HOUR`
  - `0.1` → `MIN_AGE_HOURS`
  - `24` → `HOURS_PER_DAY`
  - `2.0`, `3.0`, `0.1` → `DEFAULT_VOTE_WEIGHT`, `DEFAULT_REMIX_WEIGHT`, `DEFAULT_VIEW_WEIGHT`
  - `2` → `SCORE_DECIMAL_PLACES`
  - `24` (default trending hours) → `DEFAULT_TRENDING_HOURS`

### 3. Updated `pagination.py` Helpers
- **Replaced magic numbers**:
  - `1000` → `MAX_PAGE`
  - `100` → `MAX_PAGE_SIZE`
  - `1` → `MIN_PAGE`, `MIN_PAGE_SIZE`
  - `20` → `DEFAULT_PAGE_SIZE`
  - `1` → `DEFAULT_PAGE`

## Before vs After

### Before
```python
# In ranking.py
age_hours = (now - created_at).total_seconds() / 3600
if age_hours < 0.1:
    age_hours = 0.1
vote_weight = 2.0
remix_weight = 3.0
view_weight = 0.1
time_decay = max(1.0, 1 + (age_hours / 24))
return round(score, 2)

# In pagination.py
def validate_page_params(page: int, page_size: int, max_page: int = 1000, max_page_size: int = 100):
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
```

### After
```python
# In constants.py
SECONDS_PER_HOUR = 3600
MIN_AGE_HOURS = 0.1
HOURS_PER_DAY = 24
DEFAULT_VOTE_WEIGHT = 2.0
DEFAULT_REMIX_WEIGHT = 3.0
DEFAULT_VIEW_WEIGHT = 0.1
SCORE_DECIMAL_PLACES = 2
DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE = 1000
MAX_PAGE_SIZE = 100

# In ranking.py
from ..constants import (
    SECONDS_PER_HOUR, MIN_AGE_HOURS, HOURS_PER_DAY,
    DEFAULT_VOTE_WEIGHT, DEFAULT_REMIX_WEIGHT, DEFAULT_VIEW_WEIGHT,
    SCORE_DECIMAL_PLACES
)
age_hours = (now - created_at).total_seconds() / SECONDS_PER_HOUR
if age_hours < MIN_AGE_HOURS:
    age_hours = MIN_AGE_HOURS
vote_weight = DEFAULT_VOTE_WEIGHT
time_decay = max(1.0, 1 + (age_hours / HOURS_PER_DAY))
return round(score, SCORE_DECIMAL_PLACES)

# In pagination.py
from ..constants import DEFAULT_PAGE, DEFAULT_PAGE_SIZE, MAX_PAGE, MAX_PAGE_SIZE, MIN_PAGE, MIN_PAGE_SIZE
def validate_page_params(page: int, page_size: int, max_page: int = MAX_PAGE, max_page_size: int = MAX_PAGE_SIZE):
    if page < MIN_PAGE:
        page = DEFAULT_PAGE
    if page_size < MIN_PAGE_SIZE:
        page_size = DEFAULT_PAGE_SIZE
```

## Files Modified

1. **`constants.py`** (NEW)
   - Centralized constants module
   - Well-documented constants
   - Organized by category

2. **`services/ranking.py`**
   - Replaced all magic numbers with constants
   - Improved readability
   - Easier to maintain

3. **`helpers/pagination.py`**
   - Replaced magic numbers with constants
   - Consistent default values
   - Better maintainability

## Benefits

1. **Maintainability**: All constants in one place, easy to update
2. **Readability**: Named constants are more self-documenting than magic numbers
3. **Consistency**: Same values used across the codebase
4. **Documentation**: Constants serve as documentation of default values
5. **Type Safety**: Constants can be type-checked
6. **Testing**: Easier to test with different constant values

## Constants Defined

### Ranking Algorithm
- `DEFAULT_VOTE_WEIGHT = 2.0`
- `DEFAULT_REMIX_WEIGHT = 3.0`
- `DEFAULT_VIEW_WEIGHT = 0.1`
- `MIN_AGE_HOURS = 0.1`
- `HOURS_PER_DAY = 24`
- `DEFAULT_TRENDING_HOURS = 24`
- `SCORE_DECIMAL_PLACES = 2`

### Pagination
- `DEFAULT_PAGE = 1`
- `DEFAULT_PAGE_SIZE = 20`
- `MAX_PAGE = 1000`
- `MAX_PAGE_SIZE = 100`
- `MIN_PAGE = 1`
- `MIN_PAGE_SIZE = 1`

### Tags
- `MAX_TAGS_PER_CHAT = 10`

### Time
- `SECONDS_PER_HOUR = 3600`

### Validation
- `MIN_CHAT_AGE_HOURS = 0.1`

## Verification

- ✅ No linter errors
- ✅ All imports resolve correctly
- ✅ Constants are well-documented
- ✅ Backward compatible (no breaking changes)
- ✅ Consistent usage across codebase

## Future Improvements

Consider extracting more constants from:
- API route defaults
- Cache TTL values
- Rate limiting values
- Validation limits
- Error message templates



