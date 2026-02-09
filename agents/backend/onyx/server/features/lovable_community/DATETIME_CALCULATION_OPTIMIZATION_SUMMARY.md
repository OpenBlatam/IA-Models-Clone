# DateTime Calculation Optimization Summary

## Overview

Optimized datetime calculations by creating helper functions that encapsulate repetitive date/time calculation patterns. This improves code consistency, reduces duplication, and ensures uniform datetime handling across the codebase.

## Helper Functions Created

### 1. `calculate_age_hours(created_at, reference_time=None)`

**Purpose**: Calculate age in hours from a datetime to a reference time (or now).

**Benefits**:
- Eliminates duplicate age calculation code
- Consistent age calculation logic
- Supports custom reference times for testing

**Example Usage**:
```python
# Before
now = datetime.utcnow()
age_hours = (now - created_at).total_seconds() / SECONDS_PER_HOUR

# After
age_hours = calculate_age_hours(created_at)
age_hours = calculate_age_hours(created_at, reference_time=now)
```

### 2. `calculate_cutoff_time(hours, reference_time=None)`

**Purpose**: Calculate cutoff time by subtracting hours from reference time (or now).

**Benefits**:
- Eliminates duplicate cutoff time calculation code
- Consistent time window calculations
- Used in trending and time-based queries

**Example Usage**:
```python
# Before
from datetime import timedelta
from ..helpers import get_current_timestamp
cutoff_time = get_current_timestamp() - timedelta(hours=hours)

# After
from ..helpers.datetime_helpers import calculate_cutoff_time
cutoff_time = calculate_cutoff_time(hours)
```

### 3. `calculate_cutoff_time_days(days, reference_time=None)`

**Purpose**: Calculate cutoff time by subtracting days from reference time (or now).

**Benefits**:
- Eliminates duplicate day-based cutoff calculations
- Consistent date-based query calculations
- Used in analytics and reporting

**Example Usage**:
```python
# Before
cutoff_time = datetime.utcnow() - timedelta(days=7)

# After
cutoff_time = calculate_cutoff_time_days(7)
```

### 4. `format_datetime_iso(dt=None)`

**Purpose**: Format datetime to ISO format string.

**Benefits**:
- Eliminates duplicate ISO formatting code
- Consistent timestamp formatting in API responses
- Defaults to current time if not provided

**Example Usage**:
```python
# Before
"timestamp": datetime.utcnow().isoformat()

# After
"timestamp": format_datetime_iso()
```

### 5. `is_within_time_window(created_at, hours, reference_time=None)`

**Purpose**: Check if a datetime is within a time window (hours from reference time).

**Benefits**:
- Eliminates duplicate time window checking code
- Consistent trending window validation
- Cleaner conditional logic

**Example Usage**:
```python
# Before
now = datetime.utcnow()
age_hours = (now - created_at).total_seconds() / SECONDS_PER_HOUR
if age_hours > hours:
    return 0.0

# After
if not is_within_time_window(created_at, hours):
    return 0.0
```

## Files Modified

### 1. Created: `helpers/datetime_helpers.py`
- New module containing all datetime calculation helper functions
- Well-documented with docstrings and examples
- Type hints for better IDE support

### 2. Refactored: `services/ranking.py`
- **Methods optimized**: 2 methods
- **Lines reduced**: ~5 lines of duplicate code eliminated
- **Improvements**:
  - `calculate_score()`: Uses `calculate_age_hours()`
  - `calculate_trending_score()`: Uses `is_within_time_window()`

### 3. Refactored: `repositories/chat_repository.py`
- **Methods optimized**: 2 methods
- **Lines reduced**: ~4 lines of duplicate code eliminated
- **Improvements**:
  - `get_trending()`: Uses `calculate_cutoff_time()`
  - `_build_search_query()`: Uses `build_search_term()` (from previous optimization)

### 4. Refactored: `repositories/view_repository.py`
- **Methods optimized**: 1 method
- **Lines reduced**: ~2 lines of duplicate code eliminated
- **Improvements**:
  - `get_recent_views()`: Uses `calculate_cutoff_time()`

## Code Quality Improvements

### Before Optimization
- **Repetitive datetime calculations**: Same calculation patterns repeated multiple times
- **Inconsistent time calculations**: Slight variations in calculation logic
- **Duplicate imports**: `from datetime import timedelta` repeated
- **Harder to maintain**: Changes to calculation logic require updates in multiple places

### After Optimization
- **DRY Principle**: Datetime calculations centralized in helper functions
- **Consistent calculations**: All datetime operations follow same pattern
- **Centralized imports**: Datetime helpers imported from single module
- **Easier maintenance**: Changes to calculation logic only need to be made in one place
- **Better testability**: Helper functions can be tested independently
- **Improved readability**: Code is more focused on business logic

## Statistics

- **Total methods optimized**: 5 methods across 3 modules
- **Lines of code reduced**: ~11 lines of duplicate code eliminated
- **Helper functions created**: 5 datetime calculation helpers
- **Code maintainability**: Significantly improved (single source of truth for datetime calculations)

## Before vs After Examples

### Example 1: Age Calculation

**Before**:
```python
now = datetime.utcnow()
age_hours = (now - created_at).total_seconds() / SECONDS_PER_HOUR
if age_hours < MIN_AGE_HOURS:
    age_hours = MIN_AGE_HOURS
```

**After**:
```python
age_hours = calculate_age_hours(created_at)
if age_hours < MIN_AGE_HOURS:
    age_hours = MIN_AGE_HOURS
```

### Example 2: Cutoff Time Calculation

**Before**:
```python
from datetime import timedelta
from ..helpers import get_current_timestamp
cutoff_time = get_current_timestamp() - timedelta(hours=hours)
```

**After**:
```python
from ..helpers.datetime_helpers import calculate_cutoff_time
cutoff_time = calculate_cutoff_time(hours)
```

### Example 3: Time Window Check

**Before**:
```python
now = datetime.utcnow()
age_hours = (now - created_at).total_seconds() / SECONDS_PER_HOUR
if age_hours > hours:
    return 0.0
```

**After**:
```python
if not is_within_time_window(created_at, hours):
    return 0.0
```

## Testing Recommendations

1. **Unit tests for helper functions**: Test all datetime helpers with various inputs
2. **Integration tests**: Verify that refactored code maintains original behavior
3. **Edge case tests**: Test with None values, negative values, future dates, etc.
4. **Time zone tests**: Ensure UTC consistency across all calculations

## Future Improvements

1. Consider adding time zone support if needed
2. Could extend helpers to support more time units (minutes, weeks, months)
3. Could add helpers for date range calculations
4. Could create helpers for business day calculations if needed

## Conclusion

The optimization successfully reduces code duplication, improves maintainability, and ensures consistent datetime calculations across the codebase. The helper functions are well-documented, type-hinted, and follow Python best practices. Code is now cleaner and more focused on business logic rather than datetime calculation boilerplate.

