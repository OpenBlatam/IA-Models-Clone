# Schemas Improvement

## Overview

Improved request schemas by replacing hardcoded validation limits with constants from the centralized `constants.py` module.

## Changes Made

### 1. Added Schema Validation Constants
- **Added to `constants.py`**:
  - `MAX_TITLE_LENGTH = 200`
  - `MAX_DESCRIPTION_LENGTH = 1000`
  - `MAX_CHAT_CONTENT_LENGTH = 50000`
  - `MAX_TAG_LENGTH = 50`
  - `MIN_TITLE_LENGTH = 1`
  - `MIN_CHAT_CONTENT_LENGTH = 1`
- **Benefits**:
  - Centralized validation limits
  - Easy to update limits in one place
  - Consistent validation across schemas

### 2. Updated `PublishChatRequest` Schema
- **Replaced hardcoded values** with constants:
  - `max_length=200` → `max_length=MAX_TITLE_LENGTH`
  - `max_length=1000` → `max_length=MAX_DESCRIPTION_LENGTH`
  - `max_length=50000` → `max_length=MAX_CHAT_CONTENT_LENGTH`
  - `max_length=10` → `max_length=MAX_TAGS_PER_CHAT`
  - `[:50]` → `[:MAX_TAG_LENGTH]`
- **Updated error messages** to use constants:
  - `"Description cannot exceed 1000 characters"` → `f"Description cannot exceed {MAX_DESCRIPTION_LENGTH} characters"`
  - `"Chat content cannot exceed 50000 characters"` → `f"Chat content cannot exceed {MAX_CHAT_CONTENT_LENGTH} characters"`
  - `"Maximum 10 tags allowed"` → `f"Maximum {MAX_TAGS_PER_CHAT} tags allowed"`

## Before vs After

### Before - constants.py
```python
# Validation Constants
MIN_CHAT_AGE_HOURS = 0.1
```

### After - constants.py
```python
# Validation Constants
MIN_CHAT_AGE_HOURS = 0.1

# Schema Validation Constants
MAX_TITLE_LENGTH = 200
MAX_DESCRIPTION_LENGTH = 1000
MAX_CHAT_CONTENT_LENGTH = 50000
MAX_TAG_LENGTH = 50
MIN_TITLE_LENGTH = 1
MIN_CHAT_CONTENT_LENGTH = 1
```

### Before - requests.py
```python
title: str = Field(
    ...,
    min_length=1,
    max_length=200,
    description="Título del chat",
    examples=["Mi increíble chat sobre IA"]
)
description: Optional[str] = Field(
    None,
    max_length=1000,
    description="Descripción opcional del chat",
    examples=["Un chat sobre inteligencia artificial y machine learning"]
)
chat_content: str = Field(
    ...,
    min_length=1,
    max_length=50000,
    description="Contenido del chat (JSON o texto)",
    examples=["{\"messages\": [...]}"]
)
tags: Optional[List[str]] = Field(
    None,
    max_length=10,
    description="Tags del chat (máximo 10)",
    examples=[["ai", "machine-learning", "chat"]]
)

@field_validator('description')
def validate_description(cls, v: Optional[str]) -> Optional[str]:
    if len(v) > 1000:
        raise ValueError("Description cannot exceed 1000 characters")

@field_validator('chat_content')
def validate_chat_content(cls, v: str) -> str:
    if len(v) > 50000:
        raise ValueError("Chat content cannot exceed 50000 characters")

@field_validator('tags')
def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
    if len(v) > 10:
        raise ValueError("Maximum 10 tags allowed")
    tag_clean = tag.strip().lower()[:50]
```

### After - requests.py
```python
from ..constants import (
    MAX_TITLE_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_CHAT_CONTENT_LENGTH,
    MAX_TAG_LENGTH,
    MIN_TITLE_LENGTH,
    MIN_CHAT_CONTENT_LENGTH,
    MAX_TAGS_PER_CHAT,
)

title: str = Field(
    ...,
    min_length=MIN_TITLE_LENGTH,
    max_length=MAX_TITLE_LENGTH,
    description="Título del chat",
    examples=["Mi increíble chat sobre IA"]
)
description: Optional[str] = Field(
    None,
    max_length=MAX_DESCRIPTION_LENGTH,
    description="Descripción opcional del chat",
    examples=["Un chat sobre inteligencia artificial y machine learning"]
)
chat_content: str = Field(
    ...,
    min_length=MIN_CHAT_CONTENT_LENGTH,
    max_length=MAX_CHAT_CONTENT_LENGTH,
    description="Contenido del chat (JSON o texto)",
    examples=["{\"messages\": [...]}"]
)
tags: Optional[List[str]] = Field(
    None,
    max_length=MAX_TAGS_PER_CHAT,
    description=f"Tags del chat (máximo {MAX_TAGS_PER_CHAT})",
    examples=[["ai", "machine-learning", "chat"]]
)

@field_validator('description')
def validate_description(cls, v: Optional[str]) -> Optional[str]:
    if len(v) > MAX_DESCRIPTION_LENGTH:
        raise ValueError(f"Description cannot exceed {MAX_DESCRIPTION_LENGTH} characters")

@field_validator('chat_content')
def validate_chat_content(cls, v: str) -> str:
    if len(v) > MAX_CHAT_CONTENT_LENGTH:
        raise ValueError(f"Chat content cannot exceed {MAX_CHAT_CONTENT_LENGTH} characters")

@field_validator('tags')
def validate_tags(cls, v: Optional[List[str]]) -> Optional[List[str]]:
    if len(v) > MAX_TAGS_PER_CHAT:
        raise ValueError(f"Maximum {MAX_TAGS_PER_CHAT} tags allowed")
    tag_clean = tag.strip().lower()[:MAX_TAG_LENGTH]
```

## Files Modified

1. **`constants.py`**
   - Added schema validation constants
   - Updated `__all__` exports

2. **`schemas/requests.py`**
   - Replaced hardcoded validation limits with constants
   - Updated error messages to use constants
   - Improved description field to use constant in f-string

## Benefits

1. **Centralized Configuration**: All validation limits in one place
2. **Easy Updates**: Change limits in one place, affects all schemas
3. **Consistency**: Same limits used across all validation
4. **Better Maintainability**: Clear where limits are defined
5. **Dynamic Error Messages**: Error messages reflect actual limits
6. **Type Safety**: Constants can be type-checked

## Improvements Details

### Constants Added
- `MAX_TITLE_LENGTH = 200`
- `MAX_DESCRIPTION_LENGTH = 1000`
- `MAX_CHAT_CONTENT_LENGTH = 50000`
- `MAX_TAG_LENGTH = 50`
- `MIN_TITLE_LENGTH = 1`
- `MIN_CHAT_CONTENT_LENGTH = 1`

### Schema Updates
- All `Field()` definitions now use constants
- All validators now use constants
- Error messages use f-strings with constants

## Verification

- ✅ No linter errors
- ✅ All imports resolve correctly
- ✅ Constants are properly exported
- ✅ Validation limits are consistent
- ✅ Error messages are accurate
- ✅ Backward compatible

## Testing Recommendations

1. Test validation with values at limits
2. Test validation with values exceeding limits
3. Verify error messages show correct limits
4. Test that changing constants affects validation
5. Verify all schemas use constants consistently



