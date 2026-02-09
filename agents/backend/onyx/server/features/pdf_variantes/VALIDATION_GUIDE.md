# Validation Guide - PDF Variantes

## ✅ Recommended Validation

### `utils/validators.py` - **USE THIS**

The canonical validation file for general-purpose validations:

```python
from utils.validators import (
    validate_file_upload,
    validate_content_type,
    ValidationResult,
)
```

### `api/validators.py` - **API-Specific Validators**

API-specific validators for FastAPI requests:

```python
from api.validators import (
    FileValidator,
    PaginationValidator,
    RequestValidator,
)
```

### `utils/validation.py` - **Basic Validators**

Basic validation functions:

```python
from utils.validation import (
    Validator,
    validate_filename,
    validate_file_extension,
    validate_integer_range,
    validate_string_length,
    validate_email,
    validate_uuid,
)
```

**Features:**
- General-purpose validation utilities
- API-specific validators
- File upload validation
- Content type validation
- Request validation

## 📦 Available Validators

### General Validators (from `utils/validators.py`)

#### `validate_file_upload`
- File upload validation
- Size checking
- Type checking
- Filename validation

#### `validate_content_type`
- Content type validation
- MIME type checking
- Allowed types validation

#### `ValidationResult`
- Validation result dataclass
- Error and warning tracking
- Result formatting

### API Validators (from `api/validators.py`)

#### `FileValidator`
- File upload validation for FastAPI
- Content type checking
- Size validation
- Filename sanitization

#### `PaginationValidator`
- Pagination parameter validation
- Page number validation
- Page size validation

#### `RequestValidator`
- Request validation
- Header validation
- Parameter validation

### Basic Validators (from `utils/validation.py`)

#### `Validator`
- Base validator class
- Custom validation rules
- Error handling

#### `validate_filename`
- Filename validation
- Invalid character checking
- Path traversal prevention

#### `validate_file_extension`
- File extension validation
- Allowed extensions checking

#### `validate_integer_range`
- Integer range validation
- Min/max checking

#### `validate_string_length`
- String length validation
- Min/max length checking

#### `validate_email`
- Email format validation
- Email pattern matching

#### `validate_uuid`
- UUID format validation
- UUID pattern matching

## ⚠️ Deprecated Validation Files

The following validation files are **deprecated** and should not be used for new code:

### `api/enhanced_validation.py`
- **Status**: Deprecated
- **Reason**: Functionality moved to `api/validators.py` and `utils/validators.py`
- **Migration**: Use `api.validators.FileValidator` or `utils.validators.validate_file_upload`

### `utils/validation_utils.py`
- **Status**: ⚠️ Consider consolidating
- **Reason**: Similar functionality to `utils/validation.py` and `utils/validators.py`
- **Note**: Contains `ValidationResult`, `InputValidator`, etc. - may be useful but consider consolidation

## 🏗️ Validation Structure

```
pdf_variantes/
├── api/
│   ├── validators.py              # ✅ API-specific validators
│   └── enhanced_validation.py    # ⚠️ Deprecated
└── utils/
    ├── validators.py              # ✅ Canonical general validators
    ├── validation.py              # ✅ Basic validators
    └── validation_utils.py        # ⚠️ Consider consolidating
```

## 📝 Usage Examples

### File Upload Validation

```python
from utils.validators import validate_file_upload
from fastapi import UploadFile

# Validate file upload
result = validate_file_upload(
    file=upload_file,
    max_size_mb=100,
    allowed_types=["pdf"]
)

if not result.is_valid:
    raise ValidationError(detail="; ".join(result.errors))
```

### Using API Validators

```python
from api.validators import FileValidator
from fastapi import UploadFile

# Validate file using API validator
FileValidator.validate_upload_file(upload_file)
```

### Basic Validation

```python
from utils.validation import validate_filename, validate_email

# Validate filename
is_valid, error = validate_filename("document.pdf")
if not is_valid:
    raise ValidationError(detail=error)

# Validate email
is_valid, error = validate_email("user@example.com")
if not is_valid:
    raise ValidationError(detail=error)
```

### Custom Validation

```python
from utils.validation import Validator

class CustomValidator(Validator):
    def validate(self, value):
        if not value:
            raise ValueError("Value is required")
        return value

validator = CustomValidator()
result = validator.validate(data)
```

## 🔄 Migration Guide

### From `api/enhanced_validation.py`

```python
# Old
from api.enhanced_validation import validate_document_access

# New
from api.validators import FileValidator
from utils.validators import validate_file_upload
```

### From `utils/validation_utils.py`

```python
# Old
from utils.validation_utils import InputValidator, ValidationResult

validator = InputValidator()
result = validator.validate_email(email)

# New
from utils.validation import validate_email
from utils.validators import ValidationResult

is_valid, error = validate_email(email)
result = ValidationResult(is_valid=is_valid, errors=[error] if error else [])
```

## 📚 Additional Resources

- See `utils/validators.py` for general validators
- See `api/validators.py` for API-specific validators
- See `utils/validation.py` for basic validators
- See `api/exceptions.py` for validation exceptions
- See `REFACTORING_STATUS.md` for refactoring progress






