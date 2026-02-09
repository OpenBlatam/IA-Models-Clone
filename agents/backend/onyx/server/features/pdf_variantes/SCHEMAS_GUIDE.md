# Schemas Guide - PDF Variantes

## ✅ Recommended Schemas

### `models.py` - **USE THIS**

The canonical file containing all Pydantic models and schemas:

```python
from models import (
    PDFUploadRequest,
    PDFUploadResponse,
    PDFDocument,
    VariantGenerateRequest,
    VariantGenerateResponse,
    PDFVariant,
    TopicExtractRequest,
    TopicExtractResponse,
    TopicItem,
    BrainstormGenerateRequest,
    BrainstormGenerateResponse,
    BrainstormIdea,
    # ... and many more
)
```

**Features:**
- Comprehensive Pydantic models
- Request/Response models
- Domain models (PDFDocument, PDFVariant, etc.)
- Enums (VariantStatus, PDFProcessingStatus, TopicCategory)
- Validation and serialization
- Ultra-optimized performance edition

## 📦 Available Models

### Request Models
- `PDFUploadRequest` - PDF upload request
- `PDFEditRequest` - PDF edit request
- `VariantGenerateRequest` - Variant generation request
- `TopicExtractRequest` - Topic extraction request
- `BrainstormGenerateRequest` - Brainstorming request
- `VariantStopRequest` - Stop variant generation
- `PDFDownloadRequest` - PDF download request

### Response Models
- `PDFUploadResponse` - PDF upload response
- `PDFEditResponse` - PDF edit response
- `VariantGenerateResponse` - Variant generation response
- `TopicExtractResponse` - Topic extraction response
- `BrainstormGenerateResponse` - Brainstorming response
- `VariantStopResponse` - Stop variant response
- `PDFDownloadResponse` - PDF download response
- `VariantListResponse` - List variants response

### Domain Models
- `PDFDocument` - PDF document entity
- `PDFVariant` - PDF variant entity
- `PDFMetadata` - PDF metadata
- `TopicItem` - Topic item
- `BrainstormIdea` - Brainstorm idea
- `EditedPage` - Edited page
- `VariantConfiguration` - Variant configuration

### Enums
- `VariantStatus` - Variant generation status
- `PDFProcessingStatus` - PDF processing status
- `TopicCategory` - Topic category
- `AnnotationType` - Annotation type
- `VariantType` - Variant type

### Other Models
- `DocumentStats` - Document statistics
- `ProcessingMetrics` - Processing metrics
- `QualityMetrics` - Quality metrics
- `VariantBatch` - Batch of variants
- `OptimizationSettings` - Optimization settings

## ⚠️ Deprecated Schema Files

The following files are **deprecated** and should not be used for new code:

### `schemas.py` (Root)
- **Status**: Deprecated
- **Reason**: Duplicate of `models.py`
- **Migration**: Use `models.py` instead

### `enhanced_schemas.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `models.py` instead
- **Migration**: Use `models.py` instead

### `optimized_schemas.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `models.py` instead
- **Migration**: Use `models.py` instead

### `ultra_schemas.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `models.py` instead
- **Migration**: Use `models.py` instead

## 🏗️ Schema Structure

```
pdf_variantes/
├── models.py              # ✅ Canonical models file (all Pydantic models)
├── schemas.py             # ⚠️ Deprecated
├── enhanced_schemas.py    # ⚠️ Deprecated
├── optimized_schemas.py   # ⚠️ Deprecated
└── ultra_schemas.py       # ⚠️ Deprecated
```

## 📝 Usage Examples

### Basic Request/Response

```python
from models import PDFUploadRequest, PDFUploadResponse

# Create request
request = PDFUploadRequest(
    filename="document.pdf",
    auto_process=True,
    extract_text=True
)

# Response will be PDFUploadResponse
```

### Variant Generation

```python
from models import VariantGenerateRequest, VariantGenerateResponse, VariantType

request = VariantGenerateRequest(
    document_id="doc123",
    variant_types=[VariantType.SUMMARY, VariantType.OUTLINE],
    count=5
)
```

### Topic Extraction

```python
from models import TopicExtractRequest, TopicExtractResponse

request = TopicExtractRequest(
    document_id="doc123",
    max_topics=10
)
```

### Using Domain Models

```python
from models import PDFDocument, PDFVariant, PDFMetadata

document = PDFDocument(
    id="doc123",
    filename="document.pdf",
    metadata=PDFMetadata(
        title="My Document",
        author="John Doe",
        page_count=10
    )
)
```

## 🔄 Migration Guide

### From `schemas.py`
```python
# Old
from schemas import PDFUploadSchema, VariantGenerateSchema

# New
from models import PDFUploadRequest, VariantGenerateRequest
```

### From `enhanced_schemas.py`
```python
# Old
from enhanced_schemas import PDFUploadRequest, VariantGenerateRequest

# New
from models import PDFUploadRequest, VariantGenerateRequest
```

### From `optimized_schemas.py`
```python
# Old
from optimized_schemas import OptimizedPDFUploadRequest, OptimizedVariantRequest

# New
from models import PDFUploadRequest, VariantGenerateRequest
```

### From `ultra_schemas.py`
```python
# Old
from ultra_schemas import UltraFastPDFUploadRequest, UltraFastVariantRequest

# New
from models import PDFUploadRequest, VariantGenerateRequest
```

## 📚 Additional Resources

- See `models.py` for the complete list of all models
- See `REFACTORING_STATUS.md` for refactoring progress
- See API documentation for model usage in endpoints






