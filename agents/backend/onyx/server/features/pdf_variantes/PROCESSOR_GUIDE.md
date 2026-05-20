# Processor Guide - PDF Variantes

## ✅ Recommended Processors

### `services/pdf_service.py` - **USE THIS**

The canonical PDF processing service:

```python
from services.pdf_service import PDFVariantesService
from utils.config import get_settings

settings = get_settings()
service = PDFVariantesService(settings)
await service.initialize()

# Use the service
result = await service.upload_pdf(file, request, user_id)
```

**Features:**
- Complete PDF processing service
- AI capabilities integration
- Caching support
- File management
- Variant generation
- Topic extraction
- Brainstorming

### `utils/file_helpers.py` - **PDFProcessor Class**

Lower-level PDF processing utilities:

```python
from utils.file_helpers import PDFProcessor
from utils.config import get_settings

settings = get_settings()
processor = PDFProcessor(settings)

# Use the processor
result = await processor.process_pdf(file_content)
```

**Note**: This is a utility class used by `PDFVariantesService`. For most use cases, use the service instead.

## ⚠️ Deprecated Processor Files

The following files are **deprecated** and should not be used for new code:

### `pdf_processor.py` (Root)
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `services/pdf_service.py` instead
- **Migration**: Use `services.pdf_service.PDFVariantesService`

### `advanced_pdf_processor.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `services/pdf_service.py` instead
- **Migration**: Use `services.pdf_service.PDFVariantesService`

### `enhanced_pdf_processor.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `services/pdf_service.py` instead
- **Migration**: Use `services.pdf_service.PDFVariantesService`

### `optimized_processor.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `services/pdf_service.py` instead
- **Migration**: Use `services.pdf_service.PDFVariantesService`

### `ultra_pdf_processor.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `services/pdf_service.py` instead
- **Migration**: Use `services.pdf_service.PDFVariantesService`

## 🏗️ Processor Structure

```
pdf_variantes/
├── services/
│   └── pdf_service.py          # ✅ Canonical PDF processing service
├── utils/
│   └── file_helpers.py          # ✅ PDFProcessor utility class
├── pdf_processor.py             # ⚠️ Deprecated
├── advanced_pdf_processor.py    # ⚠️ Deprecated
├── enhanced_pdf_processor.py   # ⚠️ Deprecated
├── optimized_processor.py       # ⚠️ Deprecated
└── ultra_pdf_processor.py       # ⚠️ Deprecated
```

## 📝 Usage Examples

### Basic PDF Upload and Processing

```python
from services.pdf_service import PDFVariantesService
from utils.config import get_settings
from fastapi import UploadFile

settings = get_settings()
service = PDFVariantesService(settings)
await service.initialize()

# Upload PDF
upload_request = PDFUploadRequest(
    filename="document.pdf",
    extract_text=True,
    extract_images=True
)

result = await service.upload_pdf(
    file=upload_file,
    request=upload_request,
    user_id="user123"
)
```

### Generate Variants

```python
variant_request = VariantGenerateRequest(
    document_id=result.document.id,
    variant_types=[VariantType.SUMMARY, VariantType.OUTLINE],
    count=5
)

variants = await service.generate_variants(
    request=variant_request,
    user_id="user123"
)
```

### Extract Topics

```python
topic_request = TopicExtractRequest(
    document_id=result.document.id,
    max_topics=10
)

topics = await service.extract_topics(
    request=topic_request,
    user_id="user123"
)
```

### Brainstorming

```python
brainstorm_request = BrainstormGenerateRequest(
    document_id=result.document.id,
    max_ideas=20
)

ideas = await service.generate_brainstorm(
    request=brainstorm_request,
    user_id="user123"
)
```

## 🔄 Migration Guide

### From `pdf_processor.py`
```python
# Old
from pdf_processor import process_pdf_complete
result = await process_pdf_complete(file_content, filename)

# New
from services.pdf_service import PDFVariantesService
service = PDFVariantesService(settings)
await service.initialize()
result = await service.upload_pdf(file, request, user_id)
```

### From `advanced_pdf_processor.py`
```python
# Old
from advanced_pdf_processor import advanced_process_pdf
result = await advanced_process_pdf(file_content)

# New
from services.pdf_service import PDFVariantesService
service = PDFVariantesService(settings)
await service.initialize()
result = await service.upload_pdf(file, request, user_id)
```

### From `enhanced_pdf_processor.py`, `optimized_processor.py`, or `ultra_pdf_processor.py`
```python
# Old
from enhanced_pdf_processor import process_pdf
# or
from optimized_processor import process_pdf
# or
from ultra_pdf_processor import ultra_fast_process_pdf

# New
from services.pdf_service import PDFVariantesService
service = PDFVariantesService(settings)
await service.initialize()
result = await service.upload_pdf(file, request, user_id)
```

## 📚 Additional Resources

- See `services/pdf_service.py` for the full service implementation
- See `utils/file_helpers.py` for PDFProcessor utility class
- See `REFACTORING_STATUS.md` for refactoring progress






