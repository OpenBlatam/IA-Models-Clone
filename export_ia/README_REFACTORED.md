# Export IA - Refactored AI Document Processing System

## 🚀 Overview

Export IA is a state-of-the-art AI-powered document processing system that has been completely refactored with a modular, professional architecture. The system provides high-quality document export capabilities across multiple formats with advanced quality assurance and async processing.

## ✨ Key Improvements in Refactored Version

### 🏗️ Modular Architecture
- **Separation of Concerns**: Clear separation between core engine, exporters, quality management, and API layers
- **Plugin System**: Extensible exporter system with factory pattern
- **Configuration Management**: Centralized YAML-based configuration
- **Async Processing**: Full async/await support with proper task management

### 🎯 Enhanced Features
- **Quality Assurance**: Comprehensive quality validation and scoring system
- **Task Management**: Advanced async task processing with progress tracking
- **API Layer**: Clean REST API with FastAPI and proper error handling
- **Multiple Formats**: Support for PDF, DOCX, HTML, Markdown, RTF, TXT, JSON, XML
- **Professional Styling**: Quality-based styling with 5 different quality levels

## 🏗️ New Architecture

```
src/
├── __init__.py                 # Main package exports
├── core/                       # Core system components
│   ├── __init__.py
│   ├── engine.py              # Main ExportIAEngine
│   ├── models.py              # Data models and enums
│   ├── config.py              # Configuration management
│   ├── task_manager.py        # Async task management
│   └── quality_manager.py     # Quality assurance system
├── exporters/                  # Export format handlers
│   ├── __init__.py
│   ├── base.py                # Base exporter class
│   ├── factory.py             # Exporter factory
│   ├── pdf_exporter.py        # PDF export handler
│   ├── docx_exporter.py       # DOCX export handler
│   ├── html_exporter.py       # HTML export handler
│   ├── markdown_exporter.py   # Markdown export handler
│   ├── rtf_exporter.py        # RTF export handler
│   ├── txt_exporter.py        # Plain text export handler
│   ├── json_exporter.py       # JSON export handler
│   └── xml_exporter.py        # XML export handler
└── api/                       # API layer
    ├── __init__.py
    ├── models.py              # API request/response models
    └── fastapi_app.py         # FastAPI application
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements_refactored_v2.txt

# Optional: Install with AI features
pip install -r requirements_advanced.txt
```

### Basic Usage

```python
import asyncio
from src.core.engine import ExportIAEngine
from src.core.models import ExportConfig, ExportFormat, DocumentType, QualityLevel

async def main():
    # Sample content
    content = {
        "title": "My Document",
        "sections": [
            {
                "heading": "Introduction",
                "content": "This is a sample document for export."
            }
        ]
    }
    
    # Create export configuration
    config = ExportConfig(
        format=ExportFormat.PDF,
        document_type=DocumentType.REPORT,
        quality_level=QualityLevel.PROFESSIONAL
    )
    
    # Export document
    async with ExportIAEngine() as engine:
        task_id = await engine.export_document(content, config)
        
        # Wait for completion
        while True:
            status = await engine.get_task_status(task_id)
            if status["status"] == "completed":
                print(f"Export completed: {status['file_path']}")
                break
            await asyncio.sleep(1)

asyncio.run(main())
```

### API Usage

```bash
# Start the API server
uvicorn src.api.fastapi_app:create_app --factory --host 0.0.0.0 --port 8000

# Export a document via API
curl -X POST "http://localhost:8000/export" \
  -H "Content-Type: application/json" \
  -d '{
    "content": {
      "title": "API Document",
      "sections": [{"heading": "Test", "content": "API test"}]
    },
    "format": "pdf",
    "document_type": "report",
    "quality_level": "professional"
  }'
```

## 🔧 Configuration

The system uses a comprehensive YAML configuration file (`config/export_config.yaml`) that includes:

- **System Settings**: Output directories, task limits, timeouts
- **Quality Levels**: 5 different quality levels with specific styling
- **Document Templates**: Predefined templates for different document types
- **Format Features**: Feature lists for each export format

### Quality Levels

1. **Basic**: Simple formatting, basic fonts
2. **Standard**: Improved typography, better colors
3. **Professional**: Headers/footers, page numbers, table styling
4. **Premium**: Custom branding, advanced formatting
5. **Enterprise**: Interactive elements, accessibility features

## 📊 API Endpoints

### Core Endpoints

- `POST /export` - Create export task
- `GET /export/{task_id}/status` - Get task status
- `GET /export/{task_id}/download` - Download exported file
- `DELETE /export/{task_id}` - Cancel task

### Information Endpoints

- `GET /formats` - List supported formats
- `GET /statistics` - System statistics
- `GET /templates/{type}` - Get document template
- `POST /validate` - Validate content quality

## 🎨 Export Formats

### Supported Formats

| Format | Features | Use Case |
|--------|----------|----------|
| PDF | High quality, print ready, vector graphics | Professional documents, reports |
| DOCX | Editable, professional formatting, tables | Business documents, proposals |
| HTML | Web ready, responsive, interactive | Web content, online reports |
| Markdown | Version control friendly, lightweight | Documentation, README files |
| RTF | Cross platform, rich formatting | Legacy compatibility |
| TXT | Universal compatibility, fast | Simple text documents |
| JSON | Structured data, API friendly | Data exchange, APIs |
| XML | Structured data, validation | Enterprise systems |

## 🧪 Examples

### Basic Export
```python
# See examples/basic_usage.py for comprehensive examples
```

### API Client
```python
# See examples/api_usage.py for API usage examples
```

### Batch Processing
```python
# Export multiple documents in parallel
tasks = []
for content in documents:
    task_id = await engine.export_document(content, config)
    tasks.append(task_id)

# Wait for all to complete
for task_id in tasks:
    status = await engine.get_task_status(task_id)
    # Handle completion
```

## 🔒 Quality Assurance

The refactored system includes comprehensive quality assurance:

- **Content Validation**: Structure, completeness, professional language
- **Formatting Validation**: Typography, spacing, alignment
- **Accessibility Validation**: Alt text, heading structure, color contrast
- **Professional Standards**: Branding consistency, error-free content

### Quality Scoring

Each export receives a quality score (0.0 to 1.0) based on:
- Base export success (30%)
- Quality level bonus (10-50%)
- Format-specific features (15-20%)
- Professional features (10-30%)

## 🚀 Performance Features

### Async Processing
- Non-blocking task processing
- Concurrent export handling
- Progress tracking and status updates
- Automatic cleanup and resource management

### Task Management
- Queue-based processing
- Task prioritization
- Timeout handling
- Error recovery and retry logic

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

### Adding New Exporters
```python
from src.exporters.base import BaseExporter

class CustomExporter(BaseExporter):
    async def export(self, content, config, output_path):
        # Implementation
        pass
    
    def get_supported_features(self):
        return ["feature1", "feature2"]

# Register the exporter
from src.exporters.factory import ExporterFactory
from src.core.models import ExportFormat

ExporterFactory.register_exporter(ExportFormat.CUSTOM, CustomExporter)
```

## 📈 Migration from Legacy Version

### Key Changes
1. **Async Interface**: All operations are now async
2. **Task-based**: Export operations return task IDs for tracking
3. **Modular Structure**: Components are separated into focused modules
4. **Configuration**: YAML-based configuration instead of hardcoded values
5. **API Layer**: New REST API for external integration

### Migration Guide
```python
# Old way (legacy)
engine = ExportIAEngine()
result = engine.export_document(content, config)

# New way (refactored)
async with ExportIAEngine() as engine:
    task_id = await engine.export_document(content, config)
    status = await engine.get_task_status(task_id)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- FastAPI team for the modern web framework
- ReportLab team for PDF generation capabilities
- The open-source community for inspiration and contributions

---

**Export IA v2.0** - Refactored for the future 🚀




