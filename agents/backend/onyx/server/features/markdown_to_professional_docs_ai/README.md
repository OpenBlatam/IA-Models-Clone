# Markdown to Professional Documents AI

Convert Markdown files to professional document formats with charts, diagrams, tables, and beautiful formatting.

## 🚀 Features

- ✅ **Multiple Format Support**: Convert to Excel, PDF, Word, Tableau, Power BI, HTML, PowerPoint, and more
- 📊 **Automatic Charts**: Generate beautiful charts and graphs from tables (bar, line, pie, scatter, area, histogram)
- 📋 **Table Conversion**: Preserve and format tables in all output formats
- 🎨 **Professional Styling**: Beautiful, professional formatting for all documents
- 📁 **Batch Processing**: Convert multiple files or entire repositories
- 🔄 **Repository Support**: Convert all Markdown files in a repository at once
- ⚡ **Smart Caching**: Automatic caching of conversions for improved performance
- 🛡️ **Robust Validation**: Comprehensive input validation with helpful error messages
- 🔍 **Auto Chart Detection**: Automatically detects the best chart type for your data
- 📈 **Enhanced Markdown Parsing**: Support for blockquotes, emphasis, horizontal rules, and more
- 🎨 **Mermaid Diagrams**: Render Mermaid diagrams to images (flowcharts, sequence diagrams, etc.)
- 📊 **Metrics & Monitoring**: Comprehensive metrics collection and monitoring
- 🚦 **Rate Limiting**: Built-in rate limiting to protect the service
- 🖼️ **Image Processing**: Download, optimize, and embed images from URLs
- 🎨 **Custom Templates**: Professional, Modern, and Classic templates with full customization
- 📐 **Math Formulas**: Render LaTeX/MathJax formulas to images
- 🌍 **Multi-language**: Support for 10 languages with auto-detection
- 🔔 **Webhooks**: Async notifications for conversion events
- 🔍 **Document Comparison**: Compare documents and detect differences
- 🌐 **Translation**: Automatic translation support for multiple languages
- ✍️ **Digital Signatures**: Sign and verify documents with digital signatures
- 📦 **Multi-format Export**: Export to multiple formats simultaneously
- 📊 **Interactive Dashboard**: Web-based dashboard with real-time metrics and visualizations
- 🎨 **Advanced Templates**: Dynamic template system with custom styling and layouts
- 🔐 **Permissions & Roles**: Granular access control with role-based permissions
- 📅 **Task Scheduling**: Schedule tasks for deferred execution
- 🔌 **Plugin System**: Extensible plugin architecture for custom functionality
- 📝 **Annotations**: Add comments, highlights, and notes to documents
- 👥 **Collaboration**: Track changes and identify collaborators
- 🔎 **Search & Indexing**: Full-text search across documents
- 📧 **Notifications**: Comprehensive notification system
- 📈 **Analytics**: Advanced analytics and reporting
- 💾 **Versioning**: Document version control
- 🗄️ **Backup & Recovery**: Automated backup and recovery system
- 🔒 **Security**: Advanced security sanitization
- 📦 **Compression**: Document compression and optimization
- 👁️ **OCR**: Extract text from images
- 🔄 **Workflow Automation**: Create and execute automated workflows
- 🤖 **AI Suggestions**: AI-powered content analysis and suggestions
- ☁️ **Cloud Storage**: Upload to S3, Google Drive, Azure Blob Storage
- ✅ **Document Review**: Quality assurance and review system
- 📊 **Advanced Analytics**: Enhanced reporting with custom charts
- 🔌 **API Integrations**: REST API client for external integrations
- 🔔 **Advanced Webhooks**: Webhooks with retry, events, and delivery tracking
- 🔄 **Data Pipelines**: Transform and process data with pipelines
- 🧪 **Testing Framework**: Automated testing and quality validation
- 📊 **Prometheus Metrics**: Advanced metrics collection for monitoring
- 📝 **Structured Logging**: JSON-based structured logging
- 🚦 **Advanced Rate Limiting**: Per-user rate limiting with custom limits
- 🏥 **Health Checks**: Comprehensive health check system
- 📚 **OpenAPI Documentation**: Automatic API documentation with Swagger/ReDoc
- ⚙️ **Dynamic Configuration**: Hot-reload configuration without restart
- 🔴 **Redis Cache**: Distributed caching with Redis
- 📋 **Task Queue**: Asynchronous task processing with queue
- 🔐 **JWT Authentication**: Token-based authentication and authorization
- 📝 **Audit Logging**: Comprehensive audit trail of all actions
- 🖼️ **Advanced Image Optimization**: Intelligent image compression and optimization
- 👁️ **Document Preview**: Generate previews for documents
- 💧 **Advanced Watermarking**: Text, image, and logo watermarks

## 📦 Supported Formats

| Format | Extension | Features |
|--------|-----------|----------|
| **Excel** | `.xlsx` | Tables, charts, formulas, formatting |
| **PDF** | `.pdf` | Text, tables, charts, diagrams, images |
| **Word** | `.docx` | Text, tables, images, formatting, styles |
| **Tableau** | `.twb` | Data connections, visualizations, dashboards |
| **Power BI** | `.pbix` | Data models, visualizations, reports |
| **HTML** | `.html` | Interactive, charts, responsive design |
| **PowerPoint** | `.pptx` | Slides, charts, images, animations |
| **ODT** | `.odt` | OpenDocument text format |
| **RTF** | `.rtf` | Rich text format con tablas y colores |
| **LaTeX** | `.tex` | LaTeX document con fórmulas matemáticas |
| **EPUB** | `.epub` | E-book format para lectores digitales |

## 🛠️ Installation

```bash
cd agents/backend/onyx/server/features/markdown_to_professional_docs_ai
pip install -r requirements.txt
```

## ⚙️ Configuration

Create a `.env` file:

```env
PORT=8035
OUTPUT_DIR=./outputs
TEMP_DIR=./temp
MAX_FILE_SIZE=10485760
CHART_THEME=plotly_white
PDF_ENGINE=weasyprint
PDF_PAGE_SIZE=A4

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60

# Image Processing
MAX_IMAGE_SIZE_MB=5
MAX_IMAGE_DIMENSION=2000
```

## 🚀 Usage

### Start the server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8035 --reload
```

### API Endpoints

#### 1. Convert Markdown Content

```bash
POST /convert
Content-Type: application/json

{
  "markdown_content": "# Title\n\n## Section\n\n| Col1 | Col2 |\n|------|------|\n| A    | 1    |",
  "output_format": "excel",
  "include_charts": true,
  "include_tables": true
}
```

#### 2. Convert Markdown File

```bash
POST /convert/file
Content-Type: multipart/form-data

file: <markdown_file.md>
output_format: excel
include_charts: true
include_tables: true
```

#### 3. Batch Conversion

```bash
POST /convert/batch
Content-Type: application/json

{
  "markdown_contents": ["# Doc 1", "# Doc 2"],
  "output_format": "pdf",
  "include_charts": true,
  "include_tables": true
}
```

#### 4. Convert Repository

```bash
POST /convert/repository
Content-Type: multipart/form-data

repository_path: /path/to/repo
output_format: word
include_charts: true
include_tables: true
recursive: true
```

#### 5. Get Supported Formats

```bash
GET /formats
```

#### 6. Health Check

```bash
GET /health
```

#### 7. Cache Management

```bash
# Get cache statistics
GET /cache/stats

# Clear cache
POST /cache/clear
```

#### 8. Health Check (Enhanced)

```bash
GET /health
# Returns health status with cache information
```

#### 9. Metrics

```bash
# Get service metrics
GET /metrics

# Reset metrics
POST /metrics/reset
```

#### 10. Templates

```bash
# Get available templates
GET /templates

# Create new template
POST /templates/create
Content-Type: application/x-www-form-urlencoded

template_name: my_template
template_config: {"name": "My Template", "styles": {...}}

# Delete template
DELETE /templates/{template_name}
```

#### 11. Dashboard

```bash
# Access interactive web dashboard
GET /dashboard
```

#### 12. Document Comparison

```bash
POST /compare
Content-Type: application/x-www-form-urlencoded

doc1_path: /path/to/doc1.pdf
doc2_path: /path/to/doc2.pdf
comparison_type: content  # content, structure, or metadata
```

#### 13. Translation

```bash
POST /translate
Content-Type: application/x-www-form-urlencoded

content: Hello world
target_language: es
source_language: en  # Optional, auto-detected if not provided
```

#### 14. Digital Signatures

```bash
# Sign a document
POST /sign
Content-Type: application/x-www-form-urlencoded

document_path: /path/to/document.pdf
signer_name: John Doe
private_key: optional_private_key

# Verify signature
POST /verify
Content-Type: application/x-www-form-urlencoded

document_path: /path/to/document.pdf
```

#### 15. Multi-format Export

```bash
# Export to multiple formats simultaneously
POST /export/multiple
Content-Type: application/x-www-form-urlencoded

markdown_content: # My Document...
output_formats: pdf,word,excel,html
include_charts: true
include_tables: true

# Create format package (ZIP)
POST /export/package
Content-Type: application/x-www-form-urlencoded

markdown_content: # My Document...
output_formats: pdf,word,excel
package_name: my_package
include_charts: true
include_tables: true
```

#### 11. Languages

```bash
# Get supported languages
GET /languages
```

#### 12. Dashboard

```bash
# Access interactive web dashboard
GET /dashboard
```

#### 13. Document Comparison

```bash
POST /compare
Content-Type: application/x-www-form-urlencoded

doc1_path: /path/to/doc1.pdf
doc2_path: /path/to/doc2.pdf
comparison_type: content  # content, structure, or metadata
```

#### 14. Translation

```bash
POST /translate
Content-Type: application/x-www-form-urlencoded

content: Hello world
target_language: es
source_language: en  # Optional, auto-detected if not provided
```

#### 15. Digital Signatures

```bash
# Sign a document
POST /sign
Content-Type: application/x-www-form-urlencoded

document_path: /path/to/document.pdf
signer_name: John Doe
private_key: optional_private_key

# Verify signature
POST /verify
Content-Type: application/x-www-form-urlencoded

document_path: /path/to/document.pdf
```

#### 16. Multi-format Export

```bash
# Export to multiple formats simultaneously
POST /export/multiple
Content-Type: application/x-www-form-urlencoded

markdown_content: # My Document...
output_formats: pdf,word,excel,html
include_charts: true
include_tables: true

# Create format package (ZIP)
POST /export/package
Content-Type: application/x-www-form-urlencoded

markdown_content: # My Document...
output_formats: pdf,word,excel
package_name: my_package
include_charts: true
include_tables: true
```

#### 17. Annotations

```bash
# Add annotation
POST /annotations/add

# Get annotations for document
GET /annotations/{document_path}
```

#### 18. Search

```bash
# Search documents
GET /search?query=keyword&limit=10
```

#### 19. Analytics

```bash
# Get analytics report
GET /analytics/report?period=daily  # daily, weekly, monthly
```

#### 20. Collaboration

```bash
# Get collaboration info
GET /collaboration/{document_path}
```

#### 21. Plugins

```bash
# List plugins
GET /plugins
```

#### 22. Scheduler

```bash
# List scheduled tasks
GET /scheduler/tasks

# Get task status
GET /scheduler/task/{task_id}

# Cancel task
POST /scheduler/task/{task_id}/cancel
```

#### 23. Permissions

```bash
# List roles
GET /permissions/roles

# Assign role to user
POST /permissions/user/{user_id}/role
```

#### 24. Workflow Automation

```bash
# Create workflow
POST /workflow/create
Content-Type: application/x-www-form-urlencoded

workflow_name: my_workflow
workflow_steps: [{"name": "convert", "action": "convert", "parameters": {...}}]

# Execute workflow
POST /workflow/execute
Content-Type: application/x-www-form-urlencoded

workflow_name: my_workflow
initial_data: {"markdown_content": "..."}

# Get execution status
GET /workflow/execution/{execution_id}

# List workflows
GET /workflows
```

#### 25. AI Suggestions

```bash
# Get AI-powered suggestions
POST /ai/suggest
Content-Type: application/x-www-form-urlencoded

markdown_content: # My Document...

# Analyze content
POST /ai/analyze
Content-Type: application/x-www-form-urlencoded

markdown_content: # My Document...
```

#### 26. Cloud Storage

```bash
# Upload to cloud
POST /cloud/upload
Content-Type: application/x-www-form-urlencoded

file_path: /path/to/document.pdf
provider: s3  # s3, gdrive, azure
destination: optional_destination

# List providers
GET /cloud/providers
```

#### 27. Document Review

```bash
# Review document
POST /review
Content-Type: application/x-www-form-urlencoded

document_path: /path/to/document.pdf
reviewer: John Doe
markdown_content: optional_content

# Get reviews
GET /review/{document_path}
```

#### 28. API Integrations

```bash
# Register integration
POST /integrations/register
Content-Type: application/x-www-form-urlencoded

name: my_api
base_url: https://api.example.com
api_key: optional_api_key

# List integrations
GET /integrations
```

#### 29. Advanced Webhooks

```bash
# Register webhook
POST /webhooks/register
Content-Type: application/x-www-form-urlencoded

webhook_id: my_webhook
url: https://example.com/webhook
secret: optional_secret
events: conversion.completed,document.signed
retry_count: 3
retry_delay: 5

# List webhooks
GET /webhooks

# Get webhook deliveries
GET /webhooks/deliveries?webhook_id=my_webhook&limit=100
```

#### 30. Data Pipelines

```bash
# Create pipeline
POST /pipeline/create
Content-Type: application/x-www-form-urlencoded

pipeline_name: my_pipeline

# Execute pipeline
POST /pipeline/{pipeline_name}/execute
Content-Type: application/x-www-form-urlencoded

data: {"tables": [...]}

# List pipelines
GET /pipelines
```

#### 31. Testing

```bash
# Run all tests
POST /tests/run

# Get test summary
GET /tests/summary
```

#### 32. Prometheus Metrics

```bash
# Get Prometheus metrics
GET /metrics/prometheus
```

#### 33. Advanced Rate Limiting

```bash
# Set user rate limits
POST /rate-limit/user/{user_id}/limits
Content-Type: application/x-www-form-urlencoded

requests_per_minute: 100
requests_per_hour: 5000
requests_per_day: 50000

# Get user stats
GET /rate-limit/user/{user_id}/stats

# Reset user rate limit
POST /rate-limit/user/{user_id}/reset
```

#### 34. API Documentation

```bash
# Swagger UI
GET /docs

# ReDoc
GET /redoc

# OpenAPI JSON
GET /openapi.json
```

#### 35. Task Queue

```bash
# Enqueue task
POST /queue/task
Content-Type: application/x-www-form-urlencoded

task_type: convert
payload: {"markdown_content": "...", "output_format": "pdf"}
priority: 0

# Get task status
GET /queue/task/{task_id}

# List tasks
GET /queue/tasks?status=completed&limit=100
```

#### 36. Authentication

```bash
# Login
POST /auth/login
Content-Type: application/x-www-form-urlencoded

username: user@example.com
password: password123

# Refresh token
POST /auth/refresh
Content-Type: application/x-www-form-urlencoded

refresh_token: <refresh_token>
```

#### 37. Audit Logs

```bash
# Get audit logs
GET /audit/logs?user_id=user123&action=convert&limit=100
```

#### 38. Dynamic Configuration

```bash
# Get config
GET /config?key=rate_limit.per_minute

# Set config
POST /config
Content-Type: application/x-www-form-urlencoded

key: rate_limit.per_minute
value: 100
```

#### 39. Redis Cache

```bash
# Get Redis cache stats
GET /cache/redis/stats
```

#### 40. Image Optimization

```bash
# Optimize image
POST /images/optimize
Content-Type: application/x-www-form-urlencoded

image_path: /path/to/image.jpg
quality: medium  # low, medium, high, lossless
max_width: 1920
max_height: 1080
format: JPEG  # Optional

# Get image info
GET /images/info?image_path=/path/to/image.jpg
```

#### 41. Document Preview

```bash
# Generate preview
POST /preview/generate
Content-Type: application/x-www-form-urlencoded

document_path: /path/to/document.pdf
format: png  # png, jpg, pdf
page: 1
width: 800
height: 1000
```

#### 42. Watermarking

```bash
# Add watermark
POST /watermark/add
Content-Type: application/x-www-form-urlencoded

document_path: /path/to/document.pdf
watermark_config: {"type": "text", "text": "CONFIDENTIAL", "position": "center", "opacity": 0.5}
output_path: /path/to/output.pdf
```

## 📝 Examples

### Python Example

```python
import requests

# Convert markdown to Excel
response = requests.post(
    "http://localhost:8035/convert",
    json={
        "markdown_content": """
# Sales Report

## Q1 Results

| Product | Sales | Units |
|---------|-------|-------|
| Widget A | $10,000 | 100 |
| Widget B | $15,000 | 150 |
| Widget C | $8,000 | 80 |
        """,
        "output_format": "excel",
        "include_charts": True,
        "include_tables": True
    }
)

result = response.json()
print(f"Output file: {result['output_file']}")
```

### cURL Example

```bash
curl -X POST "http://localhost:8035/convert" \
  -H "Content-Type: application/json" \
  -d '{
    "markdown_content": "# My Document\n\n| A | B |\n|---|---|\n| 1 | 2 |",
    "output_format": "pdf",
    "include_charts": true
  }'
```

## 🎨 Features in Detail

### Charts and Diagrams

- **Automatic Chart Generation**: Tables are automatically converted to charts
- **Multiple Chart Types**: Bar, line, pie, scatter, area, histogram charts supported
- **Auto Chart Detection**: Intelligently selects the best chart type for your data
- **Interactive Charts**: HTML output includes interactive Plotly charts with hover effects
- **Professional Styling**: Charts match professional document standards with consistent colors
- **Value Labels**: Charts include value labels for better readability

### Table Handling

- **Preserved Formatting**: Tables maintain structure across all formats
- **Auto-sizing**: Column widths automatically adjusted
- **Styled Headers**: Professional header styling with colors
- **Data Type Detection**: Automatic detection of numeric data for charts

### Document Styling

- **Professional Themes**: Consistent styling across all formats
- **Custom Styling**: Support for custom styling options
- **Responsive Design**: HTML output is fully responsive
- **Print-Ready**: PDF output optimized for printing

## 📊 Architecture

```
markdown_to_professional_docs_ai/
├── main.py                 # FastAPI application
├── config.py               # Configuration
├── services/
│   ├── markdown_parser.py  # Markdown parsing
│   ├── converter_service.py # Main conversion service
│   └── converters/
│       ├── excel_converter.py
│       ├── pdf_converter.py
│       ├── word_converter.py
│       ├── html_converter.py
│       ├── tableau_converter.py
│       ├── powerbi_converter.py
│       └── ppt_converter.py
└── utils/
    └── chart_generator.py  # Chart generation utilities
```

## 🔧 Advanced Usage

### Custom Styling

```python
{
  "markdown_content": "# Title",
  "output_format": "pdf",
  "custom_styling": {
    "font_family": "Arial",
    "primary_color": "#366092",
    "header_color": "#1a1a1a",
    "table_style": "professional"
  }
}
```

### Repository Conversion

Convert all Markdown files in a repository:

```bash
curl -X POST "http://localhost:8035/convert/repository" \
  -F "repository_path=/path/to/repo" \
  -F "output_format=word" \
  -F "include_charts=true" \
  -F "recursive=true"
```

## ⚡ Performance & Caching

The service includes intelligent caching to improve performance:

- **Automatic Caching**: Conversions are automatically cached
- **Cache TTL**: Default 24 hours (configurable)
- **Cache Statistics**: Monitor cache usage via `/cache/stats`
- **Cache Management**: Clear cache via `/cache/clear`

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Chart Generation Fails**: Check that matplotlib backend is set correctly
   - The code uses 'Agg' backend for non-interactive use

3. **Tableau/Power BI Files**: These create basic structures
   - Full integration requires respective APIs/SDKs

4. **Large Files**: Adjust `MAX_FILE_SIZE` in config if needed

5. **Validation Errors**: Check error messages for specific validation issues
   - Format suggestions are provided for invalid formats
   - File size limits are clearly indicated

6. **Cache Issues**: Clear cache if experiencing stale results
   ```bash
   POST /cache/clear
   ```

7. **Mermaid Diagrams**: Install Mermaid CLI for diagram rendering
   ```bash
   npm install -g @mermaid-js/mermaid-cli
   ```
   Without the CLI, diagrams will be extracted but not rendered.

8. **Rate Limiting**: Adjust rate limits in configuration if needed
   - Default: 100 requests per 60 seconds per IP
   - Configure via `RATE_LIMIT_REQUESTS` and `RATE_LIMIT_WINDOW`

## 📚 Documentation

- [API Reference](API_REFERENCE.md)
- [Format Specifications](FORMAT_SPECS.md)
- [Examples](EXAMPLES.md)

## 🤝 Contributing

Contributions welcome! Please follow the existing code structure and add tests for new features.

## 📄 License

Part of Blatam Academy system.

## 🔗 Related Features

- [Analizador de Documentos](../analizador_de_documentos/README.md)
- [AI Document Processor](../ai_document_processor/README.md)

## 🆕 What's New

### v1.8.0 (Latest)
- ✅ Advanced batch processing with progress tracking
- ✅ Plugin system for extensibility
- ✅ Task scheduler for scheduled conversions
- ✅ Permissions and roles system
- ✅ Role-based access control
- ✅ Plugin management endpoints
- ✅ Task scheduling endpoints
- ✅ Permission management endpoints

### v1.7.0
- ✅ Annotation system for documents (comments, highlights, notes)
- ✅ Collaboration tracking with change history
- ✅ Search and indexing system
- ✅ Notification system with priorities
- ✅ Analytics and reporting engine
- ✅ Daily/weekly/monthly reports
- ✅ Document search functionality
- ✅ Collaboration statistics

### v1.6.0
- ✅ OCR support for text extraction from images
- ✅ Document versioning system
- ✅ Backup and recovery system
- ✅ EPUB converter for e-books
- ✅ Version management endpoints
- ✅ Backup management endpoints
- ✅ Automatic version creation

### v1.5.0
- ✅ Document validation system
- ✅ Document compression utilities
- ✅ Advanced security sanitization
- ✅ CSS generator from templates
- ✅ MathJax integration in HTML
- ✅ Automatic content sanitization
- ✅ Path traversal protection
- ✅ XSS prevention

### v1.4.0
- ✅ LaTeX converter for academic documents
- ✅ Enhanced RTF converter with full formatting
- ✅ Watermark generation for documents
- ✅ Parallel processing for better performance
- ✅ Advanced table processing with statistics
- ✅ Formula detection in tables
- ✅ Table type detection (matrix, pivot, data)

### v1.3.0
- ✅ Custom templates system (Professional, Modern, Classic)
- ✅ Math formula rendering (LaTeX/MathJax)
- ✅ Multi-language support (10 languages with auto-detection)
- ✅ Webhook notifications for async conversions
- ✅ Template customization and merging
- ✅ Language detection and translation

### v1.2.0
- ✅ Mermaid diagram support (flowcharts, sequence diagrams, etc.)
- ✅ Comprehensive metrics and monitoring system
- ✅ Rate limiting to protect the service
- ✅ Image processing and optimization
- ✅ Enhanced parser with Mermaid extraction
- ✅ Metrics endpoints for monitoring

### v1.1.0
- ✅ Enhanced error handling with custom exceptions
- ✅ Comprehensive input validation
- ✅ Smart caching system
- ✅ Auto chart type detection
- ✅ Enhanced Markdown parsing (blockquotes, emphasis, etc.)
- ✅ More chart types (scatter, area, histogram)
- ✅ Better chart styling and formatting
- ✅ Cache management endpoints
- ✅ Improved error messages with suggestions

See [IMPROVEMENTS.md](IMPROVEMENTS.md) and [IMPROVEMENTS_V2.md](IMPROVEMENTS_V2.md) for detailed changelogs.

---

**Port**: 8035  
**Status**: ✅ Active  
**Version**: 1.8.0

