# Advanced AI Document Processor

> Part of the [Blatam Academy Integrated Platform](../README.md)

An advanced document processing system with AI/ML capabilities, OCR, content analysis, and batch processing.

## 🚀 Key Features

### 📄 Document Processing
- **Multi-Format Support** — PDF, DOCX, DOC, TXT, RTF, ODT, PPTX, XLSX, CSV
- **Advanced OCR** — Optical character recognition with EasyOCR and Tesseract
- **Metadata Extraction** — Complete document information
- **Batch Processing** — Efficient handling of multiple documents

### 🤖 AI/ML Analysis
- **Document Classification** — Automatic categorization using transformers
- **Entity Extraction** — Identification of persons, places, organizations
- **Sentiment Analysis** — Evaluation of content tone and emotions
- **Topic Modeling** — Main topic identification (LDA)
- **Automatic Summarization** — Summary generation using BART
- **Keyword Extraction** — Identification of important terms
- **Content Analysis** — Quality and readability evaluation

### 🔍 Search & Comparison
- **Semantic Search** — Intelligent search based on meaning
- **Document Comparison** — Similarity analysis and plagiarism detection
- **Vector Database** — Efficient embedding storage

### ⚡ Advanced Features
- **Async Processing** — Efficient multi-task handling
- **WebSockets** — Real-time updates
- **Smart Cache** — Performance optimization with Redis
- **Export** — Multiple output formats (JSON, CSV, XLSX, PDF, DOCX)
- **RESTful API** — Comprehensive and well-documented interface

## 🛠️ Technologies Used

### Core Framework
- **FastAPI** — Modern and fast web framework
- **Pydantic** — Data validation and configuration
- **Uvicorn** — High-performance ASGI server

### AI/ML & NLP
- **Transformers** — Pre-trained language models
- **PyTorch** — Deep learning framework
- **spaCy** — Natural Language Processing
- **NLTK** — NLP tools
- **scikit-learn** — Machine Learning
- **sentence-transformers** — Semantic embeddings

### Document Processing
- **PyPDF2/pdfplumber** — PDF processing
- **python-docx** — Word document processing
- **openpyxl** — Excel processing
- **python-pptx** — PowerPoint processing
- **EasyOCR/Tesseract** — Optical Character Recognition

### Database & Storage
- **SQLAlchemy** — ORM for databases
- **Redis** — In-memory cache
- **ChromaDB** — Vector database
- **FAISS** — Similarity search

### Monitoring & Logging
- **Prometheus** — System metrics
- **Sentry** — Error monitoring
- **structlog** — Structured logging

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip
- Redis (optional, for cache)
- SQL Database (optional)

### Dependency Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-document-processor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy models
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"
```

### Configuration

1. Copy configuration file:
```bash
cp env.example .env
```

2. Edit `.env` with your settings:
```bash
# Basic configuration
APP_NAME="Advanced AI Document Processor"
DEBUG=true
HOST=0.0.0.0
PORT=8001

# Database (optional)
DATABASE_URL=sqlite:///./document_processor.db

# Redis (optional)
REDIS_URL=redis://localhost:6379

# AI/ML Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CLASSIFICATION_MODEL=distilbert-base-uncased
SUMMARIZATION_MODEL=facebook/bart-large-cnn
```

## 🚀 Usage

### Start Server

```bash
# Development
python app.py

# Production
uvicorn app:app --host 0.0.0.0 --port 8001 --workers 4
```

### Access Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

## 📚 API Endpoints

### Documents

#### Upload Document
```bash
curl -X POST "http://localhost:8001/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "analysis_types=content_analysis,classification,ocr" \
  -F "language=en"
```

#### Upload Batch
```bash
curl -X POST "http://localhost:8001/api/v1/documents/upload/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "batch_name=my_batch" \
  -F "analysis_types=content_analysis,classification"
```

#### Get Result
```bash
curl -X GET "http://localhost:8001/api/v1/documents/{document_id}"
```

### Specific Analysis

#### OCR Analysis
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/ocr"
```

#### Classification
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/classification"
```

#### Entity Extraction
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/entities"
```

#### Sentiment Analysis
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/sentiment"
```

#### Topic Modeling
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/topics"
```

#### Summary
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/summary"
```

#### Keywords
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/keywords"
```

#### Content Analysis
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/analyze/content"
```

### Search & Comparison

#### Semantic Search
```bash
curl -X POST "http://localhost:8001/api/v1/documents/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "documents about artificial intelligence",
    "search_type": "semantic",
    "limit": 10
  }'
```

#### Compare Documents
```bash
curl -X POST "http://localhost:8001/api/v1/documents/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "document_ids": ["doc1_id", "doc2_id"],
    "comparison_type": "similarity",
    "threshold": 0.8
  }'
```

### Export

#### Export Results
```bash
curl -X POST "http://localhost:8001/api/v1/documents/{document_id}/export" \
  -F "format=json"
```

## 🔧 Advanced Configuration

### Available Analysis Types

- `ocr`: Optical Character Recognition
- `classification`: Document classification
- `entity_extraction`: Entity extraction
- `sentiment_analysis`: Sentiment analysis
- `topic_modeling`: Topic modeling
- `summarization`: Automatic summarization
- `keyword_extraction`: Keyword extraction
- `semantic_search`: Semantic search
- `plagiarism_detection`: Plagiarism detection
- `content_analysis`: Content analysis

### Supported Formats

- **PDF**: PDF Documents
- **DOCX/DOC**: Microsoft Word Documents
- **TXT**: Plain Text Files
- **RTF**: Rich Text Format
- **ODT**: OpenDocument Text
- **PPTX**: PowerPoint Presentations
- **XLSX**: Excel Spreadsheets
- **CSV**: CSV Files
- **Images**: PNG, JPG, JPEG, TIFF, BMP (with OCR)

### Supported OCR Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)

## 🐳 Docker

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
```

## 🧪 Testing

```bash
# Run tests
pytest

# Tests with coverage
pytest --cov=.
```

## 📊 Monitoring

### Prometheus Metrics
- Endpoint: http://localhost:9091/metrics
- Available metrics:
  - `documents_processed_total`
  - `document_processing_duration_seconds`
  - `active_processing_jobs`
  - `error_rate`

## 🔒 Security

### Security Configuration
- File type validation
- File size limits
- Rate limiting
- Configurable CORS
- JWT Authentication (optional)

## 🤝 Contribution

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is under the MIT License. See `LICENSE` file for details.

---

**Developed with ❤️ using FastAPI, Transformers, and cutting-edge AI/ML technologies.**

[← Back to Main README](../README.md)