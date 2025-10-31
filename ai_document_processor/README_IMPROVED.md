# Improved Real AI Document Processor

An enhanced, production-ready AI document processing system with advanced features, real-time monitoring, and comprehensive document support.

## 🚀 Enhanced Features

### Core AI Capabilities
- **Advanced Text Analysis**: Complexity, readability, and quality metrics
- **Sentiment Analysis**: Multi-dimensional emotion detection
- **Text Classification**: Automatic categorization with confidence scores
- **Text Summarization**: Intelligent summarization with customizable length
- **Question Answering**: Context-aware Q&A system
- **Keyword Extraction**: Smart keyword extraction with relevance scoring
- **Language Detection**: Multi-language support with confidence levels
- **Named Entity Recognition**: Advanced entity extraction and classification
- **Part-of-Speech Tagging**: Comprehensive grammatical analysis

### Advanced Features
- **Similarity Analysis**: Text similarity using sentence transformers
- **Topic Modeling**: Automatic topic extraction and analysis
- **Language Pattern Analysis**: Advanced linguistic pattern detection
- **Quality Metrics**: Text quality assessment and scoring
- **Batch Processing**: Efficient multi-document processing
- **Caching System**: Redis and memory caching for performance
- **Real-time Monitoring**: Performance metrics and health monitoring

### Document Support
- **PDF**: Full text extraction with metadata
- **DOCX**: Word document processing
- **XLSX/XLS**: Excel spreadsheet analysis
- **PPTX**: PowerPoint presentation processing
- **TXT**: Plain text with encoding detection
- **Images**: OCR with Tesseract integration
- **Batch Upload**: Multiple document processing

## 🛠️ Technology Stack

### Core Technologies
- **FastAPI**: Modern, fast web framework
- **spaCy**: Advanced NLP processing
- **NLTK**: Natural language toolkit
- **Transformers**: Hugging Face models
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning algorithms

### Advanced Libraries
- **Sentence Transformers**: Semantic similarity
- **Redis**: Caching and session management
- **Tesseract**: OCR capabilities
- **OpenCV**: Image processing
- **Prometheus**: Metrics collection

## 📦 Installation

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd ai-document-processor

# Run installation script
python install_improved.py

# Start the application
python improved_app.py
```

### Manual Installation
```bash
# Install dependencies
pip install -r real_requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm

# Install NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords')"

# Install Tesseract (system dependent)
# Windows: Download from GitHub
# macOS: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr
```

### Docker Installation
```bash
# Build Docker image
docker build -t ai-doc-processor .

# Run container
docker run -p 8000:8000 ai-doc-processor
```

## 🚀 Usage

### Start the Server
```bash
python improved_app.py
```

The server will start on `http://localhost:8000`

### API Documentation
Visit `http://localhost:8000/docs` for interactive API documentation.

## 📚 API Endpoints

### Basic Processing
- `POST /api/v1/real-documents/process-text` - Basic text processing
- `POST /api/v1/real-documents/analyze-sentiment` - Sentiment analysis
- `POST /api/v1/real-documents/classify-text` - Text classification
- `POST /api/v1/real-documents/summarize-text` - Text summarization
- `POST /api/v1/real-documents/extract-keywords` - Keyword extraction
- `POST /api/v1/real-documents/detect-language` - Language detection
- `POST /api/v1/real-documents/answer-question` - Question answering

### Advanced Processing
- `POST /api/v1/advanced-documents/process-text-advanced` - Advanced text processing
- `POST /api/v1/advanced-documents/upload-document` - Document upload and processing
- `POST /api/v1/advanced-documents/analyze-complexity` - Text complexity analysis
- `POST /api/v1/advanced-documents/analyze-readability` - Readability analysis
- `POST /api/v1/advanced-documents/analyze-similarity` - Similarity analysis
- `POST /api/v1/advanced-documents/analyze-topics` - Topic analysis
- `POST /api/v1/advanced-documents/batch-process` - Batch processing

### Utility Endpoints
- `GET /api/v1/advanced-documents/supported-formats` - Supported file formats
- `GET /api/v1/advanced-documents/processing-stats` - Processing statistics
- `POST /api/v1/advanced-documents/clear-cache` - Clear processing cache
- `GET /api/v1/advanced-documents/health-advanced` - Advanced health check
- `GET /api/v1/advanced-documents/capabilities-advanced` - Advanced capabilities

### Monitoring
- `GET /health` - Basic health check
- `GET /status` - Detailed status
- `GET /metrics` - Prometheus metrics

## 💡 Examples

### Basic Text Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/real-documents/process-text" \
  -F "text=This is a great product! I love it." \
  -F "task=analyze"
```

### Advanced Analysis
```bash
curl -X POST "http://localhost:8000/api/v1/advanced-documents/process-text-advanced" \
  -F "text=Your document text here..." \
  -F "task=analyze" \
  -F "use_cache=true"
```

### Document Upload
```bash
curl -X POST "http://localhost:8000/api/v1/advanced-documents/upload-document" \
  -F "file=@document.pdf" \
  -F "task=analyze"
```

### Batch Processing
```bash
curl -X POST "http://localhost:8000/api/v1/advanced-documents/batch-process" \
  -F "texts=Text 1" \
  -F "texts=Text 2" \
  -F "texts=Text 3" \
  -F "task=analyze"
```

### Python Client Example
```python
import requests

# Basic analysis
response = requests.post(
    "http://localhost:8000/api/v1/real-documents/process-text",
    data={"text": "This is a great product!", "task": "analyze"}
)
result = response.json()

# Advanced analysis
response = requests.post(
    "http://localhost:8000/api/v1/advanced-documents/process-text-advanced",
    data={"text": "Your document text...", "task": "analyze", "use_cache": True}
)
advanced_result = response.json()

# Document upload
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/advanced-documents/upload-document",
        files={"file": f},
        data={"task": "analyze"}
    )
    document_result = response.json()
```

## ⚙️ Configuration

### Environment Variables
```env
# Application settings
DEBUG=false
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000

# AI Model settings
SPACY_MODEL=en_core_web_sm
MAX_TEXT_LENGTH=5120
MAX_SUMMARY_LENGTH=150
MIN_SUMMARY_LENGTH=30

# Processing settings
ENABLE_SPACY=true
ENABLE_NLTK=true
ENABLE_TRANSFORMERS=true
ENABLE_SENTIMENT=true
ENABLE_CLASSIFICATION=true
ENABLE_SUMMARIZATION=true
ENABLE_QA=true

# Advanced features
ENABLE_SENTENCE_TRANSFORMERS=true
ENABLE_SKLEARN=true
ENABLE_CACHING=true
ENABLE_REDIS=false

# API settings
RATE_LIMIT_PER_MINUTE=100
MAX_FILE_SIZE_MB=10

# Cache settings
CACHE_TTL=3600
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Performance settings
ENABLE_COMPRESSION=true
ENABLE_METRICS=true
ENABLE_MONITORING=true
```

## 📊 Performance

### System Requirements
- **RAM**: 4GB+ recommended (8GB+ for optimal performance)
- **CPU**: 2+ cores recommended (4+ cores for high load)
- **Storage**: 2GB+ for models and cache
- **Network**: Stable internet for model downloads

### Performance Metrics
- **Basic Analysis**: < 1 second
- **Advanced Analysis**: 1-3 seconds
- **Document Upload**: 2-5 seconds
- **Batch Processing**: 1-2 seconds per document
- **Cache Hit Rate**: 80%+ with Redis

### Optimization Tips
1. **Enable Redis**: For better caching performance
2. **Use Batch Processing**: For multiple documents
3. **Enable Compression**: For large responses
4. **Monitor Metrics**: Use `/metrics` endpoint
5. **Tune Cache TTL**: Based on your use case

## 🔧 Troubleshooting

### Common Issues

#### Models Not Loading
```bash
# Reinstall spaCy model
python -m spacy download en_core_web_sm

# Reinstall NLTK data
python -c "import nltk; nltk.download('vader_lexicon')"
```

#### Memory Issues
- Increase system RAM
- Reduce `MAX_TEXT_LENGTH`
- Enable Redis caching
- Use smaller models

#### Performance Issues
- Enable Redis caching
- Use batch processing
- Monitor with `/metrics`
- Check system resources

#### OCR Not Working
- Install Tesseract OCR
- Add to system PATH
- Check image formats
- Verify Tesseract installation

### Debug Mode
```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Start application
python improved_app.py
```

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest

# Run with coverage
pytest --cov=.
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f load_test.py --host=http://localhost:8000
```

## 📈 Monitoring

### Health Checks
- `GET /health` - Basic health
- `GET /status` - Detailed status
- `GET /metrics` - Prometheus metrics

### Key Metrics
- Request rate
- Response time
- Cache hit rate
- Error rate
- Memory usage
- CPU usage

### Prometheus Integration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'ai-doc-processor'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

## 🚀 Deployment

### Production Deployment
```bash
# Use production server
uvicorn improved_app:app --host 0.0.0.0 --port 8000 --workers 4

# With Gunicorn
gunicorn improved_app:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker Deployment
```bash
# Build image
docker build -t ai-doc-processor .

# Run with environment
docker run -p 8000:8000 \
  -e REDIS_HOST=redis-server \
  -e ENABLE_REDIS=true \
  ai-doc-processor
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-doc-processor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-doc-processor
  template:
    metadata:
      labels:
        app: ai-doc-processor
    spec:
      containers:
      - name: ai-doc-processor
        image: ai-doc-processor:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: ENABLE_REDIS
          value: "true"
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd ai-document-processor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r real_requirements.txt
pip install -r dev_requirements.txt

# Run tests
pytest

# Format code
black .
flake8 .
```

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests for new features
- Update documentation

## 📄 License

MIT License - see LICENSE file for details.

## 🆘 Support

### Getting Help
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Check system requirements
4. Monitor performance metrics
5. Create an issue on GitHub

### Community
- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Wiki: Additional documentation and guides

## 🔄 Changelog

### Version 2.0.0
- Added advanced AI processing
- Implemented document upload
- Added caching system
- Enhanced monitoring
- Improved performance
- Added batch processing
- Enhanced error handling

### Version 1.0.0
- Basic AI processing
- Core NLP features
- API endpoints
- Basic monitoring