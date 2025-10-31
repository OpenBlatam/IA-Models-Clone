# NotebookLM AI - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### 1. Prerequisites
- Python 3.9 or higher
- 4GB+ RAM (8GB+ recommended)
- GPU with CUDA support (optional, for acceleration)

### 2. Installation

```bash
# Navigate to the project directory
cd agents/backend/onyx/server/features/notebooklm_ai

# Install dependencies
pip install -r requirements_notebooklm.txt

# Install spaCy model
python -m spacy download en_core_web_sm
```

### 3. Quick Test

```bash
# Run simple test
python test_simple.py

# Run full demo
python demo_notebooklm.py
```

### 4. Basic Usage

```python
# Import components
from infrastructure.ai_engines import DocumentProcessor, CitationGenerator

# Initialize processors
doc_processor = DocumentProcessor()
citation_gen = CitationGenerator()

# Process a document
text = "Artificial Intelligence is transforming the world..."
analysis = doc_processor.process_document(text, "AI Revolution")

print(f"Word count: {analysis['word_count']}")
print(f"Sentiment: {analysis['sentiment']}")
print(f"Key points: {analysis['key_points'][:3]}")
```

### 5. Create Your First Notebook

```python
from core.entities import Notebook, Document, User, DocumentType, UserId, NotebookId, DocumentId

# Create user and notebook
user = User(
    id=UserId(),
    username="researcher",
    email="researcher@example.com"
)

notebook = Notebook(
    id=NotebookId(),
    title="My Research Notebook",
    user_id=user.id
)

# Add a document
document = Document(
    id=DocumentId(),
    title="Research Paper",
    content="Your research content here...",
    document_type=DocumentType.TXT
)

notebook.add_document(document)
print(f"Notebook created with {notebook.total_documents} documents")
```

## 🔧 Configuration

### Environment Variables
```bash
# AI Model Configuration
export NOTEBOOKLM_MODEL_NAME="microsoft/DialoGPT-medium"
export NOTEBOOKLM_MAX_LENGTH=2048
export NOTEBOOKLM_TEMPERATURE=0.7

# Database Configuration
export NOTEBOOKLM_DB_URL="postgresql://user:pass@localhost/notebooklm"
export NOTEBOOKLM_REDIS_URL="redis://localhost:6379"
```

### AI Engine Configuration
```python
from infrastructure.ai_engines import AIEngineConfig, AdvancedLLMEngine

config = AIEngineConfig(
    model_name="microsoft/DialoGPT-medium",
    max_length=1024,
    temperature=0.7,
    use_quantization=True,  # For memory efficiency
    device="auto"  # Automatically detect GPU/CPU
)

engine = AdvancedLLMEngine(config)
```

## 📊 Performance Tips

### GPU Acceleration
```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name()}")
    device = "cuda"
else:
    print("Using CPU")
    device = "cpu"
```

### Batch Processing
```python
# Process multiple documents efficiently
documents = ["doc1", "doc2", "doc3"]
results = []

for doc in documents:
    result = doc_processor.process_document(doc)
    results.append(result)
```

### Caching
```python
# Enable caching for repeated operations
import redis

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# Cache document analysis results
cache_key = f"doc_analysis_{hash(document_content)}"
cached_result = redis_client.get(cache_key)

if cached_result:
    analysis = json.loads(cached_result)
else:
    analysis = doc_processor.process_document(document_content)
    redis_client.setex(cache_key, 3600, json.dumps(analysis))
```

## 🧪 Testing

### Run Tests
```bash
# Run simple test
python test_simple.py

# Run unit tests
pytest tests/unit/ -v

# Run with coverage
pytest --cov=notebooklm_ai --cov-report=html
```

### Performance Testing
```bash
# Run performance benchmarks
python -m pytest tests/performance/ -v

# Load testing
python tests/load_test.py
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements_notebooklm.txt --force-reinstall
   ```

2. **CUDA Out of Memory**
   ```python
   # Reduce batch size or enable quantization
   config = AIEngineConfig(
       batch_size=1,  # Reduce from default
       use_quantization=True
   )
   ```

3. **Slow Processing**
   ```python
   # Enable GPU acceleration
   config = AIEngineConfig(device="cuda")
   
   # Or use smaller models
   config = AIEngineConfig(model_name="microsoft/DialoGPT-small")
   ```

4. **spaCy Model Not Found**
   ```bash
   # Install spaCy model
   python -m spacy download en_core_web_sm
   
   # Or download manually
   python -m spacy download en_core_web_sm --direct
   ```

### Performance Optimization

1. **Memory Optimization**
   ```python
   # Use quantization for large models
   config = AIEngineConfig(
       use_quantization=True,
       use_flash_attention=True
   )
   ```

2. **Processing Speed**
   ```python
   # Increase batch size for GPU
   config = AIEngineConfig(
       batch_size=8,  # Increase for GPU
       max_workers=4
   )
   ```

3. **Caching Strategy**
   ```python
   # Implement multi-level caching
   # Memory cache for frequent access
   # Redis cache for shared access
   # Disk cache for persistent storage
   ```

## 📚 Next Steps

### Advanced Usage
1. **Custom Models**: Integrate your own AI models
2. **API Development**: Build REST APIs with FastAPI
3. **Web Interface**: Create web-based notebook interface
4. **Collaboration**: Add multi-user support
5. **Analytics**: Implement usage analytics and insights

### Integration
1. **Database**: Connect to PostgreSQL, MongoDB, or Elasticsearch
2. **Cloud Storage**: Integrate with AWS S3, Google Cloud Storage
3. **Monitoring**: Add Prometheus metrics and Grafana dashboards
4. **Deployment**: Deploy with Docker and Kubernetes

### Customization
1. **Domain Models**: Adapt for specific domains (legal, medical, academic)
2. **Language Support**: Add multi-language processing
3. **Citation Styles**: Add custom citation formats
4. **Analysis Types**: Implement domain-specific analysis

## 🆘 Getting Help

### Documentation
- [Full Documentation](README.md)
- [API Reference](docs/api.md)
- [Architecture Guide](docs/architecture.md)
- [Performance Guide](docs/performance.md)

### Support
- Create an issue on GitHub
- Check the troubleshooting section
- Review the demo examples
- Test with the simple test script

### Community
- Share your use cases
- Contribute improvements
- Report bugs and issues
- Suggest new features

---

**NotebookLM AI** - Start your document intelligence journey today! 🚀 