# Intelligent Document Analyzer

> Part of the [Blatam Academy Integrated Platform](../README.md)

Advanced document analysis system with fine-tuning capabilities and adaptive learning. Provides comprehensive document analysis including classification, summarization, information extraction, sentiment analysis, and more.

## 🚀 Features

- **Multi-Task Analysis** — Classification, summarization, keyword extraction, sentiment analysis, entity recognition, topic modeling
- **Fine-Tuning** — Complete system to train custom models on your own data
- **Multi-Format** — Supports PDF, DOCX, TXT, HTML, Markdown, JSON, XML, CSV
- **REST API** — Complete endpoints for easy integration
- **Embeddings** — Generation of embeddings for semantic search and comparison
- **Question-Answering** — Answer questions about documents
- **Caching System** — Smart cache with multiple backends (memory, disk, Redis)
- **Batch Processing** — Optimized parallel processing for multiple documents
- **Rate Limiting** — Protection against abuse with configurable rate limiting
- **Metrics and Monitoring** — Complete performance metrics and statistics system
- **Performance Optimizations** — Parallel processing, caching, and memory optimizations
- **Document Comparison** — Semantic comparison and similarity detection
- **Structured Extraction** — Information extraction according to custom schemas
- **Style Analysis** — Analysis of readability, complexity, and writing quality
- **Multi-format Export** — Export results in JSON, CSV, Markdown, HTML
- **Semantic Search** — Find similar documents using embeddings
- **Plagiarism Detection** — Detect possible plagiarism by comparing with reference corpus
- **Advanced Semantic Search Engine** — Hybrid search with vector indexes
- **Workflow Automation** — Customizable workflows for automated analysis
- **Vector Databases** — Integration with Pinecone, Weaviate, Chroma, Qdrant, Milvus
- **Anomaly Detection** — Automatic detection of anomalies and inconsistencies
- **Predictive Analysis** — Forecasting and predictions based on historical trends
- **Image Analysis** — Object detection, OCR, color analysis in images
- **Alert System** — Configurable alerts with custom rules
- **Full Audit** — Log of all system actions
- **WebSockets** — Real-time updates via WebSocket

## 📋 Requirements

- Python 3.8+
- CUDA (optional, for GPU)
- 8GB+ RAM recommended
- 10GB+ disk space for models

## 🛠️ Installation

### 1. Install dependencies

```bash
cd analizador_de_documentos
pip install -r requirements.txt
```

### 2. Configure environment variables (optional)

Create `.env` file:

```env
HOST=0.0.0.0
PORT=8000
MODEL_NAME=bert-base-multilingual-cased
DEVICE=cuda  # or cpu
```

## 🚀 Quick Usage

### Start server

```bash
python main.py
```

Server will be available at `http://localhost:8000`

### API Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

## 📖 API Usage

### Analyze a document

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "document_content": "This is a document about artificial intelligence...",
    "tasks": ["classification", "summarization", "keywords"]
  }'
```

### Classify text

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/classify" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This document is about technology"
  }'
```

### Generate summary

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Long text to summarize...",
    "max_length": 150,
    "min_length": 30
  }'
```

### Upload and analyze file

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/analyze/upload" \
  -F "file=@document.pdf" \
  -F "tasks=classification,summarization"
```

## 🔧 Fine-Tuning

### Prepare training data

Data must be in JSON format:

```json
[
  {"text": "Example text 1", "label": 0},
  {"text": "Example text 2", "label": 1},
  ...
]
```

### Train model

```bash
python training/train_model.py \
  --data training_data.json \
  --num-labels 3 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 2e-5
```

### Create sample data

```bash
python training/train_model.py \
  --create-sample \
  --data sample_data.json \
  --sample-size 100
```

### Use fine-tuned model

```python
from core.document_analyzer import DocumentAnalyzer

analyzer = DocumentAnalyzer(
    fine_tuned_model_path="./models/fine_tuned/trained_model"
)

result = await analyzer.analyze_document(
    document_content="Text to analyze..."
)
```

## 📚 Project Structure

```
analizador_de_documentos/
├── core/
│   ├── document_analyzer.py      # Main analyzer
│   ├── fine_tuning_model.py       # Fine-tuning system
│   ├── document_processor.py      # Document processor
│   └── embedding_generator.py     # Embedding generator
├── api/
│   └── routes.py                  # REST API Endpoints
├── training/
│   └── train_model.py             # Training script
├── config/
│   └── config.yaml                # Configuration
├── models/                         # Saved models
│   ├── cache/                      # Model cache
│   └── fine_tuned/                 # Fine-tuned models
├── main.py                         # Main application
├── requirements.txt                # Dependencies
└── README.md                       # This documentation
```

## 🎯 Available Analysis Tasks

- **classification**: Classify document into categories
- **summarization**: Generate document summary
- **keyword_extraction**: Extract keywords
- **sentiment**: Sentiment analysis
- **entity_recognition**: Named entity recognition
- **topic_modeling**: Topic extraction
- **question_answering**: Answer questions about the document

## 🔌 Python Integration

```python
from core.document_analyzer import DocumentAnalyzer, AnalysisTask

# Initialize analyzer
analyzer = DocumentAnalyzer()

# Analyze document
result = await analyzer.analyze_document(
    document_content="Document text...",
    tasks=[
        AnalysisTask.CLASSIFICATION,
        AnalysisTask.SUMMARIZATION,
        AnalysisTask.KEYWORD_EXTRACTION
    ]
)

print(f"Summary: {result.summary}")
print(f"Keywords: {result.keywords}")
print(f"Classification: {result.classification}")
```

## 🚀 New Enhanced Features

### Caching System

The system now includes smart caching that significantly improves performance:

```python
# Configure cache backend (memory, disk, redis, auto)
import os
os.environ["CACHE_BACKEND"] = "redis"  # or "memory", "disk", "auto"

# Cache is used automatically in all operations
```

### Batch Processing

Process multiple documents in parallel:

```bash
curl -X POST "http://localhost:8000/api/analizador-documentos/batch/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"content": "Document 1...", "document_id": "doc1"},
      {"content": "Document 2...", "document_id": "doc2"}
    ],
    "tasks": ["classification", "summarization"],
    "max_workers": 10,
    "batch_size": 100
  }'
```

### Metrics and Monitoring

Access performance metrics in real-time:

```bash
# View all metrics
curl http://localhost:8000/api/analizador-documentos/metrics/

# View performance statistics
curl http://localhost:8000/api/analizador-documentos/metrics/performance

# Detailed health check
curl http://localhost:8000/api/analizador-documentos/metrics/health
```

### Rate Limiting

Automatic protection against abuse:

- Default limit: 50 requests per minute per IP
- Configurable per endpoint
- Rate limit headers in responses

## 📊 Supported Models

- `bert-base-multilingual-cased` (default)
- `distilbert-base-multilingual-cased`
- `xlm-roberta-base`
- Any HuggingFace Transformers compatible model

## 🎓 Advanced Fine-Tuning

### Custom configuration

```python
from core.fine_tuning_model import FineTuningModel, FineTuningConfig

config = FineTuningConfig(
    model_name="bert-base-multilingual-cased",
    num_labels=5,
    max_length=512,
    batch_size=32,
    learning_rate=3e-5,
    num_epochs=10,
    output_dir="./my_custom_model"
)

model = FineTuningModel(config=config)

# Prepare data
train_dataset, eval_dataset = model.prepare_dataset(texts, labels)

# Train
results = model.train(train_dataset, eval_dataset)
```

## 🐛 Troubleshooting

### Error: CUDA out of memory

Reduce `batch_size` in configuration or use CPU:

```python
analyzer = DocumentAnalyzer(device="cpu")
```

### Error: Model not found

Models are downloaded automatically the first time. If there are issues, download manually:

```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("model-name")
model = AutoModel.from_pretrained("model-name")
```

### Very long documents

The system processes large documents automatically by splitting them into chunks. For very large documents (>10MB), consider pre-processing.

## 📝 Examples

See `examples/` folder for full usage examples.

## 🤝 Contribution

Contributions are welcome. Please:

1. Fork the project
2. Create a branch for your feature
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

Proprietary — Blatam Academy

## 🔗 References

- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

[← Back to Main README](../README.md)
