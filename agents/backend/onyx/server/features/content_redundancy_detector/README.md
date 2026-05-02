# Advanced Content Redundancy Detector

> Part of the [Blatam Academy Integrated Platform](../README.md)

Functional, scalable, and optimized system to detect redundancy in text content with advanced AI/ML capabilities, following Python and FastAPI best practices.

## 🚀 Key Features

### 📊 Basic Content Analysis
- **Redundancy Analysis** — Detects repetition of words and phrases
- **Similarity Comparison** — Compares two texts and calculates similarity
- **Quality Assessment** — Evaluates readability and content quality

### 🤖 Advanced AI/ML Analysis
- **Sentiment Analysis** — Detects emotions and polarity in text
- **Language Detection** — Automatically identifies content language
- **Topic Extraction** — Discovers main topics using LDA
- **Semantic Similarity** — Compares texts using advanced embeddings
- **Plagiarism Detection** — Identifies potentially plagiarized content
- **Entity Extraction** — Finds names, places, organizations
- **Automatic Summarization** — Generates summaries using BART models
- **Readability Analysis** — Advanced evaluation of text complexity
- **Comprehensive Analysis** — Combines all features in a single call
- **Batch Processing** — Efficiently analyzes multiple texts

### ⚡ Performance and Scalability
- **Cache System** — In-memory cache with TTL to optimize responses
- **Rate Limiting** — Speed control by IP and endpoint
- **Advanced Metrics** — Real-time performance monitoring
- **Async Operations** — Optimized for high concurrency

### 🛡️ Security and Robustness
- **Error Handling** — Robust system with guard clauses
- **Advanced Validation** — Input validation with Pydantic
- **Security Headers** — Automatic protection against attacks
- **Structured Logging** — Advanced monitoring and debugging

## 📁 Project Structure

```
content_redundancy_detector/
├── app.py              # Main application with lifespan context manager
├── config.py           # Centralized configuration
├── types.py            # Pydantic models (RORO pattern)
├── utils.py            # Pure utility functions
├── services.py         # Functional services with cache
├── middleware.py       # Optimized middleware (logging, rate limiting, security)
├── routers.py          # Functional route handlers
├── cache.py            # In-memory cache system
├── metrics.py          # Metrics and monitoring system
├── rate_limiter.py     # Rate limiting system
├── tests_functional.py # Functional tests
├── requirements.txt    # Minimal dependencies
├── env.example         # Environment variables
└── README.md           # Complete documentation
```

## 🛠️ Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment (optional)
cp env.example .env

# 3. Run the application
python app.py
```

The application will be available at `http://localhost:8000`

## 📖 Usage

### Basic Content Analysis
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"content": "This is sample text to analyze"}'
```

### Similarity Comparison
```bash
curl -X POST "http://localhost:8000/similarity" \
     -H "Content-Type: application/json" \
     -d '{"text1": "Text one", "text2": "Text two", "threshold": 0.8}'
```

### Sentiment Analysis
```bash
curl -X POST "http://localhost:8000/ai/sentiment" \
     -H "Content-Type: application/json" \
     -d '{"content": "I love this product, it is fantastic!"}'
```

### Topic Extraction
```bash
curl -X POST "http://localhost:8000/ai/topics" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text about tech", "Text about sports"], "num_topics": 2}'
```

### Automatic Summarization
```bash
curl -X POST "http://localhost:8000/ai/summary" \
     -H "Content-Type: application/json" \
     -d '{"content": "Long text to summarize...", "max_length": 150}'
```

## 🔗 Endpoints

### Basic Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Basic system info |
| `GET` | `/health` | System health check |
| `POST` | `/analyze` | Analyze content for redundancy |
| `POST` | `/similarity` | Compare similarity |
| `POST` | `/quality` | Evaluate content quality |
| `GET` | `/stats` | System statistics |

### Advanced AI/ML Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ai/sentiment` | Sentiment analysis |
| `POST` | `/ai/language` | Language detection |
| `POST` | `/ai/topics` | Topic extraction |
| `POST` | `/ai/semantic-similarity` | Semantic similarity |
| `POST` | `/ai/plagiarism` | Plagiarism detection |
| `POST` | `/ai/entities` | Entity extraction |
| `POST` | `/ai/summary` | Automatic summary |
| `POST` | `/ai/readability` | Advanced readability |
| `POST` | `/ai/comprehensive` | Comprehensive analysis |
| `POST` | `/ai/batch` | Batch processing |

## 📚 Documentation
Interactive documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🧪 Testing

```bash
python -m pytest tests.py -v
```

## ⚙️ Configuration

Configurable via environment variables:
- `APP_NAME`: Application name
- `APP_VERSION`: Version
- `DEBUG`: Debug mode
- `HOST`: Server host
- `PORT`: Server port
- `LOG_LEVEL`: Logging level
- `MAX_CONTENT_LENGTH`: Max content length

## 🔧 Technologies

- **FastAPI** — High-performance web framework
- **Pydantic** — Data validation
- **Uvicorn** — ASGI server
- **Transformers** — Pre-trained language models
- **Sentence-Transformers** — Semantic embeddings
- **spaCy** — NLP
- **Redis** — In-memory cache
- **Prometheus** — Monitoring

---

[← Back to Main README](../README.md)