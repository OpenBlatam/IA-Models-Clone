# Web Content Extractor AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

Advanced system for extracting complete information from web pages using OpenRouter and multiple scraping methods.

## Features

### ✨ Advanced Extraction
- **Multiple Extraction Methods**: Trafilatura, Readability, Newspaper3k, BeautifulSoup
- **Advanced Scraping**: Support for JavaScript with Playwright
- **Table Extraction**: Structured HTML tables
- **Video Extraction**: HTML5, embeds (YouTube, Vimeo, etc.)
- **Quote Extraction**: Blockquotes and inline quotes
- **Code Extraction**: Code blocks with language detection
- **Form Extraction**: Form fields and structure
- **RSS/Atom Feeds**: Feed detection and extraction

### 🧠 Intelligent Analysis
- **AI Processing**: OpenRouter for content analysis and structuring
- **Language Detection**: Advanced detection with multiple languages
- **Quality Analysis**: Readability metrics (Flesch, Flesch-Kincaid, etc.)
- **Rich Metadata**: Author, date, keywords, Open Graph, Twitter Cards, JSON-LD
- **Structured Data**: Microdata, Schema.org

### ⚡ Performance
- **Intelligent Cache**: Cache system with TTL to optimize performance
- **Batch Scraping**: Parallel processing of multiple URLs
- **Automatic Retry**: Retries with exponential backoff
- **Rate Limiting**: Request speed control
- **Rotating User Agents**: Prevents blocking

### 🔌 REST API
- **Documented Endpoints**: FastAPI with Swagger/OpenAPI
- **Data Validation**: Pydantic for robust validation
- **Error Handling**: Structured error responses

## Installation

```bash
# Install essential dependencies
pip install -r requirements.txt

# Install optional dependencies (for advanced functionality)
pip install -r requirements-optional.txt

# Install browsers for Playwright (optional, only if using JavaScript rendering)
playwright install chromium
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
OPENROUTER_API_KEY=your_api_key_here
HOST=0.0.0.0
PORT=8000
CACHE_MAX_SIZE=1000
CACHE_TTL=3600
```

## Quick Start

```bash
# Linux/Mac
chmod +x scripts/start.sh
./scripts/start.sh

# Windows
scripts\start.bat

# Or manually
python main.py
```

## Usage

### Start Server

```bash
python main.py
```

Or with uvicorn:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Endpoints

#### Extract Content

```bash
POST /api/v1/extract
Content-Type: application/json

{
  "url": "https://example.com",
  "model": "anthropic/claude-3.5-sonnet",
  "max_tokens": 4000
}
```

#### Cache Statistics

```bash
GET /api/v1/extract/cache/stats
```

#### Clear Cache

```bash
DELETE /api/v1/extract/cache
```

#### Batch Extraction (Multiple URLs)

```bash
POST /api/v1/extract/batch
Content-Type: application/json

{
  "urls": [
    "https://example.com",
    "https://example.org",
    "https://example.net"
  ],
  "max_concurrent": 5,
  "extract_strategy": "auto"
}
```

### Example with curl

```bash
# Extract content
curl -X POST "http://localhost:8000/api/v1/extract" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "model": "anthropic/claude-3.5-sonnet"
  }'

# View cache statistics
curl "http://localhost:8000/api/v1/extract/cache/stats"

# Clear cache
curl -X DELETE "http://localhost:8000/api/v1/extract/cache"

# Batch extraction
curl -X POST "http://localhost:8000/api/v1/extract/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": ["https://example.com", "https://example.org"],
    "max_concurrent": 5
  }'
```

### Example with Python

```python
import asyncio
from example_usage import example_extract, example_cache_stats

# Extract content
asyncio.run(example_extract())

# View statistics
asyncio.run(example_cache_stats())
```

## Available Models

You can use any OpenRouter model. Recommended:

- `anthropic/claude-3.5-sonnet` (default) - Best quality
- `anthropic/claude-3-haiku` - Faster and cheaper
- `openai/gpt-4-turbo` - Excellent for analysis
- `google/gemini-pro` - Good quality/price balance

## Extraction Methods

The system automatically attempts in this order:

1. **Trafilatura** - Best for articles and structured content
2. **Readability** - Extracts clean main content
3. **Newspaper3k** - Ideal for news and articles
4. **BeautifulSoup** - Always available fallback

## Extracted Content

The scraper extracts:

- ✅ **Main Text** - Clean and structured content
- ✅ **Metadata** - Title, description, author, date, keywords
- ✅ **Links** - All links with text and normalized URLs
- ✅ **Images** - With alt text, dimensions, and URLs
- ✅ **Tables** - Complete structure with headers and rows
- ✅ **Videos** - HTML5, embeds (YouTube, Vimeo, etc.)
- ✅ **Quotes** - Blockquotes and inline quotes with authors
- ✅ **Code** - Code blocks with language detection
- ✅ **Forms** - Complete fields and structure
- ✅ **Feeds** - Detected RSS/Atom feeds
- ✅ **Structured Data** - JSON-LD, Microdata, Open Graph
- ✅ **Quality Analysis** - Readability metrics
- ✅ **Language Detection** - With confidence level

## Project Structure

```
web_content_extractor_ai/
├── main.py                          # FastAPI Server
├── config.py                        # Configuration
├── example_usage.py                 # Usage examples
├── infrastructure/
│   ├── openrouter/
│   │   └── client.py               # OpenRouter Client
│   ├── web_scraper/
│   │   └── scraper.py              # Multi-method Scraper
│   └── cache/
│       └── content_cache.py        # Cache System
├── application/
│   └── use_cases/
│       └── extract_content_use_case.py
├── api/
│   └── v1/
│       ├── controllers/
│       │   └── extract_controller.py
│       ├── schemas/
│       │   ├── requests.py
│       │   └── responses.py
│       └── routes.py
└── requirements.txt
```

## Example Response

```json
{
  "success": true,
  "url": "https://example.com",
  "raw_data": {
    "title": "Example Domain",
    "description": "...",
    "links_count": 5,
    "images_count": 2,
    "extraction_method": "trafilatura"
  },
  "extracted_info": "{\"title\": \"...\", \"content\": \"...\"}",
  "processing_metadata": {
    "model_used": "anthropic/claude-3.5-sonnet",
    "tokens_used": 1234
  },
  "message": "Content extracted successfully"
}
```

## Configuration Parameters

- `use_cache`: Use cache (default: true)
- `use_javascript`: Render JavaScript with Playwright (default: false, slower)
- `extract_strategy`: Force specific method ("auto", "trafilatura", "readability", "newspaper", "beautifulsoup")

## API Documentation

Once the server is started, visit:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker

### Build and Run

```bash
# Build image
docker build -t web-content-extractor-ai .

# Run
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your_key web-content-extractor-ai

# Or with docker-compose
docker-compose up
```

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest tests/ --cov=. --cov-report=html
```

## Notes

- Cache has a default TTL of 1 hour
- Playwright requires more resources but handles JavaScript pages better
- Trafilatura is generally the most effective method for articles
- The system automatically detects content encoding
- Rate limiting: 100 requests per minute per IP

---

[← Back to Main README](../README.md)
