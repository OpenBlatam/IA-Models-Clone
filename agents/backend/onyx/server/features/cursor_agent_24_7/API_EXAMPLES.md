# Perplexity API Examples

## Basic Query Processing

### POST /api/perplexity/query

Process a query and generate a formatted answer.

**Request:**
```json
{
  "query": "What is Python?",
  "search_results": [
    {
      "title": "Python Programming",
      "url": "https://python.org",
      "snippet": "Python is a programming language",
      "content": "Python is a high-level programming language..."
    }
  ],
  "use_llm": false,
  "include_metadata": true
}
```

**Response:**
```json
{
  "query": "What is Python?",
  "query_type": "coding",
  "answer": "Python is a high-level programming language[1]...",
  "metadata": {
    "current_date": "2025-05-13T04:31:29",
    "result_count": 1
  }
}
```

## Batch Processing

### POST /api/perplexity/batch

Process multiple queries at once.

**Request:**
```json
{
  "queries": [
    {
      "query": "What is Python?",
      "search_results": [...]
    },
    {
      "query": "What is JavaScript?",
      "search_results": [...]
    }
  ],
  "use_llm": false
}
```

**Response:**
```json
{
  "results": [
    {
      "query": "What is Python?",
      "query_type": "coding",
      "answer": "...",
      "success": true
    },
    {
      "query": "What is JavaScript?",
      "query_type": "coding",
      "answer": "...",
      "success": true
    }
  ],
  "total": 2
}
```

## Validation

### POST /api/perplexity/validate

Validate an answer against formatting rules.

**Request:**
```json
{
  "answer": "This is a test answer[1][2].",
  "query_type": "general"
}
```

**Response:**
```json
{
  "enabled": true,
  "valid": true,
  "issue_count": 0,
  "issues": []
}
```

## Cache Management

### GET /api/perplexity/cache/stats

Get cache statistics.

**Response:**
```json
{
  "enabled": true,
  "size": 150,
  "max_size": 1000,
  "ttl_seconds": 3600,
  "expired_entries": 5,
  "active_entries": 145
}
```

### POST /api/perplexity/cache/clear

Clear the cache.

**Response:**
```json
{
  "message": "Cache cleared successfully"
}
```

## Metrics

### GET /api/perplexity/metrics

Get metrics statistics.

**Response:**
```json
{
  "total_queries": 1250,
  "average_processing_time_ms": 245.5,
  "average_answer_length": 1250,
  "average_citation_count": 2.3,
  "query_type_distribution": {
    "coding": 450,
    "academic_research": 200,
    "general": 600
  },
  "error_rate": 0.02,
  "error_count": 25
}
```

### POST /api/perplexity/metrics/clear

Clear metrics.

**Response:**
```json
{
  "message": "Metrics cleared successfully"
}
```

## Query Types

### GET /api/perplexity/query-types

Get list of supported query types.

**Response:**
```json
{
  "query_types": [
    {
      "name": "academic_research",
      "description": "Academic research queries requiring detailed, scientific write-ups"
    },
    {
      "name": "coding",
      "description": "Programming and code-related queries"
    }
    // ... more types
  ]
}
```

## cURL Examples

### Basic Query
```bash
curl -X POST http://localhost:8024/api/perplexity/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "search_results": [{
      "title": "Python",
      "url": "https://python.org",
      "snippet": "Python is a programming language"
    }]
  }'
```

### Batch Processing
```bash
curl -X POST http://localhost:8024/api/perplexity/batch \
  -H "Content-Type: application/json" \
  -d '{
    "queries": [
      {"query": "Query 1", "search_results": []},
      {"query": "Query 2", "search_results": []}
    ]
  }'
```

### Get Metrics
```bash
curl http://localhost:8024/api/perplexity/metrics
```

### Clear Cache
```bash
curl -X POST http://localhost:8024/api/perplexity/cache/clear
```

## Python Client Examples

### Using requests
```python
import requests

# Process query
response = requests.post(
    "http://localhost:8024/api/perplexity/query",
    json={
        "query": "What is Python?",
        "search_results": [...]
    }
)
result = response.json()
print(result["answer"])

# Get metrics
metrics = requests.get("http://localhost:8024/api/perplexity/metrics").json()
print(f"Total queries: {metrics['total_queries']}")
```

### Using httpx (async)
```python
import httpx

async with httpx.AsyncClient() as client:
    # Process query
    response = await client.post(
        "http://localhost:8024/api/perplexity/query",
        json={
            "query": "What is Python?",
            "search_results": [...]
        }
    )
    result = response.json()
    print(result["answer"])
```

## Error Handling

### Error Response Format
```json
{
  "detail": "Error message here"
}
```

### Common Status Codes
- `200` - Success
- `400` - Bad Request (query processing error)
- `500` - Internal Server Error
- `503` - Service Unavailable (LLM provider error)

### Example Error Handling
```python
import requests

try:
    response = requests.post(
        "http://localhost:8024/api/perplexity/query",
        json={"query": "test"}
    )
    response.raise_for_status()
    result = response.json()
except requests.HTTPError as e:
    if e.response.status_code == 400:
        print("Query processing failed")
    elif e.response.status_code == 503:
        print("LLM service unavailable")
    else:
        print(f"Error: {e}")
```




