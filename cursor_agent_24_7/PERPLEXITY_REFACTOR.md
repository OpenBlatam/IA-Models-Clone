# Perplexity-Style Refactoring Summary

## Overview

The `cursor_agent_24_7` has been refactored to support Perplexity-style query processing, which focuses on answering queries using search results with proper citations and formatting.

## Changes Made

### 1. System Prompt Update

**File:** `SYSTEM_PROMPT.md`

- Replaced the Kiro IDE assistant prompt with a Perplexity-style search assistant prompt
- New prompt includes:
  - Goal: Answer queries using search results with citations
  - Format rules: Markdown formatting, citation guidelines, query type handling
  - Query types: Academic research, news, weather, people, coding, recipes, translation, creative writing, math, URL lookup
  - Restrictions: No moralization, no emojis, proper citation format

### 2. Core Components

**File:** `core/perplexity_processor.py`

Created a new module with the following components:

#### QueryTypeDetector
- Detects query types based on patterns and keywords
- Supports 11 query types (academic, news, weather, people, coding, etc.)
- Uses regex patterns to identify query intent

#### ResponseFormatter
- Formats responses according to Perplexity guidelines
- Adds citations to content
- Applies query-type-specific formatting rules
- Handles markdown formatting, tables, lists, and code blocks

#### PerplexityProcessor
- Main processor that orchestrates query processing
- Converts search results to structured format
- Processes queries and formats responses
- Manages query metadata

### 3. API Routes

**File:** `api/routes/perplexity_routes.py`

Created new API endpoints:

- `POST /api/perplexity/query` - Process a query and generate a formatted answer
- `POST /api/perplexity/process` - Process a query and return metadata only
- `GET /api/perplexity/query-types` - Get list of supported query types

### 4. Integration

**Files Modified:**
- `core/__init__.py` - Added exports for new Perplexity components
- `api/app_config.py` - Registered Perplexity routes
- `api/routes/__init__.py` - Added Perplexity router to exports

## Usage

### API Endpoint Example

```python
import requests

# Process a query with search results
response = requests.post(
    "http://localhost:8024/api/perplexity/query",
    json={
        "query": "What is the weather in New York?",
        "search_results": [
            {
                "title": "Weather Forecast",
                "url": "https://example.com/weather",
                "snippet": "Sunny, 72°F",
                "content": "Full weather details..."
            }
        ],
        "include_metadata": True
    }
)

result = response.json()
print(result["answer"])  # Formatted answer with citations
print(result["query_type"])  # "weather"
```

### Programmatic Usage

```python
from core.perplexity_processor import PerplexityProcessor

processor = PerplexityProcessor()

# Process a query
processed = processor.process_query(
    query="Explain quantum computing",
    search_results=[
        {
            "title": "Quantum Computing Guide",
            "url": "https://example.com",
            "snippet": "Quantum computing uses qubits...",
            "content": "Full content..."
        }
    ]
)

# Format a response
answer = processor.format_response(
    processed_query=processed,
    answer_content="Quantum computing is a revolutionary technology..."
)
```

## Query Types Supported

1. **Academic Research** - Detailed scientific write-ups
2. **Recent News** - News summaries with timestamps
3. **Weather** - Weather forecasts (short format)
4. **People** - Biographical information
5. **Coding** - Programming queries (code first, then explanation)
6. **Cooking Recipes** - Step-by-step recipes with ingredients
7. **Translation** - Language translation (no citations)
8. **Creative Writing** - Creative content (no citations)
9. **Science/Math** - Mathematical calculations
10. **URL Lookup** - URL content summarization
11. **General** - Default for other queries

## Features

- **Automatic Query Type Detection** - Detects query type from content
- **Citation Management** - Adds proper citations to answers
- **Format-Specific Responses** - Different formatting for different query types
- **Search Result Integration** - Processes and uses search results effectively
- **RESTful API** - Easy integration via HTTP endpoints

## Enhancements Completed

The implementation has been enhanced with:

1. **Intelligent Citation System** - Automatic citation matching using keyword and phrase analysis
2. **Prompt Builder** - Loads SYSTEM_PROMPT.md and builds complete LLM prompts
3. **LLM Integration** - Supports multiple LLM providers (LLMPipeline, OpenAI, custom)
4. **Enhanced API** - New `/prompt` endpoint for debugging and custom integrations
5. **Async Support** - Full async/await support for LLM providers

## LLM Integration

The system now supports LLM integration:

```python
# With LLM provider
answer = await processor.generate_answer(processed, llm_provider)

# Get prompt for custom LLM integration
prompt = processor.build_llm_prompt(processed)
```

See `PERPLEXITY_ENHANCEMENTS.md` for detailed usage examples.

## Testing

Test the endpoints using:

```bash
# Start the API server
python -m agents.backend.onyx.server.features.cursor_agent_24_7.main --mode api

# Test query processing
curl -X POST http://localhost:8024/api/perplexity/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "search_results": [{
      "title": "Python Programming",
      "url": "https://python.org",
      "snippet": "Python is a programming language"
    }]
  }'
```

## Notes

- **Citation System**: Now uses intelligent keyword and phrase matching
- **LLM Integration**: Fully integrated with support for multiple providers
- **Prompt Building**: Automatically loads and uses SYSTEM_PROMPT.md
- **API Endpoints**: Complete REST API with `/query`, `/prompt`, `/process`, and `/query-types`
- **Async Support**: Full async/await support for better performance

## Documentation

- `PERPLEXITY_REFACTOR.md` - Initial refactoring summary (this file)
- `PERPLEXITY_ENHANCEMENTS.md` - Detailed enhancement documentation
- `SYSTEM_PROMPT.md` - Complete Perplexity-style system prompt

