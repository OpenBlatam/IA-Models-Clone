# Perplexity Query Processing System

A comprehensive, modular system for processing queries in Perplexity style with proper formatting, citations, and query type detection.

## Quick Start

```python
from core.perplexity import PerplexityService

service = PerplexityService()
result = await service.answer_query(
    query="What is Python?",
    search_results=[...]
)
```

## Architecture

### Core Components

- **QueryTypeDetector** - Detects query types from patterns
- **ResponseFormatter** - Formats answers per Perplexity guidelines
- **CitationManager** - Manages citation matching and formatting
- **PromptBuilder** - Builds LLM prompts from system prompt
- **PerplexityProcessor** - Main orchestrator
- **PerplexityValidator** - Validates responses
- **PerplexityService** - High-level service interface

### Infrastructure

- **PerplexityCache** - Response caching
- **PerplexityMetrics** - Metrics collection
- **PerplexityConfig** - Configuration management
- **Rate Limiting** - Token bucket and sliding window
- **Middleware** - Timing, error handling, metrics

## Module Overview

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `types.py` | Data models | QueryType, SearchResult, ProcessedQuery |
| `detector.py` | Type detection | QueryTypeDetector |
| `citations.py` | Citation management | CitationManager |
| `formatter.py` | Response formatting | ResponseFormatter |
| `prompt_builder.py` | Prompt construction | PromptBuilder |
| `processor.py` | Main processor | PerplexityProcessor |
| `validator.py` | Response validation | PerplexityValidator |
| `cache.py` | Caching | PerplexityCache |
| `metrics.py` | Metrics | PerplexityMetrics |
| `service.py` | Service layer | PerplexityService |
| `config.py` | Configuration | PerplexityConfig |
| `exceptions.py` | Custom exceptions | PerplexityError, etc. |
| `middleware.py` | Middleware | RateLimiter, decorators |
| `rate_limiter.py` | Rate limiting | TokenBucketRateLimiter |
| `utils.py` | Utilities | Helper functions |
| `helpers.py` | Helpers | Common helpers |
| `constants.py` | Constants | System constants |

## Usage Patterns

### Basic Usage
```python
from core.perplexity import PerplexityService

service = PerplexityService()
result = await service.answer_query("query", search_results)
```

### With Configuration
```python
from core.perplexity import PerplexityConfig, PerplexityService

config = PerplexityConfig.from_env()
service = PerplexityService(config=config)
```

### Batch Processing
```python
results = await service.batch_process([
    {'query': 'Query 1', 'search_results': [...]},
    {'query': 'Query 2', 'search_results': [...]}
])
```

### Direct Processor
```python
from core.perplexity import PerplexityProcessor

processor = PerplexityProcessor()
processed = processor.process_query(query, search_results)
answer = await processor.generate_answer(processed, llm_provider)
```

## Configuration

### Environment Variables
- `PERPLEXITY_SYSTEM_PROMPT_PATH`
- `PERPLEXITY_ENABLE_VALIDATION`
- `PERPLEXITY_ENABLE_CACHE`
- `PERPLEXITY_ENABLE_METRICS`
- `PERPLEXITY_CACHE_TTL`
- `PERPLEXITY_CACHE_MAX_SIZE`
- `PERPLEXITY_MAX_SEARCH_RESULTS`
- `PERPLEXITY_DEFAULT_LLM_MODEL`

## Features

- ✅ 11 query types supported
- ✅ Automatic citation matching
- ✅ LaTeX math normalization
- ✅ Response validation
- ✅ Caching system
- ✅ Metrics collection
- ✅ Rate limiting
- ✅ Batch processing
- ✅ Error handling
- ✅ Configuration management

## Testing

```bash
# Run tests
pytest tests/test_perplexity.py

# Quick test
python scripts/test_perplexity.py
```

## Documentation

- `SYSTEM_PROMPT.md` - System prompt
- `API_EXAMPLES.md` - API usage examples
- `REFACTORING_FINAL.md` - Refactoring details
- `FEATURES_COMPLETE.md` - Complete features list

## License

Part of Blatam Academy project.




