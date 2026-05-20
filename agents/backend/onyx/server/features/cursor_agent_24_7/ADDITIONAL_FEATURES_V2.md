# Additional Features V2 - Enhanced Perplexity System

## New Features Added

### 1. Response Caching System

A caching system has been added to improve performance by caching processed queries and formatted responses.

**Features:**
- Automatic caching of query results
- Configurable TTL (time-to-live)
- Automatic cache eviction when full
- Hash-based cache keys for efficient lookup

**Usage:**
```python
from core.perplexity import PerplexityProcessor

# Cache enabled by default
processor = PerplexityProcessor(enable_cache=True, cache_ttl=3600)

# Process query (will check cache first)
answer = await processor.generate_answer(processed, llm_provider, search_results)

# Get cache stats
stats = processor.cache.get_stats()
print(f"Cache size: {stats['size']}/{stats['max_size']}")
```

**Cache Configuration:**
- `enable_cache`: Enable/disable caching (default: True)
- `cache_ttl`: Time-to-live in seconds (default: 3600 = 1 hour)
- `max_size`: Maximum cache entries (default: 1000)

### 2. Metrics Collection System

A comprehensive metrics system tracks query processing performance and usage statistics.

**Features:**
- Query type distribution
- Average processing times
- Answer length statistics
- Citation count tracking
- Error rate monitoring

**Usage:**
```python
from core.perplexity import PerplexityProcessor

# Metrics enabled by default
processor = PerplexityProcessor(enable_metrics=True)

# Process queries (metrics automatically recorded)
answer = await processor.generate_answer(processed, llm_provider, search_results)

# Get overall stats
stats = processor.metrics.get_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Avg processing time: {stats['average_processing_time_ms']}ms")
print(f"Query type distribution: {stats['query_type_distribution']}")

# Get stats for specific query type
type_stats = processor.metrics.get_query_type_stats("academic_research")
print(f"Academic research queries: {type_stats['count']}")
```

**Metrics Tracked:**
- Query type
- Processing time (milliseconds)
- Search result count
- Answer length (characters)
- Citation count
- Error status

### 3. Utility Functions Module

A new utilities module provides common helper functions.

**Available Functions:**
- `extract_citations(text)` - Extract citation numbers from text
- `count_citations(text)` - Count total citations
- `remove_citations(text)` - Remove all citations
- `normalize_whitespace(text)` - Normalize whitespace
- `truncate_text(text, max_length)` - Truncate text
- `format_timestamp(timestamp)` - Format datetime as ISO string
- `parse_timestamp(timestamp_str)` - Parse ISO timestamp
- `sanitize_query(query)` - Sanitize query string
- `estimate_reading_time(text)` - Estimate reading time

**Usage:**
```python
from core.perplexity import utils

# Extract citations
citations = utils.extract_citations("Text here[1][2].")
# Returns: [1, 2]

# Count citations
count = utils.count_citations("Text[1][2][3].")
# Returns: 3

# Remove citations
clean_text = utils.remove_citations("Text[1][2].")
# Returns: "Text."

# Estimate reading time
minutes = utils.estimate_reading_time(long_text)
# Returns: estimated minutes
```

## Enhanced Processor

The `PerplexityProcessor` now includes:

### Cache Integration
- Automatic cache checking before processing
- Automatic cache storage after processing
- Cache statistics available

### Metrics Integration
- Automatic metrics recording
- Performance tracking
- Usage statistics

### Configuration Options
```python
processor = PerplexityProcessor(
    system_prompt_path="path/to/prompt.md",
    enable_validation=True,      # Response validation
    enable_cache=True,            # Response caching
    enable_metrics=True,          # Metrics collection
    cache_ttl=3600                # Cache TTL in seconds
)
```

## API Enhancements

### New Endpoints (to be added)

**GET `/api/perplexity/cache/stats`**
Get cache statistics.

**GET `/api/perplexity/metrics`**
Get metrics statistics.

**POST `/api/perplexity/cache/clear`**
Clear the cache.

**POST `/api/perplexity/metrics/clear`**
Clear metrics.

## Performance Improvements

1. **Caching**: Reduces redundant processing for similar queries
2. **Metrics**: Helps identify performance bottlenecks
3. **Utilities**: Reusable functions reduce code duplication

## Example Usage

```python
from core.perplexity import PerplexityProcessor

# Initialize with all features
processor = PerplexityProcessor(
    enable_validation=True,
    enable_cache=True,
    enable_metrics=True,
    cache_ttl=7200  # 2 hours
)

# Process query
processed = processor.process_query(query, search_results)

# Generate answer (uses cache if available, records metrics)
answer = await processor.generate_answer(
    processed,
    llm_provider,
    search_results
)

# Check cache stats
cache_stats = processor.cache.get_stats()
print(f"Cache hit rate: {cache_stats['active_entries']}/{cache_stats['size']}")

# Check metrics
metrics = processor.metrics.get_stats()
print(f"Average processing time: {metrics['average_processing_time_ms']}ms")
print(f"Most common query type: {max(metrics['query_type_distribution'].items(), key=lambda x: x[1])[0]}")
```

## Benefits

1. **Performance**: Caching reduces processing time for repeated queries
2. **Monitoring**: Metrics provide insights into system usage
3. **Debugging**: Utilities make common operations easier
4. **Scalability**: Better resource management with cache limits
5. **Analytics**: Query type distribution helps understand usage patterns

## Future Enhancements

Potential additions:
1. **Redis Cache Backend**: Distributed caching
2. **Prometheus Metrics**: Export metrics to Prometheus
3. **Query Analytics Dashboard**: Visualize metrics
4. **Smart Cache Warming**: Pre-cache common queries
5. **Performance Profiling**: Detailed timing breakdowns




