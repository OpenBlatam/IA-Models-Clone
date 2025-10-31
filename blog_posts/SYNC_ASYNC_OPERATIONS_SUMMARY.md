# 🔄 Synchronous vs Asynchronous Operations Summary

## Quick Reference

### When to Use `def` (Synchronous)
- **CPU-bound operations**: Calculations, data processing, validations
- **Fast operations**: < 100ms execution time
- **No I/O**: No database, network, or file operations
- **Data transformation**: Converting between formats
- **Configuration**: Loading and managing settings

### When to Use `async def` (Asynchronous)
- **I/O-bound operations**: Database queries, API calls, file operations
- **Slow operations**: > 100ms execution time
- **External resources**: Network requests, database connections
- **Background tasks**: Long-running processes
- **Concurrent operations**: Multiple operations at once

## Key Patterns

### 1. Validation + Processing Pattern
```python
async def process_with_validation(data: Dict[str, Any], db_manager: Any) -> Dict[str, Any]:
    # ✅ Synchronous validation
    is_valid, errors = validate_data(data)
    if not is_valid:
        raise ValueError(f"Validation failed: {errors}")
    
    # ✅ Asynchronous processing
    result = await process_data_async(data, db_manager)
    
    return result
```

### 2. Configuration + Execution Pattern
```python
async def execute_with_config(operation_type: str, data: Dict[str, Any], db_manager: Any) -> Dict[str, Any]:
    # ✅ Synchronous configuration
    config = get_operation_config(operation_type)
    
    # ✅ Asynchronous execution
    result = await execute_operation_async(data, config, db_manager)
    
    return result
```

### 3. Cache + Database Pattern
```python
async def get_data_with_cache(key: str, db_manager: Any, cache: AnalysisCache) -> Dict[str, Any]:
    # ✅ Synchronous cache key generation
    cache_key = generate_cache_key(key)
    
    # ✅ Asynchronous cache check
    cached_result = await cache.get_async(cache_key)
    if cached_result:
        return cached_result
    
    # ✅ Asynchronous database query
    result = await db_manager.get_data(key)
    
    # ✅ Asynchronous cache storage
    await cache.set_async(cache_key, result, ttl_seconds=3600)
    
    return result
```

## Best Practices

### ✅ Do This
```python
# Fast validation (use def)
def validate_text(text: str) -> bool:
    return len(text) > 0 and len(text) <= 10000

# Database operations (use async def)
async def save_analysis(analysis_data: Dict[str, Any], db_manager: Any) -> int:
    return await db_manager.create_analysis(analysis_data)

# Mixed operations (combine both)
async def process_analysis(text: str, db_manager: Any) -> Dict[str, Any]:
    # Synchronous validation
    if not validate_text(text):
        raise ValueError("Invalid text")
    
    # Asynchronous processing
    analysis_id = await save_analysis({"text": text}, db_manager)
    
    return {"id": analysis_id, "status": "completed"}
```

### ❌ Don't Do This
```python
# Don't use async for CPU-bound operations
async def calculate_statistics(text: str) -> Dict[str, Any]:  # ❌ Wrong
    words = text.split()
    return {"word_count": len(words)}

# Don't use sync for I/O operations
def save_to_database(data: Dict[str, Any], db_manager: Any) -> bool:  # ❌ Wrong
    return db_manager.save(data)  # This blocks the event loop

# Don't mix sync/async incorrectly
async def bad_mixed_function(text: str, db_manager: Any) -> Dict[str, Any]:  # ❌ Wrong
    # Don't call sync functions that might block
    result = heavy_cpu_operation(text)  # This could block
    
    # Don't forget to await async operations
    db_result = db_manager.save(result)  # Missing await
    
    return db_result
```

## Performance Guidelines

### Synchronous Operations
- **Target**: < 10ms execution time
- **Use cases**: Validation, simple calculations, data transformation
- **Caching**: Use `@lru_cache` for expensive operations
- **Monitoring**: Track execution time to identify bottlenecks

### Asynchronous Operations
- **Target**: Handle multiple concurrent operations
- **Use cases**: Database queries, API calls, file operations
- **Concurrency**: Use `asyncio.gather()` for parallel operations
- **Error handling**: Implement proper exception handling

### Mixed Operations
- **Pattern**: Validate synchronously, process asynchronously
- **Caching**: Cache results of expensive operations
- **Background tasks**: Use for long-running operations
- **Monitoring**: Track both sync and async performance

## Error Handling

### Synchronous Error Handling
```python
def safe_sync_operation(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        # Process data
        result = process_data(data)
        return result
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Processing failed"}
```

### Asynchronous Error Handling
```python
async def safe_async_operation(data: Dict[str, Any], db_manager: Any) -> Dict[str, Any]:
    try:
        # Process data
        result = await db_manager.process_data(data)
        return result
    except asyncio.TimeoutError:
        logger.error("Operation timed out")
        return {"error": "Timeout"}
    except DatabaseConnectionError:
        logger.error("Database connection failed")
        return {"error": "Database unavailable"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Internal error"}
```

## Testing Strategies

### Testing Synchronous Functions
```python
def test_validate_text():
    # Test valid input
    assert validate_text("Valid text") is True
    
    # Test invalid input
    assert validate_text("") is False
    assert validate_text("A" * 15000) is False
```

### Testing Asynchronous Functions
```python
@pytest.mark.asyncio
async def test_save_analysis():
    mock_db = AsyncMock()
    mock_db.create_analysis.return_value = 123
    
    result = await save_analysis({"text": "test"}, mock_db)
    
    assert result == 123
    mock_db.create_analysis.assert_called_once()
```

### Testing Mixed Operations
```python
@pytest.mark.asyncio
async def test_process_analysis():
    mock_db = AsyncMock()
    mock_db.create_analysis.return_value = 123
    
    result = await process_analysis("Valid text", mock_db)
    
    assert result["id"] == 123
    assert result["status"] == "completed"
```

## Common Use Cases

### 1. API Route Handler
```python
@router.post("/analyze")
async def analyze_text(
    request: TextAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: Any = Depends(get_db_manager)
) -> AnalysisResponse:
    # Synchronous validation
    is_valid, errors = validate_text_content(request.text_content)
    if not is_valid:
        raise HTTPException(status_code=400, detail=errors)
    
    # Asynchronous processing
    analysis = await create_analysis_async(request, db_manager)
    
    # Background task
    background_tasks.add_task(process_background_async, analysis.id)
    
    return analysis
```

### 2. Data Processing Pipeline
```python
async def process_data_pipeline(data: List[Dict[str, Any]], db_manager: Any) -> List[Dict[str, Any]]:
    # Synchronous preprocessing
    valid_data = []
    for item in data:
        if validate_item(item):
            processed_item = preprocess_item(item)
            valid_data.append(processed_item)
    
    # Asynchronous processing
    tasks = [process_item_async(item, db_manager) for item in valid_data]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Synchronous post-processing
    final_results = []
    for result in results:
        if not isinstance(result, Exception):
            final_result = postprocess_result(result)
            final_results.append(final_result)
    
    return final_results
```

### 3. Caching Strategy
```python
async def get_analysis_with_cache(
    analysis_id: int,
    db_manager: Any,
    cache: AnalysisCache
) -> Optional[AnalysisResponse]:
    # Synchronous cache key generation
    cache_key = f"analysis:{analysis_id}"
    
    # Asynchronous cache check
    cached_result = await cache.get_async(cache_key)
    if cached_result:
        return AnalysisResponse(**cached_result)
    
    # Asynchronous database query
    analysis = await db_manager.get_analysis(analysis_id)
    if not analysis:
        return None
    
    # Synchronous response transformation
    response_data = transform_analysis_to_response(analysis)
    
    # Asynchronous cache storage
    await cache.set_async(cache_key, response_data, ttl_seconds=3600)
    
    return AnalysisResponse(**response_data)
```

## Performance Monitoring

### Metrics to Track
- **Synchronous operations**: Execution time, success rate
- **Asynchronous operations**: Execution time, concurrency, success rate
- **Cache performance**: Hit rate, miss rate, eviction rate
- **Database operations**: Query time, connection pool usage
- **Error rates**: By operation type and error category

### Monitoring Implementation
```python
class PerformanceMonitor:
    def __init__(self):
        self._metrics: List[ProcessingMetrics] = []
        self._lock = asyncio.Lock()
    
    def start_operation_sync(self, operation_type: str) -> ProcessingMetrics:
        return ProcessingMetrics(operation_type=operation_type, start_time=datetime.now())
    
    async def start_operation_async(self, operation_type: str) -> ProcessingMetrics:
        async with self._lock:
            return self.start_operation_sync(operation_type)
    
    def complete_operation_sync(self, metrics: ProcessingMetrics, success: bool = True):
        metrics.complete(success)
        self._metrics.append(metrics)
    
    async def complete_operation_async(self, metrics: ProcessingMetrics, success: bool = True):
        async with self._lock:
            self.complete_operation_sync(metrics, success)
    
    def get_stats_sync(self) -> Dict[str, Any]:
        if not self._metrics:
            return {}
        
        successful = [m for m in self._metrics if m.success]
        return {
            "total_operations": len(self._metrics),
            "successful_operations": len(successful),
            "success_rate": len(successful) / len(self._metrics) * 100,
            "average_duration_ms": sum(m.duration_ms or 0 for m in self._metrics) / len(self._metrics)
        }
```

## Checklist for Implementation

### Before Implementation
- [ ] Identify operation type (CPU-bound vs I/O-bound)
- [ ] Estimate execution time
- [ ] Consider concurrency requirements
- [ ] Plan error handling strategy
- [ ] Design caching strategy if applicable

### During Implementation
- [ ] Use `def` for CPU-bound operations
- [ ] Use `async def` for I/O-bound operations
- [ ] Implement proper error handling
- [ ] Add performance monitoring
- [ ] Write comprehensive tests

### After Implementation
- [ ] Monitor performance metrics
- [ ] Optimize based on real-world usage
- [ ] Update documentation
- [ ] Review and refactor as needed

## Key Takeaways

1. **Use `def` for fast, CPU-bound operations** (validation, calculations, transformations)
2. **Use `async def` for I/O-bound operations** (database, APIs, files)
3. **Combine both effectively** for optimal performance
4. **Monitor performance** of both sync and async operations
5. **Handle errors appropriately** for each operation type
6. **Test thoroughly** including mixed patterns
7. **Cache strategically** to improve performance
8. **Use background tasks** for heavy processing

## Related Files

- `sync_async_operations.py` - Main implementation
- `test_sync_async_operations.py` - Comprehensive test suite
- `SYNC_ASYNC_OPERATIONS_GUIDE.md` - Detailed guide
- `functional_fastapi_components.py` - Supporting components
- `declarative_routes.py` - Route implementations

This summary provides a quick reference for implementing synchronous and asynchronous operations effectively in FastAPI applications, ensuring optimal performance and maintainability. 