# 🔄 Synchronous vs Asynchronous Operations Guide

## Table of Contents

1. [Overview](#overview)
2. [When to Use `def` vs `async def`](#when-to-use-def-vs-async-def)
3. [Synchronous Operations](#synchronous-operations)
4. [Asynchronous Operations](#asynchronous-operations)
5. [Mixed Sync/Async Patterns](#mixed-syncasync-patterns)
6. [Performance Considerations](#performance-considerations)
7. [Best Practices](#best-practices)
8. [Common Patterns](#common-patterns)
9. [Error Handling](#error-handling)
10. [Testing Strategies](#testing-strategies)
11. [Real-World Examples](#real-world-examples)

## Overview

In FastAPI applications, understanding when to use `def` (synchronous) vs `async def` (asynchronous) is crucial for optimal performance and resource utilization.

### Key Principles

- **`def`**: Use for CPU-bound operations, data processing, calculations, validations
- **`async def`**: Use for I/O-bound operations, database queries, external API calls, file operations

## When to Use `def` vs `async def`

### Use `def` for Synchronous Operations

```python
# ✅ CPU-bound operations
def calculate_text_statistics(text: str) -> Dict[str, Any]:
    """Calculate text statistics (CPU-bound)."""
    words = text.split()
    return {
        "word_count": len(words),
        "character_count": len(text),
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
    }

# ✅ Data validation
def validate_text_content(text: str) -> Tuple[bool, List[str], List[str]]:
    """Validate text content (CPU-bound)."""
    errors = []
    warnings = []
    
    if not text:
        errors.append("Text content cannot be empty")
        return False, errors, warnings
    
    if len(text) > 10000:
        errors.append("Text content too long")
        return False, errors, warnings
    
    return True, errors, warnings

# ✅ Data transformation
def transform_analysis_to_response(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform database model to response format (CPU-bound)."""
    return {
        "id": analysis_data.get("id"),
        "text_content": analysis_data.get("text_content"),
        "analysis_type": analysis_data.get("analysis_type"),
        "status": analysis_data.get("status")
    }

# ✅ Cache key generation
def generate_cache_key(text: str, analysis_type: str, optimization_tier: str) -> str:
    """Generate cache key (CPU-bound)."""
    content = f"{text}:{analysis_type}:{optimization_tier}"
    hash_object = hashlib.sha256(content.encode())
    return f"analysis:{hash_object.hexdigest()}"
```

### Use `async def` for Asynchronous Operations

```python
# ✅ Database operations
async def create_analysis_async(request: TextAnalysisRequest, db_manager: Any) -> AnalysisResponse:
    """Create analysis in database (I/O-bound)."""
    analysis_data = {
        "text_content": request.text_content,
        "analysis_type": request.analysis_type,
        "optimization_tier": request.optimization_tier
    }
    
    analysis = await db_manager.create_text_analysis(analysis_data)
    return AnalysisResponse(**analysis)

# ✅ External API calls
async def call_external_api_async(url: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Call external API (I/O-bound)."""
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        return response.json()

# ✅ File operations
async def save_analysis_result_async(analysis_id: int, result: Dict[str, Any]) -> bool:
    """Save analysis result to file (I/O-bound)."""
    filename = f"analysis_{analysis_id}.json"
    async with aiofiles.open(filename, 'w') as f:
        await f.write(json.dumps(result, indent=2))
    return True

# ✅ Background tasks
async def process_analysis_background_async(
    analysis_id: int,
    text_content: str,
    analysis_type: str,
    db_manager: Any
):
    """Process analysis in background (I/O-bound)."""
    try:
        # Simulate processing
        await asyncio.sleep(2)
        
        # Update database
        update_data = {
            "status": "completed",
            "sentiment_score": 0.5,
            "processing_time_ms": 2000.0
        }
        
        await db_manager.update_text_analysis(analysis_id, update_data)
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
        await db_manager.update_text_analysis(analysis_id, {
            "status": "error",
            "error_message": str(e)
        })
```

## Synchronous Operations

### Characteristics

- **CPU-bound**: Operations that primarily use CPU resources
- **No I/O**: No network calls, database queries, or file operations
- **Fast execution**: Typically complete in milliseconds
- **Blocking**: Can block the event loop if too heavy

### Common Use Cases

```python
# Data validation
def validate_input_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate input data structure."""
    errors = []
    
    required_fields = ["text_content", "analysis_type"]
    for field in required_fields:
        if field not in data:
            errors.append(f"Missing required field: {field}")
    
    if data.get("text_content"):
        if len(data["text_content"]) > 10000:
            errors.append("Text content too long")
    
    return len(errors) == 0, errors

# Data processing
def calculate_metrics(data: List[float]) -> Dict[str, float]:
    """Calculate statistical metrics."""
    if not data:
        return {}
    
    return {
        "mean": sum(data) / len(data),
        "min": min(data),
        "max": max(data),
        "count": len(data)
    }

# Data transformation
def normalize_text(text: str) -> str:
    """Normalize text content."""
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

# Configuration management
@lru_cache(maxsize=100)
def get_config(environment: str) -> Dict[str, Any]:
    """Get configuration for environment."""
    configs = {
        "development": {
            "debug": True,
            "log_level": "DEBUG",
            "cache_ttl": 300
        },
        "production": {
            "debug": False,
            "log_level": "INFO",
            "cache_ttl": 3600
        }
    }
    return configs.get(environment, configs["development"])
```

## Asynchronous Operations

### Characteristics

- **I/O-bound**: Operations that wait for external resources
- **Non-blocking**: Don't block the event loop
- **Concurrent**: Can run multiple operations simultaneously
- **Resource efficient**: Better utilization of system resources

### Common Use Cases

```python
# Database operations
async def get_user_analyses_async(
    user_id: int,
    limit: int = 20,
    offset: int = 0,
    db_manager: Any
) -> Tuple[List[AnalysisResponse], int]:
    """Get user's analyses with pagination."""
    analyses, total_count = await db_manager.list_user_analyses(
        user_id=user_id,
        limit=limit,
        offset=offset,
        order_by="created_at",
        order_desc=True
    )
    
    response_analyses = []
    for analysis in analyses:
        response_data = transform_analysis_to_response(analysis)
        response_analyses.append(AnalysisResponse(**response_data))
    
    return response_analyses, total_count

# External API calls
async def analyze_text_with_external_api_async(
    text: str,
    api_url: str,
    api_key: str
) -> Dict[str, Any]:
    """Analyze text using external API."""
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"text": text, "analysis_type": "sentiment"}
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(api_url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()

# File operations
async def save_analysis_report_async(
    analysis_id: int,
    report_data: Dict[str, Any],
    file_path: str
) -> str:
    """Save analysis report to file."""
    filename = f"{file_path}/analysis_{analysis_id}_report.json"
    
    async with aiofiles.open(filename, 'w') as f:
        await f.write(json.dumps(report_data, indent=2, default=str))
    
    return filename

# Batch processing
async def process_batch_analyses_async(
    batch_id: int,
    texts: List[str],
    analysis_type: str,
    db_manager: Any
) -> Dict[str, Any]:
    """Process multiple analyses concurrently."""
    tasks = []
    
    for i, text in enumerate(texts):
        task = process_single_analysis_async(
            batch_id=batch_id,
            text_index=i,
            text=text,
            analysis_type=analysis_type,
            db_manager=db_manager
        )
        tasks.append(task)
    
    # Process all analyses concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Count results
    successful = sum(1 for r in results if not isinstance(r, Exception))
    failed = len(results) - successful
    
    return {
        "batch_id": batch_id,
        "total_texts": len(texts),
        "successful": successful,
        "failed": failed,
        "success_rate": (successful / len(texts)) * 100
    }
```

## Mixed Sync/Async Patterns

### Pattern 1: Sync Validation + Async Processing

```python
async def create_analysis_handler(
    request: TextAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: Any,
    cache: AnalysisCache
) -> AnalysisResponse:
    """Route handler with mixed sync/async operations."""
    
    # ✅ Synchronous validation
    is_valid, errors, warnings = validate_text_content(request.text_content)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {', '.join(errors)}"
        )
    
    # ✅ Synchronous cache key generation
    cache_key = generate_cache_key(
        request.text_content,
        request.analysis_type,
        request.optimization_tier
    )
    
    # ✅ Asynchronous cache check
    cached_result = await cache.get_async(cache_key)
    if cached_result:
        return AnalysisResponse(**cached_result)
    
    # ✅ Asynchronous database operation
    analysis = await create_analysis_async(request, db_manager)
    
    # ✅ Asynchronous cache storage
    response_data = analysis.model_dump()
    await cache.set_async(cache_key, response_data, ttl_seconds=1800)
    
    # ✅ Add background task
    background_tasks.add_task(
        process_analysis_background_async,
        analysis.id,
        request.text_content,
        request.analysis_type,
        db_manager
    )
    
    return analysis
```

### Pattern 2: Sync Data Processing + Async Storage

```python
async def process_and_store_analysis_async(
    text: str,
    analysis_type: str,
    db_manager: Any
) -> AnalysisResponse:
    """Process analysis data and store results."""
    
    # ✅ Synchronous data processing
    stats = calculate_text_statistics(text)
    complexity = calculate_complexity_score(text)
    priority = calculate_processing_priority("standard", len(text), analysis_type)
    
    # ✅ Synchronous response transformation
    analysis_data = {
        "text_content": text,
        "analysis_type": analysis_type,
        "text_statistics": stats,
        "complexity_score": complexity,
        "priority": priority,
        "status": "completed"
    }
    
    # ✅ Asynchronous database storage
    stored_analysis = await db_manager.create_text_analysis(analysis_data)
    
    # ✅ Synchronous response formatting
    response_data = transform_analysis_to_response(stored_analysis)
    
    return AnalysisResponse(**response_data)
```

### Pattern 3: Sync Configuration + Async Operations

```python
async def analyze_text_with_config_async(
    text: str,
    analysis_type: str,
    optimization_tier: str,
    db_manager: Any
) -> AnalysisResponse:
    """Analyze text using configuration-based approach."""
    
    # ✅ Synchronous configuration retrieval
    config = get_analysis_config(analysis_type, optimization_tier)
    estimated_time = estimate_processing_time(len(text), analysis_type, optimization_tier)
    
    # ✅ Asynchronous processing with config
    analysis_data = {
        "text_content": text,
        "analysis_type": analysis_type,
        "optimization_tier": optimization_tier,
        "config": config,
        "estimated_processing_time_ms": estimated_time
    }
    
    # ✅ Asynchronous database operation
    analysis = await db_manager.create_text_analysis(analysis_data)
    
    # ✅ Synchronous response transformation
    response_data = transform_analysis_to_response(analysis)
    
    return AnalysisResponse(**response_data)
```

## Performance Considerations

### Synchronous Operations

```python
# ✅ Fast operations (use def)
def quick_validation(data: Dict[str, Any]) -> bool:
    """Quick validation check."""
    return bool(data.get("required_field"))

# ✅ Cached operations (use def with @lru_cache)
@lru_cache(maxsize=1000)
def get_analysis_config(analysis_type: str, optimization_tier: str) -> Dict[str, Any]:
    """Get cached analysis configuration."""
    # Configuration lookup logic
    pass

# ❌ Avoid heavy CPU operations in request handlers
def heavy_calculation(data: List[float]) -> float:
    """Heavy calculation that could block the event loop."""
    # This could take too long and block other requests
    result = sum(x ** 2 for x in data) / len(data)
    return result
```

### Asynchronous Operations

```python
# ✅ I/O operations (use async def)
async def fetch_user_data_async(user_id: int, db_manager: Any) -> Dict[str, Any]:
    """Fetch user data from database."""
    return await db_manager.get_user(user_id)

# ✅ Concurrent operations
async def process_multiple_analyses_async(
    texts: List[str],
    analysis_type: str,
    db_manager: Any
) -> List[AnalysisResponse]:
    """Process multiple analyses concurrently."""
    tasks = [
        create_analysis_async(
            TextAnalysisRequest(text_content=text, analysis_type=analysis_type),
            db_manager
        )
        for text in texts
    ]
    
    return await asyncio.gather(*tasks)

# ✅ Background processing
async def process_analysis_background_async(
    analysis_id: int,
    text_content: str,
    analysis_type: str,
    db_manager: Any
):
    """Process analysis in background without blocking."""
    try:
        # Simulate processing time
        await asyncio.sleep(5)
        
        # Update database
        await db_manager.update_text_analysis(analysis_id, {
            "status": "completed",
            "processed_at": datetime.now()
        })
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
        await db_manager.update_text_analysis(analysis_id, {
            "status": "error",
            "error_message": str(e)
        })
```

## Best Practices

### 1. Clear Separation of Concerns

```python
# ✅ Good: Clear separation
def validate_input(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Synchronous validation."""
    # Validation logic
    pass

async def process_data_async(data: Dict[str, Any], db_manager: Any) -> Dict[str, Any]:
    """Asynchronous processing."""
    # Processing logic
    pass

async def handler(request: Request, db_manager: Any) -> Response:
    """Route handler combining sync and async operations."""
    # Validate synchronously
    is_valid, errors = validate_input(request.data)
    if not is_valid:
        raise HTTPException(status_code=400, detail=errors)
    
    # Process asynchronously
    result = await process_data_async(request.data, db_manager)
    return Response(data=result)
```

### 2. Proper Error Handling

```python
# ✅ Good: Proper error handling in async functions
async def safe_database_operation_async(db_manager: Any) -> Optional[Dict[str, Any]]:
    """Safe database operation with error handling."""
    try:
        result = await db_manager.get_data()
        return result
    except DatabaseConnectionError:
        logger.error("Database connection failed")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None

# ✅ Good: Error handling in sync functions
def safe_data_processing(data: Dict[str, Any]) -> Dict[str, Any]:
    """Safe data processing with error handling."""
    try:
        # Process data
        processed_data = transform_data(data)
        return processed_data
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return {"error": "Processing failed"}
```

### 3. Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor performance of sync/async operations."""
    
    def __init__(self):
        self._metrics: List[ProcessingMetrics] = []
        self._lock = asyncio.Lock()
    
    def start_operation_sync(self, operation_type: str) -> ProcessingMetrics:
        """Start monitoring synchronous operation."""
        return ProcessingMetrics(
            operation_type=operation_type,
            start_time=datetime.now()
        )
    
    async def start_operation_async(self, operation_type: str) -> ProcessingMetrics:
        """Start monitoring asynchronous operation."""
        async with self._lock:
            return self.start_operation_sync(operation_type)
    
    def complete_operation_sync(self, metrics: ProcessingMetrics, success: bool = True):
        """Complete monitoring synchronous operation."""
        metrics.complete(success)
        self._metrics.append(metrics)
    
    async def complete_operation_async(self, metrics: ProcessingMetrics, success: bool = True):
        """Complete monitoring asynchronous operation."""
        async with self._lock:
            self.complete_operation_sync(metrics, success)
    
    def get_stats_sync(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._metrics:
            return {}
        
        successful = [m for m in self._metrics if m.success]
        failed = [m for m in self._metrics if not m.success]
        
        return {
            "total_operations": len(self._metrics),
            "successful_operations": len(successful),
            "failed_operations": len(failed),
            "success_rate": len(successful) / len(self._metrics) * 100,
            "average_duration_ms": sum(m.duration_ms or 0 for m in self._metrics) / len(self._metrics)
        }
```

### 4. Caching Strategy

```python
class AnalysisCache:
    """Cache for analysis results."""
    
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    def get_sync(self, key: str) -> Optional[Any]:
        """Synchronous cache retrieval."""
        entry = self._cache.get(key)
        if not entry or entry.is_expired():
            if entry:
                del self._cache[key]
            return None
        
        entry.access()
        return entry.data
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Asynchronous cache retrieval."""
        async with self._lock:
            return self.get_sync(key)
    
    def set_sync(self, key: str, data: Any, ttl_seconds: int = 3600):
        """Synchronous cache storage."""
        self._cache[key] = CacheEntry(
            data=data,
            timestamp=datetime.now(),
            ttl_seconds=ttl_seconds
        )
    
    async def set_async(self, key: str, data: Any, ttl_seconds: int = 3600):
        """Asynchronous cache storage."""
        async with self._lock:
            self.set_sync(key, data, ttl_seconds)
```

## Common Patterns

### 1. Validation + Processing Pattern

```python
async def process_with_validation_async(
    data: Dict[str, Any],
    db_manager: Any
) -> Dict[str, Any]:
    """Process data with validation."""
    
    # Synchronous validation
    is_valid, errors = validate_data(data)
    if not is_valid:
        raise ValueError(f"Validation failed: {errors}")
    
    # Synchronous preprocessing
    processed_data = preprocess_data(data)
    
    # Asynchronous processing
    result = await process_data_async(processed_data, db_manager)
    
    # Synchronous post-processing
    final_result = postprocess_result(result)
    
    return final_result
```

### 2. Configuration + Execution Pattern

```python
async def execute_with_config_async(
    operation_type: str,
    data: Dict[str, Any],
    db_manager: Any
) -> Dict[str, Any]:
    """Execute operation with configuration."""
    
    # Synchronous configuration retrieval
    config = get_operation_config(operation_type)
    
    # Synchronous parameter validation
    validate_parameters(data, config)
    
    # Asynchronous execution
    result = await execute_operation_async(data, config, db_manager)
    
    # Synchronous result formatting
    formatted_result = format_result(result, config)
    
    return formatted_result
```

### 3. Batch Processing Pattern

```python
async def process_batch_async(
    items: List[Dict[str, Any]],
    db_manager: Any
) -> Dict[str, Any]:
    """Process batch of items."""
    
    # Synchronous batch validation
    valid_items = []
    errors = []
    
    for i, item in enumerate(items):
        is_valid, item_errors = validate_item(item)
        if is_valid:
            valid_items.append(item)
        else:
            errors.append(f"Item {i}: {item_errors}")
    
    # Synchronous batch preparation
    batch_config = prepare_batch_config(valid_items)
    
    # Asynchronous batch processing
    tasks = [
        process_item_async(item, batch_config, db_manager)
        for item in valid_items
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Synchronous result aggregation
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]
    
    return {
        "total_items": len(items),
        "valid_items": len(valid_items),
        "successful": len(successful),
        "failed": len(failed),
        "errors": errors,
        "results": successful
    }
```

## Error Handling

### Synchronous Error Handling

```python
def safe_sync_operation(data: Dict[str, Any]) -> Dict[str, Any]:
    """Safe synchronous operation with error handling."""
    try:
        # Validate input
        if not data:
            raise ValueError("Data cannot be empty")
        
        # Process data
        result = process_data(data)
        
        # Validate output
        if not result:
            raise ValueError("Processing failed to produce result")
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return {"error": str(e), "status": "validation_failed"}
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Internal processing error", "status": "error"}
```

### Asynchronous Error Handling

```python
async def safe_async_operation(data: Dict[str, Any], db_manager: Any) -> Dict[str, Any]:
    """Safe asynchronous operation with error handling."""
    try:
        # Asynchronous database operation
        result = await db_manager.process_data(data)
        
        if not result:
            raise ValueError("Database operation failed")
        
        return result
        
    except asyncio.TimeoutError:
        logger.error("Operation timed out")
        return {"error": "Operation timed out", "status": "timeout"}
        
    except DatabaseConnectionError:
        logger.error("Database connection failed")
        return {"error": "Database unavailable", "status": "database_error"}
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Internal error", "status": "error"}
```

### Mixed Error Handling

```python
async def mixed_operation_with_error_handling(
    data: Dict[str, Any],
    db_manager: Any
) -> Dict[str, Any]:
    """Mixed sync/async operation with comprehensive error handling."""
    
    try:
        # Synchronous validation
        is_valid, errors = validate_data(data)
        if not is_valid:
            return {"error": f"Validation failed: {errors}", "status": "validation_failed"}
        
        # Synchronous preprocessing
        processed_data = preprocess_data(data)
        
        # Asynchronous processing
        result = await process_data_async(processed_data, db_manager)
        
        # Synchronous post-processing
        final_result = postprocess_result(result)
        
        return final_result
        
    except ValueError as e:
        logger.error(f"Validation/preprocessing error: {e}")
        return {"error": str(e), "status": "processing_error"}
        
    except asyncio.TimeoutError:
        logger.error("Async operation timed out")
        return {"error": "Operation timed out", "status": "timeout"}
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"error": "Internal error", "status": "error"}
```

## Testing Strategies

### Testing Synchronous Operations

```python
class TestSynchronousOperations:
    """Test synchronous operations."""
    
    def test_validate_text_content_valid(self):
        """Test text validation with valid content."""
        is_valid, errors, warnings = validate_text_content("Valid text content")
        
        assert is_valid is True
        assert len(errors) == 0
        assert len(warnings) == 0
    
    def test_validate_text_content_empty(self):
        """Test text validation with empty content."""
        is_valid, errors, warnings = validate_text_content("")
        
        assert is_valid is False
        assert len(errors) == 1
        assert "empty" in errors[0].lower()
    
    def test_calculate_text_statistics(self):
        """Test text statistics calculation."""
        stats = calculate_text_statistics("This is a test text.")
        
        assert isinstance(stats, dict)
        assert "word_count" in stats
        assert "character_count" in stats
        assert stats["word_count"] == 5
        assert stats["character_count"] == 19
```

### Testing Asynchronous Operations

```python
class TestAsynchronousOperations:
    """Test asynchronous operations."""
    
    @pytest.mark.asyncio
    async def test_create_analysis_async(self):
        """Test asynchronous analysis creation."""
        mock_db = AsyncMock()
        mock_db.create_text_analysis.return_value = MOCK_ANALYSIS_DATA
        
        request = TextAnalysisRequest(
            text_content="Test text",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        
        result = await create_analysis_async(request, mock_db)
        
        assert isinstance(result, AnalysisResponse)
        assert result.text_content == "Test text"
        mock_db.create_text_analysis.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_analysis_async_not_found(self):
        """Test asynchronous analysis retrieval when not found."""
        mock_db = AsyncMock()
        mock_db.get_text_analysis.return_value = None
        
        result = await get_analysis_async(999, mock_db)
        
        assert result is None
```

### Testing Mixed Operations

```python
class TestMixedOperations:
    """Test mixed sync/async operations."""
    
    @pytest.mark.asyncio
    async def test_create_analysis_handler(self):
        """Test route handler with mixed operations."""
        mock_background_tasks = Mock()
        mock_db = AsyncMock()
        mock_db.create_text_analysis.return_value = MOCK_ANALYSIS_DATA
        cache = AnalysisCache()
        
        request = TextAnalysisRequest(
            text_content="Valid text content",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        
        result = await create_analysis_handler(
            request, mock_background_tasks, mock_db, cache
        )
        
        assert isinstance(result, AnalysisResponse)
        mock_background_tasks.add_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow with mixed operations."""
        cache = AnalysisCache()
        monitor = PerformanceMonitor()
        mock_db = AsyncMock()
        mock_db.create_text_analysis.return_value = MOCK_ANALYSIS_DATA
        
        # Start monitoring
        metrics = await monitor.start_operation_async("full_workflow")
        
        # Synchronous validation
        is_valid, errors, warnings = validate_text_content("Test text")
        assert is_valid is True
        
        # Synchronous statistics
        stats = calculate_text_statistics("Test text")
        assert len(stats) > 0
        
        # Asynchronous processing
        request = TextAnalysisRequest(
            text_content="Test text",
            analysis_type=AnalysisTypeEnum.SENTIMENT
        )
        analysis = await create_analysis_async(request, mock_db)
        assert isinstance(analysis, AnalysisResponse)
        
        # Complete monitoring
        await monitor.complete_operation_async(metrics, success=True)
        
        # Check performance stats
        stats = await monitor.get_stats_async()
        assert stats["total_operations"] == 1
        assert stats["successful_operations"] == 1
```

## Real-World Examples

### Example 1: Text Analysis API

```python
@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(
    request: TextAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: Any = Depends(get_db_manager),
    cache: AnalysisCache = Depends(get_cache)
) -> AnalysisResponse:
    """Analyze text with mixed sync/async operations."""
    
    # Synchronous validation
    is_valid, errors, warnings = validate_text_content(request.text_content)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation failed: {', '.join(errors)}"
        )
    
    # Synchronous cache key generation
    cache_key = generate_cache_key(
        request.text_content,
        request.analysis_type,
        request.optimization_tier
    )
    
    # Asynchronous cache check
    cached_result = await cache.get_async(cache_key)
    if cached_result:
        return AnalysisResponse(**cached_result)
    
    # Synchronous priority calculation
    priority = calculate_processing_priority(
        request.optimization_tier,
        len(request.text_content),
        request.analysis_type
    )
    
    # Asynchronous database operation
    analysis = await create_analysis_async(request, db_manager)
    
    # Asynchronous cache storage
    response_data = analysis.model_dump()
    await cache.set_async(cache_key, response_data, ttl_seconds=1800)
    
    # Add background processing
    background_tasks.add_task(
        process_analysis_background_async,
        analysis.id,
        request.text_content,
        request.analysis_type,
        db_manager
    )
    
    return analysis
```

### Example 2: Batch Processing Service

```python
@router.post("/batch", response_model=BatchAnalysisResponse)
async def create_batch_analysis(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    db_manager: Any = Depends(get_db_manager)
) -> BatchAnalysisResponse:
    """Create batch analysis with mixed operations."""
    
    # Synchronous batch validation
    for i, text in enumerate(request.texts):
        is_valid, errors, warnings = validate_text_content(text)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid text at index {i}: {', '.join(errors)}"
            )
    
    # Synchronous batch statistics
    total_texts = len(request.texts)
    total_chars = sum(len(text) for text in request.texts)
    avg_text_length = total_chars / total_texts if total_texts > 0 else 0
    
    # Asynchronous batch creation
    batch = await create_batch_analysis_async(request, db_manager)
    
    # Add background batch processing
    background_tasks.add_task(
        process_batch_texts_async,
        batch.id,
        request.texts,
        request.analysis_type,
        db_manager
    )
    
    return batch
```

### Example 3: Performance Monitoring Middleware

```python
class PerformanceMiddleware:
    """Middleware for performance monitoring."""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
    
    async def __call__(self, request: Request, call_next):
        # Start monitoring
        metrics = await self.monitor.start_operation_async(f"http_{request.method}")
        
        try:
            # Process request
            response = await call_next(request)
            
            # Complete monitoring
            await self.monitor.complete_operation_async(metrics, success=True)
            
            return response
            
        except Exception as e:
            # Complete monitoring with error
            await self.monitor.complete_operation_async(
                metrics, success=False, error_message=str(e)
            )
            raise
```

## Summary

### Key Takeaways

1. **Use `def` for CPU-bound operations**: Data validation, calculations, transformations, configuration management
2. **Use `async def` for I/O-bound operations**: Database queries, external API calls, file operations, background tasks
3. **Combine both effectively**: Use synchronous operations for validation and preprocessing, asynchronous for I/O operations
4. **Monitor performance**: Track execution times and success rates for both sync and async operations
5. **Handle errors properly**: Different error handling strategies for sync vs async operations
6. **Test comprehensively**: Test both sync and async operations, including mixed patterns
7. **Cache strategically**: Use caching to improve performance for frequently accessed data
8. **Use background tasks**: Offload heavy processing to background tasks to avoid blocking

### Best Practices Checklist

- [ ] Use `def` for CPU-bound operations (validation, calculations, transformations)
- [ ] Use `async def` for I/O-bound operations (database, APIs, files)
- [ ] Validate data synchronously before async processing
- [ ] Cache results to improve performance
- [ ] Monitor performance of both sync and async operations
- [ ] Handle errors appropriately for each operation type
- [ ] Use background tasks for heavy processing
- [ ] Test both sync and async operations thoroughly
- [ ] Document the purpose of each function (sync vs async)
- [ ] Use proper logging for debugging and monitoring

This guide provides a comprehensive approach to using `def` and `async def` effectively in FastAPI applications, ensuring optimal performance and maintainability. 