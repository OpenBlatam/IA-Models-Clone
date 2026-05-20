# Structured Routing Guide for Instagram Captions API v14.0

## Overview

This guide documents the well-structured routing system implemented in the Instagram Captions API v14.0, focusing on clear dependency injection, consistent patterns, and maintainable code organization.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Dependency Injection System](#dependency-injection-system)
3. [Route Factory Pattern](#route-factory-pattern)
4. [Route Organization](#route-organization)
5. [Best Practices](#best-practices)
6. [Examples and Patterns](#examples-and-patterns)
7. [Testing Strategy](#testing-strategy)
8. [Performance Considerations](#performance-considerations)

## Architecture Overview

### Key Principles

1. **Clear Separation of Concerns**: Routes, dependencies, and business logic are clearly separated
2. **Dependency Injection**: Centralized dependency management with clear interfaces
3. **Factory Pattern**: Route creation using factory pattern for consistency
4. **Registry Pattern**: Central router registry for easy management
5. **Builder Pattern**: Complex route configuration using builder pattern

### Directory Structure

```
routes/
├── __init__.py              # Router registry and exports
├── factory.py               # Route factory and builder patterns
├── structured_captions.py   # Well-structured captions routes
├── captions.py              # Legacy captions routes
├── performance.py           # Performance monitoring routes
├── async_flow_routes.py     # Async flow management
├── enhanced_async_routes.py # Enhanced async operations
├── shared_resources_routes.py # Shared resources management
└── lazy_loading_routes.py   # Lazy loading operations

dependencies/
└── __init__.py              # Centralized dependency injection

main_structured.py           # Main application with structured routing
```

## Dependency Injection System

### Core Dependency Classes

#### ServiceDependencies
Complete dependency set for complex operations:
```python
class ServiceDependencies:
    def __init__(
        self,
        user: Dict[str, Any] = Depends(require_authentication),
        db_pool = Depends(get_database_pool),
        api_client = Depends(get_api_client_pool),
        ai_engine = Depends(get_optimized_engine),
        cache_manager = Depends(get_cache_manager),
        lazy_loader = Depends(get_lazy_loader_manager),
        io_monitor = Depends(get_io_monitor),
        blocking_limiter = Depends(get_blocking_limiter)
    ):
        self.user = user
        self.db_pool = db_pool
        self.api_client = api_client
        self.ai_engine = ai_engine
        self.cache_manager = cache_manager
        self.lazy_loader = lazy_loader
        self.io_monitor = io_monitor
        self.blocking_limiter = blocking_limiter
```

#### CoreDependencies
Basic dependencies for simple operations:
```python
class CoreDependencies:
    def __init__(
        self,
        user: Dict[str, Any] = Depends(require_authentication),
        ai_engine = Depends(get_optimized_engine),
        cache_manager = Depends(get_cache_manager)
    ):
        self.user = user
        self.ai_engine = ai_engine
        self.cache_manager = cache_manager
```

#### AdvancedDependencies
Advanced dependencies for complex operations:
```python
class AdvancedDependencies:
    def __init__(
        self,
        user: Dict[str, Any] = Depends(require_authentication),
        db_pool = Depends(get_database_pool),
        api_client = Depends(get_api_client_pool),
        ai_engine = Depends(get_optimized_engine),
        cache_manager = Depends(get_cache_manager),
        lazy_loader = Depends(get_lazy_loader_manager),
        io_monitor = Depends(get_io_monitor)
    ):
        self.user = user
        self.db_pool = db_pool
        self.api_client = api_client
        self.ai_engine = ai_engine
        self.cache_manager = cache_manager
        self.lazy_loader = lazy_loader
        self.io_monitor = io_monitor
```

### Authentication Dependencies

```python
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    api_key = credentials.credentials
    user_id = api_key[:16]
    
    user_data = {
        "id": user_id,
        "api_key": api_key,
        "permissions": ["read", "write"],
        "rate_limit": 1000
    }
    
    request.state.user = user_data
    return user_data

async def require_permission(
    permission: str,
    user: Dict[str, Any] = Depends(require_authentication)
) -> Dict[str, Any]:
    """Require specific permission for endpoint access"""
    if permission not in user.get("permissions", []):
        raise HTTPException(
            status_code=403,
            detail=f"Permission '{permission}' required"
        )
    return user
```

## Route Factory Pattern

### RouterRegistry

Central registry for managing all application routes:

```python
class RouterRegistry:
    def __init__(self):
        self.routers: List[APIRouter] = []
        self.router_configs: dict = {}
    
    def register_router(
        self, 
        router: APIRouter, 
        prefix: str = "", 
        tags: List[str] = None,
        dependencies: List = None,
        description: str = ""
    ):
        """Register a router with configuration"""
        self.routers.append(router)
        self.router_configs[router] = {
            "prefix": prefix,
            "tags": tags or [],
            "dependencies": dependencies or [],
            "description": description
        }
```

### RouteBuilder

Builder pattern for creating complex routes:

```python
class RouteBuilder:
    def __init__(self, router: APIRouter):
        self.router = router
        self.dependencies: List = []
        self.decorators: List[Callable] = []
        self.tags: List[str] = []
        self.description: str = ""
    
    def with_dependencies(self, *deps) -> 'RouteBuilder':
        """Add dependencies to route"""
        self.dependencies.extend(deps)
        return self
    
    def with_decorators(self, *decorators) -> 'RouteBuilder':
        """Add decorators to route"""
        self.decorators.extend(decorators)
        return self
    
    def with_tags(self, *tags) -> 'RouteBuilder':
        """Add tags to route"""
        self.tags.extend(tags)
        return self
    
    def with_description(self, description: str) -> 'RouteBuilder':
        """Add description to route"""
        self.description = description
        return self
    
    def build_route(self, path: str, methods: List[str] = None):
        """Build route with all configurations"""
        def decorator(func: Callable) -> Callable:
            # Apply decorators and add to router
            decorated_func = func
            for decorator in self.decorators:
                decorated_func = decorator(decorated_func)
            
            self.router.add_api_route(
                path=path,
                endpoint=decorated_func,
                dependencies=self.dependencies,
                tags=self.tags,
                description=self.description
            )
            return decorated_func
        return decorator
```

## Route Organization

### Structured Captions Routes

Example of well-structured route implementation:

```python
@router.post("/generate", response_model=StructuredCaptionResponse)
@limit_blocking_operations(
    operation_type=OperationType.CAPTION_GENERATION,
    identifier="structured_caption_generation",
    user_id_param="user_id"
)
async def generate_structured_caption(
    request: StructuredCaptionRequest,
    deps: CoreDependencies = Depends(),
    request_context: Request = Depends()
) -> StructuredCaptionResponse:
    """
    Generate structured caption with clear dependencies
    
    Demonstrates:
    - Clear dependency injection
    - Input validation
    - Error handling
    - Performance monitoring
    """
    
    start_time = time.time()
    
    try:
        # Validate content length
        validated_content = await validate_content_length(
            request.content_description, 
            max_length=1000
        )
        
        # Generate caption using AI engine
        caption_result = await deps.ai_engine.generate_caption_optimized(
            content_description=validated_content,
            style=request.style,
            tone=request.tone,
            hashtag_count=request.hashtag_count,
            language=request.language
        )
        
        # Process result
        processing_time = time.time() - start_time
        
        # Create response
        response = StructuredCaptionResponse(
            caption=caption_result.caption,
            hashtags=caption_result.hashtags,
            style=request.style,
            tone=request.tone,
            processing_time=processing_time,
            model_used=caption_result.model_used,
            confidence_score=caption_result.confidence_score
        )
        
        # Log success
        logger.info(
            f"Caption generated successfully for user {deps.user['id']} "
            f"in {processing_time:.3f}s"
        )
        
        return response
        
    except ValidationError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except AIGenerationError as e:
        logger.error(f"AI generation error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Route Builder Example

```python
# Create route builder for monitoring endpoints
monitoring_builder = RouteBuilder(router)

@monitoring_builder.with_dependencies(Depends(require_authentication))
@monitoring_builder.with_tags("monitoring")
@monitoring_builder.with_description("Get caption generation statistics")
@monitoring_builder.build_route("/stats")
async def get_caption_stats(
    deps: AdvancedDependencies = Depends()
) -> Dict[str, Any]:
    """Get caption generation statistics"""
    
    try:
        # Get database statistics
        db_stats = await deps.db_pool.execute_query(
            query="""
                SELECT 
                    COUNT(*) as total_captions,
                    AVG(LENGTH(caption)) as avg_caption_length,
                    COUNT(DISTINCT user_id) as unique_users,
                    MAX(created_at) as last_generation
                FROM caption_history
            """
        )
        
        # Get cache statistics
        cache_stats = await deps.cache_manager.get_stats()
        
        # Get AI engine statistics
        engine_stats = deps.ai_engine.get_stats()
        
        return {
            "database": dict(db_stats[0]) if db_stats else {},
            "cache": cache_stats,
            "ai_engine": engine_stats,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get caption stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get statistics")
```

## Best Practices

### 1. Dependency Organization

- **Use appropriate dependency classes**: Choose `CoreDependencies`, `AdvancedDependencies`, or `ServiceDependencies` based on route complexity
- **Keep dependencies focused**: Each dependency should have a single responsibility
- **Validate dependencies**: Ensure all required dependencies are available before route execution

### 2. Route Structure

- **Clear naming**: Use descriptive names for routes and functions
- **Consistent patterns**: Follow the same structure for similar routes
- **Proper documentation**: Include comprehensive docstrings for all routes
- **Error handling**: Implement proper error handling with specific exceptions

### 3. Performance Optimization

- **Async operations**: Use async/await for I/O-bound operations
- **Caching**: Implement caching for expensive operations
- **Rate limiting**: Apply rate limiting to prevent abuse
- **Monitoring**: Include performance monitoring in routes

### 4. Testing

- **Unit tests**: Test individual route functions
- **Integration tests**: Test complete request/response cycles
- **Dependency mocking**: Mock dependencies for isolated testing
- **Error scenarios**: Test error handling and edge cases

## Examples and Patterns

### Basic Route Pattern

```python
@router.get("/endpoint")
async def basic_endpoint(
    deps: CoreDependencies = Depends()
) -> ResponseModel:
    """Basic endpoint with core dependencies"""
    try:
        # Business logic here
        result = await deps.ai_engine.some_operation()
        return ResponseModel(data=result)
    except Exception as e:
        logger.error(f"Error in basic_endpoint: {e}")
        raise HTTPException(status_code=500, detail="Operation failed")
```

### Advanced Route Pattern

```python
@router.post("/advanced-endpoint", response_model=AdvancedResponse)
@limit_blocking_operations(
    operation_type=OperationType.ADVANCED_OPERATION,
    identifier="advanced_endpoint",
    user_id_param="user_id"
)
async def advanced_endpoint(
    request: AdvancedRequest,
    deps: AdvancedDependencies = Depends()
) -> AdvancedResponse:
    """Advanced endpoint with full dependency set"""
    
    start_time = time.time()
    
    try:
        # Check cache
        cache_key = f"advanced:{hash(str(request))}"
        cached_result = await deps.cache_manager.get(cache_key)
        
        if cached_result:
            return AdvancedResponse(**cached_result)
        
        # Perform operation
        result = await deps.ai_engine.advanced_operation(request)
        
        # Save to database
        await deps.db_pool.execute_query(
            query="INSERT INTO operations (user_id, result) VALUES ($1, $2)",
            params=(deps.user["id"], result)
        )
        
        # Cache result
        response_data = {
            "result": result,
            "processing_time": time.time() - start_time
        }
        await deps.cache_manager.set(cache_key, response_data, ttl=3600)
        
        # Record metrics
        if deps.io_monitor:
            deps.io_monitor.record_operation(
                operation_type="advanced_endpoint",
                duration=time.time() - start_time,
                success=True
            )
        
        return AdvancedResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Advanced endpoint failed: {e}")
        raise HTTPException(status_code=500, detail="Advanced operation failed")
```

### Batch Processing Pattern

```python
@router.post("/batch-operation", response_model=List[ResponseModel])
@limit_blocking_operations(
    operation_type=OperationType.BATCH_OPERATION,
    identifier="batch_operation",
    user_id_param="user_id"
)
async def batch_operation(
    requests: List[RequestModel],
    deps: AdvancedDependencies = Depends(),
    max_concurrent: int = 5
) -> List[ResponseModel]:
    """Batch operation with concurrency control"""
    
    if len(requests) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 requests per batch")
    
    start_time = time.time()
    
    try:
        import asyncio
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_request(req: RequestModel) -> ResponseModel:
            async with semaphore:
                return await single_operation(req, deps)
        
        # Execute batch processing
        tasks = [process_single_request(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {result}")
                processed_results.append(
                    ResponseModel(error=str(result))
                )
            else:
                processed_results.append(result)
        
        logger.info(f"Batch processing completed: {len(processed_results)} items")
        return processed_results
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail="Batch processing failed")
```

## Testing Strategy

### Unit Testing

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

@pytest.fixture
def mock_dependencies():
    """Mock dependencies for testing"""
    return {
        "user": {"id": "test_user", "permissions": ["read", "write"]},
        "ai_engine": AsyncMock(),
        "cache_manager": AsyncMock(),
        "db_pool": AsyncMock(),
        "io_monitor": AsyncMock()
    }

@pytest.mark.asyncio
async def test_generate_structured_caption(mock_dependencies):
    """Test structured caption generation"""
    from routes.structured_captions import generate_structured_caption
    from dependencies import CoreDependencies
    
    # Mock dependencies
    deps = CoreDependencies(**mock_dependencies)
    
    # Mock AI engine response
    mock_dependencies["ai_engine"].generate_caption_optimized.return_value = MagicMock(
        caption="Test caption",
        hashtags=["#test"],
        model_used="gpt-3.5-turbo",
        confidence_score=0.95
    )
    
    # Test request
    request = StructuredCaptionRequest(
        content_description="Test content",
        style="casual",
        tone="friendly"
    )
    
    # Execute function
    result = await generate_structured_caption(request, deps)
    
    # Assertions
    assert result.caption == "Test caption"
    assert result.hashtags == ["#test"]
    assert result.style == "casual"
    assert result.tone == "friendly"
    assert result.model_used == "gpt-3.5-turbo"
    assert result.confidence_score == 0.95
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_caption_generation_integration():
    """Test complete caption generation flow"""
    from main_structured import app
    from fastapi.testclient import TestClient
    
    client = TestClient(app)
    
    # Test request
    response = client.post(
        "/api/v14/captions/generate",
        json={
            "content_description": "Beautiful sunset over mountains",
            "style": "casual",
            "tone": "friendly",
            "hashtag_count": 15,
            "language": "en"
        },
        headers={"Authorization": "Bearer test_api_key"}
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "caption" in data
    assert "hashtags" in data
    assert "processing_time" in data
    assert data["style"] == "casual"
    assert data["tone"] == "friendly"
```

## Performance Considerations

### 1. Dependency Injection Performance

- **Lazy loading**: Dependencies are loaded only when needed
- **Connection pooling**: Database and API connections are pooled
- **Caching**: Expensive operations are cached
- **Async operations**: All I/O operations are async

### 2. Route Performance

- **Rate limiting**: Prevents abuse and ensures fair usage
- **Circuit breakers**: Prevents cascading failures
- **Timeout handling**: Prevents hanging requests
- **Resource cleanup**: Proper cleanup of resources

### 3. Monitoring and Metrics

- **Performance tracking**: Track operation duration and success rates
- **Resource usage**: Monitor memory, CPU, and I/O usage
- **Error tracking**: Track and alert on errors
- **User analytics**: Track user behavior and usage patterns

## Conclusion

The structured routing system provides:

1. **Clear Organization**: Routes are well-organized with clear dependencies
2. **Maintainability**: Easy to maintain and extend with consistent patterns
3. **Testability**: Routes are easily testable with mocked dependencies
4. **Performance**: Optimized for performance with async operations and caching
5. **Scalability**: Designed to scale with proper resource management

This architecture ensures that the Instagram Captions API v14.0 is not only functional but also maintainable, testable, and performant. 