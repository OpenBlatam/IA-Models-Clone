# Async/Def Patterns Guide - Instagram Captions API v14.0

## Overview
This guide documents the proper usage of `async def` and `def` functions in the v14.0 optimized Instagram Captions API, following Python best practices for asynchronous programming.

## Core Principles

### Use `def` for Pure Functions
Pure functions that don't perform I/O operations, database queries, or external API calls should use `def`:

```python
def _calculate_quality_score(self, caption: str, content: str) -> float:
    """JIT-optimized quality calculation - pure function"""
    caption_len = len(caption)
    content_len = len(content)
    word_count = len(caption.split())
    
    # Optimized scoring algorithm
    length_score = min(caption_len / 200.0, 1.0) * 30
    word_score = min(word_count / 20.0, 1.0) * 40
    relevance_score = 30.0
    
    return min(length_score + word_score + relevance_score, 100.0)
```

### Use `async def` for Asynchronous Operations
Functions that perform I/O, network calls, database operations, or use `await` should use `async def`:

```python
async def generate_caption(self, request: OptimizedRequest) -> OptimizedResponse:
    """Ultra-fast caption generation with optimizations"""
    start_time = time.time()
    
    # Check cache first
    cache_key = self._generate_cache_key(request)
    if config.ENABLE_CACHE and cache_key in self.cache:
        cached_response = self.cache[cache_key]
        self.stats["cache_hits"] += 1
        return OptimizedResponse(
            **cached_response,
            cache_hit=True,
            processing_time=time.time() - start_time
        )
    
    # Generate caption
    caption = await self._generate_with_ai(request)
    hashtags = await self._generate_hashtags(request, caption)
    quality_score = self._calculate_quality_score(caption, request.content_description)
    
    # ... rest of implementation
```

## Function Categories

### Pure Functions (`def`)
These functions perform no I/O and are deterministic:

1. **Utility Functions**
   ```python
   def generate_request_id() -> str:
       """Generate unique request ID - pure function"""
       return f"v14-{secrets.token_urlsafe(8)}"
   
   def validate_api_key(api_key: str) -> bool:
       """Validate API key - pure function"""
       valid_keys = ["optimized-v14-key", "ultra-fast-key", "performance-key"]
       return api_key in valid_keys
   
   def sanitize_content(content: str) -> str:
       """Sanitize content for security - pure function"""
       harmful_patterns = ['<script', 'javascript:', 'data:']
       for pattern in harmful_patterns:
           if pattern.lower() in content.lower():
               raise ValueError(f"Potentially harmful content detected: {pattern}")
       return content.strip()
   ```

2. **Calculation Functions**
   ```python
   def _calculate_quality_score(self, caption: str, content: str) -> float:
       """JIT-optimized quality calculation - pure function"""
       # ... calculation logic
   
   def _generate_cache_key(self, request: OptimizedRequest) -> str:
       """Generate optimized cache key - pure function"""
       key_data = f"{request.content_description}:{request.style}:{request.hashtag_count}"
       return hashlib.md5(key_data.encode()).hexdigest()
   ```

3. **Fallback Functions**
   ```python
   def _fallback_generation(self, request: OptimizedRequest) -> str:
       """Fallback caption generation - pure function"""
       templates = {
           "casual": f"Just captured this amazing moment! {request.content_description} ✨",
           "professional": f"Professional insight: {request.content_description} #expertise",
           # ... more templates
       }
       return templates.get(request.style, templates["casual"])
   ```

4. **Statistics Functions**
   ```python
   def get_stats(self) -> Dict[str, Any]:
       """Get performance statistics - pure function"""
       return {
           "total_requests": self.stats["requests"],
           "cache_hits": self.stats["cache_hits"],
           # ... more stats
       }
   
   def get_performance_summary(self) -> Dict[str, Any]:
       """Get performance summary - pure function"""
       response_times = self.metrics["response_times"]
       return {
           "uptime": time.time() - self.metrics["start_time"],
           # ... more metrics
       }
   ```

### Asynchronous Functions (`async def`)
These functions perform I/O operations or use `await`:

1. **API Endpoints**
   ```python
   @app.post("/api/v14/generate", response_model=OptimizedResponse)
   async def generate_caption(
       request: OptimizedRequest,
       api_key: str = Depends(verify_api_key),
       background_tasks: BackgroundTasks = None
   ):
       """Ultra-fast single caption generation"""
       # ... async implementation
   
   @app.post("/api/v14/batch")
   async def batch_generate(
       requests: List[OptimizedRequest],
       api_key: str = Depends(verify_api_key)
   ):
       """Optimized batch processing"""
       # ... async implementation
   ```

2. **AI Generation Functions**
   ```python
   async def _generate_with_ai(self, request: OptimizedRequest) -> str:
       """Generate caption using optimized AI"""
       if not self.model or not self.tokenizer:
           return self._fallback_generation(request)
       
       try:
           prompt = f"Write a {request.style} Instagram caption about {request.content_description}:"
           inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=100, truncation=True)
           # ... async AI processing
       except Exception as e:
           logger.error(f"AI generation failed: {e}")
           return self._fallback_generation(request)
   
   async def _generate_hashtags(self, request: OptimizedRequest, caption: str) -> List[str]:
       """Generate optimized hashtags"""
       # ... async hashtag generation
   ```

3. **Initialization Functions**
   ```python
   async def _initialize_models(self):
       """Initialize models with optimizations"""
       try:
           self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
           self.model = AutoModelForCausalLM.from_pretrained(
               config.MODEL_NAME,
               torch_dtype=torch.float16 if config.MIXED_PRECISION else torch.float32
           ).to(self.device)
           # ... async initialization
       except Exception as e:
           logger.error(f"Model initialization failed: {e}")
   ```

4. **Batch Processing Functions**
   ```python
   async def batch_generate(self, requests: List[OptimizedRequest]) -> List[OptimizedResponse]:
       """Batch processing for multiple requests"""
       if not config.ENABLE_BATCHING:
           return [await self.generate_caption(req) for req in requests]
       
       # Process in batches
       batch_size = config.BATCH_SIZE
       results = []
       
       for i in range(0, len(requests), batch_size):
           batch = requests[i:i + batch_size]
           batch_results = await asyncio.gather(*[self.generate_caption(req) for req in batch])
           results.extend(batch_results)
       
       return results
   ```

5. **Testing Functions**
   ```python
   async def test_api_health(self):
       """Test API health and basic functionality"""
       try:
           async with aiohttp.ClientSession() as session:
               async with session.get(f"{self.base_url}/health") as response:
                   # ... async testing
       except Exception as e:
           print(f"❌ API Health Check: ERROR - {e}")
           return False
   
   async def test_single_generation(self):
       """Test single caption generation"""
       # ... async testing implementation
   ```

## Best Practices

### 1. Function Documentation
Always document whether a function is pure or async:
```python
def pure_function() -> str:
    """Pure function - no I/O operations"""
    return "result"

async def async_function() -> str:
    """Async function - performs I/O operations"""
    result = await some_async_operation()
    return result
```

### 2. Error Handling
- Pure functions: Use standard exception handling
- Async functions: Use `try/except` with proper async context

### 3. Performance Considerations
- Pure functions can be optimized with JIT compilation
- Async functions should use proper concurrency patterns

### 4. Testing
- Pure functions: Unit tests with standard assertions
- Async functions: Use `pytest-asyncio` for async testing

## Examples from v14.0 Implementation

### Correct Usage Examples

✅ **Pure Function (def)**
```python
def _calculate_quality_score(self, caption: str, content: str) -> float:
    """JIT-optimized quality calculation - pure function"""
    # No I/O, deterministic calculation
    return score
```

✅ **Async Function (async def)**
```python
async def generate_caption(self, request: OptimizedRequest) -> OptimizedResponse:
    """Ultra-fast caption generation with optimizations"""
    # Uses await, performs I/O operations
    caption = await self._generate_with_ai(request)
    return response
```

✅ **FastAPI Endpoint**
```python
@app.post("/api/v14/generate")
async def generate_caption(request: OptimizedRequest):
    """API endpoint - async for HTTP handling"""
    # FastAPI requires async for endpoints
    return await optimized_engine.generate_caption(request)
```

### Performance Benefits

1. **Pure Functions (`def`)**
   - Can be JIT compiled with Numba
   - No overhead from async event loop
   - Deterministic and cacheable
   - Better for CPU-intensive calculations

2. **Async Functions (`async def`)**
   - Non-blocking I/O operations
   - Better concurrency handling
   - Scalable for high-load scenarios
   - Proper resource management

## Summary

The v14.0 Instagram Captions API correctly implements async/def patterns:

- **Pure functions** use `def` for calculations, utilities, and deterministic operations
- **Async functions** use `async def` for I/O operations, API calls, and concurrent processing
- **Clear documentation** indicates function type and purpose
- **Performance optimizations** leverage the appropriate pattern for each use case

This approach ensures optimal performance, maintainability, and scalability of the API. 