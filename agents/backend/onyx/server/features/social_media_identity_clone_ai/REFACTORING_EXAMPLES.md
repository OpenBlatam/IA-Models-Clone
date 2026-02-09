# Refactoring Examples: Before and After

This document shows concrete examples of how to refactor existing code to use the helper functions.

---

## Example 1: Cache Key Generation in routes.py

### Before

```python
# api/routes.py (lines 141-147)
if request.use_cache:
    cache_key = hashlib.md5(
        f"extract_profile_{request.platform}_{request.username}".encode()
    ).hexdigest()
    if cache_key in _response_cache:
        logger.debug(f"Respuesta obtenida de caché: {request.platform}/{request.username}")
        return _response_cache[cache_key]
```

### After

```python
# api/routes.py
from ..utils.cache_helpers import generate_cache_key

if request.use_cache:
    cache_key = generate_cache_key("extract_profile", request.platform, request.username)
    if cache_key in _response_cache:
        logger.debug(f"Respuesta obtenida de caché: {request.platform}/{request.username}")
        return _response_cache[cache_key]
```

**Benefits:**
- More readable: explicit parts of the cache key
- Handles None values automatically
- Consistent across all cache key generation

---

## Example 2: HTTPException in routes.py

### Before

```python
# api/routes.py (lines 403-407)
if not identity:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Identidad no encontrada: {identity_id}"
    )
```

### After

```python
# api/routes.py
from .exception_helpers import not_found

if not identity:
    raise not_found("Identidad", identity_id)
```

**Benefits:**
- Shorter and more readable
- Consistent error structure
- Includes error code automatically

---

## Example 3: Response Formatting in routes.py

### Before

```python
# api/routes.py (lines 253-264)
return {
    "success": True,
    "identity_id": identity.profile_id,
    "identity": identity.model_dump(),
    "stats": {
        "total_videos": identity.total_videos,
        "total_posts": identity.total_posts,
        "total_comments": identity.total_comments,
        "topics_count": len(identity.content_analysis.topics),
        "themes_count": len(identity.content_analysis.themes)
    }
}
```

### After

```python
# api/routes.py
from .response_helpers import success_response

return success_response(
    data={
        "identity_id": identity.profile_id,
        "identity": identity.model_dump()
    },
    metadata={
        "stats": {
            "total_videos": identity.total_videos,
            "total_posts": identity.total_posts,
            "total_comments": identity.total_comments,
            "topics_count": len(identity.content_analysis.topics),
            "themes_count": len(identity.content_analysis.themes)
        }
    }
)
```

**Benefits:**
- Clear separation of data and metadata
- Consistent response structure
- Easier to extend with additional fields

---

## Example 4: Validation in routes.py

### Before

```python
# api/routes.py (lines 208-212)
if not any([request.tiktok_username, request.instagram_username, request.youtube_channel_id]):
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Al menos un perfil debe ser proporcionado"
    )
```

### After

```python
# api/routes.py
from ..utils.validation_helpers import validate_at_least_one

validate_at_least_one(
    request.tiktok_username,
    request.instagram_username,
    request.youtube_channel_id,
    field_names=["tiktok_username", "instagram_username", "youtube_channel_id"],
    message="Al menos un perfil debe ser proporcionado"
)
```

**Benefits:**
- More explicit about what's being validated
- Reusable validation logic
- Better error messages

---

## Example 5: Enum Validation in routes.py

### Before

```python
# api/routes.py (lines 300-307)
try:
    platform = Platform(request.platform)
    content_type = ContentType(request.content_type)
except ValueError as e:
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f"Plataforma o tipo de contenido inválido: {e}"
    )
```

### After

```python
# api/routes.py
from ..utils.validation_helpers import validate_platform, validate_content_type

platform = validate_platform(request.platform)
content_type = validate_content_type(request.content_type)
```

**Benefits:**
- Much shorter and cleaner
- Automatic error handling
- Consistent error messages

---

## Example 6: Cache Key in content_generator.py

### Before

```python
# services/content_generator.py (lines 173-175)
cache_key = hashlib.md5(
    f"instagram_{self.identity.profile_id}_{topic}_{style}_{use_lora}".encode()
).hexdigest()
```

### After

```python
# services/content_generator.py
from ..utils.cache_helpers import generate_cache_key

cache_key = generate_cache_key(
    "instagram",
    self.identity.profile_id,
    topic,
    style,
    use_lora
)
```

**Benefits:**
- More readable: each part is explicit
- Handles None values (topic, style can be None)
- Consistent with other cache key generation

---

## Example 7: Error Handling in decorators.py

### Before

```python
# api/decorators.py (lines 36-41)
except ValueError as e:
    logger.warning(f"Validation error in {func.__name__}: {e}")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(e)
    )
```

### After

```python
# api/decorators.py
from .exception_helpers import validation_error

except ValueError as e:
    logger.warning(f"Validation error in {func.__name__}: {e}")
    raise validation_error(str(e))
```

**Benefits:**
- Shorter code
- Consistent error structure
- Automatic error code

---

## Example 8: Complete Endpoint Refactoring

### Before

```python
# api/routes.py - extract_profile endpoint
@router.post("/extract-profile", status_code=status.HTTP_200_OK)
@handle_api_errors
@log_endpoint_call
async def extract_profile(request: ExtractProfileRequest):
    logger.info(f"Extrayendo perfil: {request.platform}/{request.username}")
    metrics.increment("profile_extraction_requests", tags={"platform": request.platform})
    
    # Verificar caché de respuesta si use_cache está habilitado
    if request.use_cache:
        cache_key = hashlib.md5(
            f"extract_profile_{request.platform}_{request.username}".encode()
        ).hexdigest()
        if cache_key in _response_cache:
            logger.debug(f"Respuesta obtenida de caché: {request.platform}/{request.username}")
            return _response_cache[cache_key]
    
    with metrics.timer("profile_extraction_duration", tags={"platform": request.platform}):
        extractor = ProfileExtractor()
    
    if request.platform == "tiktok":
        profile = await extractor.extract_tiktok_profile(request.username, use_cache=request.use_cache)
    elif request.platform == "instagram":
        profile = await extractor.extract_instagram_profile(request.username, use_cache=request.use_cache)
    elif request.platform == "youtube":
        profile = await extractor.extract_youtube_profile(request.username, use_cache=request.use_cache)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Plataforma no soportada: {request.platform}"
        )
    
    response = {
        "success": True,
        "platform": request.platform,
        "username": request.username,
        "profile": profile.model_dump(),
        "stats": {
            "videos": len(profile.videos),
            "posts": len(profile.posts),
            "comments": len(profile.comments)
        }
    }
    
    # Guardar en caché si está habilitado
    if request.use_cache:
        if len(_response_cache) >= _cache_max_size:
            _response_cache.popitem(last=False)
        _response_cache[cache_key] = response
    
    return response
```

### After

```python
# api/routes.py - extract_profile endpoint (refactored)
from ..utils.cache_helpers import generate_cache_key
from .response_helpers import success_response
from .exception_helpers import validation_error
from collections import OrderedDict

# Global cache (could be moved to cache_manager)
_response_cache: OrderedDict = OrderedDict()
_cache_max_size = 1000

@router.post("/extract-profile", status_code=status.HTTP_200_OK)
@handle_api_errors
@log_endpoint_call
async def extract_profile(request: ExtractProfileRequest):
    logger.info(f"Extrayendo perfil: {request.platform}/{request.username}")
    metrics.increment("profile_extraction_requests", tags={"platform": request.platform})
    
    # Verificar caché
    cache_key = generate_cache_key("extract_profile", request.platform, request.username)
    if request.use_cache and cache_key in _response_cache:
        logger.debug(f"Respuesta obtenida de caché: {request.platform}/{request.username}")
        return _response_cache[cache_key]
    
    with metrics.timer("profile_extraction_duration", tags={"platform": request.platform}):
        extractor = ProfileExtractor()
    
    # Extract profile based on platform
    platform_map = {
        "tiktok": extractor.extract_tiktok_profile,
        "instagram": extractor.extract_instagram_profile,
        "youtube": extractor.extract_youtube_profile
    }
    
    extract_func = platform_map.get(request.platform)
    if not extract_func:
        raise validation_error(f"Plataforma no soportada: {request.platform}", field="platform")
    
    profile = await extract_func(request.username, use_cache=request.use_cache)
    
    # Build response
    response = success_response(
        data={
            "platform": request.platform,
            "username": request.username,
            "profile": profile.model_dump()
        },
        metadata={
            "stats": {
                "videos": len(profile.videos),
                "posts": len(profile.posts),
                "comments": len(profile.comments)
            }
        }
    )
    
    # Guardar en caché
    if request.use_cache:
        if len(_response_cache) >= _cache_max_size:
            _response_cache.popitem(last=False)
        _response_cache[cache_key] = response
    
    return response
```

**Improvements:**
1. ✅ Cache key generation uses helper
2. ✅ Response uses standardized format
3. ✅ Error handling uses helper
4. ✅ Platform mapping is cleaner
5. ✅ Overall code is more maintainable

---

## Example 9: Service Lazy Loading

### Before

```python
# api/routes.py (lines 53-76)
def get_analytics_service():
    if not hasattr(get_analytics_service, '_instance'):
        get_analytics_service._instance = AnalyticsService()
    return get_analytics_service._instance

def get_export_service():
    if not hasattr(get_export_service, '_instance'):
        get_export_service._instance = ExportService()
    return get_export_service._instance

def get_versioning_service():
    if not hasattr(get_versioning_service, '_instance'):
        get_versioning_service._instance = VersioningService()
    return get_versioning_service._instance

def get_batch_service():
    if not hasattr(get_batch_service, '_instance'):
        get_batch_service._instance = BatchService()
    return get_batch_service._instance

def get_search_service():
    if not hasattr(get_search_service, '_instance'):
        get_search_service._instance = SearchService()
    return get_search_service._instance
```

### After

```python
# api/routes.py
from ..core.service_factory import create_service_getter

get_analytics_service = create_service_getter(AnalyticsService)
get_export_service = create_service_getter(ExportService)
get_versioning_service = create_service_getter(VersioningService)
get_batch_service = create_service_getter(BatchService)
get_search_service = create_service_getter(SearchService)
```

**Benefits:**
- Reduced from ~25 lines to 5 lines
- Consistent pattern
- Easier to add new services
- Better testability (can reset instances)

---

## Summary of Code Reduction

| Pattern | Before (lines) | After (lines) | Reduction |
|--------|---------------|---------------|-----------|
| Cache key generation | ~3-5 per use | ~1 per use | 60-70% |
| HTTPException | ~3-4 per use | ~1 per use | 70-75% |
| Response formatting | ~5-10 per use | ~3-5 per use | 40-50% |
| Validation | ~3-8 per use | ~1-2 per use | 60-75% |
| Service getters | ~5 per service | ~1 per service | 80% |

**Total estimated reduction:** ~200-300 lines of repetitive code across the codebase.








