# Detailed Refactoring Analysis: Step-by-Step Pattern Identification

This document provides a comprehensive, step-by-step analysis of how to identify repetitive patterns and create helper functions, using real code examples from the codebase.

---

## Analysis Methodology

### Step 1: Code Review and Pattern Identification

We start by examining actual code to identify repetitive patterns.

---

## Pattern Analysis #1: Cache Key Generation

### Step 1: Review the Code

**Location:** `api/routes.py`, lines 141-144

```python
if request.use_cache:
    cache_key = hashlib.md5(
        f"extract_profile_{request.platform}_{request.username}".encode()
    ).hexdigest()
    if cache_key in _response_cache:
        logger.debug(f"Respuesta obtenida de caché: {request.platform}/{request.username}")
        return _response_cache[cache_key]
```

**Similar Pattern Found:** `services/content_generator.py`, lines 173-175

```python
cache_key = hashlib.md5(
    f"instagram_{self.identity.profile_id}_{topic}_{style}_{use_lora}".encode()
).hexdigest()
```

**Similar Pattern Found:** `services/identity_analyzer.py`, lines 116-120

```python
cache_key = hashlib.md5(
    f"{tiktok_profile.username if tiktok_profile else ''}"
    f"{instagram_profile.username if instagram_profile else ''}"
    f"{youtube_profile.username if youtube_profile else ''}".encode()
).hexdigest()
```

### Step 2: Identify Repetitive Elements

**Common Elements:**
1. ✅ `hashlib.md5()` - Always used
2. ✅ `.encode()` - Always called on string
3. ✅ `.hexdigest()` - Always called on hash
4. ✅ String concatenation with underscores - Pattern varies but similar
5. ✅ Handling of None values - Inconsistent (sometimes handled, sometimes not)

**Variations:**
- Number of parts varies (2-5 parts)
- Some parts can be None
- Some use f-strings, some use concatenation

### Step 3: Consider Parameters for Flexibility

**Required Parameters:**
- Variable number of parts (strings, numbers, None values)

**Optional Parameters:**
- Separator (default: "_")
- Hash algorithm (default: MD5, but could be SHA256)

### Step 4: Explain Benefits

**Code Reusability:**
- Single function replaces 15+ occurrences
- Consistent behavior across all cache key generation

**Ease of Future Updates:**
- Change hash algorithm in one place
- Add validation or sanitization in one place
- Switch to different key generation strategy easily

**Maintainability:**
- Clear intent: `generate_cache_key("extract_profile", platform, username)`
- Self-documenting code
- Less error-prone (handles None automatically)

### Step 5: Sample Helper Function

```python
# utils/cache_helpers.py
import hashlib
from typing import Any, Union, List

def generate_cache_key(*parts: Union[str, int, float, None]) -> str:
    """
    Genera una clave de caché consistente a partir de múltiples partes.
    
    Args:
        *parts: Componentes para la clave (strings, números, o None)
        
    Returns:
        Hash MD5 hexadecimal de la clave generada
        
    Examples:
        >>> generate_cache_key("extract_profile", "tiktok", "username123")
        'a1b2c3d4e5f6...'
    """
    # Filtrar None y convertir todo a string
    parts_str = [str(p) if p is not None else "" for p in parts]
    key_string = "_".join(parts_str)
    return hashlib.md5(key_string.encode()).hexdigest()
```

### Step 6: Updated Code Snippets

**Before (api/routes.py):**
```python
if request.use_cache:
    cache_key = hashlib.md5(
        f"extract_profile_{request.platform}_{request.username}".encode()
    ).hexdigest()
    if cache_key in _response_cache:
        return _response_cache[cache_key]
```

**After (api/routes.py):**
```python
from ..utils.cache_helpers import generate_cache_key

if request.use_cache:
    cache_key = generate_cache_key("extract_profile", request.platform, request.username)
    cached_response = cache.get(cache_key)
    if cached_response:
        return cached_response
```

**Before (services/content_generator.py):**
```python
cache_key = hashlib.md5(
    f"instagram_{self.identity.profile_id}_{topic}_{style}_{use_lora}".encode()
).hexdigest()
```

**After (services/content_generator.py):**
```python
from ..utils.cache_helpers import generate_cache_key

cache_key = generate_cache_key(
    "instagram",
    self.identity.profile_id,
    topic,
    style,
    use_lora
)
```

**Improvement Metrics:**
- Lines reduced: 3-4 → 1 per occurrence
- Clarity: More explicit about cache key components
- Maintainability: Change hash algorithm in one place
- Error handling: Automatic None handling

---

## Pattern Analysis #2: Platform Handler Mapping

### Step 1: Review the Code

**Location:** `api/routes.py`, lines 152-162

```python
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
```

**Similar Pattern Found:** Multiple locations with if/elif chains for platform handling

### Step 2: Identify Repetitive Elements

**Common Elements:**
1. ✅ if/elif/else chain - Always the same structure
2. ✅ Platform string comparison - Always lowercase comparison
3. ✅ Handler function call - Pattern: `extractor.extract_{platform}_profile()`
4. ✅ Error handling - Always raises HTTPException for invalid platform
5. ✅ Same arguments passed - username and use_cache

**Variations:**
- Different handler functions
- Different arguments sometimes
- Different error messages

### Step 3: Consider Parameters for Flexibility

**Required Parameters:**
- Platform name (string)
- Handlers dictionary (mapping platform → function)

**Optional Parameters:**
- Default handler (for unknown platforms)
- Default return value (if platform not found)
- Error message customization

### Step 4: Explain Benefits

**Code Reusability:**
- Eliminates if/elif chains
- Works for any platform mapping scenario

**Ease of Future Updates:**
- Add new platform: just add to dictionary
- Change error handling: modify helper
- Consistent platform validation

**Maintainability:**
- More declarative: dictionary shows all platforms at a glance
- Less error-prone: no missed elif branches
- Easier to test: can test dictionary separately

### Step 5: Sample Helper Function

```python
# utils/platform_helpers.py
from typing import Dict, Callable, Any, Optional, TypeVar

T = TypeVar('T')

def execute_for_platform(
    platform: str,
    handlers: Dict[str, Callable[..., T]],
    *args,
    default: Optional[T] = None,
    **kwargs
) -> Optional[T]:
    """
    Ejecuta el handler apropiado para una plataforma.
    
    Args:
        platform: Nombre de la plataforma
        handlers: Diccionario de handlers por plataforma
        *args: Argumentos para el handler
        default: Valor por defecto si la plataforma no existe
        **kwargs: Keyword arguments para el handler
        
    Returns:
        Resultado del handler o default
        
    Usage:
        >>> profile = await execute_for_platform(
        ...     "tiktok",
        ...     {
        ...         "tiktok": extractor.extract_tiktok_profile,
        ...         "instagram": extractor.extract_instagram_profile
        ...     },
        ...     username,
        ...     use_cache=True
        ... )
    """
    platform_lower = platform.lower()
    handler = handlers.get(platform_lower)
    if handler:
        return handler(*args, **kwargs)
    return default
```

### Step 6: Updated Code Snippets

**Before:**
```python
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
```

**After:**
```python
from ..utils.platform_helpers import execute_for_platform
from .exception_helpers import validation_error

platform_map = {
    "tiktok": extractor.extract_tiktok_profile,
    "instagram": extractor.extract_instagram_profile,
    "youtube": extractor.extract_youtube_profile
}

profile = await execute_for_platform(
    request.platform,
    platform_map,
    request.username,
    use_cache=request.use_cache
)

if not profile:
    raise validation_error(
        f"Plataforma no soportada: {request.platform}",
        field="platform"
    )
```

**Improvement Metrics:**
- Lines reduced: 11 → 9 (but much clearer)
- Clarity: All platforms visible in dictionary
- Maintainability: Add platform = add one line to dict
- Testability: Can test platform_map separately

---

## Pattern Analysis #3: Update or Create (Upsert) Pattern

### Step 1: Review the Code

**Location:** `services/storage_service.py`, lines 37-66

```python
with get_db_session() as db:
    # Verificar si ya existe
    existing = db.query(IdentityProfileModel).filter_by(id=identity.profile_id).first()
    
    if existing:
        # Actualizar existente
        existing.username = identity.username
        existing.display_name = identity.display_name
        existing.bio = identity.bio
        existing.total_videos = identity.total_videos
        existing.total_posts = identity.total_posts
        existing.total_comments = identity.total_comments
        existing.knowledge_base = identity.knowledge_base
        existing.updated_at = datetime.utcnow()
        existing.metadata = identity.metadata
        db_model = existing
    else:
        # Crear nuevo
        db_model = IdentityProfileModel(
            id=identity.profile_id,
            username=identity.username,
            display_name=identity.display_name,
            bio=identity.bio,
            total_videos=identity.total_videos,
            total_posts=identity.total_posts,
            total_comments=identity.total_comments,
            knowledge_base=identity.knowledge_base,
            metadata=identity.metadata
        )
        db.add(db_model)
    
    db.commit()
    logger.info(f"Identidad guardada: {identity.profile_id}")
    return identity.profile_id
```

**Similar Pattern Found:** `services/storage_service.py`, `_save_social_profile` method (lines 112-155)

### Step 2: Identify Repetitive Elements

**Common Elements:**
1. ✅ Query to check if exists - Always `db.query(Model).filter_by(**identifier).first()`
2. ✅ if existing: update fields - Always same pattern
3. ✅ else: create new - Always same pattern
4. ✅ Field assignment - Repetitive for each field
5. ✅ Timestamp management - `updated_at = datetime.utcnow()` always in update
6. ✅ `db.add()` - Only in create path
7. ✅ `db.commit()` - Always at end

**Variations:**
- Different models
- Different identifier fields
- Different fields to update
- Some have `created_at`, some don't

### Step 3: Consider Parameters for Flexibility

**Required Parameters:**
- Database session
- Model class
- Identifier (dict of fields to find existing)
- Update data (dict of fields to set)

**Optional Parameters:**
- Create data (additional fields only for creation)
- Auto timestamp (whether to manage timestamps automatically)

### Step 4: Explain Benefits

**Code Reusability:**
- Single function handles all upsert operations
- Works for any model with any fields

**Ease of Future Updates:**
- Add timestamp logic: modify helper
- Add validation: modify helper
- Change update strategy: modify helper

**Maintainability:**
- Less code duplication
- Consistent behavior
- Automatic timestamp management
- Less error-prone (no missed fields)

### Step 5: Sample Helper Function

```python
# db/model_helpers.py
from typing import TypeVar, Type, Dict, Any, Optional
from sqlalchemy.orm import Session
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
T = TypeVar('T')

def upsert_model(
    db: Session,
    model_class: Type[T],
    identifier: Dict[str, Any],
    update_data: Dict[str, Any],
    create_data: Optional[Dict[str, Any]] = None,
    auto_timestamp: bool = True
) -> T:
    """
    Actualiza un modelo existente o crea uno nuevo (upsert).
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo SQLAlchemy
        identifier: Diccionario con campos para identificar el modelo
        update_data: Datos para actualizar (si existe) o crear (si no existe)
        create_data: Datos adicionales solo para creación (opcional)
        auto_timestamp: Si actualizar updated_at automáticamente
        
    Returns:
        Instancia del modelo (actualizado o creado)
    """
    # Buscar modelo existente
    existing = db.query(model_class).filter_by(**identifier).first()
    
    if existing:
        # Actualizar campos
        for key, value in update_data.items():
            setattr(existing, key, value)
        
        if auto_timestamp and hasattr(existing, 'updated_at'):
            existing.updated_at = datetime.utcnow()
        
        logger.debug(f"Updated {model_class.__name__} with {identifier}")
        return existing
    else:
        # Crear nuevo
        create_dict = {**update_data}
        if create_data:
            create_dict.update(create_data)
        
        # Agregar identifier si no está en update_data
        for key, value in identifier.items():
            if key not in create_dict:
                create_dict[key] = value
        
        if auto_timestamp:
            now = datetime.utcnow()
            if 'created_at' not in create_dict and hasattr(model_class, 'created_at'):
                create_dict['created_at'] = now
            if 'updated_at' not in create_dict and hasattr(model_class, 'updated_at'):
                create_dict['updated_at'] = now
        
        new_model = model_class(**create_dict)
        db.add(new_model)
        logger.debug(f"Created new {model_class.__name__} with {identifier}")
        return new_model
```

### Step 6: Updated Code Snippets

**Before:**
```python
with get_db_session() as db:
    existing = db.query(IdentityProfileModel).filter_by(id=identity.profile_id).first()
    
    if existing:
        existing.username = identity.username
        existing.display_name = identity.display_name
        existing.bio = identity.bio
        existing.total_videos = identity.total_videos
        existing.total_posts = identity.total_posts
        existing.total_comments = identity.total_comments
        existing.knowledge_base = identity.knowledge_base
        existing.updated_at = datetime.utcnow()
        existing.metadata = identity.metadata
        db_model = existing
    else:
        db_model = IdentityProfileModel(
            id=identity.profile_id,
            username=identity.username,
            display_name=identity.display_name,
            bio=identity.bio,
            total_videos=identity.total_videos,
            total_posts=identity.total_posts,
            total_comments=identity.total_comments,
            knowledge_base=identity.knowledge_base,
            metadata=identity.metadata
        )
        db.add(db_model)
    
    db.commit()
    logger.info(f"Identidad guardada: {identity.profile_id}")
    return identity.profile_id
```

**After:**
```python
from ..db.session_helpers import db_transaction
from ..db.model_helpers import upsert_model

with db_transaction(log_operation="save_identity") as db:
    upsert_model(
        db,
        IdentityProfileModel,
        identifier={"id": identity.profile_id},
        update_data={
            "username": identity.username,
            "display_name": identity.display_name,
            "bio": identity.bio,
            "total_videos": identity.total_videos,
            "total_posts": identity.total_posts,
            "total_comments": identity.total_comments,
            "knowledge_base": identity.knowledge_base,
            "metadata": identity.metadata
        }
    )
    return identity.profile_id
# Commit y logging automáticos
```

**Improvement Metrics:**
- Lines reduced: 35 → 15 (57% reduction)
- Clarity: Intent is clear - "upsert this model"
- Maintainability: Change upsert logic in one place
- Automatic features: Timestamps, logging, commit

---

## Pattern Analysis #4: Response Dictionary Construction

### Step 1: Review the Code

**Location:** `api/routes.py`, lines 164-174

```python
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
```

**Similar Pattern Found:** `api/routes.py`, lines 253-264

```python
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

**Similar Pattern Found:** `api/routes.py`, lines 447-452

```python
return {
    "success": True,
    "identity_id": identity_id,
    "count": len(content_list),
    "content": [c.model_dump() for c in content_list]
}
```

### Step 2: Identify Repetitive Elements

**Common Elements:**
1. ✅ `"success": True` - Always present
2. ✅ Main data fields - Varies but always present
3. ✅ Metadata/stats - Often present
4. ✅ `model_dump()` - Always used for Pydantic models
5. ✅ List comprehensions with `model_dump()` - Common pattern

**Variations:**
- Different data fields
- Sometimes has metadata, sometimes doesn't
- Sometimes has stats, sometimes doesn't
- Sometimes paginated, sometimes not

### Step 3: Consider Parameters for Flexibility

**Required Parameters:**
- Data (main response data)

**Optional Parameters:**
- Metadata (stats, pagination, etc.)
- Message (optional success message)
- Status code (for HTTP responses)

### Step 4: Explain Benefits

**Code Reusability:**
- Consistent response format across all endpoints
- Single place to change response structure

**Ease of Future Updates:**
- Add common fields (timestamps, version) in one place
- Change response format globally
- Add response validation

**Maintainability:**
- Clear separation: data vs metadata
- Self-documenting: `success_response(data=..., metadata=...)`
- Type-safe responses

### Step 5: Sample Helper Function

```python
# api/response_helpers.py
from typing import Any, Dict, Optional
from fastapi import status

def success_response(
    data: Any = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_code: int = status.HTTP_200_OK
) -> Dict[str, Any]:
    """
    Crea una respuesta de éxito estandarizada.
    
    Args:
        data: Datos principales de la respuesta
        message: Mensaje opcional de éxito
        metadata: Metadatos adicionales (stats, pagination, etc.)
        status_code: Código de estado HTTP
        
    Returns:
        Diccionario con estructura de respuesta estándar
    """
    response = {"success": True}
    
    if data is not None:
        response["data"] = data
    elif message:
        response["message"] = message
    
    if metadata:
        response["metadata"] = metadata
    
    return response
```

### Step 6: Updated Code Snippets

**Before:**
```python
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
```

**After:**
```python
from .response_helpers import success_response
from ..utils.serialization_helpers import serialize_model

response = success_response(
    data={
        "platform": request.platform,
        "username": request.username,
        "profile": serialize_model(profile)
    },
    metadata={
        "stats": {
            "videos": len(profile.videos),
            "posts": len(profile.posts),
            "comments": len(profile.comments)
        }
    }
)
```

**Before:**
```python
return {
    "success": True,
    "identity_id": identity_id,
    "count": len(content_list),
    "content": [c.model_dump() for c in content_list]
}
```

**After:**
```python
from .response_helpers import success_response
from ..utils.serialization_helpers import serialize_models

return success_response(
    data={
        "identity_id": identity_id,
        "content": serialize_models(content_list)
    },
    metadata={"count": len(content_list)}
)
```

**Improvement Metrics:**
- Lines: Similar count but clearer structure
- Consistency: All responses follow same format
- Maintainability: Change format in one place
- Extensibility: Easy to add common fields

---

## Pattern Analysis #5: Dictionary Field Extraction

### Step 1: Review the Code

**Location:** `services/profile_extractor.py`, lines 117-129

```python
profile = SocialProfile(
    platform=Platform.TIKTOK,
    username=username,
    display_name=profile_data.get("display_name"),
    bio=profile_data.get("bio"),
    profile_image_url=profile_data.get("profile_image_url"),
    followers_count=profile_data.get("followers_count"),
    following_count=profile_data.get("following_count"),
    posts_count=profile_data.get("posts_count"),
    videos=videos,
    extracted_at=datetime.now(),
    metadata=profile_data
)
```

**Similar Pattern Found:** Multiple locations extracting fields from dictionaries

### Step 2: Identify Repetitive Elements

**Common Elements:**
1. ✅ Multiple `.get()` calls - Always the same pattern
2. ✅ Field names repeated - In dict and as parameter names
3. ✅ Default values - Sometimes provided, sometimes None
4. ✅ Type conversion - Sometimes needed (int, float, etc.)

**Variations:**
- Different dictionaries
- Different fields
- Different defaults
- Sometimes need type conversion

### Step 3: Consider Parameters for Flexibility

**Required Parameters:**
- Source dictionary
- List of field names to extract

**Optional Parameters:**
- Defaults dictionary (field → default value)
- Transformers dictionary (field → transform function)

### Step 4: Explain Benefits

**Code Reusability:**
- Single function for all field extraction
- Handles defaults and transformations

**Ease of Future Updates:**
- Add validation: modify helper
- Add type checking: modify helper
- Change extraction logic: modify helper

**Maintainability:**
- Less repetitive `.get()` calls
- Clear intent: extract these fields
- Consistent default handling

### Step 5: Sample Helper Function

```python
# utils/dict_helpers.py
from typing import Dict, Any, List, Optional

def extract_fields(
    data: Dict[str, Any],
    fields: List[str],
    defaults: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extrae múltiples campos de un diccionario con defaults.
    
    Args:
        data: Diccionario fuente
        fields: Lista de campos a extraer
        defaults: Diccionario con valores por defecto por campo
        
    Returns:
        Diccionario con los campos extraídos
    """
    defaults = defaults or {}
    return {
        field: data.get(field, defaults.get(field))
        for field in fields
    }
```

### Step 6: Updated Code Snippets

**Before:**
```python
profile = SocialProfile(
    platform=Platform.TIKTOK,
    username=username,
    display_name=profile_data.get("display_name"),
    bio=profile_data.get("bio"),
    profile_image_url=profile_data.get("profile_image_url"),
    followers_count=profile_data.get("followers_count"),
    following_count=profile_data.get("following_count"),
    posts_count=profile_data.get("posts_count"),
    videos=videos,
    extracted_at=datetime.now(),
    metadata=profile_data
)
```

**After:**
```python
from ..utils.dict_helpers import extract_fields
from ..utils.datetime_helpers import now

fields = extract_fields(
    profile_data,
    ["display_name", "bio", "profile_image_url", 
     "followers_count", "following_count", "posts_count"],
    defaults={"followers_count": 0, "following_count": 0, "posts_count": 0}
)

profile = SocialProfile(
    platform=Platform.TIKTOK,
    username=username,
    **fields,
    videos=videos,
    extracted_at=now(),
    metadata=profile_data
)
```

**Improvement Metrics:**
- Lines: Similar but more declarative
- Clarity: All fields visible in one list
- Maintainability: Add/remove fields easily
- Consistency: Same default handling everywhere

---

## Summary: Pattern Identification Process

### Key Steps for Any Pattern

1. **Find Similar Code Blocks**
   - Look for repeated structures
   - Count occurrences
   - Note variations

2. **Identify Common Elements**
   - What's always the same?
   - What varies?
   - What's the core pattern?

3. **Design Flexible Parameters**
   - Required: Core functionality
   - Optional: Variations and customization
   - Defaults: Sensible defaults for optional params

4. **Explain Benefits Clearly**
   - Code reduction
   - Consistency
   - Maintainability
   - Future-proofing

5. **Create Well-Documented Helper**
   - Clear docstring
   - Type hints
   - Examples in docstring
   - Error handling

6. **Show Before/After Examples**
   - Real code from codebase
   - Clear improvement
   - Measurable benefits

---

## Complete Refactoring Example: extract_profile Endpoint

### Original Code (Before)

```python
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

### Refactored Code (After)

```python
from ..utils.cache_helpers import generate_cache_key
from ..utils.cache_manager import get_cache
from ..utils.serialization_helpers import serialize_model
from ..utils.metrics_helpers import track_operation
from ..utils.platform_helpers import execute_for_platform
from .response_helpers import success_response
from .exception_helpers import validation_error

cache = get_cache()

@router.post("/extract-profile", status_code=status.HTTP_200_OK)
@handle_api_errors
@log_endpoint_call
async def extract_profile(request: ExtractProfileRequest):
    # Cache check
    cache_key = generate_cache_key("extract_profile", request.platform, request.username)
    if request.use_cache:
        cached_response = cache.get(cache_key)
        if cached_response:
            return cached_response
    
    with track_operation("profile_extraction", tags={"platform": request.platform}):
        extractor = ProfileExtractor()
        
        # Platform handler mapping
        platform_map = {
            "tiktok": extractor.extract_tiktok_profile,
            "instagram": extractor.extract_instagram_profile,
            "youtube": extractor.extract_youtube_profile
        }
        
        profile = await execute_for_platform(
            request.platform,
            platform_map,
            request.username,
            use_cache=request.use_cache
        )
        
        if not profile:
            raise validation_error(
                f"Plataforma no soportada: {request.platform}",
                field="platform"
            )
        
        response = success_response(
            data={
                "platform": request.platform,
                "username": request.username,
                "profile": serialize_model(profile)
            },
            metadata={
                "stats": {
                    "videos": len(profile.videos),
                    "posts": len(profile.posts),
                    "comments": len(profile.comments)
                }
            }
        )
        
        # Store in cache
        if request.use_cache:
            cache.set(cache_key, response)
        
        return response
```

### Improvement Analysis

**Lines of Code:**
- Before: 55 lines
- After: 48 lines
- Reduction: 13% (but much clearer)

**Patterns Optimized:**
1. ✅ Cache key generation (3 lines → 1 line)
2. ✅ Cache management (5 lines → 2 lines)
3. ✅ Platform mapping (11 lines → 8 lines)
4. ✅ Response formatting (10 lines → 8 lines)
5. ✅ Metrics tracking (2 lines → 1 line)
6. ✅ Error handling (4 lines → 2 lines)

**Total Patterns:** 6 major patterns optimized

**Benefits:**
- **Readability:** More declarative, less imperative
- **Maintainability:** Changes in one place
- **Consistency:** Same patterns everywhere
- **Testability:** Helpers can be tested independently
- **Extensibility:** Easy to add features

---

## Conclusion

This detailed analysis demonstrates:

1. **Systematic Pattern Identification** - How to find repetitive code
2. **Parameter Design** - How to make helpers flexible
3. **Benefit Analysis** - Why helpers improve code
4. **Implementation** - How to create and use helpers
5. **Measurable Impact** - Quantifiable improvements

The refactoring process transforms imperative, repetitive code into declarative, maintainable code using well-designed helper functions.








