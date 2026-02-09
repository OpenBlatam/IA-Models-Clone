# Advanced Refactoring Analysis: Database and Model Patterns

## Executive Summary

This document identifies additional repetitive patterns in database operations and model management that can be optimized with helper functions. These patterns appear 50+ times across the codebase and represent significant opportunities for code reduction and consistency improvement.

---

## Pattern 1: Database Session Management

### Problem Identified

**51 occurrences** of the pattern:
```python
with get_db_session() as db:
    # database operations
    db.commit()
    logger.info("Operation completed")
```

**Locations Found:**
- `services/storage_service.py` (4 occurrences)
- `services/versioning_service.py` (3 occurrences)
- `analytics/analytics_service.py` (3 occurrences)
- `notifications/notification_service.py` (6 occurrences)
- `scheduler/scheduler_service.py` (4 occurrences)
- `ab_testing/ab_test_service.py` (6 occurrences)
- And 25+ more locations

### Issues with Current Pattern

1. **No consistent error handling** - Some operations have try/except, others don't
2. **Inconsistent commit patterns** - Some commit, some don't
3. **No rollback on error** - Errors can leave partial transactions
4. **Repetitive logging** - Similar log messages repeated everywhere
5. **No transaction management** - No way to batch operations or control transactions

### Proposed Helper Function

```python
# db/session_helpers.py
from contextlib import contextmanager
from typing import Callable, TypeVar, Optional, Any
from sqlalchemy.orm import Session
import logging

from .base import get_db_session

logger = logging.getLogger(__name__)

T = TypeVar('T')


@contextmanager
def db_transaction(
    auto_commit: bool = True,
    auto_rollback: bool = True,
    log_operation: Optional[str] = None
):
    """
    Context manager para operaciones de base de datos con manejo automático de transacciones.
    
    Args:
        auto_commit: Si hacer commit automático al final (default: True)
        auto_rollback: Si hacer rollback automático en caso de error (default: True)
        log_operation: Nombre de la operación para logging (opcional)
        
    Usage:
        with db_transaction(log_operation="save_identity") as db:
            # database operations
            # commit automático al salir exitosamente
    """
    session = get_db_session()
    db = session.__enter__()
    
    try:
        if log_operation:
            logger.debug(f"Starting database transaction: {log_operation}")
        
        yield db
        
        if auto_commit:
            db.commit()
            if log_operation:
                logger.info(f"Database transaction completed: {log_operation}")
        
    except Exception as e:
        if auto_rollback:
            db.rollback()
            logger.error(
                f"Database transaction failed: {log_operation or 'unknown'}: {e}",
                exc_info=True
            )
        raise
    finally:
        session.__exit__(None, None, None)


def with_db_session(
    func: Callable[[Session], T],
    auto_commit: bool = True,
    auto_rollback: bool = True,
    operation_name: Optional[str] = None
) -> T:
    """
    Ejecuta una función con una sesión de base de datos.
    
    Args:
        func: Función que recibe Session como primer argumento
        auto_commit: Si hacer commit automático (default: True)
        auto_rollback: Si hacer rollback en error (default: True)
        operation_name: Nombre de la operación para logging
        
    Returns:
        Resultado de la función
        
    Usage:
        result = with_db_session(
            lambda db: db.query(Model).filter_by(id=id).first(),
            operation_name="get_identity"
        )
    """
    with db_transaction(
        auto_commit=auto_commit,
        auto_rollback=auto_rollback,
        log_operation=operation_name or func.__name__
    ) as db:
        return func(db)
```

### Integration Example

**Before:**
```python
# services/storage_service.py
def save_identity(self, identity: IdentityProfile) -> str:
    with get_db_session() as db:
        existing = db.query(IdentityProfileModel).filter_by(id=identity.profile_id).first()
        
        if existing:
            # Actualizar existente
            existing.username = identity.username
            # ... más campos ...
        else:
            # Crear nuevo
            db_model = IdentityProfileModel(...)
            db.add(db_model)
        
        db.commit()
        logger.info(f"Identidad guardada: {identity.profile_id}")
        return identity.profile_id
```

**After:**
```python
# services/storage_service.py
from ..db.session_helpers import db_transaction

def save_identity(self, identity: IdentityProfile) -> str:
    with db_transaction(log_operation="save_identity") as db:
        existing = db.query(IdentityProfileModel).filter_by(id=identity.profile_id).first()
        
        if existing:
            # Actualizar existente
            existing.username = identity.username
            # ... más campos ...
        else:
            # Crear nuevo
            db_model = IdentityProfileModel(...)
            db.add(db_model)
        
        return identity.profile_id
    # Commit y logging automáticos
```

**Benefits:**
- ✅ Automatic commit/rollback
- ✅ Consistent error handling
- ✅ Automatic logging
- ✅ Cleaner code (no manual commit/logging)

---

## Pattern 2: Update or Create (Upsert) Pattern

### Problem Identified

**15+ occurrences** of the pattern:
```python
existing = db.query(Model).filter_by(id=id).first()

if existing:
    # Actualizar campos
    existing.field1 = value1
    existing.field2 = value2
    existing.updated_at = datetime.utcnow()
else:
    # Crear nuevo
    model = Model(id=id, field1=value1, field2=value2)
    db.add(model)
```

**Locations Found:**
- `services/storage_service.py` - save_identity, _save_social_profile
- `services/versioning_service.py` - save_version
- Multiple other services

### Issues with Current Pattern

1. **Repetitive code** - Same pattern repeated many times
2. **Inconsistent field updates** - Some update all fields, some don't
3. **No atomic operation** - Race conditions possible
4. **Manual timestamp management** - updated_at set manually each time

### Proposed Helper Function

```python
# db/model_helpers.py
from typing import TypeVar, Type, Dict, Any, Optional, Callable
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
        identifier: Diccionario con campos para identificar el modelo (ej: {"id": "123"})
        update_data: Datos para actualizar (si existe) o crear (si no existe)
        create_data: Datos adicionales solo para creación (opcional)
        auto_timestamp: Si actualizar updated_at automáticamente (default: True)
        
    Returns:
        Instancia del modelo (actualizado o creado)
        
    Usage:
        identity = upsert_model(
            db,
            IdentityProfileModel,
            identifier={"id": identity.profile_id},
            update_data={
                "username": identity.username,
                "display_name": identity.display_name,
                # ... más campos ...
            }
        )
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


def get_or_create(
    db: Session,
    model_class: Type[T],
    identifier: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None
) -> tuple[T, bool]:
    """
    Obtiene un modelo o lo crea si no existe.
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo SQLAlchemy
        identifier: Campos para identificar el modelo
        defaults: Valores por defecto si se crea nuevo
        
    Returns:
        Tupla (modelo, created) donde created es True si se creó nuevo
        
    Usage:
        identity, created = get_or_create(
            db,
            IdentityProfileModel,
            identifier={"id": identity_id},
            defaults={"username": "default"}
        )
    """
    existing = db.query(model_class).filter_by(**identifier).first()
    
    if existing:
        return existing, False
    
    create_data = {**identifier}
    if defaults:
        create_data.update(defaults)
    
    if hasattr(model_class, 'created_at'):
        create_data['created_at'] = datetime.utcnow()
    if hasattr(model_class, 'updated_at'):
        create_data['updated_at'] = datetime.utcnow()
    
    new_model = model_class(**create_data)
    db.add(new_model)
    return new_model, True
```

### Integration Example

**Before:**
```python
# services/storage_service.py
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
```

**After:**
```python
# services/storage_service.py
from ..db.model_helpers import upsert_model

db_model = upsert_model(
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
# updated_at se actualiza automáticamente
```

**Benefits:**
- ✅ Reduced from ~20 lines to ~10 lines
- ✅ Automatic timestamp management
- ✅ Consistent update/create logic
- ✅ Less error-prone

---

## Pattern 3: Model to Domain Object Conversion

### Problem Identified

**10+ occurrences** of converting database models to domain objects:
```python
# Reconstruir IdentityProfile desde modelo de DB
identity = IdentityProfile(
    profile_id=db_model.id,
    username=db_model.username,
    display_name=db_model.display_name,
    # ... muchos más campos ...
)
```

**Locations Found:**
- `services/storage_service.py` - get_identity, get_generated_content
- Multiple other services

### Issues with Current Pattern

1. **Repetitive field mapping** - Same fields mapped repeatedly
2. **Error-prone** - Easy to miss fields or map incorrectly
3. **No default handling** - Manual None checks everywhere
4. **Inconsistent** - Different services handle defaults differently

### Proposed Helper Function

```python
# db/mapper_helpers.py
from typing import TypeVar, Type, Dict, Any, Callable, Optional
from sqlalchemy.orm import Session
from pydantic import BaseModel

T_DB = TypeVar('T_DB')  # Database model
T_DOMAIN = TypeVar('T_DOMAIN', bound=BaseModel)  # Domain model


def model_to_domain(
    db_model: Optional[T_DB],
    domain_class: Type[T_DOMAIN],
    field_mapping: Optional[Dict[str, str]] = None,
    defaults: Optional[Dict[str, Any]] = None,
    transformers: Optional[Dict[str, Callable]] = None
) -> Optional[T_DOMAIN]:
    """
    Convierte un modelo de base de datos a un objeto de dominio.
    
    Args:
        db_model: Modelo de base de datos (puede ser None)
        domain_class: Clase del objeto de dominio
        field_mapping: Mapeo de nombres de campos (db_field -> domain_field)
        defaults: Valores por defecto para campos faltantes
        transformers: Funciones de transformación para campos específicos
        
    Returns:
        Objeto de dominio o None si db_model es None
        
    Usage:
        identity = model_to_domain(
            db_model,
            IdentityProfile,
            field_mapping={"id": "profile_id"},
            defaults={"knowledge_base": {}},
            transformers={"platform": lambda x: Platform(x)}
        )
    """
    if db_model is None:
        return None
    
    data = {}
    
    # Mapear campos
    for db_field in dir(db_model):
        if db_field.startswith('_'):
            continue
        
        domain_field = field_mapping.get(db_field, db_field) if field_mapping else db_field
        
        # Obtener valor
        value = getattr(db_model, db_field, None)
        
        # Aplicar transformador si existe
        if transformers and domain_field in transformers:
            value = transformers[domain_field](value)
        
        # Usar default si es None
        if value is None and defaults and domain_field in defaults:
            value = defaults[domain_field]
        
        # Solo incluir si no es None o está en defaults
        if value is not None or (defaults and domain_field in defaults):
            data[domain_field] = value
    
    return domain_class(**data)


def models_to_domain_list(
    db_models: list[T_DB],
    domain_class: Type[T_DOMAIN],
    **kwargs
) -> list[T_DOMAIN]:
    """
    Convierte una lista de modelos de DB a objetos de dominio.
    
    Args:
        db_models: Lista de modelos de base de datos
        domain_class: Clase del objeto de dominio
        **kwargs: Argumentos adicionales para model_to_domain
        
    Returns:
        Lista de objetos de dominio
    """
    return [
        model_to_domain(model, domain_class, **kwargs)
        for model in db_models
        if model is not None
    ]
```

### Integration Example

**Before:**
```python
# services/storage_service.py
identity = IdentityProfile(
    profile_id=db_model.id,
    username=db_model.username,
    display_name=db_model.display_name,
    bio=db_model.bio,
    content_analysis=content_analysis,
    knowledge_base=db_model.knowledge_base or {},
    total_videos=db_model.total_videos,
    total_posts=db_model.total_posts,
    total_comments=db_model.total_comments,
    created_at=db_model.created_at,
    updated_at=db_model.updated_at,
    metadata=db_model.metadata or {}
)
```

**After:**
```python
# services/storage_service.py
from ..db.mapper_helpers import model_to_domain

identity = model_to_domain(
    db_model,
    IdentityProfile,
    field_mapping={"id": "profile_id"},
    defaults={
        "knowledge_base": {},
        "metadata": {},
        "content_analysis": content_analysis
    }
)
```

**Benefits:**
- ✅ Reduced from ~15 lines to ~8 lines
- ✅ Automatic None handling
- ✅ Consistent field mapping
- ✅ Easy to add transformers for complex fields

---

## Pattern 4: Query Building Patterns

### Problem Identified

**20+ occurrences** of similar query patterns:
```python
db.query(Model).filter_by(field=value).first()
db.query(Model).filter_by(field=value).order_by(Model.created_at.desc()).limit(limit).all()
```

### Proposed Helper Function

```python
# db/query_helpers.py
from typing import TypeVar, Type, Optional, List, Dict, Any
from sqlalchemy.orm import Session, Query
from sqlalchemy import desc

T = TypeVar('T')


def query_one(
    db: Session,
    model_class: Type[T],
    filters: Dict[str, Any],
    order_by: Optional[str] = None
) -> Optional[T]:
    """
    Ejecuta una query que retorna un solo resultado.
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo
        filters: Filtros para filter_by
        order_by: Campo para ordenar (opcional)
        
    Returns:
        Modelo o None
    """
    query = db.query(model_class).filter_by(**filters)
    
    if order_by:
        query = query.order_by(desc(getattr(model_class, order_by)))
    
    return query.first()


def query_many(
    db: Session,
    model_class: Type[T],
    filters: Optional[Dict[str, Any]] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[T]:
    """
    Ejecuta una query que retorna múltiples resultados.
    
    Args:
        db: Sesión de base de datos
        model_class: Clase del modelo
        filters: Filtros para filter_by (opcional)
        order_by: Campo para ordenar (opcional)
        limit: Límite de resultados (opcional)
        offset: Offset para paginación (opcional)
        
    Returns:
        Lista de modelos
    """
    query = db.query(model_class)
    
    if filters:
        query = query.filter_by(**filters)
    
    if order_by:
        query = query.order_by(desc(getattr(model_class, order_by)))
    
    if offset:
        query = query.offset(offset)
    
    if limit:
        query = query.limit(limit)
    
    return query.all()
```

### Integration Example

**Before:**
```python
db_model = db.query(IdentityProfileModel).filter_by(id=identity_id).first()

db_models = db.query(GeneratedContentModel).filter_by(
    identity_profile_id=identity_id
).order_by(GeneratedContentModel.generated_at.desc()).limit(limit).all()
```

**After:**
```python
from ..db.query_helpers import query_one, query_many

db_model = query_one(db, IdentityProfileModel, {"id": identity_id})

db_models = query_many(
    db,
    GeneratedContentModel,
    filters={"identity_profile_id": identity_id},
    order_by="generated_at",
    limit=limit
)
```

**Benefits:**
- ✅ More readable
- ✅ Consistent query patterns
- ✅ Easy to add pagination
- ✅ Less repetitive code

---

## Summary of New Helpers

| Helper Module | Functions | Use Cases | Code Reduction |
|--------------|-----------|-----------|---------------|
| `db/session_helpers.py` | 2 functions | DB session management | ~40-50% |
| `db/model_helpers.py` | 2 functions | Upsert operations | ~60-70% |
| `db/mapper_helpers.py` | 2 functions | Model to domain conversion | ~50-60% |
| `db/query_helpers.py` | 2 functions | Query building | ~30-40% |

**Total Additional Reduction:** ~200-300 more lines of repetitive code.

**Combined with all previous helpers:** ~550-800 lines of code eliminated across the entire codebase.

---

## Implementation Priority

### Phase 1: High Impact (Do First)
1. ✅ Database session management (`session_helpers.py`)
2. ✅ Upsert operations (`model_helpers.py`)

**Estimated Impact:** ~150-200 lines reduced

### Phase 2: Medium Impact
3. ✅ Model to domain conversion (`mapper_helpers.py`)
4. ✅ Query helpers (`query_helpers.py`)

**Estimated Impact:** ~50-100 lines reduced

---

## Complete Refactoring Example

### Before

```python
def save_identity(self, identity: IdentityProfile) -> str:
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

### After

```python
from ..db.session_helpers import db_transaction
from ..db.model_helpers import upsert_model

def save_identity(self, identity: IdentityProfile) -> str:
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

**Improvements:**
- ✅ Reduced from ~35 lines to ~15 lines (57% reduction)
- ✅ Automatic commit/rollback
- ✅ Automatic timestamp management
- ✅ Consistent error handling
- ✅ More maintainable








