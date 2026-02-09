# Database Refactoring Examples

This document shows complete before/after examples of refactoring database operations using the new helper functions.

---

## Example 1: Complete Storage Service Refactoring

### Before

```python
# services/storage_service.py
def save_identity(self, identity: IdentityProfile) -> str:
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
        
        # Guardar análisis de contenido
        if identity.content_analysis:
            analysis = db.query(ContentAnalysisModel).filter_by(
                identity_profile_id=identity.profile_id
            ).first()
            
            if analysis:
                analysis.topics = identity.content_analysis.topics
                analysis.themes = identity.content_analysis.themes
                analysis.tone = identity.content_analysis.tone
                analysis.personality_traits = identity.content_analysis.personality_traits
                analysis.communication_style = identity.content_analysis.communication_style
                analysis.common_phrases = identity.content_analysis.common_phrases
                analysis.values = identity.content_analysis.values
                analysis.interests = identity.content_analysis.interests
                analysis.language_patterns = identity.content_analysis.language_patterns
                analysis.sentiment_analysis = identity.content_analysis.sentiment_analysis
                analysis.updated_at = datetime.utcnow()
            else:
                analysis = ContentAnalysisModel(
                    id=str(uuid.uuid4()),
                    identity_profile_id=identity.profile_id,
                    topics=identity.content_analysis.topics,
                    themes=identity.content_analysis.themes,
                    tone=identity.content_analysis.tone,
                    personality_traits=identity.content_analysis.personality_traits,
                    communication_style=identity.content_analysis.communication_style,
                    common_phrases=identity.content_analysis.common_phrases,
                    values=identity.content_analysis.values,
                    interests=identity.content_analysis.interests,
                    language_patterns=identity.content_analysis.language_patterns,
                    sentiment_analysis=identity.content_analysis.sentiment_analysis
                )
                db.add(analysis)
        
        # Guardar perfiles sociales
        for social_profile in [identity.tiktok_profile, identity.instagram_profile, identity.youtube_profile]:
            if social_profile:
                self._save_social_profile(db, identity.profile_id, social_profile)
        
        db.commit()
        logger.info(f"Identidad guardada: {identity.profile_id}")
        return identity.profile_id
```

### After

```python
# services/storage_service.py
from ..db.session_helpers import db_transaction
from ..db.model_helpers import upsert_model

def save_identity(self, identity: IdentityProfile) -> str:
    with db_transaction(log_operation="save_identity") as db:
        # Upsert identidad principal
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
        
        # Upsert análisis de contenido
        if identity.content_analysis:
            upsert_model(
                db,
                ContentAnalysisModel,
                identifier={"identity_profile_id": identity.profile_id},
                update_data={
                    "topics": identity.content_analysis.topics,
                    "themes": identity.content_analysis.themes,
                    "tone": identity.content_analysis.tone,
                    "personality_traits": identity.content_analysis.personality_traits,
                    "communication_style": identity.content_analysis.communication_style,
                    "common_phrases": identity.content_analysis.common_phrases,
                    "values": identity.content_analysis.values,
                    "interests": identity.content_analysis.interests,
                    "language_patterns": identity.content_analysis.language_patterns,
                    "sentiment_analysis": identity.content_analysis.sentiment_analysis
                },
                create_data={"id": str(uuid.uuid4())}
            )
        
        # Guardar perfiles sociales
        for social_profile in [identity.tiktok_profile, identity.instagram_profile, identity.youtube_profile]:
            if social_profile:
                self._save_social_profile(db, identity.profile_id, social_profile)
        
        return identity.profile_id
    # Commit y logging automáticos
```

**Improvements:**
- ✅ Reduced from ~75 lines to ~45 lines (40% reduction)
- ✅ Automatic commit/rollback
- ✅ Automatic timestamp management
- ✅ Consistent error handling
- ✅ No manual logging needed

---

## Example 2: Query Operations Refactoring

### Before

```python
# services/storage_service.py
def get_identity(self, identity_id: str) -> Optional[IdentityProfile]:
    with get_db_session() as db:
        db_model = db.query(IdentityProfileModel).filter_by(id=identity_id).first()
        
        if not db_model:
            return None
        
        # Reconstruir IdentityProfile desde modelo de DB
        from ..core.models import ContentAnalysis
        
        analysis = db.query(ContentAnalysisModel).filter_by(
            identity_profile_id=identity_id
        ).first()
        
        # ... resto del código ...
```

### After

```python
# services/storage_service.py
from ..db.session_helpers import db_transaction
from ..db.query_helpers import query_one

def get_identity(self, identity_id: str) -> Optional[IdentityProfile]:
    with db_transaction(auto_commit=False, log_operation="get_identity") as db:
        db_model = query_one(db, IdentityProfileModel, {"id": identity_id})
        
        if not db_model:
            return None
        
        # Reconstruir IdentityProfile desde modelo de DB
        from ..core.models import ContentAnalysis
        
        analysis = query_one(
            db,
            ContentAnalysisModel,
            {"identity_profile_id": identity_id}
        )
        
        # ... resto del código ...
```

**Improvements:**
- ✅ More readable query syntax
- ✅ Consistent query patterns
- ✅ No commit needed for read operations

---

## Example 3: List Queries with Pagination

### Before

```python
# services/storage_service.py
def get_generated_content(self, identity_id: str, limit: int = 10) -> List[GeneratedContent]:
    with get_db_session() as db:
        db_models = db.query(GeneratedContentModel).filter_by(
            identity_profile_id=identity_id
        ).order_by(GeneratedContentModel.generated_at.desc()).limit(limit).all()
        
        # ... procesar resultados ...
```

### After

```python
# services/storage_service.py
from ..db.session_helpers import db_transaction
from ..db.query_helpers import query_many

def get_generated_content(self, identity_id: str, limit: int = 10) -> List[GeneratedContent]:
    with db_transaction(auto_commit=False, log_operation="get_generated_content") as db:
        db_models = query_many(
            db,
            GeneratedContentModel,
            filters={"identity_profile_id": identity_id},
            order_by="generated_at",
            limit=limit
        )
        
        # ... procesar resultados ...
```

**Improvements:**
- ✅ More readable
- ✅ Easy to add pagination (offset parameter)
- ✅ Consistent query building

---

## Example 4: Complex Service Method

### Before

```python
# analytics/analytics_service.py
def save_metric(self, metric_type: str, value: float, identity_id: str):
    with get_db_session() as db:
        existing = db.query(MetricModel).filter_by(
            metric_type=metric_type,
            identity_id=identity_id
        ).first()
        
        if existing:
            existing.value = value
            existing.updated_at = datetime.utcnow()
            existing.count += 1
        else:
            metric = MetricModel(
                metric_type=metric_type,
                identity_id=identity_id,
                value=value,
                count=1
            )
            db.add(metric)
        
        db.commit()
        logger.info(f"Metric saved: {metric_type} for {identity_id}")
```

### After

```python
# analytics/analytics_service.py
from ..db.session_helpers import db_transaction
from ..db.model_helpers import upsert_model

def save_metric(self, metric_type: str, value: float, identity_id: str):
    with db_transaction(log_operation="save_metric") as db:
        existing = upsert_model(
            db,
            MetricModel,
            identifier={
                "metric_type": metric_type,
                "identity_id": identity_id
            },
            update_data={
                "value": value,
                "count": db.query(MetricModel).filter_by(
                    metric_type=metric_type,
                    identity_id=identity_id
                ).first().count + 1 if db.query(MetricModel).filter_by(
                    metric_type=metric_type,
                    identity_id=identity_id
                ).first() else 1
            }
        )
    # Commit y logging automáticos
```

**Or even better with get_or_create:**

```python
# analytics/analytics_service.py
from ..db.session_helpers import db_transaction
from ..db.model_helpers import get_or_create

def save_metric(self, metric_type: str, value: float, identity_id: str):
    with db_transaction(log_operation="save_metric") as db:
        metric, created = get_or_create(
            db,
            MetricModel,
            identifier={
                "metric_type": metric_type,
                "identity_id": identity_id
            },
            defaults={"value": value, "count": 1}
        )
        
        if not created:
            metric.value = value
            metric.count += 1
    # Commit y logging automáticos
```

**Improvements:**
- ✅ Cleaner code
- ✅ Automatic transaction management
- ✅ Consistent error handling

---

## Summary of Database Helpers

| Helper Module | Functions | Use Cases | Code Reduction |
|--------------|-----------|-----------|---------------|
| `db/session_helpers.py` | 2 functions | DB session management | ~40-50% |
| `db/model_helpers.py` | 2 functions | Upsert operations | ~60-70% |
| `db/query_helpers.py` | 2 functions | Query building | ~30-40% |

**Total Database Code Reduction:** ~200-300 lines of repetitive code.

**Combined with All Previous Helpers:** ~750-1100 lines of code eliminated across the entire codebase.

---

## Migration Checklist

- [ ] Review database helper functions
- [ ] Write unit tests for helpers
- [ ] Refactor `services/storage_service.py`
- [ ] Refactor `analytics/analytics_service.py`
- [ ] Refactor `notifications/notification_service.py`
- [ ] Refactor `scheduler/scheduler_service.py`
- [ ] Refactor `ab_testing/ab_test_service.py`
- [ ] Refactor all other services using `get_db_session()`
- [ ] Run full test suite
- [ ] Update documentation
- [ ] Code review
- [ ] Deploy to staging
- [ ] Monitor for issues
- [ ] Deploy to production








