# Schemas Guide - Addiction Recovery AI

## ✅ Recommended Schema Structure

### Schema Factory Pattern - **USE THIS**

The canonical way to access schemas is through the schema factory:

```python
from schemas.schema_factory import SchemaFactory, get_schema_factory

# Get factory instance
factory = get_schema_factory()

# Get a schema by domain and name
schema_class = factory.get_schema('assessment', 'AssessmentRequest')
```

**Features:**
- Centralized schema access
- Auto-discovery of schemas
- Domain-based organization
- Type-safe schema retrieval

## 🏗️ Schema Structure

```
schemas/
├── schema_factory.py          # ✅ Schema factory (canonical)
├── __init__.py                # ✅ Exports all schemas
├── common.py                   # ✅ Common schemas (ErrorResponse, etc.)
├── assessment.py               # ✅ Assessment schemas
├── progress.py                 # ✅ Progress schemas
├── relapse.py                  # ✅ Relapse schemas
├── support.py                  # ✅ Support schemas
├── recovery_plan.py            # ✅ Recovery plan schemas
├── analytics.py                 # ✅ Analytics schemas
├── notifications.py            # ✅ Notification schemas
├── users.py                    # ✅ User schemas
├── gamification.py             # ✅ Gamification schemas
├── emergency.py                # ✅ Emergency schemas
└── domains/                    # ✅ Domain-organized schemas
    ├── assessment/
    ├── progress/
    ├── relapse/
    ├── support/
    ├── recovery/
    ├── analytics/
    ├── notifications/
    ├── users/
    ├── gamification/
    └── emergency/
```

## 📦 Schema Categories

### Core Schemas

#### Common Schemas (`schemas/common.py`)
- `ErrorResponse` - Error response schema
- `SuccessResponse` - Success response schema
- `PaginatedResponse` - Paginated response schema

#### Assessment Schemas (`schemas/assessment.py`)
- `AssessmentRequest` - Assessment creation request
- `AssessmentResponse` - Assessment response
- `ProfileResponse` - User profile response
- `UpdateProfileRequest` - Profile update request

#### Progress Schemas (`schemas/progress.py`)
- `LogEntryRequest` - Progress log entry request
- `LogEntryResponse` - Progress log entry response
- `ProgressResponse` - Progress data response
- `StatsResponse` - Statistics response
- `TimelineResponse` - Timeline response

#### Relapse Schemas (`schemas/relapse.py`)
- `RelapseRiskCheckRequest` - Risk check request
- `RelapseRiskResponse` - Risk assessment response
- `CopingStrategiesRequest` - Coping strategies request
- `CopingStrategiesResponse` - Coping strategies response
- `EmergencyPlanRequest` - Emergency plan request
- `EmergencyPlanResponse` - Emergency plan response

#### Support Schemas (`schemas/support.py`)
- `CoachingSessionRequest` - Coaching session request
- `CoachingSessionResponse` - Coaching session response
- `MotivationResponse` - Motivation response
- `MilestoneRequest` - Milestone request
- `MilestoneResponse` - Milestone response
- `AchievementsResponse` - Achievements response

### Additional Schemas

- **Recovery Plan**: `recovery_plan.py`
- **Analytics**: `analytics.py`
- **Notifications**: `notifications.py`
- **Users**: `users.py`
- **Gamification**: `gamification.py`
- **Emergency**: `emergency.py`

## 📝 Usage Examples

### Using Schema Factory
```python
from schemas.schema_factory import get_schema_factory

factory = get_schema_factory()

# Get a schema class
AssessmentRequest = factory.get_schema('assessment', 'AssessmentRequest')

# Use the schema
request = AssessmentRequest(user_id="123", substance_type="alcohol")
```

### Direct Import (Recommended for Most Cases)
```python
from schemas import (
    AssessmentRequest,
    AssessmentResponse,
    ProgressResponse,
    RelapseRiskResponse
)

# Use schemas directly
request = AssessmentRequest(user_id="123", substance_type="alcohol")
```

### Using in FastAPI Routes
```python
from fastapi import APIRouter
from schemas import AssessmentRequest, AssessmentResponse

router = APIRouter()

@router.post("/assessment", response_model=AssessmentResponse)
async def create_assessment(request: AssessmentRequest):
    # request is validated Pydantic model
    return AssessmentResponse(...)
```

### Using Common Schemas
```python
from schemas import ErrorResponse, SuccessResponse, PaginatedResponse

# Error response
return ErrorResponse(message="Something went wrong", code="ERROR_001")

# Success response
return SuccessResponse(message="Operation successful", data={})

# Paginated response
return PaginatedResponse(
    items=[...],
    total=100,
    page=1,
    page_size=10
)
```

## 🎯 Schema Organization

### By Domain
Schemas are organized by functional domain:
- **Assessment**: Assessment and evaluation schemas
- **Progress**: Progress tracking schemas
- **Relapse**: Relapse prevention schemas
- **Support**: Support and coaching schemas
- **Recovery**: Recovery plan schemas
- **Analytics**: Analytics schemas
- **Notifications**: Notification schemas
- **Users**: User management schemas
- **Gamification**: Gamification schemas
- **Emergency**: Emergency schemas

### By Type
- **Request Schemas**: Input validation (suffix: `Request`)
- **Response Schemas**: Output models (suffix: `Response`)
- **Common Schemas**: Shared across domains

## 📚 Schema Factory API

```python
from schemas.schema_factory import SchemaFactory

factory = SchemaFactory()

# Get schema class
schema_class = factory.get_schema('domain', 'schema_name')

# List available schemas
schemas = factory.list_available_schemas()

# Clear cache
factory.clear_cache()
```

## 🔄 Schema Patterns

### Direct Import (Recommended)
```python
from schemas import AssessmentRequest, AssessmentResponse
```

### Schema Factory (Dynamic Access)
```python
from schemas.schema_factory import get_schema_factory

factory = get_schema_factory()
schema = factory.get_schema('domain', 'name')
```

### Domain-Specific Import
```python
from schemas.assessment import AssessmentRequest, AssessmentResponse
from schemas.progress import ProgressResponse, StatsResponse
```

## 🎯 Quick Reference

| Pattern | When to Use | Example |
|---------|-------------|---------|
| Direct Import | Standard use | `from schemas import AssessmentRequest` |
| Schema Factory | Dynamic access | `factory.get_schema('domain', 'name')` |
| Domain Import | Domain-specific | `from schemas.assessment import AssessmentRequest` |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `API_GUIDE.md` for API structure
- See `SERVICES_GUIDE.md` for services






