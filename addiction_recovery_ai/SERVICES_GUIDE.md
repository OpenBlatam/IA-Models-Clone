# Services Guide - Addiction Recovery AI

## ✅ Recommended Service Structure

### Service Factory Pattern - **USE THIS**

The canonical way to access services is through the service factory:

```python
from services.service_factory import ServiceFactory, get_service_factory

# Get factory instance
factory = get_service_factory()

# Get a service by domain and name
service = factory.get_service('assessment', 'assessment_service')
```

**Features:**
- Centralized service creation
- Singleton pattern support
- Auto-discovery of services
- Domain-based organization

## 🏗️ Service Structure

```
services/
├── service_factory.py          # ✅ Service factory (canonical)
├── functions/                  # ✅ Pure business functions
│   ├── assessment_functions.py
│   ├── progress_functions.py
│   ├── relapse_functions.py
│   ├── support_functions.py
│   ├── analytics_functions.py
│   └── gamification_functions.py
└── [domain]_service.py         # Domain-specific services (130+ files)
```

## 📦 Service Categories

### Core Services

Services are organized by domain/functionality. Examples include:

#### Assessment Services
- `assessment_service.py` - Basic assessment
- `advanced_assessment_service.py` - Advanced assessment features

#### Progress Services
- `progress_tracking_service.py` - Progress tracking
- `advanced_progress_tracking_service.py` - Advanced progress features

#### Relapse Services
- `relapse_prevention_service.py` - Relapse prevention
- `advanced_relapse_tracking_service.py` - Advanced relapse tracking

#### Support Services
- `counseling_service.py` - Counseling services
- `motivation_service.py` - Motivation services
- `coaching_service.py` - Coaching services

#### Analytics Services
- `analytics_service.py` - Analytics
- `advanced_data_analysis_service.py` - Advanced analytics

#### Integration Services
- `emergency_service.py` - Emergency services
- `health_integration_service.py` - Health integrations
- `wearable_service.py` - Wearable device integration
- `iot_integration_service.py` - IoT integration

### Pure Functions

#### `services/functions/` - **USE THIS FOR BUSINESS LOGIC**

Pure business logic functions organized by domain:

```python
from services.functions.assessment_functions import (
    calculate_severity_score,
    generate_assessment_report
)

from services.functions.progress_functions import (
    calculate_recovery_stage,
    predict_relapse_risk
)

from services.functions.relapse_functions import (
    assess_relapse_risk,
    generate_prevention_plan
)
```

**Features:**
- Pure functions (no side effects)
- Easy to test
- Reusable across services
- Domain-organized

## 📝 Usage Examples

### Using Service Factory
```python
from services.service_factory import get_service_factory

factory = get_service_factory()

# Get a service
assessment_service = factory.get_service('assessment', 'assessment_service')

# Use the service
result = await assessment_service.analyze(data)
```

### Using Pure Functions
```python
from services.functions.assessment_functions import calculate_severity_score

# Pure function - easy to test
score = calculate_severity_score(assessment_data)
```

### Direct Service Import (if needed)
```python
from services.assessment_service import AssessmentService

service = AssessmentService()
result = await service.analyze(data)
```

## 🎯 Service Organization

### By Domain
Services are organized by functional domain:
- **Assessment**: Assessment and evaluation services
- **Progress**: Progress tracking and analysis
- **Relapse**: Relapse prevention and tracking
- **Support**: Support, coaching, and motivation
- **Analytics**: Analytics and reporting
- **Integration**: Third-party integrations

### By Type
- **Core Services**: Basic domain services
- **Advanced Services**: Advanced features (prefixed with `advanced_`)
- **Integration Services**: External integrations
- **Analysis Services**: Data analysis services

## 📚 Service Factory API

```python
from services.service_factory import ServiceFactory

factory = ServiceFactory()

# Get service (singleton by default)
service = factory.get_service('domain', 'service_name')

# Get service (new instance)
service = factory.get_service('domain', 'service_name', singleton=False)

# Register custom instance
factory.register_service_instance('domain', 'service_name', instance)

# List available services
services = factory.list_available_services()
```

## 🔄 Service Patterns

### Service Factory Pattern (Recommended)
```python
from services.service_factory import get_service_factory

factory = get_service_factory()
service = factory.get_service('domain', 'service_name')
```

### Direct Import (When Needed)
```python
from services.assessment_service import AssessmentService

service = AssessmentService()
```

### Pure Functions (For Business Logic)
```python
from services.functions.assessment_functions import calculate_severity_score

score = calculate_severity_score(data)
```

## 🎯 Quick Reference

| Pattern | When to Use | Example |
|---------|-------------|---------|
| Service Factory | Standard service access | `factory.get_service('domain', 'name')` |
| Direct Import | When factory not needed | `from services.x import XService` |
| Pure Functions | Business logic, testing | `from services.functions.x import func` |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `API_GUIDE.md` for API structure
- See `UTILITIES_GUIDE.md` for utilities






