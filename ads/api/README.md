# Unified Ads API System

## Overview

The Unified Ads API System consolidates all previously scattered API implementations into a clean, modular, and maintainable architecture following Clean Architecture principles. This system eliminates duplication, improves consistency, and provides a unified interface for all ads-related functionality.

## Architecture

The API system is organized into five distinct layers, each with a specific responsibility:

```
ads/api/
├── __init__.py              # Main router that includes all sub-routers
├── core.py                  # Basic ads generation and management
├── ai.py                    # AI-powered operations and analysis
├── advanced.py              # Advanced AI features and training
├── integrated.py            # Onyx integration capabilities
├── optimized.py             # Production-ready features with optimizations
├── api_demo.py              # Comprehensive demonstration of all functionality
└── README.md                # This documentation
```

### API Layer Structure

| Layer | Purpose | Key Features | Endpoint Prefix |
|-------|---------|--------------|-----------------|
| **Core** | Basic ads functionality | Ads generation, brand voice, audience profiles | `/ads/core` |
| **AI** | AI-powered operations | Content analysis, optimization, recommendations | `/ads/ai` |
| **Advanced** | Advanced AI features | AI training, performance tracking, competitor analysis | `/ads/advanced` |
| **Integrated** | Onyx integration | Cross-platform optimization, Onyx features | `/ads/integrated` |
| **Optimized** | Production features | Rate limiting, caching, background processing | `/ads/optimized` |

## Key Benefits

### 1. **Eliminated Scattered Implementations**
- **Before**: 5+ separate API files with duplicated functionality
- **After**: Single unified system with clear separation of concerns

### 2. **Clean Architecture Principles**
- **Dependency Rule**: Dependencies point inward toward domain entities
- **Separation of Concerns**: Each layer has a single, well-defined responsibility
- **Interface Segregation**: Clear contracts between layers
- **Dependency Inversion**: High-level modules don't depend on low-level modules

### 3. **Consistent Patterns**
- Unified request/response models
- Standardized error handling
- Consistent logging and monitoring
- Shared authentication and authorization

### 4. **Improved Maintainability**
- Single source of truth for each functionality
- Easier testing and debugging
- Simplified dependency management
- Clear upgrade and migration paths

## API Endpoints

### Core API (`/ads/core`)

Basic ads generation and management functionality:

```http
POST /ads/core/generate          # Generate ads from content
POST /ads/core/brand-voice       # Analyze brand voice settings
POST /ads/core/audience-profile  # Analyze audience profiles
POST /ads/core/content-source    # Analyze content sources
POST /ads/core/project-context   # Analyze project context
GET  /ads/core/health           # Health check
GET  /ads/core/capabilities     # Available capabilities
```

### AI API (`/ads/ai`)

AI-powered content operations:

```http
POST /ads/ai/generate-ads           # AI-powered ads generation
POST /ads/ai/analyze-brand-voice    # AI brand voice analysis
POST /ads/ai/optimize-content       # AI content optimization
POST /ads/ai/generate-variations    # AI content variations
POST /ads/ai/analyze-audience       # AI audience analysis
POST /ads/ai/generate-recommendations # AI recommendations
POST /ads/ai/analyze-competitors    # AI competitor analysis
POST /ads/ai/track-performance      # AI performance tracking
GET  /ads/ai/capabilities           # AI capabilities
```

### Advanced API (`/ads/advanced`)

Advanced AI features and training:

```http
POST /ads/advanced/train-ai              # Train AI models
POST /ads/advanced/optimize-content      # Advanced content optimization
GET  /ads/advanced/audience/{segment_id} # Deep audience insights
POST /ads/advanced/brand-voice           # Advanced brand voice analysis
GET  /ads/advanced/performance/{content_id} # Performance tracking
POST /ads/advanced/recommendations       # AI recommendations
GET  /ads/advanced/impact/{content_id}   # Content impact analysis
POST /ads/advanced/audience/optimize/{segment_id} # Audience optimization
POST /ads/advanced/variations            # Content variations
POST /ads/advanced/competitor            # Competitor analysis
GET  /ads/advanced/capabilities          # Advanced capabilities
```

### Integrated API (`/ads/integrated`)

Onyx integration and cross-platform features:

```http
POST /ads/integrated/process-content     # Onyx content processing
POST /ads/integrated/generate-ads        # Onyx-enhanced ads generation
POST /ads/integrated/analyze-competitors # Onyx competitor analysis
POST /ads/integrated/track-performance   # Onyx performance tracking
POST /ads/integrated/onyx-integration    # Onyx feature integration
POST /ads/integrated/cross-platform      # Cross-platform optimization
GET  /ads/integrated/capabilities        # Integration capabilities
```

### Optimized API (`/ads/optimized`)

Production-ready features with optimizations:

```http
POST /ads/optimized/generate             # Optimized ads generation
POST /ads/optimized/remove-background    # Optimized image processing
POST /ads/optimized/analytics            # Optimized analytics tracking
POST /ads/optimized/bulk                 # Bulk operations
POST /ads/optimized/optimize-performance # Performance optimization
GET  /ads/optimized/list                 # Optimized listing
GET  /ads/optimized/stats                # User statistics
DELETE /ads/optimized/{ads_id}           # Delete ads
GET  /ads/optimized/health               # Health check
GET  /ads/optimized/capabilities         # Optimized capabilities
```

## Request/Response Models

### Standardized Request Models

All API endpoints use consistent request models with proper validation:

```python
class ContentRequest(BaseModel):
    content: str = Field(..., min_length=10, max_length=5000)
    context: Dict[str, Any] = Field(default_factory=dict)
    processing_type: str = Field("general", regex="^(general|analysis|optimization|generation)$")
```

### Standardized Response Models

All responses follow consistent patterns with metadata:

```python
class ContentAnalysisResponse(BaseModel):
    content_id: str
    analysis: Dict[str, Any]
    insights: List[str]
    recommendations: List[str]
    processing_time: float
    created_at: datetime
```

## Usage Examples

### Basic Ads Generation

```python
import httpx

async def generate_ads():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/ads/core/generate",
            json={
                "url": "https://example.com/product",
                "type": "ads",
                "prompt": "Generate engaging social media ads",
                "target_audience": "Young professionals aged 25-35",
                "context": "Tech product launch campaign",
                "keywords": ["innovation", "efficiency", "modern"]
            }
        )
        return response.json()
```

### AI-Powered Content Optimization

```python
async def optimize_content():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/ads/ai/optimize-content",
            json={
                "content": "Our tool improves team productivity by 40%",
                "target_audience": "project managers",
                "platform": "linkedin"
            }
        )
        return response.json()
```

### Advanced AI Training

```python
async def train_ai_model():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/ads/advanced/train-ai",
            json={
                "training_data": [
                    {"content": "Sample ad 1", "performance": 0.85},
                    {"content": "Sample ad 2", "performance": 0.92}
                ],
                "model_type": "ads_generation"
            }
        )
        return response.json()
```

### Onyx Integration

```python
async def integrate_onyx():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/ads/integrated/onyx-integration",
            json={
                "content": "Enterprise workflow automation platform",
                "onyx_features": ["content_analysis", "performance_prediction"],
                "integration_level": "advanced"
            }
        )
        return response.json()
```

### Production-Optimized Operations

```python
async def bulk_operations():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/ads/optimized/bulk",
            json={
                "operations": [
                    {"action": "create", "content": "Ad variation 1"},
                    {"action": "create", "content": "Ad variation 2"},
                    {"action": "optimize", "content": "Existing ad"}
                ],
                "operation_type": "create",
                "batch_size": 50
            }
        )
        return response.json()
```

## Running the Demo

The system includes a comprehensive demo that showcases all functionality:

```bash
# Run the API demo
python -m agents.backend.onyx.server.features.ads.api.api_demo

# Or import and run programmatically
from agents.backend.onyx.server.features.ads.api.api_demo import main
import asyncio

results = asyncio.run(main())
```

## Configuration

### Environment Variables

```bash
# Database configuration
ADS_DATABASE_URL=postgresql+asyncpg://user:pass@localhost/db
ADS_REDIS_URL=redis://localhost:6379/0

# API configuration
ADS_API_PREFIX=/api/v1/ads
ADS_RATE_LIMIT=100

# Cache configuration
ADS_CACHE_TTL=3600
ADS_ENABLE_CACHE=true
```

### Settings

The system uses centralized configuration through Pydantic settings:

```python
from agents.backend.onyx.server.features.ads.config.settings import settings

# Access configuration
print(settings.api_prefix)
print(settings.rate_limit)
print(settings.cache_ttl)
```

## Error Handling

All API endpoints use consistent error handling:

```python
try:
    # API operation
    result = await some_operation()
    return result
except Exception as e:
    logger.error(f"Error in operation: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Error Response Format

```json
{
    "detail": "Error message",
    "status_code": 500,
    "timestamp": "2024-01-01T12:00:00Z"
}
```

## Performance Features

### Rate Limiting

The optimized API includes rate limiting per user and operation:

- **Ads Generation**: 100 requests per hour
- **Image Processing**: 200 requests per hour
- **Analytics**: 500 requests per hour

### Caching

- Redis-based caching for frequently accessed data
- Configurable TTL and cache strategies
- Cache warming and invalidation

### Background Processing

- Asynchronous task processing
- Worker pool management
- Batch operations for improved throughput

## Testing

### Unit Tests

```bash
# Run unit tests for API layer
pytest agents/backend/onyx/server/features/ads/api/tests/unit/

# Run specific test file
pytest agents/backend/onyx/server/features/ads/api/tests/unit/test_core_api.py
```

### Integration Tests

```bash
# Run integration tests
pytest agents/backend/onyx/server/features/ads/api/tests/integration/

# Run with coverage
pytest --cov=agents.backend.onyx.server.features.ads.api tests/
```

## Migration Guide

### From Old Scattered APIs

1. **Update imports** to use new unified structure
2. **Replace direct service calls** with use case execution
3. **Update request/response models** to use new DTOs
4. **Migrate error handling** to use new patterns

### Example Migration

**Before (scattered):**
```python
from onyx.server.features.ads.service import AdsService
from onyx.server.features.ads.advanced import AdvancedAdsService

service = AdsService()
result = await service.generate_ads(content)
```

**After (unified):**
```python
from agents.backend.onyx.server.features.ads.application.use_cases import CreateAdUseCase
from agents.backend.onyx.server.features.ads.application.dto import CreateAdRequest

use_case = CreateAdUseCase()
request = CreateAdRequest(prompt=content, ad_type=AdType("ads"))
result = await use_case.execute(request)
```

## Future Enhancements

### Planned Features

1. **GraphQL Support**: Add GraphQL endpoints for complex queries
2. **WebSocket Support**: Real-time updates and notifications
3. **API Versioning**: Semantic versioning for backward compatibility
4. **OpenAPI Documentation**: Auto-generated API documentation
5. **Metrics Dashboard**: Real-time performance monitoring

### Extensibility

The system is designed for easy extension:

- **New API Layers**: Add new router modules following the established pattern
- **Custom Endpoints**: Extend existing routers with additional functionality
- **Middleware Integration**: Add custom middleware for cross-cutting concerns
- **Plugin System**: Support for third-party integrations

## Contributing

### Development Setup

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Set up environment**: Copy `.env.example` to `.env` and configure
4. **Run tests**: `pytest tests/`
5. **Start development server**: `uvicorn main:app --reload`

### Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public functions
- Include unit tests for new functionality
- Update this documentation for API changes

### Pull Request Process

1. **Create feature branch** from `main`
2. **Implement changes** following established patterns
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit pull request** with detailed description

## Support

### Documentation

- **API Reference**: This README and inline code documentation
- **Architecture Guide**: Clean Architecture principles and implementation
- **Migration Guide**: Step-by-step migration from old system
- **Examples**: Code examples and use cases

### Issues and Questions

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Wiki**: Additional documentation and guides

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note**: This API system represents a significant refactoring effort to consolidate scattered implementations into a unified, maintainable architecture. The system follows Clean Architecture principles and provides a solid foundation for future development and extensions.
