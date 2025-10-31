# AI History Comparison System - Quick Start Guide

## Overview

This guide will help you get started with the refactored AI History Comparison System quickly and easily.

## Prerequisites

- Python 3.8+
- pip or conda package manager
- Basic understanding of FastAPI and Python

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_ultimate_complete.txt
   ```

2. **Set Environment Variables** (optional):
   ```bash
   export ENVIRONMENT=development
   export DEBUG=true
   export HOST=0.0.0.0
   export PORT=8000
   ```

## Running the System

1. **Start the Application**:
   ```bash
   python main.py
   ```

2. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - System Status: http://localhost:8000/api/v1/system/status

## API Usage Examples

### 1. Content Analysis

**Analyze a single piece of content**:
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is a sample text for analysis.",
    "analysis_type": "comprehensive"
  }'
```

**Batch analysis**:
```bash
curl -X POST "http://localhost:8000/api/v1/analysis/analyze/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "contents": [
      "First content piece to analyze.",
      "Second content piece to analyze."
    ],
    "analysis_type": "comprehensive"
  }'
```

### 2. Content Comparison

**Compare two content pieces**:
```bash
curl -X POST "http://localhost:8000/api/v1/comparison/content" \
  -H "Content-Type: application/json" \
  -d '{
    "content1": "This is the first content piece.",
    "content2": "This is the second content piece.",
    "comparison_type": "similarity"
  }'
```

**Find similar content**:
```bash
curl -X POST "http://localhost:8000/api/v1/comparison/similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Content to find similar items for",
    "threshold": 0.8,
    "limit": 10
  }'
```

### 3. Trend Analysis

**Analyze trends**:
```bash
curl -X POST "http://localhost:8000/api/v1/trends/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"timestamp": 1, "value": 0.8},
      {"timestamp": 2, "value": 0.85},
      {"timestamp": 3, "value": 0.9}
    ],
    "metric": "value",
    "time_window": 30
  }'
```

**Predict future values**:
```bash
curl -X POST "http://localhost:8000/api/v1/trends/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"timestamp": 1, "value": 0.8},
      {"timestamp": 2, "value": 0.85},
      {"timestamp": 3, "value": 0.9}
    ],
    "metric": "value",
    "prediction_days": 7
  }'
```

### 4. Content Management

**Create content**:
```bash
curl -X POST "http://localhost:8000/api/v1/content/create" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "This is new content to store.",
    "content_type": "text",
    "metadata": {"author": "user1", "category": "example"}
  }'
```

**Search content**:
```bash
curl -X POST "http://localhost:8000/api/v1/content/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search term",
    "limit": 10,
    "offset": 0
  }'
```

### 5. System Management

**Get system status**:
```bash
curl -X GET "http://localhost:8000/api/v1/system/status"
```

**Get system metrics**:
```bash
curl -X GET "http://localhost:8000/api/v1/system/metrics"
```

**Get configuration**:
```bash
curl -X GET "http://localhost:8000/api/v1/system/configuration"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Environment (development/staging/production) | development |
| `DEBUG` | Debug mode | true |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 8000 |
| `DATABASE_URL` | Database connection URL | sqlite:///./ai_history.db |
| `REDIS_URL` | Redis connection URL | redis://localhost:6379 |
| `SECRET_KEY` | Secret key for security | your-secret-key-here |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |

### Feature Flags

The system supports feature flags to enable/disable functionality:

```python
# In your configuration
features = {
    "content_analysis": True,
    "trend_analysis": True,
    "comparison_engine": True,
    "real_time_streaming": True,
    "advanced_analytics": True,
    "ai_governance": True,
    "content_lifecycle": True,
    "security_monitoring": True,
    "quantum_computing": False,
    "federated_learning": False,
    "neural_architecture_search": False
}
```

## Development

### Adding New Analyzers

1. Create a new analyzer class:
```python
from ..core.base import BaseAnalyzer
from ..core.interfaces import IAnalyzer

class MyAnalyzer(BaseAnalyzer[str], IAnalyzer[str]):
    async def _analyze(self, data: str, **kwargs) -> Dict[str, Any]:
        # Your analysis logic here
        return {"result": "analysis_complete"}
    
    def get_analysis_metrics(self) -> List[str]:
        return ["metric1", "metric2"]
```

2. Add to `analyzers/__init__.py`:
```python
from .my_analyzer import MyAnalyzer

__all__ = [
    'MyAnalyzer',
    # ... other analyzers
]
```

### Adding New API Endpoints

1. Create a new endpoint module in `api/endpoints/`:
```python
from fastapi import APIRouter

router = APIRouter()

@router.post("/my-endpoint")
async def my_endpoint():
    return {"message": "Hello from my endpoint"}
```

2. Add to the main router in `api/router.py`:
```python
from .endpoints.my_endpoints import router as my_router

router.include_router(
    my_router,
    prefix="/my-feature",
    tags=["My Feature"]
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed
2. **Port Already in Use**: Change the PORT environment variable
3. **Database Connection Issues**: Check DATABASE_URL configuration
4. **Redis Connection Issues**: Check REDIS_URL configuration

### Logs

The system provides comprehensive logging:
- Application logs: Console output
- Error logs: Detailed error information
- Access logs: HTTP request logs

### Health Checks

Monitor system health:
```bash
curl -X GET "http://localhost:8000/health"
curl -X GET "http://localhost:8000/api/v1/system/status"
```

## Next Steps

1. **Explore the API**: Use the interactive documentation at `/docs`
2. **Configure Features**: Enable/disable features as needed
3. **Add Integrations**: Connect external services
4. **Customize Analysis**: Add your own analyzers
5. **Scale**: Deploy to production with proper configuration

## Support

For more information:
- Check the `REFACTORING_SUMMARY.md` for detailed architecture information
- Review the API documentation at `/docs`
- Examine the source code in the organized modules
- Check the configuration options in `core/config.py`





















