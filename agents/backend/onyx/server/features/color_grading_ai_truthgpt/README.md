# Color Grading AI TruthGPT

> Part of the [Blatam Academy Integrated Platform](../README.md)

Complete automatic color grading system with enterprise architecture, integrated with OpenRouter and TruthGPT. Similar to DaVinci Resolve but completely automated.

## 🚀 Key Features

### Processing
- ✅ Video and image processing
- ✅ Advanced color analysis
- ✅ Color matching from references
- ✅ Video quality analysis

### Management
- ✅ Predefined templates (cinematic, vintage, modern, etc.)
- ✅ Custom presets
- ✅ Professional LUTs
- ✅ Complete history
- ✅ Parameter versioning
- ✅ Backup and restore

### Infrastructure
- ✅ Event bus (pub/sub)
- ✅ Security manager
- ✅ Telemetry service
- ✅ Unified task queue
- ✅ Cloud integration (S3)

### Analytics
- ✅ Metrics and statistics
- ✅ Performance monitoring
- ✅ Analytics service
- ✅ Real-time dashboard

### Intelligence
- ✅ Recommendation engine
- ✅ ML optimizer
- ✅ Optimization engine

### Collaboration
- ✅ Webhooks
- ✅ Notifications (Email, Slack, Discord, Telegram)
- ✅ Collaboration manager
- ✅ Workflow manager

### Resilience
- ✅ Circuit breaker
- ✅ Retry manager
- ✅ Load balancer
- ✅ Feature flags

### Traffic Control
- ✅ Rate limiter
- ✅ Throttle manager
- ✅ Backpressure manager

### Lifecycle Management
- ✅ Health monitor
- ✅ Graceful shutdown
- ✅ Lifecycle manager

### Compliance & Audit
- ✅ Audit logger
- ✅ Compliance manager (GDPR, CCPA, HIPAA, SOC2, ISO27001)

## 📊 Project Statistics

- **Total services**: 63+
- **Categories**: 11
- **Design patterns**: 8+
- **Base components**: 5
- **Utilities**: 10+
- **Decorators**: 4

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full architecture details.

### Main Structure
```
core/              # Base components and agents
services/          # 63+ organized services
infrastructure/    # Clients (OpenRouter, TruthGPT)
api/              # Full REST API
config/           # Configuration
```

### Service Categories
1. **Processing** (5): Video, Image, Color Analysis, Color Matching, Quality
2. **Management** (7): Templates, Presets, LUTs, Cache, History, Version, Backup
3. **Infrastructure** (5): EventBus, Security, Telemetry, Queue, Cloud
4. **Analytics** (4): Metrics, Performance Monitor, Performance Optimizer, Analytics
5. **Intelligence** (3): Recommendations, ML Optimizer, Optimization Engine
6. **Collaboration** (4): Webhooks, Notifications, Collaboration, Workflow
7. **Resilience** (4): Circuit Breaker, Retry, Load Balancer, Feature Flags
8. **Traffic Control** (3): Rate Limiter, Throttle, Backpressure
9. **Lifecycle** (3): Health Monitor, Graceful Shutdown, Lifecycle Manager
10. **Compliance** (2): Audit Logger, Compliance Manager
11. **Support** (23+): Batch, Comparison, Export, and more...

## 📦 Installation

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### Environment Variables

```bash
export OPENROUTER_API_KEY="your-api-key"
export TRUTHGPT_ENDPOINT="optional-endpoint"  # Optional
export FFMPEG_PATH="/path/to/ffmpeg"  # Optional, defaults to "ffmpeg"
```

### Programmatic Configuration

```python
from color_grading_ai_truthgpt import ColorGradingConfig

config = ColorGradingConfig()
config.openrouter.api_key = "your-api-key"
config.max_parallel_tasks = 10
config.enable_cache = True
config.cache_ttl = 3600
```

## 🎯 Basic Usage

### Python API

```python
import asyncio
from color_grading_ai_truthgpt import UnifiedColorGradingAgent, ColorGradingConfig

async def main():
    # Create configuration
    config = ColorGradingConfig()
    
    # Create unified agent (recommended)
    agent = UnifiedColorGradingAgent(config=config)
    
    # Apply template to video
    result = await agent.grade_video(
        video_path="input.mp4",
        template_name="cinematic"
    )
    print(f"Processed video: {result['output_path']}")
    
    # Apply color matching from reference image
    result = await agent.grade_image(
        image_path="input.jpg",
        reference_image="reference.jpg"
    )
    print(f"Processed image: {result['output_path']}")
    
    # Analyze color
    analysis = await agent.analyze_media("input.mp4")
    print(f"Analysis: {analysis}")

asyncio.run(main())
```

### REST API

```bash
# Start server
uvicorn color_grading_ai_truthgpt.api.color_grading_api:app --reload

# Apply color grading
curl -X POST "http://localhost:8000/api/v1/grade/video" \
  -F "file=@input.mp4" \
  -F "template_name=cinematic"

# List templates
curl "http://localhost:8000/api/v1/templates"
```

## 📚 Documentation

See [DOCUMENTATION_INDEX.md](DOCUMENTATION_INDEX.md) for full documentation index.

### Main Documents
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete architecture
- [REFACTORING_FINAL_COMPLETE.md](REFACTORING_FINAL_COMPLETE.md) - Final refactoring
- [RESILIENCE_PATTERNS.md](RESILIENCE_PATTERNS.md) - Resilience patterns
- [TRAFFIC_CONTROL.md](TRAFFIC_CONTROL.md) - Traffic control
- [LIFECYCLE_MANAGEMENT.md](LIFECYCLE_MANAGEMENT.md) - Lifecycle management
- [COMPLIANCE_AUDIT.md](COMPLIANCE_AUDIT.md) - Compliance and audit
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guides

## 🔧 Key Components

### Agents
- **UnifiedColorGradingAgent** ⭐ (Recommended)
- ColorGradingAgent (Original, compatible)
- RefactoredColorGradingAgent (Refactored, compatible)

### Factories
- **RefactoredServiceFactory** ⭐ (Recommended)
- ServiceFactory (Original, compatible)

### Base Components
- BaseService - Base for all services
- FileManagerBase - Base for file managers
- ConfigManager - Configuration management
- ServiceGroups - Logical grouping
- ServiceAccessor - Unified access

## 🎨 Examples

### Predefined Template
```python
result = await agent.grade_video(
    video_path="input.mp4",
    template_name="vintage"
)
```

### Color Matching
```python
result = await agent.grade_video(
    video_path="input.mp4",
    reference_video="reference.mp4"
)
```

### Text Description
```python
result = await agent.grade_image(
    image_path="input.jpg",
    description="warm sunset colors with high contrast"
)
```

### Custom Parameters
```python
result = await agent.grade_video(
    video_path="input.mp4",
    color_params={
        "brightness": 1.1,
        "contrast": 1.2,
        "saturation": 1.15,
        "temperature": 5500
    }
)
```

## 🚀 Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete guides.

### Docker
```bash
docker-compose up -d
```

### Kubernetes
```bash
kubectl apply -f k8s/
```

## 📈 Enterprise Features

- ✅ **Resilience**: Circuit breaker, retry, load balancing
- ✅ **Observability**: Health monitoring, performance tracking, telemetry
- ✅ **Security**: Security manager, input validation, threat detection
- ✅ **Scalability**: Load balancing, resource pooling, cloud integration
- ✅ **Compliance**: GDPR, CCPA, HIPAA, SOC2, ISO27001
- ✅ **Audit**: Complete audit logging

## 🤝 Contributing

The project is fully functional and production-ready. To contribute:

1. Review [ARCHITECTURE.md](ARCHITECTURE.md)
2. Follow established patterns
3. Use available base components
4. Add tests for new features

## 📄 License

Proprietary - Blatam Academy

## 🙏 Acknowledgments

- OpenRouter for LLM integration
- TruthGPT for advanced optimization
- Open source community

---

[← Back to Main README](../README.md)
