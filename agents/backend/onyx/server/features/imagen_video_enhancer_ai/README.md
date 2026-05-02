# Imagen Video Enhancer AI

> Part of the [Blatam Academy Integrated Platform](../README.md)

Image and video enhancement system with SAM3 architecture, integrated with OpenRouter and TruthGPT. Similar to Krea AI, it allows uploading images/videos and enhancing them with artificial intelligence.

## Features

- ✅ SAM3 Architecture for parallel and continuous processing
- ✅ OpenRouter Integration for high-quality LLMs
- ✅ TruthGPT Integration for advanced optimization
- ✅ **Complete REST API with FastAPI** for file uploads
- ✅ **Vision Models Support** - real image analysis
- ✅ **File Validation** - images and videos
- ✅ Continuous 24/7 operation
- ✅ Parallel task execution
- ✅ Automatic task management with priority queue
- ✅ Enhancement services:
  - Image quality enhancement (with visual analysis)
  - Video quality enhancement
  - Intelligent upscaling
  - Noise reduction
  - Color and contrast enhancement
  - Old image restoration

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Configure the environment variables:

```bash
export OPENROUTER_API_KEY="your-api-key"
export TRUTHGPT_ENDPOINT="optional-endpoint"  # Optional
```

## Basic Usage

### Python API Usage

```python
import asyncio
from imagen_video_enhancer_ai import EnhancerAgent, EnhancerConfig

async def main():
    # Create configuration
    config = EnhancerConfig()
    
    # Create agent
    agent = EnhancerAgent(config=config)
    
    # Start agent (24/7 mode)
    # await agent.start()  # In production
    
    # Or use direct methods
    task_id = await agent.enhance_image(
        file_path="path/to/image.jpg",
        enhancement_type="general",
        options={"quality": "high"}
    )
    
    # Wait for result
    import time
    while True:
        status = await agent.get_task_status(task_id)
        if status["status"] == "completed":
            result = await agent.get_task_result(task_id)
            print(result)
            break
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
```

### REST API Usage

Start the API server:

```bash
uvicorn imagen_video_enhancer_ai.api.enhancer_api:app --host 0.0.0.0 --port 8000
```

Then you can use the endpoints:

**Upload and enhance image:**
```bash
curl -X POST "http://localhost:8000/upload-image" \
  -F "file=@image.jpg" \
  -F "enhancement_type=general" \
  -F "priority=5"
```

**Enhance existing image:**
```bash
curl -X POST "http://localhost:8000/enhance-image" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/path/to/image.jpg",
    "enhancement_type": "general",
    "options": {"quality": "high"},
    "priority": 5
  }'
```

**Check task status:**
```bash
curl "http://localhost:8000/task/{task_id}/status"
```

**Get result:**
```bash
curl "http://localhost:8000/task/{task_id}/result"
```

**Batch process multiple files:**
```bash
curl -X POST "http://localhost:8000/batch-process" \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "file_path": "/path/to/image1.jpg",
        "service_type": "enhance_image",
        "enhancement_type": "general"
      },
      {
        "file_path": "/path/to/image2.jpg",
        "service_type": "upscale",
        "options": {"scale_factor": 2}
      }
    ]
  }'
```

**Register webhook:**
```bash
curl -X POST "http://localhost:8000/webhooks/register" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhook",
    "events": ["task.completed", "task.failed"],
    "secret": "your-secret-key"
  }'
```

**View statistics:**
```bash
curl "http://localhost:8000/stats"
```

Interactive documentation is available at: `http://localhost:8000/docs`

## Available Services

### Image Enhancement
```python
task_id = await agent.enhance_image(
    file_path="image.jpg",
    enhancement_type="general",  # general, sharpness, colors, etc.
    options={"quality": "high"}
)
```

### Video Enhancement
```python
task_id = await agent.enhance_video(
    file_path="video.mp4",
    enhancement_type="general",
    options={"fps": 60}
)
```

### Upscaling
```python
task_id = await agent.upscale(
    file_path="image.jpg",
    scale_factor=2,  # 2x, 4x, etc.
    options={"method": "ai"}
)
```

### Noise Reduction
```python
task_id = await agent.denoise(
    file_path="image.jpg",
    noise_level="medium",  # low, medium, high
    options={"preserve_details": True}
)
```

### Restoration
```python
task_id = await agent.restore(
    file_path="old_image.jpg",
    damage_type="scratches",
    options={"preserve_style": True}
)
```

### Color Correction
```python
task_id = await agent.color_correction(
    file_path="image.jpg",
    correction_type="auto",
    options={"vibrance": 1.2}
)
```

## Architecture

The system follows the SAM3 architecture with base classes and common abstractions:
- **Base Models**: Base models with common functionality (`BaseModel`, `TimestampedModel`, `IdentifiedModel`, `StatusModel`)
- **Repository Pattern**: Base repository with common operations (`BaseRepository`, `RepositoryMixin`)
- **Manager Base**: Base manager with lifecycle and statistics (`BaseManager`, `ManagerRegistry`)
- **Base HTTP Client**: Base HTTP client with pooling and retries (`BaseHTTPClient`)
- **Config Manager**: Centralized configuration management (`ConfigManager`)
- **Lifecycle Management**: Lifecycle management system (`LifecycleManager`, `LifecycleComponent`)
- **Dependency Injection**: Dependency injection container (`DependencyContainer`)
- **Component Registry**: Component registry (`ComponentRegistry`)
- **TaskManager**: Task management with persistence (inherits from base classes)
- **ParallelExecutor**: Parallel task execution
- **ServiceHandler**: Handling of different enhancement services
- **OpenRouterClient**: Integration with LLM models (inherits from `BaseHTTPClient`)
- **TruthGPTClient**: Advanced optimization
- **Validation Helpers**: Reusable validations (`ValidationRule`, `ValidationChain`)

## Project Structure

```
imagen_video_enhancer_ai/
├── config/          # Configuration
├── core/            # Core logic
│   ├── base_models.py      # Base models (BaseModel, TimestampedModel, etc.)
│   ├── repository_base.py # Base repository and mixins
│   ├── manager_base.py     # Base manager and registry
│   ├── task_manager.py     # Task management
│   ├── enhancer_agent.py   # Main agent
│   └── services/           # Service handlers
├── infrastructure/  # OpenRouter and TruthGPT clients
├── api/             # REST endpoints with FastAPI
│   ├── routes/      # Routes organized by functionality
│   ├── models.py    # Pydantic models
│   ├── dependencies.py # Shared dependencies
│   └── middleware.py # Middleware (CORS, rate limiting)
├── utils/           # Utilities
│   ├── validation_helpers.py # Validation helpers
│   ├── formatters.py        # Data formatting
│   └── ...          # Other utilities
├── tests/           # Tests
├── examples/        # Usage examples
└── main.py          # Entry point
```

## API Endpoints

### Individual Processing
- `POST /upload-image` - Upload and enhance image
- `POST /upload-video` - Upload and enhance video
- `POST /enhance-image` - Enhance existing image
- `POST /enhance-video` - Enhance existing video
- `POST /upscale` - Upscale
- `POST /denoise` - Reduce noise
- `POST /restore` - Restore image
- `POST /color-correction` - Color correction

### Batch Processing
- `POST /batch-process` - Process multiple files in batch

### Queries
- `GET /task/{task_id}/status` - Task status
- `GET /task/{task_id}/result` - Task result
- `GET /stats` - Agent statistics
- `GET /health` - Health check
- `GET /docs` - Interactive documentation (Swagger)

### Webhooks
- `POST /webhooks/register` - Register webhook
- `DELETE /webhooks/unregister` - Unregister webhook

### Analysis
- `POST /analyze` - Analyze file (image or video)

### Export
- `POST /export-results` - Export results to various formats

### Dashboard and Monitoring
- `GET /dashboard/metrics` - Dashboard metrics
- `GET /dashboard/health` - System health status
- `GET /dashboard/trends` - Performance trends
- `GET /memory/usage` - Memory usage
- `POST /memory/optimize` - Optimize memory

### Authentication
- `POST /auth/generate-key` - Generate API key
- `GET /auth/keys` - List API keys

### Notifications
- `POST /notifications/send` - Send notification
- `GET /notifications/stats` - Notification statistics

### Configuration
- `POST /config/validate` - Validate configuration

### Advanced Metrics
- `GET /metrics/{metric_name}` - Get metric data
- `GET /metrics/{metric_name}/stats` - Metric statistics
- `GET /metrics/{histogram_name}/percentiles` - Histogram percentiles

### Events
- `GET /events/history` - Event history

## Implemented Improvements

### ✅ Complete REST API
- Endpoints for uploading files (images and videos)
- Endpoints for all enhancement services
- Automatic file validation
- Robust error handling

### ✅ Vision Models Support
- Real image analysis using vision models
- Integration with OpenRouter vision API
- Automatic quality and problem analysis
- Detailed guidelines based on real content

### ✅ File Validation
- Format validation (allowed extensions)
- Size validation (configurable limits)
- Parameter validation (types, ranges, etc.)
- Clear and descriptive error messages

### ✅ Enhanced Utilities
- File handling helpers
- Automatic file type detection
- Unique name generation
- MIME types detection

### ✅ Batch Processing
- Parallel processing of multiple files
- Progress tracking
- Per-item error handling
- Result aggregation

### ✅ Cache Management
- Intelligent cache system with TTL
- Automatic invalidation on file modification
- Cache statistics (hits, misses, hit rate)
- Automatic cleanup of expired entries

### ✅ Rate Limiting
- Token bucket rate limiter
- Limits per client (IP)
- Burst support
- Flexible configuration

### ✅ Statistics and Metrics
- Parallel executor statistics
- Cache statistics
- Performance metrics
- `/stats` endpoint for inquiry

### ✅ Webhooks
- Automatic event notifications
- Configurable multiple endpoints
- Security signature (HMAC)
- Automatic retries
- Events: task.created, task.started, task.completed, task.failed, batch.completed

### ✅ Enhanced Logging
- Centralized configuration
- Configurable levels
- Log file support
- Structured format

### ✅ Tests
- Basic tests for main components
- Validator tests
- Fixtures for configuration
- Integration tests

### ✅ Enhanced Video Processing
- Video analysis with OpenCV
- Quality problem detection
- Frame analysis
- Automatic recommendations
- Sample frame extraction

### ✅ Image Utilities
- Detailed image information
- Dimension validation
- Processing time estimation
- Property analysis

### ✅ Enhanced Error Handling
- Custom exceptions
- Consistent error format
- Context in errors
- Decorators for automatic handling

### ✅ Automatic Retry System
- Automatic retries for failed tasks
- Configurable strategies (exponential backoff, fixed delay, etc.)
- Retryable error classification
- Retry history
- Retry statistics

### ✅ Result Export
- Export to multiple formats (JSON, Markdown, CSV, HTML)
- Export of individual or all tasks
- Professional format HTML reports
- CSV export for analysis
- Markdown export for documentation

### ✅ Compression
- JSON result compression
- File compression
- Configurable compression levels
- Automatic decompression
- Compression statistics

### ✅ Monitoring Dashboard
- Real-time metrics
- System health status
- Performance trends
- Metrics history
- Automatic health score
- Trend analysis

### ✅ Memory Optimization
- Memory usage monitoring
- Automatic cache cleanup
- Optimized garbage collector
- Optimization recommendations
- Memory statistics
- Optional aggressive cleanup

### ✅ Plugin System
- Extensible architecture
- Custom plugin registration
- Automatic loading from directory
- Plugin validation
- Plugin execution
- Plugin management (enable/disable)

### ✅ Authentication and Authorization
- API key system
- Granular permissions
- Key expiration
- Key revocation
- Permission validation
- FastAPI integration

### ✅ Notification System
- Multiple channels (email, SMS, push, webhook, Slack, Discord)
- Configurable priorities
- Notification history
- Notification statistics
- Customizable handlers
- Automatic retry

### ✅ Configuration Validation
- Advanced config validation
- Path and permission validation
- Environment validation
- Automatic recommendations
- Dependency validation
- Clear error messages

### ✅ Additional Utilities
- Helpers for common operations
- File and duration formatting
- Unique ID generation
- File hash calculation
- Filename sanitization
- Safe JSON handling
- Retry and throttle decorators
- Result cache with TTL

### ✅ Enhanced Logging
- Automatic log rotation
- Flexible configuration
- Configurable levels
- Log file support
- Structured format
- Automatic cleanup of old logs

### ✅ Optimizations
- Result cache with TTL
- Call throttling
- Optimization decorators
- Automatic cache cleanup
- Efficient memory management

### ✅ Advanced Metrics System
- Time-series metrics
- Counters and gauges
- Histograms with percentiles
- Rate calculation
- Tag filtering
- Aggregated statistics
- Trend analysis

### ✅ Event System
- Pub/sub event bus
- Multiple event types
- Async handlers
- Event history
- Event filtering
- Wildcard subscriptions
- Automatic integration with the agent

### ✅ Complete Integration
- Metrics integrated into the agent
- Automatically published events
- Webhooks with events
- Comprehensive monitoring
- Integration utilities
- Advanced usage examples

### ✅ Backup and Recovery System
- Full and incremental backups
- Backup compression
- Backup restoration
- Backup listing
- Backup deletion
- Utility scripts

### ✅ Utility Scripts
- `backup_tasks.py` - Task and result backup
- `cleanup.py` - Cleanup of old files
- `export_stats.py` - Statistics export
- `dev_setup.py` - Development environment configuration
- Configurable scripts
- Dry-run support

### ✅ Development Tools
- Development utilities (`dev_helpers.py`)
- Timing and logging decorators
- Performance profiler
- Advanced error reporter
- Advanced validators
- Additional unit tests

### ✅ Complete Documentation
- Quick Start Guide
- API Documentation
- Best Practices
- Plugin System
- Deployment Guide
- Development Guide
- Refactoring Guide
- Architecture Guide
- Contributing Guide
- Advanced examples

### ✅ Advanced Utilities
- Config loader with multiple sources
- Type checker for runtime validation
- Enhanced testing helpers
- YAML and JSON support in configuration
- Configuration merge
- Serialization (JSON, Pickle, Base64)
- Response builder for consistent responses
- Data transformers (dict, list, datetime)
- Error context for better tracking
- Data transformation utilities
- Schema validator for schema validation
- API versioning system
- Formatters for data formatting
- Advanced structured logging
- Documentation helpers for doc generation
- Health check system for service monitoring
- Advanced test helpers for testing
- Migration system for schema changes
- Performance profiler for performance analysis
- Feature flags system for gradual rollouts
- Circuit breaker pattern for resilience
- Advanced rate limiter with multiple strategies
- Distributed cache with multiple backends
- Distributed tracing for observability
- Complete observability system for monitoring
- Condition-based alert system
- Advanced metrics with aggregation and percentiles
- Module system for module organization
- Agent builder pattern for flexible construction
- Service registry for service management
- Advanced middleware system for request processing
- Advanced configuration system with validation
- Advanced event system with routing and filtering
- Request validator for request validation
- Data transformer for data transformation
- Advanced serialization system with multiple formats
- Infrastructure consolidation with centralized exports
- Service providers for external services
- Modular initialization system with phases
- Consolidated imports for better organization
- Security system with encryption and hashing
- Audit system for action auditing
- Advanced throttling with multiple strategies
- Queue management with priorities
- Resource pooling for resource management
- Consolidated strategy system for retry, cache, and validation
- Consolidated service configuration
- Centralized validation manager
- Context managers for common operations
- Benchmark system for performance testing
- Performance optimizer with auto-optimization
- Automatic documentation generator
- Dynamic configuration with hot-reloading
- Advanced health checks with dependencies
- Consolidated manager registry
- System integrator for component coordination
- Error recovery system with multiple strategies
- Async utilities for common operations
- Advanced testing helpers
- CI/CD helpers for automation
- Analytics system for usage analysis
- Reporting system for report generation
- Advanced data validator with schemas
- Registry base pattern for all registries
- Executor base pattern for operation execution
- Storage base pattern for storage
- Workflow system for workflow definition and execution
- Pipeline system for data processing
- Orchestrator system for service orchestration
- State management for application state management
- Advanced cache with multiple strategies (LRU, LFU, FIFO, TTL)
- Service base pattern for all services
- Handler base pattern for all handlers
- Processor base pattern for all processors
- Coordinator for component coordination
- Integration system for external service integration
- Data pipeline for data transformation
- Advanced serializer with multiple formats
- Structured logging for structured logging
- Config builder for configuration construction
- Final utilities for final utilities
- Agent component system for component architecture
- Event handler system for event handling
- Factory base pattern for object creation
- Middleware base pattern for request/response processing
- Batch operations system for batch operations
- Scheduler system for task scheduling
- Advanced queue with priorities and scheduling
- Result aggregator for result analysis
- Performance tuner for automatic optimization
- Resource manager for system resource management
- Route decorators for common route decorators
- Response formatter for consistent response formatting
- Request validator for request validation
- Middleware helpers for middleware helpers
- Route builder for route construction with builder pattern
- Data transformer for advanced data transformation
- Cache utils for cache utilities
- Compression utils for data compression
- Encryption utils for encryption and hashing
- File utils advanced for advanced file operations
- Network utils for network utilities
- Advanced service base for advanced services
- Execution context for execution context management
- Advanced error handler for advanced error handling
- Test utils for testing utilities
- Test fixtures for advanced pytest fixtures
- Advanced assertions for advanced assertions
- Test runner for test execution with reporting
- Advanced client base for API clients with common functionality
- Response handler for HTTP response handling
- Advanced logging for structured logging and performance tracking
- Advanced monitoring for system metrics and health checks
- Config base for configuration management with multiple sources
- Advanced config validator for advanced configuration validation
- Code generator system for automatic code generation
- Seeds system for initial data seeding
- Automatic backup system with scheduling and retention
- Deployment utilities for deployment and environment checks
- Enhanced migrations system with async support
- API versioning system with multiple strategies and deprecation
- Distributed cache system with multiple backends and consistency
- Advanced logging system with rotation, filtering, and structured format
- Advanced testing system with fixtures, mocks, and testing utilities
- Advanced request validation system with schemas and rules
- Advanced data transformation system with pipelines and transformers
- Advanced middleware system for FastAPI with advanced processing
- Advanced rate limiting system with multiple strategies and limits per user
- Advanced circuit breaker system with advanced states and failure management
- Telemetry system for data collection and analysis
- Advanced performance profiler with detailed performance analysis
- Real-time metrics system for real-time metrics and aggregation
- Advanced permissions system with RBAC and granular permissions
- Advanced encryption system with multiple algorithms and key management
- Security validator system for security validation and sanitization
- Advanced audit system with detailed tracking and compliance
- Advanced health monitoring system with dependency checks and status aggregation
- Advanced retry system with multiple strategies and exponential backoff
- Advanced queue system with priorities, scheduling, and persistence
- Advanced event bus system with pub/sub, filtering, and event history
- Advanced cache strategy system with multiple eviction policies and TTL
- Advanced validation system with schemas, rules, and custom validators

## Notes

- The system analyzes real images using OpenRouter vision models
- Provides detailed guidelines based on visual analysis
- Similar to Krea AI in functionality but with SAM3 architecture and TruthGPT
- For real image/video processing, integration with additional processing tools is required

---

[← Back to Main README](../README.md)
