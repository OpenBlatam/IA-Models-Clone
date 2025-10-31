# AI Video System - Upgrade Guide

## Overview

This guide provides step-by-step instructions for upgrading from the previous version of the AI Video System to the enhanced production-ready version with all new features and improvements.

## 🚀 What's New

### Major Enhancements
- **Performance Optimization**: Advanced caching, connection pooling, rate limiting
- **Security Framework**: Input validation, encryption, session management, security auditing
- **Async Utilities**: Task management, retry mechanisms, batch processing
- **Monitoring & Observability**: Metrics collection, health checks, alerting
- **Validation Framework**: Schema validation, data validation, custom validators
- **Advanced Logging**: Structured logging, performance logging, security logging

### Breaking Changes
- Logging format changed to JSON
- Exception hierarchy updated
- Configuration structure modified
- Plugin API enhanced
- Import paths changed

## 📋 Pre-Upgrade Checklist

### 1. Backup Current System
```bash
# Backup current code
cp -r ai_video ai_video_backup

# Backup configuration
cp config.json config_backup.json

# Backup logs
cp -r logs logs_backup
```

### 2. Check Dependencies
```bash
# Check current Python version (3.8+ required)
python --version

# Check current dependencies
pip freeze > requirements_current.txt
```

### 3. Review Current Usage
- Document current configuration
- List all custom plugins
- Note any custom logging or monitoring
- Identify performance bottlenecks

## 🔄 Upgrade Process

### Step 1: Update Dependencies

#### Install New Requirements
```bash
# Install enhanced requirements
pip install -r requirements_unified.txt

# Or install production requirements
pip install -r requirements_production.txt
```

#### Verify Installation
```python
# Test imports
from ai_video.core import (
    AIVideoError, performance_monitor, security_auditor,
    metrics_collector, health_checker, alert_manager
)
print("Enhanced modules imported successfully")
```

### Step 2: Update Configuration

#### New Configuration Structure
```json
{
  "system": {
    "name": "AI Video System",
    "version": "2.0.0",
    "environment": "production"
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "ai_video.log",
    "max_file_size_mb": 100,
    "backup_count": 5,
    "directory": "logs"
  },
  "security": {
    "encryption_key": "your-secret-key",
    "max_input_length": 10000,
    "session_timeout_minutes": 60,
    "max_sessions_per_user": 5
  },
  "performance": {
    "cache_ttl": 3600,
    "max_concurrent_tasks": 100,
    "rate_limit_requests_per_minute": 60,
    "slow_query_threshold": 1.0
  },
  "monitoring": {
    "health_check_interval": 60,
    "metrics_retention_hours": 24,
    "alert_retention_days": 30
  },
  "plugins": {
    "directory": "plugins",
    "auto_load": true,
    "validation": true
  },
  "workflow": {
    "max_retries": 3,
    "timeout_seconds": 300,
    "parallel_processing": true
  }
}
```

#### Environment Variables
```bash
# Add new environment variables
export AI_VIDEO_LOG_LEVEL=INFO
export AI_VIDEO_CONFIG_PATH=/path/to/config.json
export AI_VIDEO_ENCRYPTION_KEY=your-secret-key
export AI_VIDEO_ENVIRONMENT=production
```

### Step 3: Update Code

#### Update Imports
```python
# Old imports
from ai_video.exceptions import VideoError
from ai_video.utils import log_message

# New imports
from ai_video.core import AIVideoError, main_logger
from ai_video.core import performance_monitor, security_auditor
```

#### Update Exception Handling
```python
# Old exception handling
try:
    result = generate_video(request)
except VideoError as e:
    log_message(f"Error: {e}")

# New exception handling
try:
    result = await system.generate_video(request)
except AIVideoError as e:
    main_logger.error(f"Video generation error: {e}", exc_info=True)
    security_auditor.log_security_event("video_generation_error", "error", str(e))
```

#### Add Performance Monitoring
```python
# Old code
def generate_video(request):
    # Video generation logic
    pass

# New code with performance monitoring
from ai_video.core import measure_performance

@measure_performance("video_generation")
async def generate_video(request):
    # Video generation logic
    pass
```

#### Add Security Validation
```python
# Old code
def process_request(request):
    # Process request without validation
    pass

# New code with security validation
from ai_video.core import input_validator, security_auditor

async def process_request(request):
    # Validate input
    is_valid, sanitized = input_validator.validate_string(
        request.input_text, max_length=10000, allow_html=False
    )
    if not is_valid:
        raise ValidationError(f"Input validation failed: {sanitized}")
    
    # Log security event
    security_auditor.log_data_access(
        request.user_id, "video_generation", "create", request.request_id
    )
    
    # Process request
    pass
```

### Step 4: Update Plugins

#### Plugin Structure Changes
```python
# Old plugin structure
class VideoPlugin:
    def process(self, data):
        # Plugin logic
        pass

# New plugin structure
from ai_video.core import PluginError, performance_monitor

class VideoPlugin:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(f"plugin.{self.__class__.__name__}")
    
    @measure_performance("plugin_execution")
    async def process(self, data):
        try:
            # Plugin logic
            result = await self._process_data(data)
            
            # Record metrics
            performance_monitor.record_operation(
                "plugin_execution", time.time() - start_time, "success"
            )
            
            return result
        except Exception as e:
            self.logger.error(f"Plugin execution failed: {e}", exc_info=True)
            raise PluginError(f"Plugin processing failed: {e}")
```

#### Plugin Configuration
```json
{
  "plugins": {
    "video_enhancer": {
      "enabled": true,
      "priority": 1,
      "config": {
        "quality": "high",
        "format": "mp4"
      }
    }
  }
}
```

### Step 5: Update Logging

#### Logging Configuration
```python
# Old logging
import logging
logging.info("Processing video")

# New structured logging
from ai_video.core import main_logger, log_event

main_logger.info("Processing video", extra={
    'extra_fields': {
        'request_id': request_id,
        'user_id': user_id,
        'operation': 'video_processing'
    }
})

log_event("video_processing_started", {
    "request_id": request_id,
    "user_id": user_id
})
```

#### Performance Logging
```python
from ai_video.core import performance_logger

@performance_logger.time_function("video_generation")
async def generate_video(request):
    # Video generation logic
    pass
```

### Step 6: Add Monitoring

#### Health Checks
```python
from ai_video.core import health_checker

# Register custom health check
def check_database_connection():
    try:
        # Database connection check
        return True
    except Exception:
        return False

health_checker.register_health_check("database", check_database_connection)
```

#### Metrics Collection
```python
from ai_video.core import metrics_collector

# Record custom metrics
metrics_collector.record_metric("api_requests", 1, {"endpoint": "/video"})
metrics_collector.record_metric("video_generation_duration", duration)
```

#### Alerting
```python
from ai_video.core import alert_manager

# Create alerts
alert_manager.create_alert(
    "warning",
    "High error rate detected",
    "API monitoring",
    metadata={"error_rate": 0.05}
)
```

## 🔧 Migration Scripts

### Configuration Migration
```python
#!/usr/bin/env python3
"""
Configuration migration script
"""

import json
import os
from pathlib import Path

def migrate_config():
    """Migrate old configuration to new format."""
    
    # Load old config
    old_config_path = Path("config_old.json")
    if not old_config_path.exists():
        print("Old configuration not found")
        return
    
    with open(old_config_path) as f:
        old_config = json.load(f)
    
    # Create new config structure
    new_config = {
        "system": {
            "name": "AI Video System",
            "version": "2.0.0",
            "environment": os.getenv("AI_VIDEO_ENVIRONMENT", "production")
        },
        "logging": {
            "level": old_config.get("log_level", "INFO"),
            "format": "json",
            "file": "ai_video.log",
            "max_file_size_mb": 100,
            "backup_count": 5,
            "directory": "logs"
        },
        "security": {
            "encryption_key": os.getenv("AI_VIDEO_ENCRYPTION_KEY", "default-key"),
            "max_input_length": 10000,
            "session_timeout_minutes": 60,
            "max_sessions_per_user": 5
        },
        "performance": {
            "cache_ttl": 3600,
            "max_concurrent_tasks": 100,
            "rate_limit_requests_per_minute": 60,
            "slow_query_threshold": 1.0
        },
        "monitoring": {
            "health_check_interval": 60,
            "metrics_retention_hours": 24,
            "alert_retention_days": 30
        },
        "plugins": {
            "directory": "plugins",
            "auto_load": True,
            "validation": True
        },
        "workflow": {
            "max_retries": 3,
            "timeout_seconds": 300,
            "parallel_processing": True
        }
    }
    
    # Save new config
    with open("config_new.json", "w") as f:
        json.dump(new_config, f, indent=2)
    
    print("Configuration migrated successfully")

if __name__ == "__main__":
    migrate_config()
```

### Code Migration Helper
```python
#!/usr/bin/env python3
"""
Code migration helper script
"""

import re
from pathlib import Path

def update_imports(file_path):
    """Update imports in a Python file."""
    
    with open(file_path) as f:
        content = f.read()
    
    # Update import patterns
    replacements = [
        (r'from ai_video\.exceptions import VideoError', 
         'from ai_video.core import AIVideoError'),
        (r'from ai_video\.utils import log_message', 
         'from ai_video.core import main_logger'),
        (r'VideoError', 'AIVideoError'),
        (r'log_message\(', 'main_logger.info('),
    ]
    
    for old_pattern, new_pattern in replacements:
        content = re.sub(old_pattern, new_pattern, content)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Updated imports in {file_path}")

def migrate_directory(directory):
    """Migrate all Python files in a directory."""
    
    for py_file in Path(directory).rglob("*.py"):
        if "migration" not in py_file.name:
            update_imports(py_file)

if __name__ == "__main__":
    migrate_directory(".")
```

## 🧪 Testing

### Unit Tests
```python
#!/usr/bin/env python3
"""
Test the upgraded system
"""

import asyncio
import pytest
from ai_video.core import get_system, AIVideoError
from ai_video.models import VideoRequest

async def test_system_initialization():
    """Test system initialization."""
    system = await get_system()
    assert system.is_initialized
    
    status = await system.get_system_status()
    assert status["status"] in ["operational", "degraded"]

async def test_video_generation():
    """Test video generation."""
    system = await get_system()
    
    request = VideoRequest(
        input_text="Test video generation",
        user_id="test_user"
    )
    
    response = await system.generate_video(request)
    assert response is not None
    assert response.request_id == request.request_id

async def test_error_handling():
    """Test error handling."""
    system = await get_system()
    
    # Test with invalid input
    request = VideoRequest(
        input_text="",  # Empty input should fail validation
        user_id="test_user"
    )
    
    with pytest.raises(AIVideoError):
        await system.generate_video(request)

if __name__ == "__main__":
    asyncio.run(test_system_initialization())
    asyncio.run(test_video_generation())
    asyncio.run(test_error_handling())
    print("All tests passed!")
```

### Integration Tests
```python
#!/usr/bin/env python3
"""
Integration tests for the upgraded system
"""

import asyncio
import time
from ai_video.core import (
    get_system, metrics_collector, health_checker,
    alert_manager, performance_monitor
)

async def test_full_workflow():
    """Test complete video generation workflow."""
    
    system = await get_system()
    
    # Test system status
    status = await system.get_system_status()
    print(f"System status: {status['status']}")
    
    # Test health checks
    health_results = await health_checker.run_all_health_checks()
    print(f"Health checks: {len(health_results)} passed")
    
    # Test metrics
    metrics = await system.get_metrics()
    print(f"Metrics collected: {len(metrics)}")
    
    # Test video generation
    request = VideoRequest(
        input_text="Integration test video",
        user_id="integration_test"
    )
    
    start_time = time.time()
    response = await system.generate_video(request)
    duration = time.time() - start_time
    
    print(f"Video generated in {duration:.2f} seconds")
    print(f"Response: {response}")
    
    # Test performance metrics
    perf_stats = performance_monitor.get_operation_stats("video_generation")
    print(f"Performance stats: {perf_stats}")

if __name__ == "__main__":
    asyncio.run(test_full_workflow())
```

## 🚀 Deployment

### Docker Deployment
```dockerfile
# Dockerfile for upgraded system
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements_production.txt .
RUN pip install -r requirements_production.txt

# Copy application
COPY . .

# Create log directory
RUN mkdir -p logs

# Set environment variables
ENV AI_VIDEO_ENVIRONMENT=production
ENV AI_VIDEO_LOG_LEVEL=INFO

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "ai_video.main"]
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-video:
    build: .
    ports:
      - "8000:8000"
    environment:
      - AI_VIDEO_ENVIRONMENT=production
      - AI_VIDEO_LOG_LEVEL=INFO
      - AI_VIDEO_ENCRYPTION_KEY=${ENCRYPTION_KEY}
    volumes:
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    
  monitoring:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped
```

## 🔍 Troubleshooting

### Common Issues

#### Import Errors
```python
# Error: ModuleNotFoundError: No module named 'ai_video.core'
# Solution: Install new requirements
pip install -r requirements_production.txt
```

#### Configuration Errors
```python
# Error: ConfigurationError: Invalid configuration
# Solution: Use migration script to update config
python migrate_config.py
```

#### Logging Issues
```python
# Error: Logging format issues
# Solution: Update logging configuration
setup_logging(LogConfig(format="json"))
```

#### Performance Issues
```python
# Error: Slow performance
# Solution: Enable caching and monitoring
from ai_video.core import default_cache, performance_monitor
```

### Debug Mode
```python
# Enable debug mode
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed monitoring
from ai_video.core import metrics_collector
metrics_collector.record_metric("debug_mode", 1)
```

## 📊 Post-Upgrade Verification

### 1. System Health Check
```python
from ai_video.core import get_system

async def verify_system():
    system = await get_system()
    status = await system.get_system_status()
    
    print(f"System Status: {status['status']}")
    print(f"Health Checks: {status['components']['health']}")
    print(f"Active Alerts: {status['alerts']['active_count']}")
    
    return status['status'] == 'operational'
```

### 2. Performance Verification
```python
from ai_video.core import performance_monitor, metrics_collector

def verify_performance():
    # Check performance metrics
    perf_stats = performance_monitor.get_operation_stats("video_generation")
    print(f"Performance Stats: {perf_stats}")
    
    # Check system metrics
    system_metrics = metrics_collector.get_system_metrics()
    print(f"System Metrics: {system_metrics}")
    
    return perf_stats is not None
```

### 3. Security Verification
```python
from ai_video.core import security_auditor, input_validator

def verify_security():
    # Test input validation
    is_valid, result = input_validator.validate_string("test", max_length=100)
    print(f"Input Validation: {is_valid}")
    
    # Check security events
    security_events = security_auditor.get_security_report()
    print(f"Security Events: {len(security_events['recent_events'])}")
    
    return is_valid
```

## 🎯 Next Steps

### 1. Monitor System
- Set up monitoring dashboards
- Configure alerting
- Review performance metrics
- Monitor security events

### 2. Optimize Performance
- Tune cache settings
- Optimize database queries
- Configure rate limits
- Monitor resource usage

### 3. Enhance Security
- Review security logs
- Configure additional validators
- Set up security monitoring
- Implement additional checks

### 4. Scale System
- Configure load balancing
- Set up clustering
- Optimize for high availability
- Plan for growth

## 📞 Support

### Documentation
- [Enhanced Features](ENHANCED_FEATURES.md)
- [Production Guide](PRODUCTION_GUIDE.md)
- [API Documentation](README_UNIFIED.md)

### Troubleshooting
- Check logs in `logs/` directory
- Review monitoring dashboards
- Check system status endpoint
- Review error alerts

### Getting Help
- Review this upgrade guide
- Check system documentation
- Review error logs
- Contact support team

This upgrade guide provides comprehensive instructions for migrating to the enhanced AI Video System with all new features and improvements. 