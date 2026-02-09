# Modular Architecture Guide

## 🏗️ Architecture Overview

The AWS deployment now uses a **modular plugin architecture** that follows microservices principles:

- **Plugin-based**: Each feature is a pluggable module
- **Dependency Injection**: Services are injected via interfaces
- **Configuration-driven**: Enable/disable features via configuration
- **Separation of Concerns**: Each plugin has a single responsibility
- **Testable**: Easy to mock and test individual components

## 📁 Structure

```
aws/
├── core/                          # Core framework
│   ├── interfaces.py              # Abstract interfaces
│   ├── plugin_manager.py          # Plugin registry and manager
│   ├── config_manager.py          # Modular configuration
│   └── app_factory.py             # Application factory
├── plugins/                       # Pluggable features
│   ├── middleware/                # Middleware plugins
│   │   ├── tracing_plugin.py
│   │   ├── rate_limiting_plugin.py
│   │   ├── circuit_breaker_plugin.py
│   │   ├── caching_plugin.py
│   │   ├── logging_plugin.py
│   │   └── security_headers_plugin.py
│   ├── monitoring/                # Monitoring plugins
│   │   └── prometheus_plugin.py
│   ├── security/                  # Security plugins
│   │   └── oauth2_plugin.py
│   └── messaging/                 # Messaging plugins
│       └── kafka_plugin.py
└── api_integration.py             # Main integration point
```

## 🔌 Plugin System

### Plugin Interface

All plugins implement one of these interfaces:

- `MiddlewarePlugin`: HTTP middleware
- `MonitoringPlugin`: Monitoring and metrics
- `SecurityPlugin`: Authentication and authorization
- `MessagingPlugin`: Event publishing/subscribing
- `CachePlugin`: Caching layer
- `WorkerPlugin`: Background task processing

### Creating a Custom Plugin

```python
from aws.core.interfaces import MiddlewarePlugin
from fastapi import FastAPI
from typing import Dict, Any

class MyCustomPlugin(MiddlewarePlugin):
    def get_name(self) -> str:
        return "my_custom"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        return config.get("enable_my_custom", False)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        # Your setup logic here
        return app
```

### Registering Plugins

```python
from aws.core.app_factory import AppFactory
from aws.core.config_manager import AppConfig

factory = AppFactory()
factory.plugin_manager.register_plugin(MyCustomPlugin())
app = factory.create_app(robot_config)
```

## ⚙️ Configuration

### Modular Configuration

Configuration is organized by feature:

```python
from aws.core.config_manager import AppConfig

# From environment variables
config = AppConfig.from_env()

# From JSON file
config = AppConfig.from_file("config.json")

# Custom configuration
config = AppConfig(
    middleware=MiddlewareConfig(
        enable_tracing=True,
        enable_rate_limiting=True,
    ),
    monitoring=MonitoringConfig(
        enable_prometheus=True,
    ),
    # ... other configs
)
```

### Configuration Structure

```json
{
  "middleware": {
    "enable_tracing": true,
    "enable_rate_limiting": true,
    "enable_circuit_breaker": true,
    "enable_caching": true,
    "redis_url": "redis://localhost:6379"
  },
  "monitoring": {
    "enable_prometheus": true,
    "enable_cloudwatch": true
  },
  "security": {
    "enable_oauth2": true,
    "jwt_secret_key": "your-secret-key"
  },
  "messaging": {
    "enable_kafka": false,
    "kafka_bootstrap_servers": "localhost:9092"
  },
  "worker": {
    "enable_celery": true,
    "celery_broker_url": "redis://localhost:6379"
  },
  "cache": {
    "enable_redis": true,
    "redis_url": "redis://localhost:6379"
  }
}
```

## 🚀 Usage

### Basic Usage

```python
from robot_movement_ai.config.robot_config import RobotConfig
from aws.api_integration import create_advanced_robot_app

robot_config = RobotConfig()
app = create_advanced_robot_app(robot_config)
```

### Custom Configuration

```python
from aws.core.config_manager import AppConfig, MiddlewareConfig
from aws.api_integration import create_advanced_robot_app

app_config = AppConfig(
    middleware=MiddlewareConfig(
        enable_tracing=False,  # Disable tracing
        enable_rate_limiting=True,
    )
)

app = create_advanced_robot_app(robot_config, app_config)
```

### Accessing Plugins

```python
from fastapi import Request

@app.get("/api/v1/status")
async def status(request: Request):
    plugin_manager = request.app.state.plugin_manager
    
    # Get messaging plugin
    messaging = plugin_manager.registry.get_messaging_plugin()
    if messaging:
        messaging.publish("event.type", {"data": "value"})
    
    # Get cache plugin
    cache = plugin_manager.registry.get_cache_plugin()
    if cache:
        value = await cache.get("key")
    
    return {"status": "ok"}
```

## 🔧 Extending the System

### Adding a New Middleware Plugin

1. Create plugin class:

```python
# aws/plugins/middleware/my_plugin.py
from aws.core.interfaces import MiddlewarePlugin
from fastapi import FastAPI
from typing import Dict, Any

class MyMiddlewarePlugin(MiddlewarePlugin):
    def get_name(self) -> str:
        return "my_middleware"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        return config.get("middleware", {}).get("enable_my_middleware", False)
    
    def setup(self, app: FastAPI, config: Dict[str, Any]) -> FastAPI:
        # Add middleware
        from fastapi.middleware.base import BaseHTTPMiddleware
        
        class MyMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                # Your logic
                return await call_next(request)
        
        app.add_middleware(MyMiddleware)
        return app
```

2. Register in factory:

```python
# aws/core/app_factory.py
from aws.plugins.middleware.my_plugin import MyMiddlewarePlugin

class AppFactory:
    def _register_default_plugins(self):
        # ... existing plugins
        self.plugin_manager.register_plugin(MyMiddlewarePlugin())
```

### Adding a New Messaging Plugin

```python
from aws.core.interfaces import MessagingPlugin
from typing import Dict, Any, Optional

class RabbitMQPlugin(MessagingPlugin):
    def get_name(self) -> str:
        return "rabbitmq"
    
    def is_enabled(self, config: Dict[str, Any]) -> bool:
        return config.get("messaging", {}).get("enable_rabbitmq", False)
    
    def publish(self, event_type: str, data: Dict[str, Any], key: Optional[str] = None) -> bool:
        # RabbitMQ publish logic
        pass
    
    def subscribe(self, topic: str, handler: callable) -> bool:
        # RabbitMQ subscribe logic
        pass
```

## 🧪 Testing

### Testing Plugins

```python
import pytest
from aws.plugins.middleware.tracing_plugin import TracingMiddlewarePlugin

def test_tracing_plugin():
    plugin = TracingMiddlewarePlugin()
    assert plugin.get_name() == "tracing"
    
    config = {"middleware": {"enable_tracing": True}}
    assert plugin.is_enabled(config) == True
    
    # Test setup
    from fastapi import FastAPI
    app = FastAPI()
    app = plugin.setup(app, config)
    assert app is not None
```

### Mocking Plugins

```python
from unittest.mock import Mock
from aws.core.plugin_manager import PluginManager

def test_with_mocked_plugin():
    plugin = Mock(spec=MiddlewarePlugin)
    plugin.get_name.return_value = "mock"
    plugin.is_enabled.return_value = True
    plugin.setup.return_value = FastAPI()
    
    manager = PluginManager({})
    manager.register_plugin(plugin)
    # ... test
```

## 📊 Benefits

### 1. **Modularity**
- Each feature is independent
- Easy to add/remove features
- Clear separation of concerns

### 2. **Testability**
- Plugins can be tested in isolation
- Easy to mock dependencies
- Configuration-driven testing

### 3. **Extensibility**
- Simple to add new plugins
- No need to modify core code
- Plugin registry pattern

### 4. **Configuration**
- Enable/disable features via config
- Environment-specific configurations
- Type-safe configuration classes

### 5. **Maintainability**
- Clear structure
- Single responsibility principle
- Easy to understand and modify

## 🔄 Migration from Old System

### Old Way

```python
from aws.api_integration import create_advanced_robot_app
app = create_advanced_robot_app(config)
```

### New Way (Same API, Modular Underneath)

```python
from aws.api_integration import create_advanced_robot_app
app = create_advanced_robot_app(config)  # Still works!
```

The API is backward compatible, but now uses the modular plugin system internally.

## 📚 Best Practices

1. **One Plugin, One Responsibility**: Each plugin should do one thing well
2. **Use Interfaces**: Always implement the appropriate interface
3. **Configuration-Driven**: Check `is_enabled()` before setup
4. **Error Handling**: Handle missing dependencies gracefully
5. **Logging**: Log plugin initialization and errors
6. **Documentation**: Document plugin behavior and configuration

## 🎯 Next Steps

1. Review existing plugins in `aws/plugins/`
2. Create custom plugins for your needs
3. Configure via `AppConfig`
4. Test plugins individually
5. Deploy with selected plugins enabled

---

**The system is now fully modular and ready for microservices architecture!** 🚀















