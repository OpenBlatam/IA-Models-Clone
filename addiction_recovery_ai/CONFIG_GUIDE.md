# Configuration Guide - Addiction Recovery AI

## ✅ Recommended Configuration

### `config/app_config.py` - **USE THIS**

The canonical configuration file for standard deployments:

```python
from config.app_config import get_config, AppConfig

# Get configuration instance
config = get_config()

# Access configuration
print(config.host)  # 0.0.0.0
print(config.port)  # 8018
print(config.app_name)  # Addiction Recovery AI
```

**Features:**
- Pydantic-based settings with validation
- Environment variable support
- Type-safe configuration
- Used by `main.py` and `core/app_factory.py`

**Usage:**
```python
from config.app_config import get_config

config = get_config()

# Use in application
app = FastAPI()
app.config = config
```

## 📋 Alternative Configuration Files

### `config/centralized_config.py` - AWS/Microservices Configuration
- **Status**: ✅ Active (Specialized)
- **Purpose**: Centralized configuration for microservices with AWS integration
- **Use Case**: 
  - AWS deployments
  - Microservices architecture
  - Hot-reload configuration needs
  - AWS Parameter Store / Secrets Manager integration

```python
from config.centralized_config import CentralizedConfig

config = CentralizedConfig()
# Supports hot-reload, AWS Parameter Store, Secrets Manager
```

**When to use:**
- AWS deployments
- Microservices architecture
- When you need hot-reload configuration
- When using AWS Parameter Store or Secrets Manager

### `config/aws_settings.py` - AWS-Specific Settings
- **Status**: ✅ Active (Specialized)
- **Purpose**: AWS-specific configuration settings
- **Use Case**: AWS deployments, AWS service configuration

```python
from config.aws_settings import get_aws_settings

aws_settings = get_aws_settings()
```

**When to use:**
- AWS deployments
- AWS service configuration
- When using AWS-specific features

### `config/config_manager.py` - Configuration Manager
- **Status**: ✅ Active (Utility)
- **Purpose**: Configuration management utilities
- **Use Case**: Advanced configuration management needs

### `config/settings.py` - Legacy Settings
- **Status**: ⚠️ Potentially Deprecated
- **Purpose**: Legacy configuration (older version)
- **Note**: Appears to be an older version, consider using `app_config.py` instead

## 🏗️ Configuration Structure

```
config/
├── app_config.py              # ✅ Canonical (standard deployments)
├── centralized_config.py      # ✅ Active (AWS/microservices)
├── aws_settings.py            # ✅ Active (AWS-specific)
├── config_manager.py          # ✅ Active (utility)
├── settings.py                # ⚠️ Potentially deprecated
├── default_config.yaml        # ✅ Default configuration
└── model_config.yaml          # ✅ Model configuration
```

## 📝 Usage Examples

### Standard Deployment
```python
from config.app_config import get_config

config = get_config()
# Use config.host, config.port, etc.
```

### AWS Deployment
```python
from config.centralized_config import CentralizedConfig
from config.aws_settings import get_aws_settings

# For microservices with AWS
config = CentralizedConfig()

# For AWS-specific settings
aws_settings = get_aws_settings()
```

### Using Configuration in App Factory
```python
from core.app_factory import create_app
from config.app_config import get_config

config = get_config()
app = create_app()  # Uses config internally
```

## 🎯 Quick Reference

| File | Purpose | Status | When to Use |
|------|---------|--------|-------------|
| `config/app_config.py` | Standard configuration | ✅ Canonical | Standard deployments, development |
| `config/centralized_config.py` | AWS/microservices config | ✅ Active | AWS deployments, microservices |
| `config/aws_settings.py` | AWS-specific settings | ✅ Active | AWS deployments |
| `config/config_manager.py` | Config utilities | ✅ Active | Advanced config management |
| `config/settings.py` | Legacy settings | ⚠️ Potentially deprecated | Consider migrating to app_config.py |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `ENTRY_POINTS_GUIDE.md` for entry points
- See `API_GUIDE.md` for API structure






