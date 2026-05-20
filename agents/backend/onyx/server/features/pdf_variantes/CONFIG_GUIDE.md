# Configuration Guide - PDF Variantes

## ✅ Recommended Configuration

### `utils/config.py` - **USE THIS**

The canonical configuration file for PDF Variantes:

```python
from utils.config import Settings, get_settings

# Get settings instance
settings = get_settings()

# Access configuration
print(settings.APP_NAME)
print(settings.DATABASE_URL)
print(settings.ENVIRONMENT)
```

**Features:**
- Pydantic-based settings with validation
- Environment variable support
- Environment-specific configurations (Development, Production, Testing)
- Feature flags
- Comprehensive configuration options

**Environment-Specific Settings:**
```python
from utils.config import get_settings_by_env

# Get settings for specific environment
settings = get_settings_by_env("production")
settings = get_settings_by_env("development")
settings = get_settings_by_env("testing")
```

## 📦 Configuration Files

### `api/config.py` - FastAPI App Configuration
**Status**: ✅ Active (Different Purpose)

This file is for FastAPI-specific configuration:
- Middleware setup
- OpenAPI schema generation
- App metadata

**Usage:**
```python
from api.config import create_app_config, setup_middleware

app_config = create_app_config()
setup_middleware(app)
```

**Note**: This is NOT deprecated - it serves a different purpose than `utils/config.py`

## ⚠️ Deprecated Configuration Files

The following files are **deprecated** and should not be used for new code:

### `config.py` (Root)
- **Status**: Deprecated
- **Reason**: Duplicate of `utils/config.py`
- **Migration**: Use `utils.config.Settings`

### `enhanced_config.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `utils/config.py` instead
- **Migration**: Use `utils.config.Settings`

### `real_config.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `utils/config.py` instead
- **Migration**: Use `utils.config.Settings`

### `ultra_config.py`
- **Status**: Deprecated
- **Reason**: Duplicate functionality, use `utils/config.py` instead
- **Migration**: Use `utils.config.Settings`

## 🏗️ Configuration Structure

```
pdf_variantes/
├── utils/
│   └── config.py          # ✅ Canonical configuration (Settings, get_settings)
├── api/
│   └── config.py          # ✅ FastAPI app configuration (middleware, OpenAPI)
├── config.py              # ⚠️ Deprecated
├── enhanced_config.py     # ⚠️ Deprecated
├── real_config.py         # ⚠️ Deprecated
└── ultra_config.py        # ⚠️ Deprecated
```

## 📝 Configuration Options

### Application Settings
- `APP_NAME`: Application name
- `APP_VERSION`: Application version
- `DEBUG`: Debug mode
- `ENVIRONMENT`: Environment (development, production, testing)

### API Settings
- `API_V1_STR`: API version prefix
- `SECRET_KEY`: Secret key for JWT
- `ACCESS_TOKEN_EXPIRE_MINUTES`: Access token expiration
- `REFRESH_TOKEN_EXPIRE_DAYS`: Refresh token expiration

### Database Settings
- `DATABASE_URL`: Database connection URL
- `DATABASE_POOL_SIZE`: Connection pool size
- `DATABASE_MAX_OVERFLOW`: Max overflow connections

### Redis Settings
- `REDIS_URL`: Redis connection URL
- `REDIS_PASSWORD`: Redis password
- `REDIS_DB`: Redis database number

### File Storage Settings
- `UPLOAD_PATH`: Upload directory
- `VARIANTS_PATH`: Variants directory
- `EXPORT_PATH`: Export directory
- `MAX_FILE_SIZE_MB`: Maximum file size
- `ALLOWED_FILE_TYPES`: Allowed file types

### AI Settings
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `HUGGINGFACE_API_KEY`: Hugging Face API key
- `DEFAULT_AI_MODEL`: Default AI model

### Processing Settings
- `MAX_VARIANTS_PER_REQUEST`: Max variants per request
- `MAX_TOPICS_PER_DOCUMENT`: Max topics per document
- `MAX_BRAINSTORM_IDEAS`: Max brainstorm ideas
- `DEFAULT_VARIANT_COUNT`: Default variant count

### Caching Settings
- `CACHE_TTL_SECONDS`: Cache TTL
- `CACHE_MAX_SIZE_MB`: Max cache size
- `ENABLE_CACHE`: Enable caching

### Security Settings
- `CORS_ORIGINS`: CORS allowed origins
- `ALLOWED_HOSTS`: Allowed hosts
- `RATE_LIMIT_PER_MINUTE`: Rate limit per minute
- `RATE_LIMIT_PER_HOUR`: Rate limit per hour

### Monitoring Settings
- `ENABLE_METRICS`: Enable metrics
- `METRICS_PORT`: Metrics port
- `HEALTH_CHECK_INTERVAL`: Health check interval

### Logging Settings
- `LOG_LEVEL`: Log level
- `LOG_FORMAT`: Log format
- `LOG_FILE`: Log file path

## 🔄 Migration Guide

### From `config.py`
```python
# Old
from config import Settings
settings = Settings()

# New
from utils.config import Settings, get_settings
settings = get_settings()
```

### From `enhanced_config.py`
```python
# Old
from enhanced_config import ConfigManager, PDFVariantesConfig
config = ConfigManager().get_config()

# New
from utils.config import Settings, get_settings
settings = get_settings()
```

### From `real_config.py`
```python
# Old
from real_config import RealSettings, get_real_settings
settings = get_real_settings()

# New
from utils.config import Settings, get_settings
settings = get_settings()
```

### From `ultra_config.py`
```python
# Old
from ultra_config import UltraFastConfigManager
config = UltraFastConfigManager().get_config()

# New
from utils.config import Settings, get_settings
settings = get_settings()
```

## 🌍 Environment Variables

All settings can be overridden using environment variables:

```bash
export ENVIRONMENT=production
export DEBUG=false
export DATABASE_URL=postgresql://user:pass@localhost/db
export SECRET_KEY=your-secret-key
export LOG_LEVEL=WARNING
```

Or use a `.env` file:
```env
ENVIRONMENT=production
DEBUG=false
DATABASE_URL=postgresql://user:pass@localhost/db
SECRET_KEY=your-secret-key
LOG_LEVEL=WARNING
```

## 📚 Additional Resources

- See `utils/config.py` for full configuration options
- See `api/config.py` for FastAPI-specific configuration
- See `REFACTORING_STATUS.md` for refactoring progress






