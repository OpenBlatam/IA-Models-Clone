# Health Checks Guide - Addiction Recovery AI

## ✅ Recommended Health Checks

### `api/health.py` - **USE THIS FOR STANDARD HEALTH CHECKS**

The canonical health check endpoint for standard deployments:

```python
from api.health import router

# Include in your app
app.include_router(router)
```

**Features:**
- Basic health check endpoint (`/health`)
- System information
- Database connectivity check
- Redis connectivity check (if configured)
- Uptime tracking
- Simple and lightweight

**Endpoints:**
- `GET /health` - Basic health check
- `GET /health/detailed` - Detailed health information

**Usage:**
```python
from fastapi import FastAPI
from api.health import router

app = FastAPI()
app.include_router(router)

# Health check available at /health
```

## 📋 Alternative Health Checks

### `api/health_advanced.py` - AWS-Specific Health Checks
- **Status**: ✅ Active (Specialized)
- **Purpose**: Advanced health checks for AWS deployments
- **Use Case**: When deploying on AWS with AWS services
- **Features**:
  - AWS service health checks (DynamoDB, S3, CloudWatch, SNS, SQS)
  - Readiness and liveness probes
  - Detailed service status
  - AWS-specific monitoring

```python
from api.health_advanced import router

# Include in your app (for AWS deployments)
app.include_router(router)
```

**When to use:**
- AWS deployments
- When using AWS services (DynamoDB, S3, etc.)
- Production environments with AWS infrastructure
- Kubernetes deployments with AWS services

**Endpoints:**
- `GET /health` - Basic health check
- `GET /health/ready` - Readiness probe
- `GET /health/live` - Liveness probe
- `GET /health/aws` - AWS services health check

## 🏗️ Health Checks Structure

```
api/
├── health.py              # ✅ Canonical (standard health checks)
└── health_advanced.py      # ✅ Active (AWS-specific health checks)
```

## 📝 Usage Examples

### Standard Deployment
```python
from fastapi import FastAPI
from api.health import router

app = FastAPI()
app.include_router(router)

# Health check at /health
```

### AWS Deployment
```python
from fastapi import FastAPI
from api.health_advanced import router

app = FastAPI()
app.include_router(router)

# Advanced health checks with AWS service monitoring
```

### Using Both (if needed)
```python
from fastapi import FastAPI
from api.health import router as standard_health
from api.health_advanced import router as aws_health

app = FastAPI()

# Standard health at /health
app.include_router(standard_health)

# AWS health at /health/aws
app.include_router(aws_health, prefix="/aws")
```

## 🎯 Quick Reference

| File | Purpose | Status | When to Use |
|------|---------|--------|-------------|
| `api/health.py` | Standard health checks | ✅ Canonical | Standard deployments, development |
| `api/health_advanced.py` | AWS health checks | ✅ Active | AWS deployments, production with AWS |

## 📚 Additional Resources

- See `REFACTORING_STATUS.md` for refactoring progress
- See `API_GUIDE.md` for API structure
- See `ENTRY_POINTS_GUIDE.md` for entry points






