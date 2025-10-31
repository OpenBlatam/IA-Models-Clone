# Dependencies Management Guide

## Overview

This guide provides comprehensive information about managing dependencies for the HeyGen AI FastAPI backend project. The project uses modern Python packaging standards with multiple dependency files for different use cases.

## Dependency Files Structure

### 📁 Core Dependency Files

| File | Purpose | Use Case |
|------|---------|----------|
| `requirements.txt` | Production dependencies | Full production deployment |
| `requirements-minimal.txt` | Essential dependencies only | Basic functionality |
| `requirements-dev.txt` | Development dependencies | Local development |
| `requirements-test.txt` | Testing dependencies | Running tests |
| `pyproject.toml` | Modern Python packaging | Package distribution |

### 📦 Optional Dependencies (pyproject.toml)

| Group | Purpose | Dependencies |
|-------|---------|--------------|
| `dev` | Development tools | pytest, black, flake8, mypy, etc. |
| `test` | Testing framework | pytest, factory-boy, testcontainers, etc. |
| `docs` | Documentation | mkdocs, mkdocs-material, etc. |
| `monitoring` | Observability | prometheus, sentry, jaeger, etc. |
| `ml` | Machine Learning | torch, transformers, langchain, etc. |
| `video` | Video processing | moviepy, ffmpeg, opencv, etc. |
| `audio` | Audio processing | librosa, soundfile, pydub, etc. |
| `all` | Everything | All optional dependencies |

## Installation Methods

### 🚀 Quick Start (Production)

```bash
# Install with pip
pip install -r requirements.txt

# Or install as package with all dependencies
pip install -e ".[all]"
```

### 🔧 Development Setup

```bash
# Install with development dependencies
pip install -r requirements-dev.txt

# Or install as package with dev dependencies
pip install -e ".[dev]"
```

### 🧪 Testing Setup

```bash
# Install with testing dependencies
pip install -r requirements-test.txt

# Or install as package with test dependencies
pip install -e ".[test]"
```

### 📚 Documentation Setup

```bash
# Install with documentation dependencies
pip install -e ".[docs]"
```

### 📊 Monitoring Setup

```bash
# Install with monitoring dependencies
pip install -e ".[monitoring]"
```

## Dependency Categories

### 🏗️ Core Framework
```python
# FastAPI and ASGI server
fastapi>=0.104.0,<0.105.0
uvicorn[standard]>=0.24.0,<0.25.0
gunicorn>=21.2.0,<22.0.0
pydantic>=2.5.0,<3.0.0
pydantic-settings>=2.1.0,<3.0.0
```

**Purpose**: Web framework, API server, data validation

### 🗄️ Database and ORM
```python
# Database
sqlalchemy>=2.0.23,<3.0.0
alembic>=1.13.0,<2.0.0
aiosqlite>=0.19.0,<0.20.0
asyncpg>=0.29.0,<0.30.0
redis>=5.0.1,<6.0.0
```

**Purpose**: Database operations, migrations, caching

### 🔐 Authentication and Security
```python
# Authentication
python-jose[cryptography]>=3.3.0,<4.0.0
passlib[bcrypt]>=1.7.4,<2.0.0
python-multipart>=0.0.6,<0.1.0
bcrypt>=4.1.0,<5.0.0
cryptography>=41.0.7,<42.0.0
```

**Purpose**: JWT tokens, password hashing, file uploads

### 🌐 HTTP and Networking
```python
# HTTP clients
httpx>=0.25.0,<0.26.0
requests>=2.31.0,<3.0.0
aiohttp>=3.9.0,<4.0.0
```

**Purpose**: HTTP requests, API calls, async networking

### ⚡ Background Tasks
```python
# Background processing
celery>=5.3.4,<6.0.0
asyncio-mqtt>=0.16.0,<0.17.0
asyncio-throttle>=1.0.2,<2.0.0
```

**Purpose**: Task queues, message processing, rate limiting

### 📊 Monitoring and Logging
```python
# Observability
prometheus-client>=0.19.0,<0.20.0
structlog>=23.2.0,<24.0.0
sentry-sdk[fastapi]>=1.38.0,<2.0.0
```

**Purpose**: Metrics, structured logging, error tracking

### 🚦 Rate Limiting and Caching
```python
# Rate limiting and caching
slowapi>=0.1.9,<0.2.0
cachetools>=5.3.0,<6.0.0
aioredis>=2.0.1,<3.0.0
```

**Purpose**: API rate limiting, caching, Redis operations

### 📁 File Processing
```python
# File handling
aiofiles>=23.2.1,<24.0.0
python-magic>=0.4.27,<0.5.0
Pillow>=10.1.0,<11.0.0
opencv-python>=4.8.1.78,<5.0.0
```

**Purpose**: File uploads, image processing, async file operations

### 🎬 Video and Audio Processing
```python
# Video processing
moviepy>=1.0.3,<2.0.0
ffmpeg-python>=0.2.0,<0.3.0

# Audio processing
librosa>=0.10.1,<0.11.0
soundfile>=0.12.1,<0.13.0
pydub>=0.25.1,<0.26.0
```

**Purpose**: Video editing, audio processing, media manipulation

### 🤖 AI/ML Core Libraries
```python
# Core ML
torch>=2.1.0,<3.0.0
transformers>=4.35.0,<5.0.0
diffusers>=0.24.0,<0.25.0
accelerate>=0.25.0,<0.26.0
safetensors>=0.4.0,<0.5.0
xformers>=0.0.22,<0.1.0
```

**Purpose**: Deep learning, model inference, optimization

### 🔍 Vector Stores and Embeddings
```python
# Vector databases
faiss-cpu>=1.7.4,<2.0.0
chromadb>=0.4.18,<0.5.0
sentence-transformers>=2.2.2,<3.0.0
```

**Purpose**: Vector search, embeddings, similarity matching

### 🔗 LangChain and AI Integration
```python
# AI frameworks
langchain>=0.1.0,<0.2.0
langchain-openai>=0.1.0,<0.2.0
openai>=1.3.0,<2.0.0
```

**Purpose**: LLM integration, AI workflows, prompt management

### 📊 Data Processing
```python
# Data analysis
numpy>=1.24.0,<2.0.0
pandas>=2.1.0,<3.0.0
scipy>=1.11.0,<2.0.0
scikit-learn>=1.3.0,<2.0.0
```

**Purpose**: Data manipulation, analysis, machine learning

### ⚙️ Configuration and Utilities
```python
# Configuration
python-dotenv>=1.0.0,<2.0.0
click>=8.1.0,<9.0.0
pyyaml>=6.0.1,<7.0.0
toml>=0.10.2,<0.11.0

# Utilities
psutil>=5.9.6,<6.0.0
packaging>=23.2,<24.0
tqdm>=4.66.0,<5.0.0
python-dateutil>=2.8.2,<3.0.0
```

**Purpose**: Environment management, CLI tools, system utilities

## Version Management Strategy

### 📋 Version Constraints

```python
# Example version constraint
"fastapi>=0.104.0,<0.105.0"
```

**Format**: `package>=minimum_version,<maximum_version`

**Benefits**:
- Ensures minimum required functionality
- Prevents breaking changes from major updates
- Allows security patches and bug fixes

### 🔄 Dependency Updates

#### Automatic Updates
```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Update specific package
pip install --upgrade fastapi

# Check for outdated packages
pip list --outdated
```

#### Manual Updates
```bash
# Generate requirements with current versions
pip freeze > requirements-current.txt

# Compare with existing requirements
diff requirements.txt requirements-current.txt
```

### 🔒 Security Updates

```bash
# Check for security vulnerabilities
safety check -r requirements.txt

# Update vulnerable packages
safety check -r requirements.txt --json | jq '.[] | .package'
```

## Development Workflow

### 🛠️ Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 🧪 Testing

```bash
# Install test dependencies
pip install -e ".[test]"

# Run tests
pytest

# Run with coverage
pytest --cov=heygen_ai

# Run specific test categories
pytest -m "unit"
pytest -m "integration"
pytest -m "e2e"
```

### 📚 Documentation

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

### 📊 Monitoring

```bash
# Install monitoring dependencies
pip install -e ".[monitoring]"

# Run with monitoring
python -m heygen_ai.main --enable-monitoring
```

## Production Deployment

### 🚀 Docker Deployment

```dockerfile
# Use multi-stage build for production
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run application
CMD ["uvicorn", "heygen_ai.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### ☁️ Cloud Deployment

#### AWS Lambda
```yaml
# serverless.yml
functions:
  api:
    handler: heygen_ai.main.handler
    runtime: python3.11
    layers:
      - arn:aws:lambda:us-east-1:770693421928:layer:Klayers-p311-requests:1
```

#### Google Cloud Run
```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/heygen-ai', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/heygen-ai']
```

## Dependency Conflicts and Resolution

### 🔍 Conflict Detection

```bash
# Check for dependency conflicts
pip check

# Show dependency tree
pip install pipdeptree
pipdeptree

# Show reverse dependencies
pipdeptree --reverse
```

### 🔧 Conflict Resolution

#### Common Conflicts

1. **Version Conflicts**
```bash
# Example: Multiple packages requiring different versions
Package A requires torch>=1.0.0
Package B requires torch>=2.0.0

# Resolution: Use compatible versions
torch>=2.0.0  # Satisfies both requirements
```

2. **Platform Conflicts**
```bash
# Example: Platform-specific packages
# requirements.txt
torch>=2.1.0  # CPU version
# requirements-gpu.txt
torch>=2.1.0+cu118  # GPU version
```

3. **Python Version Conflicts**
```python
# pyproject.toml
requires-python = ">=3.9"  # Minimum Python version
```

### 🛠️ Resolution Strategies

#### 1. Pin Specific Versions
```python
# requirements.txt
fastapi==0.104.1  # Exact version
pydantic>=2.5.0,<3.0.0  # Version range
```

#### 2. Use Compatible Releases
```python
# pyproject.toml
dependencies = [
    "fastapi~=0.104.0",  # Compatible release
]
```

#### 3. Conditional Dependencies
```python
# setup.py
extras_require = {
    'gpu': ['torch>=2.1.0+cu118'],
    'cpu': ['torch>=2.1.0'],
}
```

## Best Practices

### 📋 Dependency Management

1. **Use Version Constraints**
   - Always specify version ranges
   - Use `>=` for minimum versions
   - Use `<` for maximum versions

2. **Separate Concerns**
   - Production vs development dependencies
   - Core vs optional dependencies
   - Platform-specific dependencies

3. **Regular Updates**
   - Update dependencies regularly
   - Test updates thoroughly
   - Monitor for security vulnerabilities

4. **Documentation**
   - Document dependency purposes
   - Maintain changelog for major updates
   - Provide migration guides

### 🔒 Security Considerations

1. **Vulnerability Scanning**
```bash
# Regular security checks
safety check -r requirements.txt
pip-audit
```

2. **Dependency Verification**
```bash
# Verify package integrity
pip install --require-hashes -r requirements.txt
```

3. **Minimal Dependencies**
   - Only include necessary dependencies
   - Remove unused dependencies
   - Use minimal base images

### 🚀 Performance Optimization

1. **Dependency Optimization**
```bash
# Use wheels when possible
pip install --only-binary=all -r requirements.txt

# Cache dependencies
pip install --cache-dir .pip-cache -r requirements.txt
```

2. **Layer Optimization**
```dockerfile
# Optimize Docker layers
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

## Troubleshooting

### 🔍 Common Issues

#### 1. Import Errors
```bash
# Check installed packages
pip list | grep package_name

# Check import paths
python -c "import package_name; print(package_name.__file__)"
```

#### 2. Version Conflicts
```bash
# Show conflicting packages
pip check

# Show dependency tree
pipdeptree --warn silence
```

#### 3. Platform Issues
```bash
# Check platform compatibility
python -c "import platform; print(platform.platform())"

# Install platform-specific packages
pip install --platform manylinux2014_x86_64 package_name
```

### 🛠️ Debugging Commands

```bash
# Show package information
pip show package_name

# Show package files
pip show -f package_name

# Check package compatibility
pip check

# Validate requirements file
pip install --dry-run -r requirements.txt
```

## Conclusion

This dependency management system provides:

1. **Flexibility**: Multiple dependency files for different use cases
2. **Security**: Version constraints and vulnerability scanning
3. **Maintainability**: Clear organization and documentation
4. **Scalability**: Optional dependencies for different deployment scenarios
5. **Compatibility**: Cross-platform support and version management

Follow these guidelines to maintain a robust and secure dependency management system for the HeyGen AI project. 