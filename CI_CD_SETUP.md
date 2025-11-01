# 🔄 Guía de CI/CD - Blatam Academy Features

## 🚀 GitHub Actions Setup

### Workflow Básico

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=bulk --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Workflow de Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [ main ]
    tags:
      - 'v*'

jobs:
  deploy:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Registry
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          your-registry/blatam-academy:latest
          your-registry/blatam-academy:${{ github.sha }}
    
    - name: Deploy to production
      run: |
        ssh user@production-server 'cd /app && docker-compose pull && docker-compose up -d'
```

## 🧪 Pre-commit Hooks

### .pre-commit-config.yaml

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 22.3.0
    hooks:
      - id: black
        language_version: python3.9
  
  - repo: https://github.com/pycqa/flake8
    rev: 4.0.1
    hooks:
      - id: flake8
        args: ['--max-line-length=120']
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.950
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: local
    hooks:
      - id: tests
        name: tests
        entry: pytest
        language: system
        pass_filenames: false
        always_run: true
        args: ['tests/', '-v']
```

## 📦 Dockerfile Optimizado para CI/CD

```dockerfile
# Dockerfile.ci
FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Variables de entorno para CI
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Comando por defecto
CMD ["pytest", "tests/"]
```

## 🔄 Pipeline Completo

### Stage 1: Lint y Tests

```yaml
lint_and_test:
  stage: test
  script:
    - black --check .
    - flake8 .
    - pytest tests/unit/ -v
    - pytest tests/integration/ -v
  only:
    - merge_requests
    - main
```

### Stage 2: Build

```yaml
build:
  stage: build
  script:
    - docker build -t blatam-academy:$CI_COMMIT_SHA .
    - docker tag blatam-academy:$CI_COMMIT_SHA blatam-academy:latest
  only:
    - main
```

### Stage 3: Deploy Staging

```yaml
deploy_staging:
  stage: deploy
  script:
    - ./deploy.sh staging
  environment:
    name: staging
  only:
    - develop
```

### Stage 4: Deploy Production

```yaml
deploy_production:
  stage: deploy
  script:
    - ./deploy.sh production
  environment:
    name: production
  when: manual
  only:
    - main
```

## 📊 CI/CD Checklist

### Setup Inicial
- [ ] GitHub Actions / GitLab CI configurado
- [ ] Tests automatizados
- [ ] Linting configurado
- [ ] Coverage reporting
- [ ] Docker build automatizado

### Pre-Merge
- [ ] Tests pasando
- [ ] Linting OK
- [ ] Coverage > 80%
- [ ] No vulnerabilidades

### Pre-Deploy
- [ ] Tests de integración pasando
- [ ] Build exitoso
- [ ] Imágenes Docker creadas
- [ ] Health checks pasando

---

**Más información:**
- [Contributing](CONTRIBUTING.md)
- [Testing Guide](bulk/core/TESTING_GUIDE.md)
- [Production Ready](bulk/PRODUCTION_READY.md)

