# CI/CD Integration Guide

## 🚀 Integración con CI/CD

Esta guía explica cómo integrar la suite de tests con sistemas de CI/CD.

## GitHub Actions

### Workflow Básico

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov pytest-xdist
    
    - name: Run tests
      run: |
        pytest tests/ -v --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

### Workflow Avanzado

```yaml
name: Comprehensive Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -m unit -v
  
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ -m integration -v
  
  coverage:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=. --cov-report=html
      - uses: actions/upload-artifact@v3
        with:
          name: coverage-report
          path: htmlcov/
```

## GitLab CI

```yaml
stages:
  - test
  - coverage

unit-tests:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest tests/ -m unit -v

integration-tests:
  stage: test
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest tests/ -m integration -v

coverage:
  stage: coverage
  image: python:3.11
  script:
    - pip install -r requirements.txt
    - pytest tests/ --cov=. --cov-report=html
  artifacts:
    paths:
      - htmlcov/
    expire_in: 1 week
```

## Jenkins

```groovy
pipeline {
    agent any
    
    stages {
        stage('Test') {
            steps {
                sh 'pip install -r requirements.txt'
                sh 'pytest tests/ -v --junitxml=test-results.xml'
            }
            post {
                always {
                    junit 'test-results.xml'
                }
            }
        }
        
        stage('Coverage') {
            steps {
                sh 'pytest tests/ --cov=. --cov-report=html'
            }
            post {
                always {
                    publishHTML([
                        reportDir: 'htmlcov',
                        reportFiles: 'index.html',
                        reportName: 'Coverage Report'
                    ])
                }
            }
        }
    }
}
```

## Variables de Entorno para CI

```bash
# Configuración para CI
export TEST_MODE=ci
export PYTEST_ADDOPTS="-v --tb=short"
export COVERAGE_THRESHOLD=95
```

## Ejecución en CI

### Tests Rápidos (Recomendado para PRs)
```bash
pytest tests/ -m "not slow and not integration" -v
```

### Todos los Tests (Recomendado para main)
```bash
pytest tests/ -v --cov=. --cov-report=xml
```

### Tests en Paralelo
```bash
pytest tests/ -n auto  # Requiere pytest-xdist
```

## Reportes

### JUnit XML
```bash
pytest tests/ --junitxml=test-results.xml
```

### Coverage XML
```bash
pytest tests/ --cov=. --cov-report=xml
```

### HTML Reports
```bash
pytest tests/ --cov=. --cov-report=html
# Reporte disponible en htmlcov/index.html
```

## Mejores Prácticas para CI

1. **Ejecutar tests rápidos en PRs**: Solo tests unitarios y rápidos
2. **Ejecutar todos los tests en main**: Incluir tests lentos e integración
3. **Generar reportes**: Coverage y JUnit XML
4. **Fallo rápido**: Detener en primer error si es crítico
5. **Caché de dependencias**: Usar caché para pip
6. **Paralelización**: Usar pytest-xdist cuando sea posible

## Ejemplo de Script CI

```bash
#!/bin/bash
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Running unit tests..."
pytest tests/ -m unit -v

echo "Running integration tests..."
pytest tests/ -m integration -v

echo "Generating coverage report..."
pytest tests/ --cov=. --cov-report=html --cov-report=xml

echo "Tests completed successfully!"
```

