"""
CI/CD Generator - Generador de Pipelines CI/CD
===============================================

Genera pipelines de CI/CD automáticamente para los proyectos.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _generate_backend_ci() -> str:
    """
    Genera workflow de CI para backend (función pura).
    
    Returns:
        Contenido del archivo backend-ci.yml
    """
    return '''name: Backend CI

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'backend/**'
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'backend/**'

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
      working-directory: ./backend
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      working-directory: ./backend
      run: |
        pytest
    
    - name: Lint
      working-directory: ./backend
      run: |
        pip install flake8
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
'''


def _generate_frontend_ci() -> str:
    """
    Genera workflow de CI para frontend (función pura).
    
    Returns:
        Contenido del archivo frontend-ci.yml
    """
    return '''name: Frontend CI

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'frontend/**'
  pull_request:
    branches: [ main, develop ]
    paths:
      - 'frontend/**'

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
    
    - name: Install dependencies
      working-directory: ./frontend
      run: npm ci
    
    - name: Run tests
      working-directory: ./frontend
      run: npm test -- --coverage
    
    - name: Build
      working-directory: ./frontend
      run: npm run build
'''


def _generate_docker_build() -> str:
    """
    Genera workflow de Docker build (función pura).
    
    Returns:
        Contenido del archivo docker-build.yml
    """
    return '''name: Docker Build

on:
  push:
    branches: [ main ]
  tags:
    - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Build and push backend
      uses: docker/build-push-action@v4
      with:
        context: ./backend
        push: ${{ github.event_name != 'pull_request' }}
        tags: |
          ${{ secrets.DOCKER_REGISTRY }}/${{ github.repository }}-backend:latest
          ${{ secrets.DOCKER_REGISTRY }}/${{ github.repository }}-backend:${{ github.sha }}
    
    - name: Build and push frontend
      uses: docker/build-push-action@v4
      with:
        context: ./frontend
        push: ${{ github.event_name != 'pull_request' }}
        tags: |
          ${{ secrets.DOCKER_REGISTRY }}/${{ github.repository }}-frontend:latest
          ${{ secrets.DOCKER_REGISTRY }}/${{ github.repository }}-frontend:${{ github.sha }}
'''


def _generate_gitlab_ci() -> str:
    """
    Genera configuración de GitLab CI (función pura).
    
    Returns:
        Contenido del archivo .gitlab-ci.yml
    """
    return '''stages:
  - test
  - build
  - deploy

backend:test:
  stage: test
  image: python:3.11
  script:
    - cd backend
    - pip install -r requirements.txt
    - pytest
  only:
    - main
    - develop
    - merge_requests

frontend:test:
  stage: test
  image: node:18
  script:
    - cd frontend
    - npm ci
    - npm test
    - npm run build
  only:
    - main
    - develop
    - merge_requests

docker:build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA ./backend
    - docker build -t $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA ./frontend
    - docker push $CI_REGISTRY_IMAGE/backend:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE/frontend:$CI_COMMIT_SHA
  only:
    - main
'''


def _write_file_safe(file_path: Path, content: str, encoding: str = "utf-8") -> None:
    """
    Escribir archivo de forma segura (función pura).
    
    Args:
        file_path: Ruta del archivo
        content: Contenido a escribir
        encoding: Codificación del archivo
        
    Raises:
        IOError: Si no se puede escribir el archivo
    """
    try:
        file_path.write_text(content, encoding=encoding)
    except IOError as e:
        logger.error(f"Failed to write file {file_path}: {e}")
        raise


class CICDGenerator:
    """
    Generador de pipelines CI/CD.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializa el generador de CI/CD."""
        pass
    
    async def generate_github_actions(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera GitHub Actions workflows.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se pueden escribir los archivos
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not keywords:
            raise ValueError("keywords cannot be empty")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        workflows_dir = project_dir / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            _write_file_safe(workflows_dir / "backend-ci.yml", _generate_backend_ci())
            _write_file_safe(workflows_dir / "frontend-ci.yml", _generate_frontend_ci())
            _write_file_safe(workflows_dir / "docker-build.yml", _generate_docker_build())
            
            logger.info("GitHub Actions workflows generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate GitHub Actions workflows: {e}")
            raise
    
    async def generate_gitlab_ci(
        self,
        project_dir: Path,
        keywords: Dict[str, Any],
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera GitLab CI configuration.
        
        Args:
            project_dir: Directorio del proyecto
            keywords: Keywords extraídos
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede escribir el archivo
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not keywords:
            raise ValueError("keywords cannot be empty")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        try:
            _write_file_safe(project_dir / ".gitlab-ci.yml", _generate_gitlab_ci())
            logger.info("GitLab CI generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate GitLab CI: {e}")
            raise
