"""
CI/CD Generator
===============

Generador de configuraciones CI/CD para proyectos de Deep Learning.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CICDConfig:
    """Configuración de CI/CD."""
    platform: str  # 'github', 'gitlab', 'jenkins'
    test_command: str = "pytest"
    lint_command: str = "flake8"
    build_command: str = "docker build"
    deploy_command: str = "docker push"
    python_versions: List[str] = None
    use_gpu: bool = False
    
    def __post_init__(self):
        """Inicializar valores por defecto."""
        if self.python_versions is None:
            self.python_versions = ['3.9', '3.10', '3.11']


class CICDGenerator:
    """
    Generador de configuraciones CI/CD.
    """
    
    def __init__(self):
        """Inicializar generador."""
        pass
    
    def generate_github_actions(
        self,
        project_dir: Path,
        config: Optional[CICDConfig] = None
    ) -> str:
        """
        Generar workflow de GitHub Actions.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido del workflow
        """
        if config is None:
            config = CICDConfig(platform='github')
        
        workflow_content = f"""name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: {config.python_versions}
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{{{ matrix.python-version }}}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}
    
    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{{{ runner.os }}}}-pip-${{{{ hashFiles('**/requirements.txt') }}}}
        restore-keys: |
          ${{{{ runner.os }}}}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8
    
    - name: Lint with flake8
      run: |
        {config.lint_command} app/ --count --select=E9,F63,F7,F82 --show-source --statistics
        {config.lint_command} app/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
    
    - name: Test with pytest
      run: |
        {config.test_command} --cov=app --cov-report=xml --cov-report=html
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{{{ secrets.DOCKER_USERNAME }}}}
        password: ${{{{ secrets.DOCKER_PASSWORD }}}}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{{{ secrets.DOCKER_USERNAME }}}}/${{{{ github.event.repository.name }}}}:latest
"""
        
        return workflow_content
    
    def generate_gitlab_ci(
        self,
        project_dir: Path,
        config: Optional[CICDConfig] = None
    ) -> str:
        """
        Generar .gitlab-ci.yml.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido de .gitlab-ci.yml
        """
        if config is None:
            config = CICDConfig(platform='gitlab')
        
        gitlab_ci_content = f"""stages:
  - test
  - build
  - deploy

variables:
  DOCKER_DRIVER: overlay2
  DOCKER_TLS_CERTDIR: "/certs"

test:
  stage: test
  image: python:{config.python_versions[0] if config.python_versions else '3.10'}
  script:
    - pip install -r requirements.txt
    - pip install pytest pytest-cov flake8
    - {config.lint_command} app/
    - {config.test_command} --cov=app --cov-report=xml
  coverage: '/TOTAL.*\\s+(\\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

build:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only:
    - main

deploy:
  stage: deploy
  image: docker:latest
  services:
    - docker:dind
  script:
    - docker pull $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker tag $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA $CI_REGISTRY_IMAGE:latest
    - docker push $CI_REGISTRY_IMAGE:latest
  only:
    - main
"""
        
        return gitlab_ci_content
    
    def generate_jenkinsfile(
        self,
        project_dir: Path,
        config: Optional[CICDConfig] = None
    ) -> str:
        """
        Generar Jenkinsfile.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido de Jenkinsfile
        """
        if config is None:
            config = CICDConfig(platform='jenkins')
        
        jenkinsfile_content = f"""pipeline {{
    agent any
    
    stages {{
        stage('Checkout') {{
            steps {{
                checkout scm
            }}
        }}
        
        stage('Test') {{
            steps {{
                sh '''
                    python -m venv venv
                    . venv/bin/activate
                    pip install -r requirements.txt
                    pip install pytest pytest-cov flake8
                    {config.lint_command} app/
                    {config.test_command} --cov=app --cov-report=xml
                '''
            }}
        }}
        
        stage('Build') {{
            steps {{
                sh '{config.build_command} -t ${{env.JOB_NAME}}:${{env.BUILD_NUMBER}} .'
            }}
        }}
        
        stage('Deploy') {{
            when {{
                branch 'main'
            }}
            steps {{
                sh '{config.deploy_command} ${{env.JOB_NAME}}:${{env.BUILD_NUMBER}}'
            }}
        }}
    }}
    
    post {{
        always {{
            publishCoverage adapters: [coberturaAdapter('coverage.xml')]
        }}
    }}
}}
"""
        
        return jenkinsfile_content
    
    def generate_all(
        self,
        project_dir: Path,
        platform: str = "github",
        config: Optional[CICDConfig] = None
    ) -> Dict[str, str]:
        """
        Generar todas las configuraciones CI/CD.
        
        Args:
            project_dir: Directorio del proyecto
            platform: Plataforma CI/CD
            config: Configuración (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        if config is None:
            config = CICDConfig(platform=platform)
        
        files = {}
        
        if platform == "github":
            workflow_content = self.generate_github_actions(project_dir, config)
            workflow_path = project_dir / ".github" / "workflows" / "ci.yml"
            workflow_path.parent.mkdir(parents=True, exist_ok=True)
            workflow_path.write_text(workflow_content, encoding='utf-8')
            files['.github/workflows/ci.yml'] = workflow_content
        
        elif platform == "gitlab":
            gitlab_ci_content = self.generate_gitlab_ci(project_dir, config)
            gitlab_ci_path = project_dir / ".gitlab-ci.yml"
            gitlab_ci_path.write_text(gitlab_ci_content, encoding='utf-8')
            files['.gitlab-ci.yml'] = gitlab_ci_content
        
        elif platform == "jenkins":
            jenkinsfile_content = self.generate_jenkinsfile(project_dir, config)
            jenkinsfile_path = project_dir / "Jenkinsfile"
            jenkinsfile_path.write_text(jenkinsfile_content, encoding='utf-8')
            files['Jenkinsfile'] = jenkinsfile_content
        
        logger.info(f"Configuraciones CI/CD generadas para {platform}")
        
        return files


# Instancia global
_global_cicd_generator: Optional[CICDGenerator] = None


def get_cicd_generator() -> CICDGenerator:
    """
    Obtener instancia global del generador de CI/CD.
    
    Returns:
        Instancia del generador
    """
    global _global_cicd_generator
    
    if _global_cicd_generator is None:
        _global_cicd_generator = CICDGenerator()
    
    return _global_cicd_generator

