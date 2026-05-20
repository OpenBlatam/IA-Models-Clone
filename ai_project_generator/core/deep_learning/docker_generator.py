"""
Docker Generator
================

Generador de Dockerfiles optimizados para proyectos de Deep Learning.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DockerConfig:
    """Configuración de Docker."""
    base_image: str
    python_version: str
    framework: str
    cuda_version: Optional[str] = None
    requirements_file: str = "requirements.txt"
    working_dir: str = "/app"
    expose_port: Optional[int] = None


class DockerGenerator:
    """
    Generador de Dockerfiles optimizados.
    """
    
    def __init__(self):
        """Inicializar generador."""
        self.base_images = {
            'pytorch': {
                'cpu': 'pytorch/pytorch:latest',
                'cuda': 'pytorch/pytorch:latest-cuda11.8.0-cudnn8-runtime'
            },
            'tensorflow': {
                'cpu': 'tensorflow/tensorflow:latest',
                'cuda': 'tensorflow/tensorflow:latest-gpu'
            },
            'jax': {
                'cpu': 'python:3.10-slim',
                'cuda': 'nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04'
            }
        }
    
    def generate_dockerfile(
        self,
        project_dir: Path,
        config: Optional[DockerConfig] = None,
        framework: str = "pytorch",
        use_gpu: bool = False
    ) -> str:
        """
        Generar Dockerfile optimizado.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración personalizada (opcional)
            framework: Framework a usar
            use_gpu: Si usar GPU/CUDA
            
        Returns:
            Contenido del Dockerfile
        """
        if config is None:
            config = self._create_default_config(framework, use_gpu)
        
        dockerfile_content = f"""# Dockerfile optimizado para {framework.upper()}
# Generado automáticamente por DeepLearningGenerator

FROM {config.base_image}

# Variables de entorno
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    git \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR {config.working_dir}

# Copiar requirements primero (para cache de Docker)
COPY {config.requirements_file} .

# Instalar dependencias Python
RUN pip install --upgrade pip && \\
    pip install -r {config.requirements_file}

# Copiar código de la aplicación
COPY app/ ./app/
COPY *.py ./

# Exponer puerto si es necesario
"""
        
        if config.expose_port:
            dockerfile_content += f"EXPOSE {config.expose_port}\n\n"
        
        dockerfile_content += """# Comando por defecto
CMD ["python", "main.py"]
"""
        
        return dockerfile_content
    
    def generate_docker_compose(
        self,
        project_dir: Path,
        services: List[str] = None,
        use_gpu: bool = False
    ) -> str:
        """
        Generar docker-compose.yml.
        
        Args:
            project_dir: Directorio del proyecto
            services: Lista de servicios (opcional)
            use_gpu: Si usar GPU
            
        Returns:
            Contenido de docker-compose.yml
        """
        if services is None:
            services = ['app']
        
        compose_content = """version: '3.8'

services:
"""
        
        for service in services:
            compose_content += f"""  {service}:
    build:
      context: .
      dockerfile: Dockerfile
    volumes:
      - ./app:/app/app
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - PYTHONUNBUFFERED=1
"""
            if use_gpu:
                compose_content += """    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
"""
            if service == 'app':
                compose_content += """    ports:
      - "8000:8000"
"""
        
        return compose_content
    
    def generate_dockerignore(self, project_dir: Path) -> str:
        """
        Generar .dockerignore.
        
        Args:
            project_dir: Directorio del proyecto
            
        Returns:
            Contenido de .dockerignore
        """
        return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Data
data/
*.csv
*.json
*.h5
*.hdf5

# Models
models/
*.pth
*.pt
*.ckpt
*.pb

# Logs
logs/
*.log
wandb/
tensorboard/

# Cache
.cache/
*.cache

# Git
.git/
.gitignore

# Docker
Dockerfile
docker-compose.yml
.dockerignore

# Docs
docs/
*.md

# Tests
tests/
.pytest_cache/
.coverage
htmlcov/
"""
    
    def _create_default_config(
        self,
        framework: str,
        use_gpu: bool
    ) -> DockerConfig:
        """Crear configuración por defecto."""
        device = 'cuda' if use_gpu else 'cpu'
        base_image = self.base_images.get(framework, {}).get(device, 'python:3.10-slim')
        
        return DockerConfig(
            base_image=base_image,
            python_version="3.10",
            framework=framework,
            cuda_version="11.8.0" if use_gpu else None
        )
    
    def generate_all(
        self,
        project_dir: Path,
        framework: str = "pytorch",
        use_gpu: bool = False,
        expose_port: Optional[int] = 8000
    ) -> Dict[str, str]:
        """
        Generar todos los archivos Docker.
        
        Args:
            project_dir: Directorio del proyecto
            framework: Framework a usar
            use_gpu: Si usar GPU
            expose_port: Puerto a exponer (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        config = self._create_default_config(framework, use_gpu)
        config.expose_port = expose_port
        
        files = {}
        
        # Dockerfile
        dockerfile_content = self.generate_dockerfile(project_dir, config, framework, use_gpu)
        dockerfile_path = project_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content, encoding='utf-8')
        files['Dockerfile'] = dockerfile_content
        
        # docker-compose.yml
        compose_content = self.generate_docker_compose(project_dir, use_gpu=use_gpu)
        compose_path = project_dir / "docker-compose.yml"
        compose_path.write_text(compose_content, encoding='utf-8')
        files['docker-compose.yml'] = compose_content
        
        # .dockerignore
        dockerignore_content = self.generate_dockerignore(project_dir)
        dockerignore_path = project_dir / ".dockerignore"
        dockerignore_path.write_text(dockerignore_content, encoding='utf-8')
        files['.dockerignore'] = dockerignore_content
        
        logger.info(f"Archivos Docker generados en {project_dir}")
        
        return files


# Instancia global
_global_docker_generator: Optional[DockerGenerator] = None


def get_docker_generator() -> DockerGenerator:
    """
    Obtener instancia global del generador de Docker.
    
    Returns:
        Instancia del generador
    """
    global _global_docker_generator
    
    if _global_docker_generator is None:
        _global_docker_generator = DockerGenerator()
    
    return _global_docker_generator

