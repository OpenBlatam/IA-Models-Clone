"""
Container Optimizer - Optimizador de contenedores
================================================

Optimización de contenedores Docker para:
- Lightweight images
- Multi-stage builds
- Layer caching
- Security scanning
- Size optimization
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class ContainerOptimizer:
    """
    Optimizador de contenedores Docker.
    """
    
    def __init__(self, base_image: str = "python:3.11-slim") -> None:
        self.base_image = base_image
        self.optimizations: List[str] = []
    
    def generate_dockerfile(
        self,
        app_path: str = ".",
        output_path: str = "Dockerfile",
        enable_multi_stage: bool = True,
        enable_security: bool = True
    ) -> str:
        """
        Genera Dockerfile optimizado.
        
        Args:
            app_path: Ruta de la aplicación
            output_path: Ruta de salida
            enable_multi_stage: Habilitar multi-stage build
            enable_security: Habilitar optimizaciones de seguridad
        
        Returns:
            Contenido del Dockerfile
        """
        dockerfile_lines: List[str] = []
        
        if enable_multi_stage:
            # Stage 1: Builder
            dockerfile_lines.extend([
                f"FROM {self.base_image} AS builder",
                "",
                "WORKDIR /app",
                "",
                "# Install build dependencies",
                "RUN apt-get update && apt-get install -y --no-install-recommends \\",
                "    gcc \\",
                "    g++ \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "",
                "# Copy requirements and install dependencies",
                "COPY requirements.txt .",
                "RUN pip install --no-cache-dir --user -r requirements.txt",
                "",
                "# Stage 2: Runtime",
                f"FROM {self.base_image}",
                "",
                "# Create non-root user",
                "RUN groupadd -r appuser && useradd -r -g appuser appuser",
                "",
                "WORKDIR /app",
                "",
                "# Copy installed packages from builder",
                "COPY --from=builder /root/.local /home/appuser/.local",
                "",
                "# Copy application",
                f"COPY {app_path} .",
                "",
                "# Set permissions",
                "RUN chown -R appuser:appuser /app",
                "",
                "# Use non-root user",
                "USER appuser",
                "",
                "# Add local bin to PATH",
                "ENV PATH=/home/appuser/.local/bin:$PATH",
                "",
                "# Expose port",
                "EXPOSE 8000",
                "",
                "# Health check",
                "HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\",
                "    CMD python -c 'import requests; requests.get(\"http://localhost:8000/health\")' || exit 1",
                "",
                "# Run application",
                "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]"
            ])
        else:
            # Single stage
            dockerfile_lines.extend([
                f"FROM {self.base_image}",
                "",
                "WORKDIR /app",
                "",
                "# Install dependencies",
                "RUN apt-get update && apt-get install -y --no-install-recommends \\",
                "    && rm -rf /var/lib/apt/lists/*",
                "",
                "# Copy requirements and install",
                "COPY requirements.txt .",
                "RUN pip install --no-cache-dir -r requirements.txt",
                "",
                "# Copy application",
                f"COPY {app_path} .",
                "",
                "# Expose port",
                "EXPOSE 8000",
                "",
                "# Run application",
                "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8000\"]"
            ])
        
        if enable_security:
            # Agregar optimizaciones de seguridad
            security_lines = [
                "",
                "# Security: Remove unnecessary packages",
                "RUN apt-get autoremove -y && apt-get clean",
                "",
                "# Security: Set read-only filesystem where possible",
                "# Note: Adjust based on application needs"
            ]
            dockerfile_lines.extend(security_lines)
        
        dockerfile_content = "\n".join(dockerfile_lines)
        
        # Escribir archivo
        output_file = Path(output_path)
        output_file.write_text(dockerfile_content)
        
        logger.info(f"Dockerfile generated at {output_path}")
        return dockerfile_content
    
    def generate_dockerignore(self, output_path: str = ".dockerignore") -> str:
        """Genera .dockerignore optimizado"""
        ignore_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".Python",
            "*.so",
            ".git",
            ".gitignore",
            ".env",
            "venv",
            "env",
            ".venv",
            "*.egg-info",
            "dist",
            "build",
            ".pytest_cache",
            ".mypy_cache",
            ".coverage",
            "htmlcov",
            "*.log",
            ".idea",
            ".vscode",
            "*.swp",
            "*.swo",
            "*~",
            ".DS_Store"
        ]
        
        content = "\n".join(ignore_patterns)
        output_file = Path(output_path)
        output_file.write_text(content)
        
        logger.info(f".dockerignore generated at {output_path}")
        return content
    
    def generate_docker_compose(
        self,
        output_path: str = "docker-compose.yml",
        services: Optional[List[str]] = None
    ) -> str:
        """Genera docker-compose.yml"""
        services = services or ["app", "redis", "postgres"]
        
        compose_content = {
            "version": "3.8",
            "services": {}
        }
        
        # App service
        if "app" in services:
            compose_content["services"]["app"] = {
                "build": {
                    "context": ".",
                    "dockerfile": "Dockerfile"
                },
                "ports": ["8000:8000"],
                "environment": {
                    "ENVIRONMENT": "development"
                },
                "depends_on": ["redis"],
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                }
            }
        
        # Redis service
        if "redis" in services:
            compose_content["services"]["redis"] = {
                "image": "redis:7-alpine",
                "ports": ["6379:6379"],
                "volumes": ["redis_data:/data"]
            }
        
        # Postgres service
        if "postgres" in services:
            compose_content["services"]["postgres"] = {
                "image": "postgres:15-alpine",
                "ports": ["5432:5432"],
                "environment": {
                    "POSTGRES_DB": "appdb",
                    "POSTGRES_USER": "appuser",
                    "POSTGRES_PASSWORD": "apppass"
                },
                "volumes": ["postgres_data:/var/lib/postgresql/data"]
            }
        
        compose_content["volumes"] = {
            "redis_data": {},
            "postgres_data": {}
        }
        
        try:
            import yaml
            content = yaml.dump(compose_content, default_flow_style=False)
        except ImportError:
            import json
            content = json.dumps(compose_content, indent=2)
            logger.warning("yaml not available, using JSON format")
        
        output_file = Path(output_path)
        output_file.write_text(content)
        
        logger.info(f"docker-compose.yml generated at {output_path}")
        return content
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones de optimización"""
        recommendations = [
            {
                "type": "multi_stage_build",
                "priority": "high",
                "message": "Use multi-stage builds to reduce image size",
                "impact": "Can reduce image size by 50-70%"
            },
            {
                "type": "layer_caching",
                "priority": "high",
                "message": "Order Dockerfile commands to maximize cache hits",
                "impact": "Faster builds on subsequent runs"
            },
            {
                "type": "alpine_base",
                "priority": "medium",
                "message": "Use Alpine-based images for smaller size",
                "impact": "Can reduce base image size by 80%"
            },
            {
                "type": "non_root_user",
                "priority": "high",
                "message": "Run container as non-root user",
                "impact": "Improved security"
            },
            {
                "type": "health_checks",
                "priority": "medium",
                "message": "Add health checks to containers",
                "impact": "Better orchestration and monitoring"
            }
        ]
        
        return recommendations


def get_container_optimizer() -> ContainerOptimizer:
    """Obtiene optimizador de contenedores"""
    return ContainerOptimizer()

