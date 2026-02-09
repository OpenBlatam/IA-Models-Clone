"""
Deployment Utilities
====================

Utilidades para deployment y despliegue.
"""

import logging
import subprocess
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConfig:
    """Configuración de deployment."""
    environment: str
    host: str
    port: int
    workers: int = 4
    reload: bool = False
    log_level: str = "info"
    extra_config: Dict[str, Any] = field(default_factory=dict)


class DeploymentManager:
    """
    Gestor de deployment.
    
    Gestiona deployment del sistema.
    """
    
    def __init__(self):
        """Inicializar gestor de deployment."""
        self.deployment_history: List[Dict[str, Any]] = []
    
    def create_deployment_config(
        self,
        environment: str = "production",
        host: str = "0.0.0.0",
        port: int = 8010,
        workers: int = 4,
        reload: bool = False,
        log_level: str = "info"
    ) -> DeploymentConfig:
        """
        Crear configuración de deployment.
        
        Args:
            environment: Entorno (production, staging, development)
            host: Host
            port: Puerto
            workers: Número de workers
            reload: Si recargar automáticamente
            log_level: Nivel de log
            
        Returns:
            Configuración de deployment
        """
        return DeploymentConfig(
            environment=environment,
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level=log_level
        )
    
    def generate_startup_script(
        self,
        config: DeploymentConfig,
        output_file: str = "start_server.sh"
    ) -> str:
        """
        Generar script de inicio.
        
        Args:
            config: Configuración de deployment
            output_file: Archivo de salida
            
        Returns:
            Ruta del archivo generado
        """
        path = Path(output_file)
        
        if path.suffix == '.sh':
            # Script bash
            script_content = f"""#!/bin/bash
# Robot Movement AI - Startup Script
# Generated: {datetime.now().isoformat()}

export ENVIRONMENT={config.environment}
export HOST={config.host}
export PORT={config.port}
export WORKERS={config.workers}
export LOG_LEVEL={config.log_level}

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Ejecutar servidor
uvicorn robot_movement_ai.main:app \\
    --host $HOST \\
    --port $PORT \\
    --workers $WORKERS \\
    --log-level $LOG_LEVEL \\
    {"--reload" if config.reload else ""}
"""
        else:
            # Script PowerShell
            script_content = f"""# Robot Movement AI - Startup Script
# Generated: {datetime.now().isoformat()}

$env:ENVIRONMENT = "{config.environment}"
$env:HOST = "{config.host}"
$env:PORT = "{config.port}"
$env:WORKERS = "{config.workers}"
$env:LOG_LEVEL = "{config.log_level}"

# Activar entorno virtual si existe
if (Test-Path "venv") {{
    & .\\venv\\Scripts\\Activate.ps1
}}

# Ejecutar servidor
uvicorn robot_movement_ai.main:app `
    --host $env:HOST `
    --port $env:PORT `
    --workers $env:WORKERS `
    --log-level $env:LOG_LEVEL `
    {"--reload" if config.reload else ""}
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Hacer ejecutable en Unix
        if path.suffix == '.sh':
            import os
            os.chmod(path, 0o755)
        
        logger.info(f"Startup script generated: {output_file}")
        return str(path)
    
    def generate_dockerfile(
        self,
        output_file: str = "Dockerfile",
        python_version: str = "3.11"
    ) -> str:
        """
        Generar Dockerfile.
        
        Args:
            output_file: Archivo de salida
            python_version: Versión de Python
            
        Returns:
            Ruta del archivo generado
        """
        path = Path(output_file)
        
        dockerfile_content = f"""FROM python:{python_version}-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependencias Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código
COPY . .

# Exponer puerto
EXPOSE 8010

# Comando por defecto
CMD ["uvicorn", "robot_movement_ai.main:app", "--host", "0.0.0.0", "--port", "8010"]
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(dockerfile_content)
        
        logger.info(f"Dockerfile generated: {output_file}")
        return str(path)
    
    def generate_docker_compose(
        self,
        output_file: str = "docker-compose.yml"
    ) -> str:
        """
        Generar docker-compose.yml.
        
        Args:
            output_file: Archivo de salida
            
        Returns:
            Ruta del archivo generado
        """
        path = Path(output_file)
        
        compose_content = """version: '3.8'

services:
  robot-movement-ai:
    build: .
    ports:
      - "8010:8010"
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=info
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8010/health"]
      interval: 30s
      timeout: 10s
      retries: 3
"""
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(compose_content)
        
        logger.info(f"Docker Compose file generated: {output_file}")
        return str(path)
    
    def record_deployment(
        self,
        environment: str,
        version: str,
        status: str
    ) -> None:
        """
        Registrar deployment.
        
        Args:
            environment: Entorno
            version: Versión
            status: Estado
        """
        self.deployment_history.append({
            "environment": environment,
            "version": version,
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_deployment_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de deployments."""
        return self.deployment_history[-limit:]


# Instancia global
_deployment_manager: Optional[DeploymentManager] = None


def get_deployment_manager() -> DeploymentManager:
    """Obtener instancia global del gestor de deployment."""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = DeploymentManager()
    return _deployment_manager






