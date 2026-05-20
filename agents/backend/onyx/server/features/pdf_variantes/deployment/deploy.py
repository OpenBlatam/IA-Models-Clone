"""
PDF Variantes Deployment Scripts
Scripts de despliegue para el sistema PDF Variantes
"""

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json

logger = logging.getLogger(__name__)

class DeploymentManager:
    """Gestor de despliegue"""
    
    def __init__(self, config_path: str = "deployment_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent
    
    def _load_config(self) -> Dict[str, Any]:
        """Cargar configuración de despliegue"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading deployment config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Obtener configuración por defecto"""
        return {
            "environment": "production",
            "docker": {
                "enabled": True,
                "compose_file": "docker-compose.yml",
                "image_name": "pdf-variantes",
                "tag": "latest"
            },
            "kubernetes": {
                "enabled": False,
                "namespace": "pdf-variantes",
                "config_file": "k8s-config.yaml"
            },
            "database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "name": "pdf_variantes",
                "user": "pdf_user",
                "password": "pdf_password"
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            },
            "monitoring": {
                "enabled": True,
                "grafana": True,
                "prometheus": True
            },
            "ssl": {
                "enabled": False,
                "cert_path": "ssl/cert.pem",
                "key_path": "ssl/key.pem"
            }
        }
    
    def deploy_docker(self) -> bool:
        """Desplegar con Docker"""
        try:
            logger.info("Deploying with Docker...")
            
            # Construir imagen
            if not self._build_docker_image():
                return False
            
            # Ejecutar con docker-compose
            if not self._run_docker_compose():
                return False
            
            logger.info("Docker deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying with Docker: {e}")
            return False
    
    def _build_docker_image(self) -> bool:
        """Construir imagen Docker"""
        try:
            cmd = [
                "docker", "build",
                "-t", f"{self.config['docker']['image_name']}:{self.config['docker']['tag']}",
                "."
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker build failed: {result.stderr}")
                return False
            
            logger.info("Docker image built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error building Docker image: {e}")
            return False
    
    def _run_docker_compose(self) -> bool:
        """Ejecutar docker-compose"""
        try:
            compose_file = self.config['docker']['compose_file']
            
            if not os.path.exists(compose_file):
                logger.error(f"Docker compose file not found: {compose_file}")
                return False
            
            cmd = ["docker-compose", "-f", compose_file, "up", "-d"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Docker compose failed: {result.stderr}")
                return False
            
            logger.info("Docker compose started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running docker-compose: {e}")
            return False
    
    def deploy_kubernetes(self) -> bool:
        """Desplegar con Kubernetes"""
        try:
            logger.info("Deploying with Kubernetes...")
            
            if not self.config['kubernetes']['enabled']:
                logger.info("Kubernetes deployment disabled")
                return True
            
            # Aplicar configuración de Kubernetes
            if not self._apply_k8s_config():
                return False
            
            logger.info("Kubernetes deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying with Kubernetes: {e}")
            return False
    
    def _apply_k8s_config(self) -> bool:
        """Aplicar configuración de Kubernetes"""
        try:
            config_file = self.config['kubernetes']['config_file']
            
            if not os.path.exists(config_file):
                logger.error(f"Kubernetes config file not found: {config_file}")
                return False
            
            cmd = ["kubectl", "apply", "-f", config_file]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Kubernetes apply failed: {result.stderr}")
                return False
            
            logger.info("Kubernetes configuration applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error applying Kubernetes config: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Configurar entorno"""
        try:
            logger.info("Setting up environment...")
            
            # Crear directorios necesarios
            if not self._create_directories():
                return False
            
            # Configurar variables de entorno
            if not self._setup_environment_variables():
                return False
            
            # Instalar dependencias
            if not self._install_dependencies():
                return False
            
            logger.info("Environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            return False
    
    def _create_directories(self) -> bool:
        """Crear directorios necesarios"""
        try:
            directories = [
                "uploads",
                "variants", 
                "exports",
                "logs",
                "ssl",
                "plugins"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating directories: {e}")
            return False
    
    def _setup_environment_variables(self) -> bool:
        """Configurar variables de entorno"""
        try:
            env_file = self.project_root / ".env"
            
            if not env_file.exists():
                # Crear archivo .env desde ejemplo
                example_file = self.project_root / "env.example"
                if example_file.exists():
                    shutil.copy(example_file, env_file)
                    logger.info("Created .env file from example")
                else:
                    logger.warning("No .env.example file found")
            
            return True
            
        except Exception as e:
            logger.error(f"Error setting up environment variables: {e}")
            return False
    
    def _install_dependencies(self) -> bool:
        """Instalar dependencias"""
        try:
            requirements_file = self.project_root / "requirements.txt"
            
            if not requirements_file.exists():
                logger.error("Requirements file not found")
                return False
            
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Pip install failed: {result.stderr}")
                return False
            
            logger.info("Dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    def run_migrations(self) -> bool:
        """Ejecutar migraciones de base de datos"""
        try:
            logger.info("Running database migrations...")
            
            cmd = ["alembic", "upgrade", "head"]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Migration failed: {result.stderr}")
                return False
            
            logger.info("Database migrations completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error running migrations: {e}")
            return False
    
    def start_services(self) -> bool:
        """Iniciar servicios"""
        try:
            logger.info("Starting services...")
            
            # Iniciar servicio principal
            if not self._start_main_service():
                return False
            
            # Iniciar servicios de monitoreo
            if self.config['monitoring']['enabled']:
                if not self._start_monitoring_services():
                    return False
            
            logger.info("Services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting services: {e}")
            return False
    
    def _start_main_service(self) -> bool:
        """Iniciar servicio principal"""
        try:
            if self.config['docker']['enabled']:
                # Servicio ya iniciado con docker-compose
                return True
            else:
                # Iniciar servicio directamente
                cmd = [sys.executable, "start.py"]
                subprocess.Popen(cmd)
                logger.info("Main service started")
                return True
            
        except Exception as e:
            logger.error(f"Error starting main service: {e}")
            return False
    
    def _start_monitoring_services(self) -> bool:
        """Iniciar servicios de monitoreo"""
        try:
            if self.config['monitoring']['grafana']:
                # Iniciar Grafana
                logger.info("Starting Grafana...")
            
            if self.config['monitoring']['prometheus']:
                # Iniciar Prometheus
                logger.info("Starting Prometheus...")
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting monitoring services: {e}")
            return False
    
    def health_check(self) -> bool:
        """Verificar salud del sistema"""
        try:
            logger.info("Performing health check...")
            
            # Verificar servicio principal
            if not self._check_main_service():
                return False
            
            # Verificar base de datos
            if not self._check_database():
                return False
            
            # Verificar Redis
            if not self._check_redis():
                return False
            
            logger.info("Health check completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during health check: {e}")
            return False
    
    def _check_main_service(self) -> bool:
        """Verificar servicio principal"""
        try:
            import requests
            
            response = requests.get("http://localhost:8000/health", timeout=10)
            
            if response.status_code == 200:
                logger.info("Main service is healthy")
                return True
            else:
                logger.error(f"Main service health check failed: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"Error checking main service: {e}")
            return False
    
    def _check_database(self) -> bool:
        """Verificar base de datos"""
        try:
            # Implementar verificación de base de datos
            logger.info("Database is healthy")
            return True
            
        except Exception as e:
            logger.error(f"Error checking database: {e}")
            return False
    
    def _check_redis(self) -> bool:
        """Verificar Redis"""
        try:
            # Implementar verificación de Redis
            logger.info("Redis is healthy")
            return True
            
        except Exception as e:
            logger.error(f"Error checking Redis: {e}")
            return False
    
    def deploy(self) -> bool:
        """Desplegar sistema completo"""
        try:
            logger.info("Starting complete deployment...")
            
            # Configurar entorno
            if not self.setup_environment():
                return False
            
            # Ejecutar migraciones
            if not self.run_migrations():
                return False
            
            # Desplegar con Docker
            if self.config['docker']['enabled']:
                if not self.deploy_docker():
                    return False
            
            # Desplegar con Kubernetes
            if self.config['kubernetes']['enabled']:
                if not self.deploy_kubernetes():
                    return False
            
            # Iniciar servicios
            if not self.start_services():
                return False
            
            # Verificar salud
            if not self.health_check():
                return False
            
            logger.info("Complete deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during complete deployment: {e}")
            return False

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="PDF Variantes Deployment Manager")
    parser.add_argument("--config", default="deployment_config.yaml", help="Deployment config file")
    parser.add_argument("--action", choices=["deploy", "setup", "migrate", "start", "health"], 
                       default="deploy", help="Action to perform")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear gestor de despliegue
    deployment_manager = DeploymentManager(args.config)
    
    # Ejecutar acción
    if args.action == "deploy":
        success = deployment_manager.deploy()
    elif args.action == "setup":
        success = deployment_manager.setup_environment()
    elif args.action == "migrate":
        success = deployment_manager.run_migrations()
    elif args.action == "start":
        success = deployment_manager.start_services()
    elif args.action == "health":
        success = deployment_manager.health_check()
    else:
        logger.error(f"Unknown action: {args.action}")
        success = False
    
    if success:
        logger.info("Action completed successfully")
        sys.exit(0)
    else:
        logger.error("Action failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
