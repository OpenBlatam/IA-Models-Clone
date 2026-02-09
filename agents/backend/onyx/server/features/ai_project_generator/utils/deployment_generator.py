"""
Deployment Generator - Generador de Configuraciones de Despliegue
==================================================================

Genera configuraciones para desplegar proyectos en diferentes plataformas.
Optimizado siguiendo mejores prácticas de Python y FastAPI.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _generate_vercel_json() -> Dict[str, Any]:
    """
    Genera vercel.json (función pura).
    
    Returns:
        Diccionario con configuración de Vercel
    """
    return {
        "version": 2,
        "builds": [
            {
                "src": "frontend/package.json",
                "use": "@vercel/static-build",
                "config": {
                    "distDir": "dist"
                }
            }
        ],
        "routes": [
            {
                "src": "/api/(.*)",
                "dest": "backend/main.py"
            },
            {
                "src": "/(.*)",
                "dest": "/frontend/$1"
            }
        ]
    }


def _generate_vercelignore() -> str:
    """
    Genera .vercelignore (función pura).
    
    Returns:
        Contenido del archivo .vercelignore
    """
    return """backend/venv/
backend/__pycache__/
frontend/node_modules/
frontend/dist/
*.log
.env
"""


def _generate_netlify_toml() -> str:
    """
    Genera netlify.toml (función pura).
    
    Returns:
        Contenido del archivo netlify.toml
    """
    return '''[build]
  command = "cd frontend && npm install && npm run build"
  publish = "frontend/dist"

[build.environment]
  NODE_VERSION = "18"

[[redirects]]
  from = "/api/*"
  to = "/.netlify/functions/:splat"
  status = 200

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
'''


def _generate_railway_json() -> Dict[str, Any]:
    """
    Genera railway.json (función pura).
    
    Returns:
        Diccionario con configuración de Railway
    """
    return {
        "$schema": "https://railway.app/railway.schema.json",
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "startCommand": "cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT",
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 10
        }
    }


def _generate_procfile() -> str:
    """
    Genera Procfile (función pura).
    
    Returns:
        Contenido del archivo Procfile
    """
    return "web: cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT"


def _generate_runtime_txt() -> str:
    """
    Genera runtime.txt (función pura).
    
    Returns:
        Contenido del archivo runtime.txt
    """
    return "python-3.11.0"


def _generate_slugignore() -> str:
    """
    Genera .slugignore (función pura).
    
    Returns:
        Contenido del archivo .slugignore
    """
    return """frontend/
.git/
*.pyc
__pycache__/
.env
"""


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


def _write_json_file(file_path: Path, data: Dict[str, Any], encoding: str = "utf-8") -> None:
    """
    Escribir archivo JSON de forma segura (función pura).
    
    Args:
        file_path: Ruta del archivo
        data: Datos a escribir
        encoding: Codificación del archivo
        
    Raises:
        IOError: Si no se puede escribir el archivo
    """
    try:
        file_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding=encoding
        )
    except IOError as e:
        logger.error(f"Failed to write JSON file {file_path}: {e}")
        raise


class DeploymentGenerator:
    """
    Generador de configuraciones de despliegue.
    Optimizado con funciones puras y mejor manejo de errores.
    """
    
    def __init__(self) -> None:
        """Inicializa el generador de despliegue."""
        pass
    
    async def generate_vercel_config(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera configuración para Vercel.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se pueden escribir los archivos
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        try:
            _write_json_file(project_dir / "vercel.json", _generate_vercel_json())
            _write_file_safe(project_dir / ".vercelignore", _generate_vercelignore())
            
            logger.info("Vercel configuration generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate Vercel config: {e}")
            raise
    
    async def generate_netlify_config(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera configuración para Netlify.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede escribir el archivo
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        try:
            _write_file_safe(project_dir / "netlify.toml", _generate_netlify_toml())
            logger.info("Netlify configuration generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate Netlify config: {e}")
            raise
    
    async def generate_railway_config(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera configuración para Railway.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se puede escribir el archivo
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        try:
            _write_json_file(project_dir / "railway.json", _generate_railway_json())
            logger.info("Railway configuration generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate Railway config: {e}")
            raise
    
    async def generate_heroku_config(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera configuración para Heroku.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se pueden escribir los archivos
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        try:
            _write_file_safe(project_dir / "Procfile", _generate_procfile())
            _write_file_safe(project_dir / "runtime.txt", _generate_runtime_txt())
            _write_file_safe(project_dir / ".slugignore", _generate_slugignore())
            
            logger.info("Heroku configuration generated successfully")
        except IOError as e:
            logger.error(f"Failed to generate Heroku config: {e}")
            raise
    
    async def generate_all_deployments(
        self,
        project_dir: Path,
        project_info: Dict[str, Any],
    ) -> None:
        """
        Genera todas las configuraciones de despliegue.
        
        Args:
            project_dir: Directorio del proyecto
            project_info: Información del proyecto
            
        Raises:
            ValueError: Si los parámetros son inválidos
            IOError: Si no se pueden escribir los archivos
        """
        if not project_dir:
            raise ValueError("project_dir cannot be None")
        
        if not project_info:
            raise ValueError("project_info cannot be empty")
        
        try:
            await self.generate_vercel_config(project_dir, project_info)
            await self.generate_netlify_config(project_dir, project_info)
            await self.generate_railway_config(project_dir, project_info)
            await self.generate_heroku_config(project_dir, project_info)
            
            logger.info("All deployment configurations generated successfully")
        except Exception as e:
            logger.error(f"Failed to generate deployment configs: {e}")
            raise
