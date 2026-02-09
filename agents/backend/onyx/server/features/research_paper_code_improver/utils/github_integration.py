"""
GitHub Integration - Integración con GitHub API
===============================================
"""

import logging
from typing import Dict, Any, List, Optional
import os
import base64

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests no disponible. Instalar con: pip install requests")


class GitHubIntegration:
    """
    Integración con GitHub API para obtener código de repositorios.
    """
    
    def __init__(self, token: Optional[str] = None):
        """
        Inicializar integración con GitHub.
        
        Args:
            token: GitHub personal access token (opcional)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("Se requiere 'requests' para integración con GitHub")
        
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def get_file_content(
        self,
        repo: str,
        file_path: str,
        branch: Optional[str] = None
    ) -> Optional[str]:
        """
        Obtiene el contenido de un archivo de GitHub.
        
        Args:
            repo: Repositorio en formato "owner/repo"
            file_path: Ruta al archivo en el repositorio
            branch: Rama del repositorio (opcional, default: main)
            
        Returns:
            Contenido del archivo o None si hay error
        """
        try:
            if not branch:
                branch = self._get_default_branch(repo)
            
            url = f"{self.base_url}/repos/{repo}/contents/{file_path}"
            params = {"ref": branch}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("encoding") == "base64":
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content
            else:
                logger.warning(f"Encoding no soportado: {data.get('encoding')}")
                return None
                
        except requests.RequestException as e:
            logger.error(f"Error obteniendo archivo de GitHub: {e}")
            return None
        except Exception as e:
            logger.error(f"Error procesando respuesta: {e}")
            return None
    
    def list_repository_files(
        self,
        repo: str,
        branch: Optional[str] = None,
        path: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Lista archivos de un repositorio.
        
        Args:
            repo: Repositorio en formato "owner/repo"
            branch: Rama del repositorio (opcional)
            path: Ruta dentro del repositorio (opcional)
            
        Returns:
            Lista de archivos y directorios
        """
        try:
            if not branch:
                branch = self._get_default_branch(repo)
            
            url = f"{self.base_url}/repos/{repo}/contents/{path}"
            params = {"ref": branch}
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            items = response.json()
            
            files = []
            for item in items:
                files.append({
                    "name": item["name"],
                    "path": item["path"],
                    "type": item["type"],
                    "size": item.get("size", 0),
                    "url": item.get("html_url", "")
                })
            
            return files
            
        except requests.RequestException as e:
            logger.error(f"Error listando archivos: {e}")
            return []
        except Exception as e:
            logger.error(f"Error procesando respuesta: {e}")
            return []
    
    def _get_default_branch(self, repo: str) -> str:
        """Obtiene la rama por defecto del repositorio"""
        try:
            url = f"{self.base_url}/repos/{repo}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            return data.get("default_branch", "main")
            
        except Exception as e:
            logger.warning(f"Error obteniendo rama por defecto: {e}, usando 'main'")
            return "main"
    
    def get_repository_info(self, repo: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un repositorio.
        
        Args:
            repo: Repositorio en formato "owner/repo"
            
        Returns:
            Información del repositorio
        """
        try:
            url = f"{self.base_url}/repos/{repo}"
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "name": data.get("name", ""),
                "full_name": data.get("full_name", ""),
                "description": data.get("description", ""),
                "language": data.get("language", ""),
                "stars": data.get("stargazers_count", 0),
                "forks": data.get("forks_count", 0),
                "default_branch": data.get("default_branch", "main"),
                "url": data.get("html_url", "")
            }
            
        except requests.RequestException as e:
            logger.error(f"Error obteniendo información del repositorio: {e}")
            return None
        except Exception as e:
            logger.error(f"Error procesando respuesta: {e}")
            return None




