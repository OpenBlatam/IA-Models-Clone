"""
Cliente de GitHub API
"""

import logging
from typing import Optional, Dict, Any, List
import httpx
from ..config.settings import settings

logger = logging.getLogger(__name__)


class GitHubClient:
    """Cliente para interactuar con la API de GitHub"""
    
    def __init__(self, token: Optional[str] = None):
        """
        Inicializar cliente de GitHub
        
        Args:
            token: Token de autenticación de GitHub. Si es None, usa el de settings.
        """
        self.token = token or settings.GITHUB_TOKEN
        self.base_url = settings.GITHUB_API_BASE_URL
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "GitHub-Autonomous-Agent-AI"
        }
        
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Realizar petición a la API de GitHub
        
        Args:
            method: Método HTTP (GET, POST, PUT, DELETE, etc.)
            endpoint: Endpoint de la API (sin el base_url)
            **kwargs: Argumentos adicionales para httpx
            
        Returns:
            Respuesta de la API como diccionario
            
        Raises:
            httpx.HTTPError: Si hay error en la petición
        """
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            with httpx.Client(headers=self.headers, timeout=30.0) as client:
                response = client.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Error en petición a GitHub: {e}")
            raise
    
    def get_repo_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """
        Obtener información de un repositorio
        
        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            
        Returns:
            Información del repositorio
        """
        return self._request("GET", f"repos/{owner}/{repo}")
    
    def list_files(self, owner: str, repo: str, path: str = "", branch: str = "main") -> List[Dict[str, Any]]:
        """
        Listar archivos de un repositorio
        
        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            path: Ruta dentro del repositorio
            branch: Rama a consultar
            
        Returns:
            Lista de archivos
        """
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        if branch:
            endpoint += f"?ref={branch}"
        
        result = self._request("GET", endpoint)
        
        if isinstance(result, list):
            return result
        return [result]
    
    def get_file_content(self, owner: str, repo: str, path: str, branch: str = "main") -> str:
        """
        Obtener contenido de un archivo
        
        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            path: Ruta del archivo
            branch: Rama a consultar
            
        Returns:
            Contenido del archivo (decodificado)
        """
        import base64
        
        endpoint = f"repos/{owner}/{repo}/contents/{path}"
        if branch:
            endpoint += f"?ref={branch}"
        
        result = self._request("GET", endpoint)
        
        if result.get("encoding") == "base64":
            content = base64.b64decode(result["content"]).decode("utf-8")
            return content
        
        return result.get("content", "")
    
    def create_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: str = "main"
    ) -> Dict[str, Any]:
        """
        Crear o actualizar un archivo
        
        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            path: Ruta del archivo
            content: Contenido del archivo
            message: Mensaje de commit
            branch: Rama donde crear el archivo
            
        Returns:
            Información del commit creado
        """
        import base64
        
        data = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": branch
        }
        
        return self._request("PUT", f"repos/{owner}/{repo}/contents/{path}", json=data)
    
    def create_branch(self, owner: str, repo: str, branch: str, from_branch: str = "main") -> Dict[str, Any]:
        """
        Crear una nueva rama
        
        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            branch: Nombre de la nueva rama
            from_branch: Rama desde la cual crear
            
        Returns:
            Información de la rama creada
        """
        refs = self._request("GET", f"repos/{owner}/{repo}/git/refs/heads/{from_branch}")
        sha = refs["object"]["sha"]
        
        data = {
            "ref": f"refs/heads/{branch}",
            "sha": sha
        }
        
        return self._request("POST", f"repos/{owner}/{repo}/git/refs", json=data)
    
    def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main"
    ) -> Dict[str, Any]:
        """
        Crear un pull request
        
        Args:
            owner: Propietario del repositorio
            repo: Nombre del repositorio
            title: Título del PR
            body: Descripción del PR
            head: Rama de origen
            base: Rama destino
            
        Returns:
            Información del PR creado
        """
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }
        
        return self._request("POST", f"repos/{owner}/{repo}/pulls", json=data)
    
    def parse_repo_url(self, repo_url: str) -> tuple[str, str]:
        """
        Parsear URL de repositorio en owner y repo
        
        Args:
            repo_url: URL o formato owner/repo
            
        Returns:
            Tupla (owner, repo)
        """
        repo_url = repo_url.strip()
        
        if "/" in repo_url:
            if repo_url.startswith("http"):
                parts = repo_url.rstrip("/").split("/")
                owner = parts[-2]
                repo = parts[-1].replace(".git", "")
            else:
                parts = repo_url.split("/")
                owner = parts[0]
                repo = parts[1].replace(".git", "")
            
            return owner, repo
        
        raise ValueError(f"Formato de repositorio inválido: {repo_url}")




