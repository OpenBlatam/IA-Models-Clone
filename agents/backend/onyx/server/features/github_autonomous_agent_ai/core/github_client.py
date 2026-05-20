"""
GitHub Client
=============

Cliente para interactuar con la API de GitHub.
"""

import logging
from typing import Optional, Dict, Any, List
import aiohttp

logger = logging.getLogger(__name__)


class GitHubClient:
    """Cliente para la API de GitHub."""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
        }
        
        if token:
            self.headers["Authorization"] = f"token {token}"
            
    async def get_repository(self, owner: str, repo: str) -> Dict[str, Any]:
        """Obtener información de un repositorio."""
        url = f"{self.base_url}/repos/{owner}/{repo}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
                
    async def get_file_content(
        self,
        owner: str,
        repo: str,
        path: str,
        ref: Optional[str] = None
    ) -> str:
        """Obtener el contenido de un archivo."""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        params = {}
        if ref:
            params["ref"] = ref
            
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                import base64
                return base64.b64decode(data["content"]).decode("utf-8")
                
    async def create_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Crear o actualizar un archivo."""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        import base64
        data = {
            "message": message,
            "content": base64.b64encode(content.encode()).decode(),
        }
        
        if branch:
            data["branch"] = branch
            
        async with aiohttp.ClientSession() as session:
            async with session.put(url, headers=self.headers, json=data) as response:
                response.raise_for_status()
                return await response.json()
                
    async def create_pull_request(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str,
        head: str,
        base: str = "main"
    ) -> Dict[str, Any]:
        """Crear un pull request."""
        url = f"{self.base_url}/repos/{owner}/{repo}/pulls"
        
        data = {
            "title": title,
            "body": body,
            "head": head,
            "base": base
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self.headers, json=data) as response:
                response.raise_for_status()
                return await response.json()
                
    async def list_branches(
        self,
        owner: str,
        repo: str
    ) -> List[Dict[str, Any]]:
        """Listar ramas del repositorio."""
        url = f"{self.base_url}/repos/{owner}/{repo}/branches"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                return await response.json()
    
    async def update_file(
        self,
        owner: str,
        repo: str,
        path: str,
        content: str,
        message: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Actualizar un archivo existente."""
        # Primero obtener el SHA del archivo actual
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        params = {}
        if branch:
            params["ref"] = branch
            
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 404:
                    # Si no existe, crearlo
                    return await self.create_file(owner, repo, path, content, message, branch)
                response.raise_for_status()
                data = await response.json()
                sha = data["sha"]
            
            # Actualizar el archivo
            import base64
            update_data = {
                "message": message,
                "content": base64.b64encode(content.encode()).decode(),
                "sha": sha
            }
            
            if branch:
                update_data["branch"] = branch
                
            async with session.put(url, headers=self.headers, json=update_data) as update_response:
                update_response.raise_for_status()
                return await update_response.json()
    
    async def delete_file(
        self,
        owner: str,
        repo: str,
        path: str,
        message: str,
        branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Eliminar un archivo."""
        url = f"{self.base_url}/repos/{owner}/{repo}/contents/{path}"
        
        # Obtener el SHA del archivo
        params = {}
        if branch:
            params["ref"] = branch
            
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                sha = data["sha"]
            
            # Eliminar el archivo
            delete_data = {
                "message": message,
                "sha": sha
            }
            
            if branch:
                delete_data["branch"] = branch
                
            async with session.delete(url, headers=self.headers, json=delete_data) as delete_response:
                delete_response.raise_for_status()
                return await delete_response.json()
    
    async def create_branch(
        self,
        owner: str,
        repo: str,
        branch_name: str,
        from_branch: str = "main"
    ) -> Dict[str, Any]:
        """Crear una nueva rama."""
        # Primero obtener el SHA de la rama base
        url = f"{self.base_url}/repos/{owner}/{repo}/git/refs/heads/{from_branch}"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as response:
                response.raise_for_status()
                data = await response.json()
                sha = data["object"]["sha"]
            
            # Crear la nueva rama
            create_url = f"{self.base_url}/repos/{owner}/{repo}/git/refs"
            create_data = {
                "ref": f"refs/heads/{branch_name}",
                "sha": sha
            }
            
            async with session.post(create_url, headers=self.headers, json=create_data) as create_response:
                create_response.raise_for_status()
                return await create_response.json()


