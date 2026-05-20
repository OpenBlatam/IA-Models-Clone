"""
Servicio de autenticación OAuth con GitHub
"""

import logging
import secrets
import httpx
from typing import Optional, Dict, Any
from urllib.parse import urlencode
from ...config.settings import settings

logger = logging.getLogger(__name__)


class GitHubOAuthService:
    """Servicio para manejar autenticación OAuth con GitHub"""
    
    def __init__(self):
        self.client_id = settings.GITHUB_CLIENT_ID
        self.client_secret = settings.GITHUB_CLIENT_SECRET
        self.redirect_uri = settings.GITHUB_REDIRECT_URI
        self.github_oauth_url = "https://github.com/login/oauth"
        self.github_api_url = "https://api.github.com"
        
        # Almacenamiento temporal de estados OAuth (en producción usar Redis o DB)
        self._oauth_states: Dict[str, str] = {}
        # Almacenamiento de tokens de acceso (en producción usar DB segura)
        self._access_tokens: Dict[str, str] = {}
    
    def generate_state(self) -> str:
        """Generar un estado aleatorio para OAuth"""
        state = secrets.token_urlsafe(32)
        return state
    
    def get_authorization_url(self, state: Optional[str] = None) -> tuple[str, str]:
        """
        Obtener URL de autorización de GitHub
        
        Returns:
            Tupla (auth_url, state)
        """
        if not self.client_id:
            raise ValueError("GITHUB_CLIENT_ID no está configurado")
        
        if not state:
            state = self.generate_state()
        
        # Guardar el estado
        self._oauth_states[state] = "pending"
        
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "repo user:email",
            "state": state,
            "response_type": "code"
        }
        
        auth_url = f"{self.github_oauth_url}/authorize?{urlencode(params)}"
        return auth_url, state
    
    async def exchange_code_for_token(self, code: str, state: Optional[str] = None) -> Dict[str, Any]:
        """
        Intercambiar código de autorización por token de acceso
        
        Args:
            code: Código de autorización de GitHub
            state: Estado OAuth (opcional, para validación)
            
        Returns:
            Información del token y usuario
        """
        if not self.client_id or not self.client_secret:
            raise ValueError("GITHUB_CLIENT_ID y GITHUB_CLIENT_SECRET deben estar configurados")
        
        # Validar estado si se proporciona
        if state and state not in self._oauth_states:
            raise ValueError("Estado OAuth inválido")
        
        # Intercambiar código por token
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.github_oauth_url}/access_token",
                json={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "code": code,
                    "redirect_uri": self.redirect_uri
                },
                headers={"Accept": "application/json"}
            )
            response.raise_for_status()
            token_data = response.json()
        
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("No se recibió token de acceso")
        
        # Obtener información del usuario
        async with httpx.AsyncClient() as client:
            user_response = await client.get(
                f"{self.github_api_url}/user",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            user_response.raise_for_status()
            user_data = user_response.json()
        
        # Guardar token (usar user_id como clave)
        user_id = str(user_data.get("id"))
        self._access_tokens[user_id] = access_token
        
        # Limpiar estado
        if state:
            self._oauth_states.pop(state, None)
        
        return {
            "access_token": access_token,
            "token_type": token_data.get("token_type", "bearer"),
            "scope": token_data.get("scope", ""),
            "user": {
                "id": user_data.get("id"),
                "login": user_data.get("login"),
                "name": user_data.get("name"),
                "email": user_data.get("email"),
                "avatar_url": user_data.get("avatar_url")
            }
        }
    
    def get_user_token(self, user_id: str) -> Optional[str]:
        """Obtener token de acceso de un usuario"""
        return self._access_tokens.get(user_id)
    
    def set_user_token(self, user_id: str, token: str):
        """Guardar token de acceso de un usuario"""
        self._access_tokens[user_id] = token
    
    def revoke_user_token(self, user_id: str):
        """Revocar token de acceso de un usuario"""
        self._access_tokens.pop(user_id, None)
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Obtener información del usuario desde GitHub"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.github_api_url}/user",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def get_user_repositories(self, access_token: str, page: int = 1, per_page: int = 100) -> list[Dict[str, Any]]:
        """Obtener repositorios del usuario autenticado"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.github_api_url}/user/repos",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                },
                params={
                    "type": "all",
                    "sort": "updated",
                    "direction": "desc",
                    "page": page,
                    "per_page": per_page
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def search_repositories(self, access_token: str, query: str) -> list[Dict[str, Any]]:
        """Buscar repositorios"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.github_api_url}/search/repositories",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                },
                params={"q": query}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("items", [])
    
    async def get_repository(self, access_token: str, owner: str, repo: str) -> Dict[str, Any]:
        """Obtener información de un repositorio específico"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.github_api_url}/repos/{owner}/{repo}",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json"
                }
            )
            response.raise_for_status()
            return response.json()


# Instancia global del servicio
_oauth_service: Optional[GitHubOAuthService] = None


def get_oauth_service() -> GitHubOAuthService:
    """Obtener instancia del servicio OAuth"""
    global _oauth_service
    if _oauth_service is None:
        _oauth_service = GitHubOAuthService()
    return _oauth_service

