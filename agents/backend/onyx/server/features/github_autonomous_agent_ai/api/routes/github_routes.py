"""
GitHub OAuth Routes
==================

Rutas para autenticación OAuth con GitHub.
"""

import logging
import os
import secrets
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
import httpx

from ...config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/github", tags=["github"])

# Almacenamiento temporal de estados OAuth (en producción usar Redis)
_oauth_states: Dict[str, str] = {}
_oauth_sessions: Dict[str, Dict[str, Any]] = {}


@router.get("/auth/initiate")
async def initiate_auth() -> Dict[str, str]:
    """
    Iniciar el flujo de autenticación OAuth con GitHub.
    
    Returns:
        URL de autorización de GitHub
    """
    if not settings.GITHUB_CLIENT_ID:
        logger.error("GITHUB_CLIENT_ID no configurado")
        raise HTTPException(
            status_code=500,
            detail="GITHUB_CLIENT_ID no configurado. Por favor configura las credenciales de GitHub OAuth."
        )
    
    if not settings.GITHUB_REDIRECT_URI:
        logger.error("GITHUB_REDIRECT_URI no configurado")
        raise HTTPException(
            status_code=500,
            detail="GITHUB_REDIRECT_URI no configurado. Por favor configura la URL de callback."
        )
    
    try:
        # Generar estado aleatorio para prevenir CSRF
        state = secrets.token_urlsafe(32)
        _oauth_states[state] = "pending"
        
        # Construir URL de autorización
        redirect_uri = settings.GITHUB_REDIRECT_URI
        from urllib.parse import urlencode
        params = {
            "client_id": settings.GITHUB_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": "repo user:email",
            "response_type": "code"
        }
        auth_url = f"https://github.com/login/oauth/authorize?{urlencode(params)}"
        
        logger.info(f"Generada URL de autorización para estado: {state[:8]}...")
        
        return {"auth_url": auth_url, "state": state}
    except Exception as e:
        logger.error(f"Error al iniciar autenticación: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al iniciar autenticación: {str(e)}")


@router.get("/auth/status")
async def check_auth_status(request: Request) -> Dict[str, Any]:
    """
    Verificar el estado de autenticación actual.
    
    Returns:
        Estado de autenticación y usuario si está autenticado
    """
    # Obtener sesión del cookie o header
    session_id = request.cookies.get("github_session") or request.headers.get("X-Session-ID")
    
    if not session_id or session_id not in _oauth_sessions:
        return {"authenticated": False}
    
    session = _oauth_sessions[session_id]
    return {
        "authenticated": True,
        "user": session.get("user")
    }


@router.get("/auth/callback")
@router.post("/auth/callback")
async def handle_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Manejar el callback de OAuth de GitHub.
    
    Args:
        code: Código de autorización de GitHub
        state: Estado para validar la solicitud
        error: Error de GitHub si existe
        
    Returns:
        Resultado de la autenticación
    """
    # Si hay error, retornarlo
    if error:
        raise HTTPException(status_code=400, detail=f"Error de GitHub: {error}")
    
    # Obtener parámetros de query si no vienen como parámetros
    if not code:
        code = request.query_params.get("code")
    if not state:
        state = request.query_params.get("state")
    if not error:
        error = request.query_params.get("error")
    
    if error:
        raise HTTPException(status_code=400, detail=f"Error de GitHub: {error}")
    
    if not code:
        raise HTTPException(status_code=400, detail="Código de autorización no proporcionado")
    
    # Validar estado (opcional, puede no estar en el almacenamiento si se reinició el servidor)
    if state and state not in _oauth_states:
        logger.warning(f"Estado OAuth no encontrado: {state}")
    
    if not settings.GITHUB_CLIENT_ID or not settings.GITHUB_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="Credenciales de GitHub OAuth no configuradas"
        )
    
    try:
        # Intercambiar código por token de acceso
        async with httpx.AsyncClient() as client:
            # GitHub requiere redirect_uri en el callback si se usó en la autorización
            token_payload = {
                "client_id": settings.GITHUB_CLIENT_ID,
                "client_secret": settings.GITHUB_CLIENT_SECRET,
                "code": code,
            }
            # Agregar redirect_uri si está configurado (algunas apps de GitHub lo requieren)
            if settings.GITHUB_REDIRECT_URI:
                token_payload["redirect_uri"] = settings.GITHUB_REDIRECT_URI
            
            logger.info(f"Intercambiando código por token, redirect_uri: {settings.GITHUB_REDIRECT_URI}")
            
            token_response = await client.post(
                "https://github.com/login/oauth/access_token",
                data=token_payload,
                headers={"Accept": "application/json"},
            )
            
            # Verificar el status code antes de parsear JSON
            if token_response.status_code != 200:
                logger.error(f"Error de GitHub al intercambiar token: {token_response.status_code} - {token_response.text}")
                raise HTTPException(
                    status_code=token_response.status_code,
                    detail=f"Error de GitHub: {token_response.text}"
                )
            
            token_data = token_response.json()
            
            if "error" in token_data:
                raise HTTPException(
                    status_code=400,
                    detail=f"Error de GitHub: {token_data.get('error_description', token_data.get('error'))}"
                )
            
            access_token = token_data.get("access_token")
            if not access_token:
                raise HTTPException(status_code=400, detail="No se recibió token de acceso")
            
            # Obtener información del usuario
            user_response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            user_data = user_response.json()
            
            # Obtener email del usuario
            email = None
            try:
                emails_response = await client.get(
                    "https://api.github.com/user/emails",
                    headers={
                        "Authorization": f"token {access_token}",
                        "Accept": "application/vnd.github.v3+json",
                    },
                )
                emails = emails_response.json()
                primary_email = next((e for e in emails if e.get("primary")), None)
                if primary_email:
                    email = primary_email.get("email")
            except Exception as e:
                logger.warning(f"Error obteniendo email: {e}")
            
            user = {
                "login": user_data.get("login"),
                "name": user_data.get("name"),
                "avatar_url": user_data.get("avatar_url"),
                "email": email,
            }
            
            # Crear sesión
            session_id = secrets.token_urlsafe(32)
            _oauth_sessions[session_id] = {
                "access_token": access_token,
                "user": user,
            }
            
            # Limpiar estado usado
            if state and state in _oauth_states:
                del _oauth_states[state]
            
            from fastapi.responses import JSONResponse
            response = JSONResponse({
                "success": True,
                "user": user,
            })
            # Establecer cookie de sesión
            response.set_cookie(
                key="github_session",
                value=session_id,
                max_age=86400 * 30,  # 30 días
                httponly=True,
                samesite="lax"
            )
            
            return response
            
    except httpx.HTTPError as e:
        logger.error(f"Error en callback de OAuth: {e}")
        raise HTTPException(status_code=500, detail="Error al comunicarse con GitHub")
    except Exception as e:
        logger.error(f"Error procesando callback: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auth/logout")
async def logout(request: Request):
    """
    Cerrar sesión.
    
    Returns:
        Confirmación de cierre de sesión
    """
    from fastapi.responses import JSONResponse
    
    session_id = request.cookies.get("github_session") or request.headers.get("X-Session-ID")
    
    if session_id and session_id in _oauth_sessions:
        del _oauth_sessions[session_id]
    
    response = JSONResponse({"success": True, "message": "Sesión cerrada"})
    response.delete_cookie(key="github_session")
    return response


def get_github_token(request: Request) -> Optional[str]:
    """
    Obtener token de GitHub de la sesión actual.
    
    Args:
        request: Request de FastAPI
        
    Returns:
        Token de acceso o None
    """
    session_id = request.cookies.get("github_session") or request.headers.get("X-Session-ID")
    
    if session_id and session_id in _oauth_sessions:
        return _oauth_sessions[session_id].get("access_token")
    
    return None


@router.get("/repositories")
async def get_repositories(request: Request) -> list:
    """
    Obtener repositorios del usuario autenticado.
    
    Returns:
        Lista de repositorios
    """
    access_token = get_github_token(request)
    
    if not access_token:
        raise HTTPException(status_code=401, detail="No autenticado")
    
    try:
        async with httpx.AsyncClient() as client:
            repos_response = await client.get(
                "https://api.github.com/user/repos",
                params={"per_page": 100, "sort": "updated"},
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            
            if repos_response.status_code != 200:
                raise HTTPException(
                    status_code=repos_response.status_code,
                    detail="Error al obtener repositorios de GitHub"
                )
            
            repos_data = repos_response.json()
            
            # Formatear repositorios para el frontend
            repositories = []
            for repo in repos_data:
                repositories.append({
                    "id": repo.get("id"),
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "private": repo.get("private", False),
                    "html_url": repo.get("html_url"),
                    "language": repo.get("language"),
                    "updated_at": repo.get("updated_at"),
                    "default_branch": repo.get("default_branch", "main"),
                })
            
            return repositories
            
    except httpx.HTTPError as e:
        logger.error(f"Error obteniendo repositorios: {e}")
        raise HTTPException(status_code=500, detail="Error al comunicarse con GitHub")
    except Exception as e:
        logger.error(f"Error procesando repositorios: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/repositories/search")
async def search_repositories(request: Request, q: str) -> list:
    """
    Buscar repositorios.
    
    Args:
        q: Query de búsqueda
        
    Returns:
        Lista de repositorios encontrados
    """
    access_token = get_github_token(request)
    
    if not access_token:
        raise HTTPException(status_code=401, detail="No autenticado")
    
    try:
        async with httpx.AsyncClient() as client:
            repos_response = await client.get(
                "https://api.github.com/search/repositories",
                params={"q": q},
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            
            if repos_response.status_code != 200:
                raise HTTPException(
                    status_code=repos_response.status_code,
                    detail="Error al buscar repositorios en GitHub"
                )
            
            data = repos_response.json()
            repos_data = data.get("items", [])
            
            # Formatear repositorios para el frontend
            repositories = []
            for repo in repos_data:
                repositories.append({
                    "id": repo.get("id"),
                    "name": repo.get("name"),
                    "full_name": repo.get("full_name"),
                    "description": repo.get("description"),
                    "private": repo.get("private", False),
                    "html_url": repo.get("html_url"),
                    "language": repo.get("language"),
                    "updated_at": repo.get("updated_at"),
                    "default_branch": repo.get("default_branch", "main"),
                })
            
            return repositories
            
    except httpx.HTTPError as e:
        logger.error(f"Error buscando repositorios: {e}")
        raise HTTPException(status_code=500, detail="Error al comunicarse con GitHub")
    except Exception as e:
        logger.error(f"Error procesando búsqueda: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/repositories/{owner}/{repo}")
async def get_repository(request: Request, owner: str, repo: str) -> Dict[str, Any]:
    """
    Obtener información de un repositorio específico.
    
    Args:
        owner: Propietario del repositorio
        repo: Nombre del repositorio
        
    Returns:
        Información del repositorio
    """
    access_token = get_github_token(request)
    
    if not access_token:
        raise HTTPException(status_code=401, detail="No autenticado")
    
    try:
        async with httpx.AsyncClient() as client:
            repo_response = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}",
                headers={
                    "Authorization": f"token {access_token}",
                    "Accept": "application/vnd.github.v3+json",
                },
            )
            
            if repo_response.status_code != 200:
                raise HTTPException(
                    status_code=repo_response.status_code,
                    detail="Error al obtener repositorio de GitHub"
                )
            
            repo_data = repo_response.json()
            
            return {
                "id": repo_data.get("id"),
                "name": repo_data.get("name"),
                "full_name": repo_data.get("full_name"),
                "description": repo_data.get("description"),
                "private": repo_data.get("private", False),
                "html_url": repo_data.get("html_url"),
                "language": repo_data.get("language"),
                "updated_at": repo_data.get("updated_at"),
                "default_branch": repo_data.get("default_branch", "main"),
            }
            
    except httpx.HTTPError as e:
        logger.error(f"Error obteniendo repositorio: {e}")
        raise HTTPException(status_code=500, detail="Error al comunicarse con GitHub")
    except Exception as e:
        logger.error(f"Error procesando repositorio: {e}")
        raise HTTPException(status_code=500, detail=str(e))
