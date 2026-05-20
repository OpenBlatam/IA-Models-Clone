"""
Cliente API mejorado para integración con frontend.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import httpx
from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class APIClient:
    """
    Cliente API para comunicación con el backend.
    
    Soporta:
    - Requests async
    - Retry automático
    - Manejo de errores
    - Timeouts configurables
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Inicializar cliente API.
        
        Args:
            base_url: URL base de la API (default: desde settings)
            timeout: Timeout en segundos
            max_retries: Número máximo de reintentos
        """
        self.base_url = base_url or f"http://{settings.HOST}:{settings.PORT}"
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
        )
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Hacer request con retry automático.
        
        Args:
            method: Método HTTP
            endpoint: Endpoint de la API
            data: Datos del body
            params: Parámetros de query
            headers: Headers adicionales
            
        Returns:
            JSON response
            
        Raises:
            httpx.HTTPError: Si el request falla
        """
        url = endpoint if endpoint.startswith("http") else f"{self.base_url}{endpoint}"
        
        request_headers = self.client.headers.copy()
        if headers:
            request_headers.update(headers)
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    json=data,
                    params=params,
                    headers=request_headers
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in [401, 403, 404]:
                    # No reintentar errores de autenticación o no encontrado
                    raise
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request failed, retrying ({attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
            except httpx.RequestError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    logger.warning(f"Request error, retrying ({attempt + 1}/{self.max_retries}): {e}")
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
        
        if last_error:
            raise last_error
    
    # Agent endpoints
    async def get_agent_status(self) -> Dict[str, Any]:
        """Obtener estado del agente."""
        return await self._request("GET", "/api/v1/agent/status")
    
    async def start_agent(self) -> Dict[str, Any]:
        """Iniciar el agente."""
        return await self._request("POST", "/api/v1/agent/start")
    
    async def stop_agent(self) -> Dict[str, Any]:
        """Detener el agente."""
        return await self._request("POST", "/api/v1/agent/stop")
    
    async def pause_agent(self) -> Dict[str, Any]:
        """Pausar el agente."""
        return await self._request("POST", "/api/v1/agent/pause")
    
    async def resume_agent(self) -> Dict[str, Any]:
        """Reanudar el agente."""
        return await self._request("POST", "/api/v1/agent/resume")
    
    # Task endpoints
    async def create_task(
        self,
        repository_owner: str,
        repository_name: str,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Crear una nueva tarea.
        
        Args:
            repository_owner: Propietario del repositorio
            repository_name: Nombre del repositorio
            instruction: Instrucción a ejecutar
            metadata: Metadatos adicionales
            
        Returns:
            Información de la tarea creada
        """
        return await self._request(
            "POST",
            "/api/v1/tasks/",
            data={
                "repository_owner": repository_owner,
                "repository_name": repository_name,
                "instruction": instruction,
                "metadata": metadata or {}
            }
        )
    
    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Obtener una tarea por ID."""
        return await self._request("GET", f"/api/v1/tasks/{task_id}")
    
    async def list_tasks(
        self,
        status: Optional[str] = None,
        repository: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Listar tareas.
        
        Args:
            status: Filtrar por estado
            repository: Filtrar por repositorio (owner/repo)
            limit: Límite de resultados
            offset: Offset para paginación
            
        Returns:
            Lista de tareas
        """
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if repository:
            params["repository"] = repository
        
        return await self._request("GET", "/api/v1/tasks/", params=params)
    
    # GitHub endpoints
    async def get_repository_info(self, owner: str, repo: str) -> Dict[str, Any]:
        """Obtener información de un repositorio."""
        return await self._request("GET", f"/api/v1/github/repositories/{owner}/{repo}")
    
    async def list_repositories(
        self,
        owner: Optional[str] = None,
        limit: int = 30
    ) -> Dict[str, Any]:
        """Listar repositorios."""
        params = {"limit": limit}
        if owner:
            params["owner"] = owner
        return await self._request("GET", "/api/v1/github/repositories", params=params)
    
    # LLM endpoints
    async def generate_llm_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Generar respuesta de LLM."""
        return await self._request(
            "POST",
            "/api/v1/llm/generate",
            data={
                "prompt": prompt,
                "model": model,
                "system_prompt": system_prompt,
                "temperature": temperature
            }
        )
    
    async def analyze_code(
        self,
        code: str,
        language: Optional[str] = None,
        analysis_type: str = "general"
    ) -> Dict[str, Any]:
        """Analizar código con LLM."""
        return await self._request(
            "POST",
            "/api/v1/llm/analyze-code",
            data={
                "code": code,
                "language": language,
                "analysis_type": analysis_type
            }
        )
    
    # Stats endpoints
    async def get_stats_overview(self) -> Dict[str, Any]:
        """Obtener resumen de estadísticas."""
        return await self._request("GET", "/api/v1/stats/overview")
    
    async def get_tasks_summary(self) -> Dict[str, Any]:
        """Obtener resumen de tareas."""
        return await self._request("GET", "/api/v1/stats/tasks/summary")
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de rendimiento."""
        return await self._request("GET", "/api/v1/stats/performance")
    
    # Health check
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del servicio."""
        return await self._request("GET", "/health")
    
    async def close(self):
        """Cerrar cliente."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

