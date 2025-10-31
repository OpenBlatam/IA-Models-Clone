"""
External Integrations System for AI History Comparison
Sistema de integraciones externas para análisis de historial de IA
"""

import asyncio
import json
import logging
import aiohttp
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import base64
import hashlib
import hmac
import time
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Tipos de integración"""
    AI_SERVICE = "ai_service"
    CLOUD_STORAGE = "cloud_storage"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"
    DATABASE = "database"
    API_GATEWAY = "api_gateway"
    MONITORING = "monitoring"
    BACKUP = "backup"

class IntegrationStatus(Enum):
    """Estados de integración"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    CONFIGURING = "configuring"

@dataclass
class IntegrationConfig:
    """Configuración de integración"""
    id: str
    name: str
    type: IntegrationType
    status: IntegrationStatus
    endpoint: str
    api_key: Optional[str] = None
    secret_key: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None

@dataclass
class IntegrationResponse:
    """Respuesta de integración"""
    success: bool
    data: Any
    status_code: int
    headers: Dict[str, str]
    error_message: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

class ExternalIntegrations:
    """
    Sistema de integraciones externas
    """
    
    def __init__(
        self,
        enable_rate_limiting: bool = True,
        enable_caching: bool = True,
        cache_ttl: int = 300  # seconds
    ):
        self.enable_rate_limiting = enable_rate_limiting
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Configuraciones de integración
        self.integrations: Dict[str, IntegrationConfig] = {}
        
        # Cache de respuestas
        self.response_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, List[datetime]] = {}
        
        # Configuración
        self.config = {
            "default_timeout": 30,
            "max_retry_attempts": 3,
            "retry_delay": 1,  # seconds
            "cache_size_limit": 1000,
            "rate_limit_window": 60  # seconds
        }
        
        # Inicializar integraciones por defecto
        self._initialize_default_integrations()
    
    def _initialize_default_integrations(self):
        """Inicializar integraciones por defecto"""
        
        # OpenAI Integration
        openai_config = IntegrationConfig(
            id="openai",
            name="OpenAI API",
            type=IntegrationType.AI_SERVICE,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://api.openai.com/v1",
            headers={"Content-Type": "application/json"},
            rate_limit=60,
            timeout=60
        )
        
        # Anthropic Integration
        anthropic_config = IntegrationConfig(
            id="anthropic",
            name="Anthropic Claude API",
            type=IntegrationType.AI_SERVICE,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://api.anthropic.com/v1",
            headers={"Content-Type": "application/json"},
            rate_limit=50,
            timeout=60
        )
        
        # Google Cloud Storage
        gcs_config = IntegrationConfig(
            id="google_cloud_storage",
            name="Google Cloud Storage",
            type=IntegrationType.CLOUD_STORAGE,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://storage.googleapis.com",
            rate_limit=1000,
            timeout=30
        )
        
        # AWS S3
        s3_config = IntegrationConfig(
            id="aws_s3",
            name="AWS S3",
            type=IntegrationType.CLOUD_STORAGE,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://s3.amazonaws.com",
            rate_limit=1000,
            timeout=30
        )
        
        # Slack Integration
        slack_config = IntegrationConfig(
            id="slack",
            name="Slack API",
            type=IntegrationType.NOTIFICATION,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://slack.com/api",
            rate_limit=100,
            timeout=30
        )
        
        # Discord Integration
        discord_config = IntegrationConfig(
            id="discord",
            name="Discord API",
            type=IntegrationType.NOTIFICATION,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://discord.com/api/v10",
            rate_limit=50,
            timeout=30
        )
        
        # Telegram Integration
        telegram_config = IntegrationConfig(
            id="telegram",
            name="Telegram Bot API",
            type=IntegrationType.NOTIFICATION,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://api.telegram.org/bot",
            rate_limit=30,
            timeout=30
        )
        
        # Google Analytics
        ga_config = IntegrationConfig(
            id="google_analytics",
            name="Google Analytics",
            type=IntegrationType.ANALYTICS,
            status=IntegrationStatus.INACTIVE,
            endpoint="https://analyticsreporting.googleapis.com/v4",
            rate_limit=100,
            timeout=30
        )
        
        self.integrations = {
            "openai": openai_config,
            "anthropic": anthropic_config,
            "google_cloud_storage": gcs_config,
            "aws_s3": s3_config,
            "slack": slack_config,
            "discord": discord_config,
            "telegram": telegram_config,
            "google_analytics": ga_config
        }
    
    async def configure_integration(
        self,
        integration_id: str,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        custom_headers: Optional[Dict[str, str]] = None,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Configurar una integración
        
        Args:
            integration_id: ID de la integración
            api_key: Clave API
            secret_key: Clave secreta
            custom_headers: Headers personalizados
            custom_parameters: Parámetros personalizados
            
        Returns:
            True si se configuró exitosamente
        """
        try:
            if integration_id not in self.integrations:
                raise ValueError(f"Integration {integration_id} not found")
            
            integration = self.integrations[integration_id]
            
            # Actualizar configuración
            if api_key:
                integration.api_key = api_key
            if secret_key:
                integration.secret_key = secret_key
            if custom_headers:
                integration.headers.update(custom_headers)
            if custom_parameters:
                integration.parameters.update(custom_parameters)
            
            # Verificar conectividad
            if await self._test_integration(integration):
                integration.status = IntegrationStatus.ACTIVE
                logger.info(f"Integration {integration_id} configured successfully")
                return True
            else:
                integration.status = IntegrationStatus.ERROR
                logger.error(f"Integration {integration_id} configuration failed")
                return False
                
        except Exception as e:
            logger.error(f"Error configuring integration {integration_id}: {e}")
            return False
    
    async def _test_integration(self, integration: IntegrationConfig) -> bool:
        """Probar conectividad de una integración"""
        try:
            # Realizar una petición de prueba según el tipo
            if integration.type == IntegrationType.AI_SERVICE:
                return await self._test_ai_service(integration)
            elif integration.type == IntegrationType.CLOUD_STORAGE:
                return await self._test_cloud_storage(integration)
            elif integration.type == IntegrationType.NOTIFICATION:
                return await self._test_notification_service(integration)
            else:
                return await self._test_generic_service(integration)
                
        except Exception as e:
            logger.error(f"Error testing integration {integration.id}: {e}")
            return False
    
    async def _test_ai_service(self, integration: IntegrationConfig) -> bool:
        """Probar servicio de IA"""
        try:
            if integration.id == "openai":
                return await self._test_openai(integration)
            elif integration.id == "anthropic":
                return await self._test_anthropic(integration)
            else:
                return await self._test_generic_service(integration)
        except:
            return False
    
    async def _test_openai(self, integration: IntegrationConfig) -> bool:
        """Probar OpenAI API"""
        try:
            headers = {
                "Authorization": f"Bearer {integration.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                response = await client.post(
                    f"{integration.endpoint}/chat/completions",
                    headers=headers,
                    json=data
                )
                
                return response.status_code == 200
                
        except:
            return False
    
    async def _test_anthropic(self, integration: IntegrationConfig) -> bool:
        """Probar Anthropic API"""
        try:
            headers = {
                "x-api-key": integration.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 10,
                "messages": [{"role": "user", "content": "Hello"}]
            }
            
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                response = await client.post(
                    f"{integration.endpoint}/messages",
                    headers=headers,
                    json=data
                )
                
                return response.status_code == 200
                
        except:
            return False
    
    async def _test_cloud_storage(self, integration: IntegrationConfig) -> bool:
        """Probar servicio de almacenamiento en la nube"""
        try:
            # Implementación básica - en producción se harían pruebas específicas
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                response = await client.get(integration.endpoint)
                return response.status_code in [200, 403, 404]  # 403/404 son válidos para algunos servicios
        except:
            return False
    
    async def _test_notification_service(self, integration: IntegrationConfig) -> bool:
        """Probar servicio de notificaciones"""
        try:
            if integration.id == "slack":
                return await self._test_slack(integration)
            elif integration.id == "discord":
                return await self._test_discord(integration)
            elif integration.id == "telegram":
                return await self._test_telegram(integration)
            else:
                return await self._test_generic_service(integration)
        except:
            return False
    
    async def _test_slack(self, integration: IntegrationConfig) -> bool:
        """Probar Slack API"""
        try:
            headers = {"Authorization": f"Bearer {integration.api_key}"}
            
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                response = await client.get(
                    f"{integration.endpoint}/auth.test",
                    headers=headers
                )
                
                return response.status_code == 200
        except:
            return False
    
    async def _test_discord(self, integration: IntegrationConfig) -> bool:
        """Probar Discord API"""
        try:
            headers = {"Authorization": f"Bot {integration.api_key}"}
            
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                response = await client.get(
                    f"{integration.endpoint}/users/@me",
                    headers=headers
                )
                
                return response.status_code == 200
        except:
            return False
    
    async def _test_telegram(self, integration: IntegrationConfig) -> bool:
        """Probar Telegram API"""
        try:
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                response = await client.get(
                    f"{integration.endpoint}{integration.api_key}/getMe"
                )
                
                return response.status_code == 200
        except:
            return False
    
    async def _test_generic_service(self, integration: IntegrationConfig) -> bool:
        """Probar servicio genérico"""
        try:
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                response = await client.get(integration.endpoint)
                return response.status_code < 500
        except:
            return False
    
    async def call_integration(
        self,
        integration_id: str,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> IntegrationResponse:
        """
        Llamar a una integración externa
        
        Args:
            integration_id: ID de la integración
            method: Método HTTP
            endpoint: Endpoint específico
            data: Datos a enviar
            headers: Headers adicionales
            params: Parámetros de query
            
        Returns:
            Respuesta de la integración
        """
        try:
            if integration_id not in self.integrations:
                raise ValueError(f"Integration {integration_id} not found")
            
            integration = self.integrations[integration_id]
            
            if integration.status != IntegrationStatus.ACTIVE:
                raise ValueError(f"Integration {integration_id} is not active")
            
            # Verificar rate limiting
            if self.enable_rate_limiting and not self._check_rate_limit(integration_id):
                raise Exception(f"Rate limit exceeded for integration {integration_id}")
            
            # Verificar cache
            cache_key = self._generate_cache_key(integration_id, method, endpoint, data, params)
            if self.enable_caching and cache_key in self.response_cache:
                cached_data, cache_time = self.response_cache[cache_key]
                if (datetime.now() - cache_time).total_seconds() < self.cache_ttl:
                    logger.info(f"Returning cached response for {integration_id}")
                    return IntegrationResponse(
                        success=True,
                        data=cached_data,
                        status_code=200,
                        headers={},
                        response_time=0.0
                    )
            
            # Preparar headers
            request_headers = integration.headers.copy()
            if headers:
                request_headers.update(headers)
            
            # Agregar autenticación según el tipo de integración
            if integration.api_key:
                request_headers = self._add_authentication(integration, request_headers)
            
            # Construir URL
            url = f"{integration.endpoint}{endpoint}"
            if params:
                url += f"?{urlencode(params)}"
            
            # Realizar petición
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=integration.timeout) as client:
                if method.upper() == "GET":
                    response = await client.get(url, headers=request_headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=request_headers, json=data)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=request_headers, json=data)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=request_headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Procesar respuesta
            try:
                response_data = response.json()
            except:
                response_data = response.text
            
            # Crear respuesta
            integration_response = IntegrationResponse(
                success=response.status_code < 400,
                data=response_data,
                status_code=response.status_code,
                headers=dict(response.headers),
                response_time=response_time
            )
            
            # Actualizar rate limiting
            if self.enable_rate_limiting:
                self._update_rate_limit(integration_id)
            
            # Actualizar cache
            if self.enable_caching and integration_response.success:
                self._update_cache(cache_key, response_data)
            
            # Actualizar último uso
            integration.last_used = datetime.now()
            
            logger.info(f"Integration {integration_id} called successfully")
            return integration_response
            
        except Exception as e:
            logger.error(f"Error calling integration {integration_id}: {e}")
            return IntegrationResponse(
                success=False,
                data=None,
                status_code=0,
                headers={},
                error_message=str(e),
                response_time=0.0
            )
    
    def _add_authentication(self, integration: IntegrationConfig, headers: Dict[str, str]) -> Dict[str, str]:
        """Agregar autenticación a los headers"""
        if integration.id == "openai":
            headers["Authorization"] = f"Bearer {integration.api_key}"
        elif integration.id == "anthropic":
            headers["x-api-key"] = integration.api_key
        elif integration.id == "slack":
            headers["Authorization"] = f"Bearer {integration.api_key}"
        elif integration.id == "discord":
            headers["Authorization"] = f"Bot {integration.api_key}"
        elif integration.id == "telegram":
            # Telegram usa el token en la URL
            pass
        elif integration.id == "aws_s3":
            # AWS requiere firma
            headers = self._add_aws_signature(integration, headers)
        elif integration.id == "google_cloud_storage":
            headers["Authorization"] = f"Bearer {integration.api_key}"
        
        return headers
    
    def _add_aws_signature(self, integration: IntegrationConfig, headers: Dict[str, str]) -> Dict[str, str]:
        """Agregar firma AWS"""
        # Implementación básica - en producción se usaría boto3 o similar
        if integration.secret_key:
            headers["Authorization"] = f"AWS4-HMAC-SHA256 {integration.api_key}"
        return headers
    
    def _generate_cache_key(
        self,
        integration_id: str,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]],
        params: Optional[Dict[str, Any]]
    ) -> str:
        """Generar clave de cache"""
        key_data = {
            "integration_id": integration_id,
            "method": method,
            "endpoint": endpoint,
            "data": data,
            "params": params
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _check_rate_limit(self, integration_id: str) -> bool:
        """Verificar rate limit"""
        if integration_id not in self.rate_limits:
            return True
        
        integration = self.integrations[integration_id]
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=self.config["rate_limit_window"])
        
        # Filtrar requests dentro de la ventana
        self.rate_limits[integration_id] = [
            req_time for req_time in self.rate_limits[integration_id]
            if req_time > window_start
        ]
        
        # Verificar si excede el límite
        return len(self.rate_limits[integration_id]) < integration.rate_limit
    
    def _update_rate_limit(self, integration_id: str):
        """Actualizar rate limit"""
        if integration_id not in self.rate_limits:
            self.rate_limits[integration_id] = []
        
        self.rate_limits[integration_id].append(datetime.now())
    
    def _update_cache(self, cache_key: str, data: Any):
        """Actualizar cache"""
        # Limpiar cache si excede el límite
        if len(self.response_cache) >= self.config["cache_size_limit"]:
            # Remover entradas más antiguas
            oldest_key = min(self.response_cache.keys(), 
                           key=lambda k: self.response_cache[k][1])
            del self.response_cache[oldest_key]
        
        self.response_cache[cache_key] = (data, datetime.now())
    
    async def send_notification(
        self,
        service: str,
        message: str,
        channel: Optional[str] = None,
        title: Optional[str] = None
    ) -> bool:
        """
        Enviar notificación a través de un servicio
        
        Args:
            service: Servicio de notificación (slack, discord, telegram)
            message: Mensaje a enviar
            channel: Canal o chat ID
            title: Título opcional
            
        Returns:
            True si se envió exitosamente
        """
        try:
            if service not in self.integrations:
                raise ValueError(f"Notification service {service} not configured")
            
            integration = self.integrations[service]
            
            if integration.status != IntegrationStatus.ACTIVE:
                raise ValueError(f"Notification service {service} is not active")
            
            # Preparar datos según el servicio
            if service == "slack":
                return await self._send_slack_message(integration, message, channel, title)
            elif service == "discord":
                return await self._send_discord_message(integration, message, channel, title)
            elif service == "telegram":
                return await self._send_telegram_message(integration, message, channel, title)
            else:
                raise ValueError(f"Unsupported notification service: {service}")
                
        except Exception as e:
            logger.error(f"Error sending notification via {service}: {e}")
            return False
    
    async def _send_slack_message(
        self,
        integration: IntegrationConfig,
        message: str,
        channel: Optional[str],
        title: Optional[str]
    ) -> bool:
        """Enviar mensaje a Slack"""
        try:
            data = {
                "text": message,
                "channel": channel or "#general"
            }
            
            if title:
                data["attachments"] = [{"title": title, "text": message}]
            
            response = await self.call_integration(
                integration.id,
                "POST",
                "/chat.postMessage",
                data=data
            )
            
            return response.success
        except:
            return False
    
    async def _send_discord_message(
        self,
        integration: IntegrationConfig,
        message: str,
        channel: Optional[str],
        title: Optional[str]
    ) -> bool:
        """Enviar mensaje a Discord"""
        try:
            data = {"content": message}
            
            if title:
                data["embeds"] = [{"title": title, "description": message}]
            
            response = await self.call_integration(
                integration.id,
                "POST",
                f"/channels/{channel}/messages",
                data=data
            )
            
            return response.success
        except:
            return False
    
    async def _send_telegram_message(
        self,
        integration: IntegrationConfig,
        message: str,
        channel: Optional[str],
        title: Optional[str]
    ) -> bool:
        """Enviar mensaje a Telegram"""
        try:
            full_message = message
            if title:
                full_message = f"*{title}*\n\n{message}"
            
            data = {
                "chat_id": channel,
                "text": full_message,
                "parse_mode": "Markdown"
            }
            
            response = await self.call_integration(
                integration.id,
                "POST",
                f"/{integration.api_key}/sendMessage",
                data=data
            )
            
            return response.success
        except:
            return False
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Obtener estado de todas las integraciones"""
        return {
            "total_integrations": len(self.integrations),
            "active_integrations": len([i for i in self.integrations.values() if i.status == IntegrationStatus.ACTIVE]),
            "integrations": {
                integration_id: {
                    "name": integration.name,
                    "type": integration.type.value,
                    "status": integration.status.value,
                    "endpoint": integration.endpoint,
                    "rate_limit": integration.rate_limit,
                    "last_used": integration.last_used.isoformat() if integration.last_used else None,
                    "created_at": integration.created_at.isoformat()
                }
                for integration_id, integration in self.integrations.items()
            },
            "cache_stats": {
                "cached_responses": len(self.response_cache),
                "cache_ttl": self.cache_ttl
            },
            "rate_limit_stats": {
                integration_id: len(requests)
                for integration_id, requests in self.rate_limits.items()
            }
        }

























