"""
External APIs Integration
=========================

Integración con APIs externas para el sistema BUL.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
import json
import aiohttp
import base64
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Tipos de integración"""
    CRM = "crm"
    EMAIL = "email"
    CALENDAR = "calendar"
    STORAGE = "storage"
    ANALYTICS = "analytics"
    SOCIAL_MEDIA = "social_media"
    PAYMENT = "payment"
    NOTIFICATION = "notification"

class APIMethod(Enum):
    """Métodos HTTP"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class APICredentials:
    """Credenciales de API"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    integration_type: IntegrationType = IntegrationType.CRM
    api_key: str = ""
    api_secret: str = ""
    base_url: str = ""
    auth_type: str = "bearer"  # bearer, basic, oauth, api_key
    headers: Dict[str, str] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class APIRequest:
    """Solicitud de API"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    method: APIMethod = APIMethod.GET
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Any = None
    json_data: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class APIResponse:
    """Respuesta de API"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str = ""
    status_code: int = 0
    headers: Dict[str, str] = field(default_factory=dict)
    data: Any = None
    json_data: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    success: bool = False
    error_message: str = ""
    response_time: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

class CRMIntegration:
    """Integración con CRM"""
    
    def __init__(self, credentials: APICredentials):
        self.credentials = credentials
        self.session = None
    
    async def initialize(self):
        """Inicializar conexión CRM"""
        self.session = aiohttp.ClientSession()
    
    async def create_contact(self, contact_data: Dict[str, Any]) -> APIResponse:
        """Crear contacto en CRM"""
        try:
            url = f"{self.credentials.base_url}/contacts"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}",
                "Content-Type": "application/json"
            }
            
            request = APIRequest(
                method=APIMethod.POST,
                url=url,
                headers=headers,
                json_data=contact_data
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error creating CRM contact: {e}")
            raise
    
    async def update_contact(self, contact_id: str, contact_data: Dict[str, Any]) -> APIResponse:
        """Actualizar contacto en CRM"""
        try:
            url = f"{self.credentials.base_url}/contacts/{contact_id}"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}",
                "Content-Type": "application/json"
            }
            
            request = APIRequest(
                method=APIMethod.PUT,
                url=url,
                headers=headers,
                json_data=contact_data
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error updating CRM contact: {e}")
            raise
    
    async def get_contact(self, contact_id: str) -> APIResponse:
        """Obtener contacto del CRM"""
        try:
            url = f"{self.credentials.base_url}/contacts/{contact_id}"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}"
            }
            
            request = APIRequest(
                method=APIMethod.GET,
                url=url,
                headers=headers
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error getting CRM contact: {e}")
            raise
    
    async def _make_request(self, request: APIRequest) -> APIResponse:
        """Realizar solicitud HTTP"""
        start_time = datetime.now()
        
        try:
            async with self.session.request(
                method=request.method.value,
                url=request.url,
                headers=request.headers,
                params=request.params,
                data=request.data,
                json=request.json_data,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Leer respuesta
                text = await response.text()
                json_data = {}
                try:
                    json_data = await response.json()
                except:
                    pass
                
                return APIResponse(
                    request_id=request.id,
                    status_code=response.status,
                    headers=dict(response.headers),
                    data=json_data,
                    json_data=json_data,
                    text=text,
                    success=200 <= response.status < 300,
                    response_time=response_time
                )
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return APIResponse(
                request_id=request.id,
                status_code=0,
                success=False,
                error_message=str(e),
                response_time=response_time
            )
    
    async def close(self):
        """Cerrar sesión"""
        if self.session:
            await self.session.close()

class EmailIntegration:
    """Integración con servicios de email"""
    
    def __init__(self, credentials: APICredentials):
        self.credentials = credentials
        self.session = None
    
    async def initialize(self):
        """Inicializar conexión email"""
        self.session = aiohttp.ClientSession()
    
    async def send_email(self, to: List[str], subject: str, body: str, 
                        from_email: str = None, cc: List[str] = None, 
                        bcc: List[str] = None, attachments: List[Dict] = None) -> APIResponse:
        """Enviar email"""
        try:
            url = f"{self.credentials.base_url}/send"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}",
                "Content-Type": "application/json"
            }
            
            email_data = {
                "to": to,
                "subject": subject,
                "body": body,
                "from": from_email or self.credentials.headers.get("from_email")
            }
            
            if cc:
                email_data["cc"] = cc
            if bcc:
                email_data["bcc"] = bcc
            if attachments:
                email_data["attachments"] = attachments
            
            request = APIRequest(
                method=APIMethod.POST,
                url=url,
                headers=headers,
                json_data=email_data
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise
    
    async def _make_request(self, request: APIRequest) -> APIResponse:
        """Realizar solicitud HTTP"""
        start_time = datetime.now()
        
        try:
            async with self.session.request(
                method=request.method.value,
                url=request.url,
                headers=request.headers,
                json=request.json_data,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                
                text = await response.text()
                json_data = {}
                try:
                    json_data = await response.json()
                except:
                    pass
                
                return APIResponse(
                    request_id=request.id,
                    status_code=response.status,
                    headers=dict(response.headers),
                    data=json_data,
                    json_data=json_data,
                    text=text,
                    success=200 <= response.status < 300,
                    response_time=response_time
                )
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return APIResponse(
                request_id=request.id,
                status_code=0,
                success=False,
                error_message=str(e),
                response_time=response_time
            )
    
    async def close(self):
        """Cerrar sesión"""
        if self.session:
            await self.session.close()

class StorageIntegration:
    """Integración con servicios de almacenamiento"""
    
    def __init__(self, credentials: APICredentials):
        self.credentials = credentials
        self.session = None
    
    async def initialize(self):
        """Inicializar conexión storage"""
        self.session = aiohttp.ClientSession()
    
    async def upload_file(self, file_path: str, file_content: bytes, 
                         folder: str = None, public: bool = False) -> APIResponse:
        """Subir archivo"""
        try:
            url = f"{self.credentials.base_url}/upload"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}"
            }
            
            # Preparar datos del archivo
            data = aiohttp.FormData()
            data.add_field('file', file_content, filename=file_path)
            if folder:
                data.add_field('folder', folder)
            data.add_field('public', str(public))
            
            request = APIRequest(
                method=APIMethod.POST,
                url=url,
                headers=headers,
                data=data
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            raise
    
    async def download_file(self, file_id: str) -> APIResponse:
        """Descargar archivo"""
        try:
            url = f"{self.credentials.base_url}/files/{file_id}/download"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}"
            }
            
            request = APIRequest(
                method=APIMethod.GET,
                url=url,
                headers=headers
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise
    
    async def delete_file(self, file_id: str) -> APIResponse:
        """Eliminar archivo"""
        try:
            url = f"{self.credentials.base_url}/files/{file_id}"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}"
            }
            
            request = APIRequest(
                method=APIMethod.DELETE,
                url=url,
                headers=headers
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error deleting file: {e}")
            raise
    
    async def _make_request(self, request: APIRequest) -> APIResponse:
        """Realizar solicitud HTTP"""
        start_time = datetime.now()
        
        try:
            async with self.session.request(
                method=request.method.value,
                url=request.url,
                headers=request.headers,
                data=request.data,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                
                text = await response.text()
                json_data = {}
                try:
                    json_data = await response.json()
                except:
                    pass
                
                return APIResponse(
                    request_id=request.id,
                    status_code=response.status,
                    headers=dict(response.headers),
                    data=json_data,
                    json_data=json_data,
                    text=text,
                    success=200 <= response.status < 300,
                    response_time=response_time
                )
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return APIResponse(
                request_id=request.id,
                status_code=0,
                success=False,
                error_message=str(e),
                response_time=response_time
            )
    
    async def close(self):
        """Cerrar sesión"""
        if self.session:
            await self.session.close()

class AnalyticsIntegration:
    """Integración con servicios de analytics"""
    
    def __init__(self, credentials: APICredentials):
        self.credentials = credentials
        self.session = None
    
    async def initialize(self):
        """Inicializar conexión analytics"""
        self.session = aiohttp.ClientSession()
    
    async def track_event(self, event_name: str, properties: Dict[str, Any] = None,
                         user_id: str = None, session_id: str = None) -> APIResponse:
        """Rastrear evento"""
        try:
            url = f"{self.credentials.base_url}/events"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}",
                "Content-Type": "application/json"
            }
            
            event_data = {
                "event": event_name,
                "properties": properties or {},
                "timestamp": datetime.now().isoformat()
            }
            
            if user_id:
                event_data["user_id"] = user_id
            if session_id:
                event_data["session_id"] = session_id
            
            request = APIRequest(
                method=APIMethod.POST,
                url=url,
                headers=headers,
                json_data=event_data
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error tracking event: {e}")
            raise
    
    async def get_analytics(self, metric: str, start_date: str, end_date: str,
                           filters: Dict[str, Any] = None) -> APIResponse:
        """Obtener analytics"""
        try:
            url = f"{self.credentials.base_url}/analytics/{metric}"
            headers = {
                "Authorization": f"Bearer {self.credentials.api_key}"
            }
            
            params = {
                "start_date": start_date,
                "end_date": end_date
            }
            
            if filters:
                params.update(filters)
            
            request = APIRequest(
                method=APIMethod.GET,
                url=url,
                headers=headers,
                params=params
            )
            
            return await self._make_request(request)
            
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            raise
    
    async def _make_request(self, request: APIRequest) -> APIResponse:
        """Realizar solicitud HTTP"""
        start_time = datetime.now()
        
        try:
            async with self.session.request(
                method=request.method.value,
                url=request.url,
                headers=request.headers,
                params=request.params,
                json=request.json_data,
                timeout=aiohttp.ClientTimeout(total=request.timeout)
            ) as response:
                response_time = (datetime.now() - start_time).total_seconds()
                
                text = await response.text()
                json_data = {}
                try:
                    json_data = await response.json()
                except:
                    pass
                
                return APIResponse(
                    request_id=request.id,
                    status_code=response.status,
                    headers=dict(response.headers),
                    data=json_data,
                    json_data=json_data,
                    text=text,
                    success=200 <= response.status < 300,
                    response_time=response_time
                )
                
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            return APIResponse(
                request_id=request.id,
                status_code=0,
                success=False,
                error_message=str(e),
                response_time=response_time
            )
    
    async def close(self):
        """Cerrar sesión"""
        if self.session:
            await self.session.close()

class ExternalAPIManager:
    """
    Gestor de APIs Externas
    
    Maneja las integraciones con servicios externos para el sistema BUL.
    """
    
    def __init__(self):
        self.integrations: Dict[str, Any] = {}
        self.credentials: Dict[str, APICredentials] = {}
        self.is_initialized = False
        
        logger.info("External API Manager initialized")
    
    async def initialize(self) -> bool:
        """Inicializar el gestor de APIs externas"""
        try:
            # Cargar credenciales por defecto
            await self._load_default_credentials()
            
            # Inicializar integraciones
            await self._initialize_integrations()
            
            self.is_initialized = True
            logger.info("External API Manager fully initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize External API Manager: {e}")
            return False
    
    async def _load_default_credentials(self):
        """Cargar credenciales por defecto"""
        import os
        
        # CRM (Salesforce/HubSpot)
        if os.getenv("CRM_API_KEY"):
            crm_creds = APICredentials(
                name="CRM Integration",
                integration_type=IntegrationType.CRM,
                api_key=os.getenv("CRM_API_KEY"),
                base_url=os.getenv("CRM_BASE_URL", "https://api.hubapi.com"),
                auth_type="bearer"
            )
            self.credentials["crm"] = crm_creds
        
        # Email (SendGrid/Mailgun)
        if os.getenv("EMAIL_API_KEY"):
            email_creds = APICredentials(
                name="Email Service",
                integration_type=IntegrationType.EMAIL,
                api_key=os.getenv("EMAIL_API_KEY"),
                base_url=os.getenv("EMAIL_BASE_URL", "https://api.sendgrid.com/v3"),
                auth_type="bearer",
                headers={"from_email": os.getenv("FROM_EMAIL", "noreply@bul-system.com")}
            )
            self.credentials["email"] = email_creds
        
        # Storage (AWS S3/Google Cloud)
        if os.getenv("STORAGE_API_KEY"):
            storage_creds = APICredentials(
                name="Storage Service",
                integration_type=IntegrationType.STORAGE,
                api_key=os.getenv("STORAGE_API_KEY"),
                api_secret=os.getenv("STORAGE_API_SECRET", ""),
                base_url=os.getenv("STORAGE_BASE_URL", "https://s3.amazonaws.com"),
                auth_type="api_key"
            )
            self.credentials["storage"] = storage_creds
        
        # Analytics (Google Analytics/Mixpanel)
        if os.getenv("ANALYTICS_API_KEY"):
            analytics_creds = APICredentials(
                name="Analytics Service",
                integration_type=IntegrationType.ANALYTICS,
                api_key=os.getenv("ANALYTICS_API_KEY"),
                base_url=os.getenv("ANALYTICS_BASE_URL", "https://api.mixpanel.com"),
                auth_type="bearer"
            )
            self.credentials["analytics"] = analytics_creds
        
        logger.info(f"Loaded {len(self.credentials)} API credentials")
    
    async def _initialize_integrations(self):
        """Inicializar integraciones"""
        for name, creds in self.credentials.items():
            if creds.integration_type == IntegrationType.CRM:
                integration = CRMIntegration(creds)
                await integration.initialize()
                self.integrations[name] = integration
                
            elif creds.integration_type == IntegrationType.EMAIL:
                integration = EmailIntegration(creds)
                await integration.initialize()
                self.integrations[name] = integration
                
            elif creds.integration_type == IntegrationType.STORAGE:
                integration = StorageIntegration(creds)
                await integration.initialize()
                self.integrations[name] = integration
                
            elif creds.integration_type == IntegrationType.ANALYTICS:
                integration = AnalyticsIntegration(creds)
                await integration.initialize()
                self.integrations[name] = integration
        
        logger.info(f"Initialized {len(self.integrations)} integrations")
    
    async def add_credentials(self, credentials: APICredentials) -> str:
        """Agregar credenciales"""
        self.credentials[credentials.id] = credentials
        logger.info(f"Added credentials: {credentials.name}")
        return credentials.id
    
    async def get_integration(self, name: str) -> Optional[Any]:
        """Obtener integración"""
        return self.integrations.get(name)
    
    async def get_available_integrations(self) -> List[Dict[str, Any]]:
        """Obtener integraciones disponibles"""
        return [
            {
                "name": name,
                "type": creds.integration_type.value,
                "is_active": creds.is_active,
                "created_at": creds.created_at.isoformat()
            }
            for name, creds in self.credentials.items()
        ]
    
    async def test_integration(self, name: str) -> Dict[str, Any]:
        """Probar integración"""
        integration = self.integrations.get(name)
        if not integration:
            return {"success": False, "error": "Integration not found"}
        
        try:
            # Prueba básica según el tipo
            if hasattr(integration, 'get_contact'):
                # CRM integration
                result = await integration.get_contact("test")
            elif hasattr(integration, 'send_email'):
                # Email integration
                result = await integration.send_email(
                    to=["test@example.com"],
                    subject="Test Email",
                    body="This is a test email from BUL system"
                )
            elif hasattr(integration, 'track_event'):
                # Analytics integration
                result = await integration.track_event("test_event", {"test": True})
            else:
                return {"success": False, "error": "Unknown integration type"}
            
            return {
                "success": result.success,
                "status_code": result.status_code,
                "response_time": result.response_time,
                "error": result.error_message if not result.success else None
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def close_all(self):
        """Cerrar todas las integraciones"""
        for integration in self.integrations.values():
            if hasattr(integration, 'close'):
                await integration.close()
        
        logger.info("All integrations closed")

# Global external API manager instance
_external_api_manager: Optional[ExternalAPIManager] = None

async def get_global_external_api_manager() -> ExternalAPIManager:
    """Obtener la instancia global del gestor de APIs externas"""
    global _external_api_manager
    if _external_api_manager is None:
        _external_api_manager = ExternalAPIManager()
        await _external_api_manager.initialize()
    return _external_api_manager
























