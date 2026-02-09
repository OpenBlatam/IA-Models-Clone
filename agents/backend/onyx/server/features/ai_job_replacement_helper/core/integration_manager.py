"""
Integration Manager Service - Gestor de integraciones
======================================================

Sistema para gestionar integraciones con servicios externos.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    """Tipos de integración"""
    LINKEDIN = "linkedin"
    GITHUB = "github"
    GOOGLE_CALENDAR = "google_calendar"
    OUTLOOK = "outlook"
    SLACK = "slack"
    ZOOM = "zoom"
    STRIPE = "stripe"
    SENDGRID = "sendgrid"
    AWS = "aws"
    GOOGLE_ANALYTICS = "google_analytics"


class IntegrationStatus(str, Enum):
    """Estados de integración"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


@dataclass
class Integration:
    """Integración"""
    id: str
    user_id: str
    integration_type: IntegrationType
    status: IntegrationStatus
    credentials: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    last_sync: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class IntegrationManagerService:
    """Servicio de gestión de integraciones"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.integrations: Dict[str, List[Integration]] = {}  # user_id -> integrations
        logger.info("IntegrationManagerService initialized")
    
    def create_integration(
        self,
        user_id: str,
        integration_type: IntegrationType,
        credentials: Dict[str, str],
        config: Optional[Dict[str, Any]] = None
    ) -> Integration:
        """Crear nueva integración"""
        integration_id = f"integration_{user_id}_{int(datetime.now().timestamp())}"
        
        # Validar credenciales según tipo
        if not self._validate_credentials(integration_type, credentials):
            raise ValueError(f"Invalid credentials for {integration_type.value}")
        
        integration = Integration(
            id=integration_id,
            user_id=user_id,
            integration_type=integration_type,
            status=IntegrationStatus.PENDING,
            credentials=credentials,
            config=config or {},
        )
        
        # Intentar conectar
        try:
            self._test_connection(integration)
            integration.status = IntegrationStatus.ACTIVE
        except Exception as e:
            integration.status = IntegrationStatus.ERROR
            integration.error_message = str(e)
        
        if user_id not in self.integrations:
            self.integrations[user_id] = []
        
        self.integrations[user_id].append(integration)
        
        logger.info(f"Integration created: {integration_id}")
        return integration
    
    def _validate_credentials(
        self,
        integration_type: IntegrationType,
        credentials: Dict[str, str]
    ) -> bool:
        """Validar credenciales"""
        required_fields = {
            IntegrationType.LINKEDIN: ["api_key", "api_secret"],
            IntegrationType.GITHUB: ["token"],
            IntegrationType.GOOGLE_CALENDAR: ["client_id", "client_secret", "refresh_token"],
            IntegrationType.OUTLOOK: ["client_id", "client_secret", "refresh_token"],
            IntegrationType.SLACK: ["bot_token"],
            IntegrationType.ZOOM: ["api_key", "api_secret"],
        }
        
        required = required_fields.get(integration_type, [])
        return all(field in credentials for field in required)
    
    def _test_connection(self, integration: Integration) -> bool:
        """Probar conexión con servicio externo"""
        # En producción, esto haría una llamada real a la API
        # Por ahora, simulamos
        
        if integration.integration_type == IntegrationType.LINKEDIN:
            # Simular test de LinkedIn API
            return True
        elif integration.integration_type == IntegrationType.GITHUB:
            # Simular test de GitHub API
            return True
        elif integration.integration_type == IntegrationType.GOOGLE_CALENDAR:
            # Simular test de Google Calendar API
            return True
        
        return True
    
    def sync_integration(self, integration_id: str) -> Dict[str, Any]:
        """Sincronizar datos de integración"""
        integration = self._find_integration(integration_id)
        if not integration:
            raise ValueError(f"Integration {integration_id} not found")
        
        if integration.status != IntegrationStatus.ACTIVE:
            raise ValueError(f"Integration {integration_id} is not active")
        
        # En producción, esto sincronizaría datos reales
        sync_result = {
            "integration_id": integration_id,
            "sync_type": integration.integration_type.value,
            "items_synced": 0,
            "success": True,
            "synced_at": datetime.now().isoformat(),
        }
        
        # Simular sincronización según tipo
        if integration.integration_type == IntegrationType.LINKEDIN:
            sync_result["items_synced"] = 10  # Trabajos sincronizados
        elif integration.integration_type == IntegrationType.GITHUB:
            sync_result["items_synced"] = 5  # Repositorios sincronizados
        elif integration.integration_type == IntegrationType.GOOGLE_CALENDAR:
            sync_result["items_synced"] = 3  # Eventos sincronizados
        
        integration.last_sync = datetime.now()
        integration.updated_at = datetime.now()
        
        return sync_result
    
    def _find_integration(self, integration_id: str) -> Optional[Integration]:
        """Encontrar integración"""
        for integrations in self.integrations.values():
            integration = next((i for i in integrations if i.id == integration_id), None)
            if integration:
                return integration
        return None
    
    def get_user_integrations(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtener integraciones del usuario"""
        integrations = self.integrations.get(user_id, [])
        
        return [
            {
                "id": i.id,
                "type": i.integration_type.value,
                "status": i.status.value,
                "last_sync": i.last_sync.isoformat() if i.last_sync else None,
                "error_message": i.error_message,
            }
            for i in integrations
        ]
    
    def deactivate_integration(self, integration_id: str) -> bool:
        """Desactivar integración"""
        integration = self._find_integration(integration_id)
        if integration:
            integration.status = IntegrationStatus.INACTIVE
            integration.updated_at = datetime.now()
            return True
        return False
    
    def reactivate_integration(self, integration_id: str) -> bool:
        """Reactivar integración"""
        integration = self._find_integration(integration_id)
        if not integration:
            return False
        
        try:
            self._test_connection(integration)
            integration.status = IntegrationStatus.ACTIVE
            integration.error_message = None
            integration.updated_at = datetime.now()
            return True
        except Exception as e:
            integration.status = IntegrationStatus.ERROR
            integration.error_message = str(e)
            return False




