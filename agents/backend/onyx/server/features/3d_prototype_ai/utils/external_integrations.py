"""
External Integrations - Sistema de integración con servicios externos
=====================================================================
"""

import logging
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    """Tipos de integración"""
    MATERIAL_SUPPLIER = "material_supplier"
    CAD_SOFTWARE = "cad_software"
    PAYMENT_GATEWAY = "payment_gateway"
    EMAIL_SERVICE = "email_service"
    STORAGE_SERVICE = "storage_service"
    ANALYTICS_SERVICE = "analytics_service"


class ExternalIntegration:
    """Integración con servicio externo"""
    
    def __init__(self, integration_id: str, integration_type: IntegrationType,
                 base_url: str, api_key: Optional[str] = None):
        self.integration_id = integration_id
        self.integration_type = integration_type
        self.base_url = base_url
        self.api_key = api_key
        self.enabled = True
        self.last_call: Optional[datetime] = None
        self.call_count = 0
        self.error_count = 0
    
    async def call(self, endpoint: str, method: str = "GET", 
                  data: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Realiza una llamada al servicio externo"""
        if not self.enabled:
            raise Exception(f"Integración {self.integration_id} está deshabilitada")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        request_headers = headers or {}
        if self.api_key:
            request_headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=request_headers) as response:
                        result = await response.json()
                elif method.upper() == "POST":
                    async with session.post(url, json=data, headers=request_headers) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Método no soportado: {method}")
                
                self.last_call = datetime.now()
                self.call_count += 1
                
                if response.status >= 400:
                    self.error_count += 1
                    raise Exception(f"Error en integración: {response.status}")
                
                return result
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error en integración {self.integration_id}: {e}")
            raise


class ExternalIntegrationsManager:
    """Gestor de integraciones externas"""
    
    def __init__(self):
        self.integrations: Dict[str, ExternalIntegration] = {}
    
    def register_integration(self, integration: ExternalIntegration):
        """Registra una integración"""
        self.integrations[integration.integration_id] = integration
        logger.info(f"Integración registrada: {integration.integration_id}")
    
    def get_integration(self, integration_id: str) -> Optional[ExternalIntegration]:
        """Obtiene una integración"""
        return self.integrations.get(integration_id)
    
    def list_integrations(self, integration_type: Optional[IntegrationType] = None) -> List[Dict[str, Any]]:
        """Lista integraciones"""
        integrations = list(self.integrations.values())
        
        if integration_type:
            integrations = [i for i in integrations if i.integration_type == integration_type]
        
        return [
            {
                "id": i.integration_id,
                "type": i.integration_type.value,
                "enabled": i.enabled,
                "last_call": i.last_call.isoformat() if i.last_call else None,
                "call_count": i.call_count,
                "error_count": i.error_count
            }
            for i in integrations
        ]
    
    async def search_materials_external(self, query: str, 
                                       supplier_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Busca materiales en proveedores externos"""
        results = []
        
        # Buscar en integraciones de proveedores
        supplier_integrations = [
            i for i in self.integrations.values()
            if i.integration_type == IntegrationType.MATERIAL_SUPPLIER
            and (not supplier_id or i.integration_id == supplier_id)
            and i.enabled
        ]
        
        for integration in supplier_integrations:
            try:
                result = await integration.call(
                    f"/materials/search?q={query}",
                    method="GET"
                )
                if isinstance(result, list):
                    results.extend(result)
                elif isinstance(result, dict) and "results" in result:
                    results.extend(result["results"])
            except Exception as e:
                logger.warning(f"Error buscando en {integration.integration_id}: {e}")
        
        return results
    
    async def export_to_cad(self, cad_software_id: str, 
                            prototype_data: Dict[str, Any]) -> Dict[str, Any]:
        """Exporta prototipo a software CAD externo"""
        integration = self.integrations.get(cad_software_id)
        if not integration or integration.integration_type != IntegrationType.CAD_SOFTWARE:
            raise ValueError(f"Integración CAD no encontrada: {cad_software_id}")
        
        result = await integration.call(
            "/export",
            method="POST",
            data=prototype_data
        )
        
        return result




