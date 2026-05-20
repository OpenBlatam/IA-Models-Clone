"""
Enhanced Service Integration - Sistema de integración mejorada con servicios
============================================================================
"""

import logging
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Estados de servicio"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class EnhancedServiceIntegration:
    """Sistema de integración mejorada con servicios"""
    
    def __init__(self):
        self.services: Dict[str, Dict[str, Any]] = {}
        self.service_health: Dict[str, Dict[str, Any]] = {}
        self.integration_logs: List[Dict[str, Any]] = []
    
    def register_service(self, service_id: str, name: str, base_url: str,
                        service_type: str, api_key: Optional[str] = None,
                        health_check_url: Optional[str] = None) -> Dict[str, Any]:
        """Registra un servicio externo"""
        service = {
            "id": service_id,
            "name": name,
            "base_url": base_url,
            "type": service_type,
            "api_key": api_key,
            "health_check_url": health_check_url or f"{base_url}/health",
            "status": ServiceStatus.ACTIVE.value,
            "registered_at": datetime.now().isoformat(),
            "last_health_check": None,
            "success_rate": 1.0,
            "response_times": []
        }
        
        self.services[service_id] = service
        
        logger.info(f"Servicio registrado: {service_id} - {name}")
        return service
    
    async def check_service_health(self, service_id: str) -> Dict[str, Any]:
        """Verifica salud de un servicio"""
        service = self.services.get(service_id)
        if not service:
            raise ValueError(f"Servicio no encontrado: {service_id}")
        
        start_time = datetime.now()
        
        try:
            async with aiohttp.ClientSession() as session:
                timeout = aiohttp.ClientTimeout(total=5)
                async with session.get(service["health_check_url"], timeout=timeout) as response:
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    is_healthy = response.status == 200
                    
                    health_data = {
                        "service_id": service_id,
                        "healthy": is_healthy,
                        "status_code": response.status,
                        "response_time": response_time,
                        "checked_at": datetime.now().isoformat()
                    }
                    
                    # Actualizar estado del servicio
                    service["last_health_check"] = datetime.now().isoformat()
                    service["response_times"].append(response_time)
                    
                    # Mantener solo últimos 100 tiempos
                    if len(service["response_times"]) > 100:
                        service["response_times"] = service["response_times"][-100:]
                    
                    # Calcular success rate
                    if len(service["response_times"]) > 0:
                        successful_checks = sum(1 for rt in service["response_times"] if rt < 2.0)
                        service["success_rate"] = successful_checks / len(service["response_times"])
                    
                    # Actualizar estado
                    if is_healthy and service["success_rate"] > 0.9:
                        service["status"] = ServiceStatus.ACTIVE.value
                    elif service["success_rate"] > 0.5:
                        service["status"] = ServiceStatus.DEGRADED.value
                    else:
                        service["status"] = ServiceStatus.INACTIVE.value
                    
                    self.service_health[service_id] = health_data
                    
                    return health_data
        
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            
            health_data = {
                "service_id": service_id,
                "healthy": False,
                "error": str(e),
                "response_time": response_time,
                "checked_at": datetime.now().isoformat()
            }
            
            service["status"] = ServiceStatus.INACTIVE.value
            service["last_health_check"] = datetime.now().isoformat()
            self.service_health[service_id] = health_data
            
            logger.error(f"Error verificando salud de servicio {service_id}: {e}")
            return health_data
    
    async def call_service(self, service_id: str, endpoint: str, method: str = "GET",
                          data: Optional[Dict[str, Any]] = None,
                          headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Llama a un servicio externo"""
        service = self.services.get(service_id)
        if not service:
            raise ValueError(f"Servicio no encontrado: {service_id}")
        
        if service["status"] == ServiceStatus.INACTIVE.value:
            raise Exception(f"Servicio no está activo: {service_id}")
        
        url = f"{service['base_url']}/{endpoint.lstrip('/')}"
        
        request_headers = headers or {}
        if service["api_key"]:
            request_headers["Authorization"] = f"Bearer {service['api_key']}"
        
        start_time = datetime.now()
        
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
                
                response_time = (datetime.now() - start_time).total_seconds()
                
                # Registrar log
                self._log_integration(service_id, endpoint, method, response.status, response_time, True)
                
                return {
                    "status": response.status,
                    "data": result,
                    "response_time": response_time
                }
        
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._log_integration(service_id, endpoint, method, 0, response_time, False, str(e))
            raise
    
    def _log_integration(self, service_id: str, endpoint: str, method: str,
                       status_code: int, response_time: float, success: bool,
                       error: Optional[str] = None):
        """Registra log de integración"""
        log_entry = {
            "service_id": service_id,
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "response_time": response_time,
            "success": success,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self.integration_logs.append(log_entry)
        
        # Mantener solo últimos 10000 logs
        if len(self.integration_logs) > 10000:
            self.integration_logs = self.integration_logs[-10000:]
    
    def get_service_status(self, service_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de un servicio"""
        service = self.services.get(service_id)
        if not service:
            return None
        
        health = self.service_health.get(service_id)
        
        return {
            "service": service,
            "health": health,
            "avg_response_time": sum(service["response_times"]) / len(service["response_times"]) if service["response_times"] else 0,
            "success_rate": service["success_rate"]
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de integraciones"""
        total_calls = len(self.integration_logs)
        successful_calls = sum(1 for log in self.integration_logs if log["success"])
        
        return {
            "total_services": len(self.services),
            "active_services": sum(1 for s in self.services.values() if s["status"] == ServiceStatus.ACTIVE.value),
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "success_rate": (successful_calls / total_calls * 100) if total_calls > 0 else 0,
            "avg_response_time": sum(log["response_time"] for log in self.integration_logs) / total_calls if total_calls > 0 else 0
        }




