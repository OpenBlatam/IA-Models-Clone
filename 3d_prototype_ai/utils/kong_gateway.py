"""
Kong API Gateway Integration
=============================

Integración con Kong API Gateway para:
- Rate limiting
- Request transformation
- Security filtering
- Authentication
- Load balancing
"""

import logging
import requests
from typing import Dict, Optional, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class KongAdminClient:
    """Cliente para administrar Kong API Gateway"""
    
    def __init__(self, admin_url: str = "http://localhost:8001"):
        self.admin_url = admin_url.rstrip('/')
        self.session = requests.Session()
    
    def create_service(self, name: str, url: str, tags: Optional[List[str]] = None) -> Dict:
        """Crea un servicio en Kong"""
        data = {
            "name": name,
            "url": url,
            "tags": tags or []
        }
        
        response = self.session.post(
            f"{self.admin_url}/services",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def create_route(self, service_name: str, paths: List[str], 
                    methods: Optional[List[str]] = None,
                    strip_path: bool = False) -> Dict:
        """Crea una ruta en Kong"""
        data = {
            "service": {"name": service_name},
            "paths": paths,
            "strip_path": strip_path
        }
        
        if methods:
            data["methods"] = methods
        
        response = self.session.post(
            f"{self.admin_url}/routes",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def enable_rate_limiting(self, service_name: str, 
                             minute: int = 60,
                             hour: int = 1000) -> Dict:
        """Habilita rate limiting para un servicio"""
        data = {
            "name": "rate-limiting",
            "config": {
                "minute": minute,
                "hour": hour,
                "policy": "local"
            }
        }
        
        response = self.session.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def enable_cors(self, service_name: str, 
                    origins: List[str] = None) -> Dict:
        """Habilita CORS para un servicio"""
        data = {
            "name": "cors",
            "config": {
                "origins": origins or ["*"],
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "headers": ["Authorization", "Content-Type"],
                "exposed_headers": ["X-Request-ID"],
                "credentials": True,
                "max_age": 3600
            }
        }
        
        response = self.session.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def enable_request_transformer(self, service_name: str,
                                   add_headers: Optional[Dict[str, str]] = None) -> Dict:
        """Habilita transformación de requests"""
        data = {
            "name": "request-transformer",
            "config": {
                "add": {
                    "headers": [f"{k}:{v}" for k, v in (add_headers or {}).items()]
                }
            }
        }
        
        response = self.session.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def enable_response_transformer(self, service_name: str,
                                    add_headers: Optional[Dict[str, str]] = None) -> Dict:
        """Habilita transformación de responses"""
        data = {
            "name": "response-transformer",
            "config": {
                "add": {
                    "headers": [f"{k}:{v}" for k, v in (add_headers or {}).items()]
                }
            }
        }
        
        response = self.session.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def enable_jwt_auth(self, service_name: str, 
                       key_claim_name: str = "iss") -> Dict:
        """Habilita autenticación JWT"""
        data = {
            "name": "jwt",
            "config": {
                "key_claim_name": key_claim_name,
                "secret_is_base64": False
            }
        }
        
        response = self.session.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def enable_ip_restriction(self, service_name: str,
                             whitelist: Optional[List[str]] = None,
                             blacklist: Optional[List[str]] = None) -> Dict:
        """Habilita restricción por IP"""
        data = {
            "name": "ip-restriction",
            "config": {}
        }
        
        if whitelist:
            data["config"]["whitelist"] = whitelist
        if blacklist:
            data["config"]["blacklist"] = blacklist
        
        response = self.session.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def enable_circuit_breaker(self, service_name: str,
                                threshold: int = 5,
                                timeout: int = 60) -> Dict:
        """Habilita circuit breaker"""
        data = {
            "name": "circuit-breaker",
            "config": {
                "threshold": threshold,
                "timeout": timeout,
                "unhealthy": {
                    "http_statuses": [500, 502, 503, 504],
                    "tcp_failures": 3,
                    "timeouts": 3
                },
                "healthy": {
                    "http_statuses": [200, 201, 202],
                    "interval": 10
                }
            }
        }
        
        response = self.session.post(
            f"{self.admin_url}/services/{service_name}/plugins",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def setup_service(self, name: str, url: str, paths: List[str],
                     rate_limit_minute: int = 60,
                     enable_cors: bool = True,
                     enable_jwt: bool = False) -> Dict:
        """Configura un servicio completo en Kong"""
        # Crear servicio
        service = self.create_service(name, url)
        
        # Crear rutas
        route = self.create_route(name, paths)
        
        # Habilitar plugins
        plugins = []
        
        plugins.append(self.enable_rate_limiting(name, minute=rate_limit_minute))
        
        if enable_cors:
            plugins.append(self.enable_cors(name))
        
        if enable_jwt:
            plugins.append(self.enable_jwt_auth(name))
        
        plugins.append(self.enable_circuit_breaker(name))
        
        return {
            "service": service,
            "route": route,
            "plugins": plugins
        }


class KongGatewayManager:
    """Gestor de Kong API Gateway"""
    
    def __init__(self, admin_url: str = "http://localhost:8001"):
        self.client = KongAdminClient(admin_url)
        self.services: Dict[str, Dict] = {}
    
    def register_service(self, name: str, backend_url: str, paths: List[str],
                        rate_limit: int = 60,
                        enable_cors: bool = True,
                        enable_jwt: bool = False) -> Dict:
        """Registra un servicio en Kong"""
        config = self.client.setup_service(
            name=name,
            url=backend_url,
            paths=paths,
            rate_limit_minute=rate_limit,
            enable_cors=enable_cors,
            enable_jwt=enable_jwt
        )
        
        self.services[name] = config
        logger.info(f"Service {name} registered in Kong")
        return config
    
    def get_service_config(self, name: str) -> Optional[Dict]:
        """Obtiene la configuración de un servicio"""
        return self.services.get(name)
    
    def list_services(self) -> List[str]:
        """Lista todos los servicios registrados"""
        return list(self.services.keys())


# Instancia global
kong_gateway_manager = KongGatewayManager()




