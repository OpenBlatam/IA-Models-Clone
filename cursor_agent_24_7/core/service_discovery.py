"""
Service Discovery - Descubrimiento de servicios
================================================

Sistema de service discovery para microservicios.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class Service:
    """Información de un servicio."""
    name: str
    host: str
    port: int
    protocol: str = "http"
    health_endpoint: str = "/api/health"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def url(self) -> str:
        """URL completa del servicio."""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def health_url(self) -> str:
        """URL del health check."""
        return f"{self.url}{self.health_endpoint}"


class ServiceRegistry:
    """
    Registro de servicios para service discovery.
    
    Soporta múltiples backends:
    - Consul
    - etcd
    - Kubernetes DNS
    - Static configuration
    """
    
    def __init__(self, backend: str = "static"):
        """
        Inicializar registry.
        
        Args:
            backend: Backend a usar ("consul", "etcd", "k8s", "static").
        """
        self.backend = backend
        self.services: Dict[str, Service] = {}
        self._client = None
        
        if backend == "consul":
            self._init_consul()
        elif backend == "etcd":
            self._init_etcd()
        elif backend == "k8s":
            self._init_k8s()
    
    def _init_consul(self) -> None:
        """Inicializar cliente Consul."""
        try:
            import consul
            consul_host = os.getenv("CONSUL_HOST", "localhost")
            consul_port = int(os.getenv("CONSUL_PORT", "8500"))
            self._client = consul.Consul(host=consul_host, port=consul_port)
            logger.info("Consul client initialized")
        except ImportError:
            logger.warning("python-consul not installed, falling back to static")
            self.backend = "static"
        except Exception as e:
            logger.warning(f"Failed to connect to Consul: {e}, falling back to static")
            self.backend = "static"
    
    def _init_etcd(self) -> None:
        """Inicializar cliente etcd."""
        try:
            import etcd3
            etcd_host = os.getenv("ETCD_HOST", "localhost")
            etcd_port = int(os.getenv("ETCD_PORT", "2379"))
            self._client = etcd3.client(host=etcd_host, port=etcd_port)
            logger.info("etcd client initialized")
        except ImportError:
            logger.warning("etcd3 not installed, falling back to static")
            self.backend = "static"
        except Exception as e:
            logger.warning(f"Failed to connect to etcd: {e}, falling back to static")
            self.backend = "static"
    
    def _init_k8s(self) -> None:
        """Inicializar para Kubernetes DNS."""
        # Kubernetes usa DNS para service discovery
        self.backend = "k8s"
        logger.info("Using Kubernetes DNS for service discovery")
    
    def register(self, service: Service) -> None:
        """
        Registrar un servicio.
        
        Args:
            service: Servicio a registrar.
        """
        if self.backend == "consul":
            self._register_consul(service)
        elif self.backend == "etcd":
            self._register_etcd(service)
        else:
            # Static registry
            self.services[service.name] = service
            logger.info(f"Service {service.name} registered at {service.url}")
    
    def _register_consul(self, service: Service) -> None:
        """Registrar en Consul."""
        try:
            self._client.agent.service.register(
                name=service.name,
                service_id=f"{service.name}-{service.host}-{service.port}",
                address=service.host,
                port=service.port,
                check=consul.Check.http(
                    url=service.health_url,
                    interval="10s"
                ),
                tags=list(service.metadata.keys())
            )
            self.services[service.name] = service
            logger.info(f"Service {service.name} registered in Consul")
        except Exception as e:
            logger.error(f"Failed to register in Consul: {e}")
    
    def _register_etcd(self, service: Service) -> None:
        """Registrar en etcd."""
        try:
            key = f"/services/{service.name}"
            value = {
                "host": service.host,
                "port": service.port,
                "protocol": service.protocol,
                "metadata": service.metadata
            }
            import json
            self._client.put(key, json.dumps(value))
            self.services[service.name] = service
            logger.info(f"Service {service.name} registered in etcd")
        except Exception as e:
            logger.error(f"Failed to register in etcd: {e}")
    
    def discover(self, service_name: str) -> Optional[Service]:
        """
        Descubrir un servicio.
        
        Args:
            service_name: Nombre del servicio.
        
        Returns:
            Servicio encontrado o None.
        """
        if self.backend == "consul":
            return self._discover_consul(service_name)
        elif self.backend == "etcd":
            return self._discover_etcd(service_name)
        elif self.backend == "k8s":
            return self._discover_k8s(service_name)
        else:
            return self.services.get(service_name)
    
    def _discover_consul(self, service_name: str) -> Optional[Service]:
        """Descubrir en Consul."""
        try:
            _, services = self._client.health.service(service_name, passing=True)
            if services:
                svc = services[0]['Service']
                return Service(
                    name=svc['Service'],
                    host=svc['Address'],
                    port=svc['Port'],
                    metadata=svc.get('Tags', {})
                )
        except Exception as e:
            logger.error(f"Failed to discover in Consul: {e}")
        return None
    
    def _discover_etcd(self, service_name: str) -> Optional[Service]:
        """Descubrir en etcd."""
        try:
            key = f"/services/{service_name}"
            value, _ = self._client.get(key)
            if value:
                import json
                data = json.loads(value)
                return Service(
                    name=service_name,
                    host=data['host'],
                    port=data['port'],
                    protocol=data.get('protocol', 'http'),
                    metadata=data.get('metadata', {})
                )
        except Exception as e:
            logger.error(f"Failed to discover in etcd: {e}")
        return None
    
    def _discover_k8s(self, service_name: str) -> Optional[Service]:
        """Descubrir en Kubernetes usando DNS."""
        # Kubernetes service discovery usa DNS
        # Formato: <service-name>.<namespace>.svc.cluster.local
        namespace = os.getenv("KUBERNETES_NAMESPACE", "default")
        host = f"{service_name}.{namespace}.svc.cluster.local"
        
        # Obtener puerto desde environment o usar default
        port = int(os.getenv(f"{service_name.upper()}_SERVICE_PORT", "80"))
        
        return Service(
            name=service_name,
            host=host,
            port=port,
            metadata={"discovery": "k8s-dns"}
        )
    
    def list_services(self) -> List[Service]:
        """Listar todos los servicios registrados."""
        if self.backend == "consul":
            return self._list_consul()
        elif self.backend == "etcd":
            return self._list_etcd()
        else:
            return list(self.services.values())
    
    def _list_consul(self) -> List[Service]:
        """Listar servicios en Consul."""
        try:
            _, services = self._client.agent.services()
            result = []
            for svc_id, svc in services.items():
                result.append(Service(
                    name=svc['Service'],
                    host=svc['Address'],
                    port=svc['Port'],
                    metadata={'id': svc_id, 'tags': svc.get('Tags', [])}
                ))
            return result
        except Exception as e:
            logger.error(f"Failed to list services in Consul: {e}")
            return []
    
    def _list_etcd(self) -> List[Service]:
        """Listar servicios en etcd."""
        try:
            services = []
            for value, _ in self._client.get_prefix("/services/"):
                import json
                data = json.loads(value)
                service_name = data.get('name', 'unknown')
                services.append(Service(
                    name=service_name,
                    host=data['host'],
                    port=data['port'],
                    protocol=data.get('protocol', 'http'),
                    metadata=data.get('metadata', {})
                ))
            return services
        except Exception as e:
            logger.error(f"Failed to list services in etcd: {e}")
            return []


# Registry global
_service_registry: Optional[ServiceRegistry] = None


def get_service_registry() -> ServiceRegistry:
    """Obtener registry global."""
    global _service_registry
    
    if _service_registry is None:
        backend = os.getenv("SERVICE_DISCOVERY_BACKEND", "static")
        _service_registry = ServiceRegistry(backend=backend)
    
    return _service_registry




