"""
Service Discovery - Sistema de descubrimiento de servicios
==========================================================

Soporta:
- Consul
- etcd
- Kubernetes service discovery
- DNS-based discovery
"""

import logging
from typing import Dict, Optional, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceDiscoveryType(Enum):
    """Tipo de service discovery"""
    CONSUL = "consul"
    ETCD = "etcd"
    KUBERNETES = "kubernetes"
    DNS = "dns"


class ConsulServiceDiscovery:
    """Service discovery con Consul"""
    
    def __init__(self, consul_url: str = "http://localhost:8500"):
        self.consul_url = consul_url
        self.client = None
        self._setup()
    
    def _setup(self):
        """Configura cliente de Consul"""
        try:
            import consul
            self.client = consul.Consul(host=consul_url.split("://")[1].split(":")[0])
            logger.info("Consul client configured")
        except ImportError:
            logger.warning("python-consul not available. Install with: pip install python-consul")
        except Exception as e:
            logger.error(f"Failed to setup Consul: {e}")
    
    def register_service(self, name: str, address: str, port: int,
                        tags: Optional[List[str]] = None,
                        health_check: Optional[Dict] = None) -> bool:
        """Registra un servicio en Consul"""
        if not self.client:
            return False
        
        try:
            self.client.agent.service.register(
                name=name,
                service_id=f"{name}-{address}-{port}",
                address=address,
                port=port,
                tags=tags or [],
                check=health_check
            )
            logger.info(f"Service {name} registered in Consul")
            return True
        except Exception as e:
            logger.error(f"Failed to register service: {e}")
            return False
    
    def discover_service(self, name: str) -> List[Dict]:
        """Descubre instancias de un servicio"""
        if not self.client:
            return []
        
        try:
            _, services = self.client.health.service(name, passing=True)
            return [
                {
                    "address": service["Service"]["Address"],
                    "port": service["Service"]["Port"],
                    "tags": service["Service"]["Tags"]
                }
                for service in services
            ]
        except Exception as e:
            logger.error(f"Failed to discover service: {e}")
            return []


class KubernetesServiceDiscovery:
    """Service discovery para Kubernetes"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
        self.services: Dict[str, List[Dict]] = {}
    
    def discover_service(self, name: str) -> List[Dict]:
        """Descubre servicios en Kubernetes"""
        try:
            from kubernetes import client, config
            
            # Cargar configuración (in-cluster o kubeconfig)
            try:
                config.load_incluster_config()
            except:
                config.load_kube_config()
            
            v1 = client.CoreV1Api()
            
            # Obtener endpoints del servicio
            endpoints = v1.read_namespaced_endpoints(name, self.namespace)
            
            addresses = []
            for subset in endpoints.subsets:
                for address in subset.addresses:
                    addresses.append({
                        "address": address.ip,
                        "port": subset.ports[0].port if subset.ports else None
                    })
            
            return addresses
        except ImportError:
            logger.warning("kubernetes not available. Install with: pip install kubernetes")
            return []
        except Exception as e:
            logger.error(f"Failed to discover service in Kubernetes: {e}")
            return []


class DNSServiceDiscovery:
    """Service discovery basado en DNS"""
    
    def __init__(self, dns_server: Optional[str] = None):
        self.dns_server = dns_server
    
    def discover_service(self, name: str, port: Optional[int] = None) -> List[Dict]:
        """Descubre servicios usando DNS"""
        try:
            import socket
            
            # Resolver DNS
            addresses = socket.getaddrinfo(name, port or 8030, socket.AF_INET)
            
            return [
                {
                    "address": addr[4][0],
                    "port": addr[4][1]
                }
                for addr in addresses
            ]
        except Exception as e:
            logger.error(f"DNS discovery failed: {e}")
            return []


class ServiceDiscoveryManager:
    """Gestor de service discovery"""
    
    def __init__(self, discovery_type: ServiceDiscoveryType = ServiceDiscoveryType.DNS):
        self.discovery_type = discovery_type
        self.discovery: Optional[Any] = None
        
        if discovery_type == ServiceDiscoveryType.CONSUL:
            self.discovery = ConsulServiceDiscovery()
        elif discovery_type == ServiceDiscoveryType.KUBERNETES:
            self.discovery = KubernetesServiceDiscovery()
        elif discovery_type == ServiceDiscoveryType.DNS:
            self.discovery = DNSServiceDiscovery()
    
    def register_service(self, name: str, address: str, port: int, **kwargs) -> bool:
        """Registra un servicio"""
        if hasattr(self.discovery, 'register_service'):
            return self.discovery.register_service(name, address, port, **kwargs)
        return False
    
    def discover_service(self, name: str) -> List[Dict]:
        """Descubre un servicio"""
        if self.discovery:
            return self.discovery.discover_service(name)
        return []


# Instancia global
service_discovery = ServiceDiscoveryManager()




