"""
Service Mesh Integration
========================

Soporte para service mesh (Istio, Linkerd) para:
- Service discovery
- Load balancing
- Circuit breaking
- Retries
- Distributed tracing
- mTLS
"""

import logging
from typing import Dict, Optional, List, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ServiceMeshType(Enum):
    """Tipo de service mesh"""
    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL = "consul"


class IstioConfig:
    """Configuración para Istio"""
    
    @staticmethod
    def generate_virtual_service(name: str, hosts: List[str], 
                                destinations: List[Dict]) -> Dict:
        """Genera VirtualService de Istio"""
        return {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "VirtualService",
            "metadata": {
                "name": name
            },
            "spec": {
                "hosts": hosts,
                "http": [
                    {
                        "match": [{"uri": {"prefix": "/"}}],
                        "route": [
                            {
                                "destination": {
                                    "host": dest["host"],
                                    "subset": dest.get("subset", "v1")
                                },
                                "weight": dest.get("weight", 100)
                            }
                            for dest in destinations
                        ],
                        "retries": {
                            "attempts": 3,
                            "perTryTimeout": "10s",
                            "retryOn": "5xx,reset,connect-failure,refused-stream"
                        },
                        "timeout": "30s"
                    }
                ]
            }
        }
    
    @staticmethod
    def generate_destination_rule(name: str, host: str,
                                   subsets: List[Dict]) -> Dict:
        """Genera DestinationRule de Istio"""
        return {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "DestinationRule",
            "metadata": {
                "name": name
            },
            "spec": {
                "host": host,
                "subsets": subsets,
                "trafficPolicy": {
                    "loadBalancer": {
                        "simple": "LEAST_CONN"
                    },
                    "connectionPool": {
                        "tcp": {
                            "maxConnections": 100
                        },
                        "http": {
                            "http1MaxPendingRequests": 10,
                            "http2MaxRequests": 10,
                            "maxRequestsPerConnection": 2
                        }
                    },
                    "circuitBreaker": {
                        "consecutiveErrors": 5,
                        "interval": "30s",
                        "baseEjectionTime": "30s",
                        "maxEjectionPercent": 50
                    }
                }
            }
        }
    
    @staticmethod
    def generate_service_entry(name: str, hosts: List[str],
                               ports: List[Dict]) -> Dict:
        """Genera ServiceEntry de Istio"""
        return {
            "apiVersion": "networking.istio.io/v1alpha3",
            "kind": "ServiceEntry",
            "metadata": {
                "name": name
            },
            "spec": {
                "hosts": hosts,
                "ports": ports,
                "location": "MESH_EXTERNAL",
                "resolution": "DNS"
            }
        }


class LinkerdConfig:
    """Configuración para Linkerd"""
    
    @staticmethod
    def generate_service_profile(name: str, namespace: str = "default") -> Dict:
        """Genera ServiceProfile de Linkerd"""
        return {
            "apiVersion": "linkerd.io/v1alpha2",
            "kind": "ServiceProfile",
            "metadata": {
                "name": name,
                "namespace": namespace
            },
            "spec": {
                "routes": [
                    {
                        "name": "/api/v1/generate",
                        "condition": {
                            "method": "POST",
                            "pathRegex": "/api/v1/generate"
                        },
                        "timeout": "30s",
                        "retries": {
                            "budget": {
                                "retryRatio": 0.2,
                                "minRetriesPerSecond": 10,
                                "ttl": "60s"
                            }
                        }
                    }
                ]
            }
        }


class ServiceMeshManager:
    """Gestor de service mesh"""
    
    def __init__(self, mesh_type: ServiceMeshType = ServiceMeshType.ISTIO):
        self.mesh_type = mesh_type
        self.configs: Dict[str, Dict] = {}
    
    def configure_service(self, name: str, hosts: List[str],
                         destinations: List[Dict]) -> Dict:
        """Configura un servicio en el service mesh"""
        if self.mesh_type == ServiceMeshType.ISTIO:
            config = {
                "virtual_service": IstioConfig.generate_virtual_service(
                    name, hosts, destinations
                ),
                "destination_rule": IstioConfig.generate_destination_rule(
                    name, hosts[0], [{"name": "v1", "labels": {"version": "v1"}}]
                )
            }
        elif self.mesh_type == ServiceMeshType.LINKERD:
            config = {
                "service_profile": LinkerdConfig.generate_service_profile(name)
            }
        else:
            config = {}
        
        self.configs[name] = config
        logger.info(f"Service {name} configured in {self.mesh_type.value}")
        return config
    
    def get_service_config(self, name: str) -> Optional[Dict]:
        """Obtiene la configuración de un servicio"""
        return self.configs.get(name)
    
    def export_configs(self) -> Dict:
        """Exporta todas las configuraciones"""
        return self.configs


# Instancia global
service_mesh_manager = ServiceMeshManager()




