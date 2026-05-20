"""
Service Mesh Integration Patterns
Supports Istio and Linkerd patterns
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ServiceMeshType(str, Enum):
    """Service mesh types"""
    ISTIO = "istio"
    LINKERD = "linkerd"
    NONE = "none"


@dataclass
class ServiceMeshConfig:
    """Service mesh configuration"""
    mesh_type: ServiceMeshType = ServiceMeshType.NONE
    namespace: str = "default"
    service_name: str = "dermatology-ai"
    enable_tracing: bool = True
    enable_metrics: bool = True
    enable_mtls: bool = True


class ServiceMeshClient:
    """
    Client for service mesh operations.
    Handles Istio/Linkerd specific headers and patterns.
    """
    
    def __init__(self, config: ServiceMeshConfig):
        self.config = config
        self.mesh_type = config.mesh_type
    
    def get_mesh_headers(self) -> Dict[str, str]:
        """Get headers for service mesh integration"""
        headers = {}
        
        if self.mesh_type == ServiceMeshType.ISTIO:
            # Istio-specific headers
            headers["X-Request-ID"] = os.getenv("X_REQUEST_ID", "")
            headers["X-B3-TraceId"] = os.getenv("X_B3_TRACEID", "")
            headers["X-B3-SpanId"] = os.getenv("X_B3_SPANID", "")
            headers["X-B3-ParentSpanId"] = os.getenv("X_B3_PARENTSPANID", "")
            
        elif self.mesh_type == ServiceMeshType.LINKERD:
            # Linkerd-specific headers
            headers["l5d-ctx-trace"] = os.getenv("L5D_CTX_TRACE", "")
            headers["l5d-ctx-deadline"] = os.getenv("L5D_CTX_DEADLINE", "")
        
        return headers
    
    def should_use_mtls(self) -> bool:
        """Check if mTLS should be used"""
        return self.config.enable_mtls and self.mesh_type != ServiceMeshType.NONE
    
    def get_service_url(self, service_name: str) -> str:
        """
        Get service URL with service mesh DNS format
        
        Args:
            service_name: Name of the service
            
        Returns:
            Service URL in mesh format
        """
        if self.mesh_type == ServiceMeshType.ISTIO:
            # Istio uses standard Kubernetes DNS
            return f"http://{service_name}.{self.config.namespace}.svc.cluster.local"
        elif self.mesh_type == ServiceMeshType.LINKERD:
            # Linkerd also uses Kubernetes DNS
            return f"http://{service_name}.{self.config.namespace}.svc.cluster.local"
        else:
            # Fallback to direct URL
            return f"http://{service_name}"
    
    def inject_sidecar_config(self) -> Dict[str, Any]:
        """Get sidecar injection configuration"""
        if self.mesh_type == ServiceMeshType.ISTIO:
            return {
                "sidecar.istio.io/inject": "true",
                "sidecar.istio.io/proxyImage": "istio/proxyv2:latest",
            }
        elif self.mesh_type == ServiceMeshType.LINKERD:
            return {
                "linkerd.io/inject": "enabled",
            }
        return {}


def get_service_mesh_client() -> Optional[ServiceMeshClient]:
    """Get service mesh client from environment"""
    mesh_type = os.getenv("SERVICE_MESH_TYPE", "none").lower()
    
    if mesh_type == "none":
        return None
    
    config = ServiceMeshConfig(
        mesh_type=ServiceMeshType(mesh_type),
        namespace=os.getenv("KUBERNETES_NAMESPACE", "default"),
        service_name=os.getenv("SERVICE_NAME", "dermatology-ai"),
        enable_tracing=os.getenv("MESH_ENABLE_TRACING", "true").lower() == "true",
        enable_metrics=os.getenv("MESH_ENABLE_METRICS", "true").lower() == "true",
        enable_mtls=os.getenv("MESH_ENABLE_MTLS", "true").lower() == "true",
    )
    
    return ServiceMeshClient(config)


# Middleware for service mesh integration
class ServiceMeshMiddleware:
    """Middleware for service mesh request/response handling"""
    
    def __init__(self, mesh_client: Optional[ServiceMeshClient] = None):
        self.mesh_client = mesh_client or get_service_mesh_client()
    
    async def process_request(self, request) -> Dict[str, Any]:
        """Process incoming request from service mesh"""
        if not self.mesh_client:
            return {}
        
        # Extract mesh-specific headers
        mesh_headers = {}
        
        # Istio headers
        if "x-request-id" in request.headers:
            mesh_headers["request_id"] = request.headers["x-request-id"]
        if "x-b3-traceid" in request.headers:
            mesh_headers["trace_id"] = request.headers["x-b3-traceid"]
        if "x-b3-spanid" in request.headers:
            mesh_headers["span_id"] = request.headers["x-b3-spanid"]
        
        # Linkerd headers
        if "l5d-ctx-trace" in request.headers:
            mesh_headers["linkerd_trace"] = request.headers["l5d-ctx-trace"]
        
        return mesh_headers
    
    async def process_response(self, response, mesh_headers: Dict[str, Any]):
        """Process response for service mesh"""
        if not self.mesh_client:
            return
        
        # Add mesh-specific headers
        headers_to_add = self.mesh_client.get_mesh_headers()
        for key, value in headers_to_add.items():
            if value:
                response.headers[key] = value

