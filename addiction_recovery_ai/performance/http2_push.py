"""
HTTP/2 Server Push Optimizer
Hints for HTTP/2 server push to preload critical resources
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class PushResource:
    """Resource to push via HTTP/2"""
    path: str
    as_type: str  # "script", "style", "image", "font", etc.
    priority: int = 5  # 1-10, higher = more important
    crossorigin: Optional[str] = None


class HTTP2PushOptimizer:
    """
    HTTP/2 server push optimizer
    
    Features:
    - Resource push hints
    - Priority-based pushing
    - Dependency tracking
    - Critical resource identification
    """
    
    def __init__(self):
        self._push_resources: Dict[str, List[PushResource]] = {}
        self._dependencies: Dict[str, List[str]] = {}
        logger.info("✅ HTTP/2 push optimizer initialized")
    
    def register_push_resource(
        self,
        endpoint: str,
        resource: PushResource
    ):
        """
        Register resource to push for endpoint
        
        Args:
            endpoint: API endpoint path
            resource: Resource to push
        """
        if endpoint not in self._push_resources:
            self._push_resources[endpoint] = []
        
        self._push_resources[endpoint].append(resource)
        logger.debug(f"Registered push resource: {resource.path} for {endpoint}")
    
    def get_push_resources(self, endpoint: str) -> List[PushResource]:
        """
        Get resources to push for endpoint
        
        Args:
            endpoint: API endpoint path
            
        Returns:
            List of resources to push
        """
        # Get direct resources
        resources = self._push_resources.get(endpoint, []).copy()
        
        # Add dependency resources
        if endpoint in self._dependencies:
            for dep_endpoint in self._dependencies[endpoint]:
                dep_resources = self._push_resources.get(dep_endpoint, [])
                resources.extend(dep_resources)
        
        # Sort by priority
        resources.sort(key=lambda r: r.priority, reverse=True)
        
        return resources
    
    def generate_link_header(self, resources: List[PushResource]) -> str:
        """
        Generate Link header for HTTP/2 push
        
        Args:
            resources: Resources to push
            
        Returns:
            Link header value
        """
        links = []
        for resource in resources:
            link_parts = [f"<{resource.path}>"]
            link_parts.append(f"rel=preload")
            link_parts.append(f"as={resource.as_type}")
            
            if resource.crossorigin:
                link_parts.append(f"crossorigin={resource.crossorigin}")
            
            links.append("; ".join(link_parts))
        
        return ", ".join(links)
    
    def add_dependency(self, endpoint: str, dependency_endpoint: str):
        """
        Add dependency between endpoints
        
        Args:
            endpoint: Main endpoint
            dependency_endpoint: Dependent endpoint
        """
        if endpoint not in self._dependencies:
            self._dependencies[endpoint] = []
        
        if dependency_endpoint not in self._dependencies[endpoint]:
            self._dependencies[endpoint].append(dependency_endpoint)


# Global optimizer instance
_push_optimizer: Optional[HTTP2PushOptimizer] = None


def get_http2_push_optimizer() -> HTTP2PushOptimizer:
    """Get global HTTP/2 push optimizer instance"""
    global _push_optimizer
    if _push_optimizer is None:
        _push_optimizer = HTTP2PushOptimizer()
    return _push_optimizer















