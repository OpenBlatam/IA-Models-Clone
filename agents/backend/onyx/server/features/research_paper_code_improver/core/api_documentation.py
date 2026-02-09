"""
API Documentation Generator - Generador de documentación de APIs
=================================================================
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class APIParameter:
    """Parámetro de API"""
    name: str
    type: str
    required: bool = False
    description: Optional[str] = None
    default: Optional[Any] = None
    example: Optional[Any] = None


@dataclass
class APIEndpoint:
    """Endpoint de API"""
    method: str
    path: str
    summary: str
    description: Optional[str] = None
    parameters: List[APIParameter] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False


class APIDocumentationGenerator:
    """Generador de documentación de APIs"""
    
    def __init__(self, title: str = "API Documentation", version: str = "1.0.0"):
        self.title = title
        self.version = version
        self.endpoints: List[APIEndpoint] = []
        self.info: Dict[str, Any] = {
            "title": title,
            "version": version,
            "description": "",
            "contact": {},
            "license": {}
        }
    
    def add_endpoint(self, endpoint: APIEndpoint):
        """Agrega un endpoint"""
        self.endpoints.append(endpoint)
    
    def generate_openapi(self) -> Dict[str, Any]:
        """Genera especificación OpenAPI"""
        paths = {}
        
        for endpoint in self.endpoints:
            if endpoint.path not in paths:
                paths[endpoint.path] = {}
            
            paths[endpoint.path][endpoint.method.lower()] = {
                "summary": endpoint.summary,
                "description": endpoint.description,
                "tags": endpoint.tags,
                "deprecated": endpoint.deprecated,
                "parameters": [
                    {
                        "name": param.name,
                        "in": "query" if "?" in endpoint.path else "path",
                        "required": param.required,
                        "schema": {"type": param.type},
                        "description": param.description,
                        "example": param.example
                    }
                    for param in endpoint.parameters
                ],
                "requestBody": endpoint.request_body,
                "responses": endpoint.responses
            }
        
        return {
            "openapi": "3.0.0",
            "info": self.info,
            "paths": paths
        }
    
    def generate_markdown(self) -> str:
        """Genera documentación en Markdown"""
        lines = [
            f"# {self.title}",
            f"Version: {self.version}",
            "",
            "## Endpoints",
            ""
        ]
        
        for endpoint in self.endpoints:
            lines.append(f"### {endpoint.method} {endpoint.path}")
            lines.append(f"**{endpoint.summary}**")
            if endpoint.description:
                lines.append(f"\n{endpoint.description}\n")
            
            if endpoint.parameters:
                lines.append("**Parameters:**")
                for param in endpoint.parameters:
                    req = "required" if param.required else "optional"
                    lines.append(f"- `{param.name}` ({param.type}, {req}): {param.description or ''}")
                lines.append("")
            
            if endpoint.responses:
                lines.append("**Responses:**")
                for status, resp in endpoint.responses.items():
                    lines.append(f"- `{status}`: {resp.get('description', '')}")
                lines.append("")
        
        return "\n".join(lines)




