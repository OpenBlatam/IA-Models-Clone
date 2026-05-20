"""
API Documentation - Sistema de Documentación Automática
======================================================

Sistema de generación automática de documentación de API.
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class APIDocumentation:
    """Documentación de API."""
    endpoint: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


class APIDocumentationGenerator:
    """Generador de documentación de API."""
    
    def __init__(self):
        self.endpoints: Dict[str, APIDocumentation] = {}
        self.openapi_spec: Dict[str, Any] = {
            "openapi": "3.0.0",
            "info": {
                "title": "Bulk Chat API",
                "version": "1.0.0",
                "description": "API completa para el sistema de chat continuo",
            },
            "servers": [
                {
                    "url": "http://localhost:8006",
                    "description": "Servidor de desarrollo",
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {},
            },
        }
    
    def register_endpoint(self, documentation: APIDocumentation):
        """Registrar endpoint en documentación."""
        self.endpoints[f"{documentation.method}:{documentation.endpoint}"] = documentation
        logger.debug(f"Registered endpoint: {documentation.method} {documentation.endpoint}")
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generar especificación OpenAPI."""
        paths = {}
        
        for endpoint_key, doc in self.endpoints.items():
            method, path = endpoint_key.split(":", 1)
            path_key = path
            
            if path_key not in paths:
                paths[path_key] = {}
            
            paths[path_key][method.lower()] = {
                "summary": doc.summary,
                "description": doc.description,
                "tags": doc.tags,
                "parameters": doc.parameters,
                "responses": doc.responses,
            }
            
            if doc.request_body:
                paths[path_key][method.lower()]["requestBody"] = doc.request_body
            
            if doc.examples:
                paths[path_key][method.lower()]["examples"] = {
                    f"example_{i}": ex for i, ex in enumerate(doc.examples)
                }
        
        self.openapi_spec["paths"] = paths
        
        return self.openapi_spec
    
    def generate_markdown_docs(self) -> str:
        """Generar documentación en Markdown."""
        lines = []
        lines.append("# Bulk Chat API Documentation\n")
        lines.append(f"Generated at: {datetime.now().isoformat()}\n")
        
        # Agrupar por tags
        endpoints_by_tag: Dict[str, List[APIDocumentation]] = {}
        for doc in self.endpoints.values():
            for tag in doc.tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(doc)
        
        # Generar documentación por tag
        for tag, docs in endpoints_by_tag.items():
            lines.append(f"## {tag}\n")
            
            for doc in sorted(docs, key=lambda d: d.endpoint):
                lines.append(f"### {doc.method} {doc.endpoint}\n")
                lines.append(f"**Summary:** {doc.summary}\n")
                lines.append(f"**Description:** {doc.description}\n")
                
                if doc.parameters:
                    lines.append("**Parameters:**\n")
                    for param in doc.parameters:
                        lines.append(f"- `{param.get('name')}` ({param.get('schema', {}).get('type', 'unknown')}): {param.get('description', '')}\n")
                
                if doc.request_body:
                    lines.append("**Request Body:**\n")
                    lines.append(f"```json\n{json.dumps(doc.request_body.get('content', {}).get('application/json', {}).get('schema', {}), indent=2)}\n```\n")
                
                if doc.responses:
                    lines.append("**Responses:**\n")
                    for status_code, response in doc.responses.items():
                        lines.append(f"- `{status_code}`: {response.get('description', '')}\n")
                
                lines.append("\n")
        
        return "\n".join(lines)
    
    def get_endpoint_docs(self, endpoint: str, method: str) -> Optional[APIDocumentation]:
        """Obtener documentación de endpoint."""
        return self.endpoints.get(f"{method}:{endpoint}")
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """Listar todos los endpoints."""
        return [
            {
                "endpoint": doc.endpoint,
                "method": doc.method,
                "summary": doc.summary,
                "tags": doc.tags,
            }
            for doc in self.endpoints.values()
        ]
































