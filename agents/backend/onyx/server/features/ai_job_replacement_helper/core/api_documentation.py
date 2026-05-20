"""
API Documentation Service - Documentación automática de API
============================================================

Sistema para generar y mantener documentación automática de la API.
"""

import logging
import inspect
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class APIDocumentation:
    """Documentación de API"""
    version: str
    title: str
    description: str
    endpoints: List[Dict[str, Any]]
    schemas: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass
class EndpointDocumentation:
    """Documentación de endpoint"""
    path: str
    method: str
    summary: str
    description: str
    parameters: List[Dict[str, Any]]
    request_body: Optional[Dict[str, Any]] = None
    responses: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


class APIDocumentationService:
    """Servicio de documentación de API"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.documentation: Dict[str, APIDocumentation] = {}
        logger.info("APIDocumentationService initialized")
    
    def generate_documentation(
        self,
        version: str,
        title: str,
        description: str,
        endpoints: List[Dict[str, Any]]
    ) -> APIDocumentation:
        """Generar documentación de API"""
        doc = APIDocumentation(
            version=version,
            title=title,
            description=description,
            endpoints=endpoints,
            schemas=self._extract_schemas(endpoints),
            examples=self._generate_examples(endpoints),
        )
        
        self.documentation[version] = doc
        
        logger.info(f"API documentation generated for version {version}")
        return doc
    
    def _extract_schemas(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extraer schemas de endpoints"""
        schemas = []
        seen_schemas = set()
        
        for endpoint in endpoints:
            if "request_body" in endpoint:
                schema = endpoint["request_body"].get("schema")
                if schema and schema not in seen_schemas:
                    schemas.append(schema)
                    seen_schemas.add(schema)
            
            for response in endpoint.get("responses", []):
                schema = response.get("schema")
                if schema and schema not in seen_schemas:
                    schemas.append(schema)
                    seen_schemas.add(schema)
        
        return schemas
    
    def _generate_examples(self, endpoints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generar ejemplos de uso"""
        examples = []
        
        for endpoint in endpoints[:5]:  # Primeros 5 como ejemplos
            example = {
                "endpoint": endpoint.get("path"),
                "method": endpoint.get("method"),
                "request_example": self._generate_request_example(endpoint),
                "response_example": self._generate_response_example(endpoint),
            }
            examples.append(example)
        
        return examples
    
    def _generate_request_example(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Generar ejemplo de request"""
        example = {}
        
        # Parámetros de path
        if "parameters" in endpoint:
            for param in endpoint["parameters"]:
                if param.get("in") == "path":
                    example[param["name"]] = f"example_{param['name']}"
        
        # Request body
        if "request_body" in endpoint:
            body = endpoint["request_body"]
            example["body"] = body.get("example", {})
        
        return example
    
    def _generate_response_example(self, endpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Generar ejemplo de response"""
        if "responses" in endpoint and endpoint["responses"]:
            first_response = endpoint["responses"][0]
            return first_response.get("example", {})
        return {}
    
    def document_endpoint(
        self,
        path: str,
        method: str,
        summary: str,
        description: str,
        parameters: Optional[List[Dict[str, Any]]] = None,
        request_body: Optional[Dict[str, Any]] = None,
        responses: Optional[List[Dict[str, Any]]] = None
    ) -> EndpointDocumentation:
        """Documentar un endpoint"""
        doc = EndpointDocumentation(
            path=path,
            method=method,
            summary=summary,
            description=description,
            parameters=parameters or [],
            request_body=request_body,
            responses=responses or [],
        )
        
        return doc
    
    def generate_openapi_spec(self, version: str) -> Dict[str, Any]:
        """Generar especificación OpenAPI"""
        doc = self.documentation.get(version)
        if not doc:
            raise ValueError(f"Documentation for version {version} not found")
        
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {
                "title": doc.title,
                "description": doc.description,
                "version": doc.version,
            },
            "paths": {},
        }
        
        # Convertir endpoints a formato OpenAPI
        for endpoint in doc.endpoints:
            path = endpoint.get("path", "")
            method = endpoint.get("method", "get").lower()
            
            if path not in openapi_spec["paths"]:
                openapi_spec["paths"][path] = {}
            
            openapi_spec["paths"][path][method] = {
                "summary": endpoint.get("summary", ""),
                "description": endpoint.get("description", ""),
                "parameters": endpoint.get("parameters", []),
                "responses": {
                    str(resp.get("status_code", 200)): {
                        "description": resp.get("description", ""),
                        "content": {
                            "application/json": {
                                "schema": resp.get("schema", {}),
                                "example": resp.get("example", {}),
                            }
                        }
                    }
                    for resp in endpoint.get("responses", [])
                },
            }
        
        return openapi_spec
    
    def export_documentation(
        self,
        version: str,
        format: str = "json"  # json, markdown, html
    ) -> str:
        """Exportar documentación en diferentes formatos"""
        doc = self.documentation.get(version)
        if not doc:
            raise ValueError(f"Documentation for version {version} not found")
        
        if format == "json":
            import json
            return json.dumps({
                "version": doc.version,
                "title": doc.title,
                "description": doc.description,
                "endpoints": doc.endpoints,
                "generated_at": doc.generated_at.isoformat(),
            }, indent=2)
        
        elif format == "markdown":
            md = f"# {doc.title}\n\n"
            md += f"{doc.description}\n\n"
            md += f"**Version:** {doc.version}\n\n"
            md += "## Endpoints\n\n"
            
            for endpoint in doc.endpoints:
                md += f"### {endpoint.get('method', 'GET').upper()} {endpoint.get('path', '')}\n\n"
                md += f"{endpoint.get('summary', '')}\n\n"
                md += f"{endpoint.get('description', '')}\n\n"
            
            return md
        
        elif format == "html":
            html = f"<html><head><title>{doc.title}</title></head><body>"
            html += f"<h1>{doc.title}</h1>"
            html += f"<p>{doc.description}</p>"
            html += f"<p><strong>Version:</strong> {doc.version}</p>"
            html += "<h2>Endpoints</h2>"
            
            for endpoint in doc.endpoints:
                html += f"<h3>{endpoint.get('method', 'GET').upper()} {endpoint.get('path', '')}</h3>"
                html += f"<p>{endpoint.get('summary', '')}</p>"
                html += f"<p>{endpoint.get('description', '')}</p>"
            
            html += "</body></html>"
            return html
        
        raise ValueError(f"Unsupported format: {format}")




