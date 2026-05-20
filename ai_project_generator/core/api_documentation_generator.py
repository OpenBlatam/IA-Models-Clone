"""
API Documentation Generator - Generador de Documentación de API
==============================================================

Generación avanzada de documentación:
- Auto-generated docs
- Interactive docs
- API examples
- Schema generation
- Documentation export
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentationFormat(str, Enum):
    """Formatos de documentación"""
    OPENAPI = "openapi"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    POSTMAN = "postman"


class APIDocumentationGenerator:
    """
    Generador de documentación de API.
    """
    
    def __init__(self) -> None:
        self.endpoints: List[Dict[str, Any]] = []
        self.schemas: Dict[str, Any] = {}
        self.examples: Dict[str, List[Dict[str, Any]]] = {}
        self.metadata: Dict[str, Any] = {}
    
    def register_endpoint(
        self,
        path: str,
        method: str,
        summary: str,
        description: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        request_body: Optional[Dict[str, Any]] = None,
        responses: Optional[Dict[int, Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None
    ) -> None:
        """Registra endpoint"""
        endpoint = {
            "path": path,
            "method": method.upper(),
            "summary": summary,
            "description": description,
            "parameters": parameters or [],
            "request_body": request_body,
            "responses": responses or {},
            "tags": tags or [],
            "registered_at": datetime.now().isoformat()
        }
        self.endpoints.append(endpoint)
        logger.info(f"Endpoint registered: {method} {path}")
    
    def register_schema(
        self,
        name: str,
        schema: Dict[str, Any]
    ) -> None:
        """Registra schema"""
        self.schemas[name] = schema
        logger.info(f"Schema registered: {name}")
    
    def add_example(
        self,
        endpoint_path: str,
        method: str,
        example: Dict[str, Any],
        example_type: str = "request"
    ) -> None:
        """Agrega ejemplo"""
        key = f"{method.upper()}:{endpoint_path}"
        if key not in self.examples:
            self.examples[key] = []
        
        self.examples[key].append({
            "type": example_type,
            "example": example,
            "added_at": datetime.now().isoformat()
        })
    
    def generate_openapi(
        self,
        title: str = "API Documentation",
        version: str = "1.0.0",
        base_url: str = "http://localhost:8000"
    ) -> Dict[str, Any]:
        """Genera documentación OpenAPI"""
        openapi_doc = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": self.metadata.get("description", ""),
                "contact": self.metadata.get("contact", {}),
                "license": self.metadata.get("license", {})
            },
            "servers": [
                {"url": base_url, "description": "Production server"}
            ],
            "paths": {},
            "components": {
                "schemas": self.schemas
            },
            "tags": self._generate_tags()
        }
        
        # Agregar endpoints
        for endpoint in self.endpoints:
            path = endpoint["path"]
            method = endpoint["method"].lower()
            
            if path not in openapi_doc["paths"]:
                openapi_doc["paths"][path] = {}
            
            openapi_doc["paths"][path][method] = {
                "summary": endpoint["summary"],
                "description": endpoint.get("description"),
                "tags": endpoint.get("tags", []),
                "parameters": endpoint.get("parameters", []),
                "requestBody": endpoint.get("request_body"),
                "responses": endpoint.get("responses", {})
            }
        
        return openapi_doc
    
    def generate_markdown(self) -> str:
        """Genera documentación en Markdown"""
        md = f"# API Documentation\n\n"
        md += f"Generated at: {datetime.now().isoformat()}\n\n"
        
        # Agrupar por tags
        endpoints_by_tag: Dict[str, List[Dict[str, Any]]] = {}
        for endpoint in self.endpoints:
            tags = endpoint.get("tags", ["default"])
            for tag in tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(endpoint)
        
        # Generar documentación por tag
        for tag, endpoints in endpoints_by_tag.items():
            md += f"## {tag.title()}\n\n"
            
            for endpoint in endpoints:
                md += f"### {endpoint['method']} {endpoint['path']}\n\n"
                md += f"**Summary:** {endpoint['summary']}\n\n"
                
                if endpoint.get("description"):
                    md += f"{endpoint['description']}\n\n"
                
                # Parámetros
                if endpoint.get("parameters"):
                    md += "**Parameters:**\n\n"
                    for param in endpoint["parameters"]:
                        md += f"- `{param.get('name')}` ({param.get('type', 'string')}): {param.get('description', '')}\n"
                    md += "\n"
                
                # Ejemplos
                key = f"{endpoint['method']}:{endpoint['path']}"
                if key in self.examples:
                    md += "**Examples:**\n\n"
                    for ex in self.examples[key]:
                        md += f"```json\n{ex['example']}\n```\n\n"
                
                md += "---\n\n"
        
        return md
    
    def _generate_tags(self) -> List[Dict[str, str]]:
        """Genera tags únicos"""
        all_tags = set()
        for endpoint in self.endpoints:
            all_tags.update(endpoint.get("tags", []))
        
        return [{"name": tag} for tag in sorted(all_tags)]
    
    def export_documentation(
        self,
        format: DocumentationFormat = DocumentationFormat.OPENAPI,
        output_file: Optional[str] = None
    ) -> str:
        """Exporta documentación"""
        if format == DocumentationFormat.OPENAPI:
            content = self.generate_openapi()
            if output_file:
                import json
                with open(output_file, "w") as f:
                    json.dump(content, f, indent=2)
            return str(content)
        elif format == DocumentationFormat.MARKDOWN:
            content = self.generate_markdown()
            if output_file:
                with open(output_file, "w") as f:
                    f.write(content)
            return content
        else:
            raise ValueError(f"Unsupported format: {format}")


def get_api_documentation_generator() -> APIDocumentationGenerator:
    """Obtiene generador de documentación"""
    return APIDocumentationGenerator()















