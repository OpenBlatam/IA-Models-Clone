"""
API Documentation - Documentación Automática de API
===================================================

Sistema para generar y mantener documentación automática de la API.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APIDocumentation:
    """Documentación de endpoint"""
    path: str
    method: str
    summary: str
    description: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    request_body: Optional[Dict[str, Any]] = None
    responses: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    deprecated: bool = False
    examples: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIDocumentationGenerator:
    """
    Generador de documentación de API.
    
    Recopila y genera documentación automática de endpoints.
    """
    
    def __init__(self):
        self.endpoints: Dict[str, APIDocumentation] = {}
        self.base_info: Dict[str, Any] = {
            "title": "Cursor Agent API",
            "version": "1.0.0",
            "description": "API REST para el agente Cursor 24/7"
        }
    
    def register_endpoint(
        self,
        path: str,
        method: str,
        summary: str,
        description: Optional[str] = None,
        parameters: Optional[List[Dict[str, Any]]] = None,
        request_body: Optional[Dict[str, Any]] = None,
        responses: Optional[Dict[int, Dict[str, Any]]] = None,
        tags: Optional[List[str]] = None,
        deprecated: bool = False,
        examples: Optional[List[Dict[str, Any]]] = None,
        **metadata
    ) -> None:
        """
        Registrar endpoint para documentación.
        
        Args:
            path: Path del endpoint
            method: Método HTTP
            summary: Resumen del endpoint
            description: Descripción detallada
            parameters: Lista de parámetros
            request_body: Schema del body
            responses: Diccionario de respuestas por código
            tags: Tags del endpoint
            deprecated: Si está deprecado
            examples: Ejemplos de uso
            **metadata: Metadata adicional
        """
        key = f"{method.upper()}:{path}"
        
        doc = APIDocumentation(
            path=path,
            method=method.upper(),
            summary=summary,
            description=description,
            parameters=parameters or [],
            request_body=request_body,
            responses=responses or {},
            tags=tags or [],
            deprecated=deprecated,
            examples=examples or [],
            metadata=metadata
        )
        
        self.endpoints[key] = doc
        logger.debug(f"📚 Endpoint documented: {method} {path}")
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """
        Generar especificación OpenAPI.
        
        Returns:
            Diccionario con especificación OpenAPI
        """
        spec = {
            "openapi": "3.0.0",
            "info": self.base_info,
            "servers": [
                {
                    "url": "http://localhost:8024",
                    "description": "Servidor de desarrollo"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {}
            }
        }
        
        # Agrupar endpoints por path
        paths: Dict[str, Dict[str, Any]] = {}
        
        for key, doc in self.endpoints.items():
            method, path = key.split(":", 1)
            
            if path not in paths:
                paths[path] = {}
            
            path_item = {
                "summary": doc.summary,
                "description": doc.description or doc.summary,
                "operationId": f"{method.lower()}_{path.replace('/', '_').replace('{', '').replace('}', '')}",
                "tags": doc.tags,
                "deprecated": doc.deprecated
            }
            
            # Parámetros
            if doc.parameters:
                path_item["parameters"] = doc.parameters
            
            # Request body
            if doc.request_body:
                path_item["requestBody"] = {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": doc.request_body
                        }
                    }
                }
            
            # Responses
            path_item["responses"] = {}
            for status_code, response in doc.responses.items():
                path_item["responses"][str(status_code)] = response
            
            # Ejemplos
            if doc.examples:
                path_item["examples"] = doc.examples
            
            paths[path][method.lower()] = path_item
        
        spec["paths"] = paths
        
        return spec
    
    def generate_markdown_docs(self) -> str:
        """
        Generar documentación en Markdown.
        
        Returns:
            String con documentación Markdown
        """
        lines = [
            f"# {self.base_info['title']}",
            "",
            f"**Versión:** {self.base_info['version']}",
            "",
            self.base_info.get('description', ''),
            "",
            "## Endpoints",
            ""
        ]
        
        # Agrupar por tags
        endpoints_by_tag: Dict[str, List[APIDocumentation]] = {}
        for doc in self.endpoints.values():
            for tag in doc.tags:
                if tag not in endpoints_by_tag:
                    endpoints_by_tag[tag] = []
                endpoints_by_tag[tag].append(doc)
        
        # Sin tags
        untagged = [doc for doc in self.endpoints.values() if not doc.tags]
        if untagged:
            endpoints_by_tag["General"] = untagged
        
        # Generar documentación por tag
        for tag, docs in endpoints_by_tag.items():
            lines.append(f"### {tag}")
            lines.append("")
            
            for doc in sorted(docs, key=lambda x: (x.method, x.path)):
                lines.append(f"#### {doc.method} {doc.path}")
                if doc.deprecated:
                    lines.append("⚠️ **Deprecated**")
                lines.append("")
                lines.append(f"**Resumen:** {doc.summary}")
                lines.append("")
                
                if doc.description:
                    lines.append(f"**Descripción:** {doc.description}")
                    lines.append("")
                
                if doc.parameters:
                    lines.append("**Parámetros:**")
                    for param in doc.parameters:
                        param_name = param.get("name", "")
                        param_type = param.get("schema", {}).get("type", "string")
                        param_desc = param.get("description", "")
                        required = " (requerido)" if param.get("required", False) else " (opcional)"
                        lines.append(f"- `{param_name}` ({param_type}){required}: {param_desc}")
                    lines.append("")
                
                if doc.request_body:
                    lines.append("**Request Body:**")
                    lines.append("```json")
                    # Simplificar schema para mostrar
                    lines.append(str(doc.request_body))
                    lines.append("```")
                    lines.append("")
                
                if doc.responses:
                    lines.append("**Respuestas:**")
                    for status_code, response in doc.responses.items():
                        description = response.get("description", "")
                        lines.append(f"- `{status_code}`: {description}")
                    lines.append("")
                
                if doc.examples:
                    lines.append("**Ejemplos:**")
                    for example in doc.examples:
                        lines.append(f"```json")
                        lines.append(str(example))
                        lines.append("```")
                    lines.append("")
                
                lines.append("---")
                lines.append("")
        
        return "\n".join(lines)
    
    def save_docs(self, output_dir: str, formats: List[str] = ["markdown", "openapi"]) -> None:
        """
        Guardar documentación en archivos.
        
        Args:
            output_dir: Directorio de salida
            formats: Formatos a generar (markdown, openapi)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if "markdown" in formats:
            md_content = self.generate_markdown_docs()
            md_file = output_path / "API_DOCUMENTATION.md"
            md_file.write_text(md_content, encoding="utf-8")
            logger.info(f"📚 Markdown documentation saved to {md_file}")
        
        if "openapi" in formats:
            import json
            openapi_spec = self.generate_openapi_spec()
            openapi_file = output_path / "openapi.json"
            openapi_file.write_text(
                json.dumps(openapi_spec, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            logger.info(f"📚 OpenAPI specification saved to {openapi_file}")




