"""
Auto Documentation - Sistema de documentación automática mejorada
===================================================================
"""

import logging
import inspect
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoDocumentation:
    """Sistema de documentación automática"""
    
    def __init__(self):
        self.api_docs: Dict[str, Dict[str, Any]] = {}
        self.code_docs: Dict[str, str] = {}
        self.examples: Dict[str, List[Dict[str, Any]]] = {}
    
    def document_endpoint(self, endpoint: str, method: str, func: callable,
                        description: Optional[str] = None):
        """Documenta un endpoint automáticamente"""
        # Extraer información de la función
        sig = inspect.signature(func)
        docstring = inspect.getdoc(func) or description or ""
        
        parameters = []
        for param_name, param in sig.parameters.items():
            param_info = {
                "name": param_name,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "required": param.default == inspect.Parameter.empty,
                "default": param.default if param.default != inspect.Parameter.empty else None
            }
            parameters.append(param_info)
        
        doc_key = f"{method.upper()}:{endpoint}"
        
        self.api_docs[doc_key] = {
            "endpoint": endpoint,
            "method": method,
            "description": docstring,
            "parameters": parameters,
            "function_name": func.__name__,
            "documented_at": datetime.now().isoformat()
        }
        
        logger.info(f"Endpoint documentado automáticamente: {doc_key}")
    
    def add_example(self, endpoint: str, method: str, example: Dict[str, Any]):
        """Agrega un ejemplo a un endpoint"""
        doc_key = f"{method.upper()}:{endpoint}"
        
        if doc_key not in self.examples:
            self.examples[doc_key] = []
        
        self.examples[doc_key].append(example)
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Genera especificación OpenAPI automáticamente"""
        paths = {}
        
        for doc_key, doc in self.api_docs.items():
            method = doc["method"].lower()
            endpoint = doc["endpoint"]
            
            if endpoint not in paths:
                paths[endpoint] = {}
            
            paths[endpoint][method] = {
                "summary": doc["description"].split("\n")[0] if doc["description"] else "",
                "description": doc["description"],
                "parameters": [
                    {
                        "name": p["name"],
                        "in": "query" if not p["required"] else "path",
                        "required": p["required"],
                        "schema": {"type": p["type"].lower() if p["type"] != "Any" else "string"}
                    }
                    for p in doc["parameters"]
                ],
                "responses": {
                    "200": {
                        "description": "Success",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object"}
                            }
                        }
                    }
                }
            }
            
            # Agregar ejemplos
            if doc_key in self.examples:
                paths[endpoint][method]["requestBody"] = {
                    "content": {
                        "application/json": {
                            "examples": {
                                f"example_{i}": {"value": ex}
                                for i, ex in enumerate(self.examples[doc_key])
                            }
                        }
                    }
                }
        
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "3D Prototype AI API",
                "version": "2.0.0",
                "description": "API completa para generación de prototipos 3D"
            },
            "paths": paths
        }
    
    def generate_markdown_docs(self) -> str:
        """Genera documentación en Markdown"""
        md = "# API Documentation\n\n"
        md += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Agrupar por método
        by_method = {}
        for doc in self.api_docs.values():
            method = doc["method"]
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(doc)
        
        for method, docs in by_method.items():
            md += f"## {method.upper()} Endpoints\n\n"
            
            for doc in docs:
                md += f"### `{doc['endpoint']}`\n\n"
                md += f"{doc['description']}\n\n"
                
                if doc.get("parameters"):
                    md += "**Parameters:**\n\n"
                    for param in doc["parameters"]:
                        req = "required" if param["required"] else "optional"
                        md += f"- `{param['name']}` ({param['type']}, {req}): {param.get('description', '')}\n"
                    md += "\n"
                
                if doc_key := f"{method.upper()}:{doc['endpoint']}" in self.examples:
                    md += "**Examples:**\n\n"
                    for example in self.examples[doc_key]:
                        md += f"```json\n{example}\n```\n\n"
        
        return md




