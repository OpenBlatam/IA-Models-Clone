"""
Document Documentation - Generación Automática de Documentación
=============================================================

Generación automática de documentación del sistema.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import inspect

logger = logging.getLogger(__name__)


@dataclass
class APIDocumentation:
    """Documentación de API."""
    endpoint: str
    method: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    responses: List[Dict[str, Any]] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)


@dataclass
class MethodDocumentation:
    """Documentación de método."""
    method_name: str
    class_name: str
    description: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    return_type: Optional[str] = None
    examples: List[str] = field(default_factory=list)


class DocumentationGenerator:
    """Generador de documentación."""
    
    def __init__(self, analyzer):
        """Inicializar generador."""
        self.analyzer = analyzer
    
    def generate_class_documentation(self, class_obj: type) -> Dict[str, Any]:
        """Generar documentación de clase."""
        doc = {
            "class_name": class_obj.__name__,
            "docstring": inspect.getdoc(class_obj) or "",
            "methods": []
        }
        
        # Documentar métodos públicos
        for name, method in inspect.getmembers(class_obj, inspect.isfunction):
            if not name.startswith('_'):
                method_doc = self._document_method(method, class_obj.__name__)
                if method_doc:
                    doc["methods"].append(method_doc)
        
        return doc
    
    def _document_method(self, method: callable, class_name: str) -> Optional[MethodDocumentation]:
        """Documentar método."""
        signature = inspect.signature(method)
        
        parameters = []
        for param_name, param in signature.parameters.items():
            parameters.append({
                "name": param_name,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                "required": param.default == inspect.Parameter.empty
            })
        
        return_type = None
        if signature.return_annotation != inspect.Signature.empty:
            return_type = str(signature.return_annotation)
        
        return MethodDocumentation(
            method_name=method.__name__,
            class_name=class_name,
            description=inspect.getdoc(method) or "",
            parameters=parameters,
            return_type=return_type
        )
    
    def generate_api_documentation(self) -> List[APIDocumentation]:
        """Generar documentación de API."""
        docs = []
        
        # Si hay API server, obtener endpoints
        if hasattr(self.analyzer, '_api_server') and self.analyzer._api_server:
            app = self.analyzer._api_server.app
            
            for route in app.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    methods = list(route.methods)
                    if 'HEAD' in methods:
                        methods.remove('HEAD')
                    if 'OPTIONS' in methods:
                        methods.remove('OPTIONS')
                    
                    for method in methods:
                        endpoint_doc = APIDocumentation(
                            endpoint=route.path,
                            method=method,
                            description=getattr(route, 'summary', '') or getattr(route, 'description', ''),
                            parameters=[],
                            responses=[]
                        )
                        docs.append(endpoint_doc)
        
        return docs
    
    def generate_complete_documentation(self) -> Dict[str, Any]:
        """Generar documentación completa del sistema."""
        documentation = {
            "generated_at": datetime.now().isoformat(),
            "system_name": "Document Analyzer",
            "version": "1.0.0",
            "classes": [],
            "api_endpoints": [],
            "features": []
        }
        
        # Documentar clase principal
        if hasattr(self.analyzer, '__class__'):
            class_doc = self.generate_class_documentation(self.analyzer.__class__)
            documentation["classes"].append(class_doc)
        
        # Documentar API
        api_docs = self.generate_api_documentation()
        documentation["api_endpoints"] = [
            {
                "endpoint": doc.endpoint,
                "method": doc.method,
                "description": doc.description
            }
            for doc in api_docs
        ]
        
        # Listar características
        features = []
        
        # Detectar características disponibles
        if hasattr(self.analyzer, 'comparator') and self.analyzer.comparator:
            features.append("Comparación de Documentos")
        
        if hasattr(self.analyzer, 'batch_processor') and self.analyzer.batch_processor:
            features.append("Procesamiento en Batch")
        
        if hasattr(self.analyzer, 'version_manager') and self.analyzer.version_manager:
            features.append("Gestión de Versiones")
        
        if hasattr(self.analyzer, 'grammar_analyzer') and self.analyzer.grammar_analyzer:
            features.append("Análisis Gramatical")
        
        if hasattr(self.analyzer, 'semantic_search') and self.analyzer.semantic_search:
            features.append("Búsqueda Semántica")
        
        if hasattr(self.analyzer, 'cloud_storage') and self.analyzer.cloud_storage:
            features.append("Almacenamiento Cloud")
        
        if hasattr(self.analyzer, 'tagging_system') and self.analyzer.tagging_system:
            features.append("Sistema de Etiquetado")
        
        documentation["features"] = features
        
        return documentation
    
    def generate_markdown_documentation(self, output_path: str) -> str:
        """Generar documentación en Markdown."""
        doc_data = self.generate_complete_documentation()
        
        md = f"""# Document Analyzer - Documentación Completa

**Generado**: {doc_data['generated_at']}  
**Versión**: {doc_data['version']}

## Características Disponibles

"""
        
        for feature in doc_data['features']:
            md += f"- {feature}\n"
        
        md += "\n## API Endpoints\n\n"
        
        for endpoint in doc_data['api_endpoints']:
            md += f"### `{endpoint['method']} {endpoint['endpoint']}`\n\n"
            md += f"{endpoint['description']}\n\n"
        
        md += "\n## Métodos de Clase\n\n"
        
        for class_doc in doc_data['classes']:
            md += f"### {class_doc['class_name']}\n\n"
            md += f"{class_doc['docstring']}\n\n"
            
            for method in class_doc['methods']:
                md += f"#### `{method.method_name}`\n\n"
                md += f"{method.description}\n\n"
                
                if method.parameters:
                    md += "**Parámetros:**\n\n"
                    for param in method.parameters:
                        md += f"- `{param['name']}` ({param['type']}): "
                        if not param['required']:
                            md += f"Opcional. Default: {param['default']}\n"
                        else:
                            md += "Requerido\n"
                    md += "\n"
        
        output_file = output_path if output_path.endswith('.md') else f"{output_path}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        return output_file


__all__ = [
    "DocumentationGenerator",
    "APIDocumentation",
    "MethodDocumentation"
]


