"""
Documentation Generator
=======================

Generador automático de documentación.
"""

import inspect
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """
    Generador de documentación.
    
    Genera documentación automática del código.
    """
    
    def __init__(self):
        """Inicializar generador de documentación."""
        self.modules: Dict[str, Any] = {}
        self.classes: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}
    
    def analyze_module(self, module) -> Dict[str, Any]:
        """
        Analizar módulo.
        
        Args:
            module: Módulo a analizar
            
        Returns:
            Información del módulo
        """
        module_name = module.__name__
        module_doc = inspect.getdoc(module) or ""
        
        info = {
            "name": module_name,
            "docstring": module_doc,
            "classes": [],
            "functions": [],
            "constants": []
        }
        
        # Analizar clases
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_name:
                class_info = self._analyze_class(obj)
                info["classes"].append(class_info)
                self.classes[f"{module_name}.{name}"] = class_info
        
        # Analizar funciones
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module_name:
                func_info = self._analyze_function(obj)
                info["functions"].append(func_info)
                self.functions[f"{module_name}.{name}"] = func_info
        
        self.modules[module_name] = info
        return info
    
    def _analyze_class(self, cls) -> Dict[str, Any]:
        """Analizar clase."""
        class_doc = inspect.getdoc(cls) or ""
        signature = str(inspect.signature(cls.__init__))
        
        methods = []
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('_') or name in ['__init__', '__str__', '__repr__']:
                method_info = self._analyze_function(method)
                methods.append(method_info)
        
        return {
            "name": cls.__name__,
            "docstring": class_doc,
            "signature": signature,
            "methods": methods
        }
    
    def _analyze_function(self, func) -> Dict[str, Any]:
        """Analizar función."""
        func_doc = inspect.getdoc(func) or ""
        signature = str(inspect.signature(func))
        
        return {
            "name": func.__name__,
            "docstring": func_doc,
            "signature": signature
        }
    
    def generate_markdown(
        self,
        output_file: str,
        title: str = "API Documentation"
    ) -> None:
        """
        Generar documentación en Markdown.
        
        Args:
            output_file: Archivo de salida
            title: Título de la documentación
        """
        lines = [f"# {title}\n"]
        
        for module_name, module_info in self.modules.items():
            lines.append(f"## Module: {module_name}\n")
            
            if module_info["docstring"]:
                lines.append(f"{module_info['docstring']}\n")
            
            # Clases
            if module_info["classes"]:
                lines.append("### Classes\n")
                for class_info in module_info["classes"]:
                    lines.append(f"#### {class_info['name']}\n")
                    if class_info["docstring"]:
                        lines.append(f"{class_info['docstring']}\n")
                    lines.append(f"```python\n{class_info['signature']}\n```\n")
                    
                    # Métodos
                    if class_info["methods"]:
                        lines.append("**Methods:**\n")
                        for method in class_info["methods"]:
                            lines.append(f"- `{method['name']}{method['signature']}`\n")
                            if method["docstring"]:
                                lines.append(f"  {method['docstring']}\n")
            
            # Funciones
            if module_info["functions"]:
                lines.append("### Functions\n")
                for func_info in module_info["functions"]:
                    lines.append(f"#### {func_info['name']}\n")
                    if func_info["docstring"]:
                        lines.append(f"{func_info['docstring']}\n")
                    lines.append(f"```python\n{func_info['signature']}\n```\n")
        
        # Escribir archivo
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Documentation generated: {output_file}")
    
    def generate_json(self, output_file: str) -> None:
        """
        Generar documentación en JSON.
        
        Args:
            output_file: Archivo de salida
        """
        data = {
            "modules": self.modules,
            "classes": self.classes,
            "functions": self.functions
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Documentation generated: {output_file}")


# Instancia global
_doc_generator: Optional[DocumentationGenerator] = None


def get_documentation_generator() -> DocumentationGenerator:
    """Obtener instancia global del generador de documentación."""
    global _doc_generator
    if _doc_generator is None:
        _doc_generator = DocumentationGenerator()
    return _doc_generator






