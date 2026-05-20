"""
Documentation Helpers
=====================

Utilities for generating and managing documentation.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import inspect
import json


class DocGenerator:
    """Generator for code documentation."""
    
    @staticmethod
    def extract_function_doc(func: callable) -> Dict[str, Any]:
        """
        Extract documentation from function.
        
        Args:
            func: Function to document
            
        Returns:
            Documentation dictionary
        """
        doc = inspect.getdoc(func)
        sig = inspect.signature(func)
        
        return {
            "name": func.__name__,
            "docstring": doc,
            "parameters": {
                name: {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "required": param.default == inspect.Parameter.empty
                }
                for name, param in sig.parameters.items()
            },
            "return_type": str(sig.return_annotation) if sig.return_annotation != inspect.Parameter.empty else None
        }
    
    @staticmethod
    def extract_class_doc(cls: type) -> Dict[str, Any]:
        """
        Extract documentation from class.
        
        Args:
            cls: Class to document
            
        Returns:
            Documentation dictionary
        """
        doc = inspect.getdoc(cls)
        methods = {}
        
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_'):
                methods[name] = DocGenerator.extract_function_doc(method)
        
        return {
            "name": cls.__name__,
            "docstring": doc,
            "methods": methods
        }
    
    @staticmethod
    def generate_api_docs(routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate API documentation from routes.
        
        Args:
            routes: List of route dictionaries
            
        Returns:
            API documentation dictionary
        """
        return {
            "version": "1.0.0",
            "routes": routes,
            "generated_at": datetime.now().isoformat()
        }
    
    @staticmethod
    def save_docs(docs: Dict[str, Any], output_path: Path):
        """
        Save documentation to file.
        
        Args:
            docs: Documentation dictionary
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(docs, f, indent=2, ensure_ascii=False)


def generate_module_docs(module_path: str) -> Dict[str, Any]:
    """
    Generate documentation for a module.
    
    Args:
        module_path: Path to module
        
    Returns:
        Module documentation
    """
    import importlib
    from pathlib import Path
    
    # Import module
    module = importlib.import_module(module_path)
    
    docs = {
        "module": module_path,
        "docstring": inspect.getdoc(module),
        "classes": {},
        "functions": {}
    }
    
    # Extract classes
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if obj.__module__ == module_path:
            docs["classes"][name] = DocGenerator.extract_class_doc(obj)
    
    # Extract functions
    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module_path:
            docs["functions"][name] = DocGenerator.extract_function_doc(obj)
    
    return docs

