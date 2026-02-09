"""
Documentation Generator for Recovery AI
"""

import inspect
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Generate documentation from code"""
    
    def __init__(self):
        """Initialize documentation generator"""
        logger.info("DocumentationGenerator initialized")
    
    def generate_class_doc(
        self,
        cls: type,
        include_methods: bool = True
    ) -> Dict[str, Any]:
        """
        Generate documentation for class
        
        Args:
            cls: Class to document
            include_methods: Include methods
        
        Returns:
            Documentation dictionary
        """
        doc = {
            "name": cls.__name__,
            "module": cls.__module__,
            "docstring": cls.__doc__ or "",
            "bases": [base.__name__ for base in cls.__bases__]
        }
        
        if include_methods:
            methods = []
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                if not name.startswith("_"):
                    methods.append({
                        "name": name,
                        "signature": str(inspect.signature(method)),
                        "docstring": method.__doc__ or ""
                    })
            doc["methods"] = methods
        
        return doc
    
    def generate_function_doc(
        self,
        func: Any
    ) -> Dict[str, Any]:
        """
        Generate documentation for function
        
        Args:
            func: Function to document
        
        Returns:
            Documentation dictionary
        """
        sig = inspect.signature(func)
        
        return {
            "name": func.__name__,
            "module": func.__module__,
            "signature": str(sig),
            "docstring": func.__doc__ or "",
            "parameters": {
                name: {
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    "default": param.default if param.default != inspect.Parameter.empty else None,
                    "kind": str(param.kind)
                }
                for name, param in sig.parameters.items()
            }
        }
    
    def generate_module_doc(
        self,
        module: Any
    ) -> Dict[str, Any]:
        """
        Generate documentation for module
        
        Args:
            module: Module to document
        
        Returns:
            Documentation dictionary
        """
        doc = {
            "name": module.__name__,
            "docstring": module.__doc__ or "",
            "classes": [],
            "functions": []
        }
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module.__name__:
                doc["classes"].append(self.generate_class_doc(obj))
            elif inspect.isfunction(obj) and obj.__module__ == module.__name__:
                doc["functions"].append(self.generate_function_doc(obj))
        
        return doc
    
    def generate_api_doc(
        self,
        modules: List[Any]
    ) -> Dict[str, Any]:
        """
        Generate API documentation
        
        Args:
            modules: List of modules
        
        Returns:
            API documentation
        """
        api_doc = {
            "modules": []
        }
        
        for module in modules:
            api_doc["modules"].append(self.generate_module_doc(module))
        
        return api_doc

