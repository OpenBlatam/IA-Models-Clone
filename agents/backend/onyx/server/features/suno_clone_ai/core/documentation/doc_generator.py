"""
Documentation Generator

Utilities for generating documentation.
"""

import logging
import inspect
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """Generate documentation from code."""
    
    def __init__(self):
        """Initialize documentation generator."""
        pass
    
    def generate_function_doc(
        self,
        func: callable
    ) -> Dict[str, Any]:
        """
        Generate documentation for function.
        
        Args:
            func: Function to document
            
        Returns:
            Function documentation
        """
        doc = {
            'name': func.__name__,
            'docstring': inspect.getdoc(func) or '',
            'signature': str(inspect.signature(func)),
            'parameters': []
        }
        
        # Extract parameters
        sig = inspect.signature(func)
        for param_name, param in sig.parameters.items():
            param_info = {
                'name': param_name,
                'type': str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                'default': param.default if param.default != inspect.Parameter.empty else None,
                'required': param.default == inspect.Parameter.empty
            }
            doc['parameters'].append(param_info)
        
        return doc
    
    def generate_class_doc(
        self,
        cls: type
    ) -> Dict[str, Any]:
        """
        Generate documentation for class.
        
        Args:
            cls: Class to document
            
        Returns:
            Class documentation
        """
        doc = {
            'name': cls.__name__,
            'docstring': inspect.getdoc(cls) or '',
            'methods': [],
            'attributes': []
        }
        
        # Extract methods
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if not name.startswith('_'):
                method_doc = self.generate_function_doc(method)
                doc['methods'].append(method_doc)
        
        return doc
    
    def generate_module_doc(
        self,
        module: Any
    ) -> Dict[str, Any]:
        """
        Generate documentation for module.
        
        Args:
            module: Module to document
            
        Returns:
            Module documentation
        """
        doc = {
            'name': module.__name__,
            'docstring': inspect.getdoc(module) or '',
            'functions': [],
            'classes': []
        }
        
        # Extract functions and classes
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                doc['functions'].append(self.generate_function_doc(obj))
            elif inspect.isclass(obj) and not name.startswith('_'):
                doc['classes'].append(self.generate_class_doc(obj))
        
        return doc


def generate_api_docs(
    module: Any,
    output_path: str = "./docs/api.md"
) -> str:
    """
    Generate API documentation.
    
    Args:
        module: Module to document
        output_path: Output file path
        
    Returns:
        Output path
    """
    generator = DocumentationGenerator()
    doc = generator.generate_module_doc(module)
    
    # Format as Markdown
    markdown = f"# {doc['name']}\n\n"
    markdown += f"{doc['docstring']}\n\n"
    
    if doc['classes']:
        markdown += "## Classes\n\n"
        for cls in doc['classes']:
            markdown += f"### {cls['name']}\n\n"
            markdown += f"{cls['docstring']}\n\n"
    
    if doc['functions']:
        markdown += "## Functions\n\n"
        for func in doc['functions']:
            markdown += f"### {func['name']}\n\n"
            markdown += f"{func['docstring']}\n\n"
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(markdown)
    
    logger.info(f"Generated API docs: {output_path}")
    
    return output_path


def generate_model_docs(
    model_class: type,
    output_path: str = "./docs/model.md"
) -> str:
    """
    Generate model documentation.
    
    Args:
        model_class: Model class to document
        output_path: Output file path
        
    Returns:
        Output path
    """
    generator = DocumentationGenerator()
    doc = generator.generate_class_doc(model_class)
    
    # Format as Markdown
    markdown = f"# {doc['name']}\n\n"
    markdown += f"{doc['docstring']}\n\n"
    
    if doc['methods']:
        markdown += "## Methods\n\n"
        for method in doc['methods']:
            markdown += f"### {method['name']}\n\n"
            markdown += f"{method['docstring']}\n\n"
            markdown += f"**Signature:** `{method['signature']}`\n\n"
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(markdown)
    
    logger.info(f"Generated model docs: {output_path}")
    
    return output_path



