"""
Documentation Generator Module - Auto-generate documentation.

Provides:
- API documentation generation
- Code documentation extraction
- Markdown generation
- OpenAPI/Swagger spec generation
"""

import logging
import inspect
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DocSection:
    """Documentation section."""
    title: str
    content: str
    subsections: List["DocSection"] = field(default_factory=list)


class DocumentationGenerator:
    """Generate documentation from code."""
    
    def __init__(self, output_dir: str = "docs"):
        """
        Initialize documentation generator.
        
        Args:
            output_dir: Output directory for docs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_module_docs(self, module) -> str:
        """
        Generate documentation for a module.
        
        Args:
            module: Python module
            
        Returns:
            Markdown documentation
        """
        doc = f"# {module.__name__}\n\n"
        
        if module.__doc__:
            doc += f"{module.__doc__}\n\n"
        
        # Get classes
        classes = [
            obj for name, obj in inspect.getmembers(module, inspect.isclass)
            if obj.__module__ == module.__name__
        ]
        
        if classes:
            doc += "## Classes\n\n"
            for cls in classes:
                doc += self._format_class(cls)
        
        # Get functions
        functions = [
            obj for name, obj in inspect.getmembers(module, inspect.isfunction)
            if obj.__module__ == module.__name__
        ]
        
        if functions:
            doc += "## Functions\n\n"
            for func in functions:
                doc += self._format_function(func)
        
        return doc
    
    def _format_class(self, cls) -> str:
        """Format class documentation."""
        doc = f"### {cls.__name__}\n\n"
        
        if cls.__doc__:
            doc += f"{cls.__doc__}\n\n"
        
        # Get methods
        methods = [
            name for name, method in inspect.getmembers(cls, inspect.isfunction)
            if not name.startswith('_') or name in ['__init__', '__str__']
        ]
        
        if methods:
            doc += "**Methods:**\n\n"
            for method_name in methods:
                method = getattr(cls, method_name)
                doc += f"- `{method_name}`: {method.__doc__ or 'No description'}\n"
            doc += "\n"
        
        return doc
    
    def _format_function(self, func) -> str:
        """Format function documentation."""
        doc = f"### {func.__name__}\n\n"
        
        if func.__doc__:
            doc += f"{func.__doc__}\n\n"
        
        # Get signature
        sig = inspect.signature(func)
        doc += f"**Signature:** `{func.__name__}{sig}`\n\n"
        
        return doc
    
    def generate_api_docs(self, api_module) -> str:
        """
        Generate API documentation.
        
        Args:
            api_module: API module
            
        Returns:
            API documentation
        """
        doc = "# API Documentation\n\n"
        doc += f"Generated: {datetime.now().isoformat()}\n\n"
        
        # Extract endpoints (if FastAPI)
        if hasattr(api_module, 'app'):
            app = api_module.app
            routes = []
            for route in app.routes:
                if hasattr(route, 'path') and hasattr(route, 'methods'):
                    routes.append({
                        'path': route.path,
                        'methods': list(route.methods),
                    })
            
            doc += "## Endpoints\n\n"
            for route in routes:
                methods_str = ', '.join(route['methods'])
                doc += f"### {methods_str} {route['path']}\n\n"
        
        return doc
    
    def save_docs(self, content: str, filename: str) -> Path:
        """
        Save documentation to file.
        
        Args:
            content: Documentation content
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        filepath = self.output_dir / filename
        filepath.write_text(content, encoding='utf-8')
        logger.info(f"Saved documentation to {filepath}")
        return filepath












