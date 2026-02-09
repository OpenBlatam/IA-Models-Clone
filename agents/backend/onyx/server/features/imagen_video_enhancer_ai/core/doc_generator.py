"""
Documentation Generator
=======================

System for automatic documentation generation.
"""

import inspect
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DocSection:
    """Documentation section."""
    title: str
    content: str
    level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentationGenerator:
    """Automatic documentation generator."""
    
    def __init__(self):
        """Initialize documentation generator."""
        self.sections: List[DocSection] = []
    
    def add_section(self, title: str, content: str, level: int = 1, metadata: Optional[Dict[str, Any]] = None):
        """
        Add documentation section.
        
        Args:
            title: Section title
            content: Section content
            level: Heading level
            metadata: Optional metadata
        """
        section = DocSection(
            title=title,
            content=content,
            level=level,
            metadata=metadata or {}
        )
        self.sections.append(section)
    
    def generate_class_doc(self, cls: type) -> str:
        """
        Generate documentation for a class.
        
        Args:
            cls: Class to document
            
        Returns:
            Documentation string
        """
        doc = f"## {cls.__name__}\n\n"
        
        # Class docstring
        if cls.__doc__:
            doc += f"{cls.__doc__}\n\n"
        
        # Methods
        methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m, None))]
        if methods:
            doc += "### Methods\n\n"
            for method_name in methods:
                method = getattr(cls, method_name)
                if inspect.ismethod(method) or inspect.isfunction(method):
                    sig = inspect.signature(method)
                    doc += f"#### `{method_name}{sig}`\n\n"
                    if method.__doc__:
                        doc += f"{method.__doc__}\n\n"
        
        return doc
    
    def generate_module_doc(self, module) -> str:
        """
        Generate documentation for a module.
        
        Args:
            module: Module to document
            
        Returns:
            Documentation string
        """
        doc = f"# {module.__name__}\n\n"
        
        # Module docstring
        if module.__doc__:
            doc += f"{module.__doc__}\n\n"
        
        # Classes
        classes = [obj for name, obj in inspect.getmembers(module, inspect.isclass)]
        if classes:
            doc += "## Classes\n\n"
            for cls in classes:
                doc += self.generate_class_doc(cls)
                doc += "\n"
        
        # Functions
        functions = [obj for name, obj in inspect.getmembers(module, inspect.isfunction)]
        if functions:
            doc += "## Functions\n\n"
            for func in functions:
                sig = inspect.signature(func)
                doc += f"### `{func.__name__}{sig}`\n\n"
                if func.__doc__:
                    doc += f"{func.__doc__}\n\n"
        
        return doc
    
    def generate_api_doc(self, routes: List[Dict[str, Any]]) -> str:
        """
        Generate API documentation.
        
        Args:
            routes: List of route definitions
            
        Returns:
            Documentation string
        """
        doc = "# API Documentation\n\n"
        doc += f"Generated: {datetime.now().isoformat()}\n\n"
        
        # Group by tags
        by_tag: Dict[str, List[Dict[str, Any]]] = {}
        for route in routes:
            tag = route.get("tags", ["default"])[0]
            if tag not in by_tag:
                by_tag[tag] = []
            by_tag[tag].append(route)
        
        # Generate sections
        for tag, tag_routes in by_tag.items():
            doc += f"## {tag.title()}\n\n"
            
            for route in tag_routes:
                method = route.get("method", "GET")
                path = route.get("path", "")
                summary = route.get("summary", "")
                
                doc += f"### {method} {path}\n\n"
                if summary:
                    doc += f"{summary}\n\n"
                
                # Parameters
                params = route.get("parameters", [])
                if params:
                    doc += "**Parameters:**\n\n"
                    for param in params:
                        doc += f"- `{param.get('name')}` ({param.get('type', 'string')}): {param.get('description', '')}\n"
                    doc += "\n"
                
                # Response
                responses = route.get("responses", {})
                if responses:
                    doc += "**Responses:**\n\n"
                    for status, response in responses.items():
                        doc += f"- `{status}`: {response.get('description', '')}\n"
                    doc += "\n"
        
        return doc
    
    def generate_markdown(self) -> str:
        """
        Generate markdown documentation.
        
        Returns:
            Markdown string
        """
        markdown = ""
        
        for section in self.sections:
            heading = "#" * section.level
            markdown += f"{heading} {section.title}\n\n"
            markdown += f"{section.content}\n\n"
        
        return markdown
    
    def save(self, file_path: Path):
        """
        Save documentation to file.
        
        Args:
            file_path: Output file path
        """
        markdown = self.generate_markdown()
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(markdown)
        
        logger.info(f"Documentation saved to {file_path}")
    
    def clear(self):
        """Clear all sections."""
        self.sections.clear()




