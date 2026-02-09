"""
Auto Documentation System for Flux2 Clothing Changer
=====================================================

Automatic documentation generation and management.
"""

import inspect
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FunctionDoc:
    """Function documentation."""
    name: str
    module: str
    signature: str
    docstring: str
    parameters: Dict[str, Any]
    return_type: Optional[str] = None
    examples: List[str] = None
    
    def __post_init__(self):
        if self.examples is None:
            self.examples = []


@dataclass
class ClassDoc:
    """Class documentation."""
    name: str
    module: str
    docstring: str
    methods: List[FunctionDoc]
    attributes: Dict[str, Any]
    base_classes: List[str] = None
    
    def __post_init__(self):
        if self.base_classes is None:
            self.base_classes = []


class AutoDocumentation:
    """Automatic documentation generation system."""
    
    def __init__(
        self,
        output_dir: Path = Path("docs/auto"),
        format: str = "markdown",
    ):
        """
        Initialize auto documentation system.
        
        Args:
            output_dir: Output directory for documentation
            format: Documentation format (markdown, json, html)
        """
        self.output_dir = output_dir
        self.format = format
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def document_module(
        self,
        module,
        module_name: str,
    ) -> Dict[str, Any]:
        """
        Document a module.
        
        Args:
            module: Module object
            module_name: Module name
            
        Returns:
            Documentation dictionary
        """
        docs = {
            "module_name": module_name,
            "timestamp": datetime.now().isoformat(),
            "classes": [],
            "functions": [],
        }
        
        # Document classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module_name:
                class_doc = self._document_class(obj, module_name)
                docs["classes"].append(class_doc)
        
        # Document functions
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module_name:
                func_doc = self._document_function(obj, module_name)
                docs["functions"].append(func_doc)
        
        return docs
    
    def _document_class(
        self,
        cls,
        module_name: str,
    ) -> ClassDoc:
        """Document a class."""
        # Get methods
        methods = []
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith("_"):
                method_doc = self._document_function(method, module_name)
                methods.append(method_doc)
        
        # Get attributes
        attributes = {}
        for name, value in inspect.getmembers(cls):
            if not name.startswith("_") and not inspect.ismethod(value) and not inspect.isfunction(value):
                attributes[name] = str(type(value).__name__)
        
        # Get base classes
        base_classes = [base.__name__ for base in cls.__bases__ if base != object]
        
        return ClassDoc(
            name=cls.__name__,
            module=module_name,
            docstring=cls.__doc__ or "",
            methods=methods,
            attributes=attributes,
            base_classes=base_classes,
        )
    
    def _document_function(
        self,
        func,
        module_name: str,
    ) -> FunctionDoc:
        """Document a function."""
        sig = inspect.signature(func)
        
        # Get parameters
        parameters = {}
        for param_name, param in sig.parameters.items():
            parameters[param_name] = {
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "Any",
                "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                "kind": str(param.kind),
            }
        
        # Get return type
        return_type = None
        if sig.return_annotation != inspect.Signature.empty:
            return_type = str(sig.return_annotation)
        
        return FunctionDoc(
            name=func.__name__,
            module=module_name,
            signature=str(sig),
            docstring=func.__doc__ or "",
            parameters=parameters,
            return_type=return_type,
        )
    
    def generate_markdown(
        self,
        docs: Dict[str, Any],
        file_path: Path,
    ) -> None:
        """Generate markdown documentation."""
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {docs['module_name']}\n\n")
            f.write(f"*Generated: {docs['timestamp']}*\n\n")
            
            # Classes
            if docs["classes"]:
                f.write("## Classes\n\n")
                for class_doc in docs["classes"]:
                    f.write(f"### {class_doc.name}\n\n")
                    if class_doc.docstring:
                        f.write(f"{class_doc.docstring}\n\n")
                    
                    if class_doc.base_classes:
                        f.write(f"**Base classes:** {', '.join(class_doc.base_classes)}\n\n")
                    
                    if class_doc.methods:
                        f.write("#### Methods\n\n")
                        for method in class_doc.methods:
                            f.write(f"##### {method.name}\n\n")
                            f.write(f"```python\n{method.signature}\n```\n\n")
                            if method.docstring:
                                f.write(f"{method.docstring}\n\n")
            
            # Functions
            if docs["functions"]:
                f.write("## Functions\n\n")
                for func_doc in docs["functions"]:
                    f.write(f"### {func_doc.name}\n\n")
                    f.write(f"```python\n{func_doc.signature}\n```\n\n")
                    if func_doc.docstring:
                        f.write(f"{func_doc.docstring}\n\n")
    
    def generate_json(
        self,
        docs: Dict[str, Any],
        file_path: Path,
    ) -> None:
        """Generate JSON documentation."""
        # Convert dataclasses to dicts
        json_docs = {
            "module_name": docs["module_name"],
            "timestamp": docs["timestamp"],
            "classes": [asdict(c) for c in docs["classes"]],
            "functions": [asdict(f) for f in docs["functions"]],
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(json_docs, f, indent=2)
    
    def document_and_save(
        self,
        module,
        module_name: str,
    ) -> Path:
        """
        Document module and save to file.
        
        Args:
            module: Module object
            module_name: Module name
            
        Returns:
            Path to saved documentation
        """
        docs = self.document_module(module, module_name)
        
        if self.format == "markdown":
            file_path = self.output_dir / f"{module_name}.md"
            self.generate_markdown(docs, file_path)
        elif self.format == "json":
            file_path = self.output_dir / f"{module_name}.json"
            self.generate_json(docs, file_path)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
        
        logger.info(f"Documentation generated: {file_path}")
        return file_path


