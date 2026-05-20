"""
API Documentation Generator
Generate API documentation automatically
"""

from pathlib import Path
from typing import Dict, Any, List
import inspect
import logging

logger = logging.getLogger(__name__)


class APIDocGenerator:
    """
    Generate API documentation
    """
    
    def __init__(self, output_dir: Path = Path("docs/api")):
        """
        Initialize API doc generator
        
        Args:
            output_dir: Output directory for documentation
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_module_docs(self, module, module_name: str) -> str:
        """
        Generate documentation for a module
        
        Args:
            module: Module object
            module_name: Module name
            
        Returns:
            Documentation string
        """
        doc_lines = [f"# {module_name}", ""]
        
        if module.__doc__:
            doc_lines.append(module.__doc__)
            doc_lines.append("")
        
        # Get classes
        classes = inspect.getmembers(module, inspect.isclass)
        if classes:
            doc_lines.append("## Classes")
            doc_lines.append("")
            for name, cls in classes:
                if cls.__module__ == module.__name__:
                    doc_lines.append(f"### {name}")
                    doc_lines.append("")
                    if cls.__doc__:
                        doc_lines.append(cls.__doc__)
                        doc_lines.append("")
                    
                    # Get methods
                    methods = inspect.getmembers(cls, inspect.isfunction)
                    if methods:
                        doc_lines.append("#### Methods")
                        doc_lines.append("")
                        for method_name, method in methods:
                            doc_lines.append(f"- `{method_name}()`")
                            if method.__doc__:
                                doc_lines.append(f"  {method.__doc__.split(chr(10))[0]}")
                        doc_lines.append("")
        
        # Get functions
        functions = inspect.getmembers(module, inspect.isfunction)
        if functions:
            doc_lines.append("## Functions")
            doc_lines.append("")
            for name, func in functions:
                if func.__module__ == module.__name__:
                    doc_lines.append(f"### {name}")
                    doc_lines.append("")
                    if func.__doc__:
                        doc_lines.append(func.__doc__)
                        doc_lines.append("")
        
        return "\n".join(doc_lines)
    
    def generate_all_docs(self, modules: Dict[str, Any]) -> None:
        """
        Generate documentation for all modules
        
        Args:
            modules: Dictionary of module_name -> module_object
        """
        for module_name, module in modules.items():
            doc_content = self.generate_module_docs(module, module_name)
            output_file = self.output_dir / f"{module_name}.md"
            output_file.write_text(doc_content)
            logger.info(f"Generated docs for {module_name}")



