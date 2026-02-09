"""
Advanced Documentation Generator
=================================

Advanced utilities for generating documentation automatically.
"""

import ast
import inspect
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import re


@dataclass
class DocInfo:
    """Documentation information."""
    name: str
    docstring: str
    type: str  # function, class, method, module
    signature: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    returns: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedDocGenerator:
    """Advanced documentation generator."""
    
    @staticmethod
    def extract_docstring(obj: Any) -> Optional[str]:
        """
        Extract docstring from object.
        
        Args:
            obj: Object to extract docstring from
            
        Returns:
            Docstring or None
        """
        return inspect.getdoc(obj)
    
    @staticmethod
    def extract_signature(obj: Any) -> str:
        """
        Extract signature from callable.
        
        Args:
            obj: Callable object
            
        Returns:
            Signature string
        """
        try:
            sig = inspect.signature(obj)
            return str(sig)
        except Exception:
            return ""
    
    @staticmethod
    def extract_parameters(obj: Any) -> List[Dict[str, Any]]:
        """
        Extract parameters from callable.
        
        Args:
            obj: Callable object
            
        Returns:
            List of parameter information
        """
        params = []
        try:
            sig = inspect.signature(obj)
            for name, param in sig.parameters.items():
                param_info = {
                    "name": name,
                    "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else None,
                    "default": str(param.default) if param.default != inspect.Parameter.empty else None,
                    "required": param.default == inspect.Parameter.empty
                }
                params.append(param_info)
        except Exception:
            pass
        return params
    
    @staticmethod
    def extract_return_type(obj: Any) -> Optional[str]:
        """
        Extract return type from callable.
        
        Args:
            obj: Callable object
            
        Returns:
            Return type string or None
        """
        try:
            sig = inspect.signature(obj)
            if sig.return_annotation != inspect.Signature.empty:
                return str(sig.return_annotation)
        except Exception:
            pass
        return None
    
    @staticmethod
    def parse_docstring(docstring: str) -> Dict[str, Any]:
        """
        Parse docstring into structured format.
        
        Args:
            docstring: Docstring text
            
        Returns:
            Parsed docstring information
        """
        if not docstring:
            return {}
        
        result = {
            "summary": "",
            "description": "",
            "args": [],
            "returns": None,
            "raises": [],
            "examples": []
        }
        
        lines = docstring.strip().split('\n')
        current_section = "summary"
        current_text = []
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Args:"):
                current_section = "args"
                if current_text:
                    result["summary"] = "\n".join(current_text)
                current_text = []
            elif line.startswith("Returns:"):
                current_section = "returns"
                if current_text:
                    result["description"] = "\n".join(current_text)
                current_text = []
            elif line.startswith("Raises:"):
                current_section = "raises"
                current_text = []
            elif line.startswith("Example") or line.startswith("Examples:"):
                current_section = "examples"
                current_text = []
            elif line and not line.startswith(":"):
                if current_section == "summary" and not result["summary"]:
                    current_text.append(line)
                elif current_section == "args":
                    # Parse argument
                    match = re.match(r'(\w+):\s*(.+)', line)
                    if match:
                        result["args"].append({
                            "name": match.group(1),
                            "description": match.group(2)
                        })
                elif current_section == "returns":
                    result["returns"] = line
                elif current_section == "raises":
                    match = re.match(r'(\w+):\s*(.+)', line)
                    if match:
                        result["raises"].append({
                            "exception": match.group(1),
                            "description": match.group(2)
                        })
                elif current_section == "examples":
                    result["examples"].append(line)
                else:
                    current_text.append(line)
        
        if current_text and not result["summary"]:
            result["summary"] = "\n".join(current_text)
        
        return result
    
    def generate_module_docs(self, module_path: Path) -> List[DocInfo]:
        """
        Generate documentation for module.
        
        Args:
            module_path: Path to Python module
            
        Returns:
            List of documentation information
        """
        docs = []
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    doc_info = DocInfo(
                        name=node.name,
                        docstring=ast.get_docstring(node) or "",
                        type="function",
                        signature=self._extract_ast_signature(node)
                    )
                    docs.append(doc_info)
                elif isinstance(node, ast.ClassDef):
                    doc_info = DocInfo(
                        name=node.name,
                        docstring=ast.get_docstring(node) or "",
                        type="class"
                    )
                    docs.append(doc_info)
        except Exception as e:
            print(f"Error parsing module {module_path}: {e}")
        
        return docs
    
    def _extract_ast_signature(self, node: ast.FunctionDef) -> str:
        """Extract signature from AST node."""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"({', '.join(args)})"
    
    def generate_markdown(self, doc_info: DocInfo) -> str:
        """
        Generate Markdown documentation.
        
        Args:
            doc_info: Documentation information
            
        Returns:
            Markdown string
        """
        lines = []
        
        # Title
        lines.append(f"## {doc_info.name}")
        lines.append("")
        
        # Type
        lines.append(f"**Type:** {doc_info.type}")
        lines.append("")
        
        # Docstring
        if doc_info.docstring:
            lines.append(doc_info.docstring)
            lines.append("")
        
        # Signature
        if doc_info.signature:
            lines.append("### Signature")
            lines.append("```python")
            lines.append(f"def {doc_info.name}{doc_info.signature}")
            lines.append("```")
            lines.append("")
        
        # Parameters
        if doc_info.parameters:
            lines.append("### Parameters")
            for param in doc_info.parameters:
                param_str = f"- `{param['name']}`"
                if param.get('type'):
                    param_str += f" ({param['type']})"
                if param.get('default'):
                    param_str += f" = {param['default']}"
                if not param.get('required'):
                    param_str += " (optional)"
                lines.append(param_str)
            lines.append("")
        
        # Returns
        if doc_info.returns:
            lines.append("### Returns")
            lines.append(f"`{doc_info.returns}`")
            lines.append("")
        
        # Examples
        if doc_info.examples:
            lines.append("### Examples")
            lines.append("```python")
            lines.extend(doc_info.examples)
            lines.append("```")
            lines.append("")
        
        return "\n".join(lines)




