"""
Documentation Generator - Generación automática de documentación
=================================================================
"""

import logging
import ast
from typing import Dict, Any, List, Optional
import re

logger = logging.getLogger(__name__)


class DocumentationGenerator:
    """
    Genera documentación automática para código.
    """
    
    def __init__(self):
        """Inicializar generador de documentación"""
        self.supported_languages = ["python", "javascript", "typescript"]
    
    def generate_documentation(
        self,
        code: str,
        language: str = "python",
        format: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Genera documentación para código.
        
        Args:
            code: Código a documentar
            language: Lenguaje de programación
            format: Formato de salida (markdown, html, rst)
            
        Returns:
            Documentación generada
        """
        try:
            if language == "python":
                return self._generate_python_docs(code, format)
            elif language in ["javascript", "typescript"]:
                return self._generate_js_docs(code, format)
            else:
                return {
                    "documentation": "",
                    "error": f"Lenguaje no soportado: {language}"
                }
        except Exception as e:
            logger.error(f"Error generando documentación: {e}")
            return {
                "documentation": "",
                "error": str(e)
            }
    
    def _generate_python_docs(self, code: str, format: str) -> Dict[str, Any]:
        """Genera documentación para código Python"""
        try:
            tree = ast.parse(code)
            
            docs = []
            module_doc = None
            
            # Extraer docstring del módulo
            if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
                module_doc = tree.body[0].value.s
            
            # Procesar cada elemento
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_doc = self._document_function(node, format)
                    if func_doc:
                        docs.append(func_doc)
                
                elif isinstance(node, ast.ClassDef):
                    class_doc = self._document_class(node, format)
                    if class_doc:
                        docs.append(class_doc)
            
            # Generar documentación completa
            if format == "markdown":
                documentation = self._format_markdown(module_doc, docs)
            elif format == "html":
                documentation = self._format_html(module_doc, docs)
            else:
                documentation = self._format_markdown(module_doc, docs)
            
            return {
                "documentation": documentation,
                "format": format,
                "elements_documented": len(docs)
            }
            
        except SyntaxError as e:
            logger.warning(f"Error de sintaxis: {e}")
            return {
                "documentation": "",
                "error": f"Syntax error: {str(e)}"
            }
    
    def _document_function(self, node: ast.FunctionDef, format: str) -> Optional[str]:
        """Documenta una función"""
        func_name = node.name
        args = [arg.arg for arg in node.args.args]
        
        # Extraer docstring
        docstring = ast.get_docstring(node)
        
        if format == "markdown":
            doc = f"### {func_name}\n\n"
            
            if docstring:
                doc += f"{docstring}\n\n"
            
            if args:
                doc += "**Parámetros:**\n\n"
                for arg in args:
                    if arg != "self":
                        doc += f"- `{arg}`: (tipo desconocido)\n"
                doc += "\n"
            
            doc += f"**Definida en:** línea {node.lineno}\n\n"
            
            return doc
        
        return None
    
    def _document_class(self, node: ast.ClassDef, format: str) -> Optional[str]:
        """Documenta una clase"""
        class_name = node.name
        docstring = ast.get_docstring(node)
        
        # Encontrar métodos
        methods = [
            n.name for n in node.body
            if isinstance(n, ast.FunctionDef)
        ]
        
        if format == "markdown":
            doc = f"## {class_name}\n\n"
            
            if docstring:
                doc += f"{docstring}\n\n"
            
            if methods:
                doc += "**Métodos:**\n\n"
                for method in methods:
                    doc += f"- `{method}()`\n"
                doc += "\n"
            
            doc += f"**Definida en:** línea {node.lineno}\n\n"
            
            return doc
        
        return None
    
    def _generate_js_docs(self, code: str, format: str) -> Dict[str, Any]:
        """Genera documentación para código JavaScript/TypeScript"""
        docs = []
        
        # Encontrar funciones
        function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|function))'
        matches = re.finditer(function_pattern, code)
        
        for match in matches:
            func_name = match.group(1) or match.group(2)
            if func_name:
                if format == "markdown":
                    doc = f"### {func_name}\n\n"
                    doc += "Función JavaScript/TypeScript\n\n"
                    docs.append(doc)
        
        documentation = "\n".join(docs) if docs else "No se encontraron funciones para documentar."
        
        return {
            "documentation": documentation,
            "format": format,
            "elements_documented": len(docs)
        }
    
    def _format_markdown(self, module_doc: Optional[str], docs: List[str]) -> str:
        """Formatea documentación en Markdown"""
        lines = ["# Documentación del Código", ""]
        
        if module_doc:
            lines.append("## Descripción del Módulo")
            lines.append("")
            lines.append(module_doc)
            lines.append("")
        
        if docs:
            lines.append("## Elementos")
            lines.append("")
            lines.extend(docs)
        
        return "\n".join(lines)
    
    def _format_html(self, module_doc: Optional[str], docs: List[str]) -> str:
        """Formatea documentación en HTML"""
        html = ["<!DOCTYPE html>", "<html>", "<head>", "<title>Documentación</title>", "</head>", "<body>"]
        
        html.append("<h1>Documentación del Código</h1>")
        
        if module_doc:
            html.append("<h2>Descripción del Módulo</h2>")
            html.append(f"<p>{module_doc}</p>")
        
        if docs:
            html.append("<h2>Elementos</h2>")
            for doc in docs:
                # Convertir markdown básico a HTML
                doc_html = doc.replace("### ", "<h3>").replace("\n\n", "</h3><p>")
                html.append(doc_html)
        
        html.extend(["</body>", "</html>"])
        
        return "\n".join(html)




