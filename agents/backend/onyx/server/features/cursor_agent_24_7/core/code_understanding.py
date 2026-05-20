"""
Code Understanding System
==========================

Sistema de comprensión de código similar a LSP (Language Server Protocol)
para ayudar al agente a entender mejor el código base.
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import ast
import re

logger = logging.getLogger(__name__)


class CodeSymbol:
    """Símbolo de código (función, clase, variable, etc.)"""
    
    def __init__(
        self,
        name: str,
        symbol_type: str,
        file_path: str,
        line: int,
        docstring: Optional[str] = None,
        signature: Optional[str] = None
    ):
        self.name = name
        self.symbol_type = symbol_type
        self.file_path = file_path
        self.line = line
        self.docstring = docstring
        self.signature = signature
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "name": self.name,
            "type": self.symbol_type,
            "file_path": self.file_path,
            "line": self.line,
            "docstring": self.docstring,
            "signature": self.signature
        }


class CodeUnderstanding:
    """
    Sistema de comprensión de código.
    
    Proporciona capacidades similares a LSP para:
    - Encontrar definiciones
    - Encontrar referencias
    - Obtener información de hover
    - Analizar estructura del código
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar sistema de comprensión de código.
        
        Args:
            workspace_root: Raíz del workspace (opcional).
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.symbol_cache: Dict[str, List[CodeSymbol]] = {}
        logger.info(f"📚 Code understanding initialized for: {self.workspace_root}")
    
    def find_definition(
        self,
        symbol_name: str,
        file_path: str,
        line: int
    ) -> Optional[CodeSymbol]:
        """
        Encontrar definición de un símbolo.
        
        Args:
            symbol_name: Nombre del símbolo.
            file_path: Archivo donde se usa el símbolo.
            line: Línea donde se usa el símbolo.
        
        Returns:
            Símbolo encontrado o None.
        """
        try:
            file_path_obj = Path(file_path)
            if not file_path_obj.is_absolute():
                file_path_obj = self.workspace_root / file_path_obj
            
            if not file_path_obj.exists():
                logger.warning(f"File not found: {file_path_obj}")
                return None
            
            symbols = self._parse_file_symbols(str(file_path_obj))
            
            for symbol in symbols:
                if symbol.name == symbol_name:
                    return symbol
            
            logger.debug(f"Symbol '{symbol_name}' not found in {file_path}")
            return None
            
        except Exception as e:
            logger.error(f"Error finding definition: {e}", exc_info=True)
            return None
    
    def find_references(
        self,
        symbol_name: str,
        file_path: str,
        line: int
    ) -> List[CodeSymbol]:
        """
        Encontrar referencias a un símbolo.
        
        Args:
            symbol_name: Nombre del símbolo.
            file_path: Archivo donde está definido el símbolo.
            line: Línea donde está definido el símbolo.
        
        Returns:
            Lista de referencias encontradas.
        """
        references: List[CodeSymbol] = []
        
        try:
            definition = self.find_definition(symbol_name, file_path, line)
            if not definition:
                return references
            
            workspace_files = list(self.workspace_root.rglob("*.py"))
            
            for py_file in workspace_files:
                if py_file.is_file():
                    try:
                        content = py_file.read_text(encoding='utf-8')
                        if symbol_name in content:
                            lines = content.split('\n')
                            for idx, line_content in enumerate(lines, 1):
                                if symbol_name in line_content and idx != definition.line:
                                    references.append(CodeSymbol(
                                        name=symbol_name,
                                        symbol_type="reference",
                                        file_path=str(py_file.relative_to(self.workspace_root)),
                                        line=idx
                                    ))
                    except Exception as e:
                        logger.debug(f"Error reading {py_file}: {e}")
            
            return references
            
        except Exception as e:
            logger.error(f"Error finding references: {e}", exc_info=True)
            return references
    
    def get_hover_info(
        self,
        symbol_name: str,
        file_path: str,
        line: int
    ) -> Optional[Dict[str, Any]]:
        """
        Obtener información de hover sobre un símbolo.
        
        Args:
            symbol_name: Nombre del símbolo.
            file_path: Archivo donde está el símbolo.
            line: Línea donde está el símbolo.
        
        Returns:
            Información del símbolo o None.
        """
        symbol = self.find_definition(symbol_name, file_path, line)
        if not symbol:
            return None
        
        return {
            "name": symbol.name,
            "type": symbol.symbol_type,
            "file_path": symbol.file_path,
            "line": symbol.line,
            "docstring": symbol.docstring,
            "signature": symbol.signature
        }
    
    def _parse_file_symbols(self, file_path: str) -> List[CodeSymbol]:
        """
        Parsear símbolos de un archivo Python.
        
        Args:
            file_path: Ruta del archivo.
        
        Returns:
            Lista de símbolos encontrados.
        """
        if file_path in self.symbol_cache:
            return self.symbol_cache[file_path]
        
        symbols: List[CodeSymbol] = []
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists() or not file_path_obj.suffix == '.py':
            return symbols
        
        try:
            content = file_path_obj.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(file_path_obj))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    signature = self._get_function_signature(node)
                    symbols.append(CodeSymbol(
                        name=node.name,
                        symbol_type="function",
                        file_path=str(file_path_obj.relative_to(self.workspace_root)),
                        line=node.lineno,
                        docstring=docstring,
                        signature=signature
                    ))
                elif isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    symbols.append(CodeSymbol(
                        name=node.name,
                        symbol_type="class",
                        file_path=str(file_path_obj.relative_to(self.workspace_root)),
                        line=node.lineno,
                        docstring=docstring
                    ))
            
            self.symbol_cache[file_path] = symbols
            return symbols
            
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
            return symbols
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Obtener firma de función"""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                try:
                    if hasattr(ast, 'unparse'):
                        arg_str += f": {ast.unparse(arg.annotation)}"
                    else:
                        arg_str += ": Any"
                except Exception:
                    arg_str += ": Any"
            args.append(arg_str)
        
        signature = f"{node.name}({', '.join(args)})"
        if node.returns:
            try:
                if hasattr(ast, 'unparse'):
                    signature += f" -> {ast.unparse(node.returns)}"
                else:
                    signature += " -> Any"
            except Exception:
                signature += " -> Any"
        
        return signature
    
    def analyze_codebase_structure(self) -> Dict[str, Any]:
        """
        Analizar estructura del código base.
        
        Returns:
            Diccionario con información de la estructura.
        """
        structure = {
            "total_files": 0,
            "total_functions": 0,
            "total_classes": 0,
            "files": []
        }
        
        try:
            python_files = list(self.workspace_root.rglob("*.py"))
            structure["total_files"] = len(python_files)
            
            for py_file in python_files:
                if py_file.is_file():
                    symbols = self._parse_file_symbols(str(py_file))
                    functions = [s for s in symbols if s.symbol_type == "function"]
                    classes = [s for s in symbols if s.symbol_type == "class"]
                    
                    structure["total_functions"] += len(functions)
                    structure["total_classes"] += len(classes)
                    
                    structure["files"].append({
                        "path": str(py_file.relative_to(self.workspace_root)),
                        "functions": len(functions),
                        "classes": len(classes)
                    })
            
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing codebase: {e}", exc_info=True)
            return structure

