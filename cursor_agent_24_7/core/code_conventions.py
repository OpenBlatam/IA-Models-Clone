"""
Code Conventions Analyzer
=========================

Sistema que analiza y detecta convenciones de código del proyecto
para seguir las mejores prácticas de Devin: entender el estilo del código
antes de hacer cambios.
"""

import logging
import ast
import re
from typing import Optional, Dict, Any, List, Set
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class CodeConventions:
    """Convenciones de código detectadas"""
    indentation: str = "    "
    max_line_length: int = 120
    quote_style: str = '"'
    import_style: str = "absolute"
    naming_style: Dict[str, str] = field(default_factory=dict)
    docstring_style: Optional[str] = None
    type_hints: bool = True
    async_style: str = "async/await"
    libraries_used: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "indentation": self.indentation,
            "max_line_length": self.max_line_length,
            "quote_style": self.quote_style,
            "import_style": self.import_style,
            "naming_style": self.naming_style,
            "docstring_style": self.docstring_style,
            "type_hints": self.type_hints,
            "async_style": self.async_style,
            "libraries_used": sorted(list(self.libraries_used))
        }


class CodeConventionsAnalyzer:
    """
    Analizador de convenciones de código.
    
    Analiza el código existente para detectar:
    - Estilo de indentación
    - Longitud de líneas
    - Estilo de comillas
    - Estilo de imports
    - Convenciones de nombres
    - Uso de type hints
    - Librerías utilizadas
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar analizador.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.conventions: Optional[CodeConventions] = None
        logger.info(f"📐 Code conventions analyzer initialized")
    
    def analyze_workspace(self, sample_files: Optional[List[str]] = None) -> CodeConventions:
        """
        Analizar workspace para detectar convenciones.
        
        Args:
            sample_files: Lista de archivos a analizar (opcional, si None analiza todos).
        
        Returns:
            Convenciones detectadas.
        """
        conventions = CodeConventions()
        
        if sample_files:
            files_to_analyze = [Path(f) for f in sample_files]
        else:
            files_to_analyze = list(self.workspace_root.rglob("*.py"))[:20]
        
        indentation_samples = []
        line_length_samples = []
        quote_samples = []
        import_samples = []
        naming_samples = {"functions": [], "classes": [], "variables": []}
        type_hint_count = 0
        total_functions = 0
        
        for file_path in files_to_analyze:
            if not file_path.is_file():
                continue
            
            try:
                content = file_path.read_text(encoding='utf-8')
                lines = content.split('\n')
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    stripped = line.lstrip()
                    if stripped:
                        indent = line[:len(line) - len(stripped)]
                        if indent:
                            indentation_samples.append(indent)
                    
                    line_length_samples.append(len(line))
                    
                    if '"' in line:
                        quote_samples.append('"')
                    elif "'" in line:
                        quote_samples.append("'")
                    
                    if line.strip().startswith('import ') or line.strip().startswith('from '):
                        import_samples.append(line.strip())
                
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            total_functions += 1
                            naming_samples["functions"].append(node.name)
                            if node.returns or any(arg.annotation for arg in node.args.args):
                                type_hint_count += 1
                        elif isinstance(node, ast.ClassDef):
                            naming_samples["classes"].append(node.name)
                        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                            naming_samples["variables"].append(node.id)
                        
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            for alias in (node.names if hasattr(node, 'names') else []):
                                lib_name = alias.name.split('.')[0]
                                conventions.libraries_used.add(lib_name)
                except SyntaxError:
                    pass
                    
            except Exception as e:
                logger.debug(f"Error analyzing {file_path}: {e}")
        
        if indentation_samples:
            most_common_indent = max(set(indentation_samples), key=indentation_samples.count)
            conventions.indentation = most_common_indent
        
        if line_length_samples:
            conventions.max_line_length = max(line_length_samples)
        
        if quote_samples:
            conventions.quote_style = max(set(quote_samples), key=quote_samples.count)
        
        if import_samples:
            relative_imports = sum(1 for imp in import_samples if imp.startswith('from .') or imp.startswith('from ..'))
            if relative_imports > len(import_samples) * 0.3:
                conventions.import_style = "relative"
            else:
                conventions.import_style = "absolute"
        
        if naming_samples["functions"]:
            conventions.naming_style["functions"] = self._detect_naming_style(naming_samples["functions"])
        if naming_samples["classes"]:
            conventions.naming_style["classes"] = self._detect_naming_style(naming_samples["classes"])
        if naming_samples["variables"]:
            conventions.naming_style["variables"] = self._detect_naming_style(naming_samples["variables"])
        
        if total_functions > 0:
            conventions.type_hints = (type_hint_count / total_functions) > 0.5
        
        self.conventions = conventions
        return conventions
    
    def _detect_naming_style(self, names: List[str]) -> str:
        """Detectar estilo de nombres"""
        if not names:
            return "snake_case"
        
        snake_case_count = sum(1 for name in names if '_' in name and name.islower())
        camel_case_count = sum(1 for name in names if name[0].isupper() and '_' not in name)
        
        if snake_case_count > camel_case_count:
            return "snake_case"
        elif camel_case_count > snake_case_count:
            return "PascalCase"
        else:
            return "mixed"
    
    def check_library_usage(self, library_name: str) -> bool:
        """
        Verificar si una librería se usa en el proyecto.
        
        Args:
            library_name: Nombre de la librería.
        
        Returns:
            True si se usa en el proyecto.
        """
        if not self.conventions:
            self.analyze_workspace()
        
        return library_name in (self.conventions.libraries_used if self.conventions else set())
    
    def get_conventions(self) -> Optional[CodeConventions]:
        """
        Obtener convenciones detectadas.
        
        Returns:
            Convenciones o None si no se han analizado.
        """
        if not self.conventions:
            self.analyze_workspace()
        return self.conventions
    
    def check_file_conventions(self, file_path: str) -> Dict[str, Any]:
        """
        Verificar si un archivo sigue las convenciones.
        
        Args:
            file_path: Ruta del archivo.
        
        Returns:
            Diccionario con resultados de verificación.
        """
        if not self.conventions:
            self.analyze_workspace()
        
        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return {"error": "File not found"}
        
        try:
            content = file_path_obj.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            violations = []
            
            for i, line in enumerate(lines, 1):
                if len(line) > (self.conventions.max_line_length if self.conventions else 120):
                    violations.append({
                        "line": i,
                        "type": "line_length",
                        "message": f"Line exceeds max length ({len(line)} > {self.conventions.max_line_length if self.conventions else 120})"
                    })
            
            return {
                "follows_conventions": len(violations) == 0,
                "violations": violations,
                "total_lines": len(lines)
            }
        except Exception as e:
            return {"error": str(e)}

