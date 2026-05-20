"""
Context Analyzer
================

Sistema que analiza el contexto del código antes de hacer cambios,
siguiendo las mejores prácticas de Devin de entender el código
antes de modificarlo.
"""

import logging
import ast
from typing import Optional, Dict, Any, List, Set
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FileContext:
    """Contexto de un archivo"""
    file_path: str
    imports: List[str] = field(default_factory=list)
    frameworks: Set[str] = field(default_factory=set)
    libraries: Set[str] = field(default_factory=set)
    patterns: List[str] = field(default_factory=list)
    conventions: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "file_path": self.file_path,
            "imports": self.imports,
            "frameworks": sorted(list(self.frameworks)),
            "libraries": sorted(list(self.libraries)),
            "patterns": self.patterns,
            "conventions": self.conventions
        }


@dataclass
class ComponentAnalysis:
    """Análisis de componente existente"""
    component_name: str
    file_path: str
    component_type: str
    framework: Optional[str] = None
    naming_convention: Optional[str] = None
    typing_style: Optional[str] = None
    patterns_used: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "component_name": self.component_name,
            "file_path": self.file_path,
            "component_type": self.component_type,
            "framework": self.framework,
            "naming_convention": self.naming_convention,
            "typing_style": self.typing_style,
            "patterns_used": self.patterns_used,
            "dependencies": self.dependencies
        }


class ContextAnalyzer:
    """
    Analizador de contexto de código.
    
    Analiza el contexto del código antes de hacer cambios,
    siguiendo las mejores prácticas de Devin.
    """
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar analizador de contexto.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.context_cache: Dict[str, FileContext] = {}
        logger.info("🔍 Context analyzer initialized")
    
    def analyze_file_context(self, file_path: str) -> FileContext:
        """
        Analizar contexto de un archivo.
        
        Args:
            file_path: Ruta del archivo.
        
        Returns:
            Contexto del archivo.
        """
        if file_path in self.context_cache:
            return self.context_cache[file_path]
        
        file_path_obj = Path(file_path)
        if not file_path_obj.is_absolute():
            file_path_obj = self.workspace_root / file_path_obj
        
        if not file_path_obj.exists():
            return FileContext(file_path=file_path)
        
        context = FileContext(file_path=file_path)
        
        try:
            content = file_path_obj.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        lib_name = alias.name.split('.')[0]
                        context.imports.append(alias.name)
                        context.libraries.add(lib_name)
                        
                        if lib_name in ['fastapi', 'flask', 'django', 'tornado']:
                            context.frameworks.add(lib_name)
                        elif lib_name in ['pydantic', 'marshmallow']:
                            context.frameworks.add('validation')
                        elif lib_name in ['sqlalchemy', 'peewee', 'tortoise']:
                            context.frameworks.add('orm')
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        lib_name = node.module.split('.')[0]
                        context.imports.append(node.module)
                        context.libraries.add(lib_name)
                        
                        if lib_name in ['fastapi', 'flask', 'django', 'tornado']:
                            context.frameworks.add(lib_name)
                        elif lib_name in ['pydantic', 'marshmallow']:
                            context.frameworks.add('validation')
                        elif lib_name in ['sqlalchemy', 'peewee', 'tortoise']:
                            context.frameworks.add('orm')
            
            context.patterns = self._detect_patterns(content)
            context.conventions = self._detect_conventions(content)
            
        except Exception as e:
            logger.error(f"Error analyzing file context {file_path}: {e}", exc_info=True)
        
        self.context_cache[file_path] = context
        return context
    
    def analyze_component(
        self,
        component_name: str,
        component_type: str,
        search_path: Optional[str] = None
    ) -> List[ComponentAnalysis]:
        """
        Analizar componentes existentes similares.
        
        Args:
            component_name: Nombre del componente a buscar.
            component_type: Tipo de componente (class, function, etc.).
            search_path: Ruta donde buscar (opcional).
        
        Returns:
            Lista de análisis de componentes similares.
        """
        analyses: List[ComponentAnalysis] = []
        
        search_dir = Path(search_path) if search_path else self.workspace_root
        
        try:
            python_files = list(search_dir.rglob("*.py"))
            
            for py_file in python_files[:50]:
                if not py_file.is_file():
                    continue
                
                try:
                    content = py_file.read_text(encoding='utf-8')
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if component_type == "class" and isinstance(node, ast.ClassDef):
                            if component_name.lower() in node.name.lower() or \
                               any(component_name.lower() in base.id.lower() 
                                   for base in node.bases if isinstance(base, ast.Name)):
                                analysis = ComponentAnalysis(
                                    component_name=node.name,
                                    file_path=str(py_file.relative_to(self.workspace_root)),
                                    component_type="class"
                                )
                                
                                file_context = self.analyze_file_context(str(py_file))
                                analysis.frameworks = file_context.frameworks
                                analysis.naming_convention = self._detect_naming(node.name)
                                analysis.typing_style = self._detect_typing(node)
                                
                                analyses.append(analysis)
                        
                        elif component_type == "function" and isinstance(node, ast.FunctionDef):
                            if component_name.lower() in node.name.lower():
                                analysis = ComponentAnalysis(
                                    component_name=node.name,
                                    file_path=str(py_file.relative_to(self.workspace_root)),
                                    component_type="function"
                                )
                                
                                file_context = self.analyze_file_context(str(py_file))
                                analysis.frameworks = file_context.frameworks
                                analysis.naming_convention = self._detect_naming(node.name)
                                analysis.typing_style = self._detect_typing(node)
                                
                                analyses.append(analysis)
                
                except Exception as e:
                    logger.debug(f"Error analyzing {py_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error analyzing components: {e}", exc_info=True)
        
        return analyses
    
    def get_surrounding_context(
        self,
        file_path: str,
        line: int,
        context_lines: int = 10
    ) -> Dict[str, Any]:
        """
        Obtener contexto alrededor de una línea.
        
        Args:
            file_path: Ruta del archivo.
            line: Línea de interés.
            context_lines: Número de líneas de contexto.
        
        Returns:
            Contexto alrededor de la línea.
        """
        file_path_obj = Path(file_path)
        if not file_path_obj.is_absolute():
            file_path_obj = self.workspace_root / file_path_obj
        
        if not file_path_obj.exists():
            return {"error": "File not found"}
        
        try:
            content = file_path_obj.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            start_line = max(0, line - context_lines - 1)
            end_line = min(len(lines), line + context_lines)
            
            surrounding = lines[start_line:end_line]
            
            file_context = self.analyze_file_context(file_path)
            
            return {
                "file_path": file_path,
                "line": line,
                "surrounding_lines": surrounding,
                "imports": file_context.imports,
                "frameworks": sorted(list(file_context.frameworks)),
                "libraries": sorted(list(file_context.libraries))
            }
        
        except Exception as e:
            logger.error(f"Error getting surrounding context: {e}", exc_info=True)
            return {"error": str(e)}
    
    def _detect_patterns(self, content: str) -> List[str]:
        """Detectar patrones en el código"""
        patterns = []
        
        if "@dataclass" in content:
            patterns.append("dataclass")
        if "@property" in content:
            patterns.append("property")
        if "async def" in content:
            patterns.append("async")
        if "class " in content and "(" in content:
            patterns.append("inheritance")
        if "__init__" in content:
            patterns.append("constructor")
        if "try:" in content and "except" in content:
            patterns.append("error_handling")
        
        return patterns
    
    def _detect_conventions(self, content: str) -> Dict[str, Any]:
        """Detectar convenciones en el código"""
        conventions = {}
        
        lines = content.split('\n')
        if lines:
            first_line = lines[0]
            if first_line.startswith('"""') or first_line.startswith("'''"):
                conventions["docstring_style"] = "triple_quotes"
            elif first_line.startswith('"') or first_line.startswith("'"):
                conventions["docstring_style"] = "single_quotes"
        
        return conventions
    
    def _detect_naming(self, name: str) -> str:
        """Detectar convención de nombres"""
        if '_' in name and name.islower():
            return "snake_case"
        elif name[0].isupper() and '_' not in name:
            return "PascalCase"
        elif name[0].islower() and any(c.isupper() for c in name[1:]):
            return "camelCase"
        else:
            return "mixed"
    
    def _detect_typing(self, node: ast.AST) -> Optional[str]:
        """Detectar estilo de typing"""
        if isinstance(node, ast.FunctionDef):
            has_annotations = node.returns is not None or \
                            any(arg.annotation for arg in node.args.args)
            if has_annotations:
                return "type_hints"
        elif isinstance(node, ast.ClassDef):
            return "class"
        return None

