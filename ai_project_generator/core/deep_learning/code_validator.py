"""
Code Validator
==============

Validador de código generado.
"""

import logging
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Resultado de validación."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class CodeValidator:
    """
    Validador de código Python generado.
    """
    
    def __init__(self):
        """Inicializar validador."""
        self.rules: List[Callable[[str, Path], List[str]]] = []
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Registrar reglas por defecto."""
        self.rules = [
            self._check_syntax,
            self._check_imports,
            self._check_docstrings,
            self._check_type_hints,
            self._check_naming_conventions
        ]
    
    def validate_file(self, file_path: Path) -> ValidationResult:
        """
        Validar archivo Python.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Resultado de validación
        """
        result = ValidationResult(is_valid=True)
        
        if not file_path.exists():
            result.is_valid = False
            result.errors.append(f"Archivo no existe: {file_path}")
            return result
        
        if not file_path.suffix == '.py':
            result.warnings.append(f"Archivo no es Python: {file_path}")
            return result
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Error leyendo archivo: {e}")
            return result
        
        # Ejecutar todas las reglas
        for rule in self.rules:
            try:
                issues = rule(content, file_path)
                if issues:
                    result.warnings.extend(issues)
            except Exception as e:
                logger.warning(f"Error ejecutando regla: {e}")
        
        result.is_valid = len(result.errors) == 0
        return result
    
    def validate_directory(
        self,
        directory: Path,
        recursive: bool = True
    ) -> Dict[str, ValidationResult]:
        """
        Validar todos los archivos Python en un directorio.
        
        Args:
            directory: Directorio a validar
            recursive: Validar recursivamente
            
        Returns:
            Diccionario con resultados por archivo
        """
        results = {}
        
        pattern = "**/*.py" if recursive else "*.py"
        
        for py_file in directory.glob(pattern):
            results[str(py_file)] = self.validate_file(py_file)
        
        return results
    
    def _check_syntax(self, content: str, file_path: Path) -> List[str]:
        """Verificar sintaxis Python."""
        issues = []
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(f"Error de sintaxis: {e.msg} en línea {e.lineno}")
        return issues
    
    def _check_imports(self, content: str, file_path: Path) -> List[str]:
        """Verificar imports."""
        issues = []
        
        # Verificar imports no usados (básico)
        try:
            tree = ast.parse(content)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(node)
            
            # Verificar imports estándar primero
            stdlib_imports = ['os', 'sys', 'pathlib', 'typing', 'logging']
            for imp in imports:
                if isinstance(imp, ast.Import):
                    for alias in imp.names:
                        if alias.name in stdlib_imports and alias.asname:
                            issues.append(f"Considerar usar '{alias.name}' directamente en lugar de '{alias.asname}'")
        except Exception:
            pass
        
        return issues
    
    def _check_docstrings(self, content: str, file_path: Path) -> List[str]:
        """Verificar docstrings."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            # Verificar funciones y clases sin docstring
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    if not ast.get_docstring(node):
                        issues.append(f"{node.__class__.__name__} '{node.name}' sin docstring")
        except Exception:
            pass
        
        return issues
    
    def _check_type_hints(self, content: str, file_path: Path) -> List[str]:
        """Verificar type hints."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Verificar si tiene type hints en parámetros
                    has_hints = any(
                        arg.annotation is not None
                        for arg in node.args.args
                    )
                    
                    if not has_hints and len(node.args.args) > 0:
                        issues.append(f"Función '{node.name}' sin type hints en parámetros")
        except Exception:
            pass
        
        return issues
    
    def _check_naming_conventions(self, content: str, file_path: Path) -> List[str]:
        """Verificar convenciones de nombres."""
        issues = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Verificar nombres de funciones (snake_case)
                    if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                        issues.append(f"Nombre de función '{node.name}' no sigue snake_case")
                
                elif isinstance(node, ast.ClassDef):
                    # Verificar nombres de clases (PascalCase)
                    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                        issues.append(f"Nombre de clase '{node.name}' no sigue PascalCase")
        except Exception:
            pass
        
        return issues
    
    def add_rule(self, rule: Callable[[str, Path], List[str]]) -> None:
        """
        Agregar regla personalizada.
        
        Args:
            rule: Función que recibe (content, file_path) y retorna lista de issues
        """
        self.rules.append(rule)


# Instancia global
_global_validator: Optional[CodeValidator] = None


def get_validator() -> CodeValidator:
    """
    Obtener instancia global del validador.
    
    Returns:
        Instancia del validador
    """
    global _global_validator
    
    if _global_validator is None:
        _global_validator = CodeValidator()
    
    return _global_validator

