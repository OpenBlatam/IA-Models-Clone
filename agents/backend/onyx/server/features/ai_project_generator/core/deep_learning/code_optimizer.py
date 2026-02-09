"""
Code Optimizer
==============

Optimizador de código generado para mejor rendimiento.
"""

import logging
import ast
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Resultado de optimización."""
    original_size: int
    optimized_size: int
    optimizations_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CodeOptimizer:
    """
    Optimizador de código Python.
    """
    
    def __init__(self):
        """Inicializar optimizador."""
        self.optimizations: List[Callable[[str], Tuple[str, List[str]]]] = []
        self._register_default_optimizations()
    
    def _register_default_optimizations(self) -> None:
        """Registrar optimizaciones por defecto."""
        self.optimizations = [
            self._remove_unused_imports,
            self._optimize_string_concat,
            self._optimize_list_comprehensions,
            self._remove_dead_code
        ]
    
    def optimize_file(self, file_path: Path) -> OptimizationResult:
        """
        Optimizar archivo Python.
        
        Args:
            file_path: Ruta del archivo
            
        Returns:
            Resultado de optimización
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return OptimizationResult(0, 0)
        
        try:
            original_content = file_path.read_text(encoding='utf-8')
            original_size = len(original_content)
            
            optimized_content = original_content
            optimizations_applied = []
            warnings = []
            
            for optimization in self.optimizations:
                try:
                    new_content, opt_warnings = optimization(optimized_content)
                    if new_content != optimized_content:
                        optimizations_applied.append(optimization.__name__)
                        optimized_content = new_content
                    warnings.extend(opt_warnings)
                except Exception as e:
                    logger.warning(f"Error en optimización {optimization.__name__}: {e}")
            
            optimized_size = len(optimized_content)
            
            # Guardar si hay cambios
            if optimized_content != original_content:
                file_path.write_text(optimized_content, encoding='utf-8')
            
            return OptimizationResult(
                original_size=original_size,
                optimized_size=optimized_size,
                optimizations_applied=optimizations_applied,
                warnings=warnings
            )
        except Exception as e:
            logger.error(f"Error optimizando archivo {file_path}: {e}")
            return OptimizationResult(0, 0)
    
    def _remove_unused_imports(self, content: str) -> Tuple[str, List[str]]:
        """Remover imports no usados."""
        warnings = []
        try:
            tree = ast.parse(content)
            used_names = set()
            
            # Recopilar nombres usados
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used_names.add(node.attr)
            
            # Verificar imports
            lines = content.split('\n')
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    # Verificar si el import se usa
                    import_names = self._extract_import_names(line)
                    if import_names and not any(name in used_names for name in import_names):
                        warnings.append(f"Import no usado: {line.strip()}")
                        i += 1
                        continue
                new_lines.append(line)
                i += 1
            
            return '\n'.join(new_lines), warnings
        except Exception:
            return content, warnings
    
    def _extract_import_names(self, import_line: str) -> List[str]:
        """Extraer nombres de imports."""
        names = []
        try:
            if import_line.strip().startswith('import '):
                parts = import_line.replace('import', '').strip().split(',')
                for part in parts:
                    name = part.strip().split(' as ')[0].split('.')[0]
                    names.append(name)
            elif import_line.strip().startswith('from '):
                # from module import name1, name2
                if ' import ' in import_line:
                    import_part = import_line.split(' import ')[1]
                    parts = import_part.split(',')
                    for part in parts:
                        name = part.strip().split(' as ')[0]
                        names.append(name)
        except Exception:
            pass
        return names
    
    def _optimize_string_concat(self, content: str) -> Tuple[str, List[str]]:
        """Optimizar concatenación de strings."""
        warnings = []
        # Reemplazar "str1" + "str2" con "str1str2"
        pattern = r'"([^"]+)"\s*\+\s*"([^"]+)"'
        matches = re.finditer(pattern, content)
        replacements = []
        for match in matches:
            new_str = f'"{match.group(1)}{match.group(2)}"'
            replacements.append((match.start(), match.end(), new_str))
        
        if replacements:
            # Aplicar reemplazos en reverso para mantener índices
            for start, end, replacement in reversed(replacements):
                content = content[:start] + replacement + content[end:]
            warnings.append("Optimizada concatenación de strings")
        
        return content, warnings
    
    def _optimize_list_comprehensions(self, content: str) -> Tuple[str, List[str]]:
        """Optimizar list comprehensions."""
        warnings = []
        # Esta es una optimización básica, se puede expandir
        return content, warnings
    
    def _remove_dead_code(self, content: str) -> Tuple[str, List[str]]:
        """Remover código muerto."""
        warnings = []
        try:
            tree = ast.parse(content)
            # Detectar código después de return/raise
            lines = content.split('\n')
            new_lines = []
            i = 0
            while i < len(lines):
                line = lines[i]
                stripped = line.strip()
                
                # Detectar return/raise
                if stripped.startswith('return ') or stripped == 'return' or stripped.startswith('raise '):
                    new_lines.append(line)
                    i += 1
                    # Verificar si hay código después (en mismo nivel de indentación)
                    indent_level = len(line) - len(line.lstrip())
                    while i < len(lines):
                        next_line = lines[i]
                        if not next_line.strip():  # Línea vacía
                            new_lines.append(next_line)
                            i += 1
                            continue
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent_level and not next_line.strip().startswith('#'):
                            # Código después de return/raise
                            warnings.append(f"Posible código muerto después de return en línea {i+1}")
                            break
                        new_lines.append(next_line)
                        i += 1
                    continue
                
                new_lines.append(line)
                i += 1
            
            return '\n'.join(new_lines), warnings
        except Exception:
            return content, warnings
    
    def add_optimization(self, optimization: Callable[[str], Tuple[str, List[str]]]) -> None:
        """
        Agregar optimización personalizada.
        
        Args:
            optimization: Función que recibe contenido y retorna (contenido_optimizado, warnings)
        """
        self.optimizations.append(optimization)


# Instancia global
_global_optimizer: Optional[CodeOptimizer] = None


def get_optimizer() -> CodeOptimizer:
    """
    Obtener instancia global del optimizador.
    
    Returns:
        Instancia del optimizador
    """
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = CodeOptimizer()
    
    return _global_optimizer

