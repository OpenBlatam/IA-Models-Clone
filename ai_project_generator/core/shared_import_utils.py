"""
Shared Import Utilities - Utilidades compartidas de imports
==========================================================

Módulo compartido para gestión de imports condicionales que puede ser
utilizado por múltiples módulos del proyecto.

Proporciona:
- ImportGroup: Dataclass para organizar grupos de imports
- ImportManager: Clase principal para gestionar imports
- Funciones de utilidad para verificar estado de imports
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ImportPriority(Enum):
    """Prioridades de importación."""
    CRITICAL = 10
    HIGH = 9
    MEDIUM = 8
    NORMAL = 7
    LOW = 6
    OPTIONAL = 5
    EXPERIMENTAL = 4
    DEPRECATED = 3
    LEGACY = 2
    MINIMAL = 1


@dataclass
class ImportGroup:
    """
    Grupo de imports relacionados.
    
    Attributes:
        name: Nombre identificador del grupo
        module_path: Ruta del módulo relativa (ej: ".models")
        symbols: Lista de nombres de símbolos a importar
        description: Descripción del grupo (opcional)
        is_optional: Si True, el grupo es opcional y los errores se ignoran
        priority: Prioridad de importación (ImportPriority o int)
        category: Categoría del grupo (opcional, para organización)
    """
    name: str
    module_path: str
    symbols: List[str]
    description: str = ""
    is_optional: bool = True
    priority: int = 7
    category: str = "general"
    
    def __post_init__(self):
        """Convertir ImportPriority a int si es necesario."""
        if isinstance(self.priority, ImportPriority):
            self.priority = self.priority.value


class SharedImportManager:
    """
    Gestor de imports condicionales compartido.
    
    Proporciona una forma organizada de gestionar imports opcionales
    con mejor manejo de errores, logging y estadísticas.
    """
    
    def __init__(self, namespace: Dict[str, Any], verbose: bool = False):
        """
        Inicializar gestor de imports.
        
        Args:
            namespace: Namespace donde asignar los imports (típicamente globals()).
            verbose: Si True, loguea información detallada de imports.
        """
        self.namespace = namespace
        self.verbose = verbose
        self._imported_symbols: Dict[str, Any] = {}
        self._failed_imports: Set[str] = set()
        self._successful_imports: Set[str] = set()
        self._import_groups: Dict[str, ImportGroup] = {}
        self._import_results: Dict[str, Dict[str, bool]] = {}
    
    def register_group(self, group: ImportGroup) -> None:
        """
        Registrar un grupo de imports.
        
        Args:
            group: Grupo de imports a registrar.
        """
        self._import_groups[group.name] = group
    
    def import_group(self, group: ImportGroup) -> Dict[str, bool]:
        """
        Importar un grupo de símbolos.
        
        Args:
            group: Grupo de imports a importar.
            
        Returns:
            Diccionario con estado de cada símbolo (True = importado exitosamente).
        """
        results = {}
        
        try:
            fromlist = group.symbols
            module = __import__(group.module_path, fromlist=fromlist, level=1)
            
            imported_count = 0
            for symbol_name in group.symbols:
                try:
                    if hasattr(module, symbol_name):
                        symbol_value = getattr(module, symbol_name)
                        self.namespace[symbol_name] = symbol_value
                        self._imported_symbols[symbol_name] = symbol_value
                        self._successful_imports.add(symbol_name)
                        results[symbol_name] = True
                        imported_count += 1
                    else:
                        if self.verbose:
                            logger.debug(
                                f"Symbol '{symbol_name}' not found in '{group.module_path}'"
                            )
                        self.namespace[symbol_name] = None
                        self._imported_symbols[symbol_name] = None
                        self._failed_imports.add(symbol_name)
                        results[symbol_name] = False
                except AttributeError as e:
                    if self.verbose:
                        logger.debug(
                            f"Failed to import '{symbol_name}' from '{group.module_path}': {e}"
                        )
                    self.namespace[symbol_name] = None
                    self._imported_symbols[symbol_name] = None
                    self._failed_imports.add(symbol_name)
                    results[symbol_name] = False
            
            if imported_count > 0 and self.verbose:
                logger.debug(
                    f"Imported {imported_count}/{len(group.symbols)} symbols "
                    f"from {group.module_path} ({group.name})"
                )
                
        except ImportError as e:
            if self.verbose:
                logger.debug(f"{group.name} not available ({group.module_path}): {e}")
            
            for symbol_name in group.symbols:
                self.namespace[symbol_name] = None
                self._imported_symbols[symbol_name] = None
                self._failed_imports.add(symbol_name)
                results[symbol_name] = False
                
        except Exception as e:
            logger.warning(
                f"Error importing {group.name} ({group.module_path}): {e}",
                exc_info=self.verbose
            )
            
            for symbol_name in group.symbols:
                self.namespace[symbol_name] = None
                self._imported_symbols[symbol_name] = None
                self._failed_imports.add(symbol_name)
                results[symbol_name] = False
        
        self._import_results[group.name] = results
        return results
    
    def import_all_groups(self, groups: List[ImportGroup]) -> Dict[str, Dict[str, bool]]:
        """
        Importar múltiples grupos de imports.
        
        Args:
            groups: Lista de grupos de imports a importar.
            
        Returns:
            Diccionario con resultados por grupo.
        """
        results = {}
        
        sorted_groups = sorted(groups, key=lambda g: g.priority, reverse=True)
        
        for group in sorted_groups:
            results[group.name] = self.import_group(group)
        
        return results
    
    def get_import_status(self) -> Dict[str, Any]:
        """
        Obtener estado de los imports.
        
        Returns:
            Diccionario con estadísticas de imports.
        """
        total = len(self._imported_symbols)
        successful = len(self._successful_imports)
        failed = len(self._failed_imports)
        
        return {
            "total_symbols": total,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total * 100) if total > 0 else 0.0,
            "successful_symbols": list(self._successful_imports),
            "failed_symbols": list(self._failed_imports),
            "groups": {
                name: {
                    "total": len(group.symbols),
                    "successful": sum(1 for s in group.symbols if s in self._successful_imports),
                    "failed": sum(1 for s in group.symbols if s in self._failed_imports),
                }
                for name, group in self._import_groups.items()
            },
        }
    
    def check_symbol(self, symbol_name: str) -> bool:
        """
        Verificar si un símbolo está disponible.
        
        Args:
            symbol_name: Nombre del símbolo a verificar.
            
        Returns:
            True si el símbolo está disponible, False en caso contrario.
        """
        return symbol_name in self._successful_imports
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos disponibles.
        
        Returns:
            Lista de nombres de símbolos disponibles.
        """
        return list(self._successful_imports)
    
    def get_missing_symbols(self) -> List[str]:
        """
        Obtener lista de símbolos no disponibles.
        
        Returns:
            Lista de nombres de símbolos faltantes.
        """
        return list(self._failed_imports)
    
    def get_group_status(self, group_name: str) -> Optional[Dict[str, Any]]:
        """
        Obtener estado de un grupo específico.
        
        Args:
            group_name: Nombre del grupo.
            
        Returns:
            Diccionario con estado del grupo o None si no existe.
        """
        if group_name not in self._import_groups:
            return None
        
        group = self._import_groups[group_name]
        results = self._import_results.get(group_name, {})
        
        return {
            "name": group_name,
            "description": group.description,
            "category": group.category,
            "priority": group.priority,
            "total_symbols": len(group.symbols),
            "successful": sum(1 for v in results.values() if v),
            "failed": sum(1 for v in results.values() if not v),
            "symbols": results,
        }
    
    def get_category_status(self, category: str) -> Dict[str, Any]:
        """
        Obtener estado de todos los grupos de una categoría.
        
        Args:
            category: Nombre de la categoría.
            
        Returns:
            Diccionario con estado de los grupos de la categoría.
        """
        category_groups = [
            name for name, group in self._import_groups.items()
            if group.category == category
        ]
        
        return {
            "category": category,
            "groups": {
                name: self.get_group_status(name)
                for name in category_groups
            },
        }


def create_import_summary(manager: SharedImportManager) -> str:
    """
    Crear resumen legible del estado de imports.
    
    Args:
        manager: Gestor de imports.
        
    Returns:
        String con resumen formateado.
    """
    status = manager.get_import_status()
    
    lines = [
        "=" * 60,
        "Import Status Summary",
        "=" * 60,
        f"Total Symbols: {status['total_symbols']}",
        f"Successful: {status['successful']} ({status['success_rate']:.1f}%)",
        f"Failed: {status['failed']}",
        "",
        "Groups:",
    ]
    
    for group_name, group_info in status.get("groups", {}).items():
        success_rate = (
            (group_info["successful"] / group_info["total"] * 100)
            if group_info["total"] > 0
            else 0.0
        )
        lines.append(
            f"  {group_name}: {group_info['successful']}/{group_info['total']} "
            f"({success_rate:.1f}%)"
        )
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

