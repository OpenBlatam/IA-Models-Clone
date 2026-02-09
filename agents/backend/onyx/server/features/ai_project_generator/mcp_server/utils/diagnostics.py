"""
Diagnostics Utilities - Utilidades de diagnóstico para MCP Server
==================================================================

Funciones para diagnosticar problemas, verificar estado del sistema,
y proporcionar información detallada para debugging.
"""

import logging
import sys
import platform
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """
    Obtener información del sistema.
    
    Returns:
        Diccionario con información del sistema.
    """
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "python_path": sys.path,
    }


def get_module_diagnostics() -> Dict[str, Any]:
    """
    Obtener diagnóstico completo del módulo MCP Server.
    
    Returns:
        Diccionario con información de diagnóstico.
    """
    try:
        from .. import (
            get_version, check_imports, get_missing_imports,
            get_available_features, get_module_info
        )
        
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "version": get_version(),
            "system_info": get_system_info(),
            "imports": {
                "status": check_imports(),
                "missing": get_missing_imports(),
                "available_features": get_available_features(),
            },
            "module_info": get_module_info(),
        }
        
        return diagnostics
        
    except Exception as e:
        logger.error(f"Error getting module diagnostics: {e}", exc_info=True)
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


def check_module_health() -> Dict[str, Any]:
    """
    Verificar salud del módulo.
    
    Returns:
        Diccionario con estado de salud del módulo.
    """
    health_status = {
        "status": "healthy",
        "checks": {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Check imports
    try:
        from .. import check_imports
        imports_status = check_imports()
        total = len(imports_status)
        available = sum(1 for v in imports_status.values() if v)
        rate = (available / total * 100) if total > 0 else 0.0
        
        health_status["checks"]["imports"] = {
            "status": "ok" if rate > 80 else "degraded",
            "availability_rate": rate,
            "total": total,
            "available": available,
        }
        
        if rate < 50:
            health_status["status"] = "unhealthy"
        elif rate < 80:
            health_status["status"] = "degraded"
            
    except Exception as e:
        health_status["checks"]["imports"] = {
            "status": "error",
            "error": str(e),
        }
        health_status["status"] = "unhealthy"
    
    # Check core components
    try:
        from ..server import MCPServer
        from ..connectors import ConnectorRegistry
        from ..security import MCPSecurityManager
        
        health_status["checks"]["core_components"] = {
            "status": "ok",
            "components": {
                "MCPServer": MCPServer is not None,
                "ConnectorRegistry": ConnectorRegistry is not None,
                "MCPSecurityManager": MCPSecurityManager is not None,
            },
        }
    except Exception as e:
        health_status["checks"]["core_components"] = {
            "status": "error",
            "error": str(e),
        }
        health_status["status"] = "unhealthy"
    
    return health_status


def get_dependency_tree() -> Dict[str, Any]:
    """
    Obtener árbol de dependencias del módulo.
    
    Returns:
        Diccionario con estructura de dependencias.
    """
    try:
        from .._imports import IMPORT_GROUPS
        
        tree = {
            "groups": {},
            "total_groups": len(IMPORT_GROUPS),
            "total_modules": sum(len(modules) for modules in IMPORT_GROUPS.values()),
        }
        
        for group_name, modules in IMPORT_GROUPS.items():
            tree["groups"][group_name] = {
                "modules": [
                    {
                        "path": module_path,
                        "symbols": symbols.split(",") if isinstance(symbols, str) else symbols,
                    }
                    for module_path, symbols in modules
                ],
                "module_count": len(modules),
            }
        
        return tree
        
    except Exception as e:
        logger.error(f"Error getting dependency tree: {e}", exc_info=True)
        return {"error": str(e)}


def validate_module_setup() -> Tuple[bool, List[str]]:
    """
    Validar configuración del módulo.
    
    Returns:
        Tupla (is_valid, errors) donde is_valid indica si la configuración es válida
        y errors es una lista de errores encontrados.
    """
    errors = []
    
    # Check core imports
    try:
        from ..server import MCPServer
        from ..connectors import ConnectorRegistry
    except ImportError as e:
        errors.append(f"Core imports failed: {e}")
    
    # Check import manager
    try:
        from .._import_manager import ImportManager
    except ImportError as e:
        errors.append(f"ImportManager not available: {e}")
    
    # Check utilities
    try:
        from ..utils.module_info import get_cached_imports_status
    except ImportError as e:
        errors.append(f"Module info utilities not available: {e}")
    
    return len(errors) == 0, errors


def get_performance_metrics() -> Dict[str, Any]:
    """
    Obtener métricas de performance del módulo.
    
    Returns:
        Diccionario con métricas de performance.
    """
    import time
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    return {
        "memory": {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
            "percent": process.memory_percent(),
        },
        "cpu": {
            "percent": process.cpu_percent(interval=0.1),
            "num_threads": process.num_threads(),
        },
        "timestamp": datetime.now().isoformat(),
    }


def generate_diagnostic_report() -> str:
    """
    Generar reporte de diagnóstico completo.
    
    Returns:
        String con reporte formateado.
    """
    lines = [
        "=" * 80,
        "MCP Server Diagnostic Report",
        "=" * 80,
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]
    
    # System info
    system_info = get_system_info()
    lines.extend([
        "System Information:",
        "-" * 80,
        f"Platform: {system_info['platform']}",
        f"Python: {system_info['python_version'].split()[0]}",
        f"Executable: {system_info['python_executable']}",
        "",
    ])
    
    # Module health
    health = check_module_health()
    lines.extend([
        "Module Health:",
        "-" * 80,
        f"Status: {health['status'].upper()}",
        "",
    ])
    
    for check_name, check_info in health.get("checks", {}).items():
        lines.append(f"  {check_name}:")
        if isinstance(check_info, dict):
            for key, value in check_info.items():
                if key != "status":
                    lines.append(f"    {key}: {value}")
        lines.append("")
    
    # Module info
    try:
        from .. import get_module_info
        module_info = get_module_info()
        lines.extend([
            "Module Information:",
            "-" * 80,
            f"Version: {module_info.get('version', 'unknown')}",
            f"Author: {module_info.get('author', 'unknown')}",
            "",
        ])
        
        if "statistics" in module_info:
            stats = module_info["statistics"]
            lines.extend([
                "Statistics:",
                f"  Total Components: {stats.get('total_components', 0)}",
                f"  Available: {stats.get('available_components', 0)}",
                f"  Missing: {stats.get('missing_components', 0)}",
                "",
            ])
    except Exception as e:
        lines.append(f"Error getting module info: {e}\n")
    
    # Validation
    is_valid, errors = validate_module_setup()
    lines.extend([
        "Validation:",
        "-" * 80,
        f"Status: {'VALID' if is_valid else 'INVALID'}",
        "",
    ])
    
    if errors:
        lines.append("Errors:")
        for error in errors:
            lines.append(f"  - {error}")
        lines.append("")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)


def print_diagnostic_report() -> None:
    """
    Imprimir reporte de diagnóstico en consola.
    """
    print(generate_diagnostic_report())


__all__ = [
    "get_system_info",
    "get_module_diagnostics",
    "check_module_health",
    "get_dependency_tree",
    "validate_module_setup",
    "get_performance_metrics",
    "generate_diagnostic_report",
    "print_diagnostic_report",
]

