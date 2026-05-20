"""
MCP Server - Model Context Protocol Integration Layer
======================================================

Servidor MCP que expone conectores estandarizados para:
- Sistema de archivos
- Bases de datos
- APIs externas

Proporciona una capa de integración unificada para que los modelos
puedan consultar fuentes autorizadas de contexto.

Versión: 2.2.0
Autor: Blatam Academy
Licencia: Propietaria
"""

import logging
from typing import TYPE_CHECKING, Dict, Any

logger = logging.getLogger(__name__)

# Importar constantes y funciones públicas desde módulo especializado
from ._public_api import (
    __version__,
    __author__,
    __license__,
    get_version,
    check_imports,
    get_missing_imports,
    get_available_features,
    get_module_info,
    get_diagnostics,
    check_health,
    validate_setup,
)

# Imports para TYPE_CHECKING - delegar a módulo especializado
if TYPE_CHECKING:
    from ._type_checking_imports import *  # noqa: F401, F403
else:
    # Imports reales - usar ImportManager para carga dinámica
    from ._import_manager import ImportManager
    from ._imports import get_all_symbols
    
    _import_namespace: Dict[str, Any] = {}
    _all_symbols = get_all_symbols()
    
    # Inicializar todos los símbolos como None
    for symbol in _all_symbols:
        _import_namespace[symbol] = None
    
    # Usar ImportManager para importar todos los módulos
    _import_manager = ImportManager(_import_namespace)
    _import_manager.import_all()
    
    # Exportar todos los símbolos al namespace del módulo
    globals().update(_import_namespace)
    
    # Limpiar variables temporales
    del _import_namespace, _all_symbols, _import_manager

# Exportar __all__ desde módulo especializado
from ._exports import __all__  # noqa: E402
