from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import os
from pathlib import Path
    from . import ultra_performance_optimizers
    from . import optimized_video_ai
    from . import optimized_video_ai_ultra
    from . import mega_optimizer
    from . import speed_test
    from . import demo_optimizacion
    from . import performance_optimizer
    import logging
        from .demo_optimizacion import demo_optimizacion_completa
        from .speed_test import run_speed_test
        from .mega_optimizer import create_mega_optimizer as create_mega
from typing import Any, List, Dict, Optional
import asyncio
"""
OPTIMIZACIONES DE RENDIMIENTO Y ALGORITMOS AVANZADOS
===================================================

Optimizaciones de rendimiento y algoritmos avanzados

Estructura del módulo:
- ultra_performance_optimizers.py: Optimizadores ultra-avanzados con librerías especializadas
- optimized_video_ai.py: Sistema de video AI optimizado
- optimized_video_ai_ultra.py: Sistema ultra-optimizado de video AI
"""

# Importaciones automáticas

# Metadata del módulo
__module_name__ = "optimization"
__description__ = "Optimizaciones de rendimiento y algoritmos avanzados"
__version__ = "1.0.0"

# Path del módulo
MODULE_PATH = Path(__file__).parent

# Auto-discovery de archivos Python
__all__ = []
for file_path in MODULE_PATH.glob("*.py"):
    if file_path.name != "__init__.py":
        module_name = file_path.stem
        __all__.append(module_name)

def get_module_info():
    """Obtener información del módulo."""
    return {
        "name": __module_name__,
        "description": __description__,
        "version": __version__,
        "path": str(MODULE_PATH),
        "files": __all__
    }

def list_files():
    """Listar archivos en el módulo."""
    return [f.name for f in MODULE_PATH.glob("*.py")]

# Importaciones principales para facilitar el uso
try:
except ImportError as e:
    logging.warning(f"No se pudieron importar algunos módulos de optimization: {e}")

# Funciones de conveniencia
def get_available_optimizers():
    """Obtener lista de optimizadores disponibles."""
    optimizers = []
    
    # Optimizadores existentes
    for module_name in ["ultra_performance_optimizers", "optimized_video_ai", "optimized_video_ai_ultra"]:
        try:
            __import__(f".{module_name}", package=__name__)
            optimizers.append(module_name)
        except ImportError:
            pass
    
    # Nuevos optimizadores ultra-avanzados
    for module_name in ["mega_optimizer", "speed_test", "demo_optimizacion", "performance_optimizer"]:
        try:
            __import__(f".{module_name}", package=__name__)
            optimizers.append(module_name)
        except ImportError:
            pass
    
    return optimizers

# Función para ejecutar demo completo
async def run_optimization_demo():
    """Ejecutar demo completo de optimización."""
    try:
        await demo_optimizacion_completa()
    except ImportError:
        print("❌ Demo de optimización no disponible")

# Función para ejecutar speed test
async def run_speed_test():
    """Ejecutar speed test de optimizadores."""
    try:
        await run_speed_test()
    except ImportError:
        print("❌ Speed test no disponible")

# Función para crear mega optimizer
async def create_mega_optimizer():
    """Crear instancia del mega optimizer."""
    try:
        return await create_mega()
    except ImportError:
        print("❌ Mega Optimizer no disponible")
        return None 