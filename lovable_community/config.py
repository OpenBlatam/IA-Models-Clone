"""
Configuración para la comunidad Lovable (backward compatibility)

Este archivo mantiene compatibilidad hacia atrás importando desde el módulo config/.
La configuración está ahora organizada en secciones modulares para mejor mantenibilidad.

Para nuevas importaciones, use:
    from .config import Settings, settings
"""

# Import Settings from the modular structure for backward compatibility
from .config import Settings, settings

# Re-export for backward compatibility
__all__ = ["Settings", "settings"]
