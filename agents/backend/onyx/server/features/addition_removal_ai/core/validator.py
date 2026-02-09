"""
Change Validator - Validador de cambios para asegurar coherencia
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ChangeValidator:
    """Validador de cambios para verificar coherencia después de modificaciones"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el validador.

        Args:
            config: Configuración opcional
        """
        self.config = config or {}

    async def validate_change(
        self,
        original: str,
        modified: str,
        operation: str
    ) -> Dict[str, Any]:
        """
        Validar un cambio realizado.

        Args:
            original: Contenido original
            modified: Contenido modificado
            operation: Tipo de operación (add, remove, etc.)

        Returns:
            Diccionario con el resultado de la validación
        """
        validation = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "checks": {}
        }

        # Verificar que el contenido modificado no esté vacío (a menos que sea intencional)
        if not modified.strip() and original.strip():
            validation["warnings"].append("El contenido resultante está vacío")
            validation["checks"]["empty_result"] = False

        # Verificar que la longitud cambió según la operación
        if operation == "add":
            if len(modified) <= len(original):
                validation["warnings"].append("El contenido no parece haber aumentado")
                validation["checks"]["length_increase"] = False
            else:
                validation["checks"]["length_increase"] = True

        elif operation == "remove":
            if len(modified) >= len(original):
                validation["warnings"].append("El contenido no parece haber disminuido")
                validation["checks"]["length_decrease"] = False
            else:
                validation["checks"]["length_decrease"] = True

        # Verificar coherencia básica
        validation["checks"]["coherence"] = self._check_coherence(original, modified)

        # Si hay errores críticos, marcar como inválido
        if validation["errors"]:
            validation["valid"] = False

        return validation

    def _check_coherence(self, original: str, modified: str) -> bool:
        """
        Verificar coherencia básica entre original y modificado.

        Args:
            original: Contenido original
            modified: Contenido modificado

        Returns:
            True si es coherente, False en caso contrario
        """
        # Verificación básica - puede mejorarse
        # Por ahora, solo verificar que no se haya corrompido completamente
        if not modified:
            return False
        
        # Verificar que al menos parte del contenido original permanece
        # (excepto en casos de reemplazo completo)
        if len(modified) < len(original) * 0.1 and len(original) > 100:
            return False
        
        return True






