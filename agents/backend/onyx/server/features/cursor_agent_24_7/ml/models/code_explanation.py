"""
Code Explanation Model - Wrapper para compatibilidad hacia atrás
================================================================

Este archivo mantiene compatibilidad hacia atrás importando desde
la implementación modular en code_explanation/.

Proporciona una interfaz simple para usuarios que importan directamente
desde ml.models.code_explanation en lugar de ml.models.code_explanation.model.
"""

import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .code_explanation.model import CodeExplanationModel
else:
    try:
        from .code_explanation import CodeExplanationModel
    except ImportError as e:
        logger.error(
            f"Failed to import CodeExplanationModel from code_explanation module: {e}",
            exc_info=True
        )
        # Intentar importar directamente desde model.py como fallback
        try:
            from .code_explanation.model import CodeExplanationModel
        except ImportError as e2:
            logger.error(
                f"Failed to import CodeExplanationModel from code_explanation.model: {e2}",
                exc_info=True
            )
            # Re-raise con mensaje más claro
            raise ImportError(
                "CodeExplanationModel not available. "
                "Ensure code_explanation module is properly installed."
            ) from e2

__all__ = ["CodeExplanationModel"]
