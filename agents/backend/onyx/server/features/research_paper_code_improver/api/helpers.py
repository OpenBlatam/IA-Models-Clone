"""
API Helpers - Funciones auxiliares para la API
===============================================
"""

import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(model_id: Optional[str], base_dir: str = "data/models") -> Optional[str]:
    """
    Resuelve la ruta completa del modelo desde su ID.
    
    Args:
        model_id: ID del modelo (opcional)
        base_dir: Directorio base donde se almacenan los modelos
        
    Returns:
        Ruta completa al modelo o None si no se proporciona model_id
    """
    if not model_id:
        return None
    return str(Path(base_dir) / model_id)


def create_code_improver(
    model_id: Optional[str] = None,
    vector_store=None,
    use_rag: bool = True,
    use_cache: bool = True,
    use_analyzer: bool = True
):
    """
    Factory function para crear instancias de CodeImprover.
    
    Args:
        model_id: ID del modelo a usar (opcional)
        vector_store: Instancia de VectorStore para RAG
        use_rag: Usar RAG para mejoras
        use_cache: Usar cache para mejoras
        use_analyzer: Usar analizador de código
        
    Returns:
        Instancia de CodeImprover configurada
    """
    from ..core.code_improver import CodeImprover
    
    model_path = resolve_model_path(model_id)
    
    return CodeImprover(
        model_path=model_path,
        vector_store=vector_store,
        use_rag=use_rag,
        use_cache=use_cache,
        use_analyzer=use_analyzer
    )

