"""
Combined Manual Generator
========================

Generador que combina texto e imágenes.
"""

import logging
from typing import Dict, Any, Optional

from .text_generator import TextManualGenerator
from .image_generator import ImageManualGenerator

logger = logging.getLogger(__name__)


class CombinedManualGenerator:
    """Generador que combina texto e imágenes."""
    
    def __init__(
        self,
        text_generator: TextManualGenerator,
        image_generator: ImageManualGenerator
    ):
        """
        Inicializar generador combinado.
        
        Args:
            text_generator: Generador de texto
            image_generator: Generador de imágenes
        """
        self.text_generator = text_generator
        self.image_generator = image_generator
        self._logger = logger
    
    async def generate(
        self,
        problem_description: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        category: str = "general",
        model: Optional[str] = None,
        include_safety: bool = True,
        include_tools: bool = True,
        include_materials: bool = True
    ) -> Dict[str, Any]:
        """
        Generar manual combinando imagen y texto.
        
        Args:
            problem_description: Descripción del problema
            image_path: Ruta al archivo de imagen (opcional)
            image_bytes: Bytes de la imagen (opcional)
            image_base64: Imagen en base64 (opcional)
            category: Categoría del oficio
            model: Modelo a usar
            include_safety: Incluir advertencias de seguridad
            include_tools: Incluir lista de herramientas
            include_materials: Incluir lista de materiales
        
        Returns:
            Manual generado
        """
        if image_path or image_bytes or image_base64:
            return await self.image_generator.generate(
                image_path=image_path,
                image_bytes=image_bytes,
                image_base64=image_base64,
                problem_description=problem_description,
                category=category,
                model=model,
                include_safety=include_safety,
                include_tools=include_tools,
                include_materials=include_materials
            )
        else:
            return await self.text_generator.generate(
                problem_description=problem_description,
                category=category,
                model=model,
                include_safety=include_safety,
                include_tools=include_tools,
                include_materials=include_materials
            )

