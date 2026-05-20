"""
Generador de Manuales Tipo LEGO
=================================

Genera manuales paso a paso tipo LEGO para oficios populares.
Soporta procesamiento de imágenes y texto.

Refactorizado para usar módulos especializados.
"""

import logging
from typing import Dict, Any, Optional

from ..infrastructure.openrouter.openrouter_client import OpenRouterClient
from ..config.settings import get_settings
from ..utils.category_detector import CategoryDetector
from ..utils.cache_manager import get_cache
from .generation.text_generator import TextManualGenerator
from .generation.image_generator import ImageManualGenerator
from .generation.combined_generator import CombinedManualGenerator

logger = logging.getLogger(__name__)


class ManualGenerator:
    """Generador de manuales paso a paso tipo LEGO."""
    
    def __init__(
        self,
        openrouter_client: Optional[OpenRouterClient] = None,
        use_cache: bool = True
    ):
        """
        Inicializar generador de manuales.
        
        Args:
            openrouter_client: Cliente OpenRouter (opcional, se crea uno si no se proporciona)
            use_cache: Usar cache para respuestas similares
        """
        self.settings = get_settings()
        self.client = openrouter_client or OpenRouterClient()
        self.use_cache = use_cache
        self.cache = get_cache() if use_cache else None
        self.category_detector = CategoryDetector()
        self._logger = logger
        
        self.text_generator = TextManualGenerator(
            client=self.client,
            category_detector=self.category_detector,
            cache=self.cache
        )
        
        self.image_generator = ImageManualGenerator(
            client=self.client,
            category_detector=self.category_detector,
            text_generator=self.text_generator
        )
        
        self.combined_generator = CombinedManualGenerator(
            text_generator=self.text_generator,
            image_generator=self.image_generator
        )
    
    
    async def generate_manual_from_text(
        self,
        problem_description: str,
        category: str = "general",
        model: Optional[str] = None,
        include_safety: bool = True,
        include_tools: bool = True,
        include_materials: bool = True
    ) -> Dict[str, Any]:
        """
        Generar manual desde descripción de texto.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría del oficio
            model: Modelo a usar (opcional)
            include_safety: Incluir advertencias de seguridad
            include_tools: Incluir lista de herramientas
            include_materials: Incluir lista de materiales
        
        Returns:
            Manual generado
        """
        return await self.text_generator.generate(
            problem_description=problem_description,
            category=category,
            model=model,
            include_safety=include_safety,
            include_tools=include_tools,
            include_materials=include_materials
        )
    
    async def generate_manual_from_image(
        self,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        problem_description: Optional[str] = None,
        category: Optional[str] = None,
        model: Optional[str] = None,
        include_safety: bool = True,
        include_tools: bool = True,
        include_materials: bool = True
    ) -> Dict[str, Any]:
        """
        Generar manual desde imagen.
        
        Args:
            image_path: Ruta al archivo de imagen
            image_bytes: Bytes de la imagen
            image_base64: Imagen en base64
            problem_description: Descripción adicional del problema
            category: Categoría del oficio (se detecta si no se proporciona)
            model: Modelo con visión a usar
            include_safety: Incluir advertencias de seguridad
            include_tools: Incluir lista de herramientas
            include_materials: Incluir lista de materiales
        
        Returns:
            Manual generado
        """
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
    
    async def generate_manual_combined(
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
        return await self.combined_generator.generate(
            problem_description=problem_description,
            image_path=image_path,
            image_bytes=image_bytes,
            image_base64=image_base64,
            category=category,
            model=model,
            include_safety=include_safety,
            include_tools=include_tools,
            include_materials=include_materials
        )

