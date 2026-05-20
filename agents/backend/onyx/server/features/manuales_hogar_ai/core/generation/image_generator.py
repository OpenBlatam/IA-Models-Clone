"""
Image Manual Generator
=====================

Generador especializado de manuales desde imágenes.
"""

import logging
import asyncio
from typing import Dict, Any, Optional

from ...infrastructure.openrouter.openrouter_client import OpenRouterClient
from ...config.settings import get_settings
from ...utils.category_detector import CategoryDetector
from ..prompts.vision_prompt_builder import VisionPromptBuilder
from .text_generator import TextManualGenerator

logger = logging.getLogger(__name__)


class ImageManualGenerator:
    """Generador de manuales desde imágenes."""
    
    def __init__(
        self,
        client: OpenRouterClient,
        category_detector: CategoryDetector,
        text_generator: TextManualGenerator,
        vision_prompt_builder: Optional[VisionPromptBuilder] = None
    ):
        """
        Inicializar generador de imágenes.
        
        Args:
            client: Cliente OpenRouter
            category_detector: Detector de categorías
            text_generator: Generador de texto
            vision_prompt_builder: Constructor de prompts de visión (opcional)
        """
        self.settings = get_settings()
        self.client = client
        self.category_detector = category_detector
        self.text_generator = text_generator
        self.vision_prompt_builder = vision_prompt_builder or VisionPromptBuilder()
        self._logger = logger
    
    async def generate(
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
        try:
            model = model or self.settings.vision_model
            
            vision_prompt = self.vision_prompt_builder.build(problem_description)
            
            self._logger.info("Analizando imagen del problema...")
            
            analysis_response = await asyncio.wait_for(
                self.client.generate_with_vision(
                    prompt=vision_prompt,
                    image_path=image_path,
                    image_bytes=image_bytes,
                    image_base64=image_base64,
                    model=model,
                    max_tokens=2000,
                    temperature=0.5
                ),
                timeout=60.0
            )
            
            analysis_text = analysis_response['choices'][0]['message']['content']
            
            if not category:
                category = self._detect_category_from_analysis(analysis_text)
            
            description_parts = [f"ANÁLISIS DE LA IMAGEN:\n{analysis_text}\n"]
            if problem_description:
                description_parts.append(f"DESCRIPCIÓN ADICIONAL DEL USUARIO:\n{problem_description}\n")
            description_parts.append("\nBasándote en el análisis de la imagen y la descripción, genera el manual de reparación.")
            full_description = "\n".join(description_parts)
            
            manual_result = await self.text_generator.generate(
                problem_description=full_description,
                category=category or "general",
                model=model,
                include_safety=include_safety,
                include_tools=include_tools,
                include_materials=include_materials
            )
            
            if manual_result["success"]:
                manual_result["image_analysis"] = analysis_text
                manual_result["detected_category"] = category
            
            return manual_result
        
        except Exception as e:
            self._logger.error(f"Error generando manual desde imagen: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "manual": None
            }
    
    def _detect_category_from_analysis(self, analysis_text: str) -> str:
        """
        Detectar categoría del oficio desde el análisis.
        
        Args:
            analysis_text: Texto del análisis
        
        Returns:
            Categoría detectada
        """
        category, confidence = self.category_detector.detect_category(analysis_text)
        if confidence > 0.3:
            self._logger.info(f"Categoría detectada: {category} (confianza: {confidence:.2f})")
            return category
        self._logger.warning(f"Categoría con baja confianza: {category} ({confidence:.2f}), usando 'general'")
        return "general"

