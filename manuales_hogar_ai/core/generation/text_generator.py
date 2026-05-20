"""
Text Manual Generator
=====================

Generador especializado de manuales desde texto.
"""

import logging
from typing import Dict, Any, Optional

from ...infrastructure.openrouter.openrouter_client import OpenRouterClient
from ...config.settings import get_settings
from ...utils.category_detector import CategoryDetector
from ...utils.cache_manager import CacheManager
from ..prompts.manual_prompt_builder import ManualPromptBuilder

logger = logging.getLogger(__name__)


class TextManualGenerator:
    """Generador de manuales desde texto."""
    
    def __init__(
        self,
        client: OpenRouterClient,
        category_detector: CategoryDetector,
        cache: Optional[CacheManager] = None,
        prompt_builder: Optional[ManualPromptBuilder] = None
    ):
        """
        Inicializar generador de texto.
        
        Args:
            client: Cliente OpenRouter
            category_detector: Detector de categorías
            cache: Cache manager (opcional)
            prompt_builder: Constructor de prompts (opcional)
        """
        self.settings = get_settings()
        self.client = client
        self.category_detector = category_detector
        self.cache = cache
        self.prompt_builder = prompt_builder or ManualPromptBuilder()
        self._logger = logger
    
    async def generate(
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
        try:
            cache_key_params = {
                "include_safety": include_safety,
                "include_tools": include_tools,
                "include_materials": include_materials,
                "model": model or self.settings.default_model
            }
            
            if self.cache:
                cached_result = self.cache.get(
                    problem_description,
                    category,
                    **cache_key_params
                )
                if cached_result:
                    self._logger.info("Resultado obtenido del cache")
                    return cached_result
            
            model = model or self.settings.default_model
            
            if category == "general":
                category = self._detect_category(problem_description)
            
            prompt = self.prompt_builder.build(
                problem_description=problem_description,
                category=category,
                include_safety=include_safety,
                include_tools=include_tools,
                include_materials=include_materials
            )
            
            self._logger.info(f"Generando manual para categoría: {category}")
            
            response = await self.client.generate_text(
                prompt=prompt,
                model=model,
                max_tokens=self.settings.max_tokens,
                temperature=self.settings.temperature
            )
            
            manual_text = response['choices'][0]['message']['content']
            
            result = {
                "success": True,
                "manual": manual_text,
                "category": category,
                "model_used": model,
                "tokens_used": response.get('usage', {}).get('total_tokens', 0),
                "format": "lego"
            }
            
            if self.cache:
                self.cache.set(
                    problem_description,
                    category,
                    result,
                    **cache_key_params
                )
            
            return result
        
        except Exception as e:
            self._logger.error(f"Error generando manual desde texto: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "manual": None
            }
    
    def _detect_category(self, text: str) -> str:
        """
        Detectar categoría del texto.
        
        Args:
            text: Texto a analizar
        
        Returns:
            Categoría detectada
        """
        detected_category, confidence = self.category_detector.detect_category(text)
        if confidence > 0.3:
            self._logger.info(f"Categoría auto-detectada: {detected_category} (confianza: {confidence:.2f})")
            return detected_category
        self._logger.debug(f"Categoría no detectada con suficiente confianza: {confidence:.2f}")
        return "general"

