"""
Enhance Image Handler
====================

Handler for image enhancement service.
"""

from typing import Dict, Any

from .base_handler import BaseServiceHandler
from .base import BaseServiceHandlerMixin
from ..service_handler import ServiceType, ServiceConfig


class EnhanceImageHandler(BaseServiceHandler, BaseServiceHandlerMixin):
    """Handler for image enhancement service."""
    
    service_type = ServiceType.ENHANCE_IMAGE
    config = ServiceConfig.for_enhance_image()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build enhancement prompt for image."""
        file_path = parameters.get("file_path", "")
        enhancement_type = parameters.get("enhancement_type", "general")
        options = parameters.get("options", {})
        
        # Build base prompt
        prompt = self.build_base_prompt(
            file_path,
            "mejora de imagen",
            {
                "tipo_mejoramiento": enhancement_type,
                **options
            }
        )
        
        # Add specific instructions
        instructions = [
            "Describe el estado actual de la imagen (calidad, problemas, etc.)",
            "Identifica áreas que necesitan mejoramiento",
            "Proporciona técnicas específicas recomendadas",
            "Explica los resultados esperados después del mejoramiento"
        ]
        
        return self.add_instructions(prompt, instructions)

