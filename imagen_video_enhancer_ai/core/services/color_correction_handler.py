"""
Color Correction Handler
========================

Handler for color correction service.
"""

from typing import Dict, Any

from .base_handler import BaseServiceHandler
from ..service_handler import ServiceType, ServiceConfig


class ColorCorrectionHandler(BaseServiceHandler):
    """Handler for color correction service."""
    
    service_type = ServiceType.COLOR_CORRECTION
    config = ServiceConfig.for_color_correction()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build color correction prompt."""
        file_path = parameters.get("file_path", "")
        correction_type = parameters.get("correction_type", "auto")
        options = parameters.get("options", {})
        
        prompt = f"Corrige los colores de: {file_path}\n"
        prompt += f"Tipo de corrección: {correction_type}\n"
        
        if options:
            prompt += f"Opciones: {options}\n"
        
        prompt += "Proporciona una guía detallada de corrección de colores."
        
        return prompt




