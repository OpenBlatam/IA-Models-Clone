"""
Upscale Handler
===============

Handler for upscaling service.
"""

from typing import Dict, Any

from .base_handler import BaseServiceHandler
from ..service_handler import ServiceType, ServiceConfig


class UpscaleHandler(BaseServiceHandler):
    """Handler for upscale service."""
    
    service_type = ServiceType.UPSCALE
    config = ServiceConfig.for_upscale()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build upscaling prompt."""
        file_path = parameters.get("file_path", "")
        scale_factor = parameters.get("scale_factor", 2)
        options = parameters.get("options", {})
        
        prompt = f"Realiza upscaling de: {file_path}\n"
        prompt += f"Factor de escala: {scale_factor}x\n"
        
        if options:
            prompt += f"Opciones: {options}\n"
        
        prompt += "Proporciona una guía detallada de upscaling."
        
        return prompt




