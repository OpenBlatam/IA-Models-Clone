"""
Denoise Handler
===============

Handler for denoising service.
"""

from typing import Dict, Any

from .base_handler import BaseServiceHandler
from ..service_handler import ServiceType, ServiceConfig


class DenoiseHandler(BaseServiceHandler):
    """Handler for denoise service."""
    
    service_type = ServiceType.DENOISE
    config = ServiceConfig.for_denoise()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build denoising prompt."""
        file_path = parameters.get("file_path", "")
        noise_level = parameters.get("noise_level", "medium")
        options = parameters.get("options", {})
        
        prompt = f"Reduce el ruido de: {file_path}\n"
        prompt += f"Nivel de ruido: {noise_level}\n"
        
        if options:
            prompt += f"Opciones: {options}\n"
        
        prompt += "Proporciona una guía detallada de reducción de ruido."
        
        return prompt




