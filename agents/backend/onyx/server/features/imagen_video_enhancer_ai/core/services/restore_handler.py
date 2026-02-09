"""
Restore Handler
==============

Handler for restoration service.
"""

from typing import Dict, Any

from .base_handler import BaseServiceHandler
from ..service_handler import ServiceType, ServiceConfig


class RestoreHandler(BaseServiceHandler):
    """Handler for restoration service."""
    
    service_type = ServiceType.RESTORE
    config = ServiceConfig.for_restore()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build restoration prompt."""
        file_path = parameters.get("file_path", "")
        damage_type = parameters.get("damage_type")
        options = parameters.get("options", {})
        
        prompt = f"Restaura la siguiente imagen: {file_path}\n"
        
        if damage_type:
            prompt += f"Tipo de daño: {damage_type}\n"
        
        if options:
            prompt += f"Opciones: {options}\n"
        
        prompt += "Proporciona una guía detallada de restauración."
        
        return prompt




