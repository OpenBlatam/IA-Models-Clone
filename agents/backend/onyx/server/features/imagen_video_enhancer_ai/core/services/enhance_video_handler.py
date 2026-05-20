"""
Enhance Video Handler
====================

Handler for video enhancement service.
"""

from typing import Dict, Any

from .base_handler import BaseServiceHandler
from ..service_handler import ServiceType, ServiceConfig


class EnhanceVideoHandler(BaseServiceHandler):
    """Handler for video enhancement service."""
    
    service_type = ServiceType.ENHANCE_VIDEO
    config = ServiceConfig.for_enhance_video()
    
    def build_prompt(self, parameters: Dict[str, Any]) -> str:
        """Build enhancement prompt for video."""
        file_path = parameters.get("file_path", "")
        enhancement_type = parameters.get("enhancement_type", "general")
        options = parameters.get("options", {})
        video_analysis = parameters.get("video_analysis", {})
        
        prompt = f"Analiza y mejora el siguiente video: {file_path}\n"
        
        if video_analysis:
            prompt += f"\nAnálisis del video:\n"
            prompt += f"- Resolución: {video_analysis.get('resolution', 'unknown')}\n"
            prompt += f"- FPS: {video_analysis.get('fps', 'unknown')}\n"
            prompt += f"- Duración: {video_analysis.get('duration_seconds', 0):.2f}s\n"
            
            if video_analysis.get('quality_issues'):
                prompt += f"- Problemas detectados: {len(video_analysis['quality_issues'])} frames con problemas\n"
            
            if video_analysis.get('recommendations'):
                prompt += f"- Recomendaciones: {', '.join(video_analysis['recommendations'])}\n"
        
        prompt += f"\nTipo de mejoramiento solicitado: {enhancement_type}\n"
        
        if options:
            prompt += f"Opciones específicas: {options}\n"
        
        prompt += "\nPor favor:\n"
        prompt += "1. Describe el estado actual del video\n"
        prompt += "2. Identifica áreas que necesitan mejoramiento\n"
        prompt += "3. Proporciona técnicas específicas recomendadas\n"
        prompt += "4. Explica los resultados esperados después del mejoramiento"
        
        return prompt




