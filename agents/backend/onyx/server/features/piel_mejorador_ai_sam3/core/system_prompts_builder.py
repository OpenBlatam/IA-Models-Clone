"""
System prompts builder for Piel Mejorador AI SAM3.
"""

from typing import Dict


class SystemPromptsBuilder:
    """
    Builds system prompts for different skin enhancement services.
    """
    
    @staticmethod
    def build_all_prompts() -> Dict[str, str]:
        """
        Build all system prompts for different services.
        
        Returns:
            Dictionary mapping service names to their system prompts
        """
        base_prompt = SystemPromptsBuilder._build_base_prompt()
        
        return {
            "default": base_prompt,
            "mejorar_imagen": base_prompt + SystemPromptsBuilder._get_image_enhancement_specialization(),
            "mejorar_video": base_prompt + SystemPromptsBuilder._get_video_enhancement_specialization(),
            "analisis_piel": base_prompt + SystemPromptsBuilder._get_skin_analysis_specialization(),
        }
    
    @staticmethod
    def _build_base_prompt() -> str:
        """Build base system prompt."""
        return """Eres un experto en mejoramiento de piel y procesamiento de imágenes/videos con IA.
Tienes conocimiento profundo de técnicas de mejoramiento de piel, realismo fotográfico,
y procesamiento de imágenes y videos.

Tu objetivo es mejorar la apariencia de la piel en imágenes y videos de manera realista,
manteniendo las características naturales y evitando efectos artificiales o exagerados.

Siempre proporcionas resultados de alta calidad que se ven naturales y profesionales.
Cuando proceses imágenes o videos, analiza cuidadosamente la textura de la piel,
iluminación, y características faciales para mantener la autenticidad."""
    
    @staticmethod
    def _get_image_enhancement_specialization() -> str:
        """Get specialization for image enhancement."""
        return """
Especialízate en mejoramiento de piel en imágenes estáticas. Siempre:
1. Analiza la textura y condición de la piel
2. Aplica mejoras graduales según el nivel especificado
3. Mantén características faciales naturales
4. Preserva la iluminación y sombras originales
5. Evita efectos de "over-processing" o artificialidad
6. Ajusta según el nivel de realismo solicitado"""
    
    @staticmethod
    def _get_video_enhancement_specialization() -> str:
        """Get specialization for video enhancement."""
        return """
Especialízate en mejoramiento de piel en videos. Siempre:
1. Mantén consistencia entre frames
2. Aplica mejoras suaves que no causen "flickering"
3. Preserva el movimiento natural
4. Analiza frame por frame para consistencia
5. Ajusta según el nivel de mejora y realismo especificado
6. Procesa de manera eficiente para mantener fluidez temporal"""
    
    @staticmethod
    def _get_skin_analysis_specialization() -> str:
        """Get specialization for skin analysis."""
        return """
Proporciona análisis detallado de la condición de la piel:
- Textura y poros
- Tono y uniformidad
- Imperfecciones y áreas a mejorar
- Características a preservar
- Recomendaciones de nivel de mejora apropiado"""




