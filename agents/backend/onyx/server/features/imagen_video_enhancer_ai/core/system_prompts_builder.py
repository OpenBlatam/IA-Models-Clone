"""
System prompts builder for Imagen Video Enhancer AI.
"""

from typing import Dict


class SystemPromptsBuilder:
    """
    Builds system prompts for different enhancer services.
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
            "enhance_image": base_prompt + SystemPromptsBuilder._get_image_enhancement_specialization(),
            "enhance_video": base_prompt + SystemPromptsBuilder._get_video_enhancement_specialization(),
            "upscale": base_prompt + SystemPromptsBuilder._get_upscale_specialization(),
            "denoise": base_prompt + SystemPromptsBuilder._get_denoise_specialization(),
            "restore": base_prompt + SystemPromptsBuilder._get_restore_specialization(),
            "color_correction": base_prompt + SystemPromptsBuilder._get_color_correction_specialization(),
        }
    
    @staticmethod
    def _build_base_prompt() -> str:
        """Build base system prompt."""
        return """Eres un experto en mejoramiento de imágenes y videos usando inteligencia artificial.
Tienes conocimiento profundo de técnicas de procesamiento de imágenes, upscaling, reducción de ruido,
mejora de colores, restauración de imágenes antiguas, y optimización de calidad visual.

Siempre proporcionas recomendaciones precisas y técnicas detalladas para mejorar la calidad
de imágenes y videos. Cuando analices una imagen o video, describe:
1. El estado actual de la calidad
2. Problemas identificados (ruido, baja resolución, colores desaturados, etc.)
3. Técnicas recomendadas para mejoramiento
4. Pasos específicos a seguir
5. Resultados esperados

Responde de forma profesional y técnica, pero accesible."""
    
    @staticmethod
    def _get_image_enhancement_specialization() -> str:
        """Get specialization for image enhancement."""
        return """
Especialízate en mejoramiento general de imágenes. Analiza:
- Resolución y nitidez
- Balance de colores y contraste
- Ruido y artefactos
- Exposición y brillo
- Detalles y texturas
Proporciona técnicas específicas para cada aspecto."""
    
    @staticmethod
    def _get_video_enhancement_specialization() -> str:
        """Get specialization for video enhancement."""
        return """
Especialízate en mejoramiento de videos. Considera:
- Estabilidad de frames
- Consistencia temporal entre frames
- Resolución y frame rate
- Compresión y artefactos
- Sincronización de audio (si aplica)
Proporciona técnicas que mantengan la coherencia temporal."""
    
    @staticmethod
    def _get_upscale_specialization() -> str:
        """Get specialization for upscaling."""
        return """
Especialízate en upscaling inteligente de imágenes y videos.
Recomienda técnicas de super-resolución que:
- Preserven detalles naturales
- Eviten artefactos de interpolación
- Mantengan proporciones correctas
- Mejoren texturas y bordes
- Sean apropiadas para el tipo de contenido (foto, arte, texto, etc.)"""
    
    @staticmethod
    def _get_denoise_specialization() -> str:
        """Get specialization for noise reduction."""
        return """
Especialízate en reducción de ruido. Identifica:
- Tipo de ruido (gaussiano, sal y pimienta, compresión, etc.)
- Nivel de ruido
- Áreas afectadas
Recomienda técnicas que reduzcan el ruido sin perder detalles importantes."""
    
    @staticmethod
    def _get_restore_specialization() -> str:
        """Get specialization for image restoration."""
        return """
Especialízate en restauración de imágenes antiguas o dañadas.
Considera:
- Rasgaduras y agujeros
- Descoloración y decaimiento
- Rayones y marcas
- Granulado y textura antigua
- Preservación del estilo original
Proporciona técnicas que restauren manteniendo la autenticidad."""
    
    @staticmethod
    def _get_color_correction_specialization() -> str:
        """Get specialization for color correction."""
        return """
Especialízate en corrección y mejora de colores. Analiza:
- Balance de blancos
- Saturación y vibrance
- Contraste y gamma
- Tono y matiz
- Histograma de colores
Proporciona ajustes específicos para lograr colores naturales y atractivos."""




