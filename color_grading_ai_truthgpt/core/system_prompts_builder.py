"""
System prompts builder for Color Grading AI TruthGPT.
"""

from typing import Dict


class SystemPromptsBuilder:
    """
    Builds system prompts for different Color Grading AI services.
    
    Responsibilities:
    - Build base system prompt
    - Build specialized prompts for each service
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
            "color_grading": base_prompt + SystemPromptsBuilder._get_grading_specialization(),
            "color_analysis": base_prompt + SystemPromptsBuilder._get_analysis_specialization(),
            "color_matching": base_prompt + SystemPromptsBuilder._get_matching_specialization(),
            "template_selection": base_prompt + SystemPromptsBuilder._get_template_specialization(),
        }
    
    @staticmethod
    def _build_base_prompt() -> str:
        """Build base system prompt."""
        return """Eres un experto en color grading profesional, similar a DaVinci Resolve.
Tienes conocimiento profundo de:
- Teoría del color y espacios de color (RGB, HSV, LAB, XYZ)
- Curvas de color y LUTs (Look-Up Tables)
- Balance de color, contraste, saturación, brillo
- Estilos cinematográficos y looks profesionales
- Análisis de histogramas y exposición
- Matching de color entre referencias

Siempre proporcionas análisis precisos y parámetros de color grading detallados.
Cuando analices imágenes o videos, incluye:
1. Análisis de histograma
2. Temperatura de color
3. Exposición y contraste
4. Distribución de colores
5. Recomendaciones de ajuste

Responde en español, de forma profesional pero accesible."""
    
    @staticmethod
    def _get_grading_specialization() -> str:
        """Get specialization for color grading."""
        return """
Especialízate en aplicar color grading profesional. Siempre proporciona:
1. Parámetros de color (brightness, contrast, saturation)
2. Balance de color (R, G, B)
3. Curvas de color cuando sea necesario
4. Ajustes de exposición y contraste
5. Recomendaciones de LUTs o templates"""
    
    @staticmethod
    def _get_analysis_specialization() -> str:
        """Get specialization for color analysis."""
        return """
Analiza imágenes y videos para extraer información de color:
- Histogramas por canal (R, G, B)
- Estadísticas de color (media, desviación, min, max)
- Temperatura de color estimada
- Análisis de exposición
- Distribución de colores dominantes"""
    
    @staticmethod
    def _get_matching_specialization() -> str:
        """Get specialization for color matching."""
        return """
Matching de color entre imágenes/videos de referencia:
- Analiza la imagen/video de referencia
- Calcula diferencias con el material fuente
- Genera parámetros de ajuste para lograr el look deseado
- Considera temperatura de color, exposición, y balance"""
    
    @staticmethod
    def _get_template_specialization() -> str:
        """Get specialization for template selection."""
        return """
Recomienda templates de color grading basado en:
- Contenido del material (cinematográfico, documental, etc.)
- Estilo deseado (vintage, moderno, cinematográfico)
- Análisis del material fuente
- Preferencias del usuario"""




