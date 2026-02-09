"""
Prompt builder for Piel Mejorador AI SAM3.
"""

from typing import Dict, Any, Optional


class PromptBuilder:
    """
    Builds prompts for different skin enhancement services.
    """
    
    @staticmethod
    def build_image_enhancement_prompt(
        file_path: str,
        enhancement_level: str = "medium",
        realism_level: Optional[float] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build prompt for image enhancement.
        
        Args:
            file_path: Path to image file
            enhancement_level: Level of enhancement (low, medium, high, ultra)
            realism_level: Optional custom realism level (0.0 to 1.0)
            custom_instructions: Optional custom instructions
            
        Returns:
            Formatted prompt
        """
        level_desc = {
            "low": "mejoras sutiles y naturales",
            "medium": "mejoras moderadas manteniendo realismo",
            "high": "mejoras significativas con alto realismo",
            "ultra": "mejoras máximas con realismo fotográfico perfecto"
        }
        
        enhancement_desc = level_desc.get(enhancement_level, level_desc["medium"])
        
        prompt = f"""Mejora la piel en esta imagen con nivel de mejora '{enhancement_level}' ({enhancement_desc}).

Archivo: {file_path}

Requisitos:
- Aplicar mejoras según el nivel '{enhancement_level}'
- Mantener características faciales naturales
- Preservar iluminación y sombras originales
- Evitar efectos artificiales o exagerados"""
        
        if realism_level is not None:
            prompt += f"\n- Nivel de realismo: {realism_level:.2f} (0.0 = natural, 1.0 = fotográfico perfecto)"
        
        if custom_instructions:
            prompt += f"\n\nInstrucciones adicionales: {custom_instructions}"
        
        prompt += "\n\nProporciona una descripción detallada de las mejoras aplicadas."
        
        return prompt
    
    @staticmethod
    def build_video_enhancement_prompt(
        file_path: str,
        enhancement_level: str = "medium",
        realism_level: Optional[float] = None,
        custom_instructions: Optional[str] = None
    ) -> str:
        """
        Build prompt for video enhancement.
        
        Args:
            file_path: Path to video file
            enhancement_level: Level of enhancement (low, medium, high, ultra)
            realism_level: Optional custom realism level (0.0 to 1.0)
            custom_instructions: Optional custom instructions
            
        Returns:
            Formatted prompt
        """
        level_desc = {
            "low": "mejoras sutiles y naturales",
            "medium": "mejoras moderadas manteniendo realismo",
            "high": "mejoras significativas con alto realismo",
            "ultra": "mejoras máximas con realismo fotográfico perfecto"
        }
        
        enhancement_desc = level_desc.get(enhancement_level, level_desc["medium"])
        
        prompt = f"""Mejora la piel en este video con nivel de mejora '{enhancement_level}' ({enhancement_desc}).

Archivo: {file_path}

Requisitos:
- Aplicar mejoras según el nivel '{enhancement_level}'
- Mantener consistencia entre frames
- Preservar movimiento natural
- Evitar flickering o cambios bruscos
- Mantener características faciales naturales"""
        
        if realism_level is not None:
            prompt += f"\n- Nivel de realismo: {realism_level:.2f} (0.0 = natural, 1.0 = fotográfico perfecto)"
        
        if custom_instructions:
            prompt += f"\n\nInstrucciones adicionales: {custom_instructions}"
        
        prompt += "\n\nProporciona una descripción detallada de las mejoras aplicadas frame por frame."
        
        return prompt
    
    @staticmethod
    def build_skin_analysis_prompt(
        file_path: str,
        file_type: str = "image"
    ) -> str:
        """
        Build prompt for skin analysis.
        
        Args:
            file_path: Path to image or video file
            file_type: Type of file (image or video)
            
        Returns:
            Formatted prompt
        """
        return f"""Analiza la condición de la piel en este {file_type}.

Archivo: {file_path}

Proporciona un análisis detallado que incluya:
1. Textura y poros
2. Tono y uniformidad
3. Imperfecciones visibles
4. Características a preservar
5. Recomendación de nivel de mejora apropiado (low, medium, high, ultra)
6. Nivel de realismo recomendado (0.0 a 1.0)

Sé específico y detallado en tu análisis."""




