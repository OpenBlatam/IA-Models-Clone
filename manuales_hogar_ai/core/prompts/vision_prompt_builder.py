"""
Vision Prompt Builder
====================

Constructor especializado de prompts para análisis de imágenes.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VisionPromptBuilder:
    """Constructor de prompts para análisis de imágenes."""
    
    def __init__(self):
        """Inicializar constructor de prompts."""
        self._logger = logger
    
    def build(self, problem_description: Optional[str] = None) -> str:
        """
        Construir prompt para análisis de imagen.
        
        Args:
            problem_description: Descripción adicional del problema (opcional)
        
        Returns:
            Prompt para análisis de imagen
        """
        base_prompt = """Analiza esta imagen de un problema en el hogar/oficio.

"""
        
        if problem_description:
            base_prompt += f"DESCRIPCIÓN DEL PROBLEMA:\n{problem_description}\n\n"
        
        base_prompt += """Por favor, identifica:

1. QUÉ PROBLEMA SE OBSERVA
   - Descripción detallada del problema visible
   - Severidad (Leve/Moderada/Grave)
   - Urgencia (Baja/Media/Alta)

2. POSIBLES CAUSAS
   - Causas más probables basadas en lo que se ve
   - Factores contribuyentes

3. CATEGORÍA DEL OFICIO
   - ¿Es plomería, electricidad, carpintería, techos, etc.?

4. COMPLEJIDAD ESTIMADA
   - ¿Es algo que se puede hacer uno mismo o necesita profesional?

5. RECOMENDACIONES INICIALES
   - Primeros pasos a tomar
   - Qué verificar antes de empezar

Proporciona un análisis detallado y profesional."""
        
        return base_prompt

