"""
Manual Prompt Builder
=====================

Constructor especializado de prompts para manuales tipo LEGO.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ManualPromptBuilder:
    """Constructor de prompts para manuales tipo LEGO."""
    
    CATEGORY_NAMES = {
        "plomeria": "Plomería",
        "techos": "Reparación de Techos",
        "carpinteria": "Carpintería",
        "electricidad": "Electricidad",
        "albanileria": "Albañilería",
        "pintura": "Pintura",
        "herreria": "Herrería",
        "jardineria": "Jardinería",
        "general": "Reparación General"
    }
    
    def __init__(self):
        """Inicializar constructor de prompts."""
        self._logger = logger
    
    def build(
        self,
        problem_description: str,
        category: str = "general",
        include_safety: bool = True,
        include_tools: bool = True,
        include_materials: bool = True
    ) -> str:
        """
        Construir prompt para generar manual tipo LEGO.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría del oficio
            include_safety: Incluir advertencias de seguridad
            include_tools: Incluir lista de herramientas
            include_materials: Incluir lista de materiales
        
        Returns:
            Prompt formateado
        """
        category_name = self.CATEGORY_NAMES.get(
            category.lower(),
            "Reparación General"
        )
        
        prompt = f"""Eres un experto en {category_name} y generas manuales paso a paso tipo LEGO (instrucciones visuales y claras).

PROBLEMA A RESOLVER:
{problem_description}

GENERA UN MANUAL PASO A PASO con el siguiente formato:

1. TÍTULO DEL MANUAL
   - Título claro y descriptivo del problema

2. DIAGNÓSTICO
   - Análisis del problema
   - Posibles causas
   - Verificación necesaria

"""
        
        if include_safety:
            prompt += self._build_safety_section()
        
        if include_tools:
            prompt += self._build_tools_section()
        
        if include_materials:
            prompt += self._build_materials_section()
        
        prompt += self._build_steps_section()
        prompt += self._build_footer()
        
        return prompt
    
    def _build_safety_section(self) -> str:
        """Construir sección de advertencias de seguridad."""
        return """3. ADVERTENCIAS DE SEGURIDAD ⚠️
   - Riesgos potenciales
   - Equipo de protección necesario
   - Precauciones importantes

"""
    
    def _build_tools_section(self) -> str:
        """Construir sección de herramientas."""
        return """4. HERRAMIENTAS NECESARIAS 🔧
   - Lista de herramientas requeridas
   - Herramientas alternativas si no tienes las principales

"""
    
    def _build_materials_section(self) -> str:
        """Construir sección de materiales."""
        return """5. MATERIALES NECESARIOS 📦
   - Lista de materiales y cantidades
   - Dónde conseguirlos
   - Costos aproximados

"""
    
    def _build_steps_section(self) -> str:
        """Construir sección de pasos."""
        return """6. PASOS DE LA REPARACIÓN (Formato LEGO)
   Para cada paso, incluye:
   - Número de paso
   - Descripción clara y concisa
   - Ilustración verbal (descripción de cómo se ve)
   - Tiempo estimado
   - Dificultad (Fácil/Media/Difícil)
   
   Ejemplo de formato:
   PASO 1: Preparación
   📋 Descripción: [Qué hacer]
   🎨 Visual: [Cómo se ve]
   ⏱️ Tiempo: [X minutos]
   📊 Dificultad: [Fácil/Media/Difícil]
   ⚠️ Precauciones: [Si aplica]

7. VERIFICACIÓN
   - Cómo verificar que el problema está resuelto
   - Pruebas a realizar

8. MANTENIMIENTO PREVENTIVO
   - Cómo evitar que vuelva a pasar
   - Mantenimiento recomendado

9. CUANDO LLAMAR A UN PROFESIONAL
   - Señales de que necesitas ayuda experta
   - Situaciones de emergencia

"""
    
    def _build_footer(self) -> str:
        """Construir footer del prompt."""
        return """IMPORTANTE:
- Usa lenguaje claro y simple
- Cada paso debe ser independiente y comprensible
- Incluye emojis para hacer más visual
- Sé específico con medidas, tiempos y materiales
- Considera diferentes niveles de experiencia
- Si el problema es complejo, recomienda llamar a un profesional

Genera el manual completo ahora:"""

