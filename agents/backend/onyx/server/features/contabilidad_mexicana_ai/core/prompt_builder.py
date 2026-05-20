"""
Prompt builder for Contador AI.

Refactored to consolidate prompt building logic into a dedicated class.
"""

from typing import Dict, Any, Optional


class PromptBuilder:
    """
    Builds prompts for different Contador AI services.
    
    Responsibilities:
    - Build prompts for tax calculations
    - Build prompts for fiscal advice
    - Build prompts for guides
    - Build prompts for SAT procedures
    - Build prompts for declarations
    
    Single Responsibility: Handle all prompt building operations.
    """
    
    @staticmethod
    def build_calculation_prompt(
        regimen: str,
        tipo_impuesto: str,
        datos: Dict[str, Any]
    ) -> str:
        """
        Build prompt for tax calculation.
        
        Args:
            regimen: Fiscal regime
            tipo_impuesto: Tax type
            datos: Input data
            
        Returns:
            Formatted prompt
        """
        datos_str = PromptBuilder._format_data(datos)
        
        return f"""Calcula el {tipo_impuesto} para un contribuyente en régimen {regimen}.

Datos proporcionados:
{datos_str}

Proporciona:
1. Base de cálculo detallada
2. Tasa aplicable según el régimen
3. Cálculo paso a paso con fórmula
4. Resultado final
5. Fechas de pago y obligaciones
6. Recomendaciones si aplica"""
    
    @staticmethod
    def build_advice_prompt(
        pregunta: str,
        contexto: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for fiscal advice.
        
        Args:
            pregunta: Question or fiscal situation
            contexto: Additional context
            
        Returns:
            Formatted prompt
        """
        context_str = ""
        if contexto:
            context_str = f"\n\nContexto adicional:\n{PromptBuilder._format_data(contexto)}"
        
        return f"""Proporciona asesoría fiscal sobre la siguiente situación:

{pregunta}{context_str}

Proporciona:
1. Análisis de la situación
2. Opciones disponibles según la legislación
3. Recomendaciones específicas
4. Consideraciones importantes
5. Pasos a seguir si aplica"""
    
    @staticmethod
    def build_guide_prompt(
        tema: str,
        nivel_detalle: str = "completo"
    ) -> str:
        """
        Build prompt for fiscal guide.
        
        Args:
            tema: Fiscal topic
            nivel_detalle: Detail level
            
        Returns:
            Formatted prompt
        """
        return f"""Crea una guía fiscal {nivel_detalle} sobre: {tema}

La guía debe incluir:
1. Introducción y contexto
2. Conceptos clave
3. Pasos detallados
4. Ejemplos prácticos
5. Checklist de verificación
6. Errores comunes a evitar
7. Referencias legales relevantes
8. Recursos adicionales"""
    
    @staticmethod
    def build_procedure_prompt(
        tipo_tramite: str,
        detalles: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for SAT procedure information.
        
        Args:
            tipo_tramite: Procedure type
            detalles: Additional details
            
        Returns:
            Formatted prompt
        """
        detalles_str = ""
        if detalles:
            detalles_str = f"\n\nDetalles específicos:\n{PromptBuilder._format_data(detalles)}"
        
        return f"""Proporciona información completa sobre el trámite: {tipo_tramite}{detalles_str}

Incluye:
1. Descripción del trámite
2. Requisitos necesarios
3. Documentación requerida
4. Pasos detallados del proceso
5. Tiempos de respuesta estimados
6. Costos si aplica
7. Enlaces oficiales relevantes
8. Consejos y recomendaciones"""
    
    @staticmethod
    def build_declaration_prompt(
        tipo_declaracion: str,
        periodo: str,
        datos: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build prompt for declaration assistance.
        
        Args:
            tipo_declaracion: Declaration type
            periodo: Fiscal period
            datos: Taxpayer data
            
        Returns:
            Formatted prompt
        """
        datos_str = ""
        if datos:
            datos_str = f"\n\nDatos del contribuyente:\n{PromptBuilder._format_data(datos)}"
        
        return f"""Ayuda a preparar una declaración {tipo_declaracion} para el período {periodo}.{datos_str}

Proporciona:
1. Información necesaria para llenar la declaración
2. Secciones y campos importantes
3. Cómo calcular cada concepto
4. Validaciones a realizar
5. Errores comunes y cómo evitarlos
6. Pasos para presentar la declaración
7. Qué hacer después de presentar"""
    
    @staticmethod
    def build_comparison_prompt(
        regimenes: list,
        datos: Dict[str, Any]
    ) -> str:
        """
        Build prompt for regime comparison.
        
        Args:
            regimenes: List of fiscal regimes to compare
            datos: Taxpayer data
            
        Returns:
            Formatted prompt
        """
        datos_str = PromptBuilder._format_data(datos)
        
        return f"""Compara los siguientes regímenes fiscales para un contribuyente:

Regímenes a comparar: {', '.join(regimenes)}

Datos del contribuyente:
{datos_str}

Proporciona:
1. Comparación de carga fiscal para cada régimen
2. Ventajas y desventajas de cada uno
3. Recomendación basada en los datos proporcionados
4. Consideraciones importantes para cada régimen
5. Pasos para cambiar de régimen si aplica"""
    
    @staticmethod
    def _format_data(data: Dict[str, Any]) -> str:
        """Format data dictionary for prompts."""
        formatted = []
        for key, value in data.items():
            formatted.append(f"- {key}: {value}")
        return "\n".join(formatted)

