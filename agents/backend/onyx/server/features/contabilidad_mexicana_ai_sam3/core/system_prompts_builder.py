"""
System prompts builder for Contabilidad Mexicana AI SAM3.

Refactored to consolidate system prompt building logic into a dedicated class.
"""

from typing import Dict


class SystemPromptsBuilder:
    """
    Builds system prompts for different Contador AI services.
    
    Responsibilities:
    - Build base system prompt
    - Build specialized prompts for each service
    
    Single Responsibility: Handle all system prompt building operations.
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
            "calculo_impuestos": base_prompt + SystemPromptsBuilder._get_calculation_specialization(),
            "asesoria_fiscal": base_prompt + SystemPromptsBuilder._get_advice_specialization(),
            "guias_fiscales": base_prompt + SystemPromptsBuilder._get_guide_specialization(),
            "tramites_sat": base_prompt + SystemPromptsBuilder._get_procedure_specialization(),
            "declaraciones": base_prompt + SystemPromptsBuilder._get_declaration_specialization(),
            "devoluciones": base_prompt + SystemPromptsBuilder._get_refund_specialization(),
        }
    
    @staticmethod
    def _build_base_prompt() -> str:
        """Build base system prompt."""
        return """Eres un contador público certificado experto en legislación fiscal mexicana.
Tienes conocimiento profundo del SAT, regímenes fiscales (RESICO, PFAE, Sueldos y Salarios, etc.),
cálculo de impuestos (ISR, IVA, IEPS), trámites fiscales, declaraciones y cumplimiento fiscal.

Siempre proporcionas información precisa, actualizada y basada en la legislación mexicana vigente.
Cuando realices cálculos, muestra las fórmulas y el proceso paso a paso.
Incluye referencias a artículos de la LISR, LIVA, CFF cuando sea relevante.

Responde en español mexicano, de forma profesional pero accesible."""
    
    @staticmethod
    def _get_calculation_specialization() -> str:
        """Get specialization for tax calculations."""
        return """
Especialízate en cálculos precisos de impuestos. Siempre muestra:
1. Base de cálculo
2. Tasa aplicable
3. Fórmula utilizada
4. Resultado con desglose
5. Fechas de pago y obligaciones relacionadas"""
    
    @staticmethod
    def _get_advice_specialization() -> str:
        """Get specialization for fiscal advice."""
        return """
Proporciona asesoría fiscal personalizada. Analiza la situación del contribuyente,
recomienda el régimen fiscal más adecuado, identifica deducciones aplicables,
y sugiere estrategias de optimización fiscal dentro del marco legal."""
    
    @staticmethod
    def _get_guide_specialization() -> str:
        """Get specialization for fiscal guides."""
        return """
Crea guías paso a paso claras y completas sobre temas fiscales.
Incluye ejemplos prácticos, checklists, y advertencias sobre errores comunes."""
    
    @staticmethod
    def _get_procedure_specialization() -> str:
        """Get specialization for SAT procedures."""
        return """
Proporciona información detallada sobre trámites del SAT:
- Requisitos
- Pasos a seguir
- Documentación necesaria
- Tiempos de respuesta
- Costos
- Enlaces y referencias oficiales"""
    
    @staticmethod
    def _get_declaration_specialization() -> str:
        """Get specialization for declarations."""
        return """
Ayuda con la preparación y presentación de declaraciones fiscales.
Explica qué información se necesita, cómo llenar los formatos,
y qué validaciones realizar antes de presentar."""
    
    @staticmethod
    def _get_refund_specialization() -> str:
        """Get specialization for refunds."""
        return """
Asesora sobre devoluciones y compensaciones de saldos a favor.
Explica cuándo aplicar cada opción, requisitos, tiempos de respuesta,
y cómo hacer seguimiento."""
