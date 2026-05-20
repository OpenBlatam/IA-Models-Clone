"""
AI Helpers
==========

Utilidades para mejorar las interacciones con IA.
"""

import json
import logging
from typing import Dict, Any, Optional, List
import re

logger = logging.getLogger(__name__)


class AIHelper:
    """Utilidades para IA."""
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """
        Extraer JSON del texto de respuesta de IA.
        
        Args:
            text: Texto de respuesta
        
        Returns:
            Diccionario JSON o None
        """
        # Buscar bloques JSON
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # Intentar parsear todo el texto
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        return None
    
    @staticmethod
    def create_structured_prompt(
        system_role: str,
        user_request: str,
        context: Optional[Dict[str, Any]] = None,
        format_instructions: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Crear prompt estructurado para conversación.
        
        Args:
            system_role: Rol del sistema
            user_request: Solicitud del usuario
            context: Contexto adicional
            format_instructions: Instrucciones de formato
        
        Returns:
            Lista de mensajes para la API
        """
        messages = [
            {"role": "system", "content": system_role}
        ]
        
        user_content = user_request
        
        if context:
            user_content += f"\n\nContexto:\n{json.dumps(context, indent=2, ensure_ascii=False)}"
        
        if format_instructions:
            user_content += f"\n\nFormato requerido:\n{format_instructions}"
        
        messages.append({"role": "user", "content": user_content})
        
        return messages
    
    @staticmethod
    def parse_ai_response(
        response: Dict[str, Any],
        expected_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Parsear respuesta de IA de forma robusta.
        
        Args:
            response: Respuesta de la API
            expected_fields: Campos esperados
        
        Returns:
            Datos parseados
        """
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        
        # Intentar extraer JSON
        parsed_data = AIHelper.extract_json_from_text(content)
        
        if parsed_data:
            # Validar campos esperados
            if expected_fields:
                missing_fields = [f for f in expected_fields if f not in parsed_data]
                if missing_fields:
                    logger.warning(f"Missing expected fields: {missing_fields}")
            
            return parsed_data
        
        # Si no hay JSON, crear estructura básica
        return {
            "raw_content": content,
            "parsed": False
        }
    
    @staticmethod
    def improve_prompt_for_wardrobe(
        event_data: Dict[str, Any],
        wardrobe_items: List[Dict[str, Any]],
        protocols: List[Dict[str, Any]]
    ) -> str:
        """
        Mejorar prompt para recomendaciones de vestimenta.
        
        Args:
            event_data: Datos del evento
            wardrobe_items: Items disponibles
            protocols: Protocolos aplicables
        
        Returns:
            Prompt mejorado
        """
        prompt = f"""Eres un estilista profesional especializado en vestimenta para artistas. Tu tarea es recomendar el outfit perfecto.

EVENTO:
- Título: {event_data.get('title', 'N/A')}
- Tipo: {event_data.get('event_type', 'N/A')}
- Fecha y hora: {event_data.get('start_time', 'N/A')}
- Ubicación: {event_data.get('location', 'No especificada')}
- Descripción: {event_data.get('description', 'N/A')}

PROTOCOLOS APLICABLES:
{json.dumps(protocols, indent=2, ensure_ascii=False) if protocols else 'Ninguno especificado'}

ITEMS DISPONIBLES EN EL GUARDARROPA:
{json.dumps(wardrobe_items[:30], indent=2, ensure_ascii=False) if wardrobe_items else 'No hay items disponibles'}

INSTRUCCIONES:
1. Analiza el tipo de evento y determina el código de vestimenta apropiado
2. Considera los protocolos aplicables
3. Selecciona items específicos del guardarropa disponible
4. Proporciona razonamiento claro para tu recomendación
5. Considera el clima si es relevante
6. Sugiere alternativas si es posible

RESPONDE EN FORMATO JSON CON:
{{
    "dress_code": "código_de_vestimenta",
    "recommended_items": ["id1", "id2", ...],
    "reasoning": "explicación detallada",
    "weather_considerations": "consideraciones del clima",
    "alternatives": ["alternativa1", "alternativa2"]
}}"""
        
        return prompt
    
    @staticmethod
    def improve_prompt_for_compliance(
        event_data: Dict[str, Any],
        protocols: List[Dict[str, Any]]
    ) -> str:
        """
        Mejorar prompt para verificación de cumplimiento.
        
        Args:
            event_data: Datos del evento
            protocols: Protocolos a verificar
        
        Returns:
            Prompt mejorado
        """
        prompt = f"""Eres un auditor profesional de protocolos para artistas. Evalúa el cumplimiento de protocolos.

EVENTO:
- Título: {event_data.get('title', 'N/A')}
- Tipo: {event_data.get('event_type', 'N/A')}
- Fecha: {event_data.get('start_time', 'N/A')}
- Descripción: {event_data.get('description', 'N/A')}

PROTOCOLOS A VERIFICAR:
{json.dumps(protocols, indent=2, ensure_ascii=False)}

INSTRUCCIONES:
1. Evalúa cada protocolo individualmente
2. Identifica cualquier violación potencial
3. Proporciona recomendaciones específicas
4. Sé objetivo y constructivo

RESPONDE EN FORMATO JSON CON:
{{
    "overall_compliant": true/false,
    "protocol_checks": [
        {{
            "protocol_id": "id",
            "compliant": true/false,
            "violations": ["violación1", "violación2"],
            "notes": "notas adicionales"
        }}
    ],
    "recommendations": "recomendaciones generales"
}}"""
        
        return prompt




