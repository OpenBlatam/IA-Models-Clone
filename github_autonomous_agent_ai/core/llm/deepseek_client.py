"""
DeepSeek LLM Client
===================

Cliente para interactuar con la API de DeepSeek.
"""

import logging
import httpx
from typing import Optional, Dict, Any, List
from ...config.settings import settings

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Cliente para la API de DeepSeek."""
    
    def __init__(self):
        self.api_key = settings.DEEPSEEK_API_KEY
        self.api_base_url = settings.DEEPSEEK_API_BASE_URL
        self.model = settings.DEEPSEEK_MODEL
        self.enabled = settings.LLM_ENABLED and bool(self.api_key)
        
        if not self.enabled:
            logger.warning("DeepSeek LLM está deshabilitado o no hay API key configurada")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Optional[str]:
        """
        Enviar un mensaje al modelo de DeepSeek.
        
        Args:
            messages: Lista de mensajes en formato [{"role": "user", "content": "..."}]
            temperature: Temperatura para la generación (0.0-1.0)
            max_tokens: Máximo de tokens a generar
            
        Returns:
            Respuesta del modelo o None si hay error
        """
        if not self.enabled:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.api_base_url}/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature or settings.LLM_TEMPERATURE,
                        "max_tokens": max_tokens or settings.LLM_MAX_TOKENS
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content")
                else:
                    logger.error(f"Error en DeepSeek API: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error comunicándose con DeepSeek: {e}")
            return None
    
    async def process_instruction(
        self,
        instruction: str,
        repository: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Procesar una instrucción usando DeepSeek para generar un plan de acción.
        
        Args:
            instruction: Instrucción del usuario
            repository: Repositorio en formato owner/repo
            context: Contexto adicional (archivos, estructura, etc.)
            
        Returns:
            Plan de acción estructurado
        """
        if not self.enabled:
            return {
                "type": "fallback",
                "message": "LLM deshabilitado, usando procesamiento básico"
            }
        
        system_prompt = """Eres un asistente experto en desarrollo de software que ayuda a procesar instrucciones para modificar repositorios de GitHub.

Analiza la instrucción del usuario y genera un plan de acción estructurado en formato JSON con la siguiente estructura:
{
    "action": "create_file|update_file|delete_file|create_branch|create_pr|read_file",
    "file_path": "ruta/del/archivo" (si aplica),
    "content": "contenido del archivo" (si aplica),
    "branch_name": "nombre-de-rama" (si aplica),
    "commit_message": "mensaje del commit",
    "pr_title": "título del PR" (si aplica),
    "pr_body": "descripción del PR" (si aplica),
    "reasoning": "explicación breve de la acción"
}

Responde SOLO con el JSON, sin texto adicional."""

        user_prompt = f"""Repositorio: {repository}
Instrucción: {instruction}

{f'Contexto adicional: {context}' if context else ''}

Genera el plan de acción en formato JSON."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = await self.chat(messages)
        
        if not response:
            return {
                "type": "fallback",
                "message": "No se pudo obtener respuesta de DeepSeek"
            }
        
        # Intentar parsear JSON de la respuesta
        try:
            import json
            # Limpiar la respuesta (puede tener markdown code blocks)
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.startswith("```"):
                cleaned_response = cleaned_response[3:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            plan = json.loads(cleaned_response)
            plan["llm_used"] = True
            plan["llm_response"] = response
            return plan
            
        except json.JSONDecodeError:
            logger.warning(f"No se pudo parsear JSON de DeepSeek: {response}")
            return {
                "type": "fallback",
                "message": "Respuesta de LLM no válida",
                "raw_response": response
            }
    
    async def generate_code(
        self,
        description: str,
        language: str = "python",
        context: Optional[str] = None
    ) -> Optional[str]:
        """
        Generar código basado en una descripción.
        
        Args:
            description: Descripción de lo que debe hacer el código
            language: Lenguaje de programación
            context: Contexto adicional (código existente, etc.)
            
        Returns:
            Código generado o None
        """
        if not self.enabled:
            return None
        
        prompt = f"""Genera código en {language} que:
{description}

{f'Contexto: {context}' if context else ''}

Responde SOLO con el código, sin explicaciones adicionales."""
        
        messages = [
            {"role": "system", "content": "Eres un experto programador. Genera código limpio y bien estructurado."},
            {"role": "user", "content": prompt}
        ]
        
        return await self.chat(messages, temperature=0.3)



