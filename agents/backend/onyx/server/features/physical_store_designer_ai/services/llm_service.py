"""
LLM Service - Servicio unificado para integración con modelos de lenguaje
"""

import logging
import os
import json
from typing import Optional, List, Dict, Any
from openai import OpenAI
import openai

logger = logging.getLogger(__name__)


class LLMService:
    """Servicio para interactuar con modelos de lenguaje"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generar texto usando LLM"""
        if not self.client:
            logger.warning("No hay API key configurada, retornando respuesta por defecto")
            return self._default_response(prompt)
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generando con LLM: {e}")
            return self._default_response(prompt)
    
    async def generate_structured(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generar respuesta estructurada (JSON)"""
        if not self.client:
            return {}
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Agregar instrucciones para formato JSON
            json_prompt = f"{prompt}\n\nResponde SOLO con un objeto JSON válido, sin texto adicional."
            messages.append({"role": "user", "content": json_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,  # Menor temperatura para respuestas más consistentes
                response_format={"type": "json_object"} if response_format is None else response_format
            )
            
            content = response.choices[0].message.content
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Error parseando JSON: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error generando respuesta estructurada: {e}")
            return {}
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Generar respuesta en formato chat"""
        if not self.client:
            # Si no hay LLM, usar última respuesta del usuario como contexto
            user_messages = [m for m in messages if m.get("role") == "user"]
            if user_messages:
                return self._generate_simple_response(user_messages[-1].get("content", ""))
            return "No puedo procesar tu mensaje en este momento."
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error en chat: {e}")
            return self._generate_simple_response(messages[-1].get("content", "") if messages else "")
    
    def _default_response(self, prompt: str) -> str:
        """Respuesta por defecto cuando no hay LLM"""
        return "Lo siento, no puedo procesar tu solicitud en este momento. Por favor, intenta más tarde."
    
    def _generate_simple_response(self, user_message: str) -> str:
        """Generar respuesta simple sin LLM"""
        message_lower = user_message.lower()
        
        if any(word in message_lower for word in ["hola", "hi", "buenos"]):
            return "¡Hola! Estoy aquí para ayudarte a diseñar tu local. ¿Qué tipo de tienda quieres abrir?"
        
        if any(word in message_lower for word in ["gracias", "thanks"]):
            return "¡De nada! ¿Hay algo más en lo que pueda ayudarte?"
        
        return "Entiendo. ¿Puedes darme más detalles sobre lo que necesitas?"




