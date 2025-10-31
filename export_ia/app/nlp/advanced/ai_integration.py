"""
AI Integration Manager - Integración con modelos de IA externos
"""

import logging
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)


class AIIntegrationManager:
    """Gestor de integración con modelos de IA externos."""
    
    def __init__(self):
        self._initialized = False
        self.api_keys = {}
        self.endpoints = {}
        self.session = None
        
        # Configuración de APIs
        self.api_configs = {
            "openai": {
                "base_url": "https://api.openai.com/v1",
                "models": {
                    "gpt-3.5-turbo": "chat/completions",
                    "gpt-4": "chat/completions",
                    "text-davinci-003": "completions",
                    "text-embedding-ada-002": "embeddings"
                }
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com/v1",
                "models": {
                    "claude-3-sonnet": "messages",
                    "claude-3-haiku": "messages"
                }
            },
            "cohere": {
                "base_url": "https://api.cohere.ai/v1",
                "models": {
                    "command": "generate",
                    "embed-english-v2.0": "embed"
                }
            },
            "huggingface": {
                "base_url": "https://api-inference.huggingface.co/models",
                "models": {
                    "microsoft/DialoGPT-medium": "conversational",
                    "facebook/blenderbot-400M-distill": "conversational"
                }
            }
        }
    
    async def initialize(self):
        """Inicializar el gestor de integración."""
        if not self._initialized:
            try:
                logger.info("Inicializando AIIntegrationManager")
                
                # Cargar API keys desde variables de entorno
                self._load_api_keys()
                
                # Crear sesión HTTP
                self.session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=30)
                )
                
                self._initialized = True
                logger.info("AIIntegrationManager inicializado exitosamente")
                
            except Exception as e:
                logger.error(f"Error al inicializar AIIntegrationManager: {e}")
                raise
    
    async def shutdown(self):
        """Cerrar el gestor de integración."""
        if self._initialized:
            try:
                # Cerrar sesión HTTP
                if self.session:
                    await self.session.close()
                
                self._initialized = False
                logger.info("AIIntegrationManager cerrado")
                
            except Exception as e:
                logger.error(f"Error al cerrar AIIntegrationManager: {e}")
    
    def _load_api_keys(self):
        """Cargar API keys desde variables de entorno."""
        api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "cohere": os.getenv("COHERE_API_KEY"),
            "huggingface": os.getenv("HUGGINGFACE_API_KEY")
        }
        
        for provider, key in api_keys.items():
            if key:
                self.api_keys[provider] = key
                logger.info(f"API key cargada para {provider}")
            else:
                logger.warning(f"API key no encontrada para {provider}")
    
    async def chat_completion(self, messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo", provider: str = "openai") -> Dict[str, Any]:
        """Generar completación de chat."""
        if not self._initialized:
            await self.initialize()
        
        if provider not in self.api_keys:
            raise ValueError(f"API key no disponible para {provider}")
        
        try:
            if provider == "openai":
                return await self._openai_chat_completion(messages, model)
            elif provider == "anthropic":
                return await self._anthropic_chat_completion(messages, model)
            else:
                raise ValueError(f"Proveedor no soportado: {provider}")
                
        except Exception as e:
            logger.error(f"Error en chat completion: {e}")
            raise
    
    async def _openai_chat_completion(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Completación de chat con OpenAI."""
        url = f"{self.api_configs['openai']['base_url']}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['openai']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "provider": "openai",
                    "model": model,
                    "response": result["choices"][0]["message"]["content"],
                    "usage": result.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Error OpenAI: {response.status} - {error_text}")
    
    async def _anthropic_chat_completion(self, messages: List[Dict[str, str]], model: str) -> Dict[str, Any]:
        """Completación de chat con Anthropic."""
        url = f"{self.api_configs['anthropic']['base_url']}/messages"
        
        headers = {
            "x-api-key": self.api_keys["anthropic"],
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        # Convertir mensajes al formato de Anthropic
        system_message = None
        user_messages = []
        
        for message in messages:
            if message["role"] == "system":
                system_message = message["content"]
            else:
                user_messages.append(message)
        
        data = {
            "model": model,
            "max_tokens": 1000,
            "messages": user_messages
        }
        
        if system_message:
            data["system"] = system_message
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "provider": "anthropic",
                    "model": model,
                    "response": result["content"][0]["text"],
                    "usage": result.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Error Anthropic: {response.status} - {error_text}")
    
    async def generate_text(self, prompt: str, model: str = "text-davinci-003", provider: str = "openai") -> Dict[str, Any]:
        """Generar texto con modelo de IA."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if provider == "openai":
                return await self._openai_text_generation(prompt, model)
            elif provider == "cohere":
                return await self._cohere_text_generation(prompt, model)
            else:
                raise ValueError(f"Proveedor no soportado: {provider}")
                
        except Exception as e:
            logger.error(f"Error en generación de texto: {e}")
            raise
    
    async def _openai_text_generation(self, prompt: str, model: str) -> Dict[str, Any]:
        """Generación de texto con OpenAI."""
        url = f"{self.api_configs['openai']['base_url']}/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['openai']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "provider": "openai",
                    "model": model,
                    "prompt": prompt,
                    "generated_text": result["choices"][0]["text"],
                    "usage": result.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Error OpenAI: {response.status} - {error_text}")
    
    async def _cohere_text_generation(self, prompt: str, model: str) -> Dict[str, Any]:
        """Generación de texto con Cohere."""
        url = f"{self.api_configs['cohere']['base_url']}/generate"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['cohere']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "prompt": prompt,
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "provider": "cohere",
                    "model": model,
                    "prompt": prompt,
                    "generated_text": result["generations"][0]["text"],
                    "usage": result.get("meta", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Error Cohere: {response.status} - {error_text}")
    
    async def get_embeddings(self, texts: List[str], model: str = "text-embedding-ada-002", provider: str = "openai") -> Dict[str, Any]:
        """Obtener embeddings de textos."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if provider == "openai":
                return await self._openai_embeddings(texts, model)
            elif provider == "cohere":
                return await self._cohere_embeddings(texts, model)
            else:
                raise ValueError(f"Proveedor no soportado: {provider}")
                
        except Exception as e:
            logger.error(f"Error al obtener embeddings: {e}")
            raise
    
    async def _openai_embeddings(self, texts: List[str], model: str) -> Dict[str, Any]:
        """Obtener embeddings con OpenAI."""
        url = f"{self.api_configs['openai']['base_url']}/embeddings"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['openai']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "input": texts
        }
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                embeddings = [item["embedding"] for item in result["data"]]
                return {
                    "provider": "openai",
                    "model": model,
                    "embeddings": embeddings,
                    "usage": result.get("usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Error OpenAI: {response.status} - {error_text}")
    
    async def _cohere_embeddings(self, texts: List[str], model: str) -> Dict[str, Any]:
        """Obtener embeddings con Cohere."""
        url = f"{self.api_configs['cohere']['base_url']}/embed"
        
        headers = {
            "Authorization": f"Bearer {self.api_keys['cohere']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "texts": texts
        }
        
        async with self.session.post(url, headers=headers, json=data) as response:
            if response.status == 200:
                result = await response.json()
                return {
                    "provider": "cohere",
                    "model": model,
                    "embeddings": result["embeddings"],
                    "usage": result.get("meta", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                error_text = await response.text()
                raise Exception(f"Error Cohere: {response.status} - {error_text}")
    
    async def analyze_sentiment_ai(self, text: str, provider: str = "openai") -> Dict[str, Any]:
        """Análisis de sentimiento con IA."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "Eres un experto en análisis de sentimiento. Analiza el sentimiento del texto y proporciona una respuesta en formato JSON con: sentiment (positive/negative/neutral), confidence (0-1), reasoning (explicación breve)."
                },
                {
                    "role": "user",
                    "content": f"Analiza el sentimiento de este texto: {text}"
                }
            ]
            
            result = await self.chat_completion(messages, provider=provider)
            
            # Intentar parsear la respuesta JSON
            try:
                sentiment_data = json.loads(result["response"])
                return {
                    "text": text,
                    "sentiment": sentiment_data.get("sentiment", "neutral"),
                    "confidence": sentiment_data.get("confidence", 0.5),
                    "reasoning": sentiment_data.get("reasoning", ""),
                    "provider": provider,
                    "timestamp": datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                # Si no es JSON válido, usar análisis básico
                response_lower = result["response"].lower()
                if "positive" in response_lower:
                    sentiment = "positive"
                elif "negative" in response_lower:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
                
                return {
                    "text": text,
                    "sentiment": sentiment,
                    "confidence": 0.7,
                    "reasoning": result["response"],
                    "provider": provider,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error en análisis de sentimiento con IA: {e}")
            raise
    
    async def summarize_text_ai(self, text: str, max_length: int = 150, provider: str = "openai") -> Dict[str, Any]:
        """Resumir texto con IA."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"Eres un experto en resumir texto. Crea un resumen conciso de máximo {max_length} palabras que capture los puntos principales del texto."
                },
                {
                    "role": "user",
                    "content": f"Resume este texto: {text}"
                }
            ]
            
            result = await self.chat_completion(messages, provider=provider)
            
            return {
                "original_text": text,
                "summary": result["response"],
                "max_length": max_length,
                "provider": provider,
                "usage": result.get("usage", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en resumen con IA: {e}")
            raise
    
    async def translate_text_ai(self, text: str, target_language: str, provider: str = "openai") -> Dict[str, Any]:
        """Traducir texto con IA."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"Eres un traductor experto. Traduce el texto al {target_language} manteniendo el significado y el tono original."
                },
                {
                    "role": "user",
                    "content": f"Traduce este texto: {text}"
                }
            ]
            
            result = await self.chat_completion(messages, provider=provider)
            
            return {
                "original_text": text,
                "translated_text": result["response"],
                "target_language": target_language,
                "provider": provider,
                "usage": result.get("usage", {}),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en traducción con IA: {e}")
            raise
    
    async def get_available_providers(self) -> Dict[str, Any]:
        """Obtener proveedores disponibles."""
        available_providers = {}
        
        for provider, config in self.api_configs.items():
            available_providers[provider] = {
                "available": provider in self.api_keys,
                "models": list(config["models"].keys()),
                "base_url": config["base_url"]
            }
        
        return {
            "providers": available_providers,
            "total_providers": len(available_providers),
            "available_count": sum(1 for p in available_providers.values() if p["available"]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del gestor de integración."""
        try:
            return {
                "status": "healthy" if self._initialized else "unhealthy",
                "initialized": self._initialized,
                "available_providers": await self.get_available_providers(),
                "session_active": self.session is not None and not self.session.closed,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }




