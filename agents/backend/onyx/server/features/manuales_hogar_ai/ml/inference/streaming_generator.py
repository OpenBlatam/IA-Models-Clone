"""
Generador con Streaming
=======================

Generación de texto con streaming para respuestas en tiempo real.
"""

import logging
import torch
from typing import AsyncIterator, Optional, Dict, Any
from transformers import TextIteratorStreamer
from threading import Thread
import asyncio

logger = logging.getLogger(__name__)


class StreamingGenerator:
    """Generador con streaming para respuestas en tiempo real."""
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Inicializar generador con streaming.
        
        Args:
            model: Modelo de transformers
            tokenizer: Tokenizer
            device: Dispositivo
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._logger = logger
    
    async def generate_stream(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generar texto con streaming.
        
        Args:
            prompt: Prompt de entrada
            max_length: Longitud máxima
            temperature: Temperatura
            top_p: Top-p sampling
            **kwargs: Otros parámetros
        
        Yields:
            Tokens generados
        """
        try:
            # Tokenizar
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Crear streamer
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )
            
            # Configuración de generación
            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "streamer": streamer,
                "pad_token_id": self.tokenizer.pad_token_id,
                **kwargs
            }
            
            # Generar en thread separado
            thread = Thread(
                target=self._generate_in_thread,
                args=(generation_kwargs,)
            )
            thread.start()
            
            # Stream tokens
            for token in streamer:
                yield token
            
            thread.join()
        
        except Exception as e:
            self._logger.error(f"Error en streaming: {str(e)}")
            yield f"[ERROR: {str(e)}]"
    
    def _generate_in_thread(self, generation_kwargs: Dict[str, Any]):
        """Generar en thread separado."""
        try:
            with torch.no_grad():
                self.model.generate(**generation_kwargs)
        except Exception as e:
            self._logger.error(f"Error en generación: {str(e)}")
    
    async def generate_stream_manual(
        self,
        problem_description: str,
        category: str = "general",
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generar manual con streaming.
        
        Args:
            problem_description: Descripción del problema
            category: Categoría
            **kwargs: Otros parámetros
        
        Yields:
            Texto del manual generado
        """
        prompt = self._build_manual_prompt(problem_description, category)
        
        async for token in self.generate_stream(prompt, **kwargs):
            yield token
    
    def _build_manual_prompt(
        self,
        problem_description: str,
        category: str
    ) -> str:
        """Construir prompt para generación."""
        category_names = {
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
        
        category_name = category_names.get(category, "Reparación General")
        
        prompt = f"""Genera un manual paso a paso tipo LEGO para {category_name}.

PROBLEMA:
{problem_description}

MANUAL:
"""
        return prompt




