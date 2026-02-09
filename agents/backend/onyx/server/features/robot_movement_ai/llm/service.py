"""
LLM Service - Servicio de modelos de lenguaje
"""
from typing import List, Dict, Any, Optional
from .model_manager import ModelManager
from .inference_engine import InferenceEngine
from .prompt_engine import PromptEngine


class LLMService:
    """Servicio principal de LLM"""
    
    def __init__(self):
        self.model_manager = ModelManager()
        self.inference_engine = InferenceEngine()
        self.prompt_engine = PromptEngine()
    
    async def generate(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Genera texto usando un modelo"""
        model = await self.model_manager.get_model(model_name)
        return await self.inference_engine.generate(model, prompt, **kwargs)
    
    async def embed(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """Genera embeddings"""
        model = await self.model_manager.get_model(model_name)
        return await self.inference_engine.embed(model, text)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs
    ) -> str:
        """Genera respuesta en formato chat"""
        model = await self.model_manager.get_model(model_name)
        return await self.inference_engine.chat(model, messages, **kwargs)

