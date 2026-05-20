"""
AI Engine - Motor de IA para análisis y sugerencias inteligentes
"""

import logging
import os
from typing import Dict, Any, Optional, List
import json

logger = logging.getLogger(__name__)

# Intentar importar OpenAI
try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI no disponible. Funcionalidades de IA limitadas.")

# Intentar importar LangChain
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain.schema import SystemMessage, HumanMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger.warning("LangChain no disponible. Usando análisis básico.")


class AIEngine:
    """Motor de IA para análisis inteligente y sugerencias"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inicializar el motor de IA.

        Args:
            config: Configuración opcional
        """
        self.config = config or {}
        self.ai_config = self.config.get("ai", {})
        
        # Inicializar cliente OpenAI si está disponible
        self.openai_client = None
        if OPENAI_AVAILABLE:
            api_key = os.getenv("OPENAI_API_KEY") or self.ai_config.get("api_key")
            if api_key:
                self.openai_client = AsyncOpenAI(api_key=api_key)
        
        # Inicializar LangChain si está disponible
        self.langchain_llm = None
        if LANGCHAIN_AVAILABLE:
            model_name = self.ai_config.get("model", "gpt-3.5-turbo")
            temperature = self.ai_config.get("temperature", 0.7)
            try:
                self.langchain_llm = ChatOpenAI(
                    model_name=model_name,
                    temperature=temperature,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
            except Exception as e:
                logger.warning(f"No se pudo inicializar LangChain: {e}")

    async def analyze_context_intelligent(
        self,
        content: str,
        operation: str = "add",
        addition: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Análisis inteligente de contexto usando IA.

        Args:
            content: Contenido a analizar
            operation: Tipo de operación (add, remove)
            addition: Contenido a agregar (si aplica)

        Returns:
            Análisis inteligente del contexto
        """
        if not self.openai_client and not self.langchain_llm:
            return {"intelligent_analysis": False, "reason": "IA no disponible"}

        try:
            if operation == "add" and addition:
                prompt = f"""Analiza el siguiente contenido y sugiere la mejor posición para agregar el nuevo texto.

CONTENIDO ORIGINAL:
{content[:2000]}

NUEVO CONTENIDO A AGREGAR:
{addition[:500]}

Proporciona:
1. La mejor posición (start, end, before_section, after_section, specific_position)
2. Razón de la sugerencia
3. Coherencia temática (0-1)
4. Sugerencias de integración

Responde en formato JSON:
{{
    "position": "end",
    "reason": "El nuevo contenido complementa el tema principal",
    "coherence": 0.85,
    "suggestions": ["Mantener el tono formal", "Agregar transición"]
}}"""
            else:
                prompt = f"""Analiza el siguiente contenido y sugiere qué elementos podrían eliminarse para mejorar la calidad.

CONTENIDO:
{content[:2000]}

Proporciona:
1. Elementos redundantes o innecesarios
2. Secciones que podrían eliminarse
3. Razón de cada sugerencia
4. Impacto en la coherencia

Responde en formato JSON."""
            
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=self.ai_config.get("model", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": "Eres un experto en análisis de contenido y edición de texto."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.ai_config.get("temperature", 0.7),
                    max_tokens=self.ai_config.get("max_tokens", 1000)
                )
                result_text = response.choices[0].message.content
                try:
                    return json.loads(result_text)
                except:
                    return {"analysis": result_text, "intelligent_analysis": True}
            
            elif self.langchain_llm:
                messages = [
                    SystemMessage(content="Eres un experto en análisis de contenido y edición de texto."),
                    HumanMessage(content=prompt)
                ]
                result = await self.langchain_llm.ainvoke(messages)
                try:
                    return json.loads(result.content)
                except:
                    return {"analysis": result.content, "intelligent_analysis": True}
        
        except Exception as e:
            logger.error(f"Error en análisis inteligente: {e}")
            return {"intelligent_analysis": False, "error": str(e)}

    async def suggest_optimal_position(
        self,
        content: str,
        addition: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Sugerir la posición óptima usando IA.

        Args:
            content: Contenido original
            addition: Contenido a agregar
            context: Contexto adicional

        Returns:
            Sugerencia de posición con razones
        """
        analysis = await self.analyze_context_intelligent(content, "add", addition)
        
        if analysis.get("intelligent_analysis"):
            return {
                "position": analysis.get("position", "end"),
                "reason": analysis.get("reason", ""),
                "confidence": analysis.get("coherence", 0.5),
                "suggestions": analysis.get("suggestions", [])
            }
        
        # Fallback a análisis básico
        return {
            "position": "end",
            "reason": "Análisis básico: agregar al final",
            "confidence": 0.5,
            "suggestions": []
        }

    async def suggest_removals(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Sugerir elementos a eliminar usando IA.

        Args:
            content: Contenido a analizar
            context: Contexto adicional

        Returns:
            Lista de sugerencias de eliminación
        """
        analysis = await self.analyze_context_intelligent(content, "remove")
        
        if analysis.get("intelligent_analysis"):
            # Procesar sugerencias
            suggestions = analysis.get("suggestions", [])
            if isinstance(suggestions, list):
                return suggestions
            elif isinstance(suggestions, str):
                return [{"text": suggestions, "reason": analysis.get("reason", "")}]
        
        return []

    async def validate_semantic_coherence(
        self,
        original: str,
        modified: str,
        operation: str
    ) -> Dict[str, Any]:
        """
        Validar coherencia semántica usando IA.

        Args:
            original: Contenido original
            modified: Contenido modificado
            operation: Tipo de operación

        Returns:
            Validación semántica
        """
        if not self.openai_client and not self.langchain_llm:
            return {"semantic_validation": False, "coherent": True}

        try:
            prompt = f"""Evalúa la coherencia semántica entre el contenido original y modificado.

CONTENIDO ORIGINAL:
{original[:1500]}

CONTENIDO MODIFICADO:
{modified[:1500]}

OPERACIÓN: {operation}

Evalúa:
1. Coherencia temática (0-1)
2. Fluidez del texto (0-1)
3. Mantenimiento del tono (0-1)
4. Problemas detectados
5. Sugerencias de mejora

Responde en formato JSON:
{{
    "coherent": true,
    "thematic_coherence": 0.9,
    "fluency": 0.85,
    "tone_consistency": 0.9,
    "issues": [],
    "suggestions": []
}}"""
            
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model=self.ai_config.get("model", "gpt-3.5-turbo"),
                    messages=[
                        {"role": "system", "content": "Eres un experto en evaluación de calidad de texto."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                result_text = response.choices[0].message.content
                try:
                    return json.loads(result_text)
                except:
                    return {"semantic_validation": True, "coherent": True, "raw": result_text}
        
        except Exception as e:
            logger.error(f"Error en validación semántica: {e}")
            return {"semantic_validation": False, "coherent": True, "error": str(e)}






