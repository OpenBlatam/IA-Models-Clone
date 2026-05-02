"""
Agent Operations Module
=======================

Operaciones básicas del agente encapsuladas.
Proporciona una capa de abstracción para think, act, observe.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .prompt_builder import PromptBuilder
from .memory_context_builder import MemoryContextBuilder

logger = logging.getLogger(__name__)


class AgentOperations:
    """
    Encapsula operaciones básicas del agente.
    
    Proporciona métodos para:
    - Thinking (razonamiento)
    - Acting (acción)
    - Observing (observación)
    """
    
    def __init__(
        self,
        llm_client,
        memory_context_builder: Optional[MemoryContextBuilder] = None,
        metrics_manager=None,
        state=None
    ):
        """
        Inicializar operaciones del agente.
        
        Args:
            llm_client: Cliente LLM
            memory_context_builder: Constructor de contexto de memoria
            metrics_manager: Manager de métricas
            state: Estado del agente
        """
        self.llm_client = llm_client
        self.memory_context_builder = memory_context_builder
        self.metrics_manager = metrics_manager
        self.state = state
    
    async def think_async(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        use_memory: bool = True
    ) -> Dict[str, Any]:
        """
        Pensar sobre una tarea (versión async).
        
        Args:
            task: Descripción de la tarea
            context: Contexto adicional
            use_memory: Si se debe usar contexto de memoria
            
        Returns:
            Resultado del pensamiento
        """
        if self.state:
            self.state.status = getattr(self.state, 'status', None)
            if hasattr(self.state, 'current_task'):
                self.state.current_task = task
        
        # Construir contexto de memoria si está disponible
        memory_context = None
        if use_memory and self.memory_context_builder:
            try:
                memory_context = self.memory_context_builder.build_context_for_task(task)
            except Exception as e:
                logger.warning(f"Error building memory context: {e}")
        
        # Construir prompt
        thinking_prompt = PromptBuilder.build_thinking_prompt(
            task=task,
            context=context,
            memory_context=memory_context
        )
        
        try:
            start_time = datetime.now()
            response = await self.llm_client.generate_text(
                prompt=thinking_prompt,
                max_tokens=2000,
                temperature=0.7
            )
            elapsed = (datetime.now() - start_time).total_seconds()
            
            generated_text = response.get("generated_text", "")
            usage = response.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            
            # Registrar métricas
            if self.metrics_manager:
                self.metrics_manager.record_llm_call(tokens_used, elapsed)
            
            result = {
                "task": task,
                "reasoning": generated_text,
                "tokens_used": tokens_used,
                "response_time": elapsed,
                "context": context or {},
                "memory_used": memory_context is not None
            }
            
            if self.state and hasattr(self.state, 'add_step'):
                self.state.add_step("thinking", result)
            
            return result
        
        except Exception as e:
            logger.error(f"Error in think: {e}", exc_info=True)
            if self.metrics_manager:
                self.metrics_manager.record_error()
            if self.state:
                self.state.status = getattr(self.state, 'status', None)
                if hasattr(self.state, 'add_step'):
                    self.state.add_step("error", {"error": str(e), "task": task})
            raise
    
    def think_sync(
        self,
        observation: str,
        context: Optional[Dict] = None
    ) -> str:
        """
        Pensar sobre una observación (versión sync para compatibilidad).
        
        Args:
            observation: Observación actual
            context: Contexto adicional
            
        Returns:
            Pensamiento como string
        """
        try:
            result = asyncio.run(self.think_async(observation, context))
            return result.get("reasoning", f"Thinking about: {observation}")
        except Exception as e:
            logger.error(f"Error in think: {e}", exc_info=True)
            return f"Error thinking: {str(e)}"
    
    def act(
        self,
        action: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        tool_registry=None
    ) -> Dict[str, Any]:
        """
        Ejecutar una acción.
        
        Args:
            action: Acción a ejecutar
            context: Contexto adicional
            tool_registry: Registro de herramientas
            
        Returns:
            Resultado de la acción
        """
        if self.state:
            self.state.status = getattr(self.state, 'status', None)
        
        action_type = action.get("type", "process")
        action_description = action.get("description", "")
        
        try:
            if action_type == "tool_call":
                # Ejecutar herramienta
                tool_name = action.get("tool", "")
                tool_args = action.get("args", {})
                
                if tool_registry:
                    tool = tool_registry.get_tool(tool_name)
                    if tool:
                        result = tool.execute(**tool_args)
                    else:
                        result = {"error": f"Tool {tool_name} not found"}
                else:
                    result = {"error": "Tool registry not available"}
            
            elif action_type == "process":
                # Procesar acción genérica
                result = {"status": "processed", "action": action_description}
            
            else:
                result = {"status": "unknown_action_type", "type": action_type}
            
            if self.state and hasattr(self.state, 'add_step'):
                self.state.add_step("action", {
                    "type": action_type,
                    "result": result,
                    "context": context or {}
                })
            
            return result
        
        except Exception as e:
            logger.error(f"Error in act: {e}", exc_info=True)
            if self.state and hasattr(self.state, 'add_step'):
                self.state.add_step("error", {"error": str(e), "action": action})
            raise
    
    def observe(
        self,
        observation: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Observar un resultado.
        
        Args:
            observation: Observación a procesar
            context: Contexto adicional
            
        Returns:
            Observación procesada
        """
        if self.state:
            self.state.status = getattr(self.state, 'status', None)
        
        try:
            # Procesar observación
            if isinstance(observation, dict):
                processed = observation
            elif isinstance(observation, str):
                processed = {"observation": observation, "type": "text"}
            else:
                processed = {"observation": str(observation), "type": "unknown"}
            
            processed["context"] = context or {}
            processed["timestamp"] = datetime.now().isoformat()
            
            if self.state and hasattr(self.state, 'add_step'):
                self.state.add_step("observation", processed)
            
            return processed
        
        except Exception as e:
            logger.error(f"Error in observe: {e}", exc_info=True)
            if self.state and hasattr(self.state, 'add_step'):
                self.state.add_step("error", {"error": str(e), "observation": str(observation)})
            raise
