"""
Hook Manager Module
===================

Gestión centralizada de hooks para el ciclo de vida del agente.
Proporciona registro y ejecución estructurada de hooks.
"""

import asyncio
import logging
from typing import List, Callable, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class HookType(Enum):
    """Tipos de hooks disponibles."""
    PRE_START = "pre_start"
    POST_START = "post_start"
    PRE_STOP = "pre_stop"
    POST_STOP = "post_stop"
    PRE_CLEANUP = "pre_cleanup"
    POST_CLEANUP = "post_cleanup"
    PRE_TASK = "pre_task"
    POST_TASK = "post_task"
    PRE_REFLECTION = "pre_reflection"
    POST_REFLECTION = "post_reflection"


class HookManager:
    """
    Gestor de hooks para el ciclo de vida del agente.
    
    Permite registrar y ejecutar hooks de forma estructurada
    en diferentes puntos del ciclo de vida.
    """
    
    def __init__(self):
        """Inicializar gestor de hooks."""
        self.hooks: Dict[HookType, List[Callable]] = {
            hook_type: [] for hook_type in HookType
        }
    
    def register_hook(
        self,
        hook_type: HookType,
        hook: Callable,
        priority: int = 0
    ) -> None:
        """
        Registrar un hook.
        
        Args:
            hook_type: Tipo de hook
            hook: Función a ejecutar
            priority: Prioridad (mayor = se ejecuta primero)
        """
        if hook not in self.hooks[hook_type]:
            # Insertar según prioridad
            inserted = False
            for i, (existing_hook, existing_priority) in enumerate(
                [(h, getattr(h, '_hook_priority', 0)) for h in self.hooks[hook_type]]
            ):
                if priority > existing_priority:
                    self.hooks[hook_type].insert(i, hook)
                    inserted = True
                    break
            
            if not inserted:
                self.hooks[hook_type].append(hook)
            
            # Guardar prioridad
            hook._hook_priority = priority
            logger.debug(f"Registered {hook_type.value} hook with priority {priority}")
    
    def unregister_hook(self, hook_type: HookType, hook: Callable) -> None:
        """
        Desregistrar un hook.
        
        Args:
            hook_type: Tipo de hook
            hook: Función a remover
        """
        if hook in self.hooks[hook_type]:
            self.hooks[hook_type].remove(hook)
            logger.debug(f"Unregistered {hook_type.value} hook")
    
    async def execute_hooks(
        self,
        hook_type: HookType,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ejecutar todos los hooks de un tipo.
        
        Args:
            hook_type: Tipo de hooks a ejecutar
            context: Contexto opcional para pasar a los hooks
        """
        hooks = self.hooks.get(hook_type, [])
        
        if not hooks:
            return
        
        logger.debug(f"Executing {len(hooks)} {hook_type.value} hooks")
        
        for hook in hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    if context:
                        await hook(**context)
                    else:
                        await hook()
                else:
                    if context:
                        hook(**context)
                    else:
                        hook()
            except Exception as e:
                logger.error(
                    f"Error executing {hook_type.value} hook: {e}",
                    exc_info=True
                )
                # Continuar con otros hooks aunque uno falle
    
    def get_hooks(self, hook_type: HookType) -> List[Callable]:
        """
        Obtener lista de hooks de un tipo.
        
        Args:
            hook_type: Tipo de hooks
            
        Returns:
            Lista de hooks
        """
        return self.hooks.get(hook_type, []).copy()
    
    def clear_hooks(self, hook_type: Optional[HookType] = None) -> None:
        """
        Limpiar hooks.
        
        Args:
            hook_type: Tipo de hooks a limpiar (None = todos)
        """
        if hook_type:
            self.hooks[hook_type] = []
            logger.debug(f"Cleared {hook_type.value} hooks")
        else:
            self.hooks = {hook_type: [] for hook_type in HookType}
            logger.debug("Cleared all hooks")
    
    def get_hook_count(self, hook_type: Optional[HookType] = None) -> int:
        """
        Obtener número de hooks registrados.
        
        Args:
            hook_type: Tipo de hooks (None = todos)
            
        Returns:
            Número de hooks
        """
        if hook_type:
            return len(self.hooks.get(hook_type, []))
        return sum(len(hooks) for hooks in self.hooks.values())


def create_hook_manager() -> HookManager:
    """
    Factory function para crear HookManager.
    
    Returns:
        Instancia de HookManager
    """
    return HookManager()


