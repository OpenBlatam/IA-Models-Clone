"""
State Machine System
===================

Sistema de máquinas de estado.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


@dataclass
class State:
    """Estado de la máquina."""
    state_id: str
    name: str
    on_enter: Optional[Callable] = None
    on_exit: Optional[Callable] = None
    on_stay: Optional[Callable] = None


@dataclass
class Transition:
    """Transición entre estados."""
    transition_id: str
    from_state: str
    to_state: str
    condition: Optional[Callable] = None
    on_transition: Optional[Callable] = None


class StateMachine:
    """
    Máquina de estado.
    
    Gestiona estados y transiciones.
    """
    
    def __init__(self, machine_id: str, name: str, initial_state: str):
        """
        Inicializar máquina de estado.
        
        Args:
            machine_id: ID único de la máquina
            name: Nombre de la máquina
            initial_state: Estado inicial
        """
        self.machine_id = machine_id
        self.name = name
        self.states: Dict[str, State] = {}
        self.transitions: List[Transition] = []
        self.current_state: Optional[str] = initial_state
        self.state_history: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
    
    def add_state(
        self,
        state_id: str,
        name: str,
        on_enter: Optional[Callable] = None,
        on_exit: Optional[Callable] = None,
        on_stay: Optional[Callable] = None
    ) -> State:
        """
        Agregar estado.
        
        Args:
            state_id: ID único del estado
            name: Nombre del estado
            on_enter: Callback al entrar
            on_exit: Callback al salir
            on_stay: Callback al permanecer
            
        Returns:
            Estado creado
        """
        state = State(
            state_id=state_id,
            name=name,
            on_enter=on_enter,
            on_exit=on_exit,
            on_stay=on_stay
        )
        
        self.states[state_id] = state
        return state
    
    def add_transition(
        self,
        transition_id: str,
        from_state: str,
        to_state: str,
        condition: Optional[Callable] = None,
        on_transition: Optional[Callable] = None
    ) -> Transition:
        """
        Agregar transición.
        
        Args:
            transition_id: ID único de la transición
            from_state: Estado origen
            to_state: Estado destino
            condition: Condición para la transición
            on_transition: Callback en la transición
            
        Returns:
            Transición creada
        """
        transition = Transition(
            transition_id=transition_id,
            from_state=from_state,
            to_state=to_state,
            condition=condition,
            on_transition=on_transition
        )
        
        self.transitions.append(transition)
        return transition
    
    async def transition_to(
        self,
        target_state: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transicionar a estado.
        
        Args:
            target_state: Estado destino
            context: Contexto adicional
            
        Returns:
            True si la transición fue exitosa
        """
        if context:
            self.context.update(context)
        
        if target_state not in self.states:
            logger.error(f"State not found: {target_state}")
            return False
        
        if self.current_state == target_state:
            # Permanecer en el estado
            state = self.states[target_state]
            if state.on_stay:
                try:
                    if asyncio.iscoroutinefunction(state.on_stay):
                        await state.on_stay(self.context)
                    else:
                        state.on_stay(self.context)
                except Exception as e:
                    logger.error(f"Error in on_stay callback: {e}")
            return True
        
        # Buscar transición válida
        valid_transition = None
        for transition in self.transitions:
            if transition.from_state == self.current_state and transition.to_state == target_state:
                if transition.condition is None:
                    valid_transition = transition
                    break
                else:
                    try:
                        if asyncio.iscoroutinefunction(transition.condition):
                            if await transition.condition(self.context):
                                valid_transition = transition
                                break
                        else:
                            if transition.condition(self.context):
                                valid_transition = transition
                                break
                    except Exception as e:
                        logger.error(f"Error in transition condition: {e}")
        
        if not valid_transition:
            logger.warning(f"No valid transition from {self.current_state} to {target_state}")
            return False
        
        # Ejecutar callbacks
        if self.current_state:
            current_state_obj = self.states[self.current_state]
            if current_state_obj.on_exit:
                try:
                    if asyncio.iscoroutinefunction(current_state_obj.on_exit):
                        await current_state_obj.on_exit(self.context)
                    else:
                        current_state_obj.on_exit(self.context)
                except Exception as e:
                    logger.error(f"Error in on_exit callback: {e}")
        
        if valid_transition.on_transition:
            try:
                if asyncio.iscoroutinefunction(valid_transition.on_transition):
                    await valid_transition.on_transition(self.context)
                else:
                    valid_transition.on_transition(self.context)
            except Exception as e:
                logger.error(f"Error in on_transition callback: {e}")
        
        # Cambiar estado
        old_state = self.current_state
        self.current_state = target_state
        
        new_state_obj = self.states[target_state]
        if new_state_obj.on_enter:
            try:
                if asyncio.iscoroutinefunction(new_state_obj.on_enter):
                    await new_state_obj.on_enter(self.context)
                else:
                    new_state_obj.on_enter(self.context)
            except Exception as e:
                logger.error(f"Error in on_enter callback: {e}")
        
        # Registrar en historial
        self.state_history.append({
            "from_state": old_state,
            "to_state": target_state,
            "transition_id": valid_transition.transition_id,
            "timestamp": __import__('datetime').datetime.now().isoformat()
        })
        
        logger.info(f"State transition: {old_state} -> {target_state}")
        return True
    
    def get_current_state(self) -> Optional[str]:
        """Obtener estado actual."""
        return self.current_state
    
    def get_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de estados."""
        return self.state_history[-limit:]


class StateMachineManager:
    """Gestor de máquinas de estado."""
    
    def __init__(self):
        """Inicializar gestor."""
        self.machines: Dict[str, StateMachine] = {}
    
    def create_machine(
        self,
        machine_id: str,
        name: str,
        initial_state: str
    ) -> StateMachine:
        """Crear nueva máquina de estado."""
        machine = StateMachine(machine_id, name, initial_state)
        self.machines[machine_id] = machine
        return machine
    
    def get_machine(self, machine_id: str) -> Optional[StateMachine]:
        """Obtener máquina de estado."""
        return self.machines.get(machine_id)
    
    def list_machines(self) -> List[StateMachine]:
        """Listar todas las máquinas."""
        return list(self.machines.values())


# Instancia global
_state_machine_manager: Optional[StateMachineManager] = None


def get_state_machine_manager() -> StateMachineManager:
    """Obtener instancia global del gestor de máquinas de estado."""
    global _state_machine_manager
    if _state_machine_manager is None:
        _state_machine_manager = StateMachineManager()
    return _state_machine_manager

