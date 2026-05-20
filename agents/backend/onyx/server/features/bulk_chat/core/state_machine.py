"""
State Machine - Máquina de Estados Avanzada
===========================================

Sistema de máquina de estados con transiciones, validaciones y eventos.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Tipo de transición."""
    AUTOMATIC = "automatic"
    MANUAL = "manual"
    CONDITIONAL = "conditional"
    TIMED = "timed"


@dataclass
class StateTransition:
    """Transición de estado."""
    transition_id: str
    from_state: str
    to_state: str
    transition_type: TransitionType
    condition: Optional[Callable] = None
    on_transition: Optional[Callable] = None
    timeout: Optional[float] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateMachine:
    """Máquina de estados."""
    machine_id: str
    name: str
    initial_state: str
    states: List[str] = field(default_factory=list)
    transitions: List[StateTransition] = field(default_factory=list)
    current_state: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateHistory:
    """Historial de estados."""
    machine_id: str
    timestamp: datetime
    from_state: str
    to_state: str
    transition_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateMachineManager:
    """Gestor de máquinas de estados."""
    
    def __init__(self):
        self.machines: Dict[str, StateMachine] = {}
        self.state_history: deque = deque(maxlen=100000)
        self._lock = asyncio.Lock()
    
    def create_state_machine(
        self,
        machine_id: str,
        name: str,
        initial_state: str,
        states: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Crear máquina de estados."""
        machine = StateMachine(
            machine_id=machine_id,
            name=name,
            initial_state=initial_state,
            states=states or [],
            current_state=initial_state,
            metadata=metadata or {},
        )
        
        async def save_machine():
            async with self._lock:
                self.machines[machine_id] = machine
        
        asyncio.create_task(save_machine())
        
        logger.info(f"Created state machine: {machine_id} - {name}")
        return machine_id
    
    def add_transition(
        self,
        machine_id: str,
        transition_id: str,
        from_state: str,
        to_state: str,
        transition_type: TransitionType = TransitionType.MANUAL,
        condition: Optional[Callable] = None,
        on_transition: Optional[Callable] = None,
        timeout: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Agregar transición."""
        transition = StateTransition(
            transition_id=transition_id,
            from_state=from_state,
            to_state=to_state,
            transition_type=transition_type,
            condition=condition,
            on_transition=on_transition,
            timeout=timeout,
            metadata=metadata or {},
        )
        
        async def save_transition():
            async with self._lock:
                machine = self.machines.get(machine_id)
                if not machine:
                    raise ValueError(f"State machine {machine_id} not found")
                machine.transitions.append(transition)
        
        asyncio.create_task(save_transition())
        
        logger.info(f"Added transition {transition_id} to machine {machine_id}")
        return transition_id
    
    async def transition(
        self,
        machine_id: str,
        to_state: str,
        transition_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Realizar transición."""
        machine = self.machines.get(machine_id)
        if not machine:
            return False
        
        current_state = machine.current_state
        
        # Buscar transición válida
        valid_transition = None
        if transition_id:
            valid_transition = next(
                (t for t in machine.transitions if t.transition_id == transition_id),
                None
            )
        else:
            valid_transition = next(
                (t for t in machine.transitions
                 if t.from_state == current_state and t.to_state == to_state and t.enabled),
                None
            )
        
        if not valid_transition:
            return False
        
        # Verificar condición si existe
        if valid_transition.condition:
            try:
                if asyncio.iscoroutinefunction(valid_transition.condition):
                    condition_result = await valid_transition.condition(machine)
                else:
                    condition_result = valid_transition.condition(machine)
                
                if not condition_result:
                    return False
            except Exception as e:
                logger.error(f"Error evaluating transition condition: {e}")
                return False
        
        # Realizar transición
        old_state = machine.current_state
        machine.current_state = to_state
        
        # Ejecutar callback si existe
        if valid_transition.on_transition:
            try:
                if asyncio.iscoroutinefunction(valid_transition.on_transition):
                    await valid_transition.on_transition(machine, old_state, to_state)
                else:
                    valid_transition.on_transition(machine, old_state, to_state)
            except Exception as e:
                logger.error(f"Error executing transition callback: {e}")
        
        # Guardar historial
        async with self._lock:
            self.state_history.append(StateHistory(
                machine_id=machine_id,
                timestamp=datetime.now(),
                from_state=old_state,
                to_state=to_state,
                transition_id=valid_transition.transition_id,
                metadata=metadata or {},
            ))
        
        logger.info(f"State machine {machine_id} transitioned: {old_state} -> {to_state}")
        return True
    
    async def auto_transition(self, machine_id: str):
        """Intentar transición automática."""
        machine = self.machines.get(machine_id)
        if not machine:
            return
        
        # Buscar transiciones automáticas desde el estado actual
        auto_transitions = [
            t for t in machine.transitions
            if t.from_state == machine.current_state
            and t.transition_type == TransitionType.AUTOMATIC
            and t.enabled
        ]
        
        for transition in auto_transitions:
            # Verificar condición si existe
            if transition.condition:
                try:
                    if asyncio.iscoroutinefunction(transition.condition):
                        condition_result = await transition.condition(machine)
                    else:
                        condition_result = transition.condition(machine)
                    
                    if not condition_result:
                        continue
                except Exception:
                    continue
            
            # Realizar transición
            await self.transition(machine_id, transition.to_state, transition.transition_id)
            break
    
    def get_state_machine(self, machine_id: str) -> Optional[Dict[str, Any]]:
        """Obtener información de máquina de estados."""
        machine = self.machines.get(machine_id)
        if not machine:
            return None
        
        return {
            "machine_id": machine.machine_id,
            "name": machine.name,
            "initial_state": machine.initial_state,
            "current_state": machine.current_state,
            "states": machine.states,
            "transitions": [
                {
                    "transition_id": t.transition_id,
                    "from_state": t.from_state,
                    "to_state": t.to_state,
                    "transition_type": t.transition_type.value,
                    "enabled": t.enabled,
                }
                for t in machine.transitions
            ],
            "created_at": machine.created_at.isoformat(),
        }
    
    def get_state_history(self, machine_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener historial de estados."""
        history = list(self.state_history)
        
        if machine_id:
            history = [h for h in history if h.machine_id == machine_id]
        
        history.sort(key=lambda h: h.timestamp, reverse=True)
        
        return [
            {
                "machine_id": h.machine_id,
                "timestamp": h.timestamp.isoformat(),
                "from_state": h.from_state,
                "to_state": h.to_state,
                "transition_id": h.transition_id,
            }
            for h in history[:limit]
        ]
    
    def get_state_machine_manager_summary(self) -> Dict[str, Any]:
        """Obtener resumen del gestor."""
        return {
            "total_machines": len(self.machines),
            "total_transitions": sum(len(m.transitions) for m in self.machines.values()),
            "total_history": len(self.state_history),
        }


