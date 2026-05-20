"""
Initialization System
=====================

System for modular component initialization.
"""

import logging
from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class InitializationPhase(Enum):
    """Initialization phase."""
    SETUP = "setup"
    CLIENTS = "clients"
    MANAGERS = "managers"
    PROCESSORS = "processors"
    SERVICES = "services"
    FINALIZE = "finalize"


@dataclass
class InitializationStep:
    """Initialization step definition."""
    name: str
    phase: InitializationPhase
    initializer: Callable[[], Awaitable[None]]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 0  # Lower priority runs first


class InitializationManager:
    """Manager for component initialization."""
    
    def __init__(self):
        """Initialize initialization manager."""
        self.steps: Dict[str, InitializationStep] = {}
        self.completed: List[str] = []
    
    def register_step(
        self,
        name: str,
        phase: InitializationPhase,
        initializer: Callable[[], Awaitable[None]],
        dependencies: Optional[List[str]] = None,
        priority: int = 0
    ):
        """
        Register an initialization step.
        
        Args:
            name: Step name
            phase: Initialization phase
            initializer: Async initializer function
            dependencies: Optional step dependencies
            priority: Step priority
        """
        step = InitializationStep(
            name=name,
            phase=phase,
            initializer=initializer,
            dependencies=dependencies or [],
            priority=priority
        )
        self.steps[name] = step
        logger.info(f"Registered initialization step: {name} ({phase.value})")
    
    async def initialize_phase(self, phase: InitializationPhase):
        """
        Initialize a specific phase.
        
        Args:
            phase: Phase to initialize
        """
        phase_steps = [
            step for step in self.steps.values()
            if step.phase == phase
        ]
        
        # Sort by priority
        phase_steps.sort(key=lambda s: s.priority)
        
        # Resolve dependencies
        ordered_steps = self._resolve_dependencies(phase_steps)
        
        # Execute steps
        for step in ordered_steps:
            if step.name in self.completed:
                continue
            
            # Check dependencies
            deps_satisfied = all(
                dep in self.completed for dep in step.dependencies
            )
            
            if not deps_satisfied:
                missing = [d for d in step.dependencies if d not in self.completed]
                raise RuntimeError(
                    f"Step '{step.name}' has unsatisfied dependencies: {missing}"
                )
            
            try:
                await step.initializer()
                self.completed.append(step.name)
                logger.info(f"Completed initialization step: {step.name}")
            except Exception as e:
                logger.error(f"Error in initialization step '{step.name}': {e}")
                raise
    
    async def initialize_all(self):
        """Initialize all phases in order."""
        phases = [
            InitializationPhase.SETUP,
            InitializationPhase.CLIENTS,
            InitializationPhase.MANAGERS,
            InitializationPhase.PROCESSORS,
            InitializationPhase.SERVICES,
            InitializationPhase.FINALIZE
        ]
        
        for phase in phases:
            await self.initialize_phase(phase)
    
    def _resolve_dependencies(self, steps: List[InitializationStep]) -> List[InitializationStep]:
        """Resolve step dependencies."""
        resolved = []
        remaining = steps.copy()
        
        while remaining:
            progress = False
            
            for step in remaining[:]:
                deps_satisfied = all(
                    dep in [s.name for s in resolved] or
                    dep in self.completed
                    for dep in step.dependencies
                )
                
                if deps_satisfied:
                    resolved.append(step)
                    remaining.remove(step)
                    progress = True
            
            if not progress:
                # Circular dependency or missing dependency
                unresolved = [s.name for s in remaining]
                logger.warning(f"Could not resolve dependencies for: {unresolved}")
                resolved.extend(remaining)
                break
        
        return resolved




