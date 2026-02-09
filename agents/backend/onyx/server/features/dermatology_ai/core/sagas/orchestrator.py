"""
Saga Orchestrator - Manages multiple sagas
"""

from typing import Dict, Optional
from .saga import Saga, SagaState
import logging

logger = logging.getLogger(__name__)


class SagaOrchestrator:
    """
    Orchestrator for managing multiple sagas.
    Tracks saga execution and provides query interface.
    """
    
    def __init__(self):
        self.sagas: Dict[str, Saga] = {}
    
    async def execute_saga(self, saga: Saga) -> bool:
        """
        Execute saga and track it
        
        Args:
            saga: Saga to execute
            
        Returns:
            True if successful
        """
        self.sagas[saga.saga_id] = saga
        result = await saga.execute()
        return result
    
    def get_saga(self, saga_id: str) -> Optional[Saga]:
        """Get saga by ID"""
        return self.sagas.get(saga_id)
    
    def get_saga_state(self, saga_id: str) -> Optional[Dict[str, Any]]:
        """Get saga state"""
        saga = self.get_saga(saga_id)
        if saga:
            return saga.get_state()
        return None
    
    def list_sagas(
        self,
        state: Optional[SagaState] = None,
        limit: int = 100
    ) -> list[Saga]:
        """List sagas, optionally filtered by state"""
        sagas = list(self.sagas.values())
        
        if state:
            sagas = [s for s in sagas if s.state == state]
        
        return sagas[:limit]


# Global orchestrator
_saga_orchestrator: Optional[SagaOrchestrator] = None


def get_saga_orchestrator() -> SagaOrchestrator:
    """Get or create global saga orchestrator"""
    global _saga_orchestrator
    if _saga_orchestrator is None:
        _saga_orchestrator = SagaOrchestrator()
    return _saga_orchestrator















