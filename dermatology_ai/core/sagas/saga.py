"""
Saga Pattern Implementation
Manages distributed transactions with compensation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Dict, Callable
from datetime import datetime
import uuid
import logging

logger = logging.getLogger(__name__)


class SagaState(str, Enum):
    """Saga execution state"""
    PENDING = "pending"
    RUNNING = "running"
    COMPENSATING = "compensating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SagaStep:
    """Single step in a saga"""
    name: str
    execute: Callable  # Function to execute step
    compensate: Optional[Callable] = None  # Compensation function
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: Optional[float] = None


class Saga:
    """
    Saga for managing distributed transactions.
    Implements compensation pattern for rollback.
    """
    
    def __init__(self, saga_id: Optional[str] = None):
        self.saga_id = saga_id or str(uuid.uuid4())
        self.steps: list[SagaStep] = []
        self.current_step: int = 0
        self.state: SagaState = SagaState.PENDING
        self.context: Dict[str, Any] = {}
        self.created_at: datetime = datetime.utcnow()
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
    
    def add_step(
        self,
        name: str,
        execute: Callable,
        compensate: Optional[Callable] = None,
        max_retries: int = 3,
        timeout_seconds: Optional[float] = None
    ):
        """Add step to saga"""
        step = SagaStep(
            name=name,
            execute=execute,
            compensate=compensate,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds
        )
        self.steps.append(step)
        return self
    
    async def execute(self) -> bool:
        """
        Execute saga steps sequentially
        
        Returns:
            True if all steps completed successfully
        """
        self.state = SagaState.RUNNING
        logger.info(f"Starting saga {self.saga_id} with {len(self.steps)} steps")
        
        try:
            for i, step in enumerate(self.steps):
                self.current_step = i
                logger.info(f"Executing step {i+1}/{len(self.steps)}: {step.name}")
                
                # Execute step with retry
                success = False
                for attempt in range(step.max_retries + 1):
                    try:
                        if step.timeout_seconds:
                            import asyncio
                            result = await asyncio.wait_for(
                                step.execute(self.context),
                                timeout=step.timeout_seconds
                            )
                        else:
                            result = await step.execute(self.context)
                        
                        # Store result in context
                        self.context[step.name] = result
                        success = True
                        break
                        
                    except Exception as e:
                        step.retry_count = attempt + 1
                        if attempt < step.max_retries:
                            logger.warning(
                                f"Step {step.name} failed (attempt {attempt+1}/{step.max_retries+1}): {e}"
                            )
                            import asyncio
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        else:
                            logger.error(f"Step {step.name} failed after {step.max_retries+1} attempts: {e}")
                            raise
                
                if not success:
                    raise Exception(f"Step {step.name} failed")
            
            self.state = SagaState.COMPLETED
            self.completed_at = datetime.utcnow()
            logger.info(f"Saga {self.saga_id} completed successfully")
            return True
            
        except Exception as e:
            self.state = SagaState.FAILED
            self.error = str(e)
            self.completed_at = datetime.utcnow()
            logger.error(f"Saga {self.saga_id} failed: {e}")
            
            # Compensate (rollback)
            await self.compensate()
            return False
    
    async def compensate(self):
        """Compensate (rollback) all executed steps"""
        if self.state != SagaState.FAILED:
            return
        
        self.state = SagaState.COMPENSATING
        logger.info(f"Compensating saga {self.saga_id}")
        
        # Compensate in reverse order
        for i in range(self.current_step, -1, -1):
            step = self.steps[i]
            
            if step.compensate:
                try:
                    logger.info(f"Compensating step {i+1}: {step.name}")
                    await step.compensate(self.context)
                except Exception as e:
                    logger.error(f"Compensation failed for step {step.name}: {e}")
                    # Continue with other compensations
    
    def get_state(self) -> Dict[str, Any]:
        """Get saga state"""
        return {
            "saga_id": self.saga_id,
            "state": self.state.value,
            "current_step": self.current_step,
            "total_steps": len(self.steps),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
        }

