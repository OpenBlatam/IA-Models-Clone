"""
🚀 TruthGPT SOTA Injected Base Agent - System 5.9 Gold Standard
Refactored by: Autonomous Code Injector (Layer 2.8)
Pattern Applied: [Circuit Breaker + Forensic State Persistence + Observer Hooks]
"""

import time
import json
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict
from ..models import AgentResponse
from optimization_core.truthgpt_collected.injected.upgraded_execution_kernel import exec_kernel

logger = logging.getLogger("TruthGPT.SOTA.Agent")

class MemoryEntry(BaseModel):
    """SOTA UPGRADE: Added sequence numbering and metadata slots."""
    sequence: int = Field(default=0)
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="The message content")
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AgentStatus(BaseModel):
    """SOTA UPGRADE: Added health metrics and circuit status."""
    name: str
    role: str
    memory_size: int = 0
    is_active: bool = True
    circuit_state: str = "CLOSED"  # CLOSED, OPEN, HALF-OPEN
    consecutive_failures: int = 0
    uptime: float = 0.0

class BaseAgent(ABC):
    """
    OpenClaw Base Agent Interface (SOTA Injected).
    
    Advanced Features:
    - Persistent State: Auto-recovery from disk.
    - Circuit Breaker: Safety cut-off on repetitive failures.
    - Hooks: Real-time telemetry integration.
    """

    def __init__(self, name: str, role: str, persistence_dir: Optional[str] = None) -> None:
        self.name = name
        self.role = role
        self.memory: List[MemoryEntry] = []
        self.start_time = time.time()
        self.consecutive_failures = 0
        self.circuit_state = "CLOSED"
        self.persistence_path = Path(persistence_dir) / f"{name}_state.json" if persistence_dir else None
        
        # SOTA Hooks (Observer Pattern)
        self.on_query: List[Callable] = []
        self.on_response: List[Callable] = []
        
        if self.persistence_path and self.persistence_path.exists():
            self._load_state()

    async def run_shell_command(self, command: str) -> str:
        """SOTA Tool: Execute a real shell command via the Execution Kernel."""
        logger.info(f"Agent {self.name} is requesting shell execution: {command}")
        res = exec_kernel.run_command(command)
        if res.exit_code == 0:
            return f"Success: {res.stdout}"
        else:
            return f"Error: {res.stderr}"

    @abstractmethod
    async def process(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """Process a query with Circuit Breaker safety."""
        pass

    async def safe_process(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Wrapper with Circuit Breaker and Telemetry."""
        if self.circuit_state == "OPEN":
            logger.warning(f"Circuit OPEN for agent {self.name}. Rejecting query.")
            return AgentResponse(content=f"Error: Agent {self.name} is in cooling-off period.", action_type="error")

        # Telemetry: Hook Start
        for hook in self.on_query: await hook(query)

        try:
            response = await self.process(query, context)
            self.consecutive_failures = 0
            # Telemetry: Hook Success
            for hook in self.on_response: await hook(response)
            return response
        except Exception as e:
            self.consecutive_failures += 1
            if self.consecutive_failures >= 3:
                self.circuit_state = "OPEN"
                logger.error(f"Circuit TRIPPED for agent {self.name} after {self.consecutive_failures} failures.")
            raise e

    def add_to_memory(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        entry = MemoryEntry(
            sequence=len(self.memory) + 1,
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.memory.append(entry)
        if self.persistence_path:
            self._save_state()

    def _save_state(self):
        """Forensic persistence to disk."""
        if not self.persistence_path: return
        state = {
            "name": self.name,
            "memory": [m.model_dump() for m in self.memory],
            "failures": self.consecutive_failures,
            "circuit": self.circuit_state
        }
        self.persistence_path.write_text(json.dumps(state, indent=2))

    def _load_state(self):
        """Restore agent from forensic state."""
        try:
            state = json.loads(self.persistence_path.read_text())
            self.memory = [MemoryEntry(**m) for m in state.get("memory", [])]
            self.consecutive_failures = state.get("failures", 0)
            self.circuit_state = state.get("circuit", "CLOSED")
            logger.info(f"Agent {self.name} state restored from {self.persistence_path}")
        except Exception as e:
            logger.error(f"Failed to restore agent state: {e}")

    def get_status(self) -> AgentStatus:
        return AgentStatus(
            name=self.name,
            role=self.role,
            memory_size=len(self.memory),
            is_active=self.circuit_state != "OPEN",
            circuit_state=self.circuit_state,
            consecutive_failures=self.consecutive_failures,
            uptime=time.time() - self.start_time
        )
