"""
State Management and Persistence
Handles agent state persistence and recovery
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class AgentState:
    """Agent state for persistence"""
    agent_id: str
    instruction: str
    status: str
    metrics: Dict[str, Any]
    created_at: str
    last_saved: str


class StateManager:
    """Manages agent state persistence and recovery"""
    
    def __init__(self, storage_path: str = "./data/autonomous_agent/state"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
    
    def _get_state_file(self, agent_id: str) -> Path:
        """Get path to state file for agent"""
        return self.storage_path / f"{agent_id}.json"
    
    async def save_state(
        self,
        agent_id: str,
        instruction: str,
        status: str,
        metrics: Dict[str, Any],
        created_at: Optional[datetime] = None
    ) -> None:
        """Save agent state to disk"""
        try:
            async with self._lock:
                state = AgentState(
                    agent_id=agent_id,
                    instruction=instruction,
                    status=status,
                    metrics=metrics,
                    created_at=(created_at or datetime.utcnow()).isoformat(),
                    last_saved=datetime.utcnow().isoformat()
                )
                
                state_file = self._get_state_file(agent_id)
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump(asdict(state), f, indent=2)
                
                logger.debug(f"Saved state for agent {agent_id}")
        except Exception as e:
            logger.error(f"Error saving state for agent {agent_id}: {e}")
    
    async def load_state(self, agent_id: str) -> Optional[AgentState]:
        """Load agent state from disk"""
        try:
            state_file = self._get_state_file(agent_id)
            if not state_file.exists():
                return None
            
            async with self._lock:
                with open(state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return AgentState(**data)
        except Exception as e:
            logger.error(f"Error loading state for agent {agent_id}: {e}")
            return None
    
    async def delete_state(self, agent_id: str) -> None:
        """Delete agent state file"""
        try:
            state_file = self._get_state_file(agent_id)
            if state_file.exists():
                state_file.unlink()
                logger.debug(f"Deleted state for agent {agent_id}")
        except Exception as e:
            logger.error(f"Error deleting state for agent {agent_id}: {e}")
    
    async def list_all_states(self) -> list:
        """List all saved agent states"""
        try:
            states = []
            for state_file in self.storage_path.glob("*.json"):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        states.append(AgentState(**data))
                except Exception as e:
                    logger.warning(f"Error loading state from {state_file}: {e}")
            return states
        except Exception as e:
            logger.error(f"Error listing states: {e}")
            return []




