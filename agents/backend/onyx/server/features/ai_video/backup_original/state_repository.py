from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from __future__ import annotations
import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional
from threading import Lock

from typing import Any, List, Dict, Optional
import asyncio
# Assuming WorkflowState is defined in video_workflow.py
# To avoid circular dependency, we can use a forward reference or a placeholder
# from .video_workflow import WorkflowState
class WorkflowState: ...

logger = logging.getLogger(__name__)

class StateRepository(ABC):
    """Abstract base class for state persistence."""

    @abstractmethod
    def save(self, state: WorkflowState) -> None:
        pass

    @abstractmethod
    def load(self, workflow_id: str) -> Optional[WorkflowState]:
        pass

class FileStateRepository(StateRepository):
    """
    A simple file-based repository for storing workflow state as JSON.
    Each workflow state is saved in a file named after its workflow_id.
    """
    def __init__(self, directory: str):
        
    """__init__ function."""
self.directory = directory
        os.makedirs(self.directory, exist_ok=True)
        self._lock = Lock()

    def _get_path(self, workflow_id: str) -> str:
        return os.path.join(self.directory, f"workflow_{workflow_id}.json")

    def save(self, state: WorkflowState) -> None:
        path = self._get_path(state.workflow_id)
        with self._lock:
            try:
                with open(path, 'w', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    # Exclude the lock from serialization
                    f.write(state.model_dump_json(indent=2))
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                logger.info(f"Workflow state for '{state.workflow_id}' saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save state for '{state.workflow_id}': {e}")

    def load(self, workflow_id: str) -> Optional[WorkflowState]:
        path = self._get_path(workflow_id)
        if not os.path.exists(path):
            return None
        
        with self._lock:
            try:
                with open(path, 'r', encoding='utf-8') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    data = json.load(f)
                    state = WorkflowState(**data)
                    logger.info(f"Workflow state for '{workflow_id}' loaded from {path}")
                    return state
            except Exception as e:
                logger.error(f"Failed to load or parse state for '{workflow_id}': {e}")
                return None 