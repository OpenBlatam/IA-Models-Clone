"""
Experience Driven Learning
Implements retrieval-based learning from past experiences
"""

import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from .async_helpers import safe_async_call

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """A single experience record"""
    task_id: str
    instruction: str
    plan: Optional[str]
    result: str
    outcome: str  # success/failure
    timestamp: datetime
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "plan": self.plan,
            "result": self.result,
            "outcome": self.outcome,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experience":
        return cls(
            task_id=data["task_id"],
            instruction=data["instruction"],
            plan=data.get("plan"),
            result=data["result"],
            outcome=data["outcome"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data.get("metadata", {})
        )


class ExperienceDrivenLearning:
    """
    Experience Driven Learning Module
    Stores and retrieves past experiences to improve future performance.
    """

    def __init__(
        self,
        storage_path: str = "./data/unified_ai/experience",
        max_experiences: int = 1000
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_experiences = max_experiences
        self._experiences: List[Experience] = []
        self._lock = asyncio.Lock()
        self._load_from_disk()

    def _get_storage_file(self) -> Path:
        return self.storage_path / "experiences.json"

    def _load_from_disk(self) -> None:
        storage_file = self._get_storage_file()
        if not storage_file.exists():
            return

        try:
            with open(storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for exp_data in data.get("experiences", []):
                    self._experiences.append(Experience.from_dict(exp_data))
            logger.info(f"Loaded {len(self._experiences)} experiences")
        except Exception as e:
            logger.error(f"Error loading experiences: {e}")

    async def _save_to_disk(self) -> None:
        storage_file = self._get_storage_file()
        try:
            async with self._lock:
                data = {
                    "experiences": [e.to_dict() for e in self._experiences],
                    "last_updated": datetime.utcnow().isoformat()
                }
                with open(storage_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving experiences: {e}")

    async def add_experience(
        self,
        task_id: str,
        instruction: str,
        result: str,
        plan: Optional[str] = None,
        outcome: str = "success",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new experience"""
        experience = Experience(
            task_id=task_id,
            instruction=instruction,
            plan=plan,
            result=result,
            outcome=outcome,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )

        async with self._lock:
            self._experiences.append(experience)
            if len(self._experiences) > self.max_experiences:
                self._experiences = self._experiences[-self.max_experiences:]

        await self._save_to_disk()
        logger.debug(f"Added experience for task {task_id}")

    async def retrieve_similar(
        self,
        instruction: str,
        limit: int = 3
    ) -> List[Experience]:
        """
        Retrieve similar experiences based on simple keyword matching.
        In a production system, this would use vector embeddings.
        """
        instruction_words = set(instruction.lower().split())
        
        scored_experiences = []
        async with self._lock:
            for exp in self._experiences:
                # Simple Jaccard similarity on words
                exp_words = set(exp.instruction.lower().split())
                intersection = instruction_words.intersection(exp_words)
                union = instruction_words.union(exp_words)
                
                if not union:
                    score = 0
                else:
                    score = len(intersection) / len(union)
                
                if score > 0.1: # Threshold
                    scored_experiences.append((score, exp))

        # Sort by score descending
        scored_experiences.sort(key=lambda x: x[0], reverse=True)
        
        return [exp for _, exp in scored_experiences[:limit]]

    async def format_experiences_for_prompt(self, experiences: List[Experience]) -> str:
        """Format experiences for inclusion in LLM prompt"""
        if not experiences:
            return ""

        formatted = "Here are some similar past tasks and their solutions:\n\n"
        for i, exp in enumerate(experiences, 1):
            formatted += f"--- Example {i} ---\n"
            formatted += f"Task: {exp.instruction}\n"
            if exp.plan:
                formatted += f"Plan: {exp.plan}\n"
            formatted += f"Result: {exp.result}\n\n"
        
        return formatted
