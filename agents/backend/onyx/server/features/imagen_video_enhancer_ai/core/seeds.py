"""
Seeds System
============

System for seeding initial data and configurations.
"""

import asyncio
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class SeedStatus(Enum):
    """Seed status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Seed:
    """Seed definition."""
    id: str
    name: str
    description: str
    seed_function: Callable[[], Any]
    dependencies: List[str] = field(default_factory=list)
    enabled: bool = True
    priority: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "dependencies": self.dependencies,
            "enabled": self.enabled,
            "priority": self.priority
        }


class SeedRunner:
    """Seed runner and manager."""
    
    def __init__(self, seeds_dir: Optional[Path] = None):
        """
        Initialize seed runner.
        
        Args:
            seeds_dir: Directory for seed files and history
        """
        self.seeds_dir = seeds_dir or Path("seeds")
        self.seeds_dir.mkdir(parents=True, exist_ok=True)
        self.seeds: Dict[str, Seed] = {}
        self.history_file = self.seeds_dir / "seed_history.json"
        self._history: List[Dict[str, Any]] = []
        self._load_history()
    
    def _load_history(self):
        """Load seed history."""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self._history = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading seed history: {e}")
                self._history = []
    
    def _save_history(self):
        """Save seed history."""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self._history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving seed history: {e}")
    
    def register(self, seed: Seed):
        """
        Register a seed.
        
        Args:
            seed: Seed to register
        """
        self.seeds[seed.id] = seed
        logger.info(f"Registered seed: {seed.id} ({seed.name})")
    
    def get_applied_seeds(self) -> List[str]:
        """
        Get list of applied seed IDs.
        
        Returns:
            List of seed IDs
        """
        return [
            entry["seed_id"]
            for entry in self._history
            if entry.get("status") == SeedStatus.COMPLETED.value
        ]
    
    def get_pending_seeds(self) -> List[Seed]:
        """
        Get list of pending seeds.
        
        Returns:
            List of pending seeds
        """
        applied = set(self.get_applied_seeds())
        pending = [
            seed
            for seed in self.seeds.values()
            if seed.enabled and seed.id not in applied
        ]
        
        # Sort by priority and dependencies
        return self._sort_seeds(pending)
    
    def _sort_seeds(self, seeds: List[Seed]) -> List[Seed]:
        """Sort seeds by dependencies and priority."""
        sorted_seeds = []
        remaining = seeds.copy()
        applied = set(self.get_applied_seeds())
        
        # Sort by priority first
        remaining.sort(key=lambda s: s.priority, reverse=True)
        
        while remaining:
            progress = False
            for seed in remaining[:]:
                # Check if all dependencies are satisfied
                deps_satisfied = all(
                    dep in applied or any(s.id == dep for s in sorted_seeds)
                    for dep in seed.dependencies
                )
                
                if deps_satisfied:
                    sorted_seeds.append(seed)
                    remaining.remove(seed)
                    progress = True
            
            if not progress:
                # Circular dependency or missing dependency
                logger.warning("Could not resolve all dependencies")
                sorted_seeds.extend(remaining)
                break
        
        return sorted_seeds
    
    async def run_seed(self, seed_id: str, force: bool = False) -> bool:
        """
        Run a specific seed.
        
        Args:
            seed_id: Seed ID
            force: Force re-run even if already applied
            
        Returns:
            True if successful
        """
        if seed_id not in self.seeds:
            logger.error(f"Seed not found: {seed_id}")
            return False
        
        seed = self.seeds[seed_id]
        
        # Check if already applied
        if not force and seed_id in self.get_applied_seeds():
            logger.info(f"Seed already applied: {seed_id}")
            return True
        
        if not seed.enabled:
            logger.info(f"Seed disabled: {seed_id}")
            return False
        
        # Record start
        history_entry = {
            "seed_id": seed_id,
            "status": SeedStatus.RUNNING.value,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "error": None
        }
        self._history.append(history_entry)
        self._save_history()
        
        try:
            # Run seed
            if asyncio.iscoroutinefunction(seed.seed_function):
                await seed.seed_function()
            else:
                seed.seed_function()
            
            # Record success
            history_entry["status"] = SeedStatus.COMPLETED.value
            history_entry["completed_at"] = datetime.now().isoformat()
            self._save_history()
            
            logger.info(f"Seed completed: {seed_id}")
            return True
            
        except Exception as e:
            # Record failure
            history_entry["status"] = SeedStatus.FAILED.value
            history_entry["error"] = str(e)
            history_entry["completed_at"] = datetime.now().isoformat()
            self._save_history()
            
            logger.error(f"Seed failed: {seed_id} - {e}")
            return False
    
    async def run_all_pending(self, force: bool = False) -> Dict[str, bool]:
        """
        Run all pending seeds.
        
        Args:
            force: Force re-run even if already applied
            
        Returns:
            Dictionary of seed_id -> success
        """
        pending = self.get_pending_seeds()
        results = {}
        
        for seed in pending:
            success = await self.run_seed(seed.id, force=force)
            results[seed.id] = success
            
            if not success:
                # Continue on failure (seeds are usually independent)
                logger.warning(f"Seed failed but continuing: {seed.id}")
        
        return results
    
    def reset_seed(self, seed_id: str) -> bool:
        """
        Reset a seed (remove from history).
        
        Args:
            seed_id: Seed ID
            
        Returns:
            True if successful
        """
        if seed_id not in self.seeds:
            logger.error(f"Seed not found: {seed_id}")
            return False
        
        # Remove from history
        self._history = [
            entry for entry in self._history
            if entry.get("seed_id") != seed_id
        ]
        self._save_history()
        
        logger.info(f"Seed reset: {seed_id}")
        return True
    
    def get_seed_history(self, seed_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get seed history.
        
        Args:
            seed_id: Optional seed ID to filter
            
        Returns:
            List of history entries
        """
        if seed_id:
            return [
                entry for entry in self._history
                if entry.get("seed_id") == seed_id
            ]
        return self._history.copy()



