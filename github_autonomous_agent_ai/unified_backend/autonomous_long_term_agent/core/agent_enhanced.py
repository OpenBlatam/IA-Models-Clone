"""
Enhanced Autonomous Long-Term Agent
Integrates research papers and optimization techniques for improved performance
"""

import asyncio
import logging
import uuid
import sys
from pathlib import Path
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Add optimization_core to path if available
_optimization_core_path = Path(__file__).parent.parent.parent.parent / "Frontier-Model-run-polyglot" / "scripts" / "TruthGPT-main" / "optimization_core"
if _optimization_core_path.exists() and str(_optimization_core_path) not in sys.path:
    sys.path.insert(0, str(_optimization_core_path))

try:
    from core.papers import (
        ModelEnhancer,
        PaperRegistry,
        EnhancementConfig,
        get_paper_registry
    )
    from core.papers.paper_validator import PaperValidator
    PAPERS_AVAILABLE = True
except ImportError:
    PAPERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Papers module not available, running in basic mode")

from ..config import settings
from .agent import AutonomousLongTermAgent, AgentStatus
from .reasoning_engine import ReasoningResult
from .agent_cache import AgentCache
from .enhanced_task_processor import EnhancedTaskProcessor
from .task_queue import Task

logger = logging.getLogger(__name__)


class EnhancedAutonomousAgent(AutonomousLongTermAgent):
    """
    Enhanced autonomous agent with paper-based optimizations
    
    Inherits from AutonomousLongTermAgent and adds:
    - Paper-based reasoning enhancements
    - Optimized knowledge retrieval with caching
    - Better long-horizon reasoning
    - Performance optimizations
    - Enhanced metrics and monitoring
    
    Fully compatible with AutonomousLongTermAgent API.
    """
    
    def __init__(
        self,
        agent_id: Optional[str] = None,
        instruction: Optional[str] = None,
        enable_papers: bool = True
    ):
        # Initialize base class first
        super().__init__(agent_id=agent_id, instruction=instruction)
        
        # Enhanced features
        self.enable_papers = enable_papers and PAPERS_AVAILABLE
        
        # Paper-based enhancements
        if self.enable_papers:
            try:
                self.paper_registry = get_paper_registry()
                self.model_enhancer = ModelEnhancer(self.paper_registry)
                self._setup_paper_enhancements()
                logger.info("✅ Paper enhancements enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize papers: {e}")
                self.enable_papers = False
        else:
            self.paper_registry = None
            self.model_enhancer = None
        
        # Performance optimizations (additional to base class)
        self._cache = AgentCache(
            max_reasoning_size=100,
            max_knowledge_size=50
        )
        
        # Replace base TaskProcessor with EnhancedTaskProcessor
        self._task_processor = EnhancedTaskProcessor(
            reasoning_engine=self.reasoning_engine,
            learning_engine=self.learning_engine,
            knowledge_base=self.knowledge_base,
            metrics_manager=self._metrics_manager,
            observer_manager=self._observer_manager,
            cache=self._cache,
            enable_papers=self.enable_papers
        )
    
    def _setup_paper_enhancements(self):
        """Setup paper-based enhancements for reasoning"""
        if not self.enable_papers:
            return
        
        try:
            # Get papers for memory and reasoning improvements
            memory_papers = self.paper_registry.search_papers(
                category="memory",
                min_speedup=1.5
            )
            reasoning_papers = self.paper_registry.search_papers(
                category="research",
                min_accuracy=5.0
            )
            
            # Log available enhancements
            logger.info(f"Found {len(memory_papers)} memory papers")
            logger.info(f"Found {len(reasoning_papers)} reasoning papers")
            
        except Exception as e:
            logger.warning(f"Error setting up paper enhancements: {e}")
    
    async def start(self) -> None:
        """Start the enhanced autonomous agent"""
        if self.status == AgentStatus.RUNNING:
            logger.warning(f"Agent {self.agent_id} already running")
            return
        
        logger.info(f"🚀 Starting enhanced autonomous agent {self.agent_id}")
        if self.enable_papers:
            logger.info("📚 Paper-based optimizations enabled")
        logger.info("⚠️  Agent will run continuously until explicitly stopped")
        
        # Use base class start (handles loop, metrics, etc.)
        await super().start()
    
    async def stop(self) -> None:
        """Stop the enhanced autonomous agent"""
        logger.info(f"⏹️  Stopping enhanced agent {self.agent_id}")
        await super().stop()
        logger.info(f"✅ Enhanced agent {self.agent_id} stopped")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get enhanced agent status and metrics (extends base)"""
        # Get base status
        base_status = await super().get_status()
        
        # Add enhanced information
        base_status["enhanced"] = True
        base_status["papers_enabled"] = self.enable_papers
        
        # Get paper registry stats if available
        if self.enable_papers and self.paper_registry:
            try:
                base_status["paper_stats"] = self.paper_registry.get_statistics()
            except Exception:
                pass
        
        # Add cache statistics from EnhancedTaskProcessor
        if isinstance(self._task_processor, EnhancedTaskProcessor):
            cache_stats = self._task_processor.get_cache_stats()
            base_status["cache_stats"] = cache_stats
            
            # Add enhanced metrics to existing metrics
            base_status["metrics"]["papers_applied"] = self._task_processor.get_papers_applied()
            base_status["metrics"]["cache_hit_rate"] = cache_stats.get("cache_hit_rate", 0.0)
        
        return base_status

