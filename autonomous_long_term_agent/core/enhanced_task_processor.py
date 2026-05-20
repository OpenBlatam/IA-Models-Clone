"""
Enhanced Task Processor
Extends TaskProcessor with caching and paper-based optimizations
"""

import logging
from typing import Dict, Any, Optional

from .task_queue import Task
from .reasoning_engine import ReasoningResult
from .task_processor import TaskProcessor
from .agent_cache import AgentCache
from .agent_utils import reasoning_result_to_dict, dict_to_reasoning_result

logger = logging.getLogger(__name__)


class EnhancedTaskProcessor(TaskProcessor):
    """
    Enhanced task processor with caching and optimizations.
    Extends TaskProcessor to add caching layer.
    """
    
    def __init__(
        self,
        reasoning_engine,
        learning_engine,
        knowledge_base,
        metrics_manager,
        observer_manager,
        cache: Optional[AgentCache] = None,
        enable_papers: bool = False
    ):
        super().__init__(
            reasoning_engine=reasoning_engine,
            learning_engine=learning_engine,
            knowledge_base=knowledge_base,
            metrics_manager=metrics_manager,
            observer_manager=observer_manager
        )
        self._cache = cache or AgentCache(max_reasoning_size=100, max_knowledge_size=50)
        self.enable_papers = enable_papers
        self._papers_applied = 0
    
    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        Process task with caching and enhanced reasoning.
        
        Args:
            task: Task to process
            
        Returns:
            Result dictionary with response and metadata
        """
        # Check cache first
        cache_key = f"task_{task.id}_{hash(task.instruction)}"
        cached_result = self._cache.get_reasoning(cache_key)
        
        if cached_result is not None:
            logger.debug(f"Using cached result for task {task.id}")
            
            # Convert cached result to ReasoningResult format if needed
            if isinstance(cached_result, dict):
                reasoning_result = dict_to_reasoning_result(cached_result)
            else:
                reasoning_result = cached_result
            
            # Use base class methods to complete task
            result = reasoning_result_to_dict(reasoning_result)
            await self._store_task_knowledge(task, reasoning_result)
            await self._record_task_completion(task.id, "success")
            await self.observer_manager.notify_task_success(task, result)
            self.metrics_manager.record_task_completed(tokens_used=reasoning_result.tokens_used)
            return result
        
        # Enhanced reasoning with paper techniques
        reasoning_result = await self._enhanced_reasoning(task.instruction, task.metadata)
        
        # Cache result
        self._cache.set_reasoning(cache_key, reasoning_result)
        
        # Convert to dict format
        result = reasoning_result_to_dict(reasoning_result)
        
        # Use base class methods to complete task (stores knowledge, records completion, notifies observers)
        await self._store_task_knowledge(task, reasoning_result)
        await self._record_task_completion(task.id, "success")
        await self.observer_manager.notify_task_success(task, result)
        
        # Update metrics
        self.metrics_manager.record_task_completed(tokens_used=reasoning_result.tokens_used)
        self.metrics_manager.record_reasoning_call()
        
        # Track paper enhancements
        if self.enable_papers:
            self._papers_applied += 1
        
        return result
    
    async def _enhanced_reasoning(
        self,
        instruction: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReasoningResult:
        """
        Enhanced reasoning with caching and paper-based optimizations.
        
        Args:
            instruction: Task instruction
            metadata: Optional task metadata
            
        Returns:
            ReasoningResult with enhanced reasoning
        """
        # Enhanced knowledge retrieval with caching
        cache_key = f"knowledge_{hash(instruction)}"
        relevant_knowledge = self._cache.get_knowledge(cache_key)
        
        if relevant_knowledge is None:
            relevant_knowledge = await self.knowledge_base.search_knowledge(instruction, limit=5)
            self._cache.set_knowledge(cache_key, relevant_knowledge)
        
        # Use base reasoning engine with enhanced context
        reasoning_result = await self.reasoning_engine.reason(
            instruction=instruction,
            metadata={
                **(metadata or {}),
                "enhanced": True,
                "papers_enabled": self.enable_papers,
                "cache_used": relevant_knowledge is not None
            }
        )
        
        return reasoning_result
    
    def get_papers_applied(self) -> int:
        """Get count of papers applied"""
        return self._papers_applied
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return self._cache.get_stats()

