"""
Fast Content Editor with Optimizations
"""

import logging
from typing import Dict, Any, Optional, List
from functools import lru_cache

from .editor import ContentEditor
from .fast_ai_engine import FastAIEngine, create_fast_ai_engine

logger = logging.getLogger(__name__)


class FastContentEditor(ContentEditor):
    """Fast version of ContentEditor with optimizations"""
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        use_fast_ai: bool = True,
        batch_size: int = 32
    ):
        """
        Initialize fast content editor
        
        Args:
            config: Configuration
            use_fast_ai: Use fast AI engine
            batch_size: Batch size for operations
        """
        super().__init__(config)
        
        self.batch_size = batch_size
        
        # Replace AI engine with fast version
        if use_fast_ai:
            self.ai_engine = create_fast_ai_engine(
                config=config,
                use_gpu=True,
                batch_size=batch_size
            )
            # Also update analyzer's AI engine if it has one
            if hasattr(self.analyzer, 'ai_engine'):
                self.analyzer.ai_engine = self.ai_engine
        
        logger.info(f"FastContentEditor initialized with batch_size={batch_size}")
    
    @lru_cache(maxsize=500)
    def add_cached(
        self,
        content: str,
        addition: str,
        position: str = "end"
    ) -> Dict[str, Any]:
        """
        Cached add operation
        
        Args:
            content: Original content
            addition: Content to add
            position: Position to add
            
        Returns:
            Result dictionary
        """
        return super().add(content, addition, position)
    
    def add_batch(
        self,
        operations: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Batch add operations
        
        Args:
            operations: List of {content, addition, position} dicts
            
        Returns:
            List of results
        """
        results = []
        for i in range(0, len(operations), self.batch_size):
            batch = operations[i:i+self.batch_size]
            batch_results = [
                self.add(
                    op.get("content", ""),
                    op.get("addition", ""),
                    op.get("position", "end")
                )
                for op in batch
            ]
            results.extend(batch_results)
        return results
    
    def remove_batch(
        self,
        operations: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Batch remove operations
        
        Args:
            operations: List of {content, pattern} dicts
            
        Returns:
            List of results
        """
        results = []
        for i in range(0, len(operations), self.batch_size):
            batch = operations[i:i+self.batch_size]
            batch_results = [
                self.remove(
                    op.get("content", ""),
                    op.get("pattern", "")
                )
                for op in batch
            ]
            results.extend(batch_results)
        return results
    
    def analyze_batch(self, contents: List[str]) -> List[Dict[str, Any]]:
        """
        Batch content analysis
        
        Args:
            contents: List of contents to analyze
            
        Returns:
            List of analysis results
        """
        if hasattr(self, 'ai_engine') and self.ai_engine:
            return self.ai_engine.analyze_batch(contents)
        return [self.analyzer.analyze(c) for c in contents]


def create_fast_editor(
    config: Optional[Dict[str, Any]] = None,
    batch_size: int = 32
) -> FastContentEditor:
    """Factory function to create fast editor"""
    return FastContentEditor(config=config, batch_size=batch_size)

