"""
Consensus service for aggregating model responses
Handles consensus algorithms and response aggregation
"""

import logging
from typing import List, Optional, Dict
from ...api.schemas import ModelResponse, ModelConfig
from ..consensus import apply_consensus
from ...api.helpers import get_weights_map

logger = logging.getLogger(__name__)


class ConsensusService:
    """Service for consensus and response aggregation"""
    
    def aggregate_responses(
        self,
        responses: List[ModelResponse],
        strategy: str,
        consensus_method: str = "majority",
        weights: Optional[Dict[str, float]] = None,
        enabled_models: Optional[List[ModelConfig]] = None
    ) -> Optional[str]:
        """
        Aggregate multiple model responses
        
        Args:
            responses: List of model responses
            strategy: Execution strategy (parallel, sequential, consensus)
            consensus_method: Consensus method to use
            weights: Optional weights for weighted voting
            enabled_models: Optional list of enabled models for weight calculation
            
        Returns:
            Aggregated response string or None
        """
        # Filter successful responses - optimized single pass
        successful_responses = [
            r for r in responses
            if r.success and r.response
        ]
        
        if not successful_responses:
            logger.warning("No successful responses to aggregate")
            return None
        
        # Early return for single response
        if len(successful_responses) == 1:
            return successful_responses[0].response
        
        # Apply consensus for consensus strategy
        if strategy == "consensus":
            # Use provided weights or calculate from enabled models
            if weights is None and enabled_models:
                weights = get_weights_map(enabled_models)
            
            return apply_consensus(
                successful_responses,
                consensus_method,
                weights
            )
        
        # For parallel/sequential, combine all responses
        # Optimized: use list comprehension with pre-formatted strings
        parts = [
            f"**{r.model_type.value}** (latency: {r.latency_ms:.2f}ms):\n{r.response}"
            if r.latency_ms is not None
            else f"**{r.model_type.value}** (latency: N/A):\n{r.response}"
            for r in successful_responses
        ]
        
        return "\n\n---\n\n".join(parts)




