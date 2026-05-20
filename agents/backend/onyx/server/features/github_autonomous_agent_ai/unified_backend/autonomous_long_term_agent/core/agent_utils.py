"""
Agent Utilities
Shared utility functions for agent operations
"""

from typing import Dict, Any
from .reasoning_engine import ReasoningResult


def reasoning_result_to_dict(result: ReasoningResult) -> Dict[str, Any]:
    """
    Convert ReasoningResult to dictionary format.
    
    This is the single source of truth for converting ReasoningResult
    to the dictionary format used throughout the system.
    
    Args:
        result: ReasoningResult instance
        
    Returns:
        Dictionary representation of the result
    """
    return {
        "response": result.response,
        "tokens_used": result.tokens_used,
        "reasoning_steps": result.reasoning_steps,
        "confidence": result.confidence
    }


def dict_to_reasoning_result(data: Dict[str, Any]) -> ReasoningResult:
    """
    Convert dictionary to ReasoningResult.
    
    Args:
        data: Dictionary with reasoning result data
        
    Returns:
        ReasoningResult instance
    """
    return ReasoningResult(
        response=data.get("response", ""),
        tokens_used=data.get("tokens_used", 0),
        reasoning_steps=data.get("reasoning_steps", []),
        confidence=data.get("confidence", 0.0)
    )

