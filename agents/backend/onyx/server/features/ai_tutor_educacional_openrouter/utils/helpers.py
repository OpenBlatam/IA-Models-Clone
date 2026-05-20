"""
Helper utility functions.
"""

from typing import Dict, Any, Optional
from ..config.tutor_config import TutorConfig


def format_response(data: Dict[str, Any], include_metadata: bool = True) -> Dict[str, Any]:
    """Format API response."""
    response = {
        "answer": data.get("answer", ""),
        "timestamp": data.get("timestamp", "")
    }
    
    if include_metadata:
        response["metadata"] = {
            "model": data.get("model", ""),
            "usage": data.get("usage", {})
        }
    
    return response


def validate_subject(subject: str, config: Optional[TutorConfig] = None) -> bool:
    """Validate subject is supported."""
    if config is None:
        config = TutorConfig()
    return subject.lower() in [s.lower() for s in config.subjects]


def validate_difficulty(difficulty: str, config: Optional[TutorConfig] = None) -> bool:
    """Validate difficulty level."""
    if config is None:
        config = TutorConfig()
    return difficulty.lower() in [d.lower() for d in config.difficulty_levels]






