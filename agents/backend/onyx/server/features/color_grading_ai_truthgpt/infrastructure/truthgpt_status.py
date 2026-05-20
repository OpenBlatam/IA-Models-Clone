"""
TruthGPT availability and status utilities.

Refactored to consolidate TruthGPT availability checks.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Try to import TruthGPT modules
try:
    import sys
    from pathlib import Path
    truthgpt_path = Path(__file__).parent.parent.parent.parent / "Frontier-Model-run-polyglot" / "scripts" / "TruthGPT-main"
    if truthgpt_path.exists():
        sys.path.insert(0, str(truthgpt_path))
    
    from optimization_core.utils.modules import (
        TruthGPTIntegrationManager,
        TruthGPTConfigManager,
        TruthGPTAnalyticsManager,
    )
    TRUTHGPT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"TruthGPT modules not available: {e}")
    TRUTHGPT_AVAILABLE = False


class TruthGPTStatus:
    """
    Manages TruthGPT availability and provides consistent status checks.
    
    Responsibilities:
    - Check TruthGPT availability
    - Provide fallback responses
    - Centralize availability logic
    """
    
    @staticmethod
    def is_available() -> bool:
        """Check if TruthGPT is available."""
        return TRUTHGPT_AVAILABLE
    
    @staticmethod
    def get_fallback_response(query: str, message: str = "TruthGPT not available") -> Dict[str, Any]:
        """
        Get fallback response when TruthGPT is not available.
        
        Args:
            query: Original query
            message: Fallback message
            
        Returns:
            Fallback response dictionary
        """
        return {
            "result": query,
            "truthgpt_enhanced": False,
            "message": message
        }
    
    @staticmethod
    def get_error_response(query: str, error: Exception) -> Dict[str, Any]:
        """
        Get error response when TruthGPT processing fails.
        
        Args:
            query: Original query
            error: Exception that occurred
            
        Returns:
            Error response dictionary
        """
        return {
            "result": query,
            "truthgpt_enhanced": False,
            "error": str(error)
        }




