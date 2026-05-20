"""
TruthGPT Client for Contabilidad Mexicana AI SAM3
=================================================

Client for TruthGPT integration with optimization and advanced features.

Refactored to:
- Use centralized helpers for common patterns
- Eliminate duplicate availability checks
- Simplify error handling
"""

import logging
from typing import Dict, Any, Optional

from .truthgpt_status import TruthGPTStatus
from .truthgpt_helpers import (
    check_truthgpt_ready,
    safe_truthgpt_call
)

# Import TruthGPT modules from truthgpt_status
# These are conditionally imported there
try:
    from optimization_core.utils.modules import (
        TruthGPTIntegrationManager,
        TruthGPTConfigManager,
        TruthGPTAnalyticsManager,
    )
except ImportError:
    TruthGPTIntegrationManager = None
    TruthGPTConfigManager = None
    TruthGPTAnalyticsManager = None

logger = logging.getLogger(__name__)


class TruthGPTClient:
    """
    Client for TruthGPT integration.
    
    Features:
    - Integration with TruthGPT optimization modules
    - Analytics and monitoring
    - Advanced processing capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TruthGPT client.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self._integration_manager = None
        self._analytics_manager = None
        self._config_manager = None
        
        if TruthGPTStatus.is_available():
            self._initialize_truthgpt()
        else:
            logger.warning("TruthGPT modules not available, running in limited mode")
    
    def _initialize_truthgpt(self):
        """Initialize TruthGPT modules."""
        if not TruthGPTStatus.is_available():
            return
        
        try:
            # Initialize managers
            self._config_manager = TruthGPTConfigManager()
            self._analytics_manager = TruthGPTAnalyticsManager()
            
            # Integration manager with API type
            integration_config = {
                "integration_type": "api",
                "api_endpoint": self.config.get("truthgpt_endpoint", ""),
                "timeout": self.config.get("timeout", 60.0),
            }
            self._integration_manager = TruthGPTIntegrationManager(integration_config)
            
            logger.info("TruthGPT modules initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing TruthGPT modules: {e}")
    
    async def process_with_truthgpt(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process query with TruthGPT optimization.
        
        Args:
            query: Query to process
            context: Additional context
            
        Returns:
            Processed result with TruthGPT enhancements
        """
        if not check_truthgpt_ready(self._integration_manager, self._analytics_manager):
            return TruthGPTStatus.get_fallback_response(query)
        
        try:
            # Prepare data for TruthGPT
            data = {
                "query": query,
                "context": context or {},
                "service": "contabilidad_mexicana_ai_sam3",
            }
            
            # Integrate with TruthGPT
            result = self._integration_manager.integrate(data)
            
            # Track analytics
            if self._analytics_manager:
                self._analytics_manager.track_query(
                    query=query,
                    service="contabilidad_mexicana_ai_sam3",
                    result=result
                )
            
            return {
                "result": result,
                "truthgpt_enhanced": True,
                "analytics": self._analytics_manager.get_stats() if self._analytics_manager else {}
            }
            
        except Exception as e:
            logger.error(f"Error processing with TruthGPT: {e}")
            return TruthGPTStatus.get_error_response(query, e)
    
    async def optimize_query(
        self,
        query: str,
        optimization_type: str = "standard"
    ) -> str:
        """
        Optimize query using TruthGPT optimization.
        
        Args:
            query: Original query
            optimization_type: Type of optimization (currently unused, reserved for future use)
            
        Returns:
            Optimized query or original query if optimization fails
        """
        if not TruthGPTStatus.is_available():
            return query
        
        async def _optimize():
            optimized = await self.process_with_truthgpt(query)
            return optimized.get("result", query)
        
        return await safe_truthgpt_call(
            query,
            _optimize,
            query,  # fallback to original query
            "optimize query"
        )
    
    async def get_analytics(self) -> Dict[str, Any]:
        """
        Get analytics from TruthGPT.
        
        Returns:
            Analytics dictionary or empty dict if unavailable
        """
        if not check_truthgpt_ready(self._integration_manager, self._analytics_manager):
            return {}
        
        try:
            return self._analytics_manager.get_stats() if self._analytics_manager else {}
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {}
    
    async def close(self):
        """Close TruthGPT client and cleanup resources."""
        logger.info("TruthGPTClient closed")

