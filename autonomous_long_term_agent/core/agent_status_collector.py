"""
Agent Status Collector
Centralizes status collection from optional components
Follows Single Responsibility Principle
"""

import logging
from typing import Dict, Any, Optional, Callable, Awaitable, List, Tuple

logger = logging.getLogger(__name__)


class StatusCollector:
    """
    Collects status from optional components.
    Handles errors gracefully and provides consistent interface.
    """
    
    @staticmethod
    async def collect_optional_status(
        component_name: str,
        component: Optional[Any],
        get_stats_coroutine: Awaitable[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Collect status from an optional component.
        
        Args:
            component_name: Name of component for logging
            component: Component instance (may be None)
            get_stats_coroutine: Awaitable coroutine that returns stats (already called)
            
        Returns:
            Stats dictionary or None if component not available or error
        """
        if not component:
            return None
        
        try:
            return await get_stats_coroutine
        except Exception as e:
            logger.warning(f"Error getting {component_name} stats: {e}")
            return None
    
    @staticmethod
    async def collect_status_dict(
        status_dict: Dict[str, Any],
        component_name: str,
        component: Optional[Any],
        get_stats_method: Optional[Callable[[], Awaitable[Dict[str, Any]]]]
    ) -> None:
        """
        Collect status and add to status dictionary.
        
        Args:
            status_dict: Dictionary to update
            component_name: Name of component
            component: Component instance
            get_stats_method: Async method to get stats (callable, not result)
        """
        if component and get_stats_method:
            try:
                stats = await get_stats_method()
                if stats:
                    status_dict[component_name] = stats
            except Exception as e:
                logger.warning(f"Error getting {component_name} stats: {e}")
    
    @staticmethod
    async def collect_multiple_status(
        components: List[Tuple[str, Optional[Any], Callable[[], Awaitable[Dict[str, Any]]]]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect status from multiple optional components in one call.
        
        Args:
            components: List of tuples (status_key, component, get_stats_method)
                - status_key: Key to use in returned dictionary
                - component: Component instance (may be None)
                - get_stats_method: Async method to get stats
        
        Returns:
            Dictionary mapping status_key to stats (only includes available components)
        """
        results = {}
        
        for status_key, component, get_stats_method in components:
            if component and get_stats_method:
                try:
                    stats = await get_stats_method()
                    if stats:
                        results[status_key] = stats
                except Exception as e:
                    logger.warning(f"Error getting {status_key} stats: {e}")
        
        return results

