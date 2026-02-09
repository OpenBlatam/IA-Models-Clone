"""
Connection Pool

Utilities for connection pooling.
"""

import logging
from typing import Callable, Optional
from core.pool.resource_pool import ResourcePool

logger = logging.getLogger(__name__)


class ConnectionPool(ResourcePool):
    """Pool for managing connections."""
    
    def __init__(
        self,
        connection_factory: Callable,
        max_connections: int = 10,
        min_connections: int = 2
    ):
        """
        Initialize connection pool.
        
        Args:
            connection_factory: Factory function to create connections
            max_connections: Maximum connections
            min_connections: Minimum connections
        """
        super().__init__(
            factory=connection_factory,
            max_size=max_connections,
            min_size=min_connections
        )
        logger.info(f"Connection pool initialized: {min_connections}-{max_connections} connections")
    
    def get_connection(self, timeout: Optional[float] = None):
        """Get connection from pool."""
        return self.get(timeout)
    
    def return_connection(self, connection) -> None:
        """Return connection to pool."""
        self.put(connection)


def create_connection_pool(
    connection_factory: Callable,
    **kwargs
) -> ConnectionPool:
    """Create connection pool."""
    return ConnectionPool(connection_factory, **kwargs)



