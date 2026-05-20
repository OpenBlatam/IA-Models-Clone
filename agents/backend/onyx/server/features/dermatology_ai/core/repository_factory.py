"""
Repository factory for creating domain repositories
Extracted from composition_root.py for better organization
"""

from typing import Dict, Any
import logging

from .domain.interfaces import (
    IAnalysisRepository,
    IUserRepository,
    IProductRepository,
)
from .infrastructure.repositories import (
    AnalysisRepository,
    UserRepository,
    ProductRepository,
)
from .infrastructure.adapters import IDatabaseAdapter

logger = logging.getLogger(__name__)


class RepositoryFactory:
    """Factory for creating domain repositories"""
    
    @staticmethod
    def create_repositories(database_adapter: IDatabaseAdapter) -> Dict[str, Any]:
        """
        Create all domain repositories
        
        Args:
            database_adapter: Database adapter instance
        
        Returns:
            Dictionary mapping repository names to instances
        """
        return {
            "analysis_repository": AnalysisRepository(database_adapter),
            "user_repository": UserRepository(database_adapter),
            "product_repository": ProductRepository(database_adapter),
        }







