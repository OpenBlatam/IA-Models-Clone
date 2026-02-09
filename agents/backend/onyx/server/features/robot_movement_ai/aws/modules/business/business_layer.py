"""
Business Layer
==============

Main business layer class.
"""

import logging
from typing import Optional
from aws.modules.business.service_factory import ServiceFactory
from aws.modules.business.use_cases import UseCaseExecutor
from aws.modules.ports.repository_port import RepositoryPort
from aws.modules.ports.cache_port import CachePort
from aws.modules.ports.messaging_port import MessagingPort

logger = logging.getLogger(__name__)


class BusinessLayer:
    """Business layer manager."""
    
    def __init__(
        self,
        repository: Optional[RepositoryPort] = None,
        cache: Optional[CachePort] = None,
        messaging: Optional[MessagingPort] = None
    ):
        self.service_factory = ServiceFactory(
            repository=repository,
            cache=cache,
            messaging=messaging
        )
        self.use_case_executor = UseCaseExecutor()
    
    def get_service_factory(self) -> ServiceFactory:
        """Get service factory."""
        return self.service_factory
    
    def get_use_case_executor(self) -> UseCaseExecutor:
        """Get use case executor."""
        return self.use_case_executor















