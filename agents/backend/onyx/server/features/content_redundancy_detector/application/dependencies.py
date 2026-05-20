"""
Dependency Injection - FastAPI dependencies for services
"""

from typing import Annotated
from fastapi import Depends, Request
from functools import lru_cache

from ..domain.interfaces import (
    IAnalysisRepository,
    ICacheService,
    IMLService,
    IEventBus,
    IExportService
)
from ..infrastructure.adapters import (
    InMemoryAnalysisRepository,
    RedisCacheAdapter,
    AIMLServiceAdapter,
    InMemoryEventBus,
    ExportServiceAdapter
)
from ..core.config import get_settings
from .services import AnalysisService, BatchService, QualityService


# Adapter instances (singletons)
_cache_adapter: ICacheService = None
_repository: IAnalysisRepository = None
_ml_service: IMLService = None
_event_bus: IEventBus = None
_export_service: IExportService = None


def get_cache_service() -> ICacheService:
    """Get cache service adapter"""
    global _cache_adapter
    if _cache_adapter is None:
        settings = get_settings()
        _cache_adapter = RedisCacheAdapter(settings)
    return _cache_adapter


def get_analysis_repository() -> IAnalysisRepository:
    """Get analysis repository adapter"""
    global _repository
    if _repository is None:
        _repository = InMemoryAnalysisRepository()
    return _repository


def get_ml_service() -> IMLService:
    """Get ML service adapter"""
    global _ml_service
    if _ml_service is None:
        settings = get_settings()
        _ml_service = AIMLServiceAdapter(settings)
    return _ml_service


def get_event_bus() -> IEventBus:
    """Get event bus adapter"""
    global _event_bus
    if _event_bus is None:
        _event_bus = InMemoryEventBus()
    return _event_bus


def get_export_service() -> IExportService:
    """Get export service adapter"""
    global _export_service
    if _export_service is None:
        _export_service = ExportServiceAdapter()
    return _export_service


# Application Services
def get_analysis_service(
    repository: Annotated[IAnalysisRepository, Depends(get_analysis_repository)],
    cache_service: Annotated[ICacheService, Depends(get_cache_service)],
    ml_service: Annotated[IMLService, Depends(get_ml_service)],
    event_bus: Annotated[IEventBus, Depends(get_event_bus)]
) -> AnalysisService:
    """Get analysis service with injected dependencies"""
    return AnalysisService(
        repository=repository,
        cache_service=cache_service,
        ml_service=ml_service,
        event_bus=event_bus
    )


def get_batch_service(
    analysis_service: Annotated[AnalysisService, Depends(get_analysis_service)]
) -> BatchService:
    """Get batch processing service"""
    return BatchService(analysis_service=analysis_service)


def get_quality_service(
    ml_service: Annotated[IMLService, Depends(get_ml_service)],
    cache_service: Annotated[ICacheService, Depends(get_cache_service)]
) -> QualityService:
    """Get quality assessment service"""
    return QualityService(
        ml_service=ml_service,
        cache_service=cache_service
    )
