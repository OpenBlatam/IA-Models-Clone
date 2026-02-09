"""
Use Case Tests
Tests for application layer use cases
"""

import pytest
from unittest.mock import Mock, AsyncMock

from core.application import (
    AnalyzeImageUseCase,
    GetRecommendationsUseCase,
    GetAnalysisHistoryUseCase,
)
from core.domain.entities import Analysis, AnalysisStatus, SkinMetrics, Condition
from core.domain.interfaces import IAnalysisRepository, IImageProcessor, IAnalysisService


@pytest.mark.asyncio
async def test_analyze_image_use_case(mock_analysis_repository, mock_image_processor):
    """Test analyze image use case"""
    # Create mock analysis service
    mock_analysis_service = Mock(spec=IAnalysisService)
    mock_analysis_service.analyze_image = AsyncMock(return_value=Analysis(
        id="test-1",
        user_id="user-1",
        metrics=SkinMetrics(overall_score=80.0, texture_score=85.0, hydration_score=75.0,
                           elasticity_score=80.0, pigmentation_score=85.0, pore_size_score=75.0,
                           wrinkles_score=80.0, redness_score=85.0, dark_spots_score=80.0),
        conditions=[Condition(name="acne", confidence=0.65, severity="moderate")],
        status=AnalysisStatus.COMPLETED
    ))
    
    # Create use case
    use_case = AnalyzeImageUseCase(
        analysis_repository=mock_analysis_repository,
        image_processor=mock_image_processor,
        analysis_service=mock_analysis_service
    )
    
    # Execute
    result = await use_case.execute(
        user_id="user-1",
        image_data=b"image_bytes"
    )
    
    # Assertions
    assert result.status == AnalysisStatus.COMPLETED
    assert result.metrics is not None
    mock_analysis_repository.create.assert_called_once()
    mock_analysis_repository.update.assert_called_once()
    mock_image_processor.validate.assert_called_once()


@pytest.mark.asyncio
async def test_get_history_use_case(mock_analysis_repository):
    """Test get history use case"""
    # Setup mock
    mock_analysis = Analysis(
        id="test-1",
        user_id="user-1",
        status=AnalysisStatus.COMPLETED
    )
    mock_analysis_repository.get_by_user = AsyncMock(return_value=[mock_analysis])
    
    # Create use case
    use_case = GetAnalysisHistoryUseCase(
        analysis_repository=mock_analysis_repository
    )
    
    # Execute
    result = await use_case.execute(
        user_id="user-1",
        limit=10,
        offset=0
    )
    
    # Assertions
    assert len(result) == 1
    assert result[0].id == "test-1"
    mock_analysis_repository.get_by_user.assert_called_once_with("user-1", 10)













