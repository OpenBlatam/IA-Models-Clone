"""
Comprehensive Use Case Tests
Tests for all application layer use cases
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from core.application.analyze_image_use_case import AnalyzeImageUseCase
from core.application.get_recommendations_use_case import GetRecommendationsUseCase
from core.application.get_history_use_case import GetAnalysisHistoryUseCase
from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    Recommendation,
    Product
)
from core.domain.interfaces import (
    IAnalysisRepository,
    IImageProcessor,
    IAnalysisService,
    IRecommendationService,
    IEventPublisher
)
from core.application.exceptions import ValidationError, ProcessingError
from tests.test_base import BaseUseCaseTest
from tests.test_helpers import build_analysis, build_metrics, create_service_mock


class TestAnalyzeImageUseCase(BaseUseCaseTest):
    """Tests for AnalyzeImageUseCase"""
    
    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_analysis_repository,
        mock_image_processor,
        mock_cache_service
    ):
        """Test successful image analysis"""
        # Setup mocks
        mock_analysis_service = Mock(spec=IAnalysisService)
        mock_analysis_service.analyze_image = AsyncMock(return_value=Analysis(
            id=str(uuid.uuid4()),
            user_id="user-123",
            metrics=SkinMetrics(
                overall_score=80.0,
                texture_score=85.0,
                hydration_score=75.0,
                elasticity_score=80.0,
                pigmentation_score=85.0,
                pore_size_score=75.0,
                wrinkles_score=80.0,
                redness_score=85.0,
                dark_spots_score=80.0
            ),
            conditions=[
                Condition(
                    name="acne",
                    confidence=0.65,
                    severity="moderate",
                    description="Mild acne detected"
                )
            ],
            skin_type=SkinType.COMBINATION,
            status=AnalysisStatus.COMPLETED
        ))
        
        mock_image_processor.validate = AsyncMock(return_value=True)
        mock_analysis_repository.create = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        ))
        mock_analysis_repository.update = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
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
            user_id="user-123",
            image_data=b"image_bytes",
            metadata={"filename": "test.jpg"}
        )
        
        # Assertions
        assert result.status == AnalysisStatus.COMPLETED
        assert result.metrics is not None
        assert result.metrics.overall_score == 80.0
        assert len(result.conditions) == 1
        mock_analysis_repository.create.assert_called_once()
        mock_analysis_repository.update.assert_called_once()
        mock_image_processor.validate.assert_called_once()
        mock_analysis_service.analyze_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_invalid_image(
        self,
        mock_analysis_repository,
        mock_image_processor
    ):
        """Test image analysis with invalid image"""
        mock_image_processor.validate = AsyncMock(return_value=False)
        
        mock_analysis_service = Mock(spec=IAnalysisService)
        
        use_case = AnalyzeImageUseCase(
            analysis_repository=mock_analysis_repository,
            image_processor=mock_image_processor,
            analysis_service=mock_analysis_service
        )
        
        with pytest.raises(ValidationError):
            await use_case.execute(
                user_id="user-123",
                image_data=b"invalid_image"
            )
    
    @pytest.mark.asyncio
    async def test_execute_with_event_publisher(
        self,
        mock_analysis_repository,
        mock_image_processor,
        mock_event_publisher
    ):
        """Test image analysis with event publishing"""
        mock_analysis_service = Mock(spec=IAnalysisService)
        mock_analysis_service.analyze_image = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            metrics=SkinMetrics(
                overall_score=80.0,
                texture_score=85.0,
                hydration_score=75.0,
                elasticity_score=80.0,
                pigmentation_score=85.0,
                pore_size_score=75.0,
                wrinkles_score=80.0,
                redness_score=85.0,
                dark_spots_score=80.0
            ),
            conditions=[],
            status=AnalysisStatus.COMPLETED
        ))
        
        mock_image_processor.validate = AsyncMock(return_value=True)
        mock_analysis_repository.create = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        ))
        mock_analysis_repository.update = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        ))
        
        use_case = AnalyzeImageUseCase(
            analysis_repository=mock_analysis_repository,
            image_processor=mock_image_processor,
            analysis_service=mock_analysis_service,
            event_publisher=mock_event_publisher
        )
        
        result = await use_case.execute(
            user_id="user-123",
            image_data=b"image_bytes"
        )
        
        assert result.status == AnalysisStatus.COMPLETED
        # Verify events were published (check if publish was called)
        assert mock_event_publisher.publish.called
    
    @pytest.mark.asyncio
    async def test_execute_processing_error(
        self,
        mock_analysis_repository,
        mock_image_processor
    ):
        """Test image analysis with processing error"""
        mock_analysis_service = Mock(spec=IAnalysisService)
        mock_analysis_service.analyze_image = AsyncMock(side_effect=Exception("Processing failed"))
        
        mock_image_processor.validate = AsyncMock(return_value=True)
        mock_analysis_repository.create = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        ))
        mock_analysis_repository.update = AsyncMock()
        
        use_case = AnalyzeImageUseCase(
            analysis_repository=mock_analysis_repository,
            image_processor=mock_image_processor,
            analysis_service=mock_analysis_service
        )
        
        with pytest.raises(ProcessingError):
            await use_case.execute(
                user_id="user-123",
                image_data=b"image_bytes"
            )
        
        # Verify failure was handled
        assert mock_analysis_repository.update.called


class TestGetRecommendationsUseCase:
    """Tests for GetRecommendationsUseCase"""
    
    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_analysis_repository,
        mock_cache_service
    ):
        """Test successful recommendations generation"""
        # Create mock analysis
        analysis = Analysis(
            id="test-1",
            user_id="user-123",
            metrics=SkinMetrics(
                overall_score=75.0,
                texture_score=80.0,
                hydration_score=70.0,
                elasticity_score=75.0,
                pigmentation_score=80.0,
                pore_size_score=70.0,
                wrinkles_score=75.0,
                redness_score=80.0,
                dark_spots_score=75.0
            ),
            conditions=[
                Condition(
                    name="acne",
                    confidence=0.65,
                    severity="moderate"
                )
            ],
            skin_type=SkinType.COMBINATION,
            status=AnalysisStatus.COMPLETED
        )
        
        mock_analysis_repository.get_by_id = AsyncMock(return_value=analysis)
        
        mock_recommendation_service = Mock(spec=IRecommendationService)
        mock_recommendation_service.generate_recommendations = AsyncMock(return_value={
            "routine": {
                "morning": [
                    Recommendation(
                        name="Cleanser",
                        category="cleanser",
                        description="Gentle cleanser",
                        priority=1
                    )
                ],
                "evening": [],
                "weekly": []
            },
            "specific_recommendations": [],
            "tips": []
        })
        
        use_case = GetRecommendationsUseCase(
            analysis_repository=mock_analysis_repository,
            recommendation_service=mock_recommendation_service
        )
        
        result = await use_case.execute(
            analysis_id="test-1",
            include_routine=True
        )
        
        assert result is not None
        assert "routine" in result
        assert "morning" in result["routine"]
        mock_analysis_repository.get_by_id.assert_called_once_with("test-1")
        mock_recommendation_service.generate_recommendations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_analysis_not_found(
        self,
        mock_analysis_repository
    ):
        """Test recommendations with non-existent analysis"""
        mock_analysis_repository.get_by_id = AsyncMock(return_value=None)
        
        mock_recommendation_service = Mock(spec=IRecommendationService)
        
        use_case = GetRecommendationsUseCase(
            analysis_repository=mock_analysis_repository,
            recommendation_service=mock_recommendation_service
        )
        
        with pytest.raises(ValidationError):
            await use_case.execute(
                analysis_id="non-existent",
                include_routine=True
            )


class TestGetHistoryUseCase:
    """Tests for GetHistoryUseCase"""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_analysis_repository):
        """Test successful history retrieval"""
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED,
                created_at=datetime.utcnow()
            )
            for i in range(5)
        ]
        
        mock_analysis_repository.get_by_user = AsyncMock(return_value=analyses)
        
        use_case = GetHistoryUseCase(
            analysis_repository=mock_analysis_repository
        )
        
        result = await use_case.execute(
            user_id="user-123",
            limit=10,
            offset=0
        )
        
        assert len(result) == 5
        assert all(a.user_id == "user-123" for a in result)
        mock_analysis_repository.get_by_user.assert_called_once_with("user-123", 10, 0)
    
    @pytest.mark.asyncio
    async def test_execute_empty_history(self, mock_analysis_repository):
        """Test history retrieval with no results"""
        mock_analysis_repository.get_by_user = AsyncMock(return_value=[])
        
        use_case = GetHistoryUseCase(
            analysis_repository=mock_analysis_repository
        )
        
        result = await use_case.execute(
            user_id="user-123",
            limit=10,
            offset=0
        )
        
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_pagination(self, mock_analysis_repository):
        """Test history retrieval with pagination"""
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED
            )
            for i in range(3)
        ]
        
        mock_analysis_repository.get_by_user = AsyncMock(return_value=analyses)
        
        use_case = GetAnalysisHistoryUseCase(
            analysis_repository=mock_analysis_repository
        )
        
        result = await use_case.execute(
            user_id="user-123",
            limit=3,
            offset=0
        )
        
        assert len(result) == 3
        # get_by_user is called with limit + offset for pagination
        mock_analysis_repository.get_by_user.assert_called_once_with("user-123", 3)


Tests for all application layer use cases
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
import uuid

from core.application.analyze_image_use_case import AnalyzeImageUseCase
from core.application.get_recommendations_use_case import GetRecommendationsUseCase
from core.application.get_history_use_case import GetAnalysisHistoryUseCase
from core.domain.entities import (
    Analysis,
    AnalysisStatus,
    SkinMetrics,
    Condition,
    SkinType,
    Recommendation,
    Product
)
from core.domain.interfaces import (
    IAnalysisRepository,
    IImageProcessor,
    IAnalysisService,
    IRecommendationService,
    IEventPublisher
)
from core.application.exceptions import ValidationError, ProcessingError
from tests.test_base import BaseUseCaseTest
from tests.test_helpers import build_analysis, build_user, build_metrics


class TestAnalyzeImageUseCase(BaseUseCaseTest):
    """Tests for AnalyzeImageUseCase"""
    
    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_analysis_repository,
        mock_image_processor,
        mock_cache_service
    ):
        """Test successful image analysis"""
        # Setup mocks
        mock_analysis_service = Mock(spec=IAnalysisService)
        mock_analysis_service.analyze_image = AsyncMock(return_value=Analysis(
            id=str(uuid.uuid4()),
            user_id="user-123",
            metrics=SkinMetrics(
                overall_score=80.0,
                texture_score=85.0,
                hydration_score=75.0,
                elasticity_score=80.0,
                pigmentation_score=85.0,
                pore_size_score=75.0,
                wrinkles_score=80.0,
                redness_score=85.0,
                dark_spots_score=80.0
            ),
            conditions=[
                Condition(
                    name="acne",
                    confidence=0.65,
                    severity="moderate",
                    description="Mild acne detected"
                )
            ],
            skin_type=SkinType.COMBINATION,
            status=AnalysisStatus.COMPLETED
        ))
        
        mock_image_processor.validate = AsyncMock(return_value=True)
        mock_analysis_repository.create = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        ))
        mock_analysis_repository.update = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
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
            user_id="user-123",
            image_data=b"image_bytes",
            metadata={"filename": "test.jpg"}
        )
        
        # Assertions
        assert result.status == AnalysisStatus.COMPLETED
        assert result.metrics is not None
        assert result.metrics.overall_score == 80.0
        assert len(result.conditions) == 1
        mock_analysis_repository.create.assert_called_once()
        mock_analysis_repository.update.assert_called_once()
        mock_image_processor.validate.assert_called_once()
        mock_analysis_service.analyze_image.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_invalid_image(
        self,
        mock_analysis_repository,
        mock_image_processor
    ):
        """Test image analysis with invalid image"""
        mock_image_processor.validate = AsyncMock(return_value=False)
        
        mock_analysis_service = Mock(spec=IAnalysisService)
        
        use_case = AnalyzeImageUseCase(
            analysis_repository=mock_analysis_repository,
            image_processor=mock_image_processor,
            analysis_service=mock_analysis_service
        )
        
        with pytest.raises(ValidationError):
            await use_case.execute(
                user_id="user-123",
                image_data=b"invalid_image"
            )
    
    @pytest.mark.asyncio
    async def test_execute_with_event_publisher(
        self,
        mock_analysis_repository,
        mock_image_processor,
        mock_event_publisher
    ):
        """Test image analysis with event publishing"""
        mock_analysis_service = Mock(spec=IAnalysisService)
        mock_analysis_service.analyze_image = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            metrics=SkinMetrics(
                overall_score=80.0,
                texture_score=85.0,
                hydration_score=75.0,
                elasticity_score=80.0,
                pigmentation_score=85.0,
                pore_size_score=75.0,
                wrinkles_score=80.0,
                redness_score=85.0,
                dark_spots_score=80.0
            ),
            conditions=[],
            status=AnalysisStatus.COMPLETED
        ))
        
        mock_image_processor.validate = AsyncMock(return_value=True)
        mock_analysis_repository.create = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        ))
        mock_analysis_repository.update = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.COMPLETED
        ))
        
        use_case = AnalyzeImageUseCase(
            analysis_repository=mock_analysis_repository,
            image_processor=mock_image_processor,
            analysis_service=mock_analysis_service,
            event_publisher=mock_event_publisher
        )
        
        result = await use_case.execute(
            user_id="user-123",
            image_data=b"image_bytes"
        )
        
        assert result.status == AnalysisStatus.COMPLETED
        # Verify events were published (check if publish was called)
        assert mock_event_publisher.publish.called
    
    @pytest.mark.asyncio
    async def test_execute_processing_error(
        self,
        mock_analysis_repository,
        mock_image_processor
    ):
        """Test image analysis with processing error"""
        mock_analysis_service = Mock(spec=IAnalysisService)
        mock_analysis_service.analyze_image = AsyncMock(side_effect=Exception("Processing failed"))
        
        mock_image_processor.validate = AsyncMock(return_value=True)
        mock_analysis_repository.create = AsyncMock(return_value=Analysis(
            id="test-1",
            user_id="user-123",
            status=AnalysisStatus.PROCESSING
        ))
        mock_analysis_repository.update = AsyncMock()
        
        use_case = AnalyzeImageUseCase(
            analysis_repository=mock_analysis_repository,
            image_processor=mock_image_processor,
            analysis_service=mock_analysis_service
        )
        
        with pytest.raises(ProcessingError):
            await use_case.execute(
                user_id="user-123",
                image_data=b"image_bytes"
            )
        
        # Verify failure was handled
        assert mock_analysis_repository.update.called


class TestGetRecommendationsUseCase:
    """Tests for GetRecommendationsUseCase"""
    
    @pytest.mark.asyncio
    async def test_execute_success(
        self,
        mock_analysis_repository,
        mock_cache_service
    ):
        """Test successful recommendations generation"""
        # Create mock analysis
        analysis = Analysis(
            id="test-1",
            user_id="user-123",
            metrics=SkinMetrics(
                overall_score=75.0,
                texture_score=80.0,
                hydration_score=70.0,
                elasticity_score=75.0,
                pigmentation_score=80.0,
                pore_size_score=70.0,
                wrinkles_score=75.0,
                redness_score=80.0,
                dark_spots_score=75.0
            ),
            conditions=[
                Condition(
                    name="acne",
                    confidence=0.65,
                    severity="moderate"
                )
            ],
            skin_type=SkinType.COMBINATION,
            status=AnalysisStatus.COMPLETED
        )
        
        mock_analysis_repository.get_by_id = AsyncMock(return_value=analysis)
        
        mock_recommendation_service = Mock(spec=IRecommendationService)
        mock_recommendation_service.generate_recommendations = AsyncMock(return_value={
            "routine": {
                "morning": [
                    Recommendation(
                        name="Cleanser",
                        category="cleanser",
                        description="Gentle cleanser",
                        priority=1
                    )
                ],
                "evening": [],
                "weekly": []
            },
            "specific_recommendations": [],
            "tips": []
        })
        
        use_case = GetRecommendationsUseCase(
            analysis_repository=mock_analysis_repository,
            recommendation_service=mock_recommendation_service
        )
        
        result = await use_case.execute(
            analysis_id="test-1",
            include_routine=True
        )
        
        assert result is not None
        assert "routine" in result
        assert "morning" in result["routine"]
        mock_analysis_repository.get_by_id.assert_called_once_with("test-1")
        mock_recommendation_service.generate_recommendations.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_analysis_not_found(
        self,
        mock_analysis_repository
    ):
        """Test recommendations with non-existent analysis"""
        mock_analysis_repository.get_by_id = AsyncMock(return_value=None)
        
        mock_recommendation_service = Mock(spec=IRecommendationService)
        
        use_case = GetRecommendationsUseCase(
            analysis_repository=mock_analysis_repository,
            recommendation_service=mock_recommendation_service
        )
        
        with pytest.raises(ValidationError):
            await use_case.execute(
                analysis_id="non-existent",
                include_routine=True
            )


class TestGetHistoryUseCase:
    """Tests for GetHistoryUseCase"""
    
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_analysis_repository):
        """Test successful history retrieval"""
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED,
                created_at=datetime.utcnow()
            )
            for i in range(5)
        ]
        
        mock_analysis_repository.get_by_user = AsyncMock(return_value=analyses)
        
        use_case = GetHistoryUseCase(
            analysis_repository=mock_analysis_repository
        )
        
        result = await use_case.execute(
            user_id="user-123",
            limit=10,
            offset=0
        )
        
        assert len(result) == 5
        assert all(a.user_id == "user-123" for a in result)
        mock_analysis_repository.get_by_user.assert_called_once_with("user-123", 10, 0)
    
    @pytest.mark.asyncio
    async def test_execute_empty_history(self, mock_analysis_repository):
        """Test history retrieval with no results"""
        mock_analysis_repository.get_by_user = AsyncMock(return_value=[])
        
        use_case = GetHistoryUseCase(
            analysis_repository=mock_analysis_repository
        )
        
        result = await use_case.execute(
            user_id="user-123",
            limit=10,
            offset=0
        )
        
        assert len(result) == 0
    
    @pytest.mark.asyncio
    async def test_execute_with_pagination(self, mock_analysis_repository):
        """Test history retrieval with pagination"""
        analyses = [
            Analysis(
                id=f"test-{i}",
                user_id="user-123",
                status=AnalysisStatus.COMPLETED
            )
            for i in range(3)
        ]
        
        mock_analysis_repository.get_by_user = AsyncMock(return_value=analyses)
        
        use_case = GetAnalysisHistoryUseCase(
            analysis_repository=mock_analysis_repository
        )
        
        result = await use_case.execute(
            user_id="user-123",
            limit=3,
            offset=0
        )
        
        assert len(result) == 3
        # get_by_user is called with limit + offset for pagination
        mock_analysis_repository.get_by_user.assert_called_once_with("user-123", 3)

