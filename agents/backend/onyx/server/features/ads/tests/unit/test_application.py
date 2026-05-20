"""
Unit tests for the ads application layer.

This module consolidates tests for:
- Use cases (CreateAdUseCase, ApproveAdUseCase, etc.)
- DTOs (request/response models)
- Application services
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock

from agents.backend.onyx.server.features.ads.application.dto import (
    CreateAdRequest, CreateAdResponse, ApproveAdRequest, ApproveAdResponse,
    ActivateAdRequest, ActivateAdResponse, PauseAdRequest, PauseAdResponse,
    ArchiveAdRequest, ArchiveAdResponse, CreateCampaignRequest, CreateCampaignResponse,
    ActivateCampaignRequest, ActivateCampaignResponse, PauseCampaignRequest, PauseCampaignResponse,
    OptimizeAdRequest, OptimizeAdResponse, PerformancePredictionRequest, PerformancePredictionResponse,
    ErrorResponse
)
from agents.backend.onyx.server.features.ads.application.use_cases import (
    CreateAdUseCase, ApproveAdUseCase, ActivateAdUseCase, PauseAdUseCase, ArchiveAdUseCase,
    CreateCampaignUseCase, ActivateCampaignUseCase, PauseCampaignUseCase,
    OptimizeAdUseCase, PredictPerformanceUseCase
)
from agents.backend.onyx.server.features.ads.domain.entities import (
    Ad, AdCampaign, AdGroup, AdPerformance
)
from agents.backend.onyx.server.features.ads.domain.value_objects import (
    AdStatus, AdType, Platform, Budget, TargetingCriteria
)


class TestCreateAdRequest:
    """Test CreateAdRequest DTO."""
    
    def test_create_ad_request_creation(self):
        """Test CreateAdRequest creation with valid values."""
        request = CreateAdRequest(
            campaign_id="campaign_123",
            group_id="group_456",
            name="Test Ad",
            content="This is a test ad",
            ad_type=AdType.TEXT,
            platform=Platform.FACEBOOK,
            targeting_criteria={
                "demographics": {"age_range": "25-34"},
                "interests": ["technology"],
                "location": {},
                "behavior": []
            },
            budget={
                "daily_limit": "50.00",
                "total_limit": "500.00",
                "currency": "USD"
            }
        )
        assert request.campaign_id == "campaign_123"
        assert request.group_id == "group_456"
        assert request.name == "Test Ad"
        assert request.content == "This is a test ad"
        assert request.ad_type == AdType.TEXT
        assert request.platform == Platform.FACEBOOK
        assert request.targeting_criteria["demographics"]["age_range"] == "25-34"
        assert request.budget["daily_limit"] == "50.00"
    
    def test_create_ad_request_validation_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValueError, match="Name is required"):
            CreateAdRequest(
                campaign_id="campaign_123",
                group_id="group_456",
                name="",
                content="This is a test ad",
                ad_type=AdType.TEXT,
                platform=Platform.FACEBOOK,
                targeting_criteria={},
                budget={}
            )
    
    def test_create_ad_request_validation_content_required(self):
        """Test that content is required."""
        with pytest.raises(ValueError, match="Content is required"):
            CreateAdRequest(
                campaign_id="campaign_123",
                group_id="group_456",
                name="Test Ad",
                content="",
                ad_type=AdType.TEXT,
                platform=Platform.FACEBOOK,
                targeting_criteria={},
                budget={}
            )


class TestCreateAdResponse:
    """Test CreateAdResponse DTO."""
    
    def test_create_ad_response_creation(self):
        """Test CreateAdResponse creation with valid values."""
        response = CreateAdResponse(
            success=True,
            ad_id="ad_123",
            message="Ad created successfully",
            created_at=datetime.now()
        )
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad created successfully"
        assert response.created_at is not None


class TestApproveAdRequest:
    """Test ApproveAdRequest DTO."""
    
    def test_approve_ad_request_creation(self):
        """Test ApproveAdRequest creation with valid values."""
        request = ApproveAdRequest(
            ad_id="ad_123",
            approver_id="user_456",
            approval_notes="Looks good to me"
        )
        assert request.ad_id == "ad_123"
        assert request.approver_id == "user_456"
        assert request.approval_notes == "Looks good to me"


class TestApproveAdResponse:
    """Test ApproveAdResponse DTO."""
    
    def test_approve_ad_response_creation(self):
        """Test ApproveAdResponse creation with valid values."""
        response = ApproveAdResponse(
            success=True,
            ad_id="ad_123",
            message="Ad approved successfully",
            approved_at=datetime.now(),
            approver_id="user_456"
        )
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad approved successfully"
        assert response.approved_at is not None
        assert response.approver_id == "user_456"


class TestActivateAdRequest:
    """Test ActivateAdRequest DTO."""
    
    def test_activate_ad_request_creation(self):
        """Test ActivateAdRequest creation with valid values."""
        request = ActivateAdRequest(
            ad_id="ad_123",
            activated_by="user_456"
        )
        assert request.ad_id == "ad_123"
        assert request.activated_by == "user_456"


class TestActivateAdResponse:
    """Test ActivateAdResponse DTO."""
    
    def test_activate_ad_response_creation(self):
        """Test ActivateAdResponse creation with valid values."""
        response = ActivateAdResponse(
            success=True,
            ad_id="ad_123",
            message="Ad activated successfully",
            activated_at=datetime.now(),
            activated_by="user_456"
        )
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad activated successfully"
        assert response.activated_at is not None
        assert response.activated_by == "user_456"


class TestPauseAdRequest:
    """Test PauseAdRequest DTO."""
    
    def test_pause_ad_request_creation(self):
        """Test PauseAdRequest creation with valid values."""
        request = PauseAdRequest(
            ad_id="ad_123",
            paused_by="user_456",
            pause_reason="Budget limit reached"
        )
        assert request.ad_id == "ad_123"
        assert request.paused_by == "user_456"
        assert request.pause_reason == "Budget limit reached"


class TestPauseAdResponse:
    """Test PauseAdResponse DTO."""
    
    def test_pause_ad_response_creation(self):
        """Test PauseAdResponse creation with valid values."""
        response = PauseAdResponse(
            success=True,
            ad_id="ad_123",
            message="Ad paused successfully",
            paused_at=datetime.now(),
            paused_by="user_456"
        )
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad paused successfully"
        assert response.paused_at is not None
        assert response.paused_by == "user_456"


class TestArchiveAdRequest:
    """Test ArchiveAdRequest DTO."""
    
    def test_archive_ad_request_creation(self):
        """Test ArchiveAdRequest creation with valid values."""
        request = ArchiveAdRequest(
            ad_id="ad_123",
            archived_by="user_456",
            archive_reason="Campaign completed"
        )
        assert request.ad_id == "ad_123"
        assert request.archived_by == "user_456"
        assert request.archive_reason == "Campaign completed"


class TestArchiveAdResponse:
    """Test ArchiveAdResponse DTO."""
    
    def test_archive_ad_response_creation(self):
        """Test ArchiveAdResponse creation with valid values."""
        response = ArchiveAdResponse(
            success=True,
            ad_id="ad_123",
            message="Ad archived successfully",
            archived_at=datetime.now(),
            archived_by="user_456"
        )
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad archived successfully"
        assert response.archived_at is not None
        assert response.archived_by == "user_456"


class TestCreateCampaignRequest:
    """Test CreateCampaignRequest DTO."""
    
    def test_create_campaign_request_creation(self):
        """Test CreateCampaignRequest creation with valid values."""
        request = CreateCampaignRequest(
            name="Test Campaign",
            description="A test advertising campaign",
            objective="awareness",
            budget={
                "daily_limit": "200.00",
                "total_limit": "2000.00",
                "currency": "USD"
            },
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=60)
        )
        assert request.name == "Test Campaign"
        assert request.description == "A test advertising campaign"
        assert request.objective == "awareness"
        assert request.budget["daily_limit"] == "200.00"
        assert request.start_date is not None
        assert request.end_date is not None
    
    def test_create_campaign_request_validation_name_required(self):
        """Test that name is required."""
        with pytest.raises(ValueError, match="Name is required"):
            CreateCampaignRequest(
                name="",
                description="A test advertising campaign",
                objective="awareness",
                budget={},
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=60)
            )


class TestCreateCampaignResponse:
    """Test CreateCampaignResponse DTO."""
    
    def test_create_campaign_response_creation(self):
        """Test CreateCampaignResponse creation with valid values."""
        response = CreateCampaignResponse(
            success=True,
            campaign_id="campaign_123",
            message="Campaign created successfully",
            created_at=datetime.now()
        )
        assert response.success is True
        assert response.campaign_id == "campaign_123"
        assert response.message == "Campaign created successfully"
        assert response.created_at is not None


class TestActivateCampaignRequest:
    """Test ActivateCampaignRequest DTO."""
    
    def test_activate_campaign_request_creation(self):
        """Test ActivateCampaignRequest creation with valid values."""
        request = ActivateCampaignRequest(
            campaign_id="campaign_123",
            activated_by="user_456"
        )
        assert request.campaign_id == "campaign_123"
        assert request.activated_by == "user_456"


class TestActivateCampaignResponse:
    """Test ActivateCampaignResponse DTO."""
    
    def test_activate_campaign_response_creation(self):
        """Test ActivateCampaignResponse creation with valid values."""
        response = ActivateCampaignResponse(
            success=True,
            campaign_id="campaign_123",
            message="Campaign activated successfully",
            activated_at=datetime.now(),
            activated_by="user_456"
        )
        assert response.success is True
        assert response.campaign_id == "campaign_123"
        assert response.message == "Campaign activated successfully"
        assert response.activated_at is not None
        assert response.activated_by == "user_456"


class TestPauseCampaignRequest:
    """Test PauseCampaignRequest DTO."""
    
    def test_pause_campaign_request_creation(self):
        """Test PauseCampaignRequest creation with valid values."""
        request = PauseCampaignRequest(
            campaign_id="campaign_123",
            paused_by="user_456",
            pause_reason="Budget limit reached"
        )
        assert request.campaign_id == "campaign_123"
        assert request.paused_by == "user_456"
        assert request.pause_reason == "Budget limit reached"


class TestPauseCampaignResponse:
    """Test PauseCampaignResponse DTO."""
    
    def test_pause_campaign_response_creation(self):
        """Test PauseCampaignResponse creation with valid values."""
        response = PauseCampaignResponse(
            success=True,
            campaign_id="campaign_123",
            message="Campaign paused successfully",
            paused_at=datetime.now(),
            paused_by="user_456"
        )
        assert response.success is True
        assert response.campaign_id == "campaign_123"
        assert response.message == "Campaign paused successfully"
        assert response.paused_at is not None
        assert response.paused_by == "user_456"


class TestOptimizeAdRequest:
    """Test OptimizeAdRequest DTO."""
    
    def test_optimize_ad_request_creation(self):
        """Test OptimizeAdRequest creation with valid values."""
        request = OptimizeAdRequest(
            ad_id="ad_123",
            optimization_type="performance",
            parameters={
                "target_ctr": 0.08,
                "max_cpc": "3.00",
                "budget_adjustment": 0.1
            }
        )
        assert request.ad_id == "ad_123"
        assert request.optimization_type == "performance"
        assert request.parameters["target_ctr"] == 0.08
        assert request.parameters["max_cpc"] == "3.00"


class TestOptimizeAdResponse:
    """Test OptimizeAdResponse DTO."""
    
    def test_optimize_ad_response_creation(self):
        """Test OptimizeAdResponse creation with valid values."""
        response = OptimizeAdResponse(
            success=True,
            ad_id="ad_123",
            message="Ad optimized successfully",
            optimization_results={
                "ctr_improvement": 0.02,
                "cpc_reduction": 0.15,
                "budget_efficiency": 1.1
            },
            optimized_at=datetime.now()
        )
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad optimized successfully"
        assert response.optimization_results["ctr_improvement"] == 0.02
        assert response.optimized_at is not None


class TestPerformancePredictionRequest:
    """Test PerformancePredictionRequest DTO."""
    
    def test_performance_prediction_request_creation(self):
        """Test PerformancePredictionRequest creation with valid values."""
        request = PerformancePredictionRequest(
            ad_id="ad_123",
            prediction_horizon=30,
            features={
                "historical_ctr": 0.05,
                "audience_size": 100000,
                "seasonality_factor": 1.2
            }
        )
        assert request.ad_id == "ad_123"
        assert request.prediction_horizon == 30
        assert request.features["historical_ctr"] == 0.05


class TestPerformancePredictionResponse:
    """Test PerformancePredictionResponse DTO."""
    
    def test_performance_prediction_response_creation(self):
        """Test PerformancePredictionResponse creation with valid values."""
        response = PerformancePredictionResponse(
            success=True,
            ad_id="ad_123",
            predictions={
                "predicted_ctr": 0.06,
                "predicted_cpc": "2.50",
                "predicted_impressions": 50000,
                "confidence_interval": 0.95
            },
            predicted_at=datetime.now()
        )
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.predictions["predicted_ctr"] == 0.06
        assert response.predictions["confidence_interval"] == 0.95
        assert response.predicted_at is not None


class TestErrorResponse:
    """Test ErrorResponse DTO."""
    
    def test_error_response_creation(self):
        """Test ErrorResponse creation with valid values."""
        response = ErrorResponse(
            success=False,
            error_code="VALIDATION_ERROR",
            message="Invalid input data",
            details={"field": "name", "issue": "Required field missing"}
        )
        assert response.success is False
        assert response.error_code == "VALIDATION_ERROR"
        assert response.message == "Invalid input data"
        assert response.details["field"] == "name"


class TestCreateAdUseCase:
    """Test CreateAdUseCase."""
    
    @pytest.fixture
    def mock_ad_service(self):
        """Mock AdService for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def mock_campaign_service(self):
        """Mock CampaignService for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def create_ad_use_case(self, mock_ad_service, mock_campaign_service):
        """Create CreateAdUseCase instance with mocked dependencies."""
        return CreateAdUseCase(
            ad_service=mock_ad_service,
            campaign_service=mock_campaign_service
        )
    
    @pytest.mark.asyncio
    async def test_create_ad_success(self, create_ad_use_case, mock_ad_service):
        """Test successful ad creation."""
        # Arrange
        request = CreateAdRequest(
            campaign_id="campaign_123",
            group_id="group_456",
            name="Test Ad",
            content="This is a test ad",
            ad_type=AdType.TEXT,
            platform=Platform.FACEBOOK,
            targeting_criteria={},
            budget={}
        )
        
        mock_ad_service.create_ad.return_value = Ad(
            id="ad_123",
            campaign_id="campaign_123",
            group_id="group_456",
            name="Test Ad",
            content="This is a test ad",
            ad_type=AdType.TEXT,
            status=AdStatus.DRAFT,
            platform=Platform.FACEBOOK,
            targeting_criteria=TargetingCriteria(
                demographics={},
                interests=[],
                location={},
                behavior=[]
            ),
            budget=Budget(
                daily_limit=Decimal("50.00"),
                total_limit=Decimal("500.00"),
                currency="USD"
            ),
            created_at=datetime.now()
        )
        
        # Act
        response = await create_ad_use_case.execute(request)
        
        # Assert
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad created successfully"
        mock_ad_service.create_ad.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_ad_validation_error(self, create_ad_use_case):
        """Test ad creation with validation error."""
        # Arrange
        request = CreateAdRequest(
            campaign_id="campaign_123",
            group_id="group_456",
            name="",  # Invalid: empty name
            content="This is a test ad",
            ad_type=AdType.TEXT,
            platform=Platform.FACEBOOK,
            targeting_criteria={},
            budget={}
        )
        
        # Act & Assert
        with pytest.raises(ValueError, match="Name is required"):
            await create_ad_use_case.execute(request)


class TestApproveAdUseCase:
    """Test ApproveAdUseCase."""
    
    @pytest.fixture
    def mock_ad_service(self):
        """Mock AdService for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def approve_ad_use_case(self, mock_ad_service):
        """Create ApproveAdUseCase instance with mocked dependencies."""
        return ApproveAdUseCase(ad_service=mock_ad_service)
    
    @pytest.mark.asyncio
    async def test_approve_ad_success(self, approve_ad_use_case, mock_ad_service):
        """Test successful ad approval."""
        # Arrange
        request = ApproveAdRequest(
            ad_id="ad_123",
            approver_id="user_456",
            approval_notes="Looks good to me"
        )
        
        mock_ad_service.approve_ad.return_value = True
        
        # Act
        response = await approve_ad_use_case.execute(request)
        
        # Assert
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad approved successfully"
        assert response.approver_id == "user_456"
        mock_ad_service.approve_ad.assert_called_once_with(
            "ad_123", "user_456", "Looks good to me"
        )


class TestActivateAdUseCase:
    """Test ActivateAdUseCase."""
    
    @pytest.fixture
    def mock_ad_service(self):
        """Mock AdService for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def activate_ad_use_case(self, mock_ad_service):
        """Create ActivateAdUseCase instance with mocked dependencies."""
        return ActivateAdUseCase(ad_service=mock_ad_service)
    
    @pytest.mark.asyncio
    async def test_activate_ad_success(self, activate_ad_use_case, mock_ad_service):
        """Test successful ad activation."""
        # Arrange
        request = ActivateAdRequest(
            ad_id="ad_123",
            activated_by="user_456"
        )
        
        mock_ad_service.activate_ad.return_value = True
        
        # Act
        response = await activate_ad_use_case.execute(request)
        
        # Assert
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad activated successfully"
        assert response.activated_by == "user_456"
        mock_ad_service.activate_ad.assert_called_once_with("ad_123", "user_456")


class TestCreateCampaignUseCase:
    """Test CreateCampaignUseCase."""
    
    @pytest.fixture
    def mock_campaign_service(self):
        """Mock CampaignService for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def create_campaign_use_case(self, mock_campaign_service):
        """Create CreateCampaignUseCase instance with mocked dependencies."""
        return CreateCampaignUseCase(campaign_service=mock_campaign_service)
    
    @pytest.mark.asyncio
    async def test_create_campaign_success(self, create_campaign_use_case, mock_campaign_service):
        """Test successful campaign creation."""
        # Arrange
        request = CreateCampaignRequest(
            name="Test Campaign",
            description="A test advertising campaign",
            objective="awareness",
            budget={},
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=60)
        )
        
        mock_campaign_service.create_campaign.return_value = AdCampaign(
            id="campaign_123",
            name="Test Campaign",
            description="A test advertising campaign",
            objective="awareness",
            status=AdStatus.DRAFT,
            budget=Budget(
                daily_limit=Decimal("200.00"),
                total_limit=Decimal("2000.00"),
                currency="USD"
            ),
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=60),
            created_at=datetime.now()
        )
        
        # Act
        response = await create_campaign_use_case.execute(request)
        
        # Assert
        assert response.success is True
        assert response.campaign_id == "campaign_123"
        assert response.message == "Campaign created successfully"
        mock_campaign_service.create_campaign.assert_called_once()


class TestOptimizeAdUseCase:
    """Test OptimizeAdUseCase."""
    
    @pytest.fixture
    def mock_optimization_service(self):
        """Mock OptimizationService for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def optimize_ad_use_case(self, mock_optimization_service):
        """Create OptimizeAdUseCase instance with mocked dependencies."""
        return OptimizeAdUseCase(optimization_service=mock_optimization_service)
    
    @pytest.mark.asyncio
    async def test_optimize_ad_success(self, optimize_ad_use_case, mock_optimization_service):
        """Test successful ad optimization."""
        # Arrange
        request = OptimizeAdRequest(
            ad_id="ad_123",
            optimization_type="performance",
            parameters={
                "target_ctr": 0.08,
                "max_cpc": "3.00"
            }
        )
        
        mock_optimization_service.optimize_ad.return_value = {
            "ctr_improvement": 0.02,
            "cpc_reduction": 0.15,
            "budget_efficiency": 1.1
        }
        
        # Act
        response = await optimize_ad_use_case.execute(request)
        
        # Assert
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.message == "Ad optimized successfully"
        assert response.optimization_results["ctr_improvement"] == 0.02
        mock_optimization_service.optimize_ad.assert_called_once_with(
            "ad_123", "performance", {"target_ctr": 0.08, "max_cpc": "3.00"}
        )


class TestPredictPerformanceUseCase:
    """Test PredictPerformanceUseCase."""
    
    @pytest.fixture
    def mock_optimization_service(self):
        """Mock OptimizationService for testing."""
        return AsyncMock()
    
    @pytest.fixture
    def predict_performance_use_case(self, mock_optimization_service):
        """Create PredictPerformanceUseCase instance with mocked dependencies."""
        return PredictPerformanceUseCase(optimization_service=mock_optimization_service)
    
    @pytest.mark.asyncio
    async def test_predict_performance_success(self, predict_performance_use_case, mock_optimization_service):
        """Test successful performance prediction."""
        # Arrange
        request = PerformancePredictionRequest(
            ad_id="ad_123",
            prediction_horizon=30,
            features={
                "historical_ctr": 0.05,
                "audience_size": 100000
            }
        )
        
        mock_optimization_service.predict_performance.return_value = {
            "predicted_ctr": 0.06,
            "predicted_cpc": "2.50",
            "predicted_impressions": 50000,
            "confidence_interval": 0.95
        }
        
        # Act
        response = await predict_performance_use_case.execute(request)
        
        # Assert
        assert response.success is True
        assert response.ad_id == "ad_123"
        assert response.predictions["predicted_ctr"] == 0.06
        assert response.predictions["confidence_interval"] == 0.95
        mock_optimization_service.predict_performance.assert_called_once_with(
            "ad_123", 30, {"historical_ctr": 0.05, "audience_size": 100000}
        )


if __name__ == "__main__":
    pytest.main([__file__])
