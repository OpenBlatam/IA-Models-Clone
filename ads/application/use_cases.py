"""
🎯 ADS Application Layer - Use Cases

Use cases implement the application logic that orchestrates domain services
and coordinates business operations. They handle the flow of data between
the presentation layer and the domain layer.
"""

import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime, timezone

from ..domain.services import AdService, CampaignService, OptimizationService
from .dto import (
    CreateAdRequest, CreateAdResponse, ApproveAdRequest, ApproveAdResponse,
    ActivateAdRequest, ActivateAdResponse, PauseAdRequest, PauseAdResponse,
    ArchiveAdRequest, ArchiveAdResponse, CreateCampaignRequest, CreateCampaignResponse,
    ActivateCampaignRequest, ActivateCampaignResponse, PauseCampaignRequest, PauseCampaignResponse,
    OptimizationRequest, OptimizationResponse, PerformancePredictionRequest, PerformancePredictionResponse,
    ErrorResponse, SuccessResponse
)

logger = logging.getLogger(__name__)


class CreateAdUseCase:
    """Use case for creating a new advertisement."""
    
    def __init__(self, ad_service: AdService, campaign_service: Optional[CampaignService] = None):
        self.ad_service = ad_service
        self.campaign_service = campaign_service
    
    async def execute(self, request: CreateAdRequest) -> CreateAdResponse:
        """Execute the create ad use case."""
        try:
            logger.info(f"Creating advertisement: {request.name}")
            
            # Create advertisement using domain service (mocked in tests)
            ad = await self.ad_service.create_ad(
                request
            )
            response = CreateAdResponse(
                success=True,
                ad_id=ad.id,
                message="Ad created successfully",
            )
            
            logger.info(f"Advertisement created successfully: {ad.id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error creating advertisement: {e}")
            return CreateAdResponse(
                success=False,
                message="Validation error",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Error creating advertisement: {e}")
            return CreateAdResponse(
                success=False,
                message="Internal server error",
                errors=["An unexpected error occurred"]
            )
    
    # Kept for backward compatibility, unused by current tests
    def _convert_request_to_domain_data(self, request: CreateAdRequest) -> Dict[str, Any]:
        return {
            'name': request.name,
            'description': request.description,
            'ad_type': request.ad_type,
            'platform': request.platform,
            'headline': request.headline,
            'body_text': request.body_text,
            'image_url': request.image_url,
            'video_url': request.video_url,
            'call_to_action': request.call_to_action,
            'campaign_id': request.campaign_id,
            'ad_group_id': request.ad_group_id,
            'targeting': getattr(request, 'targeting_criteria', None) or getattr(request, 'targeting', None),
            'budget': request.budget,
            'schedule': request.schedule,
        }


class ApproveAdUseCase:
    """Use case for approving an advertisement."""
    
    def __init__(self, ad_service: AdService):
        self.ad_service = ad_service
    
    async def execute(self, request: ApproveAdRequest) -> ApproveAdResponse:
        """Execute the approve ad use case."""
        try:
            logger.info(f"Approving advertisement: {request.ad_id}")
            
            # Approve advertisement using domain service
            approved = await self.ad_service.approve_ad(request.ad_id, request.approver_id, request.approval_notes)
            
            # Convert domain entity to response DTO
            response = ApproveAdResponse(
                success=True,
                ad_id=request.ad_id,
                message="Ad approved successfully",
                approver_id=request.approver_id,
            )
            
            logger.info(f"Advertisement approved successfully: {request.ad_id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error approving advertisement: {e}")
            return ApproveAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Validation error",
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Error approving advertisement: {e}")
            return ApproveAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Internal server error",
                errors=["An unexpected error occurred"],
            )


class ActivateAdUseCase:
    """Use case for activating an advertisement."""
    
    def __init__(self, ad_service: AdService):
        self.ad_service = ad_service
    
    async def execute(self, request: ActivateAdRequest) -> ActivateAdResponse:
        """Execute the activate ad use case."""
        try:
            logger.info(f"Activating advertisement: {request.ad_id}")
            
            # Activate advertisement using domain service
            activated = await self.ad_service.activate_ad(request.ad_id, request.activated_by)
            
            # Convert domain entity to response DTO
            response = ActivateAdResponse(
                success=True,
                ad_id=request.ad_id,
                message="Ad activated successfully",
                activated_by=request.activated_by,
            )
            
            logger.info(f"Advertisement activated successfully: {request.ad_id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error activating advertisement: {e}")
            return ActivateAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Validation error",
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Error activating advertisement: {e}")
            return ActivateAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Internal server error",
                errors=["An unexpected error occurred"],
            )


class PauseAdUseCase:
    """Use case for pausing an advertisement."""
    
    def __init__(self, ad_service: AdService):
        self.ad_service = ad_service
    
    async def execute(self, request: PauseAdRequest) -> PauseAdResponse:
        """Execute the pause ad use case."""
        try:
            logger.info(f"Pausing advertisement: {request.ad_id}")
            
            # Pause advertisement using domain service
            paused = await self.ad_service.pause_ad(request.ad_id, request.paused_by, request.pause_reason)
            
            # Convert domain entity to response DTO
            response = PauseAdResponse(
                success=True,
                ad_id=request.ad_id,
                message="Ad paused successfully",
                paused_by=request.paused_by,
            )
            
            logger.info(f"Advertisement paused successfully: {request.ad_id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error pausing advertisement: {e}")
            return PauseAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Validation error",
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Error pausing advertisement: {e}")
            return PauseAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Internal server error",
                errors=["An unexpected error occurred"],
            )


class ArchiveAdUseCase:
    """Use case for archiving an advertisement."""
    
    def __init__(self, ad_service: AdService):
        self.ad_service = ad_service
    
    async def execute(self, request: ArchiveAdRequest) -> ArchiveAdResponse:
        """Execute the archive ad use case."""
        try:
            logger.info(f"Archiving advertisement: {request.ad_id}")
            
            # Archive advertisement using domain service
            archived = await self.ad_service.archive_ad(request.ad_id, request.archived_by, request.archive_reason)
            
            # Convert domain entity to response DTO
            response = ArchiveAdResponse(
                success=True,
                ad_id=request.ad_id,
                message="Ad archived successfully",
                archived_by=request.archived_by,
            )
            
            logger.info(f"Advertisement archived successfully: {request.ad_id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error archiving advertisement: {e}")
            return ArchiveAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Validation error",
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Error archiving advertisement: {e}")
            return ArchiveAdResponse(
                success=False,
                ad_id=request.ad_id,
                message="Internal server error",
                errors=["An unexpected error occurred"],
            )


class CreateCampaignUseCase:
    """Use case for creating a new campaign."""
    
    def __init__(self, campaign_service: CampaignService):
        self.campaign_service = campaign_service
    
    async def execute(self, request: CreateCampaignRequest) -> CreateCampaignResponse:
        """Execute the create campaign use case."""
        try:
            logger.info(f"Creating campaign: {request.name}")
            
            # Convert DTO to domain data
            campaign_data = self._convert_request_to_domain_data(request)
            
            # Create campaign using domain service
            campaign = await self.campaign_service.create_campaign(campaign_data)
            
            # Convert domain entity to response DTO
            response = CreateCampaignResponse(
                success=True,
                campaign_id=campaign.id,
                message="Campaign created successfully",
            )
            
            logger.info(f"Campaign created successfully: {campaign.id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error creating campaign: {e}")
            return CreateCampaignResponse(
                success=False,
                message="Validation error",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Error creating campaign: {e}")
            return CreateCampaignResponse(
                success=False,
                message="Internal server error",
                errors=["An unexpected error occurred"]
            )
    
    def _convert_request_to_domain_data(self, request: CreateCampaignRequest) -> Dict[str, Any]:
        """Convert request DTO to domain data format."""
        campaign_data = {
            'name': request.name,
            'description': request.description,
            'objective': request.objective,
            'platform': request.platform
        }
        
        # Convert budget data if provided
        if request.budget:
            campaign_data['budget'] = request.budget
        
        # Convert schedule data if provided
        if request.schedule:
            campaign_data['schedule'] = request.schedule
        
        # Convert targeting data if provided
        if request.targeting:
            campaign_data['targeting'] = request.targeting
        
        return campaign_data


class ActivateCampaignUseCase:
    """Use case for activating a campaign."""
    
    def __init__(self, campaign_service: CampaignService):
        self.campaign_service = campaign_service
    
    async def execute(self, request: ActivateCampaignRequest) -> ActivateCampaignResponse:
        """Execute the activate campaign use case."""
        try:
            logger.info(f"Activating campaign: {request.campaign_id}")
            
            # Activate campaign using domain service
            activated = await self.campaign_service.activate_campaign(request.campaign_id, request.activated_by)
            
            # Convert domain entity to response DTO
            response = ActivateCampaignResponse(
                success=True,
                campaign_id=request.campaign_id,
                message="Campaign activated successfully",
                activated_by=request.activated_by,
            )
            
            logger.info(f"Campaign activated successfully: {campaign.id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error activating campaign: {e}")
            return ActivateCampaignResponse(
                success=False,
                campaign_id=request.campaign_id,
                message="Validation error",
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Error activating campaign: {e}")
            return ActivateCampaignResponse(
                success=False,
                campaign_id=request.campaign_id,
                message="Internal server error",
                errors=["An unexpected error occurred"],
            )


class PauseCampaignUseCase:
    """Use case for pausing a campaign."""
    
    def __init__(self, campaign_service: CampaignService):
        self.campaign_service = campaign_service
    
    async def execute(self, request: PauseCampaignRequest) -> PauseCampaignResponse:
        """Execute the pause campaign use case."""
        try:
            logger.info(f"Pausing campaign: {request.campaign_id}")
            
            # Pause campaign using domain service
            paused = await self.campaign_service.pause_campaign(request.campaign_id, request.paused_by, request.pause_reason)
            
            # Convert domain entity to response DTO
            response = PauseCampaignResponse(
                success=True,
                campaign_id=request.campaign_id,
                message="Campaign paused successfully",
                paused_by=request.paused_by,
            )
            
            logger.info(f"Campaign paused successfully: {campaign.id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error pausing campaign: {e}")
            return PauseCampaignResponse(
                success=False,
                campaign_id=request.campaign_id,
                message="Validation error",
                errors=[str(e)],
            )
        except Exception as e:
            logger.error(f"Error pausing campaign: {e}")
            return PauseCampaignResponse(
                success=False,
                campaign_id=request.campaign_id,
                message="Internal server error",
                errors=["An unexpected error occurred"],
            )


class OptimizeAdUseCase:
    """Use case for optimizing advertisement performance."""
    
    def __init__(self, optimization_service: OptimizationService):
        self.optimization_service = optimization_service
    
    async def execute(self, request: OptimizationRequest) -> OptimizationResponse:
        """Execute the optimize ad use case."""
        try:
            logger.info(f"Optimizing advertisement: {request.ad_id}")
            
            # Optimize advertisement using domain service
            optimization_result = await self.optimization_service.optimize_ad(request.ad_id, request.optimization_type, request.parameters)
            
            # Convert domain result to response DTO
            response = OptimizationResponse(
                success=True,
                ad_id=request.ad_id,
                message="Ad optimized successfully",
                optimization_results=optimization_result,
            )
            
            logger.info(f"Advertisement optimization completed successfully: {request.ad_id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error optimizing advertisement: {e}")
            return OptimizationResponse(
                success=False,
                ad_id=request.ad_id,
                message="Validation error",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Error optimizing advertisement: {e}")
            return OptimizationResponse(
                success=False,
                ad_id=request.ad_id,
                message="Internal server error",
                errors=["An unexpected error occurred"]
            )


class PredictPerformanceUseCase:
    """Use case for predicting advertisement performance."""
    
    def __init__(self, optimization_service: OptimizationService):
        self.optimization_service = optimization_service
    
    async def execute(self, request: PerformancePredictionRequest) -> PerformancePredictionResponse:
        """Execute the predict performance use case."""
        try:
            logger.info(f"Predicting performance for advertisement: {request.ad_id}")
            
            # Predict performance using domain service
            prediction_result = await self.optimization_service.predict_performance(
                request.ad_id,
                request.prediction_horizon,
                request.features,
            )
            
            # Convert domain result to response DTO
            response = PerformancePredictionResponse(
                success=True,
                ad_id=request.ad_id,
                predictions=prediction_result,
                message="Performance prediction completed successfully",
            )
            
            logger.info(f"Performance prediction completed successfully: {request.ad_id}")
            return response
            
        except ValueError as e:
            logger.warning(f"Validation error predicting performance: {e}")
            return PerformancePredictionResponse(
                success=False,
                ad_id=request.ad_id,
                message="Validation error",
                errors=[str(e)]
            )
        except Exception as e:
            logger.error(f"Error predicting performance: {e}")
            return PerformancePredictionResponse(
                success=False,
                ad_id=request.ad_id,
                message="Internal server error",
                errors=["An unexpected error occurred"]
            )
