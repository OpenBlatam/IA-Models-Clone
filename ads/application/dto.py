"""
🎯 ADS Application Layer - Data Transfer Objects (DTOs)

DTOs define the data structures used for communication between layers,
ensuring clean separation and preventing domain entities from leaking
into the application or infrastructure layers.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from uuid import UUID
from decimal import Decimal
from datetime import datetime

from ..domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria


# =============================================================================
# Advertisement DTOs
# =============================================================================

@dataclass
class CreateAdRequest:
    """Request DTO for creating an advertisement.

    Matches test expectations (see tests/unit/test_application.py).
    """
    campaign_id: Optional[Union[UUID, str]]
    group_id: Optional[Union[UUID, str]]
    name: str
    content: str
    ad_type: AdType
    platform: Platform
    targeting_criteria: Dict[str, Any]
    budget: Dict[str, Any]

    # Back-compat optional fields used in other parts of the codebase
    description: Optional[str] = None
    headline: str = ""
    body_text: str = ""
    image_url: Optional[str] = None
    video_url: Optional[str] = None
    call_to_action: Optional[str] = None
    schedule: Optional[Dict[str, Any]] = None
    ad_group_id: Optional[Union[UUID, str]] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Name is required")
        if not self.content:
            raise ValueError("Content is required")


@dataclass
class CreateAdResponse:
    """Response DTO for advertisement creation."""
    success: bool
    ad_id: Optional[UUID] = None
    message: str = ""
    errors: Optional[List[str]] = None
    ad_data: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class ApproveAdRequest:
    """Request DTO for approving an advertisement."""
    ad_id: UUID
    approver_id: UUID
    approval_notes: Optional[str] = None


@dataclass
class ActivateAdRequest:
    """Request DTO for activating an advertisement."""
    ad_id: UUID
    activated_by: Optional[UUID] = None
    activation_notes: Optional[str] = None


@dataclass
class PauseAdRequest:
    """Request DTO for pausing an advertisement."""
    ad_id: UUID
    paused_by: Optional[UUID] = None
    pause_reason: Optional[str] = None


@dataclass
class ArchiveAdRequest:
    """Request DTO for archiving an advertisement."""
    ad_id: UUID
    archived_by: Optional[UUID] = None
    archive_reason: Optional[str] = None


@dataclass
class AdResponse:
    """Response DTO for advertisement operations."""
    success: bool
    ad_id: UUID
    message: str = ""
    errors: Optional[List[str]] = None
    ad_data: Optional[Dict[str, Any]] = None


# =============================================================================
# Campaign DTOs
# =============================================================================

@dataclass
class CreateCampaignRequest:
    """Request DTO for creating a campaign."""
    name: str
    description: Optional[str] = None
    objective: str = ""
    platform: str = "facebook"
    budget: Optional[Dict[str, Any]] = None
    schedule: Optional[Dict[str, Any]] = None
    targeting: Optional[Dict[str, Any]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Name is required")


@dataclass
class CreateCampaignResponse:
    """Response DTO for campaign creation."""
    success: bool
    campaign_id: Optional[UUID] = None
    message: str = ""
    errors: Optional[List[str]] = None
    campaign_data: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


@dataclass
class ActivateCampaignResponse:
    """Response DTO for activating a campaign."""
    success: bool
    campaign_id: UUID
    message: str = ""
    activated_at: Optional[datetime] = None
    activated_by: Optional[UUID] = None
    errors: Optional[List[str]] = None


@dataclass
class PauseCampaignResponse:
    """Response DTO for pausing a campaign."""
    success: bool
    campaign_id: UUID
    message: str = ""
    paused_at: Optional[datetime] = None
    paused_by: Optional[UUID] = None
    errors: Optional[List[str]] = None


@dataclass
class ActivateCampaignRequest:
    """Request DTO for activating a campaign."""
    campaign_id: UUID
    activated_by: Optional[UUID] = None
    activation_notes: Optional[str] = None


@dataclass
class PauseCampaignRequest:
    """Request DTO for pausing a campaign."""
    campaign_id: UUID
    paused_by: Optional[UUID] = None
    pause_reason: Optional[str] = None


@dataclass
class CampaignResponse:
    """Response DTO for campaign operations."""
    success: bool
    campaign_id: UUID
    message: str = ""
    errors: Optional[List[str]] = None
    campaign_data: Optional[Dict[str, Any]] = None


# =============================================================================
# Optimization DTOs
# =============================================================================

@dataclass
class OptimizationRequest:
    """Request DTO for ad optimization."""
    ad_id: UUID
    optimization_type: str = "performance"  # performance, budget, targeting
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResponse:
    """Response DTO for optimization operations."""
    success: bool
    ad_id: UUID
    optimization_results: Optional[Dict[str, Any]] = None
    optimized_at: Optional[datetime] = None
    message: str = ""
    errors: Optional[List[str]] = None


# =============================================================================
# Additional Response DTOs referenced by tests
# =============================================================================

@dataclass
class ApproveAdResponse:
    """Response DTO for approving an advertisement."""
    success: bool
    ad_id: UUID
    message: str = ""
    approved_at: Optional[datetime] = None
    approver_id: Optional[UUID] = None
    errors: Optional[List[str]] = None


@dataclass
class ActivateAdResponse:
    """Response DTO for activating an advertisement."""
    success: bool
    ad_id: UUID
    message: str = ""
    activated_at: Optional[datetime] = None
    activated_by: Optional[UUID] = None
    errors: Optional[List[str]] = None


@dataclass
class PauseAdResponse:
    """Response DTO for pausing an advertisement."""
    success: bool
    ad_id: UUID
    message: str = ""
    paused_at: Optional[datetime] = None
    paused_by: Optional[UUID] = None
    errors: Optional[List[str]] = None


@dataclass
class ArchiveAdResponse:
    """Response DTO for archiving an advertisement."""
    success: bool
    ad_id: UUID
    message: str = ""
    archived_at: Optional[datetime] = None
    archived_by: Optional[UUID] = None
    errors: Optional[List[str]] = None


# =============================================================================
# Performance Prediction DTOs
# =============================================================================

@dataclass
class PerformancePredictionRequest:
    """Request DTO for performance prediction.

    Matches test expectations with prediction_horizon and features.
    """
    ad_id: UUID
    prediction_horizon: int
    features: Dict[str, Any]


@dataclass
class PerformancePredictionResponse:
    """Response DTO for performance prediction."""
    success: bool
    ad_id: UUID
    predictions: Dict[str, Any]
    predicted_at: Optional[datetime] = None
    message: str = ""
    errors: Optional[List[str]] = None


# =============================================================================
# Analytics DTOs
# =============================================================================

@dataclass
class AnalyticsRequest:
    """Request DTO for analytics data."""
    entity_type: str  # ad, campaign, group
    start_date: str
    end_date: str
    entity_id: Optional[UUID] = None
    metrics: List[str] = None  # impressions, clicks, conversions, spend
    group_by: Optional[str] = None  # date, hour, platform, ad_type
    limit: int = 100


@dataclass
class AnalyticsResponse:
    """Response DTO for analytics data."""
    success: bool
    data: Optional[List[Dict[str, Any]]] = None
    summary: Optional[Dict[str, Any]] = None
    message: str = ""
    errors: Optional[List[str]] = None


# =============================================================================
# Search and Filter DTOs
# =============================================================================

@dataclass
class SearchRequest:
    """Request DTO for search operations."""
    query: str
    entity_type: str = "ad"  # ad, campaign, group
    filters: Optional[Dict[str, Any]] = None
    sort_by: Optional[str] = None
    sort_order: str = "desc"  # asc, desc
    skip: int = 0
    limit: int = 100


@dataclass
class SearchResponse:
    """Response DTO for search operations."""
    success: bool
    results: Optional[List[Dict[str, Any]]] = None
    total_count: int = 0
    page_info: Optional[Dict[str, Any]] = None
    message: str = ""
    errors: Optional[List[str]] = None


# =============================================================================
# Bulk Operations DTOs
# =============================================================================

@dataclass
class BulkOperationRequest:
    """Request DTO for bulk operations."""
    operation: str  # activate, pause, archive, update
    entity_ids: List[UUID]
    entity_type: str = "ad"  # ad, campaign, group
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class BulkOperationResponse:
    """Response DTO for bulk operations."""
    success: bool
    operation: str
    total_entities: int
    successful_operations: int
    failed_operations: int
    results: Optional[List[Dict[str, Any]]] = None
    message: str = ""
    errors: Optional[List[str]] = None


# =============================================================================
# Validation DTOs
# =============================================================================

@dataclass
class ValidationRequest:
    """Request DTO for validation operations."""
    entity_data: Dict[str, Any]
    entity_type: str = "ad"  # ad, campaign, group
    validation_rules: Optional[List[str]] = None


@dataclass
class ValidationResponse:
    """Response DTO for validation operations."""
    success: bool
    is_valid: bool
    validation_errors: Optional[List[str]] = None
    validation_warnings: Optional[List[str]] = None
    message: str = ""
    suggestions: Optional[List[str]] = None


# =============================================================================
# Error Response DTOs
# =============================================================================

@dataclass
class ErrorResponse:
    """Standard error response DTO."""
    success: bool = False
    error_code: Optional[str] = None
    error_type: Optional[str] = None
    message: str = "An error occurred"
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


@dataclass
class SuccessResponse:
    """Standard success response DTO."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None


# -----------------------------------------------------------------------------
# Backwards-compatibility aliases for tests
# -----------------------------------------------------------------------------
# Some tests reference `OptimizeAdRequest` / `OptimizeAdResponse` naming.
# Provide aliases to the canonical Optimization* names to keep imports working.
OptimizeAdRequest = OptimizationRequest
OptimizeAdResponse = OptimizationResponse
