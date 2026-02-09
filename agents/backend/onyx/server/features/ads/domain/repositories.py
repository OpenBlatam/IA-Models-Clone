"""
🎯 ADS Domain - Repository Interfaces

Repository interfaces define the contracts for data access operations.
These are part of the domain layer and should be implemented in the infrastructure layer.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID

from .entities import Ad, AdCampaign, AdGroup, AdPerformance
from .value_objects import AdStatus, AdType, Platform


class AdRepository(ABC):
    """Repository interface for advertisement entities."""
    
    @abstractmethod
    async def create(self, ad: Ad) -> Ad:
        """Create a new advertisement."""
        pass
    
    @abstractmethod
    async def get_by_id(self, ad_id: UUID) -> Optional[Ad]:
        """Get advertisement by ID."""
        pass
    
    @abstractmethod
    async def get_by_campaign(self, campaign_id: UUID) -> List[Ad]:
        """Get all advertisements in a campaign."""
        pass
    
    @abstractmethod
    async def get_by_group(self, group_id: UUID) -> List[Ad]:
        """Get all advertisements in a group."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: AdStatus) -> List[Ad]:
        """Get advertisements by status."""
        pass
    
    @abstractmethod
    async def get_by_platform(self, platform: Platform) -> List[Ad]:
        """Get advertisements by platform."""
        pass
    
    @abstractmethod
    async def get_by_type(self, ad_type: AdType) -> List[Ad]:
        """Get advertisements by type."""
        pass
    
    @abstractmethod
    async def update(self, ad: Ad) -> Ad:
        """Update an advertisement."""
        pass
    
    @abstractmethod
    async def delete(self, ad_id: UUID) -> bool:
        """Delete an advertisement."""
        pass
    
    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[Ad]:
        """List all advertisements with pagination."""
        pass
    
    @abstractmethod
    async def search(self, query: str, skip: int = 0, limit: int = 100) -> List[Ad]:
        """Search advertisements by text."""
        pass
    
    @abstractmethod
    async def get_active_ads(self) -> List[Ad]:
        """Get all currently active advertisements."""
        pass
    
    @abstractmethod
    async def get_ads_needing_review(self) -> List[Ad]:
        """Get advertisements that need review."""
        pass


class CampaignRepository(ABC):
    """Repository interface for campaign entities."""
    
    @abstractmethod
    async def create(self, campaign: AdCampaign) -> AdCampaign:
        """Create a new campaign."""
        pass
    
    @abstractmethod
    async def get_by_id(self, campaign_id: UUID) -> Optional[AdCampaign]:
        """Get campaign by ID."""
        pass
    
    @abstractmethod
    async def get_by_user(self, user_id: UUID) -> List[AdCampaign]:
        """Get campaigns by user."""
        pass
    
    @abstractmethod
    async def get_by_platform(self, platform: Platform) -> List[AdCampaign]:
        """Get campaigns by platform."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: AdStatus) -> List[AdCampaign]:
        """Get campaigns by status."""
        pass
    
    @abstractmethod
    async def update(self, campaign: AdCampaign) -> AdCampaign:
        """Update a campaign."""
        pass
    
    @abstractmethod
    async def delete(self, campaign_id: UUID) -> bool:
        """Delete a campaign."""
        pass
    
    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[AdCampaign]:
        """List all campaigns with pagination."""
        pass
    
    @abstractmethod
    async def search(self, query: str, skip: int = 0, limit: int = 100) -> List[AdCampaign]:
        """Search campaigns by text."""
        pass
    
    @abstractmethod
    async def get_active_campaigns(self) -> List[AdCampaign]:
        """Get all currently active campaigns."""
        pass
    
    @abstractmethod
    async def get_campaigns_by_objective(self, objective: str) -> List[AdCampaign]:
        """Get campaigns by objective."""
        pass


class GroupRepository(ABC):
    """Repository interface for ad group entities."""
    
    @abstractmethod
    async def create(self, group: AdGroup) -> AdGroup:
        """Create a new ad group."""
        pass
    
    @abstractmethod
    async def get_by_id(self, group_id: UUID) -> Optional[AdGroup]:
        """Get ad group by ID."""
        pass
    
    @abstractmethod
    async def get_by_campaign(self, campaign_id: UUID) -> List[AdGroup]:
        """Get ad groups by campaign."""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: AdStatus) -> List[AdGroup]:
        """Get ad groups by status."""
        pass
    
    @abstractmethod
    async def update(self, group: AdGroup) -> AdGroup:
        """Update an ad group."""
        pass
    
    @abstractmethod
    async def delete(self, group_id: UUID) -> bool:
        """Delete an ad group."""
        pass
    
    @abstractmethod
    async def list_all(self, skip: int = 0, limit: int = 100) -> List[AdGroup]:
        """List all ad groups with pagination."""
        pass


class PerformanceRepository(ABC):
    """Repository interface for performance tracking entities."""
    
    @abstractmethod
    async def create(self, performance: AdPerformance) -> AdPerformance:
        """Create a new performance record."""
        pass
    
    @abstractmethod
    async def get_by_id(self, performance_id: UUID) -> Optional[AdPerformance]:
        """Get performance record by ID."""
        pass
    
    @abstractmethod
    async def get_by_ad(self, ad_id: UUID) -> List[AdPerformance]:
        """Get performance records by advertisement."""
        pass
    
    @abstractmethod
    async def get_by_campaign(self, campaign_id: UUID) -> List[AdPerformance]:
        """Get performance records by campaign."""
        pass
    
    @abstractmethod
    async def get_by_date_range(self, start_date: str, end_date: str) -> List[AdPerformance]:
        """Get performance records by date range."""
        pass
    
    @abstractmethod
    async def get_by_hour(self, ad_id: UUID, date: str, hour: int) -> Optional[AdPerformance]:
        """Get performance record by specific hour."""
        pass
    
    @abstractmethod
    async def update(self, performance: AdPerformance) -> AdPerformance:
        """Update a performance record."""
        pass
    
    @abstractmethod
    async def delete(self, performance_id: UUID) -> bool:
        """Delete a performance record."""
        pass
    
    @abstractmethod
    async def get_aggregated_metrics(self, ad_id: UUID, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get aggregated performance metrics for an advertisement."""
        pass
    
    @abstractmethod
    async def get_campaign_aggregated_metrics(self, campaign_id: UUID, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get aggregated performance metrics for a campaign."""
        pass
    
    @abstractmethod
    async def get_top_performing_ads(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing advertisements."""
        pass


class AnalyticsRepository(ABC):
    """Repository interface for analytics and reporting."""
    
    @abstractmethod
    async def get_daily_performance(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get daily performance data."""
        pass
    
    @abstractmethod
    async def get_hourly_performance(self, date: str) -> List[Dict[str, Any]]:
        """Get hourly performance data for a specific date."""
        pass
    
    @abstractmethod
    async def get_platform_performance(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get performance data by platform."""
        pass
    
    @abstractmethod
    async def get_ad_type_performance(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get performance data by ad type."""
        pass
    
    @abstractmethod
    async def get_targeting_performance(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get performance data by targeting criteria."""
        pass
    
    @abstractmethod
    async def get_budget_utilization(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get budget utilization data."""
        pass
    
    @abstractmethod
    async def get_conversion_funnel(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get conversion funnel data."""
        pass
    
    @abstractmethod
    async def get_audience_insights(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Get audience insights data."""
        pass


class OptimizationRepository(ABC):
    """Repository interface for optimization data."""
    
    @abstractmethod
    async def save_optimization_result(self, result: Dict[str, Any]) -> bool:
        """Save optimization result."""
        pass
    
    @abstractmethod
    async def get_optimization_history(self, ad_id: UUID) -> List[Dict[str, Any]]:
        """Get optimization history for an advertisement."""
        pass
    
    @abstractmethod
    async def get_best_performing_settings(self, ad_type: AdType, platform: Platform) -> Dict[str, Any]:
        """Get best performing settings for ad type and platform."""
        pass
    
    @abstractmethod
    async def save_performance_prediction(self, prediction: Dict[str, Any]) -> bool:
        """Save performance prediction."""
        pass
    
    @abstractmethod
    async def get_performance_predictions(self, ad_id: UUID) -> List[Dict[str, Any]]:
        """Get performance predictions for an advertisement."""
        pass
