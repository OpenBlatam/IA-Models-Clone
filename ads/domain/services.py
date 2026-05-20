"""
🎯 ADS Domain - Domain Services

Domain services contain business logic that doesn't belong to a single entity
or value object, but rather orchestrates multiple entities and business rules.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from decimal import Decimal
from datetime import datetime, timezone

from .entities import Ad, AdCampaign, AdGroup, AdPerformance
from .value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria
from .repositories import (
    AdRepository, CampaignRepository, GroupRepository, 
    PerformanceRepository, AnalyticsRepository, OptimizationRepository
)


class AdService:
    """Domain service for advertisement operations."""
    
    def __init__(self, ad_repository: AdRepository):
        self.ad_repository = ad_repository
    
    async def create_ad(self, ad_data: Dict[str, Any]) -> Ad:
        """Create a new advertisement with business rules validation."""
        # Validate business rules
        self._validate_ad_creation(ad_data)
        
        # Create ad entity
        ad = Ad(**ad_data)
        
        # Set initial status
        ad.status = AdStatus.DRAFT
        
        # Save to repository
        return await self.ad_repository.create(ad)
    
    async def approve_ad(self, ad_id: UUID, approver_id: UUID) -> Ad:
        """Approve an advertisement with business rules validation."""
        ad = await self.ad_repository.get_by_id(ad_id)
        if not ad:
            raise ValueError("Advertisement not found")
        
        # Business rule: Only pending review ads can be approved
        if ad.status != AdStatus.PENDING_REVIEW:
            raise ValueError("Only pending review advertisements can be approved")
        
        # Business rule: Ad must have valid content
        if not self._validate_ad_content(ad):
            raise ValueError("Advertisement content is not valid for approval")
        
        # Business rule: Ad must have valid targeting
        if not ad.targeting.is_valid():
            raise ValueError("Advertisement targeting is not valid for approval")
        
        # Approve the ad
        ad.approve()
        
        # Update in repository
        return await self.ad_repository.update(ad)
    
    async def activate_ad(self, ad_id: UUID) -> Ad:
        """Activate an advertisement with business rules validation."""
        ad = await self.ad_repository.get_by_id(ad_id)
        if not ad:
            raise ValueError("Advertisement not found")
        
        # Business rule: Only approved ads can be activated
        if ad.status != AdStatus.APPROVED:
            raise ValueError("Only approved advertisements can be activated")
        
        # Business rule: Ad must have active schedule
        if ad.schedule and not ad.schedule.is_active():
            raise ValueError("Advertisement schedule is not active")
        
        # Business rule: Ad must have sufficient budget
        if not self._validate_ad_budget(ad):
            raise ValueError("Advertisement budget is insufficient for activation")
        
        # Activate the ad
        ad.activate()
        
        # Update in repository
        return await self.ad_repository.update(ad)
    
    async def pause_ad(self, ad_id: UUID) -> Ad:
        """Pause an advertisement."""
        ad = await self.ad_repository.get_by_id(ad_id)
        if not ad:
            raise ValueError("Advertisement not found")
        
        # Business rule: Only active ads can be paused
        if ad.status != AdStatus.ACTIVE:
            raise ValueError("Only active advertisements can be paused")
        
        # Pause the ad
        ad.pause()
        
        # Update in repository
        return await self.ad_repository.update(ad)
    
    async def archive_ad(self, ad_id: UUID) -> Ad:
        """Archive an advertisement."""
        ad = await self.ad_repository.get_by_id(ad_id)
        if not ad:
            raise ValueError("Advertisement not found")
        
        # Business rule: Active or paused ads cannot be archived
        if ad.status in [AdStatus.ACTIVE, AdStatus.PAUSED]:
            raise ValueError("Active or paused advertisements cannot be archived")
        
        # Archive the ad
        ad.archive()
        
        # Update in repository
        return await self.ad_repository.update(ad)
    
    def _validate_ad_creation(self, ad_data: Dict[str, Any]) -> None:
        """Validate business rules for ad creation."""
        # Business rule: Ad name must be unique (would need to check repository)
        if not ad_data.get('name', '').strip():
            raise ValueError("Advertisement name is required")
        
        # Business rule: Headline must be within character limits
        headline = ad_data.get('headline', '')
        if len(headline) > 40:  # Example limit
            raise ValueError("Advertisement headline is too long")
        
        # Business rule: Body text must be within character limits
        body_text = ad_data.get('body_text', '')
        if len(body_text) > 125:  # Example limit
            raise ValueError("Advertisement body text is too long")
    
    def _validate_ad_content(self, ad: Ad) -> bool:
        """Validate advertisement content for approval."""
        # Business rule: Image ads must have image URL
        if ad.ad_type == AdType.IMAGE and not ad.image_url:
            return False
        
        # Business rule: Video ads must have video URL
        if ad.ad_type == AdType.VIDEO and not ad.video_url:
            return False
        
        # Business rule: All ads must have call to action
        if not ad.call_to_action:
            return False
        
        return True
    
    def _validate_ad_budget(self, ad: Ad) -> bool:
        """Validate advertisement budget for activation."""
        # Business rule: Ad must have positive budget
        if ad.budget.amount <= 0:
            return False
        
        # Business rule: Daily limit must be reasonable
        if ad.budget.daily_limit and ad.budget.daily_limit < Decimal('10'):
            return False
        
        return True


class CampaignService:
    """Domain service for campaign operations."""
    
    def __init__(self, campaign_repository: CampaignRepository):
        self.campaign_repository = campaign_repository
    
    async def create_campaign(self, campaign_data: Dict[str, Any]) -> AdCampaign:
        """Create a new campaign with business rules validation."""
        # Validate business rules
        self._validate_campaign_creation(campaign_data)
        
        # Create campaign entity
        campaign = AdCampaign(**campaign_data)
        
        # Set initial status
        campaign.status = AdStatus.DRAFT
        
        # Save to repository
        return await self.campaign_repository.create(campaign)
    
    async def activate_campaign(self, campaign_id: UUID) -> AdCampaign:
        """Activate a campaign with business rules validation."""
        campaign = await self.campaign_repository.get_by_id(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")
        
        # Business rule: Only approved campaigns can be activated
        if campaign.status != AdStatus.APPROVED:
            raise ValueError("Only approved campaigns can be activated")
        
        # Business rule: Campaign must have active schedule
        if campaign.schedule and not campaign.schedule.is_active():
            raise ValueError("Campaign schedule is not active")
        
        # Business rule: Campaign must have sufficient budget
        if not self._validate_campaign_budget(campaign):
            raise ValueError("Campaign budget is insufficient for activation")
        
        # Activate the campaign
        campaign.status = AdStatus.ACTIVE
        campaign.updated_at = datetime.now(timezone.utc)
        
        # Update in repository
        return await self.campaign_repository.update(campaign)
    
    async def pause_campaign(self, campaign_id: UUID) -> AdCampaign:
        """Pause a campaign."""
        campaign = await self.campaign_repository.get_by_id(campaign_id)
        if not campaign:
            raise ValueError("Campaign not found")
        
        # Business rule: Only active campaigns can be paused
        if campaign.status != AdStatus.ACTIVE:
            raise ValueError("Only active campaigns can be paused")
        
        # Pause the campaign
        campaign.status = AdStatus.PAUSED
        campaign.updated_at = datetime.now(timezone.utc)
        
        # Update in repository
        return await self.campaign_repository.update(campaign)
    
    def _validate_campaign_creation(self, campaign_data: Dict[str, Any]) -> None:
        """Validate business rules for campaign creation."""
        # Business rule: Campaign name must be unique
        if not campaign_data.get('name', '').strip():
            raise ValueError("Campaign name is required")
        
        # Business rule: Campaign objective must be specified
        if not campaign_data.get('objective', '').strip():
            raise ValueError("Campaign objective is required")
        
        # Business rule: Campaign budget must be reasonable
        budget = campaign_data.get('budget')
        if budget and budget.amount < Decimal('100'):
            raise ValueError("Campaign budget must be at least $100")
    
    def _validate_campaign_budget(self, campaign: AdCampaign) -> bool:
        """Validate campaign budget for activation."""
        # Business rule: Campaign must have positive budget
        if campaign.budget.amount <= 0:
            return False
        
        # Business rule: Daily limit must be reasonable
        if campaign.budget.daily_limit and campaign.budget.daily_limit < Decimal('50'):
            return False
        
        return True


class OptimizationService:
    """Domain service for advertisement optimization."""
    
    def __init__(self, 
                 ad_repository: AdRepository,
                 performance_repository: PerformanceRepository,
                 optimization_repository: OptimizationRepository):
        self.ad_repository = ad_repository
        self.performance_repository = performance_repository
        self.optimization_repository = optimization_repository
    
    async def optimize_ad_performance(self, ad_id: UUID) -> Dict[str, Any]:
        """Optimize advertisement performance based on historical data."""
        ad = await self.ad_repository.get_by_id(ad_id)
        if not ad:
            raise ValueError("Advertisement not found")
        
        # Get performance history
        performance_history = await self.performance_repository.get_by_ad(ad_id)
        
        # Analyze performance patterns
        optimization_recommendations = self._analyze_performance_patterns(performance_history)
        
        # Generate optimization suggestions
        suggestions = self._generate_optimization_suggestions(ad, optimization_recommendations)
        
        # Save optimization result
        optimization_result = {
            'ad_id': str(ad_id),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'recommendations': optimization_recommendations,
            'suggestions': suggestions
        }
        
        await self.optimization_repository.save_optimization_result(optimization_result)
        
        return {
            'ad_id': str(ad_id),
            'optimization_recommendations': optimization_recommendations,
            'suggestions': suggestions
        }
    
    async def predict_ad_performance(self, ad_id: UUID, 
                                   target_date: str) -> Dict[str, Any]:
        """Predict advertisement performance for a target date."""
        ad = await self.ad_repository.get_by_id(ad_id)
        if not ad:
            raise ValueError("Advertisement not found")
        
        # Get historical performance data
        performance_data = await self.performance_repository.get_by_ad(ad_id)
        
        # Calculate performance trends
        trends = self._calculate_performance_trends(performance_data)
        
        # Generate performance prediction
        prediction = self._generate_performance_prediction(ad, trends, target_date)
        
        # Save prediction
        await self.optimization_repository.save_performance_prediction(prediction)
        
        return prediction
    
    def _analyze_performance_patterns(self, performance_history: List[AdPerformance]) -> Dict[str, Any]:
        """Analyze performance patterns from historical data."""
        if not performance_history:
            return {'message': 'Insufficient data for analysis'}
        
        # Calculate average metrics
        total_impressions = sum(p.impressions for p in performance_history)
        total_clicks = sum(p.clicks for p in performance_history)
        total_conversions = sum(p.conversions for p in performance_history)
        total_spend = sum(p.spend for p in performance_history)
        
        avg_ctr = (total_clicks / total_impressions * 100) if total_impressions > 0 else 0
        avg_cpc = (total_spend / total_clicks) if total_clicks > 0 else 0
        avg_conversion_rate = (total_conversions / total_clicks * 100) if total_clicks > 0 else 0
        
        # Identify patterns
        patterns = {
            'average_metrics': {
                'ctr': float(avg_ctr),
                'cpc': float(avg_cpc),
                'conversion_rate': float(avg_conversion_rate)
            },
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'total_conversions': total_conversions,
            'total_spend': float(total_spend)
        }
        
        return patterns
    
    def _generate_optimization_suggestions(self, ad: Ad, 
                                         recommendations: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on recommendations."""
        suggestions = []
        
        # Check CTR performance
        if 'average_metrics' in recommendations:
            avg_ctr = recommendations['average_metrics'].get('ctr', 0)
            if avg_ctr < 1.0:  # Below 1% CTR
                suggestions.append("Consider improving ad creative to increase click-through rate")
                suggestions.append("Review targeting criteria for better audience match")
            
            avg_cpc = recommendations['average_metrics'].get('cpc', 0)
            if avg_cpc > 2.0:  # High cost per click
                suggestions.append("Optimize bidding strategy to reduce cost per click")
                suggestions.append("Review keyword selection and negative keywords")
        
        # Check budget utilization
        if ad.budget.daily_limit:
            daily_spend = ad.metrics.spend
            if daily_spend > ad.budget.daily_limit * Decimal('0.8'):
                suggestions.append("Daily budget is being utilized efficiently")
            else:
                suggestions.append("Consider increasing daily budget or improving ad performance")
        
        return suggestions
    
    def _calculate_performance_trends(self, performance_data: List[AdPerformance]) -> Dict[str, Any]:
        """Calculate performance trends from historical data."""
        if len(performance_data) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        # Sort by date
        sorted_data = sorted(performance_data, key=lambda x: x.date)
        
        # Calculate day-over-day changes
        trends = {
            'impression_trend': [],
            'click_trend': [],
            'conversion_trend': [],
            'spend_trend': []
        }
        
        for i in range(1, len(sorted_data)):
            prev = sorted_data[i-1]
            curr = sorted_data[i]
            
            trends['impression_trend'].append(curr.impressions - prev.impressions)
            trends['click_trend'].append(curr.clicks - prev.clicks)
            trends['conversion_trend'].append(curr.conversions - prev.conversions)
            trends['spend_trend'].append(float(curr.spend - prev.spend))
        
        return trends
    
    def _generate_performance_prediction(self, ad: Ad, trends: Dict[str, Any], 
                                       target_date: str) -> Dict[str, Any]:
        """Generate performance prediction for target date."""
        prediction = {
            'ad_id': str(ad.id),
            'target_date': target_date,
            'prediction_timestamp': datetime.now(timezone.utc).isoformat(),
            'predicted_metrics': {},
            'confidence_level': 'medium'
        }
        
        if 'message' in trends:
            prediction['predicted_metrics'] = {'message': trends['message']}
            prediction['confidence_level'] = 'low'
        else:
            # Simple prediction based on trends (in production, use ML models)
            avg_impression_change = sum(trends['impression_trend']) / len(trends['impression_trend'])
            avg_click_change = sum(trends['click_trend']) / len(trends['click_trend'])
            
            prediction['predicted_metrics'] = {
                'impressions': max(0, ad.metrics.impressions + int(avg_impression_change)),
                'clicks': max(0, ad.metrics.clicks + int(avg_click_change)),
                'estimated_spend': float(ad.metrics.spend * Decimal('1.1'))  # 10% increase estimate
            }
        
        return prediction
