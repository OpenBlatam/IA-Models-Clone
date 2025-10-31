"""
Test Mock Utilities for the ads feature.

This module provides comprehensive mock functions and utilities for testing:
- Mock factories for common objects
- Mock data generators
- Specialized mock objects
- Mock configuration utilities
- Mock behavior customization
"""

import asyncio
import json
import random
import string
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Callable
from unittest.mock import Mock, MagicMock, AsyncMock, patch, PropertyMock

import pytest
from pydantic import BaseModel, ValidationError

from ...domain.entities import Ad, AdCampaign, AdGroup, AdPerformance
from ...domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria, AdMetrics, AdSchedule
from ...domain.repositories import (
    AdRepository, CampaignRepository, GroupRepository, PerformanceRepository,
    AnalyticsRepository, OptimizationRepository
)
from ...domain.services import AdService, CampaignService, OptimizationService
from ...application.use_cases import (
    CreateAdUseCase, ApproveAdUseCase, ActivateAdUseCase, PauseAdUseCase,
    ArchiveAdUseCase, CreateCampaignUseCase, ActivateCampaignUseCase,
    PauseCampaignUseCase, OptimizeAdUseCase, PredictPerformanceUseCase
)
from ...infrastructure.database import DatabaseManager, ConnectionPool
from ...infrastructure.storage import StorageService, FileStorageManager
from ...infrastructure.cache import CacheService, CacheManager
from ...infrastructure.external_services import ExternalServiceManager, AIProviderService

T = TypeVar('T', bound=BaseModel)


class MockDataGenerator:
    """Generates mock data for testing."""
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_id(prefix: str = "test") -> str:
        """Generate a random ID with prefix."""
        return f"{prefix}_{MockDataGenerator.random_string(8)}"
    
    @staticmethod
    def random_email() -> str:
        """Generate a random email address."""
        username = MockDataGenerator.random_string(8)
        domain = MockDataGenerator.random_string(6)
        return f"{username}@{domain}.com"
    
    @staticmethod
    def random_phone() -> str:
        """Generate a random phone number."""
        return f"+1{random.randint(1000000000, 9999999999)}"
    
    @staticmethod
    def random_date(start_date: datetime = None, end_date: datetime = None) -> datetime:
        """Generate a random date between start_date and end_date."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now() + timedelta(days=365)
        
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        return start_date + timedelta(days=random_number_of_days)
    
    @staticmethod
    def random_budget(min_amount: float = 100.0, max_amount: float = 10000.0) -> float:
        """Generate a random budget amount."""
        return round(random.uniform(min_amount, max_amount), 2)
    
    @staticmethod
    def random_percentage(min_percent: float = 0.0, max_percent: float = 100.0) -> float:
        """Generate a random percentage."""
        return round(random.uniform(min_percent, max_percent), 2)


class MockEntityFactory:
    """Factory for creating mock entities."""
    
    @staticmethod
    def create_mock_ad(
        ad_id: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[AdStatus] = None,
        ad_type: Optional[AdType] = None,
        platform: Optional[Platform] = None,
        budget: Optional[Budget] = None,
        targeting_criteria: Optional[TargetingCriteria] = None,
        metrics: Optional[AdMetrics] = None,
        schedule: Optional[AdSchedule] = None
    ) -> Mock:
        """Create a mock Ad entity."""
        mock_ad = Mock(spec=Ad)
        
        # Set basic attributes
        mock_ad.id = ad_id or MockDataGenerator.random_id("ad")
        mock_ad.title = title or f"Test Ad {MockDataGenerator.random_string(5)}"
        mock_ad.description = description or f"Test description for {mock_ad.title}"
        mock_ad.status = status or random.choice(list(AdStatus))
        mock_ad.ad_type = ad_type or random.choice(list(AdType))
        mock_ad.platform = platform or random.choice(list(Platform))
        mock_ad.created_at = datetime.now()
        mock_ad.updated_at = datetime.now()
        
        # Set complex attributes
        if budget is None:
            budget = Budget(
                daily_budget=MockDataGenerator.random_budget(100, 1000),
                total_budget=MockDataGenerator.random_budget(1000, 10000),
                currency="USD"
            )
        mock_ad.budget = budget
        
        if targeting_criteria is None:
            targeting_criteria = TargetingCriteria(
                age_range=(18, 65),
                gender=["male", "female"],
                interests=["technology", "business"],
                location=["US", "CA"],
                language=["en"]
            )
        mock_ad.targeting_criteria = targeting_criteria
        
        if metrics is None:
            metrics = AdMetrics(
                impressions=random.randint(1000, 100000),
                clicks=random.randint(100, 10000),
                conversions=random.randint(10, 1000),
                spend=MockDataGenerator.random_budget(100, 5000)
            )
        mock_ad.metrics = metrics
        
        if schedule is None:
            schedule = AdSchedule(
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=30),
                active_hours=list(range(9, 18)),
                active_days=list(range(1, 6))
            )
        mock_ad.schedule = schedule
        
        # Mock methods
        mock_ad.approve = Mock(return_value=True)
        mock_ad.activate = Mock(return_value=True)
        mock_ad.pause = Mock(return_value=True)
        mock_ad.archive = Mock(return_value=True)
        mock_ad.update_metrics = Mock(return_value=True)
        mock_ad.is_active = PropertyMock(return_value=mock_ad.status == AdStatus.ACTIVE)
        mock_ad.is_paused = PropertyMock(return_value=mock_ad.status == AdStatus.PAUSED)
        mock_ad.is_archived = PropertyMock(return_value=mock_ad.status == AdStatus.ARCHIVED)
        
        return mock_ad
    
    @staticmethod
    def create_mock_campaign(
        campaign_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[AdStatus] = None,
        budget: Optional[Budget] = None,
        ads: Optional[List[Mock]] = None
    ) -> Mock:
        """Create a mock AdCampaign entity."""
        mock_campaign = Mock(spec=AdCampaign)
        
        # Set basic attributes
        mock_campaign.id = campaign_id or MockDataGenerator.random_id("campaign")
        mock_campaign.name = name or f"Test Campaign {MockDataGenerator.random_string(5)}"
        mock_campaign.description = description or f"Test campaign description for {mock_campaign.name}"
        mock_campaign.status = status or random.choice(list(AdStatus))
        mock_campaign.created_at = datetime.now()
        mock_campaign.updated_at = datetime.now()
        
        # Set complex attributes
        if budget is None:
            budget = Budget(
                daily_budget=MockDataGenerator.random_budget(500, 5000),
                total_budget=MockDataGenerator.random_budget(5000, 50000),
                currency="USD"
            )
        mock_campaign.budget = budget
        
        if ads is None:
            ads = [MockEntityFactory.create_mock_ad() for _ in range(random.randint(1, 5))]
        mock_campaign.ads = ads
        
        # Mock methods
        mock_campaign.activate = Mock(return_value=True)
        mock_campaign.pause = Mock(return_value=True)
        mock_campaign.archive = Mock(return_value=True)
        mock_campaign.add_ad = Mock(return_value=True)
        mock_campaign.remove_ad = Mock(return_value=True)
        mock_campaign.is_active = PropertyMock(return_value=mock_campaign.status == AdStatus.ACTIVE)
        mock_campaign.is_paused = PropertyMock(return_value=mock_campaign.status == AdStatus.PAUSED)
        mock_campaign.is_archived = PropertyMock(return_value=mock_campaign.status == AdStatus.ARCHIVED)
        
        return mock_campaign
    
    @staticmethod
    def create_mock_ad_group(
        group_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        ads: Optional[List[Mock]] = None
    ) -> Mock:
        """Create a mock AdGroup entity."""
        mock_group = Mock(spec=AdGroup)
        
        # Set basic attributes
        mock_group.id = group_id or MockDataGenerator.random_id("group")
        mock_group.name = name or f"Test Group {MockDataGenerator.random_string(5)}"
        mock_group.description = description or f"Test group description for {mock_group.name}"
        mock_group.created_at = datetime.now()
        mock_group.updated_at = datetime.now()
        
        # Set complex attributes
        if ads is None:
            ads = [MockEntityFactory.create_mock_ad() for _ in range(random.randint(2, 8))]
        mock_group.ads = ads
        
        # Mock methods
        mock_group.add_ad = Mock(return_value=True)
        mock_group.remove_ad = Mock(return_value=True)
        mock_group.get_ad_count = Mock(return_value=len(ads))
        
        return mock_group
    
    @staticmethod
    def create_mock_performance(
        performance_id: Optional[str] = None,
        ad_id: Optional[str] = None,
        metrics: Optional[AdMetrics] = None,
        date: Optional[datetime] = None
    ) -> Mock:
        """Create a mock AdPerformance entity."""
        mock_performance = Mock(spec=AdPerformance)
        
        # Set basic attributes
        mock_performance.id = performance_id or MockDataGenerator.random_id("perf")
        mock_performance.ad_id = ad_id or MockDataGenerator.random_id("ad")
        mock_performance.date = date or MockDataGenerator.random_date()
        mock_performance.created_at = datetime.now()
        
        # Set complex attributes
        if metrics is None:
            metrics = AdMetrics(
                impressions=random.randint(1000, 100000),
                clicks=random.randint(100, 10000),
                conversions=random.randint(10, 1000),
                spend=MockDataGenerator.random_budget(100, 5000)
            )
        mock_performance.metrics = metrics
        
        # Mock methods
        mock_performance.update_metrics = Mock(return_value=True)
        mock_performance.calculate_ctr = Mock(return_value=random.uniform(0.01, 0.10))
        mock_performance.calculate_conversion_rate = Mock(return_value=random.uniform(0.001, 0.05))
        mock_performance.calculate_roas = Mock(return_value=random.uniform(1.0, 10.0))
        
        return mock_performance


class MockRepositoryFactory:
    """Factory for creating mock repositories."""
    
    @staticmethod
    def create_mock_ad_repository() -> Mock:
        """Create a mock AdRepository."""
        mock_repo = Mock(spec=AdRepository)
        
        # Mock data
        mock_ads = [MockEntityFactory.create_mock_ad() for _ in range(5)]
        
        # Mock methods
        mock_repo.create = AsyncMock(return_value=mock_ads[0])
        mock_repo.get_by_id = AsyncMock(return_value=mock_ads[0])
        mock_repo.get_all = AsyncMock(return_value=mock_ads)
        mock_repo.update = AsyncMock(return_value=True)
        mock_repo.delete = AsyncMock(return_value=True)
        mock_repo.exists = AsyncMock(return_value=True)
        mock_repo.get_by_status = AsyncMock(return_value=[ad for ad in mock_ads if ad.status == AdStatus.ACTIVE])
        mock_repo.get_by_platform = AsyncMock(return_value=mock_ads)
        mock_repo.get_by_campaign = AsyncMock(return_value=mock_ads[:3])
        mock_repo.search = AsyncMock(return_value=mock_ads)
        mock_repo.count = AsyncMock(return_value=len(mock_ads))
        
        return mock_repo
    
    @staticmethod
    def create_mock_campaign_repository() -> Mock:
        """Create a mock CampaignRepository."""
        mock_repo = Mock(spec=CampaignRepository)
        
        # Mock data
        mock_campaigns = [MockEntityFactory.create_mock_campaign() for _ in range(3)]
        
        # Mock methods
        mock_repo.create = AsyncMock(return_value=mock_campaigns[0])
        mock_repo.get_by_id = AsyncMock(return_value=mock_campaigns[0])
        mock_repo.get_all = AsyncMock(return_value=mock_campaigns)
        mock_repo.update = AsyncMock(return_value=True)
        mock_repo.delete = AsyncMock(return_value=True)
        mock_repo.exists = AsyncMock(return_value=True)
        mock_repo.get_by_status = AsyncMock(return_value=[c for c in mock_campaigns if c.status == AdStatus.ACTIVE])
        mock_repo.get_by_budget_range = AsyncMock(return_value=mock_campaigns)
        mock_repo.get_active_campaigns = AsyncMock(return_value=[c for c in mock_campaigns if c.status == AdStatus.ACTIVE])
        mock_repo.search = AsyncMock(return_value=mock_campaigns)
        mock_repo.count = AsyncMock(return_value=len(mock_campaigns))
        
        return mock_repo
    
    @staticmethod
    def create_mock_group_repository() -> Mock:
        """Create a mock GroupRepository."""
        mock_repo = Mock(spec=GroupRepository)
        
        # Mock data
        mock_groups = [MockEntityFactory.create_mock_ad_group() for _ in range(4)]
        
        # Mock methods
        mock_repo.create = AsyncMock(return_value=mock_groups[0])
        mock_repo.get_by_id = AsyncMock(return_value=mock_groups[0])
        mock_repo.get_all = AsyncMock(return_value=mock_groups)
        mock_repo.update = AsyncMock(return_value=True)
        mock_repo.delete = AsyncMock(return_value=True)
        mock_repo.exists = AsyncMock(return_value=True)
        mock_repo.get_by_campaign = AsyncMock(return_value=mock_groups[:2])
        mock_repo.add_ad_to_group = AsyncMock(return_value=True)
        mock_repo.remove_ad_from_group = AsyncMock(return_value=True)
        mock_repo.get_group_ads = AsyncMock(return_value=mock_groups[0].ads)
        
        return mock_repo
    
    @staticmethod
    def create_mock_performance_repository() -> Mock:
        """Create a mock PerformanceRepository."""
        mock_repo = Mock(spec=PerformanceRepository)
        
        # Mock data
        mock_performances = [MockEntityFactory.create_mock_performance() for _ in range(10)]
        
        # Mock methods
        mock_repo.create = AsyncMock(return_value=mock_performances[0])
        mock_repo.get_by_id = AsyncMock(return_value=mock_performances[0])
        mock_repo.get_all = AsyncMock(return_value=mock_performances)
        mock_repo.update = AsyncMock(return_value=True)
        mock_repo.delete = AsyncMock(return_value=True)
        mock_repo.exists = AsyncMock(return_value=True)
        mock_repo.get_by_ad_id = AsyncMock(return_value=mock_performances[:3])
        mock_repo.get_by_date_range = AsyncMock(return_value=mock_performances)
        mock_repo.get_aggregated_metrics = AsyncMock(return_value={
            'total_impressions': sum(p.metrics.impressions for p in mock_performances),
            'total_clicks': sum(p.metrics.clicks for p in mock_performances),
            'total_conversions': sum(p.metrics.conversions for p in mock_performances),
            'total_spend': sum(p.metrics.spend for p in mock_performances)
        })
        mock_repo.get_trends = AsyncMock(return_value=mock_performances)
        
        return mock_repo
    
    @staticmethod
    def create_mock_analytics_repository() -> Mock:
        """Create a mock AnalyticsRepository."""
        mock_repo = Mock(spec=AnalyticsRepository)
        
        # Mock methods
        mock_repo.get_performance_summary = AsyncMock(return_value={
            'total_ads': 25,
            'active_campaigns': 8,
            'total_spend': 15000.0,
            'avg_ctr': 0.025,
            'avg_conversion_rate': 0.015,
            'avg_roas': 3.2
        })
        mock_repo.get_campaign_performance = AsyncMock(return_value={
            'campaign_id': 'campaign_123',
            'impressions': 50000,
            'clicks': 1250,
            'conversions': 75,
            'spend': 2500.0,
            'ctr': 0.025,
            'conversion_rate': 0.06,
            'roas': 3.0
        })
        mock_repo.get_audience_insights = AsyncMock(return_value={
            'top_age_groups': ['25-34', '35-44'],
            'top_interests': ['technology', 'business'],
            'top_locations': ['US', 'CA'],
            'gender_distribution': {'male': 0.6, 'female': 0.4}
        })
        mock_repo.get_competitor_analysis = AsyncMock(return_value={
            'competitor_1': {'share_of_voice': 0.25, 'avg_cpc': 2.5},
            'competitor_2': {'share_of_voice': 0.20, 'avg_cpc': 2.8},
            'competitor_3': {'share_of_voice': 0.15, 'avg_cpc': 3.1}
        })
        
        return mock_repo
    
    @staticmethod
    def create_mock_optimization_repository() -> Mock:
        """Create a mock OptimizationRepository."""
        mock_repo = Mock(spec=OptimizationRepository)
        
        # Mock methods
        mock_repo.save_optimization_result = AsyncMock(return_value=True)
        mock_repo.get_optimization_history = AsyncMock(return_value=[
            {'id': 'opt_1', 'ad_id': 'ad_1', 'optimization_type': 'bid', 'result': 'success'},
            {'id': 'opt_2', 'ad_id': 'ad_2', 'optimization_type': 'targeting', 'result': 'success'},
            {'id': 'opt_3', 'ad_id': 'ad_3', 'optimization_type': 'creative', 'result': 'partial'}
        ])
        mock_repo.get_optimization_recommendations = AsyncMock(return_value=[
            {'ad_id': 'ad_1', 'recommendation': 'Increase bid by 15%', 'expected_impact': 'high'},
            {'ad_id': 'ad_2', 'recommendation': 'Refine targeting criteria', 'expected_impact': 'medium'},
            {'ad_id': 'ad_3', 'recommendation': 'Update ad creative', 'expected_impact': 'medium'}
        ])
        mock_repo.get_optimization_metrics = AsyncMock(return_value={
            'total_optimizations': 150,
            'successful_optimizations': 120,
            'failed_optimizations': 30,
            'avg_performance_improvement': 0.25
        })
        
        return mock_repo


class MockServiceFactory:
    """Factory for creating mock services."""
    
    @staticmethod
    def create_mock_ad_service() -> Mock:
        """Create a mock AdService."""
        mock_service = Mock(spec=AdService)
        
        # Mock methods
        mock_service.create_ad = AsyncMock(return_value=MockEntityFactory.create_mock_ad())
        mock_service.get_ad = AsyncMock(return_value=MockEntityFactory.create_mock_ad())
        mock_service.update_ad = AsyncMock(return_value=True)
        mock_service.delete_ad = AsyncMock(return_value=True)
        mock_service.approve_ad = AsyncMock(return_value=True)
        mock_service.activate_ad = AsyncMock(return_value=True)
        mock_service.pause_ad = AsyncMock(return_value=True)
        mock_service.archive_ad = AsyncMock(return_value=True)
        mock_service.update_ad_metrics = AsyncMock(return_value=True)
        mock_service.validate_ad = Mock(return_value=True)
        mock_service.get_ad_performance = AsyncMock(return_value=MockEntityFactory.create_mock_performance())
        
        return mock_service
    
    @staticmethod
    def create_mock_campaign_service() -> Mock:
        """Create a mock CampaignService."""
        mock_service = Mock(spec=CampaignService)
        
        # Mock methods
        mock_service.create_campaign = AsyncMock(return_value=MockEntityFactory.create_mock_campaign())
        mock_service.get_campaign = AsyncMock(return_value=MockEntityFactory.create_mock_campaign())
        mock_service.update_campaign = AsyncMock(return_value=True)
        mock_service.delete_campaign = AsyncMock(return_value=True)
        mock_service.activate_campaign = AsyncMock(return_value=True)
        mock_service.pause_campaign = AsyncMock(return_value=True)
        mock_service.archive_campaign = AsyncMock(return_value=True)
        mock_service.add_ad_to_campaign = AsyncMock(return_value=True)
        mock_service.remove_ad_from_campaign = AsyncMock(return_value=True)
        mock_service.validate_campaign = Mock(return_value=True)
        mock_service.get_campaign_performance = AsyncMock(return_value={
            'total_impressions': 100000,
            'total_clicks': 2500,
            'total_conversions': 150,
            'total_spend': 5000.0
        })
        
        return mock_service
    
    @staticmethod
    def create_mock_optimization_service() -> Mock:
        """Create a mock OptimizationService."""
        mock_service = Mock(spec=OptimizationService)
        
        # Mock methods
        mock_service.optimize_ad = AsyncMock(return_value={
            'ad_id': 'ad_123',
            'optimization_type': 'bid_optimization',
            'old_bid': 2.5,
            'new_bid': 3.0,
            'expected_improvement': 0.20,
            'status': 'success'
        })
        mock_service.optimize_campaign = AsyncMock(return_value={
            'campaign_id': 'campaign_123',
            'optimizations_applied': ['bid_adjustment', 'targeting_refinement'],
            'expected_improvement': 0.15,
            'status': 'success'
        })
        mock_service.get_optimization_recommendations = AsyncMock(return_value=[
            {'type': 'bid_optimization', 'description': 'Increase bid by 20%', 'priority': 'high'},
            {'type': 'targeting_refinement', 'description': 'Narrow age range', 'priority': 'medium'},
            {'type': 'creative_optimization', 'description': 'Update ad copy', 'priority': 'low'}
        ])
        mock_service.validate_optimization = Mock(return_value=True)
        mock_service.get_optimization_history = AsyncMock(return_value=[
            {'id': 'opt_1', 'timestamp': datetime.now(), 'type': 'bid', 'result': 'success'},
            {'id': 'opt_2', 'timestamp': datetime.now(), 'type': 'targeting', 'result': 'success'}
        ])
        
        return mock_service


class MockUseCaseFactory:
    """Factory for creating mock use cases."""
    
    @staticmethod
    def create_mock_create_ad_use_case() -> Mock:
        """Create a mock CreateAdUseCase."""
        mock_use_case = Mock(spec=CreateAdUseCase)
        
        # Mock methods
        mock_use_case.execute = AsyncMock(return_value={
            'success': True,
            'ad_id': 'ad_123',
            'message': 'Ad created successfully',
            'ad': MockEntityFactory.create_mock_ad()
        })
        mock_use_case.validate_request = Mock(return_value=True)
        mock_use_case.handle_errors = Mock(return_value=None)
        
        return mock_use_case
    
    @staticmethod
    def create_mock_approve_ad_use_case() -> Mock:
        """Create a mock ApproveAdUseCase."""
        mock_use_case = Mock(spec=ApproveAdUseCase)
        
        # Mock methods
        mock_use_case.execute = AsyncMock(return_value={
            'success': True,
            'ad_id': 'ad_123',
            'message': 'Ad approved successfully',
            'status': AdStatus.APPROVED
        })
        mock_use_case.validate_request = Mock(return_value=True)
        mock_use_case.handle_errors = Mock(return_value=None)
        
        return mock_use_case
    
    @staticmethod
    def create_mock_optimize_ad_use_case() -> Mock:
        """Create a mock OptimizeAdUseCase."""
        mock_use_case = Mock(spec=OptimizeAdUseCase)
        
        # Mock methods
        mock_use_case.execute = AsyncMock(return_value={
            'success': True,
            'ad_id': 'ad_123',
            'optimization_type': 'bid_optimization',
            'old_metrics': {'ctr': 0.02, 'conversion_rate': 0.01},
            'new_metrics': {'ctr': 0.025, 'conversion_rate': 0.012},
            'improvement': 0.25
        })
        mock_use_case.validate_request = Mock(return_value=True)
        mock_use_case.handle_errors = Mock(return_value=None)
        
        return mock_use_case


class MockInfrastructureFactory:
    """Factory for creating mock infrastructure components."""
    
    @staticmethod
    def create_mock_database_manager() -> Mock:
        """Create a mock DatabaseManager."""
        mock_db = Mock(spec=DatabaseManager)
        
        # Mock methods
        mock_db.get_session = AsyncMock(return_value=Mock())
        mock_db.execute_query = AsyncMock(return_value=[])
        mock_db.execute_transaction = AsyncMock(return_value=True)
        mock_db.health_check = AsyncMock(return_value={'status': 'healthy', 'connections': 5})
        mock_db.get_stats = Mock(return_value={
            'total_queries': 1000,
            'active_connections': 5,
            'connection_pool_size': 10
        })
        
        return mock_db
    
    @staticmethod
    def create_mock_storage_manager() -> Mock:
        """Create a mock FileStorageManager."""
        mock_storage = Mock(spec=FileStorageManager)
        
        # Mock methods
        mock_storage.save_file = AsyncMock(return_value='file_url_123')
        mock_storage.get_file = AsyncMock(return_value=b'file_content')
        mock_storage.delete_file = AsyncMock(return_value=True)
        mock_storage.file_exists = AsyncMock(return_value=True)
        mock_storage.get_file_url = Mock(return_value='https://storage.example.com/file_123')
        mock_storage.get_file_info = AsyncMock(return_value={
            'size': 1024,
            'type': 'image/jpeg',
            'created_at': datetime.now()
        })
        
        return mock_storage
    
    @staticmethod
    def create_mock_cache_manager() -> Mock:
        """Create a mock CacheManager."""
        mock_cache = Mock(spec=CacheManager)
        
        # Mock methods
        mock_cache.get = AsyncMock(return_value='cached_value')
        mock_cache.set = AsyncMock(return_value=True)
        mock_cache.delete = AsyncMock(return_value=True)
        mock_cache.exists = AsyncMock(return_value=True)
        mock_cache.clear = AsyncMock(return_value=True)
        mock_cache.get_stats = Mock(return_value={
            'hits': 150,
            'misses': 50,
            'hit_rate': 0.75,
            'total_keys': 100
        })
        
        return mock_cache
    
    @staticmethod
    def create_mock_external_service_manager() -> Mock:
        """Create a mock ExternalServiceManager."""
        mock_service = Mock(spec=ExternalServiceManager)
        
        # Mock methods
        mock_service.make_request = AsyncMock(return_value={
            'status_code': 200,
            'data': {'result': 'success'},
            'headers': {'content-type': 'application/json'}
        })
        mock_service.health_check = AsyncMock(return_value={'status': 'healthy'})
        mock_service.get_rate_limit_info = Mock(return_value={
            'remaining': 95,
            'reset_time': datetime.now() + timedelta(hours=1)
        })
        
        return mock_service


class MockConfigurationFactory:
    """Factory for creating mock configurations."""
    
    @staticmethod
    def create_mock_database_config() -> Dict[str, Any]:
        """Create a mock database configuration."""
        return {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'username': 'test_user',
            'password': 'test_password',
            'pool_size': 10,
            'max_overflow': 20,
            'pool_timeout': 30,
            'pool_recycle': 3600
        }
    
    @staticmethod
    def create_mock_redis_config() -> Dict[str, Any]:
        """Create a mock Redis configuration."""
        return {
            'host': 'localhost',
            'port': 6379,
            'database': 0,
            'password': None,
            'max_connections': 20,
            'socket_timeout': 5,
            'socket_connect_timeout': 5
        }
    
    @staticmethod
    def create_mock_storage_config() -> Dict[str, Any]:
        """Create a mock storage configuration."""
        return {
            'type': 'local',
            'base_path': '/tmp/test_storage',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'allowed_types': ['jpg', 'png', 'gif', 'mp4'],
            'compression': True,
            'encryption': False
        }
    
    @staticmethod
    def create_mock_api_config() -> Dict[str, Any]:
        """Create a mock API configuration."""
        return {
            'host': '0.0.0.0',
            'port': 8000,
            'workers': 4,
            'max_requests': 1000,
            'timeout': 30,
            'cors_origins': ['http://localhost:3000'],
            'rate_limit': {
                'requests_per_minute': 100,
                'burst_size': 20
            }
        }


class MockBehaviorCustomizer:
    """Utility for customizing mock behavior."""
    
    @staticmethod
    def make_mock_raise_exception(
        mock: Mock,
        method_name: str,
        exception: Exception,
        call_count: int = 1
    ) -> None:
        """Make a mock method raise an exception after a certain number of calls."""
        method = getattr(mock, method_name)
        method.side_effect = [None] * (call_count - 1) + [exception]
    
    @staticmethod
    def make_mock_return_different_values(
        mock: Mock,
        method_name: str,
        values: List[Any]
    ) -> None:
        """Make a mock method return different values on subsequent calls."""
        method = getattr(mock, method_name)
        method.side_effect = values
    
    @staticmethod
    def make_mock_delay_response(
        mock: Mock,
        method_name: str,
        delay_seconds: float = 1.0
    ) -> None:
        """Make a mock method delay its response."""
        async def delayed_response(*args, **kwargs):
            await asyncio.sleep(delay_seconds)
            return mock.return_value
        
        method = getattr(mock, method_name)
        method.side_effect = delayed_response
    
    @staticmethod
    def make_mock_validate_input(
        mock: Mock,
        method_name: str,
        validator: Callable
    ) -> None:
        """Make a mock method validate its input using a custom validator."""
        def validated_method(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError("Input validation failed")
            return mock.return_value
        
        method = getattr(mock, method_name)
        method.side_effect = validated_method


# Export all mock factory classes and functions
__all__ = [
    'MockDataGenerator',
    'MockEntityFactory',
    'MockRepositoryFactory',
    'MockServiceFactory',
    'MockUseCaseFactory',
    'MockInfrastructureFactory',
    'MockConfigurationFactory',
    'MockBehaviorCustomizer'
]


# Backwards-compatibility helpers expected by conftest
def create_mock_repository(*args, **kwargs) -> Mock:
    """Return a simple repository mock for tests."""
    return Mock()


def create_mock_service(service_class: Any, dependencies: Optional[Dict[str, Any]] = None) -> Mock:
    """Return a simple service mock; attach provided dependency mocks as attributes."""
    svc = Mock(spec=service_class) if service_class is not None else Mock()
    if isinstance(dependencies, dict):
        for key, dep in dependencies.items():
            setattr(svc, key, dep)
    return svc
