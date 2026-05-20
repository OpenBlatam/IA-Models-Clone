"""
Test Helper Utilities for the ads feature.

This module provides comprehensive helper functions and utilities for testing:
- Data generation and manipulation
- Validation helpers
- Performance testing utilities
- Common testing operations
- Test data factories
- Assertion helpers
- Mock data generators
"""

import asyncio
import json
import random
import string
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from unittest.mock import Mock, MagicMock, AsyncMock

import pytest
from pydantic import BaseModel, ValidationError

from ...domain.entities import Ad, AdCampaign, AdGroup, AdPerformance
from ...domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria, AdMetrics, AdSchedule
from ...application.dto import (
    CreateAdRequest, CreateAdResponse, ApproveAdRequest, ApproveAdResponse,
    ActivateAdRequest, ActivateAdResponse, PauseAdRequest, PauseAdResponse,
    ArchiveAdRequest, ArchiveAdResponse, CreateCampaignRequest, CreateCampaignResponse,
    ActivateCampaignRequest, ActivateCampaignResponse, PauseCampaignRequest, PauseCampaignResponse,
    OptimizeAdRequest, OptimizeAdResponse, PerformancePredictionRequest, PerformancePredictionResponse
)

T = TypeVar('T', bound=BaseModel)


class TestDataGenerator:
    """Generates test data for various test scenarios."""
    
    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate a random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    
    @staticmethod
    def random_email() -> str:
        """Generate a random email address."""
        username = TestDataGenerator.random_string(8)
        domain = TestDataGenerator.random_string(6)
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


class EntityFactory:
    """Factory for creating test entities."""
    
    @staticmethod
    def create_ad(
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
    ) -> Ad:
        """Create a test Ad entity."""
        if ad_id is None:
            ad_id = f"ad_{TestDataGenerator.random_string(8)}"
        if title is None:
            title = f"Test Ad {TestDataGenerator.random_string(5)}"
        if description is None:
            description = f"Test description for {title}"
        if status is None:
            status = random.choice(list(AdStatus))
        if ad_type is None:
            ad_type = random.choice(list(AdType))
        if platform is None:
            platform = random.choice(list(Platform))
        if budget is None:
            budget = Budget(
                daily_budget=TestDataGenerator.random_budget(100, 1000),
                total_budget=TestDataGenerator.random_budget(1000, 10000),
                currency="USD"
            )
        if targeting_criteria is None:
            targeting_criteria = TargetingCriteria(
                age_range=(18, 65),
                gender=["male", "female"],
                interests=["technology", "business"],
                location=["US", "CA"],
                language=["en"]
            )
        if metrics is None:
            metrics = AdMetrics(
                impressions=random.randint(1000, 100000),
                clicks=random.randint(100, 10000),
                conversions=random.randint(10, 1000),
                spend=TestDataGenerator.random_budget(100, 5000)
            )
        if schedule is None:
            schedule = AdSchedule(
                start_date=datetime.now(),
                end_date=datetime.now() + timedelta(days=30),
                active_hours=list(range(9, 18)),
                active_days=list(range(1, 6))
            )
        
        return Ad(
            id=ad_id,
            title=title,
            description=description,
            status=status,
            ad_type=ad_type,
            platform=platform,
            budget=budget,
            targeting_criteria=targeting_criteria,
            metrics=metrics,
            schedule=schedule,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @staticmethod
    def create_campaign(
        campaign_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[AdStatus] = None,
        budget: Optional[Budget] = None,
        ads: Optional[List[Ad]] = None
    ) -> AdCampaign:
        """Create a test AdCampaign entity."""
        if campaign_id is None:
            campaign_id = f"campaign_{TestDataGenerator.random_string(8)}"
        if name is None:
            name = f"Test Campaign {TestDataGenerator.random_string(5)}"
        if description is None:
            description = f"Test campaign description for {name}"
        if status is None:
            status = random.choice(list(AdStatus))
        if budget is None:
            budget = Budget(
                daily_budget=TestDataGenerator.random_budget(500, 5000),
                total_budget=TestDataGenerator.random_budget(5000, 50000),
                currency="USD"
            )
        if ads is None:
            ads = [EntityFactory.create_ad() for _ in range(random.randint(1, 5))]
        
        return AdCampaign(
            id=campaign_id,
            name=name,
            description=description,
            status=status,
            budget=budget,
            ads=ads,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @staticmethod
    def create_ad_group(
        group_id: Optional[str] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        ads: Optional[List[Ad]] = None
    ) -> AdGroup:
        """Create a test AdGroup entity."""
        if group_id is None:
            group_id = f"group_{TestDataGenerator.random_string(8)}"
        if name is None:
            name = f"Test Group {TestDataGenerator.random_string(5)}"
        if description is None:
            description = f"Test group description for {name}"
        if ads is None:
            ads = [EntityFactory.create_ad() for _ in range(random.randint(2, 8))]
        
        return AdGroup(
            id=group_id,
            name=name,
            description=description,
            ads=ads,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
    
    @staticmethod
    def create_performance(
        performance_id: Optional[str] = None,
        ad_id: Optional[str] = None,
        metrics: Optional[AdMetrics] = None,
        date: Optional[datetime] = None
    ) -> AdPerformance:
        """Create a test AdPerformance entity."""
        if performance_id is None:
            performance_id = f"perf_{TestDataGenerator.random_string(8)}"
        if ad_id is None:
            ad_id = f"ad_{TestDataGenerator.random_string(8)}"
        if metrics is None:
            metrics = AdMetrics(
                impressions=random.randint(1000, 100000),
                clicks=random.randint(100, 10000),
                conversions=random.randint(10, 1000),
                spend=TestDataGenerator.random_budget(100, 5000)
            )
        if date is None:
            date = TestDataGenerator.random_date()
        
        return AdPerformance(
            id=performance_id,
            ad_id=ad_id,
            metrics=metrics,
            date=date,
            created_at=datetime.now()
        )


class DTOFactory:
    """Factory for creating test DTOs."""
    
    @staticmethod
    def create_ad_request(
        title: Optional[str] = None,
        description: Optional[str] = None,
        ad_type: Optional[AdType] = None,
        platform: Optional[Platform] = None,
        budget: Optional[float] = None,
        targeting_criteria: Optional[Dict[str, Any]] = None
    ) -> CreateAdRequest:
        """Create a test CreateAdRequest DTO."""
        if title is None:
            title = f"Test Ad {TestDataGenerator.random_string(5)}"
        if description is None:
            description = f"Test description for {title}"
        if ad_type is None:
            ad_type = random.choice(list(AdType))
        if platform is None:
            platform = random.choice(list(Platform))
        if budget is None:
            budget = TestDataGenerator.random_budget(100, 1000)
        if targeting_criteria is None:
            targeting_criteria = {
                "age_range": [18, 65],
                "gender": ["male", "female"],
                "interests": ["technology", "business"],
                "location": ["US", "CA"],
                "language": ["en"]
            }
        
        return CreateAdRequest(
            title=title,
            description=description,
            ad_type=ad_type,
            platform=platform,
            budget=budget,
            targeting_criteria=targeting_criteria
        )
    
    @staticmethod
    def create_campaign_request(
        name: Optional[str] = None,
        description: Optional[str] = None,
        budget: Optional[float] = None
    ) -> CreateCampaignRequest:
        """Create a test CreateCampaignRequest DTO."""
        if name is None:
            name = f"Test Campaign {TestDataGenerator.random_string(5)}"
        if description is None:
            description = f"Test campaign description for {name}"
        if budget is None:
            budget = TestDataGenerator.random_budget(1000, 10000)
        
        return CreateCampaignRequest(
            name=name,
            description=description,
            budget=budget
        )
    
    @staticmethod
    def create_optimization_request(
        ad_id: Optional[str] = None,
        optimization_level: Optional[str] = None,
        target_metrics: Optional[List[str]] = None
    ) -> OptimizeAdRequest:
        """Create a test OptimizeAdRequest DTO."""
        if ad_id is None:
            ad_id = f"ad_{TestDataGenerator.random_string(8)}"
        if optimization_level is None:
            optimization_level = random.choice(["light", "standard", "aggressive", "extreme"])
        if target_metrics is None:
            target_metrics = ["ctr", "conversion_rate", "roas"]
        
        return OptimizeAdRequest(
            ad_id=ad_id,
            optimization_level=optimization_level,
            target_metrics=target_metrics
        )


class ValidationHelper:
    """Helper functions for validation testing."""
    
    @staticmethod
    def assert_valid_entity(entity: Any, entity_type: Type[T]) -> None:
        """Assert that an entity is valid and of the correct type."""
        assert entity is not None
        assert isinstance(entity, entity_type)
        assert hasattr(entity, 'id')
        assert entity.id is not None
    
    @staticmethod
    def assert_valid_dto(dto: Any, dto_type: Type[T]) -> None:
        """Assert that a DTO is valid and of the correct type."""
        assert dto is not None
        assert isinstance(dto, dto_type)
    
    @staticmethod
    def assert_entity_has_required_fields(entity: Any, required_fields: List[str]) -> None:
        """Assert that an entity has all required fields."""
        for field in required_fields:
            assert hasattr(entity, field), f"Entity missing required field: {field}"
            assert getattr(entity, field) is not None, f"Required field {field} is None"
    
    @staticmethod
    def assert_dto_has_required_fields(dto: Any, required_fields: List[str]) -> None:
        """Assert that a DTO has all required fields."""
        for field in required_fields:
            assert hasattr(dto, field), f"DTO missing required field: {field}"
            assert getattr(dto, field) is not None, f"Required field {field} is None"
    
    @staticmethod
    def assert_valid_status_transition(
        entity: Any,
        old_status: Any,
        new_status: Any,
        valid_transitions: Dict[Any, List[Any]]
    ) -> None:
        """Assert that a status transition is valid."""
        assert old_status in valid_transitions, f"Old status {old_status} not found in valid transitions"
        assert new_status in valid_transitions[old_status], f"Invalid transition from {old_status} to {new_status}"
    
    @staticmethod
    def assert_metrics_improvement(
        old_metrics: AdMetrics,
        new_metrics: AdMetrics,
        target_metrics: List[str]
    ) -> None:
        """Assert that metrics have improved."""
        for metric in target_metrics:
            if hasattr(old_metrics, metric) and hasattr(new_metrics, metric):
                old_value = getattr(old_metrics, metric)
                new_value = getattr(new_metrics, metric)
                if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
                    assert new_value >= old_value, f"Metric {metric} did not improve: {old_value} -> {new_value}"


class PerformanceHelper:
    """Helper functions for performance testing."""
    
    @staticmethod
    async def measure_execution_time(coro) -> float:
        """Measure the execution time of a coroutine."""
        start_time = time.time()
        await coro
        end_time = time.time()
        return end_time - start_time
    
    @staticmethod
    def assert_performance_threshold(execution_time: float, max_time: float) -> None:
        """Assert that execution time is within acceptable threshold."""
        assert execution_time <= max_time, f"Execution time {execution_time}s exceeds threshold {max_time}s"
    
    @staticmethod
    async def stress_test(
        operation,
        num_iterations: int = 100,
        max_concurrent: int = 10,
        max_time_per_operation: float = 1.0
    ) -> Dict[str, Any]:
        """Perform stress testing on an operation."""
        start_time = time.time()
        results = []
        errors = []
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_operation():
            async with semaphore:
                try:
                    op_start = time.time()
                    result = await operation()
                    op_time = time.time() - op_start
                    
                    if op_time > max_time_per_operation:
                        errors.append(f"Operation exceeded time limit: {op_time}s")
                    
                    results.append({
                        'result': result,
                        'execution_time': op_time,
                        'success': True
                    })
                except Exception as e:
                    errors.append(str(e))
                    results.append({
                        'result': None,
                        'execution_time': 0,
                        'success': False,
                        'error': str(e)
                    })
        
        # Run operations concurrently
        tasks = [run_operation() for _ in range(num_iterations)]
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        successful_ops = sum(1 for r in results if r['success'])
        failed_ops = num_iterations - successful_ops
        
        return {
            'total_iterations': num_iterations,
            'successful_operations': successful_ops,
            'failed_operations': failed_ops,
            'total_time': total_time,
            'average_time_per_operation': total_time / num_iterations,
            'success_rate': successful_ops / num_iterations,
            'errors': errors,
            'results': results
        }
    
    @staticmethod
    def assert_stress_test_results(results: Dict[str, Any], min_success_rate: float = 0.95) -> None:
        """Assert that stress test results meet minimum requirements."""
        assert results['success_rate'] >= min_success_rate, f"Success rate {results['success_rate']} below threshold {min_success_rate}"
        assert results['failed_operations'] == 0, f"Stress test had {results['failed_operations']} failures"


class AsyncTestHelper:
    """Helper functions for async testing."""
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout: float = 10.0,
        check_interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true with timeout."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await condition_func():
                return True
            await asyncio.sleep(check_interval)
        return False
    
    @staticmethod
    async def retry_operation(
        operation,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0
    ):
        """Retry an operation with exponential backoff."""
        last_exception = None
        current_delay = delay
        
        for attempt in range(max_retries):
            try:
                return await operation()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
        
        raise last_exception
    
    @staticmethod
    async def run_concurrent_operations(
        operations: List,
        max_concurrent: int = 5
    ) -> List[Any]:
        """Run multiple operations concurrently with limited concurrency."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_with_semaphore(operation):
            async with semaphore:
                return await operation()
        
        tasks = [run_with_semaphore(op) for op in operations]
        return await asyncio.gather(*tasks)


class MockHelper:
    """Helper functions for creating mocks."""
    
    @staticmethod
    def create_mock_repository() -> Mock:
        """Create a mock repository with common methods."""
        mock_repo = Mock()
        mock_repo.create = AsyncMock()
        mock_repo.get_by_id = AsyncMock()
        mock_repo.get_all = AsyncMock()
        mock_repo.update = AsyncMock()
        mock_repo.delete = AsyncMock()
        mock_repo.exists = AsyncMock()
        return mock_repo
    
    @staticmethod
    def create_mock_service() -> Mock:
        """Create a mock service with common methods."""
        mock_service = Mock()
        mock_service.execute = AsyncMock()
        mock_service.validate = Mock()
        mock_service.process = AsyncMock()
        return mock_service
    
    @staticmethod
    def create_mock_use_case() -> Mock:
        """Create a mock use case with common methods."""
        mock_use_case = Mock()
        mock_use_case.execute = AsyncMock()
        mock_use_case.validate_request = Mock()
        mock_use_case.handle_errors = Mock()
        return mock_use_case
    
    @staticmethod
    def create_mock_external_service() -> Mock:
        """Create a mock external service with common methods."""
        mock_service = Mock()
        mock_service.call_api = AsyncMock()
        mock_service.validate_response = Mock()
        mock_service.handle_errors = Mock()
        mock_service.rate_limit = Mock()
        return mock_service


class TestScenarioHelper:
    """Helper functions for creating test scenarios."""
    
    @staticmethod
    def create_basic_scenario() -> Dict[str, Any]:
        """Create a basic test scenario."""
        return {
            'campaign': EntityFactory.create_campaign(),
            'ads': [EntityFactory.create_ad() for _ in range(3)],
            'requests': [
                DTOFactory.create_ad_request(),
                DTOFactory.create_campaign_request(),
                DTOFactory.create_optimization_request()
            ]
        }
    
    @staticmethod
    def create_performance_scenario() -> Dict[str, Any]:
        """Create a performance test scenario."""
        return {
            'campaigns': [EntityFactory.create_campaign() for _ in range(10)],
            'ads': [EntityFactory.create_ad() for _ in range(50)],
            'requests': [DTOFactory.create_ad_request() for _ in range(100)],
            'performance_thresholds': {
                'max_response_time': 1.0,
                'min_success_rate': 0.95,
                'max_memory_usage': 100 * 1024 * 1024  # 100MB
            }
        }
    
    @staticmethod
    def create_error_scenario() -> Dict[str, Any]:
        """Create an error test scenario."""
        return {
            'invalid_requests': [
                {'title': '', 'description': None, 'budget': -100},
                {'title': 'A' * 1000, 'description': 'B' * 10000, 'budget': 999999999},
                {'title': 'Test', 'description': 'Test', 'budget': 'invalid'}
            ],
            'expected_errors': [
                'Title cannot be empty',
                'Title too long',
                'Invalid budget format'
            ]
        }
    
    @staticmethod
    def create_integration_scenario() -> Dict[str, Any]:
        """Create an integration test scenario."""
        return {
            'workflow': [
                'create_campaign',
                'create_ads',
                'activate_campaign',
                'optimize_ads',
                'track_performance',
                'generate_report'
            ],
            'dependencies': {
                'campaign': ['ads'],
                'ads': ['performance'],
                'performance': ['analytics']
            }
        }


# Export all helper classes and functions
__all__ = [
    'TestDataGenerator',
    'EntityFactory',
    'DTOFactory',
    'ValidationHelper',
    'PerformanceHelper',
    'AsyncTestHelper',
    'MockHelper',
    'TestScenarioHelper'
]


# Backwards-compatibility helpers expected by conftest
def create_test_data(*args, **kwargs):
    """No-op test data setup helper for compatibility."""
    return True


def cleanup_test_data(*args, **kwargs):
    """No-op test data cleanup helper for compatibility."""
    return True
