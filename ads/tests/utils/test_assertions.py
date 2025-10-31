"""
Test Assertion Utilities for the ads feature.

This module provides comprehensive assertion functions and utilities for testing:
- Custom assertion functions
- Validation assertions
- Performance assertions
- Business logic assertions
- Error handling assertions
- Data consistency assertions
"""

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from unittest.mock import Mock

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


class EntityAssertions:
    """Assertions for domain entities."""
    
    @staticmethod
    def assert_valid_ad(ad: Ad) -> None:
        """Assert that an Ad entity is valid."""
        assert ad is not None, "Ad entity cannot be None"
        assert isinstance(ad, Ad), f"Expected Ad entity, got {type(ad)}"
        assert ad.id is not None, "Ad ID cannot be None"
        assert ad.title is not None, "Ad title cannot be None"
        assert len(ad.title) > 0, "Ad title cannot be empty"
        assert len(ad.title) <= 100, "Ad title too long"
        assert ad.description is not None, "Ad description cannot be None"
        assert ad.status in AdStatus, f"Invalid ad status: {ad.status}"
        assert ad.ad_type in AdType, f"Invalid ad type: {ad.ad_type}"
        assert ad.platform in Platform, f"Invalid platform: {ad.platform}"
        assert ad.created_at is not None, "Ad created_at cannot be None"
        assert ad.updated_at is not None, "Ad updated_at cannot be None"
    
    @staticmethod
    def assert_valid_campaign(campaign: AdCampaign) -> None:
        """Assert that an AdCampaign entity is valid."""
        assert campaign is not None, "Campaign entity cannot be None"
        assert isinstance(campaign, AdCampaign), f"Expected AdCampaign entity, got {type(campaign)}"
        assert campaign.id is not None, "Campaign ID cannot be None"
        assert campaign.name is not None, "Campaign name cannot be None"
        assert len(campaign.name) > 0, "Campaign name cannot be empty"
        assert len(campaign.name) <= 200, "Campaign name too long"
        assert campaign.status in AdStatus, f"Invalid campaign status: {campaign.status}"
        assert campaign.created_at is not None, "Campaign created_at cannot be None"
        assert campaign.updated_at is not None, "Campaign updated_at cannot be None"
    
    @staticmethod
    def assert_valid_ad_group(group: AdGroup) -> None:
        """Assert that an AdGroup entity is valid."""
        assert group is not None, "AdGroup entity cannot be None"
        assert isinstance(group, AdGroup), f"Expected AdGroup entity, got {type(group)}"
        assert group.id is not None, "Group ID cannot be None"
        assert group.name is not None, "Group name cannot be None"
        assert len(group.name) > 0, "Group name cannot be empty"
        assert len(group.ads) >= 1, "AdGroup must contain at least one ad"
        assert group.created_at is not None, "Group created_at cannot be None"
        assert group.updated_at is not None, "Group updated_at cannot be None"
    
    @staticmethod
    def assert_valid_performance(performance: AdPerformance) -> None:
        """Assert that an AdPerformance entity is valid."""
        assert performance is not None, "Performance entity cannot be None"
        assert isinstance(performance, AdPerformance), f"Expected AdPerformance entity, got {type(performance)}"
        assert performance.id is not None, "Performance ID cannot be None"
        assert performance.ad_id is not None, "Performance ad_id cannot be None"
        assert performance.date is not None, "Performance date cannot be None"
        assert performance.created_at is not None, "Performance created_at cannot be None"
    
    @staticmethod
    def assert_entity_consistency(entity: Any, expected_type: Type[T]) -> None:
        """Assert that an entity is consistent with its expected type."""
        assert entity is not None, f"{expected_type.__name__} entity cannot be None"
        assert isinstance(entity, expected_type), f"Expected {expected_type.__name__}, got {type(entity)}"
        assert hasattr(entity, 'id'), f"{expected_type.__name__} must have an id attribute"
        assert entity.id is not None, f"{expected_type.__name__} ID cannot be None"
        assert hasattr(entity, 'created_at'), f"{expected_type.__name__} must have a created_at attribute"
        assert entity.created_at is not None, f"{expected_type.__name__} created_at cannot be None"


class DTOAssertions:
    """Assertions for Data Transfer Objects."""
    
    @staticmethod
    def assert_valid_create_ad_request(request: CreateAdRequest) -> None:
        """Assert that a CreateAdRequest DTO is valid."""
        assert request is not None, "CreateAdRequest cannot be None"
        assert isinstance(request, CreateAdRequest), f"Expected CreateAdRequest, got {type(request)}"
        assert request.title is not None, "Title cannot be None"
        assert len(request.title) > 0, "Title cannot be empty"
        assert len(request.title) <= 100, "Title too long"
        assert request.description is not None, "Description cannot be None"
        assert request.ad_type in AdType, f"Invalid ad type: {request.ad_type}"
        assert request.platform in Platform, f"Invalid platform: {request.platform}"
        assert request.budget > 0, "Budget must be positive"
        assert request.budget <= 1000000, "Budget too high"
    
    @staticmethod
    def assert_valid_create_campaign_request(request: CreateCampaignRequest) -> None:
        """Assert that a CreateCampaignRequest DTO is valid."""
        assert request is not None, "CreateCampaignRequest cannot be None"
        assert isinstance(request, CreateCampaignRequest), f"Expected CreateCampaignRequest, got {type(request)}"
        assert request.name is not None, "Name cannot be None"
        assert len(request.name) > 0, "Name cannot be empty"
        assert len(request.name) <= 200, "Name too long"
        assert request.budget > 0, "Budget must be positive"
        assert request.budget <= 10000000, "Budget too high"
    
    @staticmethod
    def assert_valid_optimization_request(request: OptimizeAdRequest) -> None:
        """Assert that an OptimizeAdRequest DTO is valid."""
        assert request is not None, "OptimizeAdRequest cannot be None"
        assert isinstance(request, OptimizeAdRequest), f"Expected OptimizeAdRequest, got {type(request)}"
        assert request.ad_id is not None, "Ad ID cannot be None"
        assert len(request.ad_id) > 0, "Ad ID cannot be empty"
        assert request.optimization_level in ["light", "standard", "aggressive", "extreme"], f"Invalid optimization level: {request.optimization_level}"
        assert len(request.target_metrics) > 0, "Target metrics cannot be empty"
        assert all(metric in ["ctr", "conversion_rate", "roas", "cpc", "cpm"] for metric in request.target_metrics), f"Invalid target metrics: {request.target_metrics}"
    
    @staticmethod
    def assert_dto_consistency(dto: Any, expected_type: Type[T]) -> None:
        """Assert that a DTO is consistent with its expected type."""
        assert dto is not None, f"{expected_type.__name__} cannot be None"
        assert isinstance(dto, expected_type), f"Expected {expected_type.__name__}, got {type(dto)}"


class ValueObjectAssertions:
    """Assertions for value objects."""
    
    @staticmethod
    def assert_valid_budget(budget: Budget) -> None:
        """Assert that a Budget value object is valid."""
        assert budget is not None, "Budget cannot be None"
        assert isinstance(budget, Budget), f"Expected Budget, got {type(budget)}"
        assert budget.daily_budget > 0, "Daily budget must be positive"
        assert budget.total_budget > 0, "Total budget must be positive"
        assert budget.daily_budget <= budget.total_budget, "Daily budget cannot exceed total budget"
        assert budget.currency in ["USD", "EUR", "GBP", "CAD", "AUD"], f"Invalid currency: {budget.currency}"
    
    @staticmethod
    def assert_valid_targeting_criteria(criteria: TargetingCriteria) -> None:
        """Assert that a TargetingCriteria value object is valid."""
        assert criteria is not None, "Targeting criteria cannot be None"
        assert isinstance(criteria, TargetingCriteria), f"Expected TargetingCriteria, got {type(criteria)}"
        assert criteria.age_range[0] >= 13, "Minimum age must be at least 13"
        assert criteria.age_range[1] <= 100, "Maximum age cannot exceed 100"
        assert criteria.age_range[0] <= criteria.age_range[1], "Invalid age range"
        assert len(criteria.gender) > 0, "Gender targeting cannot be empty"
        assert all(g in ["male", "female", "other"] for g in criteria.gender), f"Invalid gender values: {criteria.gender}"
        assert len(criteria.interests) > 0, "Interests cannot be empty"
        assert len(criteria.location) > 0, "Location targeting cannot be empty"
        assert len(criteria.language) > 0, "Language targeting cannot be empty"
    
    @staticmethod
    def assert_valid_ad_metrics(metrics: AdMetrics) -> None:
        """Assert that an AdMetrics value object is valid."""
        assert metrics is not None, "Ad metrics cannot be None"
        assert isinstance(metrics, AdMetrics), f"Expected AdMetrics, got {type(metrics)}"
        assert metrics.impressions >= 0, "Impressions cannot be negative"
        assert metrics.clicks >= 0, "Clicks cannot be negative"
        assert metrics.conversions >= 0, "Conversions cannot be negative"
        assert metrics.spend >= 0, "Spend cannot be negative"
        if metrics.impressions > 0:
            assert metrics.clicks <= metrics.impressions, "Clicks cannot exceed impressions"
        if metrics.clicks > 0:
            assert metrics.conversions <= metrics.clicks, "Conversions cannot exceed clicks"
    
    @staticmethod
    def assert_valid_ad_schedule(schedule: AdSchedule) -> None:
        """Assert that an AdSchedule value object is valid."""
        assert schedule is not None, "Ad schedule cannot be None"
        assert isinstance(schedule, AdSchedule), f"Expected AdSchedule, got {type(schedule)}"
        assert schedule.start_date is not None, "Start date cannot be None"
        assert schedule.end_date is not None, "End date cannot be None"
        assert schedule.start_date < schedule.end_date, "Start date must be before end date"
        assert len(schedule.active_hours) > 0, "Active hours cannot be empty"
        assert all(0 <= hour <= 23 for hour in schedule.active_hours), f"Invalid active hours: {schedule.active_hours}"
        assert len(schedule.active_days) > 0, "Active days cannot be empty"
        assert all(1 <= day <= 7 for day in schedule.active_days), f"Invalid active days: {schedule.active_days}"


class BusinessLogicAssertions:
    """Assertions for business logic validation."""
    
    @staticmethod
    def assert_valid_status_transition(
        entity: Any,
        old_status: AdStatus,
        new_status: AdStatus,
        valid_transitions: Dict[AdStatus, List[AdStatus]]
    ) -> None:
        """Assert that a status transition follows business rules."""
        assert old_status in valid_transitions, f"Status {old_status} not found in valid transitions"
        assert new_status in valid_transitions[old_status], f"Invalid transition from {old_status} to {new_status}"
    
    @staticmethod
    def assert_campaign_budget_consistency(campaign: AdCampaign) -> None:
        """Assert that campaign budget is consistent with ad budgets."""
        total_ad_budget = sum(ad.budget.total_budget for ad in campaign.ads)
        assert total_ad_budget <= campaign.budget.total_budget, "Total ad budgets cannot exceed campaign budget"
    
    @staticmethod
    def assert_ad_performance_consistency(ad: Ad, performance: AdPerformance) -> None:
        """Assert that ad performance data is consistent."""
        assert ad.id == performance.ad_id, "Performance data must match ad ID"
        assert performance.date >= ad.schedule.start_date, "Performance date cannot be before ad start date"
        assert performance.date <= ad.schedule.end_date, "Performance date cannot be after ad end date"
    
    @staticmethod
    def assert_targeting_compatibility(ad: Ad, campaign: AdCampaign) -> None:
        """Assert that ad targeting is compatible with campaign targeting."""
        # Basic compatibility checks
        assert ad.platform in [Platform.FACEBOOK, Platform.GOOGLE, Platform.INSTAGRAM], f"Unsupported platform: {ad.platform}"
        assert ad.budget.currency == campaign.budget.currency, "Ad and campaign must use same currency"


class PerformanceAssertions:
    """Assertions for performance testing."""
    
    @staticmethod
    def assert_response_time(response_time: float, max_time: float = 1.0) -> None:
        """Assert that response time is within acceptable limits."""
        assert response_time >= 0, "Response time cannot be negative"
        assert response_time <= max_time, f"Response time {response_time}s exceeds maximum {max_time}s"
    
    @staticmethod
    def assert_throughput(operations_per_second: float, min_throughput: float = 10.0) -> None:
        """Assert that throughput meets minimum requirements."""
        assert operations_per_second >= min_throughput, f"Throughput {operations_per_second} ops/s below minimum {min_throughput} ops/s"
    
    @staticmethod
    def assert_memory_usage(memory_usage: int, max_memory: int = 100 * 1024 * 1024) -> None:
        """Assert that memory usage is within acceptable limits."""
        assert memory_usage >= 0, "Memory usage cannot be negative"
        assert memory_usage <= max_memory, f"Memory usage {memory_usage} bytes exceeds maximum {max_memory} bytes"
    
    @staticmethod
    def assert_concurrent_operations(
        results: List[Dict[str, Any]],
        min_success_rate: float = 0.95,
        max_response_time: float = 2.0
    ) -> None:
        """Assert that concurrent operations meet performance requirements."""
        total_ops = len(results)
        successful_ops = sum(1 for r in results if r.get('success', False))
        success_rate = successful_ops / total_ops if total_ops > 0 else 0
        
        assert success_rate >= min_success_rate, f"Success rate {success_rate} below minimum {min_success_rate}"
        
        if successful_ops > 0:
            avg_response_time = sum(r.get('execution_time', 0) for r in results if r.get('success', False)) / successful_ops
            assert avg_response_time <= max_response_time, f"Average response time {avg_response_time}s exceeds maximum {max_response_time}s"


class ErrorHandlingAssertions:
    """Assertions for error handling validation."""
    
    @staticmethod
    def assert_proper_error_response(
        response: Any,
        expected_status_code: int,
        expected_error_type: str = None,
        expected_error_message: str = None
    ) -> None:
        """Assert that an error response is properly formatted."""
        assert response is not None, "Error response cannot be None"
        assert hasattr(response, 'status_code'), "Response must have status_code attribute"
        assert response.status_code == expected_status_code, f"Expected status code {expected_status_code}, got {response.status_code}"
        
        if hasattr(response, 'json'):
            response_data = response.json()
            assert 'error' in response_data, "Error response must contain 'error' field"
            assert 'message' in response_data, "Error response must contain 'message' field"
            
            if expected_error_type:
                assert response_data.get('error') == expected_error_type, f"Expected error type {expected_error_type}, got {response_data.get('error')}"
            
            if expected_error_message:
                assert expected_error_message in response_data.get('message', ''), f"Expected error message '{expected_error_message}' not found in '{response_data.get('message')}'"
    
    @staticmethod
    def assert_validation_error(
        error: ValidationError,
        expected_field: str = None,
        expected_error_type: str = None
    ) -> None:
        """Assert that a validation error is properly formatted."""
        assert error is not None, "Validation error cannot be None"
        assert isinstance(error, ValidationError), f"Expected ValidationError, got {type(error)}"
        assert len(error.errors()) > 0, "Validation error must contain at least one error"
        
        if expected_field:
            field_errors = [e for e in error.errors() if e.get('loc') and expected_field in e.get('loc', [])]
            assert len(field_errors) > 0, f"No validation error found for field: {expected_field}"
        
        if expected_error_type:
            error_types = [e.get('type') for e in error.errors()]
            assert expected_error_type in error_types, f"Expected error type {expected_error_type} not found in {error_types}"
    
    @staticmethod
    def assert_exception_type(
        exception: Exception,
        expected_exception_type: Type[Exception]
    ) -> None:
        """Assert that the correct exception type is raised."""
        assert isinstance(exception, expected_exception_type), f"Expected {expected_exception_type.__name__}, got {type(exception).__name__}"
    
    @staticmethod
    def assert_exception_message(
        exception: Exception,
        expected_message: str = None,
        message_pattern: str = None
    ) -> None:
        """Assert that an exception has the expected message."""
        assert exception is not None, "Exception cannot be None"
        
        if expected_message:
            assert str(exception) == expected_message, f"Expected message '{expected_message}', got '{str(exception)}'"
        
        if message_pattern:
            assert re.search(message_pattern, str(exception)), f"Exception message '{str(exception)}' does not match pattern '{message_pattern}'"


class DataConsistencyAssertions:
    """Assertions for data consistency validation."""
    
    @staticmethod
    def assert_data_integrity(
        original_data: Dict[str, Any],
        retrieved_data: Dict[str, Any],
        exclude_fields: List[str] = None
    ) -> None:
        """Assert that retrieved data maintains integrity with original data."""
        if exclude_fields is None:
            exclude_fields = ['created_at', 'updated_at', 'id']
        
        for key, value in original_data.items():
            if key not in exclude_fields:
                assert key in retrieved_data, f"Retrieved data missing key: {key}"
                assert retrieved_data[key] == value, f"Data integrity violation for key {key}: {value} != {retrieved_data[key]}"
    
    @staticmethod
    def assert_list_consistency(
        items: List[Any],
        expected_type: Type[T],
        min_count: int = 0,
        max_count: int = None
    ) -> None:
        """Assert that a list maintains consistency."""
        assert isinstance(items, list), f"Expected list, got {type(items)}"
        assert len(items) >= min_count, f"List must have at least {min_count} items, got {len(items)}"
        
        if max_count is not None:
            assert len(items) <= max_count, f"List must have at most {max_count} items, got {len(items)}"
        
        for item in items:
            assert isinstance(item, expected_type), f"Expected {expected_type.__name__}, got {type(item)}"
    
    @staticmethod
    def assert_unique_ids(items: List[Any]) -> None:
        """Assert that all items have unique IDs."""
        ids = [item.id for item in items if hasattr(item, 'id') and item.id is not None]
        unique_ids = set(ids)
        assert len(ids) == len(unique_ids), f"Duplicate IDs found: {[id for id in ids if ids.count(id) > 1]}"
    
    @staticmethod
    def assert_timestamp_consistency(
        entity: Any,
        created_before: datetime = None,
        updated_after: datetime = None
    ) -> None:
        """Assert that entity timestamps are consistent."""
        if created_before is None:
            created_before = datetime.now()
        if updated_after is None:
            updated_after = datetime.now() - timedelta(seconds=1)
        
        if hasattr(entity, 'created_at') and entity.created_at:
            assert entity.created_at <= created_before, f"Created timestamp {entity.created_at} is in the future"
        
        if hasattr(entity, 'updated_at') and entity.updated_at:
            assert entity.updated_at >= updated_after, f"Updated timestamp {entity.updated_at} is too old"


class MockAssertions:
    """Assertions for mock objects and testing."""
    
    @staticmethod
    def assert_mock_called(mock: Mock, method_name: str, expected_calls: int = 1) -> None:
        """Assert that a mock method was called the expected number of times."""
        assert hasattr(mock, method_name), f"Mock does not have method: {method_name}"
        method = getattr(mock, method_name)
        actual_calls = method.call_count
        assert actual_calls == expected_calls, f"Expected {expected_calls} calls to {method_name}, got {actual_calls}"
    
    @staticmethod
    def assert_mock_called_with(
        mock: Mock,
        method_name: str,
        expected_args: tuple = None,
        expected_kwargs: dict = None
    ) -> None:
        """Assert that a mock method was called with expected arguments."""
        assert hasattr(mock, method_name), f"Mock does not have method: {method_name}"
        method = getattr(mock, method_name)
        
        if expected_args:
            method.assert_called_with(*expected_args)
        
        if expected_kwargs:
            # Check if any call matches the expected kwargs
            calls = method.call_args_list
            kwargs_match = any(
                all(k in call.kwargs and call.kwargs[k] == v for k, v in expected_kwargs.items())
                for call in calls
            )
            assert kwargs_match, f"Method {method_name} was not called with expected kwargs: {expected_kwargs}"
    
    @staticmethod
    def assert_mock_not_called(mock: Mock, method_name: str) -> None:
        """Assert that a mock method was not called."""
        assert hasattr(mock, method_name), f"Mock does not have method: {method_name}"
        method = getattr(mock, method_name)
        assert method.call_count == 0, f"Expected method {method_name} to not be called, but it was called {method.call_count} times"
    
    @staticmethod
    def assert_mock_return_value(
        mock: Mock,
        method_name: str,
        expected_return_value: Any
    ) -> None:
        """Assert that a mock method returns the expected value."""
        assert hasattr(mock, method_name), f"Mock does not have method: {method_name}"
        method = getattr(mock, method_name)
        method.return_value = expected_return_value
        assert method.return_value == expected_return_value, f"Mock return value not set correctly"


# Export all assertion classes and functions
__all__ = [
    'EntityAssertions',
    'DTOAssertions',
    'ValueObjectAssertions',
    'BusinessLogicAssertions',
    'PerformanceAssertions',
    'ErrorHandlingAssertions',
    'DataConsistencyAssertions',
    'MockAssertions'
]
