"""
Extended Base Test Classes
Additional base classes for specialized testing scenarios
"""

import pytest
from typing import Any, Dict, Optional
from unittest.mock import Mock, AsyncMock
from abc import ABC


class BaseValidatorTest(ABC):
    """Base class for validator tests"""
    
    def assert_validation_passes(self, validator, data: Dict[str, Any]):
        """Assert validation passes"""
        result = validator.validate(data)
        assert result is True or result.get("valid") is True
    
    def assert_validation_fails(self, validator, data: Dict[str, Any], expected_errors: Optional[List[str]] = None):
        """Assert validation fails"""
        result = validator.validate(data)
        assert result is False or result.get("valid") is False
        if expected_errors:
            errors = result.get("errors", []) if isinstance(result, dict) else []
            for expected_error in expected_errors:
                assert any(expected_error in str(error) for error in errors), \
                    f"Expected error '{expected_error}' not found in {errors}"


class BaseMapperTest(ABC):
    """Base class for mapper tests"""
    
    def assert_mapping_correct(self, mapper, entity, expected_dict: Dict[str, Any]):
        """Assert entity maps to expected dictionary"""
        result = mapper.to_dict(entity)
        for key, value in expected_dict.items():
            assert key in result
            assert result[key] == value
    
    def assert_reverse_mapping_correct(self, mapper, data: Dict[str, Any], expected_entity_type: type):
        """Assert dictionary maps to expected entity type"""
        result = mapper.to_entity(data)
        assert isinstance(result, expected_entity_type)


class BaseEventTest(ABC):
    """Base class for event tests"""
    
    def assert_event_published(self, event_publisher, event_type: str, event_data: Dict[str, Any]):
        """Assert event was published"""
        # Check if publish was called with expected event
        assert event_publisher.publish.called
        # Additional checks can be added based on implementation
    
    def assert_event_not_published(self, event_publisher):
        """Assert event was not published"""
        assert not event_publisher.publish.called


class BaseCacheTest(ABC):
    """Base class for cache tests"""
    
    def assert_cache_hit(self, cache_service, key: str, expected_value: Any):
        """Assert cache hit with expected value"""
        result = cache_service.get(key)
        assert result == expected_value
    
    def assert_cache_miss(self, cache_service, key: str):
        """Assert cache miss"""
        result = cache_service.get(key)
        assert result is None
    
    def assert_cache_set(self, cache_service, key: str, value: Any):
        """Assert value was set in cache"""
        cache_service.set.assert_called_with(key, value)


class BaseRepositoryTestExtended(ABC):
    """Extended base class for repository tests with more utilities"""
    
    def assert_repository_create(self, repository, entity, expected_id: Optional[str] = None):
        """Assert entity was created in repository"""
        result = repository.create(entity)
        assert result is not None
        if expected_id:
            assert result.id == expected_id
    
    def assert_repository_get(self, repository, entity_id: str, expected_entity: Any):
        """Assert entity was retrieved from repository"""
        result = repository.get_by_id(entity_id)
        assert result == expected_entity
    
    def assert_repository_update(self, repository, entity):
        """Assert entity was updated in repository"""
        result = repository.update(entity)
        assert result is not None
    
    def assert_repository_delete(self, repository, entity_id: str):
        """Assert entity was deleted from repository"""
        result = repository.delete(entity_id)
        assert result is True


class BaseServiceTestExtended(ABC):
    """Extended base class for service tests"""
    
    def assert_service_method_called(self, service, method_name: str, *args, **kwargs):
        """Assert service method was called"""
        method = getattr(service, method_name)
        if args or kwargs:
            method.assert_called_once_with(*args, **kwargs)
        else:
            method.assert_called_once()
    
    def assert_service_result_type(self, result: Any, expected_type: type):
        """Assert service result is of expected type"""
        assert isinstance(result, expected_type), \
            f"Result is {type(result)}, expected {expected_type}"
    
    def assert_service_result_contains(self, result: Dict[str, Any], required_keys: List[str]):
        """Assert service result contains required keys"""
        for key in required_keys:
            assert key in result, f"Missing key: {key}"


class BaseAPITestExtended(ABC):
    """Extended base class for API tests with more utilities"""
    
    def assert_response_contains(self, response, keys: List[str]):
        """Assert response contains required keys"""
        data = response.json()
        for key in keys:
            assert key in data, f"Response missing key: {key}"
    
    def assert_response_value(self, response, key: str, expected_value: Any):
        """Assert response has expected value for key"""
        data = response.json()
        assert key in data
        assert data[key] == expected_value
    
    def assert_response_type(self, response, key: str, expected_type: type):
        """Assert response value is of expected type"""
        data = response.json()
        assert key in data
        assert isinstance(data[key], expected_type), \
            f"Value for '{key}' is {type(data[key])}, expected {expected_type}"
    
    def assert_error_response(self, response, expected_status: int = 400, expected_message: Optional[str] = None):
        """Assert error response structure"""
        self.assert_status_code(response, expected_status)
        data = response.json()
        assert "error" in data or "message" in data
        if expected_message:
            error_text = data.get("error") or data.get("message", "")
            assert expected_message.lower() in str(error_text).lower()



