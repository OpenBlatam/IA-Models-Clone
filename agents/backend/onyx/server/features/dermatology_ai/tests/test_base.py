"""
Base Test Classes
Base classes for common test patterns
"""

import pytest
from unittest.mock import Mock, AsyncMock
from typing import Any, Dict, Optional
from abc import ABC


class BaseTest(ABC):
    """Base class for all tests with common utilities"""
    
    @staticmethod
    def create_mock(return_value: Any = None, side_effect: Any = None) -> Mock:
        """Create a mock with return value or side effect"""
        mock = Mock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    @staticmethod
    def create_async_mock(return_value: Any = None, side_effect: Any = None) -> AsyncMock:
        """Create an async mock with return value or side effect"""
        mock = AsyncMock()
        if return_value is not None:
            mock.return_value = return_value
        if side_effect is not None:
            mock.side_effect = side_effect
        return mock
    
    def assert_dict_contains(self, dict_obj: Dict[str, Any], required_keys: list[str]):
        """Assert that dict contains required keys"""
        for key in required_keys:
            assert key in dict_obj, f"Missing required key: {key}"
    
    def assert_response_success(self, response_data: Dict[str, Any]):
        """Assert that API response indicates success"""
        assert "success" in response_data or response_data.get("status") == "success"
    
    def assert_response_error(self, response_data: Dict[str, Any], expected_status: Optional[int] = None):
        """Assert that API response indicates error"""
        assert "error" in response_data or "message" in response_data
        if expected_status:
            assert response_data.get("status_code") == expected_status or response_data.get("status") == expected_status


class BaseAPITest(BaseTest):
    """Base class for API endpoint tests"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from main import create_application
        app = create_application()
        return TestClient(app)
    
    def assert_status_code(self, response, expected_code: int):
        """Assert response status code"""
        assert response.status_code == expected_code, \
            f"Expected {expected_code}, got {response.status_code}: {response.text}"
    
    def assert_json_response(self, response, expected_keys: Optional[list[str]] = None):
        """Assert JSON response structure"""
        assert response.headers.get("content-type") == "application/json"
        data = response.json()
        if expected_keys:
            self.assert_dict_contains(data, expected_keys)
        return data


class BaseRepositoryTest(BaseTest):
    """Base class for repository tests"""
    
    def assert_repository_method_called(self, mock_repo, method_name: str, *args, **kwargs):
        """Assert repository method was called with specific arguments"""
        method = getattr(mock_repo, method_name)
        if args or kwargs:
            method.assert_called_once_with(*args, **kwargs)
        else:
            method.assert_called_once()


class BaseServiceTest(BaseTest):
    """Base class for service tests"""
    
    def assert_service_result_valid(self, result: Any, expected_type: type = None):
        """Assert service result is valid"""
        assert result is not None
        if expected_type:
            assert isinstance(result, expected_type)


class BaseUseCaseTest(BaseTest):
    """Base class for use case tests"""
    
    def assert_use_case_success(self, result: Any):
        """Assert use case executed successfully"""
        assert result is not None
        # Add more specific assertions based on use case return type


class BaseIntegrationTest(BaseTest):
    """Base class for integration tests"""
    
    @pytest.fixture(autouse=True)
    def setup_integration(self):
        """Setup for integration tests"""
        # Common setup logic
        yield
        # Common teardown logic
    
    def assert_integration_flow_complete(self, flow_result: Dict[str, Any]):
        """Assert integration flow completed successfully"""
        assert "success" in flow_result or flow_result.get("status") == "completed"


class BaseMiddlewareTest(BaseTest):
    """Base class for middleware tests"""
    
    def create_mock_request(self, method: str = "GET", path: str = "/", headers: Optional[Dict] = None):
        """Create mock request for middleware testing"""
        request = Mock()
        request.method = method
        request.url.path = path
        request.headers = headers or {}
        return request
    
    def create_mock_response(self, status_code: int = 200, content: Any = None):
        """Create mock response"""
        response = Mock()
        response.status_code = status_code
        response.content = content
        return response


class BaseDecoratorTest(BaseTest):
    """Base class for decorator tests"""
    
    def create_test_function(self, return_value: Any = "result", should_fail: bool = False):
        """Create test function for decorator testing"""
        call_count = 0
        
        async def test_func(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if should_fail:
                raise Exception("Function failed")
            return return_value
        
        test_func.call_count = lambda: call_count
        return test_func



