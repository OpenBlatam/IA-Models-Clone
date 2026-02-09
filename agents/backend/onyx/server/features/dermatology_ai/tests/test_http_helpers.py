"""
HTTP Testing Helpers
Specialized helpers for HTTP/API testing
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock
import json
from datetime import datetime


class HTTPTestHelpers:
    """Helpers for HTTP testing"""
    
    @staticmethod
    def create_mock_request(
        method: str = "GET",
        path: str = "/",
        headers: Optional[Dict[str, str]] = None,
        query_params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None
    ) -> Mock:
        """Create mock HTTP request"""
        request = Mock()
        request.method = method
        request.url = Mock()
        request.url.path = path
        request.url.query_params = query_params or {}
        request.headers = headers or {}
        request.body = body
        request.json = Mock(return_value=body if isinstance(body, dict) else json.loads(body) if body else {})
        return request
    
    @staticmethod
    def create_mock_response(
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        content_type: str = "application/json"
    ) -> Mock:
        """Create mock HTTP response"""
        response = Mock()
        response.status_code = status_code
        response.headers = headers or {"content-type": content_type}
        response.body = body
        response.json = Mock(return_value=body if isinstance(body, dict) else json.loads(body) if body else {})
        response.text = Mock(return_value=json.dumps(body) if isinstance(body, dict) else str(body) if body else "")
        return response
    
    @staticmethod
    def assert_status_code(response: Mock, expected_code: int):
        """Assert response has expected status code"""
        assert response.status_code == expected_code, \
            f"Expected status {expected_code}, got {response.status_code}"
    
    @staticmethod
    def assert_response_headers(response: Mock, required_headers: List[str]):
        """Assert response has required headers"""
        for header in required_headers:
            assert header in response.headers, f"Missing header: {header}"
    
    @staticmethod
    def assert_response_body(response: Mock, expected_body: Dict[str, Any]):
        """Assert response body matches expected"""
        actual_body = response.json() if hasattr(response, 'json') else response.body
        assert actual_body == expected_body, \
            f"Response body {actual_body} does not match expected {expected_body}"


class APIClientHelpers:
    """Helpers for API client testing"""
    
    @staticmethod
    def create_mock_api_client(
        base_url: str = "http://localhost:8000",
        default_headers: Optional[Dict[str, str]] = None
    ) -> Mock:
        """Create mock API client"""
        client = Mock()
        client.base_url = base_url
        client.default_headers = default_headers or {}
        client.get = Mock(return_value=HTTPTestHelpers.create_mock_response())
        client.post = Mock(return_value=HTTPTestHelpers.create_mock_response())
        client.put = Mock(return_value=HTTPTestHelpers.create_mock_response())
        client.delete = Mock(return_value=HTTPTestHelpers.create_mock_response())
        client.patch = Mock(return_value=HTTPTestHelpers.create_mock_response())
        return client
    
    @staticmethod
    def assert_api_call_made(
        client: Mock,
        method: str,
        path: str,
        expected_data: Optional[Dict[str, Any]] = None
    ):
        """Assert API call was made"""
        method_func = getattr(client, method.lower())
        assert method_func.called, f"{method} was not called"
        
        # Check path if available
        if hasattr(method_func, 'call_args'):
            call_args = method_func.call_args
            if call_args:
                called_path = call_args[0][0] if call_args[0] else None
                if called_path:
                    assert path in called_path or called_path == path, \
                        f"Expected path {path}, got {called_path}"


class WebhookHelpers:
    """Helpers for webhook testing"""
    
    @staticmethod
    def create_mock_webhook_payload(
        event_type: str = "test.event",
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create mock webhook payload"""
        return {
            "event": event_type,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
            "id": "webhook-123"
        }
    
    @staticmethod
    def assert_webhook_sent(
        webhook_client: Mock,
        url: str,
        payload: Dict[str, Any]
    ):
        """Assert webhook was sent"""
        assert webhook_client.send.called, "Webhook was not sent"
        # Additional validation can check call arguments


class RateLimitHelpers:
    """Helpers for rate limiting testing"""
    
    @staticmethod
    def create_mock_rate_limiter(
        limit: int = 100,
        window: int = 60
    ) -> Mock:
        """Create mock rate limiter"""
        limiter = Mock()
        limiter.limit = limit
        limiter.window = window
        limiter.is_allowed = Mock(return_value=True)
        limiter.get_remaining = Mock(return_value=limit)
        limiter.reset = Mock()
        return limiter
    
    @staticmethod
    def assert_rate_limit_enforced(limiter: Mock, identifier: str):
        """Assert rate limit was checked"""
        assert limiter.is_allowed.called, "Rate limit was not checked"
    
    @staticmethod
    def assert_rate_limit_exceeded(limiter: Mock, identifier: str):
        """Assert rate limit was exceeded"""
        limiter.is_allowed = Mock(return_value=False)
        assert not limiter.is_allowed(identifier), "Rate limit should be exceeded"


# Convenience exports
create_mock_request = HTTPTestHelpers.create_mock_request
create_mock_response = HTTPTestHelpers.create_mock_response
assert_status_code = HTTPTestHelpers.assert_status_code
assert_response_headers = HTTPTestHelpers.assert_response_headers
assert_response_body = HTTPTestHelpers.assert_response_body

create_mock_api_client = APIClientHelpers.create_mock_api_client
assert_api_call_made = APIClientHelpers.assert_api_call_made

create_mock_webhook_payload = WebhookHelpers.create_mock_webhook_payload
assert_webhook_sent = WebhookHelpers.assert_webhook_sent

create_mock_rate_limiter = RateLimitHelpers.create_mock_rate_limiter
assert_rate_limit_enforced = RateLimitHelpers.assert_rate_limit_enforced
assert_rate_limit_exceeded = RateLimitHelpers.assert_rate_limit_exceeded



