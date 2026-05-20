"""
Playwright Mixins
=================
Mixins for adding functionality to test classes.
"""

from playwright.sync_api import Page, Response
from typing import Dict, Any, Optional, List
import time
from playwright_utils import create_request_builder, validate_response, create_test_data
from playwright_debug import create_debugger
from playwright_analytics import create_analytics


class RequestMixin:
    """Mixin for making HTTP requests."""
    
    def make_request(
        self,
        page: Page,
        method: str,
        endpoint: str,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Response:
        """Make HTTP request using request builder."""
        builder = create_request_builder(page, base_url)
        
        if method.upper() == "GET":
            builder = builder.get(endpoint)
        elif method.upper() == "POST":
            builder = builder.post(endpoint)
        elif method.upper() == "PUT":
            builder = builder.put(endpoint)
        elif method.upper() == "DELETE":
            builder = builder.delete(endpoint)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if headers:
            builder = builder.with_headers(headers)
        
        if "json" in kwargs:
            builder = builder.with_json(kwargs.pop("json"))
        if "multipart" in kwargs:
            builder = builder.with_multipart(kwargs.pop("multipart"))
        if "params" in kwargs:
            builder = builder.with_query(kwargs.pop("params"))
        
        return builder.execute()
    
    def make_request_simple(
        self,
        page: Page,
        method: str,
        endpoint: str,
        base_url: str,
        headers: Optional[Dict[str, str]] = None
    ) -> Response:
        """Make simple HTTP request."""
        url = f"{base_url}{endpoint}"
        request_headers = headers or {}
        
        if method.upper() == "GET":
            return page.request.get(url, headers=request_headers)
        elif method.upper() == "POST":
            return page.request.post(url, headers=request_headers)
        elif method.upper() == "PUT":
            return page.request.put(url, headers=request_headers)
        elif method.upper() == "DELETE":
            return page.request.delete(url, headers=request_headers)
        else:
            raise ValueError(f"Unsupported method: {method}")


class AssertionMixin:
    """Mixin for assertions."""
    
    def assert_success(self, response: Response, expected_status: int = 200):
        """Assert response is successful."""
        assert response.status == expected_status, \
            f"Expected {expected_status}, got {response.status}: {response.text()[:200]}"
    
    def assert_json_response(
        self,
        response: Response,
        expected_keys: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Assert response is JSON and optionally check keys."""
        validated = validate_response(response)
        data = validated.assert_status_range(200, 299).assert_json()
        
        if expected_keys and isinstance(data, dict):
            for key in expected_keys:
                assert key in data, f"Missing key: {key}"
        
        return data
    
    def assert_response_time(
        self,
        response_time: float,
        max_time: float = 1.0
    ):
        """Assert response time is acceptable."""
        assert response_time < max_time, \
            f"Response time {response_time:.3f}s exceeds maximum {max_time}s"


class APIOperationsMixin:
    """Mixin for API operations."""
    
    def upload_pdf(
        self,
        page: Page,
        base_url: str,
        filename: str,
        content: bytes,
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Upload PDF file."""
        factory = create_test_data()
        files = factory.create_upload_files(filename, content)
        
        response = (
            create_request_builder(page, base_url)
            .post("/pdf/upload")
            .with_headers(headers)
            .with_multipart(files)
            .execute()
        )
        
        return validate_response(response).assert_status_range(200, 201).assert_json()
    
    def generate_variant(
        self,
        page: Page,
        base_url: str,
        file_id: str,
        variant_type: str,
        headers: Dict[str, str],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate variant."""
        factory = create_test_data()
        variant_request = factory.create_variant_request(variant_type, options)
        
        response = (
            create_request_builder(page, base_url)
            .post(f"/pdf/{file_id}/variants")
            .with_headers(headers)
            .with_json(variant_request)
            .execute()
        )
        
        return validate_response(response).assert_status_range(200, 202).assert_json()
    
    def extract_topics(
        self,
        page: Page,
        base_url: str,
        file_id: str,
        headers: Dict[str, str],
        min_relevance: float = 0.5,
        max_topics: int = 10
    ) -> Dict[str, Any]:
        """Extract topics."""
        response = (
            create_request_builder(page, base_url)
            .get(f"/pdf/{file_id}/topics")
            .with_headers(headers)
            .with_query({"min_relevance": min_relevance, "max_topics": max_topics})
            .execute()
        )
        
        return validate_response(response).assert_status_range(200, 202).assert_json()
    
    def get_preview(
        self,
        page: Page,
        base_url: str,
        file_id: str,
        headers: Dict[str, str],
        page_number: int = 1
    ) -> Dict[str, Any]:
        """Get preview."""
        response = (
            create_request_builder(page, base_url)
            .get(f"/pdf/{file_id}/preview")
            .with_headers(headers)
            .with_query({"page_number": page_number})
            .execute()
        )
        
        return validate_response(response).assert_status_range(200, 202).assert_json()
    
    def get_file_id_from_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract file_id from response."""
        return response_data.get("file_id") or response_data.get("id")


class PerformanceMixin:
    """Mixin for performance testing."""
    
    def measure_response_time(
        self,
        page: Page,
        base_url: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        iterations: int = 10
    ) -> Dict[str, float]:
        """Measure response time."""
        times = []
        
        for _ in range(iterations):
            start = time.time()
            response = page.request.get(f"{base_url}{endpoint}", headers=headers or {})
            elapsed = time.time() - start
            times.append(elapsed)
            assert response.status == 200
        
        sorted_times = sorted(times)
        return {
            "min": min(times),
            "max": max(times),
            "avg": sum(times) / len(times),
            "median": sorted_times[len(times) // 2],
            "p95": sorted_times[int(len(times) * 0.95)] if len(times) > 0 else 0,
            "p99": sorted_times[int(len(times) * 0.99)] if len(times) > 0 else 0
        }
    
    def assert_performance_threshold(
        self,
        metrics: Dict[str, float],
        max_avg: float = 1.0,
        max_p95: float = 2.0,
        max_p99: float = 3.0
    ):
        """Assert performance metrics meet thresholds."""
        assert metrics["avg"] < max_avg, \
            f"Average time {metrics['avg']:.3f}s exceeds {max_avg}s"
        assert metrics["p95"] < max_p95, \
            f"P95 time {metrics['p95']:.3f}s exceeds {max_p95}s"
        assert metrics["p99"] < max_p99, \
            f"P99 time {metrics['p99']:.3f}s exceeds {max_p99}s"


class DebuggingMixin:
    """Mixin for debugging functionality."""
    
    def setup_debugger(self, page: Page, output_dir: str = "debug_output"):
        """Setup debugger for test."""
        return create_debugger(page, output_dir)
    
    def capture_debug_info(
        self,
        page: Page,
        test_name: str,
        output_dir: str = "debug_output"
    ) -> str:
        """Capture debug information."""
        debugger = create_debugger(page, output_dir)
        debug_file = debugger.save_debug_info(test_name)
        return str(debug_file)


class AnalyticsMixin:
    """Mixin for analytics."""
    
    def setup_analytics(self, output_dir: str = "analytics"):
        """Setup analytics for test."""
        return create_analytics(output_dir)
    
    def record_test_metrics(
        self,
        analytics,
        test_name: str,
        duration: float,
        status: str,
        requests: List[Dict[str, Any]],
        memory_usage: Optional[float] = None,
        cpu_usage: Optional[float] = None
    ):
        """Record test metrics."""
        return analytics.record_test_metrics(
            test_name, duration, status, requests, memory_usage, cpu_usage
        )


class WorkflowMixin(RequestMixin, APIOperationsMixin, AssertionMixin):
    """Mixin combining common workflow operations."""
    
    def complete_workflow(
        self,
        page: Page,
        base_url: str,
        filename: str,
        content: bytes,
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Complete workflow: upload -> variant -> topics -> preview."""
        # Upload
        upload_result = self.upload_pdf(page, base_url, filename, content, headers)
        file_id = self.get_file_id_from_response(upload_result)
        
        if not file_id:
            raise ValueError("Failed to get file_id from upload")
        
        # Generate variant
        variant_result = self.generate_variant(
            page, base_url, file_id, "summary", headers
        )
        
        # Extract topics
        topics_result = self.extract_topics(page, base_url, file_id, headers)
        
        # Get preview
        preview_result = self.get_preview(page, base_url, file_id, headers)
        
        return {
            "file_id": file_id,
            "upload": upload_result,
            "variant": variant_result,
            "topics": topics_result,
            "preview": preview_result
        }



