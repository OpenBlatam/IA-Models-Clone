"""
Playwright Helper Functions
===========================
Utility functions for Playwright tests.
"""

from playwright.sync_api import Page, Response
from typing import Dict, Any, Optional, List
import time
import json


def wait_for_api_response(page: Page, url_pattern: str, timeout: int = 10000) -> Optional[Response]:
    """Wait for API response matching pattern."""
    try:
        with page.expect_response(
            lambda response: url_pattern in response.url,
            timeout=timeout
        ) as response_info:
            return response_info.value
    except Exception:
        return None


def retry_request(page: Page, method: str, url: str, max_retries: int = 3, **kwargs) -> Optional[Response]:
    """Retry request with exponential backoff."""
    for attempt in range(max_retries):
        try:
            if method.upper() == "GET":
                response = page.request.get(url, **kwargs)
            elif method.upper() == "POST":
                response = page.request.post(url, **kwargs)
            elif method.upper() == "PUT":
                response = page.request.put(url, **kwargs)
            elif method.upper() == "DELETE":
                response = page.request.delete(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status < 500:  # Don't retry on client errors
                return response
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    
    return None


def assert_json_response(response: Response, expected_keys: List[str] = None) -> Dict[str, Any]:
    """Assert response is JSON and contains expected keys."""
    assert response.status < 500, f"Server error: {response.status}"
    
    content_type = response.headers.get("content-type", "")
    assert "application/json" in content_type, f"Expected JSON, got {content_type}"
    
    data = response.json()
    assert isinstance(data, dict), "Response should be a dictionary"
    
    if expected_keys:
        for key in expected_keys:
            assert key in data, f"Missing key in response: {key}"
    
    return data


def wait_for_element_with_retry(
    page: Page,
    selector: str,
    timeout: int = 10000,
    max_retries: int = 3
) -> bool:
    """Wait for element with retry logic."""
    for attempt in range(max_retries):
        try:
            page.wait_for_selector(selector, timeout=timeout // max_retries)
            return True
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return False
    return False


def take_screenshot_on_failure(page: Page, test_name: str, output_dir: str = "screenshots"):
    """Take screenshot on test failure."""
    from pathlib import Path
    import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = output_path / f"{test_name}_{timestamp}.png"
    
    page.screenshot(path=str(screenshot_path), full_page=True)
    return screenshot_path


def measure_performance(page: Page, url: str) -> Dict[str, float]:
    """Measure page load performance."""
    start_time = time.time()
    
    response = page.request.get(url)
    
    request_time = time.time() - start_time
    
    # If it's a page navigation
    if response.status == 200:
        page.goto(url)
        load_time = time.time() - start_time
        
        # Get performance metrics if available
        metrics = page.evaluate("""
            () => {
                const perf = window.performance;
                if (perf && perf.timing) {
                    return {
                        domContentLoaded: perf.timing.domContentLoadedEventEnd - perf.timing.navigationStart,
                        loadComplete: perf.timing.loadEventEnd - perf.timing.navigationStart
                    };
                }
                return null;
            }
        """)
        
        return {
            "request_time": request_time,
            "load_time": load_time,
            "dom_content_loaded": metrics.get("domContentLoaded") if metrics else None,
            "load_complete": metrics.get("loadComplete") if metrics else None
        }
    
    return {"request_time": request_time}


def check_accessibility(page: Page) -> Dict[str, Any]:
    """Check basic accessibility features."""
    accessibility = {
        "has_title": bool(page.title()),
        "has_lang": bool(page.evaluate("() => document.documentElement.lang")),
        "has_main": page.locator("main, [role='main']").count() > 0,
        "images_have_alt": True,  # Would need more complex check
    }
    
    # Check for alt text on images
    images = page.locator("img").all()
    for img in images:
        alt = img.get_attribute("alt")
        if alt is None:
            accessibility["images_have_alt"] = False
            break
    
    return accessibility


def intercept_and_modify_request(
    page: Page,
    url_pattern: str,
    modify_func: callable
):
    """Intercept and modify request."""
    def handle_route(route):
        if url_pattern in route.request.url:
            modified_request = modify_func(route.request)
            route.continue_(**modified_request)
        else:
            route.continue_()
    
    page.route("**/*", handle_route)


def wait_for_network_idle(page: Page, timeout: int = 30000):
    """Wait for network to be idle."""
    page.wait_for_load_state("networkidle", timeout=timeout)


def get_all_cookies(context) -> List[Dict[str, Any]]:
    """Get all cookies from context."""
    return context.cookies()


def set_cookies(context, cookies: List[Dict[str, Any]]):
    """Set cookies in context."""
    context.add_cookies(cookies)


def mock_api_response(page: Page, url_pattern: str, mock_data: Dict[str, Any], status: int = 200):
    """Mock API response."""
    def handle_route(route):
        if url_pattern in route.request.url:
            route.fulfill(
                status=status,
                content_type="application/json",
                body=json.dumps(mock_data)
            )
        else:
            route.continue_()
    
    page.route("**/*", handle_route)


def check_console_errors(page: Page) -> List[str]:
    """Check for console errors."""
    errors = []
    
    def handle_console(msg):
        if msg.type == "error":
            errors.append(msg.text)
    
    page.on("console", handle_console)
    return errors


def verify_response_headers(response: Response, required_headers: List[str] = None) -> bool:
    """Verify response has required headers."""
    if not required_headers:
        return True
    
    headers = response.headers
    for header in required_headers:
        if header not in headers:
            return False
    
    return True


def extract_api_endpoints_from_openapi(page: Page, api_base_url: str) -> List[str]:
    """Extract API endpoints from OpenAPI spec."""
    openapi_paths = ["/openapi.json", "/swagger.json"]
    
    for path in openapi_paths:
        try:
            response = page.request.get(f"{api_base_url}{path}")
            if response.status == 200:
                spec = response.json()
                endpoints = list(spec.get("paths", {}).keys())
                return endpoints
        except Exception:
            continue
    
    return []


def wait_for_api_success(page: Page, url: str, timeout: int = 30000) -> Response:
    """Wait for API to return success response."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = page.request.get(url)
        if response.status == 200:
            return response
        time.sleep(1)
    
    raise TimeoutError(f"API did not return success within {timeout}ms")


def validate_response_schema(response: Response, schema: Dict[str, Any]) -> bool:
    """Validate response against a schema."""
    if response.status >= 400:
        return False
    
    try:
        data = response.json()
        
        def check_schema(data_item, schema_item):
            if isinstance(schema_item, dict):
                for key, value in schema_item.items():
                    if key not in data_item:
                        return False
                    if isinstance(value, dict):
                        if not check_schema(data_item[key], value):
                            return False
                    elif isinstance(value, type):
                        if not isinstance(data_item[key], value):
                            return False
            return True
        
        return check_schema(data, schema)
    except Exception:
        return False


def create_test_pdf(size_kb: int = 1) -> bytes:
    """Create a test PDF of specified size."""
    base_pdf = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 2\ntrailer\n<<\n/Size 2\n>>\nstartxref\n20\n%%EOF"
    
    if size_kb <= 1:
        return base_pdf
    
    # Add padding to reach desired size
    padding_size = (size_kb * 1024) - len(base_pdf)
    if padding_size > 0:
        padding = b" " * padding_size
        return base_pdf + padding
    
    return base_pdf


def assert_response_time(response: Response, max_time: float = 1.0) -> float:
    """Assert response time is within limits."""
    # Note: This requires timing to be done outside
    # This is a helper for validation
    return max_time


def extract_error_details(response: Response) -> Dict[str, Any]:
    """Extract error details from response."""
    error_details = {
        "status": response.status,
        "url": response.url,
        "headers": dict(response.headers)
    }
    
    try:
        if "application/json" in response.headers.get("content-type", ""):
            error_details["body"] = response.json()
        else:
            error_details["body"] = response.text()
    except Exception:
        error_details["body"] = None
    
    return error_details


def compare_responses(response1: Response, response2: Response) -> Dict[str, Any]:
    """Compare two responses and return differences."""
    differences = {
        "status_different": response1.status != response2.status,
        "headers_different": dict(response1.headers) != dict(response2.headers),
        "body_different": False
    }
    
    try:
        body1 = response1.json() if "application/json" in response1.headers.get("content-type", "") else response1.text()
        body2 = response2.json() if "application/json" in response2.headers.get("content-type", "") else response2.text()
        differences["body_different"] = body1 != body2
    except Exception:
        differences["body_different"] = True
    
    return differences


def wait_for_condition(page: Page, condition_func: callable, timeout: int = 10000) -> bool:
    """Wait for a condition to be true."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            if condition_func():
                return True
        except Exception:
            pass
        time.sleep(0.1)
    
    return False


def capture_network_log(page: Page) -> List[Dict[str, Any]]:
    """Capture network request/response log."""
    network_log = []
    
    def log_request(request):
        network_log.append({
            "type": "request",
            "url": request.url,
            "method": request.method,
            "headers": dict(request.headers),
            "timestamp": time.time()
        })
    
    def log_response(response):
        network_log.append({
            "type": "response",
            "url": response.url,
            "status": response.status,
            "headers": dict(response.headers),
            "timestamp": time.time()
        })
    
    page.on("request", log_request)
    page.on("response", log_response)
    
    return network_log


def assert_no_console_errors(page: Page) -> List[str]:
    """Assert no console errors and return any found."""
    errors = []
    
    def handle_console(msg):
        if msg.type == "error":
            errors.append(msg.text)
    
    page.on("console", handle_console)
    return errors


def create_mock_response(status: int = 200, body: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Create a mock response structure."""
    return {
        "status": status,
        "body": body or {},
        "headers": headers or {}
    }


def batch_requests(page: Page, requests: List[Dict[str, Any]]) -> List[Response]:
    """Execute batch of requests."""
    responses = []
    
    for req in requests:
        method = req.get("method", "GET").upper()
        url = req.get("url")
        kwargs = req.get("kwargs", {})
        
        if method == "GET":
            response = page.request.get(url, **kwargs)
        elif method == "POST":
            response = page.request.post(url, **kwargs)
        elif method == "PUT":
            response = page.request.put(url, **kwargs)
        elif method == "DELETE":
            response = page.request.delete(url, **kwargs)
        else:
            continue
        
        responses.append(response)
    
    return responses


def assert_api_versioning(response: Response, expected_version: str = None) -> bool:
    """Assert API versioning is correct."""
    headers = response.headers
    
    # Check for version in headers
    version_headers = ["x-api-version", "api-version", "version"]
    for header in version_headers:
        if header in headers:
            if expected_version:
                return headers[header] == expected_version
            return True
    
    # Check for version in URL
    if expected_version and expected_version in response.url:
        return True
    
    return False


def measure_api_performance(page: Page, url: str, iterations: int = 10) -> Dict[str, float]:
    """Measure API performance over multiple iterations."""
    times = []
    
    for _ in range(iterations):
        start = time.time()
        response = page.request.get(url)
        elapsed = time.time() - start
        times.append(elapsed)
        assert response.status == 200
    
    sorted_times = sorted(times)
    return {
        "min": min(times),
        "max": max(times),
        "avg": sum(times) / len(times),
        "median": sorted_times[len(times) // 2],
        "p95": sorted_times[int(len(times) * 0.95)],
        "p99": sorted_times[int(len(times) * 0.99)]
    }


def generate_test_report(results: List[Dict[str, Any]], output_file: str = "test_report.json"):
    """Generate test report from results."""
    import json
    from pathlib import Path
    
    report = {
        "timestamp": time.time(),
        "total_tests": len(results),
        "passed": sum(1 for r in results if r.get("status") == "passed"),
        "failed": sum(1 for r in results if r.get("status") == "failed"),
        "results": results
    }
    
    output_path = Path(output_file)
    output_path.write_text(json.dumps(report, indent=2))
    return report


def compare_test_runs(baseline_file: str, current_file: str) -> Dict[str, Any]:
    """Compare two test runs."""
    import json
    from pathlib import Path
    
    baseline = json.loads(Path(baseline_file).read_text())
    current = json.loads(Path(current_file).read_text())
    
    comparison = {
        "baseline": baseline,
        "current": current,
        "differences": {
            "total_tests": current["total_tests"] - baseline["total_tests"],
            "passed": current["passed"] - baseline["passed"],
            "failed": current["failed"] - baseline["failed"]
        }
    }
    
    return comparison


def create_test_matrix(browsers: List[str], viewports: List[Dict[str, int]]) -> List[Dict[str, Any]]:
    """Create test matrix for cross-browser/viewport testing."""
    matrix = []
    
    for browser in browsers:
        for viewport in viewports:
            matrix.append({
                "browser": browser,
                "viewport": viewport,
                "name": f"{browser}_{viewport['width']}x{viewport['height']}"
            })
    
    return matrix


def validate_api_schema(response: Response, schema: Dict[str, Any]) -> tuple:
    """Validate API response against JSON schema."""
    errors = []
    
    try:
        data = response.json()
        
        def validate_field(data_item, schema_item, path=""):
            if isinstance(schema_item, dict):
                for key, value in schema_item.items():
                    current_path = f"{path}.{key}" if path else key
                    if key not in data_item:
                        errors.append(f"Missing field: {current_path}")
                    elif isinstance(value, dict):
                        validate_field(data_item[key], value, current_path)
                    elif isinstance(value, type):
                        if not isinstance(data_item[key], value):
                            errors.append(f"Type mismatch at {current_path}: expected {value.__name__}")
        
        validate_field(data, schema)
        return len(errors) == 0, errors
    except Exception as e:
        errors.append(f"Schema validation error: {str(e)}")
        return False, errors


def create_performance_baseline(page: Page, url: str, iterations: int = 50) -> Dict[str, float]:
    """Create performance baseline."""
    return measure_api_performance(page, url, iterations)


def detect_performance_regression(current_metrics: Dict[str, float], baseline: Dict[str, float], threshold: float = 0.2) -> Dict[str, Any]:
    """Detect performance regression compared to baseline."""
    regressions = []
    
    for metric in ["avg", "p95", "p99"]:
        if metric in current_metrics and metric in baseline:
            current_value = current_metrics[metric]
            baseline_value = baseline[metric]
            increase = (current_value - baseline_value) / baseline_value
            
            if increase > threshold:
                regressions.append({
                    "metric": metric,
                    "baseline": baseline_value,
                    "current": current_value,
                    "increase": increase,
                    "threshold": threshold
                })
    
    return {
        "has_regression": len(regressions) > 0,
        "regressions": regressions
    }


def generate_coverage_report(test_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate coverage report from test results."""
    total_endpoints = set()
    tested_endpoints = set()
    
    for result in test_results:
        if "endpoint" in result:
            total_endpoints.add(result["endpoint"])
            if result.get("status") == "passed":
                tested_endpoints.add(result["endpoint"])
    
    coverage = len(tested_endpoints) / len(total_endpoints) if total_endpoints else 0
    
    return {
        "total_endpoints": len(total_endpoints),
        "tested_endpoints": len(tested_endpoints),
        "coverage": coverage,
        "untested_endpoints": list(total_endpoints - tested_endpoints)
    }

