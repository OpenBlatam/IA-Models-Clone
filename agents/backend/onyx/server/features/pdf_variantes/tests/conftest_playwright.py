"""
Playwright Configuration for Tests
===================================
Fixtures and configuration for Playwright tests.
"""

import pytest
import time
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page


@pytest.fixture(scope="session")
def browser_type_launch_args():
    """Browser launch arguments."""
    return {
        "headless": True,
        "slow_mo": 0,  # Slow down operations (ms)
    }


@pytest.fixture(scope="session")
def browser(browser_type_launch_args):
    """Create browser instance."""
    with sync_playwright() as p:
        browser = p.chromium.launch(**browser_type_launch_args)
        yield browser
        browser.close()


@pytest.fixture
def context(browser):
    """Create browser context with default settings."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    yield context
    context.close()


@pytest.fixture
def page(context):
    """Create page for testing."""
    page = context.new_page()
    
    # Set default timeout
    page.set_default_timeout(10000)  # 10 seconds
    page.set_default_navigation_timeout(30000)  # 30 seconds for navigation
    
    # Enable request/response logging for debugging
    def handle_response(response):
        if response.status >= 400:
            print(f"Error response: {response.url} - {response.status}")
    
    page.on("response", handle_response)
    
    yield page
    page.close()


@pytest.fixture
def mobile_page(context):
    """Create page with mobile viewport."""
    mobile_context = context.browser.new_context(
        viewport={"width": 375, "height": 667},  # iPhone SE size
        user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
    )
    page = mobile_context.new_page()
    page.set_default_timeout(10000)
    yield page
    page.close()
    mobile_context.close()


@pytest.fixture
def tablet_page(context):
    """Create page with tablet viewport."""
    tablet_context = context.browser.new_context(
        viewport={"width": 768, "height": 1024},  # iPad size
        user_agent="Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
    )
    page = tablet_context.new_page()
    page.set_default_timeout(10000)
    yield page
    page.close()
    tablet_context.close()


@pytest.fixture
def authenticated_context(browser):
    """Create authenticated browser context."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        extra_http_headers={
            "Authorization": "Bearer test_token_123",
            "X-User-ID": "test_user_123"
        }
    )
    yield context
    context.close()


@pytest.fixture
def authenticated_page(authenticated_context):
    """Create authenticated page."""
    page = authenticated_context.new_page()
    page.set_default_timeout(10000)
    yield page
    page.close()


@pytest.fixture
def page_with_tracing(context):
    """Create page with tracing enabled."""
    # Start tracing
    context.tracing.start(screenshots=True, snapshots=True, sources=True)
    
    page = context.new_page()
    page.set_default_timeout(10000)
    
    yield page
    
    # Stop tracing
    from pathlib import Path
    import tempfile
    
    trace_path = Path(tempfile.gettempdir()) / f"trace_{time.time()}.zip"
    context.tracing.stop(path=str(trace_path))
    
    page.close()


@pytest.fixture
def page_with_video(context):
    """Create page with video recording."""
    context = context.browser.new_context(
        viewport={"width": 1920, "height": 1080},
        record_video_dir="videos/",
        record_video_size={"width": 1920, "height": 1080}
    )
    
    page = context.new_page()
    page.set_default_timeout(10000)
    
    yield page
    
    page.close()
    context.close()


@pytest.fixture
def page_with_har(context):
    """Create page with HAR recording."""
    context = context.browser.new_context(
        viewport={"width": 1920, "height": 1080}
    )
    
    page = context.new_page()
    page.set_default_timeout(10000)
    
    # Start HAR recording
    page.route_from_har("network.har", update=False)
    
    yield page
    
    page.close()
    context.close()


@pytest.fixture
def slow_network_context(browser):
    """Create context with slow network simulation."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}
    )
    
    # Simulate slow 3G
    cdp_session = context.new_cdp_session(context.pages[0] if context.pages else None)
    if cdp_session:
        try:
            cdp_session.send("Network.emulateNetworkConditions", {
                "offline": False,
                "downloadThroughput": 1.5 * 1024 * 1024 / 8,  # 1.5 Mbps
                "uploadThroughput": 750 * 1024 / 8,  # 750 Kbps
                "latency": 562.5  # ms
            })
        except Exception:
            pass  # CDP may not be available
    
    yield context
    context.close()


@pytest.fixture
def offline_context(browser):
    """Create context with offline mode."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080}
    )
    context.set_offline(True)
    
    yield context
    context.set_offline(False)
    context.close()


@pytest.fixture
def sample_pdf_content():
    """Sample PDF content for testing."""
    return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\nxref\n0 2\ntrailer\n<<\n/Size 2\n>>\nstartxref\n20\n%%EOF"


@pytest.fixture
def large_pdf_content(sample_pdf_content):
    """Large PDF content for testing."""
    # Create 5MB PDF
    padding = b" " * (5 * 1024 * 1024 - len(sample_pdf_content))
    return sample_pdf_content + padding


@pytest.fixture
def test_data():
    """Test data for various scenarios."""
    return {
        "valid_file_ids": ["file_1", "file_2", "file_3"],
        "invalid_file_ids": ["", "nonexistent", "invalid_id_123"],
        "variant_types": ["summary", "outline", "highlights", "notes", "quiz", "presentation"],
        "invalid_variant_types": ["invalid", "unknown", ""],
        "valid_options": {
            "max_length": 500,
            "style": "academic",
            "language": "en"
        },
        "invalid_options": {
            "max_length": -1,
            "style": "",
            "language": "invalid"
        }
    }


# Import common fixtures to avoid duplication
try:
    from .fixtures_common import api_base_url, auth_headers, sample_pdf, test_data, ci_timeout, performance_thresholds
except ImportError:
    # Fallback if fixtures_common not available
    @pytest.fixture
    def api_base_url(pytestconfig):
        """Get API base URL from config or use default."""
        return pytestconfig.getoption("--api-url", default="http://localhost:8000")


def pytest_addoption(parser):
    """Add custom pytest options."""
    parser.addoption(
        "--api-url",
        action="store",
        default="http://localhost:8000",
        help="Base URL for API"
    )
    parser.addoption(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode"
    )
    parser.addoption(
        "--browser",
        action="store",
        default="chromium",
        choices=["chromium", "firefox", "webkit"],
        help="Browser to use for tests"
    )

