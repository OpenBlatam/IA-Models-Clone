"""
Playwright Configuration
========================
Centralized configuration for Playwright tests.
"""

import os
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class PlaywrightConfig:
    """Configuration for Playwright tests."""
    api_base_url: str = os.getenv("API_BASE_URL", "http://localhost:8000")
    default_timeout: int = 30000
    headless: bool = True
    browser: str = "chromium"
    viewport_width: int = 1920
    viewport_height: int = 1080
    slow_mo: int = 0
    retry_count: int = 3
    retry_delay: float = 1.0
    screenshot_on_failure: bool = True
    video_on_failure: bool = False
    trace_on_failure: bool = False


@dataclass
class APIConfig:
    """Configuration for API tests."""
    default_auth_token: str = "test_token_123"
    default_user_id: str = "test_user_123"
    max_file_size_mb: int = 100
    default_timeout: int = 30000
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0


@dataclass
class PerformanceConfig:
    """Configuration for performance tests."""
    response_time_threshold: float = 1.0
    p95_threshold: float = 2.0
    p99_threshold: float = 3.0
    throughput_min: float = 5.0
    memory_increase_max_mb: float = 100.0
    iterations_default: int = 20


@dataclass
class SecurityConfig:
    """Configuration for security tests."""
    test_injection_payloads: List[str] = None
    test_xss_payloads: List[str] = None
    test_sql_payloads: List[str] = None
    
    def __post_init__(self):
        if self.test_injection_payloads is None:
            self.test_injection_payloads = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../etc/passwd"
            ]
        if self.test_xss_payloads is None:
            self.test_xss_payloads = [
                "<script>alert('xss')</script>",
                "<img src=x onerror=alert('xss')>",
                "javascript:alert('xss')"
            ]
        if self.test_sql_payloads is None:
            self.test_sql_payloads = [
                "'; DROP TABLE users; --",
                "' OR '1'='1",
                "'; INSERT INTO users VALUES ('hacker', 'pass'); --"
            ]


@dataclass
class TestDataConfig:
    """Configuration for test data."""
    sample_pdf_size_kb: int = 1
    large_pdf_size_kb: int = 5000
    default_filename: str = "test.pdf"
    variant_types: List[str] = None
    
    def __post_init__(self):
        if self.variant_types is None:
            self.variant_types = [
                "summary",
                "outline",
                "highlights",
                "notes",
                "quiz",
                "presentation"
            ]


# Global configuration instances
playwright_config = PlaywrightConfig()
api_config = APIConfig()
performance_config = PerformanceConfig()
security_config = SecurityConfig()
test_data_config = TestDataConfig()


def get_config() -> Dict[str, Any]:
    """Get all configurations as dictionary."""
    return {
        "playwright": playwright_config,
        "api": api_config,
        "performance": performance_config,
        "security": security_config,
        "test_data": test_data_config
    }


def update_config_from_env():
    """Update configuration from environment variables."""
    global playwright_config, api_config, performance_config
    
    # Playwright config
    if os.getenv("PLAYWRIGHT_HEADLESS"):
        playwright_config.headless = os.getenv("PLAYWRIGHT_HEADLESS").lower() == "true"
    
    if os.getenv("PLAYWRIGHT_BROWSER"):
        playwright_config.browser = os.getenv("PLAYWRIGHT_BROWSER")
    
    if os.getenv("PLAYWRIGHT_TIMEOUT"):
        playwright_config.default_timeout = int(os.getenv("PLAYWRIGHT_TIMEOUT"))
    
    # API config
    if os.getenv("API_BASE_URL"):
        playwright_config.api_base_url = os.getenv("API_BASE_URL")
    
    if os.getenv("API_AUTH_TOKEN"):
        api_config.default_auth_token = os.getenv("API_AUTH_TOKEN")
    
    # Performance config
    if os.getenv("PERF_RESPONSE_TIME_THRESHOLD"):
        performance_config.response_time_threshold = float(
            os.getenv("PERF_RESPONSE_TIME_THRESHOLD")
        )


# Update config from environment on import
update_config_from_env()



