"""
Playwright Test Decorators
==========================
Decorators for Playwright tests to add common functionality.
"""

import pytest
import time
from functools import wraps
from typing import Callable, Any
from playwright.sync_api import Page, Response


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry test on failure."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except AssertionError as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


def measure_performance(func: Callable) -> Callable:
    """Decorator to measure test performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"Test {func.__name__} took {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Test {func.__name__} failed after {elapsed:.3f}s")
            raise e
    return wrapper


def capture_screenshot_on_failure(func: Callable) -> Callable:
    """Decorator to capture screenshot on test failure."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        page = None
        for arg in args:
            if isinstance(arg, Page):
                page = arg
                break
        
        if not page:
            for key, value in kwargs.items():
                if isinstance(value, Page):
                    page = value
                    break
        
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if page:
                try:
                    screenshot_path = f"screenshots/failure_{func.__name__}_{int(time.time())}.png"
                    page.screenshot(path=screenshot_path, full_page=True)
                    print(f"Screenshot saved: {screenshot_path}")
                except Exception:
                    pass
            raise e
    return wrapper


def validate_response_time(max_time: float = 1.0):
    """Decorator to validate response time."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            assert elapsed < max_time, \
                f"Test {func.__name__} took {elapsed:.3f}s, exceeds {max_time}s"
            
            return result
        return wrapper
    return decorator


def skip_if_api_unavailable(func: Callable) -> Callable:
    """Decorator to skip test if API is unavailable."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Try to find api_base_url in args or kwargs
        api_base_url = None
        for arg in args:
            if isinstance(arg, str) and "http" in arg:
                api_base_url = arg
                break
        
        if not api_base_url:
            api_base_url = kwargs.get("api_base_url")
        
        if api_base_url:
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    response = page.request.get(f"{api_base_url}/health", timeout=2000)
                    browser.close()
                    
                    if response.status != 200:
                        pytest.skip("API unavailable")
            except Exception:
                pytest.skip("API unavailable")
        
        return func(*args, **kwargs)
    return wrapper


def log_test_execution(func: Callable) -> Callable:
    """Decorator to log test execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Starting test: {func.__name__}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            print(f"Test {func.__name__} passed in {elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"Test {func.__name__} failed after {elapsed:.3f}s: {str(e)}")
            raise e
    return wrapper


def require_auth(func: Callable) -> Callable:
    """Decorator to ensure authentication is available."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if "auth_headers" not in kwargs:
            # Try to find in args
            auth_headers = None
            for arg in args:
                if isinstance(arg, dict) and "Authorization" in arg:
                    auth_headers = arg
                    break
            
            if not auth_headers:
                pytest.skip("Authentication required but not provided")
        
        return func(*args, **kwargs)
    return wrapper



