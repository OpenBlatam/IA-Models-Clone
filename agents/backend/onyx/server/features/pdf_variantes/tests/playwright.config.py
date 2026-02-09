"""
Playwright Configuration
========================
Configuration for Playwright tests.
"""

import pytest
from playwright.sync_api import sync_playwright


def pytest_configure(config):
    """Configure pytest for Playwright."""
    config.addinivalue_line(
        "markers", "playwright: Playwright browser automation tests"
    )


@pytest.fixture(scope="session")
def playwright():
    """Playwright instance."""
    with sync_playwright() as p:
        yield p


@pytest.fixture(scope="session")
def browser_type(playwright, pytestconfig):
    """Browser type based on config."""
    browser_name = pytestconfig.getoption("--browser", default="chromium")
    
    browsers = {
        "chromium": playwright.chromium,
        "firefox": playwright.firefox,
        "webkit": playwright.webkit
    }
    
    return browsers.get(browser_name, playwright.chromium)


@pytest.fixture(scope="session")
def browser(browser_type, pytestconfig):
    """Browser instance."""
    headless = pytestconfig.getoption("--headless", default=True)
    
    browser = browser_type.launch(headless=headless)
    yield browser
    browser.close()


@pytest.fixture
def context(browser):
    """Browser context."""
    context = browser.new_context(
        viewport={"width": 1920, "height": 1080},
        record_video_dir="videos/" if pytest.getoption("--record-video") else None
    )
    yield context
    context.close()


@pytest.fixture
def page(context):
    """Page instance."""
    page = context.new_page()
    page.set_default_timeout(30000)  # 30 seconds
    yield page
    page.close()



