"""
Playwright Debugging Utilities
================================
Utilities for debugging and troubleshooting Playwright tests.
"""

from playwright.sync_api import Page, Response
from typing import Dict, Any, List, Optional
import json
import time
from pathlib import Path


class PlaywrightDebugger:
    """Debugging utilities for Playwright tests."""
    
    def __init__(self, page: Page, output_dir: str = "debug_output"):
        self.page = page
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.network_logs: List[Dict] = []
        self.console_logs: List[Dict] = []
        self.errors: List[Dict] = []
        self._setup_listeners()
    
    def _setup_listeners(self):
        """Setup event listeners for debugging."""
        # Network logs
        def handle_request(request):
            self.network_logs.append({
                "type": "request",
                "url": request.url,
                "method": request.method,
                "headers": request.headers,
                "post_data": request.post_data,
                "timestamp": time.time()
            })
        
        def handle_response(response):
            self.network_logs.append({
                "type": "response",
                "url": response.url,
                "status": response.status,
                "headers": dict(response.headers),
                "timestamp": time.time()
            })
        
        self.page.on("request", handle_request)
        self.page.on("response", handle_response)
        
        # Console logs
        def handle_console(msg):
            log_entry = {
                "type": msg.type,
                "text": msg.text,
                "location": {
                    "url": msg.location.get("url", ""),
                    "line": msg.location.get("lineNumber", 0),
                    "column": msg.location.get("columnNumber", 0)
                },
                "timestamp": time.time()
            }
            self.console_logs.append(log_entry)
            
            if msg.type == "error":
                self.errors.append(log_entry)
        
        self.page.on("console", handle_console)
        
        # Page errors
        def handle_page_error(error):
            self.errors.append({
                "type": "page_error",
                "message": str(error),
                "timestamp": time.time()
            })
        
        self.page.on("pageerror", handle_page_error)
    
    def capture_screenshot(self, name: str = None) -> Path:
        """Capture screenshot for debugging."""
        if name is None:
            name = f"screenshot_{int(time.time())}.png"
        
        file_path = self.output_dir / name
        self.page.screenshot(path=str(file_path), full_page=True)
        return file_path
    
    def capture_network_log(self) -> Dict[str, Any]:
        """Capture network activity log."""
        return {
            "total_requests": len([l for l in self.network_logs if l["type"] == "request"]),
            "total_responses": len([l for l in self.network_logs if l["type"] == "response"]),
            "failed_requests": len([
                l for l in self.network_logs 
                if l["type"] == "response" and l.get("status", 0) >= 400
            ]),
            "logs": self.network_logs
        }
    
    def capture_console_log(self) -> Dict[str, Any]:
        """Capture console log."""
        return {
            "total_logs": len(self.console_logs),
            "errors": len([l for l in self.console_logs if l["type"] == "error"]),
            "warnings": len([l for l in self.console_logs if l["type"] == "warning"]),
            "logs": self.console_logs
        }
    
    def capture_all_errors(self) -> List[Dict]:
        """Capture all errors."""
        return self.errors
    
    def save_debug_info(self, test_name: str):
        """Save all debug information."""
        debug_info = {
            "test_name": test_name,
            "timestamp": time.time(),
            "url": self.page.url,
            "title": self.page.title(),
            "network": self.capture_network_log(),
            "console": self.capture_console_log(),
            "errors": self.capture_all_errors(),
            "screenshot": str(self.capture_screenshot(f"{test_name}_screenshot.png"))
        }
        
        file_path = self.output_dir / f"{test_name}_debug.json"
        file_path.write_text(json.dumps(debug_info, indent=2))
        return file_path
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze page performance."""
        performance_metrics = self.page.evaluate("""
            () => {
                const perf = window.performance;
                const timing = perf.timing;
                return {
                    dom_loading: timing.domLoading - timing.navigationStart,
                    dom_complete: timing.domComplete - timing.navigationStart,
                    load_complete: timing.loadEventEnd - timing.navigationStart,
                    first_paint: perf.getEntriesByType('paint')[0]?.startTime || 0,
                    first_contentful_paint: perf.getEntriesByType('paint')[1]?.startTime || 0
                };
            }
        """)
        
        return {
            "metrics": performance_metrics,
            "network_requests": len(self.network_logs),
            "slow_requests": [
                log for log in self.network_logs
                if log.get("duration", 0) > 1000
            ]
        }
    
    def wait_and_debug(self, condition, timeout: int = 5000, interval: int = 100):
        """Wait for condition and capture debug info if it fails."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if condition():
                    return True
            except Exception as e:
                pass
            time.sleep(interval / 1000)
        
        # Condition failed, capture debug info
        self.capture_screenshot("wait_failed.png")
        return False
    
    def compare_responses(self, response1: Response, response2: Response) -> Dict[str, Any]:
        """Compare two responses."""
        return {
            "status_diff": response1.status != response2.status,
            "headers_diff": set(response1.headers.items()) != set(response2.headers.items()),
            "url_diff": response1.url != response2.url,
            "response1": {
                "status": response1.status,
                "url": response1.url,
                "headers": dict(response1.headers)
            },
            "response2": {
                "status": response2.status,
                "url": response2.url,
                "headers": dict(response2.headers)
            }
        }


class PlaywrightTroubleshooter:
    """Troubleshooting utilities."""
    
    @staticmethod
    def diagnose_timeout(page: Page, timeout: int = 5000) -> Dict[str, Any]:
        """Diagnose timeout issues."""
        issues = []
        
        # Check for pending requests
        pending_requests = page.evaluate("""
            () => {
                return window.performance.getEntriesByType('resource')
                    .filter(r => r.responseEnd === 0).length;
            }
        """)
        
        if pending_requests > 0:
            issues.append(f"{pending_requests} pending requests detected")
        
        # Check for JavaScript errors
        console_errors = page.evaluate("""
            () => {
                return window.console._errors || [];
            }
        """)
        
        if console_errors:
            issues.append(f"{len(console_errors)} JavaScript errors detected")
        
        # Check page load state
        load_state = page.evaluate("document.readyState")
        if load_state != "complete":
            issues.append(f"Page not fully loaded: {load_state}")
        
        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "pending_requests": pending_requests,
            "console_errors": len(console_errors),
            "load_state": load_state
        }
    
    @staticmethod
    def diagnose_slow_performance(page: Page) -> Dict[str, Any]:
        """Diagnose slow performance."""
        performance_data = page.evaluate("""
            () => {
                const perf = window.performance;
                const timing = perf.timing;
                const resources = perf.getEntriesByType('resource');
                
                const slow_resources = resources
                    .filter(r => r.duration > 1000)
                    .map(r => ({
                        name: r.name,
                        duration: r.duration,
                        size: r.transferSize || 0
                    }));
                
                return {
                    dom_loading: timing.domLoading - timing.navigationStart,
                    dom_complete: timing.domComplete - timing.navigationStart,
                    load_complete: timing.loadEventEnd - timing.navigationStart,
                    slow_resources: slow_resources,
                    total_resources: resources.length
                };
            }
        """)
        
        issues = []
        if performance_data["dom_loading"] > 3000:
            issues.append("Slow DOM loading")
        if performance_data["load_complete"] > 5000:
            issues.append("Slow page load")
        if len(performance_data["slow_resources"]) > 0:
            issues.append(f"{len(performance_data['slow_resources'])} slow resources")
        
        return {
            "has_issues": len(issues) > 0,
            "issues": issues,
            "performance_data": performance_data
        }
    
    @staticmethod
    def diagnose_api_errors(page: Page) -> List[Dict[str, Any]]:
        """Diagnose API errors from network logs."""
        # This would need to be called after requests are made
        # and would analyze the network_logs from PlaywrightDebugger
        return []


def create_debugger(page: Page, output_dir: str = "debug_output") -> PlaywrightDebugger:
    """Create a debugger instance."""
    return PlaywrightDebugger(page, output_dir)


def troubleshoot_timeout(page: Page) -> Dict[str, Any]:
    """Quick timeout troubleshooting."""
    return PlaywrightTroubleshooter.diagnose_timeout(page)


def troubleshoot_performance(page: Page) -> Dict[str, Any]:
    """Quick performance troubleshooting."""
    return PlaywrightTroubleshooter.diagnose_slow_performance(page)



