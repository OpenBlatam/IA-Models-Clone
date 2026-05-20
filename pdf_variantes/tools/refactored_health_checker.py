"""
Refactored Health Checker
=========================
Health checker using base classes and improved structure.
"""

from typing import Dict, Any, List
from pathlib import Path
from .base import BaseAPITool, ToolResult
from .config import get_config
from .utils import print_success, print_error, print_warning
import time


class HealthChecker(BaseAPITool):
    """Refactored health checker."""
    
    def check_endpoint(self, endpoint: str) -> Dict[str, Any]:
        """Check a single endpoint."""
        start_time = time.time()
        
        try:
            response = self.make_request("GET", endpoint)
            response_time = (time.time() - start_time) * 1000
            
            return {
                "endpoint": endpoint,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": response_time,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "endpoint": endpoint,
                "status": "unhealthy",
                "status_code": 0,
                "response_time": (time.time() - start_time) * 1000,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def run(self, endpoints: List[str] = None, **kwargs) -> ToolResult:
        """Run health check."""
        if endpoints is None:
            endpoints = ["/health", "/", "/docs"]
        
        print("🔍 Running health check...")
        
        results = []
        for endpoint in endpoints:
            result = self.check_endpoint(endpoint)
            results.append(result)
            self.results.append(result)
        
        # Calculate overall status
        healthy_count = sum(1 for r in results if r["status"] == "healthy")
        total = len(results)
        
        if healthy_count == total:
            status = "healthy"
            message = f"All {total} endpoints are healthy"
            print_success(message)
        elif healthy_count > 0:
            status = "degraded"
            message = f"{healthy_count}/{total} endpoints are healthy"
            print_warning(message)
        else:
            status = "unhealthy"
            message = "All endpoints are unhealthy"
            print_error(message)
        
        return ToolResult(
            success=status == "healthy",
            message=message,
            data={
                "overall_status": status,
                "results": results,
                "healthy": healthy_count,
                "total": total
            }
        )


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health Checker (Refactored)")
    parser.add_argument("--url", help="API base URL")
    parser.add_argument("--endpoints", nargs="+", help="Endpoints to check")
    parser.add_argument("--export", help="Export results")
    
    args = parser.parse_args()
    
    config = get_config()
    base_url = args.url or config.base_url
    
    checker = HealthChecker(base_url=base_url)
    result = checker.run(endpoints=args.endpoints)
    
    if args.export:
        checker.export_results(args.export)
    
    return 0 if result.success else 1


if __name__ == "__main__":
    exit(main())



