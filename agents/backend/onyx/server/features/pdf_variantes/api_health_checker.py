#!/usr/bin/env python3
"""
API Health Checker
==================
Comprehensive health checking tool with detailed diagnostics.

⚠️ DEPRECATED: This file is deprecated. Use `tools.refactored_health_checker.HealthChecker` instead.

For new code, use:
    from tools.refactored_health_checker import HealthChecker
    # or
    from tools.manager import ToolManager
    manager = ToolManager()
    result = manager.run_tool("health")
"""
import warnings

warnings.warn(
    "api_health_checker.py is deprecated. Use 'tools.refactored_health_checker.HealthChecker' instead.",
    DeprecationWarning,
    stacklevel=2
)

import requests
import time
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class HealthCheckResult:
    """Health check result."""
    endpoint: str
    status: str  # healthy, degraded, unhealthy
    status_code: int
    response_time: float
    timestamp: str
    details: Dict[str, Any]
    errors: List[str]


class APIHealthChecker:
    """Comprehensive API health checker."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
        self.results: List[HealthCheckResult] = []
    
    def check_health_endpoint(self) -> HealthCheckResult:
        """Check /health endpoint."""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    details = data
                    status = "healthy" if data.get("status") in ["healthy", "ok", "up"] else "degraded"
                except:
                    status = "degraded"
                    errors.append("Could not parse JSON response")
            else:
                status = "unhealthy"
                errors.append(f"Status code: {response.status_code}")
            
            return HealthCheckResult(
                endpoint="/health",
                status=status,
                status_code=response.status_code,
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                details=details,
                errors=errors
            )
        except requests.exceptions.Timeout:
            return HealthCheckResult(
                endpoint="/health",
                status="unhealthy",
                status_code=0,
                response_time=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                details={},
                errors=["Request timeout"]
            )
        except requests.exceptions.ConnectionError:
            return HealthCheckResult(
                endpoint="/health",
                status="unhealthy",
                status_code=0,
                response_time=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                details={},
                errors=["Connection error - API may not be running"]
            )
        except Exception as e:
            return HealthCheckResult(
                endpoint="/health",
                status="unhealthy",
                status_code=0,
                response_time=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                details={},
                errors=[str(e)]
            )
    
    def check_endpoint(self, endpoint: str, expected_status: int = 200) -> HealthCheckResult:
        """Check a specific endpoint."""
        start_time = time.time()
        errors = []
        details = {}
        
        try:
            response = self.session.get(f"{self.base_url}{endpoint}", timeout=5)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == expected_status:
                status = "healthy"
                try:
                    details = response.json()
                except:
                    details = {"text": response.text[:200]}
            elif response.status_code < 500:
                status = "degraded"
                errors.append(f"Unexpected status: {response.status_code} (expected {expected_status})")
            else:
                status = "unhealthy"
                errors.append(f"Server error: {response.status_code}")
            
            return HealthCheckResult(
                endpoint=endpoint,
                status=status,
                status_code=response.status_code,
                response_time=response_time,
                timestamp=datetime.now().isoformat(),
                details=details,
                errors=errors
            )
        except Exception as e:
            return HealthCheckResult(
                endpoint=endpoint,
                status="unhealthy",
                status_code=0,
                response_time=(time.time() - start_time) * 1000,
                timestamp=datetime.now().isoformat(),
                details={},
                errors=[str(e)]
            )
    
    def check_multiple_endpoints(self, endpoints: List[str]) -> List[HealthCheckResult]:
        """Check multiple endpoints."""
        results = []
        for endpoint in endpoints:
            result = self.check_endpoint(endpoint)
            results.append(result)
            self.results.append(result)
        return results
    
    def comprehensive_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        print("🔍 Performing comprehensive health check...")
        print("=" * 60)
        
        # Check health endpoint
        health_result = self.check_health_endpoint()
        self.results.append(health_result)
        
        # Check common endpoints
        common_endpoints = ["/", "/docs", "/openapi.json"]
        endpoint_results = self.check_multiple_endpoints(common_endpoints)
        
        # Calculate overall status
        all_results = [health_result] + endpoint_results
        healthy_count = sum(1 for r in all_results if r.status == "healthy")
        degraded_count = sum(1 for r in all_results if r.status == "degraded")
        unhealthy_count = sum(1 for r in all_results if r.status == "unhealthy")
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        summary = {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "results": {
                "healthy": healthy_count,
                "degraded": degraded_count,
                "unhealthy": unhealthy_count,
                "total": len(all_results)
            },
            "checks": [asdict(r) for r in all_results]
        }
        
        return summary
    
    def print_results(self, summary: Optional[Dict[str, Any]] = None):
        """Print health check results."""
        if summary is None:
            summary = {
                "overall_status": "unknown",
                "checks": [asdict(r) for r in self.results]
            }
        
        print("\n" + "=" * 60)
        print("📊 Health Check Results")
        print("=" * 60)
        
        status_emoji = {
            "healthy": "🟢",
            "degraded": "🟡",
            "unhealthy": "🔴"
        }
        
        emoji = status_emoji.get(summary["overall_status"], "⚪")
        print(f"{emoji} Overall Status: {summary['overall_status'].upper()}")
        print()
        
        if "results" in summary:
            print("Summary:")
            print(f"  Healthy: {summary['results']['healthy']}")
            print(f"  Degraded: {summary['results']['degraded']}")
            print(f"  Unhealthy: {summary['results']['unhealthy']}")
            print(f"  Total: {summary['results']['total']}")
            print()
        
        print("Detailed Results:")
        for check in summary.get("checks", []):
            emoji = status_emoji.get(check["status"], "⚪")
            print(f"\n{emoji} {check['endpoint']}")
            print(f"   Status: {check['status']}")
            print(f"   Status Code: {check['status_code']}")
            print(f"   Response Time: {check['response_time']:.2f}ms")
            
            if check.get("errors"):
                print(f"   Errors:")
                for error in check["errors"]:
                    print(f"     - {error}")
            
            if check.get("details"):
                print(f"   Details: {json.dumps(check['details'], indent=6)}")
        
        print("\n" + "=" * 60)
    
    def export_results(self, file_path: Path):
        """Export health check results."""
        summary = {
            "overall_status": "unknown",
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "checks": [asdict(r) for r in self.results]
        }
        
        with open(file_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Health check results exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Health Checker")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--endpoint", help="Check specific endpoint")
    parser.add_argument("--export", help="Export results to file")
    
    args = parser.parse_args()
    
    checker = APIHealthChecker(base_url=args.url)
    
    if args.endpoint:
        result = checker.check_endpoint(args.endpoint)
        checker.results.append(result)
        summary = {
            "overall_status": result.status,
            "checks": [asdict(result)]
        }
        checker.print_results(summary)
    else:
        summary = checker.comprehensive_check()
        checker.print_results(summary)
    
    if args.export:
        checker.export_results(Path(args.export))


if __name__ == "__main__":
    main()



