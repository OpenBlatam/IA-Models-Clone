#!/usr/bin/env python3
"""
API Comparator
==============
Compare API responses and performance between different versions/configurations.

⚠️ DEPRECATED: This file is deprecated. Consider migrating to the new tools structure.

For new code, use:
    from tools.manager import ToolManager
    manager = ToolManager()
    # Tools can be extended with comparison capabilities
"""
import warnings

warnings.warn(
    "api_comparator.py is deprecated. Consider migrating to the new tools structure in tools/.",
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
from playwright_comparison import PlaywrightComparator, ComparisonResult


@dataclass
class ComparisonData:
    """Comparison data."""
    endpoint: str
    method: str
    url1: str
    url2: str
    response1: Dict[str, Any]
    response2: Dict[str, Any]
    comparison: ComparisonResult
    timestamp: str


class APIComparator:
    """API comparison tool."""
    
    def __init__(self, base_url1: str, base_url2: str):
        self.base_url1 = base_url1
        self.base_url2 = base_url2
        self.session1 = requests.Session()
        self.session2 = requests.Session()
        self.comparisons: List[ComparisonData] = []
    
    def compare_endpoint(
        self,
        method: str,
        endpoint: str,
        compare_body: bool = True,
        compare_headers: bool = True,
        **kwargs
    ) -> ComparisonData:
        """Compare endpoint between two APIs."""
        print(f"🔍 Comparing {method} {endpoint}...")
        
        url1 = f"{self.base_url1}{endpoint}"
        url2 = f"{self.base_url2}{endpoint}"
        
        # Make requests
        start1 = time.time()
        try:
            if method.upper() == "GET":
                response1 = self.session1.get(url1, timeout=10, **kwargs)
            elif method.upper() == "POST":
                response1 = self.session1.post(url1, timeout=10, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            time1 = (time.time() - start1) * 1000
        except Exception as e:
            response1 = None
            time1 = 0
            error1 = str(e)
        
        start2 = time.time()
        try:
            if method.upper() == "GET":
                response2 = self.session2.get(url2, timeout=10, **kwargs)
            elif method.upper() == "POST":
                response2 = self.session2.post(url2, timeout=10, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")
            time2 = (time.time() - start2) * 1000
        except Exception as e:
            response2 = None
            time2 = 0
            error2 = str(e)
        
        # Prepare data
        data1 = {
            "status": response1.status_code if response1 else 0,
            "time": time1,
            "headers": dict(response1.headers) if response1 else {},
            "body": response1.json() if response1 and response1.status_code == 200 else None,
            "error": error1 if not response1 else None
        }
        
        data2 = {
            "status": response2.status_code if response2 else 0,
            "time": time2,
            "headers": dict(response2.headers) if response2 else {},
            "body": response2.json() if response2 and response2.status_code == 200 else None,
            "error": error2 if not response2 else None
        }
        
        # Compare
        if response1 and response2:
            comparison = PlaywrightComparator.compare_responses(
                response1, response2, compare_body=compare_body, compare_headers=compare_headers
            )
        else:
            comparison = ComparisonResult(
                are_equal=False,
                differences=["One or both requests failed"],
                details={}
            )
        
        # Add performance comparison
        if time1 > 0 and time2 > 0:
            time_diff = time2 - time1
            time_pct = (time_diff / time1 * 100) if time1 > 0 else 0
            comparison.details["performance"] = {
                "time1": time1,
                "time2": time2,
                "difference": time_diff,
                "percent_change": time_pct
            }
        
        comparison_data = ComparisonData(
            endpoint=endpoint,
            method=method.upper(),
            url1=url1,
            url2=url2,
            response1=data1,
            response2=data2,
            comparison=comparison,
            timestamp=datetime.now().isoformat()
        )
        
        self.comparisons.append(comparison_data)
        return comparison_data
    
    def print_comparison(self, comparison: ComparisonData):
        """Print comparison results."""
        print("\n" + "=" * 70)
        print(f"📊 Comparison: {comparison.method} {comparison.endpoint}")
        print("=" * 70)
        
        status_emoji = "✅" if comparison.comparison.are_equal else "❌"
        print(f"{status_emoji} Are Equal: {comparison.comparison.are_equal}")
        print()
        
        print("Response 1:")
        print(f"  URL: {comparison.url1}")
        print(f"  Status: {comparison.response1['status']}")
        print(f"  Time: {comparison.response1['time']:.2f}ms")
        if comparison.response1.get('error'):
            print(f"  Error: {comparison.response1['error']}")
        
        print("\nResponse 2:")
        print(f"  URL: {comparison.url2}")
        print(f"  Status: {comparison.response2['status']}")
        print(f"  Time: {comparison.response2['time']:.2f}ms")
        if comparison.response2.get('error'):
            print(f"  Error: {comparison.response2['error']}")
        
        if comparison.comparison.differences:
            print("\nDifferences:")
            for diff in comparison.comparison.differences:
                print(f"  - {diff}")
        
        if "performance" in comparison.comparison.details:
            perf = comparison.comparison.details["performance"]
            print(f"\nPerformance:")
            print(f"  Time Difference: {perf['difference']:+.2f}ms ({perf['percent_change']:+.2f}%)")
        
        print("=" * 70)
    
    def export_comparisons(self, file_path: Path):
        """Export comparison results."""
        data = {
            "comparisons": [
                {
                    "endpoint": c.endpoint,
                    "method": c.method,
                    "url1": c.url1,
                    "url2": c.url2,
                    "are_equal": c.comparison.are_equal,
                    "differences": c.comparison.differences,
                    "details": c.comparison.details,
                    "timestamp": c.timestamp
                }
                for c in self.comparisons
            ],
            "exported_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Comparison results exported to {file_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="API Comparator")
    parser.add_argument("--url1", required=True, help="First API base URL")
    parser.add_argument("--url2", required=True, help="Second API base URL")
    parser.add_argument("--endpoint", default="/health", help="Endpoint to compare")
    parser.add_argument("--method", default="GET", help="HTTP method")
    parser.add_argument("--export", help="Export results to file")
    
    args = parser.parse_args()
    
    comparator = APIComparator(base_url1=args.url1, base_url2=args.url2)
    
    comparison = comparator.compare_endpoint(
        method=args.method,
        endpoint=args.endpoint
    )
    
    comparator.print_comparison(comparison)
    
    if args.export:
        comparator.export_comparisons(Path(args.export))


if __name__ == "__main__":
    main()



