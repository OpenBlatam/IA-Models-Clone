"""
Playwright Comparison Utilities
=================================
Utilities for comparing test results, responses, and performance.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from playwright.sync_api import Response


@dataclass
class ComparisonResult:
    """Result of a comparison."""
    are_equal: bool
    differences: List[str]
    details: Dict[str, Any]


class PlaywrightComparator:
    """Utilities for comparing test results."""
    
    @staticmethod
    def compare_responses(
        response1: Response,
        response2: Response,
        compare_body: bool = True,
        compare_headers: bool = True
    ) -> ComparisonResult:
        """Compare two API responses."""
        differences = []
        details = {}
        
        # Compare status
        if response1.status != response2.status:
            differences.append(f"Status: {response1.status} != {response2.status}")
        
        # Compare headers
        if compare_headers:
            headers1 = set(response1.headers.items())
            headers2 = set(response2.headers.items())
            
            if headers1 != headers2:
                missing = headers1 - headers2
                extra = headers2 - headers1
                
                if missing:
                    differences.append(f"Missing headers in response2: {missing}")
                if extra:
                    differences.append(f"Extra headers in response2: {extra}")
        
        # Compare body
        if compare_body:
            try:
                body1 = response1.json()
                body2 = response2.json()
                
                if body1 != body2:
                    differences.append("Response bodies differ")
                    details["body_diff"] = {
                        "response1": body1,
                        "response2": body2
                    }
            except Exception:
                # Not JSON, compare text
                body1 = response1.text()
                body2 = response2.text()
                
                if body1 != body2:
                    differences.append("Response bodies differ")
        
        return ComparisonResult(
            are_equal=len(differences) == 0,
            differences=differences,
            details=details
        )
    
    @staticmethod
    def compare_test_results(
        result1: Dict[str, Any],
        result2: Dict[str, Any]
    ) -> ComparisonResult:
        """Compare two test results."""
        differences = []
        details = {}
        
        # Compare status
        if result1.get("status") != result2.get("status"):
            differences.append(
                f"Status: {result1.get('status')} != {result2.get('status')}"
            )
        
        # Compare duration
        duration1 = result1.get("duration", 0)
        duration2 = result2.get("duration", 0)
        
        if abs(duration1 - duration2) > 0.1:  # More than 100ms difference
            differences.append(
                f"Duration: {duration1:.3f}s != {duration2:.3f}s"
            )
            details["duration_diff"] = duration1 - duration2
        
        # Compare response times
        rt1 = result1.get("avg_response_time", 0)
        rt2 = result2.get("avg_response_time", 0)
        
        if abs(rt1 - rt2) > 10:  # More than 10ms difference
            differences.append(
                f"Response time: {rt1:.3f}ms != {rt2:.3f}ms"
            )
            details["response_time_diff"] = rt1 - rt2
        
        return ComparisonResult(
            are_equal=len(differences) == 0,
            differences=differences,
            details=details
        )
    
    @staticmethod
    def compare_performance_metrics(
        metrics1: Dict[str, float],
        metrics2: Dict[str, float],
        thresholds: Optional[Dict[str, float]] = None
    ) -> ComparisonResult:
        """Compare performance metrics."""
        differences = []
        details = {}
        
        if thresholds is None:
            thresholds = {
                "duration": 0.1,  # 100ms
                "response_time": 10,  # 10ms
                "memory": 10  # 10MB
            }
        
        for key in set(metrics1.keys()) | set(metrics2.keys()):
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            threshold = thresholds.get(key, 0)
            
            diff = abs(val1 - val2)
            if diff > threshold:
                differences.append(
                    f"{key}: {val1:.3f} != {val2:.3f} (diff: {diff:.3f})"
                )
                details[key] = {
                    "value1": val1,
                    "value2": val2,
                    "difference": diff
                }
        
        return ComparisonResult(
            are_equal=len(differences) == 0,
            differences=differences,
            details=details
        )
    
    @staticmethod
    def compare_json_structures(
        json1: Dict[str, Any],
        json2: Dict[str, Any],
        ignore_keys: Optional[List[str]] = None
    ) -> ComparisonResult:
        """Compare JSON structures, ignoring specified keys."""
        if ignore_keys is None:
            ignore_keys = []
        
        def filter_dict(d: Dict, ignore: List[str]) -> Dict:
            return {
                k: v for k, v in d.items()
                if k not in ignore
            }
        
        filtered1 = filter_dict(json1, ignore_keys)
        filtered2 = filter_dict(json2, ignore_keys)
        
        differences = []
        details = {}
        
        if filtered1 != filtered2:
            differences.append("JSON structures differ")
            
            # Find specific differences
            keys1 = set(filtered1.keys())
            keys2 = set(filtered2.keys())
            
            missing = keys1 - keys2
            extra = keys2 - keys1
            common = keys1 & keys2
            
            if missing:
                differences.append(f"Missing keys in json2: {missing}")
            if extra:
                differences.append(f"Extra keys in json2: {extra}")
            
            for key in common:
                if filtered1[key] != filtered2[key]:
                    differences.append(f"Different values for key '{key}'")
                    details[key] = {
                        "value1": filtered1[key],
                        "value2": filtered2[key]
                    }
        
        return ComparisonResult(
            are_equal=len(differences) == 0,
            differences=differences,
            details=details
        )
    
    @staticmethod
    def compare_file_contents(
        file1: Path,
        file2: Path
    ) -> ComparisonResult:
        """Compare contents of two files."""
        differences = []
        details = {}
        
        if not file1.exists():
            differences.append(f"File1 does not exist: {file1}")
            return ComparisonResult(
                are_equal=False,
                differences=differences,
                details=details
            )
        
        if not file2.exists():
            differences.append(f"File2 does not exist: {file2}")
            return ComparisonResult(
                are_equal=False,
                differences=differences,
                details=details
            )
        
        content1 = file1.read_text()
        content2 = file2.read_text()
        
        if content1 != content2:
            differences.append("File contents differ")
            details["size_diff"] = abs(len(content1) - len(content2))
            
            # Try to parse as JSON for better comparison
            try:
                json1 = json.loads(content1)
                json2 = json.loads(content2)
                json_comparison = PlaywrightComparator.compare_json_structures(
                    json1, json2
                )
                if not json_comparison.are_equal:
                    differences.extend(json_comparison.differences)
                    details["json_differences"] = json_comparison.details
            except Exception:
                pass
        
        return ComparisonResult(
            are_equal=len(differences) == 0,
            differences=differences,
            details=details
        )


def compare_responses(
    response1: Response,
    response2: Response,
    compare_body: bool = True
) -> ComparisonResult:
    """Quick response comparison."""
    return PlaywrightComparator.compare_responses(
        response1, response2, compare_body=compare_body
    )


def compare_test_results(
    result1: Dict[str, Any],
    result2: Dict[str, Any]
) -> ComparisonResult:
    """Quick test result comparison."""
    return PlaywrightComparator.compare_test_results(result1, result2)



