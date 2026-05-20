from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
Simple test runner for Onyx AI Video System.

This script provides a quick way to run tests and generate basic reports.
"""


def run_tests():
    """Run all tests and generate reports."""
    
    # Get the project root
    project_root = Path(__file__).parent.parent
    test_dir = project_root / "tests"
    
    print("🚀 Onyx AI Video System - Test Runner")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not test_dir.exists():
        print("❌ Tests directory not found!")
        print(f"Expected: {test_dir}")
        return False
    
    # Create reports directory
    reports_dir = project_root / "test_reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Run different test categories
    test_categories = [
        ("unit", "Unit Tests"),
        ("integration", "Integration Tests"), 
        ("system", "System Tests")
    ]
    
    results = {}
    total_start_time = time.time()
    
    for category, description in test_categories:
        print(f"\n📋 Running {description}...")
        
        category_dir = test_dir / category
        if not category_dir.exists():
            print(f"⚠️  {description} directory not found, skipping...")
            continue
        
        start_time = time.time()
        
        # Run pytest for this category
        cmd = [
            sys.executable, "-m", "pytest",
            str(category_dir),
            "-v",
            "--tb=short",
            "--asyncio-mode=auto",
            "--cov=onyx_ai_video",
            f"--cov-report=html:{reports_dir}/coverage_{category}",
            f"--cov-report=json:{reports_dir}/coverage_{category}.json",
            "--cov-report=term-missing",
            f"--html={reports_dir}/report_{category}.html",
            "--self-contained-html",
            f"--junitxml={reports_dir}/junit_{category}.xml"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            end_time = time.time()
            
            results[category] = {
                "exit_code": result.returncode,
                "duration": end_time - start_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if result.returncode == 0:
                print(f"✅ {description} completed successfully ({end_time - start_time:.2f}s)")
            else:
                print(f"❌ {description} failed ({end_time - start_time:.2f}s)")
                print(f"Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out")
            results[category] = {
                "exit_code": -1,
                "duration": 600,
                "stdout": "",
                "stderr": "Test timeout"
            }
        except Exception as e:
            print(f"❌ {description} error: {e}")
            results[category] = {
                "exit_code": -1,
                "duration": 0,
                "stdout": "",
                "stderr": str(e)
            }
    
    total_end_time = time.time()
    
    # Generate summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for category, result in results.items():
        if result["exit_code"] == 0:
            status = "✅ PASSED"
            total_passed += 1
        else:
            status = "❌ FAILED"
            total_failed += 1
        
        total_tests += 1
        print(f"{category.upper():15} {status:10} ({result['duration']:.2f}s)")
    
    print(f"\nTotal Duration: {total_end_time - total_start_time:.2f}s")
    print(f"Success Rate: {(total_passed/total_tests*100):.1f}%" if total_tests > 0 else "No tests run")
    
    # Show report locations
    print(f"\n📊 Reports generated in: {reports_dir}")
    print("Available reports:")
    for category in results.keys():
        print(f"  - {category}: {reports_dir}/report_{category}.html")
        print(f"  - Coverage: {reports_dir}/coverage_{category}/index.html")
    
    return total_failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1) 