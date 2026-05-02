"""
Test Summary Generator
Generates a summary of test results from JSON exports
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

def load_json_results(file_path: str) -> Dict[str, Any]:
    """Load test results from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)

def format_summary(results: Dict[str, Any]) -> str:
    """Format test results summary"""
    total = results.get('total_tests', 0)
    passed = results.get('passed', 0)
    failures = results.get('failures', 0)
    errors = results.get('errors', 0)
    skipped = results.get('skipped', 0)
    success_rate = results.get('success_rate', 0.0)
    execution_time = results.get('execution_time', 0.0)
    tests_per_second = results.get('tests_per_second', 0.0)
    
    category = results.get('category', 'all')
    
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    TEST RESULTS SUMMARY                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Category:        {category:<40} ║
║  Total Tests:     {total:>6}                                           ║
║  Passed:          {passed:>6}                                           ║
║  Failures:        {failures:>6}                                           ║
║  Errors:          {errors:>6}                                           ║
║  Skipped:         {skipped:>6}                                           ║
║  Success Rate:   {success_rate:>5.1f}%                                        ║
║                                                              ║
║  Execution Time:  {execution_time:>6.2f}s                                      ║
║  Tests/Second:    {tests_per_second:>6.1f}                                      ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    
    return summary

def compare_results(current: Dict[str, Any], previous: Dict[str, Any]) -> str:
    """Compare current results with previous results"""
    current_total = current.get('total_tests', 0)
    previous_total = previous.get('total_tests', 0)
    current_passed = current.get('passed', 0)
    previous_passed = previous.get('passed', 0)
    current_rate = current.get('success_rate', 0.0)
    previous_rate = previous.get('success_rate', 0.0)
    
    comparison = []
    comparison.append("\n" + "=" * 60)
    comparison.append("COMPARISON WITH PREVIOUS RUN")
    comparison.append("=" * 60)
    comparison.append("")
    
    # Total tests comparison
    if current_total != previous_total:
        diff = current_total - previous_total
        comparison.append(f"Total Tests: {previous_total} → {current_total} ({diff:+d})")
    else:
        comparison.append(f"Total Tests: {current_total} (unchanged)")
    
    # Passed tests comparison
    diff_passed = current_passed - previous_passed
    if diff_passed != 0:
        comparison.append(f"Passed: {previous_passed} → {current_passed} ({diff_passed:+d})")
    else:
        comparison.append(f"Passed: {current_passed} (unchanged)")
    
    # Success rate comparison
    diff_rate = current_rate - previous_rate
    if abs(diff_rate) > 0.1:
        comparison.append(f"Success Rate: {previous_rate:.1f}% → {current_rate:.1f}% ({diff_rate:+.1f}%)")
    else:
        comparison.append(f"Success Rate: {current_rate:.1f}% (unchanged)")
    
    comparison.append("")
    
    return "\n".join(comparison)

def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_summary.py <results.json> [previous_results.json]")
        print()
        print("Examples:")
        print("  python test_summary.py results.json")
        print("  python test_summary.py results.json previous_results.json")
        sys.exit(1)
    
    results_file = sys.argv[1]
    results = load_json_results(results_file)
    
    # Print summary
    summary = format_summary(results)
    print(summary)
    
    # Compare with previous if provided
    if len(sys.argv) >= 3:
        previous_file = sys.argv[2]
        previous_results = load_json_results(previous_file)
        comparison = compare_results(results, previous_results)
        print(comparison)
    
    # Save summary to file
    summary_file = Path(results_file).stem + "_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summary)
        if len(sys.argv) >= 3:
            f.write(comparison)
    
    print(f"✅ Summary saved to {summary_file}")

if __name__ == "__main__":
    main()







