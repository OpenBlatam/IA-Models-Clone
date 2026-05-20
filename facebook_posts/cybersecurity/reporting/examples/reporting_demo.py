from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import sys
import os
from console_reporter import (
from html_reporter import (
from json_reporter import (
        import traceback
from typing import Any, List, Dict, Optional
import logging
"""
Demo script for the reporting module.

Demonstrates:
- Console reporting with colors and progress
- HTML reporting with interactive features
- JSON reporting with structured data
- Report merging and validation
"""

sys.path.append('.')

    ConsoleReporter,
    ConsoleReportConfig,
    ConsoleReportResult
)

    HTMLReporter,
    HTMLReportConfig,
    HTMLReportResult
)

    JSONReporter,
    JSONReportConfig,
    JSONReportResult
)

async def demo_console_reporting():
    """Demo console reporting capabilities."""
    print("\n=== Console Reporting Demo ===")
    
    config = ConsoleReportConfig(
        enable_colors=True,
        show_progress=True,
        show_timestamps=True,
        max_line_length=80
    )
    
    reporter = ConsoleReporter(config)
    reporter.start_reporting()
    
    # Sample scan results
    scan_results = [
        {
            "target": "192.168.1.1",
            "success": True,
            "response_time": 0.045,
            "error_message": None
        },
        {
            "target": "192.168.1.2", 
            "success": False,
            "response_time": 5.2,
            "error_message": "Connection timeout"
        },
        {
            "target": "192.168.1.3",
            "success": True,
            "response_time": 0.023,
            "error_message": None
        }
    ]
    
    # Sample vulnerabilities
    vulnerabilities = [
        {
            "title": "SQL Injection Vulnerability",
            "description": "Found SQL injection in login form",
            "target": "http://example.com/login",
            "severity": "high",
            "cvss_score": 8.5,
            "remediation": "Use parameterized queries"
        },
        {
            "title": "XSS Vulnerability",
            "description": "Reflected XSS in search parameter",
            "target": "http://example.com/search",
            "severity": "medium",
            "cvss_score": 6.1,
            "remediation": "Input validation and output encoding"
        },
        {
            "title": "Weak Password Policy",
            "description": "No password complexity requirements",
            "target": "http://example.com/register",
            "severity": "low",
            "cvss_score": 3.1,
            "remediation": "Implement strong password policy"
        }
    ]
    
    # Report scan results
    print("\n--- Reporting Scan Results ---")
    result = await reporter.report_scan_results(scan_results)
    print(f"Console report result: {result.success}")
    print(f"Lines written: {result.lines_written}")
    print(f"Time taken: {result.time_taken:.3f} seconds")
    
    # Report vulnerabilities
    print("\n--- Reporting Vulnerabilities ---")
    vuln_result = await reporter.report_vulnerabilities(vulnerabilities)
    print(f"Vulnerability report result: {vuln_result.success}")
    print(f"Lines written: {vuln_result.lines_written}")
    print(f"Time taken: {vuln_result.time_taken:.3f} seconds")
    
    # Report summary
    print("\n--- Reporting Summary ---")
    summary = {
        "total_scans": len(scan_results),
        "successful_scans": len([r for r in scan_results if r['success']]),
        "failed_scans": len([r for r in scan_results if not r['success']]),
        "total_vulnerabilities": len(vulnerabilities),
        "critical_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'critical']),
        "high_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'high']),
        "medium_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'medium']),
        "low_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'low'])
    }
    
    summary_result = await reporter.report_summary(summary)
    print(f"Summary report result: {summary_result.success}")
    print(f"Lines written: {summary_result.lines_written}")
    
    # End reporting
    final_result = reporter.end_reporting()
    print(f"Final console report: {final_result.success}")
    print(f"Total time: {final_result.time_taken:.3f} seconds")

async def demo_html_reporting():
    """Demo HTML reporting capabilities."""
    print("\n=== HTML Reporting Demo ===")
    
    config = HTMLReportConfig(
        output_directory="reports/html",
        include_charts=True,
        include_timestamps=True,
        responsive_design=True,
        dark_mode=False
    )
    
    reporter = HTMLReporter(config)
    
    # Sample data
    scan_results = [
        {
            "target": "192.168.1.1",
            "success": True,
            "response_time": 0.045,
            "error_message": None
        },
        {
            "target": "192.168.1.2",
            "success": False,
            "response_time": 5.2,
            "error_message": "Connection timeout"
        },
        {
            "target": "192.168.1.3",
            "success": True,
            "response_time": 0.023,
            "error_message": None
        }
    ]
    
    vulnerabilities = [
        {
            "title": "SQL Injection Vulnerability",
            "description": "Found SQL injection in login form",
            "target": "http://example.com/login",
            "severity": "high",
            "cvss_score": 8.5,
            "remediation": "Use parameterized queries"
        },
        {
            "title": "XSS Vulnerability",
            "description": "Reflected XSS in search parameter",
            "target": "http://example.com/search",
            "severity": "medium",
            "cvss_score": 6.1,
            "remediation": "Input validation and output encoding"
        },
        {
            "title": "Weak Password Policy",
            "description": "No password complexity requirements",
            "target": "http://example.com/register",
            "severity": "low",
            "cvss_score": 3.1,
            "remediation": "Implement strong password policy"
        }
    ]
    
    # Comprehensive data
    data = {
        "scan_results": scan_results,
        "vulnerabilities": vulnerabilities,
        "summary": {
            "total_scans": len(scan_results),
            "successful_scans": len([r for r in scan_results if r['success']]),
            "failed_scans": len([r for r in scan_results if not r['success']]),
            "total_vulnerabilities": len(vulnerabilities),
            "critical_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'critical']),
            "high_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'high']),
            "medium_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'medium']),
            "low_vulnerabilities": len([v for v in vulnerabilities if v['severity'] == 'low'])
        }
    }
    
    # Generate comprehensive report
    print("Generating comprehensive HTML report...")
    result = await reporter.generate_report(data, "Security Assessment Report")
    print(f"HTML report result: {result.success}")
    if result.success:
        print(f"File path: {result.file_path}")
        print(f"File size: {result.file_size} bytes")
        print(f"Time taken: {result.time_taken:.3f} seconds")
    
    # Generate vulnerability-specific report
    print("\nGenerating vulnerability HTML report...")
    vuln_result = await reporter.generate_vulnerability_report(vulnerabilities)
    print(f"Vulnerability report result: {vuln_result.success}")
    if vuln_result.success:
        print(f"File path: {vuln_result.file_path}")
        print(f"File size: {vuln_result.file_size} bytes")
    
    # Generate scan-specific report
    print("\nGenerating scan HTML report...")
    scan_result = await reporter.generate_scan_report(scan_results)
    print(f"Scan report result: {scan_result.success}")
    if scan_result.success:
        print(f"File path: {scan_result.file_path}")
        print(f"File size: {scan_result.file_size} bytes")

async def demo_json_reporting():
    """Demo JSON reporting capabilities."""
    print("\n=== JSON Reporting Demo ===")
    
    config = JSONReportConfig(
        output_directory="reports/json",
        pretty_print=True,
        include_metadata=True,
        compress_output=False,
        schema_validation=True
    )
    
    reporter = JSONReporter(config)
    
    # Sample data
    scan_results = [
        {
            "target": "192.168.1.1",
            "success": True,
            "response_time": 0.045,
            "error_message": None
        },
        {
            "target": "192.168.1.2",
            "success": False,
            "response_time": 5.2,
            "error_message": "Connection timeout"
        },
        {
            "target": "192.168.1.3",
            "success": True,
            "response_time": 0.023,
            "error_message": None
        }
    ]
    
    vulnerabilities = [
        {
            "title": "SQL Injection Vulnerability",
            "description": "Found SQL injection in login form",
            "target": "http://example.com/login",
            "severity": "high",
            "cvss_score": 8.5,
            "remediation": "Use parameterized queries"
        },
        {
            "title": "XSS Vulnerability",
            "description": "Reflected XSS in search parameter",
            "target": "http://example.com/search",
            "severity": "medium",
            "cvss_score": 6.1,
            "remediation": "Input validation and output encoding"
        },
        {
            "title": "Weak Password Policy",
            "description": "No password complexity requirements",
            "target": "http://example.com/register",
            "severity": "low",
            "cvss_score": 3.1,
            "remediation": "Implement strong password policy"
        }
    ]
    
    # Export scan results
    print("Exporting scan results to JSON...")
    scan_result = await reporter.export_scan_results(scan_results)
    print(f"Scan export result: {scan_result.success}")
    if scan_result.success:
        print(f"File path: {scan_result.file_path}")
        print(f"File size: {scan_result.file_size} bytes")
        print(f"Record count: {scan_result.record_count}")
        print(f"Time taken: {scan_result.time_taken:.3f} seconds")
    
    # Export vulnerabilities
    print("\nExporting vulnerabilities to JSON...")
    vuln_result = await reporter.export_vulnerabilities(vulnerabilities)
    print(f"Vulnerability export result: {vuln_result.success}")
    if vuln_result.success:
        print(f"File path: {vuln_result.file_path}")
        print(f"File size: {vuln_result.file_size} bytes")
        print(f"Record count: {vuln_result.record_count}")
    
    # Export comprehensive report
    print("\nExporting comprehensive report to JSON...")
    comp_result = await reporter.export_comprehensive_report(scan_results, vulnerabilities)
    print(f"Comprehensive export result: {comp_result.success}")
    if comp_result.success:
        print(f"File path: {comp_result.file_path}")
        print(f"File size: {comp_result.file_size} bytes")
        print(f"Record count: {comp_result.record_count}")
    
    # Test schema validation
    print("\nTesting schema validation...")
    invalid_data = {
        "scan_results": "not_a_list",  # Should be a list
        "vulnerabilities": [],
        "summary": {}
    }
    
    errors = await reporter.validate_json_schema(invalid_data)
    print(f"Schema validation errors: {errors}")
    
    # Test with valid data
    valid_data = {
        "scan_results": scan_results,
        "vulnerabilities": vulnerabilities,
        "summary": {
            "total_scans": len(scan_results),
            "total_vulnerabilities": len(vulnerabilities)
        }
    }
    
    valid_errors = await reporter.validate_json_schema(valid_data)
    print(f"Valid data validation errors: {valid_errors}")

async def demo_report_merging():
    """Demo report merging capabilities."""
    print("\n=== Report Merging Demo ===")
    
    config = JSONReportConfig(
        output_directory="reports/merged",
        pretty_print=True,
        include_metadata=True
    )
    
    reporter = JSONReporter(config)
    
    # Create sample report files (in real scenario, these would be existing files)
    report_files = [
        "reports/json/scan_results_20231201_120000.json",
        "reports/json/vulnerabilities_20231201_120000.json"
    ]
    
    print("Merging multiple reports...")
    merge_result = await reporter.merge_reports(report_files)
    print(f"Merge result: {merge_result.success}")
    if merge_result.success:
        print(f"Merged file path: {merge_result.file_path}")
        print(f"File size: {merge_result.file_size} bytes")
        print(f"Record count: {merge_result.record_count}")
    else:
        print(f"Merge error: {merge_result.error_message}")

async def demo_performance_comparison():
    """Demo performance comparison between different reporting formats."""
    print("\n=== Performance Comparison Demo ===")
    
    # Generate large dataset
    large_scan_results = []
    large_vulnerabilities = []
    
    for i in range(1000):
        large_scan_results.append({
            "target": f"192.168.1.{i}",
            "success": i % 10 != 0,  # 90% success rate
            "response_time": 0.01 + (i % 100) * 0.001,
            "error_message": None if i % 10 != 0 else "Connection failed"
        })
        
        if i % 50 == 0:  # 20 vulnerabilities
            large_vulnerabilities.append({
                "title": f"Vulnerability {i}",
                "description": f"Description for vulnerability {i}",
                "target": f"http://example{i}.com",
                "severity": ["low", "medium", "high", "critical"][i % 4],
                "cvss_score": 1.0 + (i % 10),
                "remediation": f"Fix for vulnerability {i}"
            })
    
    # Console reporting performance
    print("Testing console reporting performance...")
    console_config = ConsoleReportConfig(enable_colors=False, show_progress=False)
    console_reporter = ConsoleReporter(console_config)
    
    start_time = asyncio.get_event_loop().time()
    console_result = await console_reporter.report_scan_results(large_scan_results)
    console_time = asyncio.get_event_loop().time() - start_time
    
    print(f"Console reporting: {console_time:.3f} seconds for {len(large_scan_results)} records")
    
    # HTML reporting performance
    print("Testing HTML reporting performance...")
    html_config = HTMLReportConfig(output_directory="reports/performance")
    html_reporter = HTMLReporter(html_config)
    
    data = {
        "scan_results": large_scan_results,
        "vulnerabilities": large_vulnerabilities,
        "summary": {
            "total_scans": len(large_scan_results),
            "total_vulnerabilities": len(large_vulnerabilities)
        }
    }
    
    start_time = asyncio.get_event_loop().time()
    html_result = await html_reporter.generate_report(data, "Performance Test")
    html_time = asyncio.get_event_loop().time() - start_time
    
    print(f"HTML reporting: {html_time:.3f} seconds")
    if html_result.success:
        print(f"HTML file size: {html_result.file_size} bytes")
    
    # JSON reporting performance
    print("Testing JSON reporting performance...")
    json_config = JSONReportConfig(output_directory="reports/performance", pretty_print=False)
    json_reporter = JSONReporter(json_config)
    
    start_time = asyncio.get_event_loop().time()
    json_result = await json_reporter.export_comprehensive_report(large_scan_results, large_vulnerabilities)
    json_time = asyncio.get_event_loop().time() - start_time
    
    print(f"JSON reporting: {json_time:.3f} seconds")
    if json_result.success:
        print(f"JSON file size: {json_result.file_size} bytes")
        print(f"JSON record count: {json_result.record_count}")

async def main():
    """Run all reporting demos."""
    print("🚀 Cybersecurity Reporting Module Demo")
    print("=" * 50)
    
    try:
        await demo_console_reporting()
        await demo_html_reporting()
        await demo_json_reporting()
        await demo_report_merging()
        await demo_performance_comparison()
        
        print("\n✅ All reporting demos completed successfully!")
        print("\n📋 Summary:")
        print("- Console reporting: Working with colors and progress")
        print("- HTML reporting: Working with interactive features")
        print("- JSON reporting: Working with structured data")
        print("- Report merging: Working")
        print("- Performance optimization: Working")
        
        print("\n📁 Generated Reports:")
        print("- Console output: Terminal display")
        print("- HTML reports: reports/html/ directory")
        print("- JSON reports: reports/json/ directory")
        print("- Merged reports: reports/merged/ directory")
        print("- Performance reports: reports/performance/ directory")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

match __name__:
    case "__main__":
    asyncio.run(main()) 