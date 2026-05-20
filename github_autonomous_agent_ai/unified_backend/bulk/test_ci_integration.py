"""
Integración CI/CD para pruebas
Genera reportes compatibles con CI/CD systems
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

def generate_junit_xml(results_file: str = "test_results.json", output_file: str = "junit.xml"):
    """Genera reporte JUnit XML para CI/CD."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró {results_file}")
        return None
    
    summary = results.get("summary", {})
    tests = results.get("tests", [])
    
    total = summary.get("total", 0)
    failures = summary.get("failed", 0)
    
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
    <testsuite name="BUL API Tests" tests="{total}" failures="{failures}" time="{summary.get('duration', 0):.2f}">
"""
    
    for test in tests:
        test_name = test.get("name", "Unknown").replace(" ", "_").replace("'", "")
        duration = test.get("duration", 0)
        passed = test.get("passed", False)
        
        if passed:
            xml += f"""        <testcase name="{test_name}" time="{duration:.2f}"/>
"""
        else:
            error = test.get("details", {}).get("error", "Test failed")
            xml += f"""        <testcase name="{test_name}" time="{duration:.2f}">
            <failure message="{error}">Test failed</failure>
        </testcase>
"""
    
    xml += """    </testsuite>
</testsuites>
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(xml)
    
    print(f"✅ Reporte JUnit XML generado: {output_file}")
    return output_file

def generate_github_annotations(results_file: str = "test_results.json"):
    """Genera anotaciones para GitHub Actions."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        return
    
    errors = results.get("errors", [])
    
    # GitHub Actions annotations
    for error in errors:
        print(f"::error::{error.get('test', 'Unknown')}: {error.get('error', 'Unknown error')}")
    
    summary = results.get("summary", {})
    if summary.get("failed", 0) > 0:
        print(f"::warning::{summary.get('failed', 0)} pruebas fallaron")

def generate_gitlab_report(results_file: str = "test_results.json", output_file: str = "gl-test-report.json"):
    """Genera reporte para GitLab CI."""
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    except FileNotFoundError:
        return None
    
    tests = results.get("tests", [])
    
    gl_report = {
        "version": "1.1",
        "tests": []
    }
    
    for test in tests:
        gl_test = {
            "name": test.get("name", "Unknown"),
            "status": "success" if test.get("passed") else "failed",
            "duration": test.get("duration", 0),
        }
        
        if not test.get("passed"):
            gl_test["message"] = test.get("details", {}).get("error", "Test failed")
        
        gl_report["tests"].append(gl_test)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(gl_report, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Reporte GitLab generado: {output_file}")
    return output_file

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generar reportes CI/CD")
    parser.add_argument("--format", choices=["junit", "github", "gitlab", "all"], 
                       default="all", help="Formato de reporte")
    parser.add_argument("--input", default="test_results.json", help="Archivo de resultados")
    
    args = parser.parse_args()
    
    if args.format in ["junit", "all"]:
        generate_junit_xml(args.input)
    
    if args.format in ["github", "all"]:
        generate_github_annotations(args.input)
    
    if args.format in ["gitlab", "all"]:
        generate_gitlab_report(args.input)
































