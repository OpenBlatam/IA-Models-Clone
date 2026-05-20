"""
Test Result Exporter
Exports test results to various formats (JSON, XML, HTML)
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class TestResultExporter:
    """Exports test results to various formats"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def export_json(
        self,
        results: Dict[str, Any],
        output_file: str = "test_results.json"
    ) -> Path:
        """Export results to JSON"""
        output_path = self.project_root / output_file
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': results.get('total_tests', 0),
                'passed': results.get('passed', 0),
                'failed': results.get('failed', 0),
                'errors': results.get('errors', 0),
                'skipped': results.get('skipped', 0),
                'success_rate': results.get('success_rate', 0),
                'execution_time': results.get('execution_time', 0)
            },
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        return output_path
    
    def export_xml(
        self,
        results: Dict[str, Any],
        output_file: str = "test_results.xml"
    ) -> Path:
        """Export results to JUnit XML format"""
        output_path = self.project_root / output_file
        
        # Create root element
        root = ET.Element("testsuites")
        root.set("name", "TruthGPT Tests")
        root.set("tests", str(results.get('total_tests', 0)))
        root.set("failures", str(results.get('failed', 0)))
        root.set("errors", str(results.get('errors', 0)))
        root.set("skipped", str(results.get('skipped', 0)))
        root.set("time", str(results.get('execution_time', 0)))
        root.set("timestamp", datetime.now().isoformat())
        
        # Add testsuite element
        testsuite = ET.SubElement(root, "testsuite")
        testsuite.set("name", "TruthGPT")
        testsuite.set("tests", str(results.get('total_tests', 0)))
        testsuite.set("failures", str(results.get('failed', 0)))
        testsuite.set("errors", str(results.get('errors', 0)))
        testsuite.set("skipped", str(results.get('skipped', 0)))
        testsuite.set("time", str(results.get('execution_time', 0)))
        
        # Add test cases (if available)
        if 'test_cases' in results:
            for test_case in results['test_cases']:
                testcase = ET.SubElement(testsuite, "testcase")
                testcase.set("name", test_case.get('name', 'unknown'))
                testcase.set("classname", test_case.get('classname', 'unknown'))
                testcase.set("time", str(test_case.get('time', 0)))
                
                if test_case.get('status') == 'failed':
                    failure = ET.SubElement(testcase, "failure")
                    failure.set("message", test_case.get('message', 'Test failed'))
                    failure.text = test_case.get('traceback', '')
                elif test_case.get('status') == 'error':
                    error = ET.SubElement(testcase, "error")
                    error.set("message", test_case.get('message', 'Test error'))
                    error.text = test_case.get('traceback', '')
                elif test_case.get('status') == 'skipped':
                    skipped = ET.SubElement(testcase, "skipped")
                    skipped.set("message", test_case.get('message', 'Test skipped'))
        
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ")
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return output_path
    
    def export_html(
        self,
        results: Dict[str, Any],
        output_file: str = "test_results.html"
    ) -> Path:
        """Export results to HTML"""
        output_path = self.project_root / output_file
        
        total = results.get('total_tests', 0)
        passed = results.get('passed', 0)
        failed = results.get('failed', 0)
        errors = results.get('errors', 0)
        skipped = results.get('skipped', 0)
        success_rate = results.get('success_rate', 0)
        execution_time = results.get('execution_time', 0)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TruthGPT Test Results</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .stat-card {{
            background: #f9f9f9;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #4CAF50;
        }}
        .stat-card.failed {{
            border-left-color: #f44336;
        }}
        .stat-card.error {{
            border-left-color: #ff9800;
        }}
        .stat-card.skipped {{
            border-left-color: #2196F3;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }}
        .stat-label {{
            color: #666;
            margin-top: 5px;
        }}
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }}
        .timestamp {{
            color: #999;
            font-size: 0.9em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 TruthGPT Test Results</h1>
        
        <div class="summary">
            <div class="stat-card">
                <div class="stat-value">{total}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{passed}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card failed">
                <div class="stat-value">{failed}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card error">
                <div class="stat-value">{errors}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card skipped">
                <div class="stat-value">{skipped}</div>
                <div class="stat-label">Skipped</div>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" style="width: {success_rate}%">
                {success_rate:.1f}%
            </div>
        </div>
        
        <div style="margin-top: 30px;">
            <h2>Execution Details</h2>
            <p><strong>Execution Time:</strong> {execution_time:.2f} seconds</p>
            <p><strong>Tests per Second:</strong> {(total / execution_time if execution_time > 0 else 0):.1f}</p>
        </div>
        
        <div class="timestamp">
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def export_all(
        self,
        results: Dict[str, Any],
        base_name: str = "test_results"
    ) -> Dict[str, Path]:
        """Export to all formats"""
        return {
            'json': self.export_json(results, f"{base_name}.json"),
            'xml': self.export_xml(results, f"{base_name}.xml"),
            'html': self.export_html(results, f"{base_name}.html")
        }

def main():
    """Main function"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    # Example results
    example_results = {
        'total_tests': 204,
        'passed': 200,
        'failed': 2,
        'errors': 0,
        'skipped': 2,
        'success_rate': 98.0,
        'execution_time': 45.3
    }
    
    exporter = TestResultExporter(project_root)
    exported = exporter.export_all(example_results)
    
    print("✅ Exported test results:")
    for format_name, path in exported.items():
        print(f"  {format_name.upper()}: {path}")

if __name__ == "__main__":
    main()







