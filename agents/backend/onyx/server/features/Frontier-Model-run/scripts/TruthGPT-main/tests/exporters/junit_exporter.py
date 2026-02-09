"""
JUnit XML Exporter
Export test results to JUnit XML format
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
from datetime import datetime

class JUnitExporter:
    """Export test results to JUnit XML format"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def export_to_junit_xml(
        self,
        test_results: Dict,
        output_file: str = "test_results.xml"
    ) -> Path:
        """Export test results to JUnit XML format"""
        output_path = self.project_root / output_file
        
        # Create root element
        testsuites = ET.Element('testsuites')
        testsuites.set('name', 'TruthGPT Tests')
        testsuites.set('tests', str(test_results.get('total_tests', 0)))
        testsuites.set('failures', str(test_results.get('failures', 0)))
        testsuites.set('errors', str(test_results.get('errors', 0)))
        testsuites.set('time', str(test_results.get('execution_time', 0)))
        
        # Create testsuite
        testsuite = ET.SubElement(testsuites, 'testsuite')
        testsuite.set('name', 'TruthGPT Test Suite')
        testsuite.set('tests', str(test_results.get('total_tests', 0)))
        testsuite.set('failures', str(test_results.get('failures', 0)))
        testsuite.set('errors', str(test_results.get('errors', 0)))
        testsuite.set('skipped', str(test_results.get('skipped', 0)))
        testsuite.set('time', str(test_results.get('execution_time', 0)))
        testsuite.set('timestamp', datetime.now().isoformat())
        
        # Add test cases
        test_details = test_results.get('test_details', {})
        
        # Passed tests
        passed = test_results.get('total_tests', 0) - test_results.get('failures', 0) - test_results.get('errors', 0) - test_results.get('skipped', 0)
        for i in range(passed):
            testcase = ET.SubElement(testsuite, 'testcase')
            testcase.set('name', f'test_passed_{i+1}')
            testcase.set('classname', 'TruthGPT')
            testcase.set('time', '0.1')
        
        # Failed tests
        for failure in test_details.get('failures', []):
            testcase = ET.SubElement(testsuite, 'testcase')
            testcase.set('name', str(failure.get('test', 'unknown')))
            testcase.set('classname', 'TruthGPT')
            testcase.set('time', '0.1')
            
            failure_elem = ET.SubElement(testcase, 'failure')
            failure_elem.set('message', str(failure.get('message', '')))
            failure_elem.text = str(failure.get('traceback', ''))
        
        # Error tests
        for error in test_details.get('errors', []):
            testcase = ET.SubElement(testsuite, 'testcase')
            testcase.set('name', str(error.get('test', 'unknown')))
            testcase.set('classname', 'TruthGPT')
            testcase.set('time', '0.1')
            
            error_elem = ET.SubElement(testcase, 'error')
            error_elem.set('message', str(error.get('message', '')))
            error_elem.text = str(error.get('traceback', ''))
        
        # Skipped tests
        for skipped in test_details.get('skipped', []):
            testcase = ET.SubElement(testsuite, 'testcase')
            testcase.set('name', str(skipped.get('test', 'unknown')))
            testcase.set('classname', 'TruthGPT')
            testcase.set('time', '0.1')
            
            ET.SubElement(testcase, 'skipped')
        
        # Write XML
        tree = ET.ElementTree(testsuites)
        ET.indent(tree, space='  ')
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        return output_path

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    exporter = JUnitExporter(project_root)
    
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failures': 2,
        'errors': 0,
        'skipped': 2,
        'execution_time': 45.3,
        'test_details': {
            'failures': [
                {'test': 'test_example', 'message': 'Assertion failed', 'traceback': 'Traceback...'}
            ]
        }
    }
    
    xml_path = exporter.export_to_junit_xml(test_results)
    print(f"✅ JUnit XML exported to: {xml_path}")

if __name__ == "__main__":
    main()







