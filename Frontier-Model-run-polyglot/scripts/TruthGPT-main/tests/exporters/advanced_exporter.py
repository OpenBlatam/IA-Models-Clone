"""
Advanced Test Result Exporter
Export to multiple formats with advanced options
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import html

class AdvancedExporter:
    """Advanced exporter for test results"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def export_to_csv(
        self,
        test_results: List[Dict],
        output_file: str = "test_results.csv"
    ) -> Path:
        """Export test results to CSV"""
        output_path = self.project_root / output_file
        
        if not test_results:
            return output_path
        
        # Get all unique keys
        all_keys = set()
        for result in test_results:
            all_keys.update(result.keys())
        
        fieldnames = sorted(all_keys)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_results)
        
        return output_path
    
    def export_to_markdown(
        self,
        test_results: Dict,
        output_file: str = "test_results.md"
    ) -> Path:
        """Export test results to Markdown"""
        output_path = self.project_root / output_file
        
        lines = []
        lines.append("# Test Results Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Tests**: {test_results.get('total_tests', 0)}")
        lines.append(f"- **Passed**: {test_results.get('passed', 0)}")
        lines.append(f"- **Failed**: {test_results.get('failures', 0)}")
        lines.append(f"- **Errors**: {test_results.get('errors', 0)}")
        lines.append(f"- **Skipped**: {test_results.get('skipped', 0)}")
        lines.append(f"- **Success Rate**: {test_results.get('success_rate', 0):.1f}%")
        lines.append(f"- **Execution Time**: {test_results.get('execution_time', 0):.2f}s")
        lines.append("")
        
        # Failures
        if test_results.get('failures'):
            lines.append("## Failures")
            lines.append("")
            for test, traceback in test_results.get('failures', []):
                lines.append(f"### {test}")
                lines.append("")
                lines.append("```")
                lines.append(traceback[:500])
                lines.append("```")
                lines.append("")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def export_to_yaml(
        self,
        test_results: Dict,
        output_file: str = "test_results.yaml"
    ) -> Path:
        """Export test results to YAML"""
        try:
            import yaml
        except ImportError:
            print("⚠️  PyYAML not installed. Install with: pip install pyyaml")
            return None
        
        output_path = self.project_root / output_file
        
        yaml_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tests': test_results.get('total_tests', 0),
                'passed': test_results.get('passed', 0),
                'failed': test_results.get('failures', 0),
                'errors': test_results.get('errors', 0),
                'skipped': test_results.get('skipped', 0),
                'success_rate': test_results.get('success_rate', 0),
                'execution_time': test_results.get('execution_time', 0)
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        return output_path
    
    def export_to_excel(
        self,
        test_results: List[Dict],
        output_file: str = "test_results.xlsx"
    ) -> Path:
        """Export test results to Excel"""
        try:
            import openpyxl
        except ImportError:
            print("⚠️  openpyxl not installed. Install with: pip install openpyxl")
            return None
        
        output_path = self.project_root / output_file
        
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Test Results"
        
        if not test_results:
            wb.save(output_path)
            return output_path
        
        # Headers
        headers = list(test_results[0].keys())
        ws.append(headers)
        
        # Data
        for result in test_results:
            row = [result.get(h, '') for h in headers]
            ws.append(row)
        
        wb.save(output_path)
        return output_path
    
    def export_all_formats(
        self,
        test_results: Dict,
        base_name: str = "test_results"
    ) -> Dict[str, Path]:
        """Export to all available formats"""
        exported = {}
        
        # JSON (always available)
        json_path = self.project_root / f"{base_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2)
        exported['json'] = json_path
        
        # Markdown (always available)
        exported['markdown'] = self.export_to_markdown(test_results, f"{base_name}.md")
        
        # CSV (if list of results)
        if isinstance(test_results, list):
            exported['csv'] = self.export_to_csv(test_results, f"{base_name}.csv")
        
        # YAML (if available)
        yaml_path = self.export_to_yaml(test_results, f"{base_name}.yaml")
        if yaml_path:
            exported['yaml'] = yaml_path
        
        # Excel (if available)
        if isinstance(test_results, list):
            excel_path = self.export_to_excel(test_results, f"{base_name}.xlsx")
            if excel_path:
                exported['excel'] = excel_path
        
        return exported

def main():
    """Example usage"""
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    
    exporter = AdvancedExporter(project_root)
    
    test_results = {
        'total_tests': 204,
        'passed': 200,
        'failures': 2,
        'errors': 0,
        'skipped': 2,
        'success_rate': 98.0,
        'execution_time': 45.3
    }
    
    exported = exporter.export_all_formats(test_results)
    
    print("✅ Exported to formats:")
    for format_name, path in exported.items():
        print(f"  {format_name.upper()}: {path}")

if __name__ == "__main__":
    main()







