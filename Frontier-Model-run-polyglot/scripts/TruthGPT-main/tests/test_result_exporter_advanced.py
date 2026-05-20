"""
Advanced Test Result Exporter
Export results to multiple formats with customization
"""

import json
import csv
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml


class AdvancedTestResultExporter:
    """Export test results to various formats"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
        self.exports_dir = project_root / "exports"
        self.exports_dir.mkdir(exist_ok=True)
    
    def export_to_json(
        self,
        results: Dict,
        output_file: Path = None,
        pretty: bool = True
    ) -> Path:
        """Export to JSON format"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.exports_dir / f"results_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            if pretty:
                json.dump(results, f, indent=2, ensure_ascii=False)
            else:
                json.dump(results, f, ensure_ascii=False)
        
        print(f"✅ Exported to JSON: {output_file}")
        return output_file
    
    def export_to_csv(
        self,
        results: Dict,
        output_file: Path = None
    ) -> Path:
        """Export to CSV format"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.exports_dir / f"results_{timestamp}.csv"
        
        test_details = results.get('test_details', {})
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Test Name', 'Status', 'Duration', 'Error Message', 'Timestamp'
            ])
            
            # Data
            for test_name, test_data in test_details.items():
                writer.writerow([
                    test_name,
                    test_data.get('status', 'unknown'),
                    test_data.get('duration', 0),
                    test_data.get('error_message', '')[:100],  # Truncate
                    results.get('timestamp', '')
                ])
        
        print(f"✅ Exported to CSV: {output_file}")
        return output_file
    
    def export_to_xml(
        self,
        results: Dict,
        output_file: Path = None
    ) -> Path:
        """Export to XML format"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.exports_dir / f"results_{timestamp}.xml"
        
        root = ET.Element('testresults')
        root.set('timestamp', results.get('timestamp', ''))
        root.set('run_name', results.get('run_name', ''))
        
        # Summary
        summary = ET.SubElement(root, 'summary')
        summary_elem = results.get('summary', {})
        for key, value in summary_elem.items():
            summary.set(key, str(value))
        
        # Test details
        tests = ET.SubElement(root, 'tests')
        test_details = results.get('test_details', {})
        
        for test_name, test_data in test_details.items():
            test_elem = ET.SubElement(tests, 'test')
            test_elem.set('name', test_name)
            test_elem.set('status', test_data.get('status', 'unknown'))
            test_elem.set('duration', str(test_data.get('duration', 0)))
            
            if test_data.get('error_message'):
                error_elem = ET.SubElement(test_elem, 'error')
                error_elem.text = test_data.get('error_message', '')
        
        # Write XML
        tree = ET.ElementTree(root)
        ET.indent(tree, space='  ')
        tree.write(output_file, encoding='utf-8', xml_declaration=True)
        
        print(f"✅ Exported to XML: {output_file}")
        return output_file
    
    def export_to_yaml(
        self,
        results: Dict,
        output_file: Path = None
    ) -> Path:
        """Export to YAML format"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.exports_dir / f"results_{timestamp}.yaml"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
        
        print(f"✅ Exported to YAML: {output_file}")
        return output_file
    
    def export_to_html_table(
        self,
        results: Dict,
        output_file: Path = None
    ) -> Path:
        """Export to HTML table format"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.exports_dir / f"results_{timestamp}.html"
        
        test_details = results.get('test_details', {})
        summary = results.get('summary', {})
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Results - {results.get('run_name', 'Unknown')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .error {{ color: orange; }}
        .summary {{ margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>Test Results</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Run Name:</strong> {results.get('run_name', 'Unknown')}</p>
        <p><strong>Timestamp:</strong> {results.get('timestamp', 'Unknown')}</p>
        <p><strong>Total Tests:</strong> {summary.get('total_tests', 0)}</p>
        <p><strong>Passed:</strong> {summary.get('passed', 0)}</p>
        <p><strong>Failed:</strong> {summary.get('failed', 0)}</p>
        <p><strong>Success Rate:</strong> {summary.get('success_rate', 0):.1f}%</p>
    </div>
    
    <h2>Test Details</h2>
    <table>
        <tr>
            <th>Test Name</th>
            <th>Status</th>
            <th>Duration (s)</th>
            <th>Error Message</th>
        </tr>
"""
        
        for test_name, test_data in test_details.items():
            status = test_data.get('status', 'unknown')
            status_class = status
            html += f"""
        <tr>
            <td>{test_name}</td>
            <td class="{status_class}">{status}</td>
            <td>{test_data.get('duration', 0):.2f}</td>
            <td>{test_data.get('error_message', '')[:200]}</td>
        </tr>
"""
        
        html += """
    </table>
</body>
</html>
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✅ Exported to HTML: {output_file}")
        return output_file
    
    def export_to_markdown(
        self,
        results: Dict,
        output_file: Path = None
    ) -> Path:
        """Export to Markdown format"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.exports_dir / f"results_{timestamp}.md"
        
        summary = results.get('summary', {})
        test_details = results.get('test_details', {})
        
        md = f"""# Test Results

**Run Name:** {results.get('run_name', 'Unknown')}  
**Timestamp:** {results.get('timestamp', 'Unknown')}

## Summary

- **Total Tests:** {summary.get('total_tests', 0)}
- **Passed:** {summary.get('passed', 0)}
- **Failed:** {summary.get('failed', 0)}
- **Errors:** {summary.get('errors', 0)}
- **Skipped:** {summary.get('skipped', 0)}
- **Success Rate:** {summary.get('success_rate', 0):.1f}%
- **Execution Time:** {summary.get('execution_time', 0):.2f}s

## Test Details

| Test Name | Status | Duration (s) | Error Message |
|-----------|--------|--------------|---------------|
"""
        
        for test_name, test_data in test_details.items():
            status = test_data.get('status', 'unknown')
            duration = test_data.get('duration', 0)
            error = test_data.get('error_message', '').replace('|', '\\|')[:100]
            md += f"| {test_name} | {status} | {duration:.2f} | {error} |\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"✅ Exported to Markdown: {output_file}")
        return output_file
    
    def export_multi_format(
        self,
        results: Dict,
        formats: List[str] = None
    ) -> Dict[str, Path]:
        """Export to multiple formats at once"""
        if formats is None:
            formats = ['json', 'csv', 'html', 'markdown']
        
        exported = {}
        
        for fmt in formats:
            try:
                if fmt == 'json':
                    exported['json'] = self.export_to_json(results)
                elif fmt == 'csv':
                    exported['csv'] = self.export_to_csv(results)
                elif fmt == 'xml':
                    exported['xml'] = self.export_to_xml(results)
                elif fmt == 'yaml':
                    exported['yaml'] = self.export_to_yaml(results)
                elif fmt == 'html':
                    exported['html'] = self.export_to_html_table(results)
                elif fmt == 'markdown':
                    exported['markdown'] = self.export_to_markdown(results)
            except Exception as e:
                print(f"Error exporting to {fmt}: {e}")
        
        return exported


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Test Result Exporter')
    parser.add_argument('--results', type=str, required=True, help='Results file to export')
    parser.add_argument('--format', choices=['json', 'csv', 'xml', 'yaml', 'html', 'markdown', 'all'],
                       default='all', help='Export format')
    parser.add_argument('--output', type=str, help='Output file (for single format)')
    parser.add_argument('--project-root', type=str, help='Project root directory')
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root) if args.project_root else Path(__file__).parent
    
    exporter = AdvancedTestResultExporter(project_root)
    
    # Load results
    with open(args.results, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    if args.format == 'all':
        print("📤 Exporting to all formats...")
        exported = exporter.export_multi_format(results)
        print(f"\n✅ Exported to {len(exported)} formats:")
        for fmt, path in exported.items():
            print(f"  {fmt}: {path}")
    else:
        output_path = Path(args.output) if args.output else None
        
        if args.format == 'json':
            exporter.export_to_json(results, output_path)
        elif args.format == 'csv':
            exporter.export_to_csv(results, output_path)
        elif args.format == 'xml':
            exporter.export_to_xml(results, output_path)
        elif args.format == 'yaml':
            exporter.export_to_yaml(results, output_path)
        elif args.format == 'html':
            exporter.export_to_html_table(results, output_path)
        elif args.format == 'markdown':
            exporter.export_to_markdown(results, output_path)


if __name__ == '__main__':
    main()

