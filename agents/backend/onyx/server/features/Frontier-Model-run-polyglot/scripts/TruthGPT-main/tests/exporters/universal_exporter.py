"""
Universal Exporter
Export test results to any format
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

class UniversalExporter:
    """Export test results to any format"""
    
    SUPPORTED_FORMATS = [
        'json', 'xml', 'html', 'csv', 'markdown', 
        'yaml', 'excel', 'pdf', 'junit', 'txt'
    ]
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results_dir = project_root / "test_results"
    
    def export(
        self,
        result_file: str,
        formats: List[str],
        output_dir: Optional[str] = None
    ) -> Dict[str, Path]:
        """Export to multiple formats"""
        result_path = self.results_dir / result_file
        
        if not result_path.exists():
            return {'error': f'File not found: {result_file}'}
        
        try:
            with open(result_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return {'error': f'Error reading file: {e}'}
        
        if output_dir is None:
            output_dir = self.results_dir / "exports"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exported = {}
        base_name = Path(result_file).stem
        
        for format_type in formats:
            if format_type not in self.SUPPORTED_FORMATS:
                continue
            
            try:
                output_path = self._export_to_format(data, format_type, output_dir, base_name)
                if output_path:
                    exported[format_type] = output_path
            except Exception as e:
                print(f"⚠️  Error exporting to {format_type}: {e}")
        
        return exported
    
    def _export_to_format(
        self,
        data: Dict,
        format_type: str,
        output_dir: Path,
        base_name: str
    ) -> Optional[Path]:
        """Export to specific format"""
        if format_type == 'json':
            output_path = output_dir / f"{base_name}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return output_path
        
        elif format_type == 'txt':
            output_path = output_dir / f"{base_name}.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Test Results Summary\n")
                f.write(f"Total Tests: {data.get('total_tests', 0)}\n")
                f.write(f"Passed: {data.get('passed', 0)}\n")
                f.write(f"Failures: {data.get('failures', 0)}\n")
                f.write(f"Success Rate: {data.get('success_rate', 0):.1f}%\n")
            return output_path
        
        elif format_type == 'markdown':
            output_path = output_dir / f"{base_name}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# Test Results\n\n")
                f.write(f"- Total Tests: {data.get('total_tests', 0)}\n")
                f.write(f"- Passed: {data.get('passed', 0)}\n")
                f.write(f"- Failures: {data.get('failures', 0)}\n")
                f.write(f"- Success Rate: {data.get('success_rate', 0):.1f}%\n")
            return output_path
        
        # For other formats, delegate to existing exporters
        elif format_type == 'junit':
            from tests.junit_exporter import JUnitExporter
            exporter = JUnitExporter(self.project_root)
            return exporter.export_to_junit_xml(data, f"{base_name}.xml")
        
        elif format_type == 'csv':
            from tests.advanced_exporter import AdvancedExporter
            exporter = AdvancedExporter(self.project_root)
            return exporter.export_to_csv([data], f"{base_name}.csv")
        
        elif format_type == 'html':
            from tests.html_report_generator import HTMLReportGenerator
            generator = HTMLReportGenerator(self.project_root)
            return generator.generate_html_report(data, f"{base_name}.html")
        
        elif format_type == 'pdf':
            from tests.pdf_exporter import PDFExporter
            exporter = PDFExporter(self.project_root)
            return exporter.export_to_pdf(data, f"{base_name}.pdf")
        
        return None
    
    def list_supported_formats(self) -> List[str]:
        """List all supported export formats"""
        return self.SUPPORTED_FORMATS

def main():
    """Example usage"""
    from pathlib import Path
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python universal_exporter.py <file.json> <format1> [format2] ...")
        print(f"Supported formats: {', '.join(UniversalExporter.SUPPORTED_FORMATS)}")
        return
    
    project_root = Path(__file__).parent.parent
    exporter = UniversalExporter(project_root)
    
    result_file = sys.argv[1]
    formats = sys.argv[2:]
    
    exported = exporter.export(result_file, formats)
    
    if 'error' in exported:
        print(f"❌ {exported['error']}")
    else:
        print(f"✅ Exported to {len(exported)} formats:")
        for format_type, path in exported.items():
            print(f"  {format_type.upper()}: {path}")

if __name__ == "__main__":
    main()







