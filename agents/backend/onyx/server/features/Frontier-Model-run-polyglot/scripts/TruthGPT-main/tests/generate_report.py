#!/usr/bin/env python3
"""
Generador de Reportes Avanzado
Genera reportes completos de tests con visualizaciones
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import xml.etree.ElementTree as ET


class ReportGenerator:
    """Generador de reportes"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.report_data = {}
    
    def run_tests_with_coverage(self) -> Dict:
        """Ejecutar tests con cobertura"""
        print("🧪 Ejecutando tests con cobertura...")
        
        result = subprocess.run(
            ['pytest', 'core/', '--cov=..', '--cov-report=xml', '--cov-report=json', '-v'],
            cwd=self.base_path,
            capture_output=True,
            text=True
        )
        
        coverage_data = {}
        if (self.base_path / 'coverage.json').exists():
            with open(self.base_path / 'coverage.json') as f:
                coverage_data = json.load(f)
        
        return {
            'exit_code': result.returncode,
            'output': result.stdout,
            'coverage': coverage_data
        }
    
    def parse_junit_xml(self, xml_path: Path) -> Dict:
        """Parsear XML de JUnit"""
        if not xml_path.exists():
            return {}
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        stats = {
            'tests': 0,
            'failures': 0,
            'errors': 0,
            'skipped': 0,
            'time': 0.0,
            'testcases': []
        }
        
        for testsuite in root.findall('testsuite'):
            stats['tests'] += int(testsuite.get('tests', 0))
            stats['failures'] += int(testsuite.get('failures', 0))
            stats['errors'] += int(testsuite.get('errors', 0))
            stats['skipped'] += int(testsuite.get('skipped', 0))
            stats['time'] += float(testsuite.get('time', 0))
            
            for testcase in testsuite.findall('testcase'):
                stats['testcases'].append({
                    'name': testcase.get('name'),
                    'classname': testcase.get('classname'),
                    'time': float(testcase.get('time', 0)),
                    'status': 'passed' if len(testcase) == 0 else 'failed'
                })
        
        return stats
    
    def generate_html_report(self, data: Dict) -> str:
        """Generar reporte HTML"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TruthGPT Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
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
            margin: 20px 0;
        }}
        .card {{
            background: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .card h3 {{
            margin: 0 0 10px 0;
            color: #666;
            font-size: 14px;
        }}
        .card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .success {{ border-left-color: #4CAF50; }}
        .warning {{ border-left-color: #FF9800; }}
        .error {{ border-left-color: #F44336; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }}
        .badge-success {{ background-color: #4CAF50; color: white; }}
        .badge-danger {{ background-color: #F44336; color: white; }}
        .badge-warning {{ background-color: #FF9800; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 TruthGPT Test Report</h1>
        <p>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <div class="summary">
            <div class="card success">
                <h3>Total Tests</h3>
                <div class="value">{data.get('total_tests', 0)}</div>
            </div>
            <div class="card success">
                <h3>Passed</h3>
                <div class="value">{data.get('passed', 0)}</div>
            </div>
            <div class="card error">
                <h3>Failed</h3>
                <div class="value">{data.get('failed', 0)}</div>
            </div>
            <div class="card warning">
                <h3>Skipped</h3>
                <div class="value">{data.get('skipped', 0)}</div>
            </div>
        </div>
        
        <h2>Coverage</h2>
        <div class="card">
            <h3>Coverage: {data.get('coverage_percent', 0):.2f}%</h3>
        </div>
        
        <h2>Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Test</th>
                    <th>Status</th>
                    <th>Time</th>
                </tr>
            </thead>
            <tbody>
"""
        
        for test in data.get('testcases', [])[:50]:  # Mostrar primeros 50
            status_class = 'success' if test.get('status') == 'passed' else 'error'
            status_badge = '✅ Passed' if test.get('status') == 'passed' else '❌ Failed'
            
            html += f"""
                <tr>
                    <td>{test.get('name', 'N/A')}</td>
                    <td><span class="badge badge-{status_class}">{status_badge}</span></td>
                    <td>{test.get('time', 0):.3f}s</td>
                </tr>
"""
        
        html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        return html
    
    def generate_report(self) -> Dict:
        """Generar reporte completo"""
        print("📊 Generando reporte completo...\n")
        
        # Ejecutar tests
        test_results = self.run_tests_with_coverage()
        
        # Parsear resultados
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'coverage_percent': 0.0,
            'testcases': []
        }
        
        # Extraer información de cobertura
        if test_results.get('coverage'):
            totals = test_results['coverage'].get('totals', {})
            if totals:
                report['coverage_percent'] = totals.get('percent_covered', 0.0)
        
        # Generar HTML
        html_content = self.generate_html_report(report)
        
        return {
            'json': report,
            'html': html_content
        }
    
    def save_report(self, report: Dict, output_dir: Path):
        """Guardar reporte"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Guardar JSON
        json_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, 'w') as f:
            json.dump(report['json'], f, indent=2)
        
        # Guardar HTML
        html_path = output_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(html_path, 'w') as f:
            f.write(report['html'])
        
        print(f"✅ Reporte JSON guardado: {json_path}")
        print(f"✅ Reporte HTML guardado: {html_path}")
        
        return json_path, html_path


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generar reporte de tests')
    parser.add_argument('--output', type=Path, default=Path('test-results'),
                       help='Directorio de salida')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    generator = ReportGenerator(args.base_path)
    report = generator.generate_report()
    generator.save_report(report, args.output)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

