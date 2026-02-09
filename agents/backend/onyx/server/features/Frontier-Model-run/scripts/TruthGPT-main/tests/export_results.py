#!/usr/bin/env python3
"""
Exportador de Resultados
Exporta resultados de tests a múltiples formatos
"""

import sys
import json
import csv
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import xml.etree.ElementTree as ET


class ResultsExporter:
    """Exportador de resultados"""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Cargar datos"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {self.data_path}")
        
        with open(self.data_path, 'r') as f:
            return json.load(f)
    
    def export_json(self, output_path: Path) -> Path:
        """Exportar a JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.data, f, indent=2)
        return output_path
    
    def export_csv(self, output_path: Path) -> Path:
        """Exportar a CSV"""
        stats = self.data.get('stats', {})
        history = stats.get('history', [])
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'success', 'elapsed', 'exit_code'])
            
            for run in history:
                writer.writerow([
                    run.get('timestamp', ''),
                    run.get('success', False),
                    run.get('elapsed', 0),
                    run.get('exit_code', 0)
                ])
        
        return output_path
    
    def export_xml(self, output_path: Path) -> Path:
        """Exportar a XML (JUnit format)"""
        root = ET.Element('testsuites')
        root.set('name', 'TruthGPT Tests')
        root.set('tests', str(self.data.get('stats', {}).get('total_runs', 0)))
        
        stats = self.data.get('stats', {})
        history = stats.get('history', [])
        
        testsuite = ET.SubElement(root, 'testsuite')
        testsuite.set('name', 'All Tests')
        testsuite.set('tests', str(len(history)))
        testsuite.set('failures', str(stats.get('failed', 0)))
        
        for run in history:
            testcase = ET.SubElement(testsuite, 'testcase')
            testcase.set('name', f"run_{run.get('timestamp', '')}")
            testcase.set('time', str(run.get('elapsed', 0)))
            
            if not run.get('success', False):
                failure = ET.SubElement(testcase, 'failure')
                failure.set('message', 'Test failed')
                failure.text = run.get('stderr', '')
        
        tree = ET.ElementTree(root)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        return output_path
    
    def export_markdown(self, output_path: Path) -> Path:
        """Exportar a Markdown"""
        stats = self.data.get('stats', {})
        history = stats.get('history', [])
        
        md = f"""# Test Results Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Runs**: {stats.get('total_runs', 0)}
- **Successful**: {stats.get('successful', 0)}
- **Failed**: {stats.get('failed', 0)}
- **Success Rate**: {(stats.get('successful', 0) / stats.get('total_runs', 1) * 100):.1f}%

## Recent Runs

| Timestamp | Status | Elapsed Time |
|-----------|--------|--------------|
"""
        
        for run in history[-20:]:
            status = "✅ Pass" if run.get('success') else "❌ Fail"
            md += f"| {run.get('timestamp', '')[:19]} | {status} | {run.get('elapsed', 0):.2f}s |\n"
        
        with open(output_path, 'w') as f:
            f.write(md)
        
        return output_path
    
    def export_all(self, output_dir: Path):
        """Exportar a todos los formatos"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        formats = {
            'json': self.export_json,
            'csv': self.export_csv,
            'xml': self.export_xml,
            'md': self.export_markdown
        }
        
        exported = {}
        for fmt, exporter in formats.items():
            output_path = output_dir / f"results_{timestamp}.{fmt}"
            exported[fmt] = exporter(output_path)
            print(f"✅ Exportado {fmt.upper()}: {output_path}")
        
        return exported


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Exportar resultados de tests')
    parser.add_argument('--input', type=Path, required=True,
                       help='Archivo JSON de entrada')
    parser.add_argument('--format', choices=['json', 'csv', 'xml', 'md', 'all'],
                       default='all', help='Formato de exportación')
    parser.add_argument('--output', type=Path,
                       help='Archivo o directorio de salida')
    
    args = parser.parse_args()
    
    exporter = ResultsExporter(args.input)
    
    if args.format == 'all':
        output_dir = args.output or Path('exports')
        exporter.export_all(output_dir)
    else:
        if not args.output:
            args.output = Path(f"results.{args.format}")
        
        exporters = {
            'json': exporter.export_json,
            'csv': exporter.export_csv,
            'xml': exporter.export_xml,
            'md': exporter.export_markdown
        }
        
        exporters[args.format](args.output)
        print(f"✅ Exportado: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

