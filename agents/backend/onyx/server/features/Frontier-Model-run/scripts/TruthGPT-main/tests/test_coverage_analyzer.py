#!/usr/bin/env python3
"""
Analizador de Cobertura
Analiza y reporta cobertura de tests de manera avanzada
"""

import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class CoverageAnalyzer:
    """Analizador de cobertura de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def run_coverage(self, category: str = 'all') -> Dict:
        """Ejecutar análisis de cobertura"""
        print(f"📊 Ejecutando análisis de cobertura para: {category}...")
        
        test_path = self.base_path / 'core' if category == 'all' else self.base_path / 'core' / category
        
        result = subprocess.run(
            ['pytest', str(test_path), '--cov=..', '--cov-report=json', '--cov-report=term', '-q'],
            cwd=self.base_path.parent,
            capture_output=True,
            text=True
        )
        
        coverage_data = {}
        coverage_file = self.base_path.parent / 'coverage.json'
        
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                coverage_data = json.load(f)
        
        return {
            'exit_code': result.returncode,
            'coverage_data': coverage_data,
            'stdout': result.stdout
        }
    
    def analyze_coverage(self, coverage_data: Dict) -> Dict:
        """Analizar datos de cobertura"""
        if not coverage_data:
            return {'error': 'No hay datos de cobertura'}
        
        totals = coverage_data.get('totals', {})
        
        analysis = {
            'overall_coverage': totals.get('percent_covered', 0.0),
            'lines_covered': totals.get('covered_lines', 0),
            'lines_total': totals.get('num_statements', 0),
            'files_covered': totals.get('covered_branches', 0),
            'files_total': totals.get('num_branches', 0),
            'status': self._get_coverage_status(totals.get('percent_covered', 0))
        }
        
        # Análisis por archivo
        files_analysis = []
        for file_path, file_data in coverage_data.get('files', {}).items():
            file_coverage = file_data.get('summary', {}).get('percent_covered', 0)
            if file_coverage < 80:  # Archivos con baja cobertura
                files_analysis.append({
                    'file': file_path,
                    'coverage': file_coverage,
                    'lines_covered': file_data.get('summary', {}).get('covered_lines', 0),
                    'lines_total': file_data.get('summary', {}).get('num_statements', 0)
                })
        
        analysis['low_coverage_files'] = sorted(files_analysis, key=lambda x: x['coverage'])[:10]
        
        return analysis
    
    def _get_coverage_status(self, coverage: float) -> str:
        """Determinar estado de cobertura"""
        if coverage >= 90:
            return 'excellent'
        elif coverage >= 80:
            return 'good'
        elif coverage >= 70:
            return 'fair'
        else:
            return 'poor'
    
    def generate_coverage_report(self, category: str = 'all') -> Dict:
        """Generar reporte de cobertura"""
        print("📊 Generando reporte de cobertura...\n")
        
        coverage_result = self.run_coverage(category)
        analysis = self.analyze_coverage(coverage_result['coverage_data'])
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'category': category,
            'analysis': analysis,
            'recommendations': self._generate_recommendations(analysis)
        }
        
        return report
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generar recomendaciones"""
        recommendations = []
        
        coverage = analysis.get('overall_coverage', 0)
        status = analysis.get('status', 'poor')
        
        if status == 'poor':
            recommendations.append("🔴 Cobertura crítica: Aumentar cobertura de tests urgentemente")
            recommendations.append("   - Agregar tests para código no cubierto")
            recommendations.append("   - Revisar archivos con baja cobertura")
        elif status == 'fair':
            recommendations.append("🟡 Mejorar cobertura: Objetivo 80%+")
            recommendations.append("   - Enfocarse en archivos con <80% cobertura")
        elif status == 'good':
            recommendations.append("🟢 Mantener cobertura: Objetivo 90%+")
        else:
            recommendations.append("✅ Excelente cobertura: Mantener nivel actual")
        
        if analysis.get('low_coverage_files'):
            recommendations.append(f"\n📋 Archivos con baja cobertura ({len(analysis['low_coverage_files'])}):")
            for file_info in analysis['low_coverage_files'][:5]:
                recommendations.append(f"   - {file_info['file']}: {file_info['coverage']:.1f}%")
        
        return recommendations
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("📊 REPORTE DE COBERTURA")
        print("=" * 60)
        print()
        
        analysis = report['analysis']
        print(f"📈 Cobertura General: {analysis['overall_coverage']:.2f}%")
        print(f"   Estado: {analysis['status'].upper()}")
        print(f"   Líneas cubiertas: {analysis['lines_covered']}/{analysis['lines_total']}")
        print()
        
        if report['recommendations']:
            print("💡 Recomendaciones:")
            for rec in report['recommendations']:
                print(f"   {rec}")
            print()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar cobertura de tests')
    parser.add_argument('--category', default='all',
                       help='Categoría a analizar')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    analyzer = CoverageAnalyzer(args.base_path)
    report = analyzer.generate_coverage_report(args.category)
    analyzer.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Reporte guardado en: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

