#!/usr/bin/env python3
"""
Health Check
Verificación completa de salud del sistema de tests
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import json


class HealthChecker:
    """Verificador de salud del sistema de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.checks = []
        self.issues = []
        self.warnings = []
    
    def check_structure(self) -> bool:
        """Verificar estructura de directorios"""
        print("📁 Verificando estructura...")
        
        required_dirs = [
            'core', 'analyzers', 'systems',
            'reporters', 'exporters', 'utilities'
        ]
        
        missing = []
        for dir_name in required_dirs:
            if not (self.base_path / dir_name).exists():
                missing.append(dir_name)
        
        if missing:
            self.issues.append(f"Directorios faltantes: {', '.join(missing)}")
            return False
        
        self.checks.append(('Estructura', '✅ OK'))
        return True
    
    def check_config_files(self) -> bool:
        """Verificar archivos de configuración"""
        print("⚙️  Verificando configuración...")
        
        required_files = [
            'pytest.ini', 'conftest.py',
            'requirements-test.txt', 'Makefile'
        ]
        
        missing = []
        for file_name in required_files:
            if not (self.base_path / file_name).exists():
                missing.append(file_name)
        
        if missing:
            self.warnings.append(f"Archivos de configuración faltantes: {', '.join(missing)}")
            return False
        
        self.checks.append(('Configuración', '✅ OK'))
        return True
    
    def check_dependencies(self) -> bool:
        """Verificar dependencias"""
        print("📦 Verificando dependencias...")
        
        try:
            import pytest
            import coverage
            self.checks.append(('Dependencias', '✅ OK'))
            return True
        except ImportError as e:
            self.issues.append(f"Dependencia faltante: {e.name}")
            return False
    
    def check_scripts(self) -> bool:
        """Verificar scripts principales"""
        print("🔧 Verificando scripts...")
        
        required_scripts = [
            'run_tests.py',
            'validate_structure.py',
            'setup.py'
        ]
        
        missing = []
        for script in required_scripts:
            if not (self.base_path / script).exists():
                missing.append(script)
        
        if missing:
            self.warnings.append(f"Scripts faltantes: {', '.join(missing)}")
            return False
        
        self.checks.append(('Scripts', '✅ OK'))
        return True
    
    def check_test_files(self) -> bool:
        """Verificar archivos de test"""
        print("🧪 Verificando archivos de test...")
        
        test_files = list(self.base_path.rglob('test_*.py'))
        
        if len(test_files) < 10:
            self.warnings.append(f"Pocos archivos de test encontrados: {len(test_files)}")
            return False
        
        self.checks.append(('Archivos de test', f'✅ {len(test_files)} encontrados'))
        return True
    
    def run_quick_test(self) -> bool:
        """Ejecutar test rápido"""
        print("⚡ Ejecutando test rápido...")
        
        try:
            result = subprocess.run(
                ['python', 'run_tests.py', 'unit', '-q'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.checks.append(('Test rápido', '✅ OK'))
                return True
            else:
                self.issues.append("Test rápido falló")
                return False
        except Exception as e:
            self.warnings.append(f"Error ejecutando test rápido: {e}")
            return False
    
    def generate_health_report(self) -> Dict:
        """Generar reporte de salud"""
        print("\n🏥 Ejecutando health check completo...\n")
        
        checks_passed = [
            self.check_structure(),
            self.check_config_files(),
            self.check_dependencies(),
            self.check_scripts(),
            self.check_test_files(),
            self.run_quick_test()
        ]
        
        passed_count = sum(checks_passed)
        total_checks = len(checks_passed)
        
        # Calcular score de salud
        health_score = (passed_count / total_checks * 100) if total_checks > 0 else 0
        
        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 75:
            status = 'good'
        elif health_score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_score': health_score,
            'status': status,
            'checks_passed': passed_count,
            'total_checks': total_checks,
            'checks': self.checks,
            'issues': self.issues,
            'warnings': self.warnings
        }
    
    def print_report(self, report: Dict):
        """Imprimir reporte formateado"""
        print("=" * 60)
        print("🏥 HEALTH CHECK REPORT")
        print("=" * 60)
        print()
        
        print(f"📊 Score de Salud: {report['health_score']:.1f}/100")
        print(f"   Estado: {report['status'].upper()}")
        print(f"   Checks pasados: {report['checks_passed']}/{report['total_checks']}")
        print()
        
        print("✅ Checks:")
        for check_name, result in report['checks']:
            print(f"   {check_name}: {result}")
        print()
        
        if report['issues']:
            print("❌ Problemas:")
            for issue in report['issues']:
                print(f"   - {issue}")
            print()
        
        if report['warnings']:
            print("⚠️  Advertencias:")
            for warning in report['warnings']:
                print(f"   - {warning}")
            print()
        
        # Recomendaciones
        if report['status'] == 'poor':
            print("💡 Recomendaciones:")
            print("   - Resolver problemas críticos")
            print("   - Ejecutar: make setup")
            print("   - Revisar: python validate_structure.py")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Health check del sistema de tests')
    parser.add_argument('--output', type=Path,
                       help='Archivo de salida JSON')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    
    args = parser.parse_args()
    
    checker = HealthChecker(args.base_path)
    report = checker.generate_health_report()
    checker.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Reporte guardado en: {args.output}")
    
    # Exit code basado en salud
    if report['status'] in ['excellent', 'good']:
        return 0
    elif report['status'] == 'fair':
        return 1
    else:
        return 2


if __name__ == '__main__':
    sys.exit(main())

