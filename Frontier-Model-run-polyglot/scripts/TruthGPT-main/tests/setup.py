#!/usr/bin/env python3
"""
Setup Script
Configuración inicial y verificación del entorno de tests
"""

import sys
import subprocess
from pathlib import Path
from typing import List, Tuple


class TestSetup:
    """Configurador del entorno de tests"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.issues = []
        self.warnings = []
    
    def check_python_version(self) -> bool:
        """Verificar versión de Python"""
        print("🐍 Verificando versión de Python...")
        
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            self.issues.append(f"Python 3.8+ requerido, encontrado: {version.major}.{version.minor}")
            return False
        
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_dependencies(self) -> bool:
        """Verificar dependencias"""
        print("📦 Verificando dependencias...")
        
        required = ['pytest', 'coverage']
        missing = []
        
        for package in required:
            try:
                __import__(package)
                print(f"   ✅ {package}")
            except ImportError:
                missing.append(package)
                print(f"   ❌ {package} - FALTANTE")
        
        if missing:
            self.issues.append(f"Dependencias faltantes: {', '.join(missing)}")
            print(f"\n   💡 Instalar con: pip install -r requirements-test.txt")
            return False
        
        return True
    
    def check_structure(self) -> bool:
        """Verificar estructura de directorios"""
        print("📁 Verificando estructura...")
        
        required_dirs = [
            'core',
            'analyzers',
            'systems',
            'reporters',
            'exporters',
            'utilities'
        ]
        
        missing = []
        for dir_name in required_dirs:
            dir_path = self.base_path / dir_name
            if dir_path.exists():
                print(f"   ✅ {dir_name}/")
            else:
                missing.append(dir_name)
                print(f"   ❌ {dir_name}/ - FALTANTE")
        
        if missing:
            self.issues.append(f"Directorios faltantes: {', '.join(missing)}")
            return False
        
        return True
    
    def check_config_files(self) -> bool:
        """Verificar archivos de configuración"""
        print("⚙️  Verificando archivos de configuración...")
        
        required_files = [
            'pytest.ini',
            'conftest.py',
            'requirements-test.txt',
            'Makefile'
        ]
        
        missing = []
        for file_name in required_files:
            file_path = self.base_path / file_name
            if file_path.exists():
                print(f"   ✅ {file_name}")
            else:
                missing.append(file_name)
                print(f"   ❌ {file_name} - FALTANTE")
        
        if missing:
            self.warnings.append(f"Archivos de configuración faltantes: {', '.join(missing)}")
            return False
        
        return True
    
    def setup_config(self) -> bool:
        """Configurar archivos de configuración"""
        print("⚙️  Configurando archivos...")
        
        # Crear directorio de config si no existe
        config_dir = self.base_path / 'config'
        config_dir.mkdir(exist_ok=True)
        
        # Copiar template de notificaciones si no existe
        notifications_file = config_dir / 'notifications.json'
        notifications_example = config_dir / 'notifications.json.example'
        
        if not notifications_file.exists() and notifications_example.exists():
            import shutil
            shutil.copy(notifications_example, notifications_file)
            print(f"   ✅ Creado {notifications_file}")
            self.warnings.append(f"Configurar {notifications_file} con tus credenciales")
        
        return True
    
    def run_validation(self) -> bool:
        """Ejecutar validación de estructura"""
        print("🔍 Ejecutando validación...")
        
        try:
            result = subprocess.run(
                ['python', 'validate_structure.py'],
                cwd=self.base_path,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("   ✅ Validación pasada")
                return True
            else:
                self.warnings.append("Validación encontró algunos problemas")
                print("   ⚠️  Validación con advertencias")
                return True  # No crítico
        except Exception as e:
            self.warnings.append(f"Error ejecutando validación: {e}")
            return True  # No crítico
    
    def print_summary(self):
        """Imprimir resumen"""
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE CONFIGURACIÓN")
        print("=" * 60)
        
        if self.issues:
            print("\n❌ PROBLEMAS CRÍTICOS:")
            for issue in self.issues:
                print(f"   - {issue}")
        
        if self.warnings:
            print("\n⚠️  ADVERTENCIAS:")
            for warning in self.warnings:
                print(f"   - {warning}")
        
        if not self.issues and not self.warnings:
            print("\n✅ Todo configurado correctamente!")
        elif not self.issues:
            print("\n✅ Configuración básica OK (revisar advertencias)")
        else:
            print("\n❌ Resolver problemas críticos antes de continuar")
        
        print("\n💡 Próximos pasos:")
        print("   1. Revisar QUICK_START.md")
        print("   2. Ejecutar: make test")
        print("   3. Configurar notificaciones (opcional)")
    
    def setup(self) -> bool:
        """Ejecutar setup completo"""
        print("🚀 Configurando TruthGPT Test Suite...\n")
        
        checks = [
            self.check_python_version(),
            self.check_dependencies(),
            self.check_structure(),
            self.check_config_files(),
            self.setup_config(),
            self.run_validation()
        ]
        
        # Setup exitoso si no hay problemas críticos
        success = not any(self.issues)
        
        self.print_summary()
        
        return success


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configurar entorno de tests')
    parser.add_argument('--base-path', type=Path, default=Path.cwd(),
                       help='Ruta base de tests')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Saltar validación')
    
    args = parser.parse_args()
    
    setup = TestSetup(args.base_path)
    
    if args.skip_validation:
        setup.run_validation = lambda: True
    
    success = setup.setup()
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

