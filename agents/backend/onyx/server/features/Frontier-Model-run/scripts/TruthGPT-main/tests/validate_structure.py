#!/usr/bin/env python3
"""
Script de Validación de Estructura
Valida que la estructura de tests esté correctamente organizada
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Rutas esperadas
EXPECTED_STRUCTURE = {
    'core': ['unit', 'integration', 'fixtures'],
    'analyzers': [
        'general', 'cost', 'performance', 'quality', 'security',
        'compliance', 'coverage', 'dependency', 'trend',
        'flakiness', 'regression', 'optimization'
    ],
    'utilities': ['integration', 'results'],
}

REQUIRED_FILES = [
    'README.md',
    'INDEX.md',
    'run_tests.py',
    'pytest.ini',
    'conftest.py',
    '__init__.py',
]

def check_directory_structure(base_path: Path) -> Tuple[bool, List[str]]:
    """Verifica la estructura de directorios"""
    errors = []
    base_path = Path(base_path)
    
    # Verificar carpetas principales
    for main_dir, subdirs in EXPECTED_STRUCTURE.items():
        main_path = base_path / main_dir
        if not main_path.exists():
            errors.append(f"❌ Falta directorio: {main_dir}/")
            continue
        
        # Verificar subdirectorios
        for subdir in subdirs:
            sub_path = main_path / subdir
            if not sub_path.exists():
                errors.append(f"❌ Falta subdirectorio: {main_dir}/{subdir}/")
            else:
                # Verificar __init__.py
                init_file = sub_path / '__init__.py'
                if not init_file.exists():
                    errors.append(f"⚠️  Falta __init__.py en: {main_dir}/{subdir}/")
    
    return len(errors) == 0, errors

def check_required_files(base_path: Path) -> Tuple[bool, List[str]]:
    """Verifica archivos requeridos"""
    errors = []
    base_path = Path(base_path)
    
    for file in REQUIRED_FILES:
        file_path = base_path / file
        if not file_path.exists():
            errors.append(f"❌ Falta archivo requerido: {file}")
    
    return len(errors) == 0, errors

def check_python_files(base_path: Path) -> Tuple[bool, List[str]]:
    """Verifica que los archivos Python tengan estructura correcta"""
    errors = []
    base_path = Path(base_path)
    
    # Verificar que no haya archivos sueltos en la raíz de analyzers
    analyzers_path = base_path / 'analyzers'
    if analyzers_path.exists():
        for item in analyzers_path.iterdir():
            if item.is_file() and item.suffix == '.py' and item.name != '__init__.py':
                # Verificar que no esté en la raíz (debe estar en subcarpeta)
                if item.parent == analyzers_path:
                    errors.append(f"⚠️  Archivo Python suelto en analyzers/: {item.name}")
    
    return len(errors) == 0, errors

def check_test_files(base_path: Path) -> Tuple[bool, List[str]]:
    """Verifica que los archivos de test estén en lugares correctos"""
    errors = []
    base_path = Path(base_path)
    
    # Tests deben estar en core/
    core_path = base_path / 'core'
    if core_path.exists():
        # Buscar tests fuera de core/
        for test_file in base_path.rglob('test_*.py'):
            if 'core' not in str(test_file.relative_to(base_path)):
                errors.append(f"⚠️  Test fuera de core/: {test_file.relative_to(base_path)}")
    
    return len(errors) == 0, errors

def main():
    """Función principal"""
    base_path = Path(__file__).parent
    
    print("🔍 Validando estructura de tests...\n")
    
    all_valid = True
    all_errors = []
    
    # Verificar estructura de directorios
    print("📁 Verificando estructura de directorios...")
    valid, errors = check_directory_structure(base_path)
    if errors:
        all_errors.extend(errors)
        all_valid = False
        for error in errors:
            print(f"  {error}")
    else:
        print("  ✅ Estructura de directorios correcta")
    
    # Verificar archivos requeridos
    print("\n📄 Verificando archivos requeridos...")
    valid, errors = check_required_files(base_path)
    if errors:
        all_errors.extend(errors)
        all_valid = False
        for error in errors:
            print(f"  {error}")
    else:
        print("  ✅ Archivos requeridos presentes")
    
    # Verificar archivos Python
    print("\n🐍 Verificando archivos Python...")
    valid, errors = check_python_files(base_path)
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  {error}")
    else:
        print("  ✅ Archivos Python bien organizados")
    
    # Verificar archivos de test
    print("\n🧪 Verificando archivos de test...")
    valid, errors = check_test_files(base_path)
    if errors:
        all_errors.extend(errors)
        for error in errors:
            print(f"  {error}")
    else:
        print("  ✅ Archivos de test bien organizados")
    
    # Resumen
    print("\n" + "="*50)
    if all_valid and len(all_errors) == 0:
        print("✅ Validación completada: Todo correcto!")
        return 0
    else:
        print(f"⚠️  Validación completada con {len(all_errors)} advertencia(s)")
        if all_errors:
            print("\nErrores encontrados:")
            for error in all_errors:
                print(f"  {error}")
        return 1

if __name__ == '__main__':
    sys.exit(main())

