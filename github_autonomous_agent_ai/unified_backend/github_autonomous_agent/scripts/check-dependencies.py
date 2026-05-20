#!/usr/bin/env python3
"""
Script para verificar dependencias instaladas y detectar problemas.
"""

import sys
import subprocess
import importlib
from typing import List, Tuple, Dict
from pathlib import Path

# Colores para terminal
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Verifica si un paquete está instalado."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError:
        return False, None

def read_requirements(file_path: Path) -> List[str]:
    """Lee un archivo requirements.txt y extrae nombres de paquetes."""
    packages = []
    if not file_path.exists():
        return packages
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignorar comentarios y líneas vacías
            if not line or line.startswith('#'):
                continue
            # Extraer nombre del paquete (antes de >=, ==, etc.)
            package = line.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()
            if package:
                packages.append(package)
    
    return packages

def get_import_name(package_name: str) -> str:
    """Convierte nombre de paquete a nombre de import."""
    # Mapeo de nombres de paquetes a nombres de import
    mapping = {
        'python-dotenv': 'dotenv',
        'PyGithub': 'github',
        'gitpython': 'git',
        'pydantic-settings': 'pydantic_settings',
        'python-multipart': None,  # No se importa directamente
        'aiosqlite': 'aiosqlite',
        'asyncpg': 'asyncpg',
        'PyJWT': 'jwt',
        'passlib': 'passlib',
        'structlog': 'structlog',
        'python-json-logger': None,  # No se importa directamente
    }
    
    if package_name in mapping:
        return mapping[package_name]
    
    # Convertir guiones a guiones bajos
    return package_name.replace('-', '_').lower()

def check_critical_dependencies() -> Dict[str, bool]:
    """Verifica dependencias críticas."""
    critical = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
        'PyGithub': 'github',
        'gitpython': 'git',
        'celery': 'celery',
        'redis': 'redis',
        'sqlalchemy': 'sqlalchemy',
    }
    
    results = {}
    for package, import_name in critical.items():
        installed, version = check_package(package, import_name)
        results[package] = installed
        if installed:
            print_success(f"{package} ({version})")
        else:
            print_error(f"{package} - NO INSTALADO")
    
    return results

def check_requirements_file(file_path: Path) -> Dict[str, bool]:
    """Verifica todas las dependencias de un archivo requirements."""
    if not file_path.exists():
        print_warning(f"Archivo no encontrado: {file_path}")
        return {}
    
    print_info(f"\nVerificando: {file_path.name}")
    packages = read_requirements(file_path)
    
    results = {}
    installed_count = 0
    missing_count = 0
    
    for package in packages:
        import_name = get_import_name(package)
        if import_name is None:
            # Paquete que no se importa directamente, asumir instalado
            results[package] = True
            installed_count += 1
            continue
        
        installed, version = check_package(package, import_name)
        results[package] = installed
        
        if installed:
            installed_count += 1
            print_success(f"  {package} ({version})")
        else:
            missing_count += 1
            print_error(f"  {package} - FALTANTE")
    
    print_info(f"Instalados: {installed_count}, Faltantes: {missing_count}")
    return results

def check_python_version():
    """Verifica versión de Python."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print_success(f"Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_error(f"Python {version.major}.{version.minor}.{version.micro} - Se requiere 3.10+")
        return False

def check_pip():
    """Verifica que pip esté disponible."""
    try:
        result = subprocess.run(['pip', '--version'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        print_success(f"pip disponible: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print_error("pip no está disponible")
        return False

def check_redis():
    """Verifica si Redis está corriendo."""
    try:
        result = subprocess.run(['redis-cli', 'ping'], 
                              capture_output=True, 
                              text=True, 
                              timeout=2)
        if result.returncode == 0 and 'PONG' in result.stdout:
            print_success("Redis está corriendo")
            return True
        else:
            print_warning("Redis no está corriendo (opcional para Celery)")
            return False
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        print_warning("Redis no está disponible (opcional para Celery)")
        return False

def main():
    """Función principal."""
    print("=" * 60)
    print_info("Verificación de Dependencias - GitHub Autonomous Agent")
    print("=" * 60)
    
    # Cambiar al directorio del script
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    # Verificaciones básicas
    print("\n📋 Verificaciones Básicas:")
    python_ok = check_python_version()
    pip_ok = check_pip()
    
    if not python_ok or not pip_ok:
        print_error("\nVerificaciones básicas fallaron. Corrige estos problemas primero.")
        sys.exit(1)
    
    # Verificar dependencias críticas
    print("\n🔑 Dependencias Críticas:")
    critical_results = check_critical_dependencies()
    
    if not all(critical_results.values()):
        print_error("\n⚠️  Algunas dependencias críticas faltan!")
        print_info("Ejecuta: pip install -r requirements.txt")
    
    # Verificar requirements.txt
    requirements_file = script_dir / "requirements.txt"
    if requirements_file.exists():
        check_requirements_file(requirements_file)
    
    # Verificar Redis
    print("\n🔴 Redis (Opcional):")
    check_redis()
    
    # Resumen
    print("\n" + "=" * 60)
    if all(critical_results.values()):
        print_success("✅ Todas las dependencias críticas están instaladas")
    else:
        print_error("❌ Faltan dependencias críticas")
        print_info("Ejecuta: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    import os
    main()




