#!/usr/bin/env python3
"""
Script para inicializar un nuevo proyecto desde cero.
Crea estructura de directorios, archivos base y configuración inicial.
"""

import sys
from pathlib import Path
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def create_directory_structure(base_dir: Path):
    """Crea estructura de directorios."""
    directories = [
        'api/routes',
        'api/schemas',
        'core',
        'config',
        'storage/tasks',
        'storage/logs',
        'storage/cache',
        'tests',
        'tests/integration',
        'scripts',
        'backups',
    ]
    
    created = []
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        created.append(dir_path)
    
    return created

def create_init_files(base_dir: Path):
    """Crea archivos __init__.py necesarios."""
    init_files = [
        'api/__init__.py',
        'api/routes/__init__.py',
        'api/schemas/__init__.py',
        'core/__init__.py',
        'config/__init__.py',
        'tests/__init__.py',
    ]
    
    created = []
    for init_file in init_files:
        full_path = base_dir / init_file
        if not full_path.exists():
            full_path.write_text('"""Module initialization."""\n')
            created.append(init_file)
    
    return created

def create_gitignore(base_dir: Path):
    """Crea .gitignore si no existe."""
    gitignore = base_dir / '.gitignore'
    if not gitignore.exists():
        # Usar el .gitignore que ya creamos
        print_info(".gitignore ya debería existir")
    return gitignore.exists()

def create_env_example(base_dir: Path):
    """Crea .env.example si no existe."""
    env_example = base_dir / '.env.example'
    if not env_example.exists():
        print_info(".env.example ya debería existir")
    return env_example.exists()

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inicializar estructura del proyecto')
    parser.add_argument('--path', type=str, default='.',
                       help='Ruta donde inicializar (default: directorio actual)')
    parser.add_argument('--force', action='store_true',
                       help='Sobrescribir archivos existentes')
    
    args = parser.parse_args()
    
    base_dir = Path(args.path).resolve()
    
    print_header("Inicialización de Proyecto - GitHub Autonomous Agent")
    print_info(f"Directorio base: {base_dir}\n")
    
    # Crear estructura
    print_info("Creando estructura de directorios...")
    dirs = create_directory_structure(base_dir)
    print_success(f"Creados {len(dirs)} directorios")
    
    # Crear __init__.py
    print_info("\nCreando archivos __init__.py...")
    inits = create_init_files(base_dir)
    if inits:
        print_success(f"Creados {len(inits)} archivos __init__.py")
    else:
        print_info("Todos los __init__.py ya existen")
    
    # Verificar archivos importantes
    print_info("\nVerificando archivos de configuración...")
    gitignore_ok = create_gitignore(base_dir)
    env_ok = create_env_example(base_dir)
    
    if gitignore_ok:
        print_success(".gitignore existe")
    else:
        print_info("⚠️  Crea un .gitignore")
    
    if env_ok:
        print_success(".env.example existe")
    else:
        print_info("⚠️  Crea un .env.example")
    
    # Resumen
    print_header("Resumen de Inicialización")
    print_success("✅ Estructura del proyecto inicializada")
    print_info("\nPróximos pasos:")
    print("  1. Configurar .env desde .env.example")
    print("  2. Instalar dependencias: pip install -r requirements.txt")
    print("  3. Ejecutar migraciones: python scripts/migrate-db.py upgrade")
    print("  4. Iniciar desarrollo: make run-dev")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())




