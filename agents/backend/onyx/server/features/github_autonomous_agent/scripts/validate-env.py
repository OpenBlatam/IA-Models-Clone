#!/usr/bin/env python3
"""
Script para validar archivo .env y verificar configuración.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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

def validate_env_file(env_path: Path) -> Tuple[bool, List[str]]:
    """Valida el archivo .env."""
    if not env_path.exists():
        return False, ["Archivo .env no encontrado"]
    
    errors = []
    warnings = []
    
    with open(env_path, 'r') as f:
        lines = f.readlines()
    
    required_vars = {
        'GITHUB_TOKEN': 'Token de GitHub (obligatorio)',
        'SECRET_KEY': 'Clave secreta (obligatorio)',
    }
    
    recommended_vars = {
        'DATABASE_URL': 'URL de base de datos',
        'REDIS_URL': 'URL de Redis',
    }
    
    found_vars = {}
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Ignorar comentarios y líneas vacías
        if not line or line.startswith('#'):
            continue
        
        if '=' not in line:
            warnings.append(f"Línea {line_num}: Formato inválido (no contiene '=')")
            continue
        
        key, value = line.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        found_vars[key] = (value, line_num)
    
    # Verificar variables requeridas
    for var, description in required_vars.items():
        if var not in found_vars:
            errors.append(f"Variable requerida faltante: {var} ({description})")
        else:
            value, _ = found_vars[var]
            if not value or value in ['your_github_token_here', 'change-me-in-production']:
                errors.append(f"Variable {var} no configurada correctamente (valor por defecto)")
    
    # Verificar variables recomendadas
    for var, description in recommended_vars.items():
        if var not in found_vars:
            warnings.append(f"Variable recomendada faltante: {var} ({description})")
    
    # Validaciones específicas
    if 'GITHUB_TOKEN' in found_vars:
        token, _ = found_vars['GITHUB_TOKEN']
        if len(token) < 20:
            warnings.append("GITHUB_TOKEN parece ser muy corto (verifica que sea válido)")
    
    if 'SECRET_KEY' in found_vars:
        secret, _ = found_vars['SECRET_KEY']
        if len(secret) < 32:
            warnings.append("SECRET_KEY debería tener al menos 32 caracteres para seguridad")
        if secret == 'change-me-in-production':
            errors.append("SECRET_KEY debe cambiarse del valor por defecto")
    
    if 'DEBUG' in found_vars:
        debug_val, _ = found_vars['DEBUG']
        if debug_val.lower() == 'true':
            warnings.append("DEBUG está habilitado (deshabilitar en producción)")
    
    if 'DATABASE_URL' in found_vars:
        db_url, _ = found_vars['DATABASE_URL']
        if 'sqlite' in db_url.lower() and 'production' in os.getenv('ENVIRONMENT', '').lower():
            warnings.append("SQLite no es recomendado para producción, usa PostgreSQL")
    
    return len(errors) == 0, errors + warnings

def main():
    """Función principal."""
    script_dir = Path(__file__).parent.parent
    env_path = script_dir / ".env"
    env_example_path = script_dir / ".env.example"
    
    print("=" * 60)
    print_info("Validación de Configuración - GitHub Autonomous Agent")
    print("=" * 60)
    
    # Verificar si existe .env
    if not env_path.exists():
        print_error("\nArchivo .env no encontrado")
        if env_example_path.exists():
            print_info("Copia .env.example a .env:")
            print(f"  cp .env.example .env")
        sys.exit(1)
    
    print_info(f"\nValidando: {env_path}")
    
    # Validar archivo
    is_valid, issues = validate_env_file(env_path)
    
    # Mostrar resultados
    errors = [i for i in issues if i.startswith("Variable requerida") or "no configurada" in i]
    warnings = [i for i in issues if i not in errors]
    
    if errors:
        print("\n❌ Errores encontrados:")
        for error in errors:
            print_error(f"  {error}")
    
    if warnings:
        print("\n⚠️  Advertencias:")
        for warning in warnings:
            print_warning(f"  {warning}")
    
    if not errors and not warnings:
        print_success("\n✅ Configuración válida")
    elif not errors:
        print_warning("\n⚠️  Configuración válida con advertencias")
    else:
        print_error("\n❌ Configuración inválida - corrige los errores")
        sys.exit(1)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()




