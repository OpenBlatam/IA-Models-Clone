#!/usr/bin/env python3
"""
Script para generar requirements-lock.txt con versiones exactas.
Útil para deployments reproducibles.
"""

import subprocess
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

def generate_lock_file(requirements_file: Path, output_file: Path):
    """Genera archivo de lock con versiones exactas."""
    print_info(f"Generando lock file desde: {requirements_file.name}")
    
    try:
        # Usar pip freeze pero solo para paquetes en requirements
        result = subprocess.run(
            ["pip", "freeze"],
            capture_output=True,
            text=True,
            check=True
        )
        
        installed = {}
        for line in result.stdout.strip().split('\n'):
            if '==' in line:
                name, version = line.split('==', 1)
                installed[name.lower()] = (name, version)
        
        # Leer requirements original
        with open(requirements_file, 'r') as f:
            requirements_content = f.read()
        
        # Generar lock file
        with open(output_file, 'w') as f:
            f.write("# ============================================================================\n")
            f.write("# GitHub Autonomous Agent - Locked Dependencies\n")
            f.write("# ============================================================================\n")
            f.write(f"# Generado: {datetime.now().isoformat()}\n")
            f.write("# Este archivo contiene versiones exactas de todas las dependencias\n")
            f.write("# Útil para deployments reproducibles\n")
            f.write("# ============================================================================\n\n")
            
            # Mantener estructura y comentarios del original
            in_section = False
            current_section = ""
            
            for line in requirements_content.split('\n'):
                stripped = line.strip()
                
                # Preservar comentarios y secciones
                if stripped.startswith('#'):
                    if stripped.startswith('# ---'):
                        in_section = True
                        current_section = stripped
                    f.write(line + '\n')
                    continue
                
                if not stripped:
                    f.write('\n')
                    continue
                
                # Procesar dependencia
                if '>=' in stripped or '==' in stripped or '<' in stripped:
                    # Extraer nombre del paquete
                    package_name = stripped.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()
                    
                    # Buscar versión instalada
                    if package_name.lower() in installed:
                        name, version = installed[package_name.lower()]
                        # Mantener comentario original si existe
                        comment = ""
                        if '#' in stripped:
                            comment = "  # " + stripped.split('#', 1)[1].strip()
                        f.write(f"{name}=={version}{comment}\n")
                    else:
                        # Si no está instalado, mantener línea original
                        f.write(line + '\n')
                else:
                    f.write(line + '\n')
        
        print_success(f"Lock file generado: {output_file.name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error generando lock file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal."""
    import os
    
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print_header("Generación de Requirements Lock File")
    
    # Archivos
    requirements_file = script_dir / "requirements.txt"
    lock_file = script_dir / "requirements-lock.txt"
    
    if not requirements_file.exists():
        print(f"❌ Archivo no encontrado: {requirements_file}")
        sys.exit(1)
    
    # Verificar que estamos en un entorno virtual
    import sys
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Advertencia: No parece estar en un entorno virtual")
        response = input("¿Continuar de todos modos? (s/N): ")
        if response.lower() != 's':
            sys.exit(0)
    
    # Generar lock file
    if generate_lock_file(requirements_file, lock_file):
        print_info("\nPróximos pasos:")
        print("  1. Revisa requirements-lock.txt")
        print("  2. Commitea si deseas versiones exactas en producción")
        print("  3. Para instalar: pip install -r requirements-lock.txt")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()




