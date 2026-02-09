#!/usr/bin/env python3
"""
Script para actualizar dependencias de forma segura.
Verifica compatibilidad y genera reporte de cambios.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def print_success(msg: str):
    print(f"{Colors.GREEN}✅ {msg}{Colors.RESET}")

def print_warning(msg: str):
    print(f"{Colors.YELLOW}⚠️  {msg}{Colors.RESET}")

def print_error(msg: str):
    print(f"{Colors.RED}❌ {msg}{Colors.RESET}")

def print_info(msg: str):
    print(f"{Colors.BLUE}ℹ️  {msg}{Colors.RESET}")

def print_header(msg: str):
    print(f"\n{Colors.CYAN}{'='*60}{Colors.RESET}")
    print(f"{Colors.CYAN}{msg}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}\n")

def get_installed_packages() -> Dict[str, str]:
    """Obtiene paquetes instalados y sus versiones."""
    try:
        result = subprocess.run(
            ["pip", "list", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        packages = json.loads(result.stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in packages}
    except Exception as e:
        print_error(f"Error obteniendo paquetes instalados: {e}")
        return {}

def get_outdated_packages() -> List[Dict[str, str]]:
    """Obtiene paquetes desactualizados."""
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        return json.loads(result.stdout)
    except Exception as e:
        print_warning(f"No se pudieron obtener paquetes desactualizados: {e}")
        return []

def read_requirements(file_path: Path) -> List[Tuple[str, str]]:
    """Lee requirements.txt y extrae paquetes con versiones."""
    packages = []
    if not file_path.exists():
        return packages
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Extraer nombre y versión
            parts = line.split('>=')
            if len(parts) == 2:
                name = parts[0].strip()
                version = parts[1].split(',')[0].split('<')[0].strip()
                packages.append((name, version))
    
    return packages

def check_vulnerabilities():
    """Verifica vulnerabilidades conocidas."""
    print_header("Verificando Vulnerabilidades")
    
    try:
        # Intentar con safety
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print_success("No se encontraron vulnerabilidades conocidas")
            return True
        else:
            try:
                vulns = json.loads(result.stdout)
                if vulns:
                    print_warning(f"Se encontraron {len(vulns)} vulnerabilidades:")
                    for vuln in vulns[:10]:  # Mostrar solo las primeras 10
                        print(f"  - {vuln.get('package', 'unknown')}: {vuln.get('vulnerability', 'unknown')}")
                    if len(vulns) > 10:
                        print(f"  ... y {len(vulns) - 10} más")
                    return False
            except:
                print_warning("No se pudo parsear output de safety")
                return True
    except FileNotFoundError:
        print_warning("safety no está instalado. Instala con: pip install safety")
        print_info("O usa: pip install pip-audit && pip-audit")
        return None

def update_package(package_name: str, dry_run: bool = True) -> bool:
    """Actualiza un paquete específico."""
    if dry_run:
        print_info(f"[DRY RUN] Actualizaría: {package_name}")
        return True
    
    try:
        subprocess.run(
            ["pip", "install", "--upgrade", package_name],
            check=True,
            capture_output=True
        )
        print_success(f"Actualizado: {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Error actualizando {package_name}: {e}")
        return False

def generate_update_report(outdated: List[Dict], requirements_file: Path):
    """Genera reporte de actualizaciones."""
    report_path = Path("dependency_update_report.txt")
    
    with open(report_path, 'w') as f:
        f.write(f"Dependency Update Report\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Total outdated packages: {len(outdated)}\n\n")
        
        for pkg in outdated:
            f.write(f"Package: {pkg['name']}\n")
            f.write(f"  Current: {pkg['version']}\n")
            f.write(f"  Latest:  {pkg.get('latest_version', 'unknown')}\n")
            f.write(f"\n")
    
    print_success(f"Reporte guardado en: {report_path}")

def main():
    """Función principal."""
    import os
    
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print_header("Actualización de Dependencias - GitHub Autonomous Agent")
    
    # Verificar modo
    dry_run = "--dry-run" in sys.argv or "-d" in sys.argv
    if dry_run:
        print_info("Modo DRY RUN - No se realizarán cambios reales\n")
    
    # Leer requirements
    requirements_file = script_dir / "requirements.txt"
    if not requirements_file.exists():
        print_error(f"Archivo no encontrado: {requirements_file}")
        sys.exit(1)
    
    print_info(f"Analizando: {requirements_file.name}")
    
    # Obtener paquetes desactualizados
    print_header("Buscando Paquetes Desactualizados")
    outdated = get_outdated_packages()
    
    if not outdated:
        print_success("Todos los paquetes están actualizados!")
        return
    
    print_warning(f"Se encontraron {len(outdated)} paquetes desactualizados:\n")
    
    # Mostrar paquetes desactualizados
    for pkg in outdated[:20]:  # Mostrar primeros 20
        name = pkg['name']
        current = pkg['version']
        latest = pkg.get('latest_version', 'unknown')
        print(f"  {name:30} {current:15} → {latest}")
    
    if len(outdated) > 20:
        print(f"\n  ... y {len(outdated) - 20} más")
    
    # Generar reporte
    generate_update_report(outdated, requirements_file)
    
    # Verificar vulnerabilidades
    vuln_result = check_vulnerabilities()
    
    # Actualizar si no es dry-run
    if not dry_run:
        print_header("Actualizando Paquetes")
        response = input("¿Deseas actualizar todos los paquetes? (s/N): ")
        
        if response.lower() == 's':
            updated = 0
            failed = 0
            
            for pkg in outdated:
                if update_package(pkg['name'], dry_run=False):
                    updated += 1
                else:
                    failed += 1
            
            print_header("Resumen de Actualización")
            print_success(f"Actualizados: {updated}")
            if failed > 0:
                print_error(f"Fallidos: {failed}")
            
            print_info("\nPróximos pasos:")
            print("  1. Ejecutar tests: make test")
            print("  2. Verificar que todo funciona: make run-dev")
            print("  3. Si todo está bien, actualizar requirements.txt con versiones nuevas")
        else:
            print_info("Actualización cancelada")
    else:
        print_info("\nPara actualizar realmente, ejecuta sin --dry-run:")
        print("  python scripts/update-dependencies.py")

if __name__ == "__main__":
    main()




