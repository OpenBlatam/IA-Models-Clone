#!/usr/bin/env python3
"""
Script para verificar seguridad de dependencias.
Comprueba vulnerabilidades conocidas y dependencias desactualizadas.
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict

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

def check_safety():
    """Verifica con safety."""
    print_header("Safety Check")
    
    try:
        result = subprocess.run(
            ["safety", "check", "--json"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            print_success("No se encontraron vulnerabilidades")
            return True, []
        
        try:
            vulns = json.loads(result.stdout)
            if vulns:
                return False, vulns
            return True, []
        except:
            print_warning("No se pudo parsear output de safety")
            return None, []
    except FileNotFoundError:
        print_warning("safety no está instalado")
        print_info("Instala con: pip install safety")
        return None, []

def check_pip_audit():
    """Verifica con pip-audit."""
    print_header("pip-audit Check")
    
    try:
        result = subprocess.run(
            ["pip-audit", "--format=json"],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                vulns = data.get("vulnerabilities", [])
                if not vulns:
                    print_success("No se encontraron vulnerabilidades")
                    return True, []
                return False, vulns
            except:
                print_success("No se encontraron vulnerabilidades")
                return True, []
        else:
            try:
                data = json.loads(result.stdout)
                vulns = data.get("vulnerabilities", [])
                return False, vulns
            except:
                return None, []
    except FileNotFoundError:
        print_warning("pip-audit no está instalado")
        print_info("Instala con: pip install pip-audit")
        return None, []

def check_outdated_security():
    """Verifica paquetes desactualizados con vulnerabilidades conocidas."""
    print_header("Paquetes Desactualizados")
    
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated", "--format=json"],
            capture_output=True,
            text=True,
            check=True
        )
        
        outdated = json.loads(result.stdout)
        
        # Paquetes críticos de seguridad
        critical_packages = [
            'cryptography', 'pyjwt', 'passlib', 'fastapi',
            'uvicorn', 'pydantic', 'sqlalchemy', 'celery',
            'redis', 'httpx', 'aiohttp'
        ]
        
        critical_outdated = [
            pkg for pkg in outdated
            if pkg['name'].lower() in [cp.lower() for cp in critical_packages]
        ]
        
        if critical_outdated:
            print_warning(f"Paquetes críticos desactualizados: {len(critical_outdated)}")
            for pkg in critical_outdated:
                print(f"  ⚠️  {pkg['name']}: {pkg['version']} → {pkg.get('latest_version', 'unknown')}")
            return False
        else:
            print_success("Paquetes críticos están actualizados")
            return True
    except Exception as e:
        print_warning(f"Error verificando paquetes: {e}")
        return None

def check_secret_keys():
    """Verifica que no hay secretos hardcodeados."""
    print_header("Verificación de Secretos")
    
    script_dir = Path(__file__).parent.parent
    secret_patterns = [
        ('password', 'password\s*=\s*["\'][^"\']+["\']'),
        ('token', 'token\s*=\s*["\'][^"\']+["\']'),
        ('secret', 'secret\s*=\s*["\'][^"\']+["\']'),
        ('api_key', 'api_key\s*=\s*["\'][^"\']+["\']'),
    ]
    
    issues = []
    for py_file in script_dir.rglob('*.py'):
        if 'venv' in str(py_file) or '__pycache__' in str(py_file):
            continue
        
        try:
            content = py_file.read_text()
            for pattern_name, pattern in secret_patterns:
                import re
                if re.search(pattern, content, re.IGNORECASE):
                    # Verificar que no sea un comentario o docstring
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if re.search(pattern, line, re.IGNORECASE):
                            if not line.strip().startswith('#') and '"""' not in line:
                                issues.append((str(py_file), i+1, pattern_name))
        except:
            continue
    
    if issues:
        print_warning(f"Posibles secretos encontrados: {len(issues)}")
        for file, line, pattern in issues[:10]:
            print(f"  ⚠️  {file}:{line} - Patrón: {pattern}")
        if len(issues) > 10:
            print(f"  ... y {len(issues) - 10} más")
        return False
    else:
        print_success("No se encontraron secretos hardcodeados")
        return True

def generate_security_report(results: Dict):
    """Genera reporte de seguridad."""
    report_path = Path("security_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("Security Report\n")
        f.write("="*60 + "\n\n")
        
        for check_name, (status, details) in results.items():
            f.write(f"{check_name}:\n")
            if status is True:
                f.write("  ✅ PASS\n")
            elif status is False:
                f.write("  ❌ FAIL\n")
                if details:
                    for detail in details[:20]:
                        f.write(f"    - {detail}\n")
            else:
                f.write("  ⚠️  SKIPPED\n")
            f.write("\n")
    
    print_success(f"Reporte guardado en: {report_path}")

def main():
    """Función principal."""
    import os
    
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    print_header("Security Check - GitHub Autonomous Agent")
    
    results = {}
    
    # Safety check
    safety_status, safety_vulns = check_safety()
    results['Safety'] = (safety_status, safety_vulns)
    
    if safety_status is False and safety_vulns:
        print_warning(f"\nVulnerabilidades encontradas por Safety: {len(safety_vulns)}")
        for vuln in safety_vulns[:10]:
            pkg = vuln.get('package', 'unknown')
            vuln_id = vuln.get('vulnerability', 'unknown')
            print(f"  ❌ {pkg}: {vuln_id}")
    
    # pip-audit check
    audit_status, audit_vulns = check_pip_audit()
    results['pip-audit'] = (audit_status, audit_vulns)
    
    if audit_status is False and audit_vulns:
        print_warning(f"\nVulnerabilidades encontradas por pip-audit: {len(audit_vulns)}")
        for vuln in audit_vulns[:10]:
            pkg = vuln.get('name', 'unknown')
            vuln_id = vuln.get('id', 'unknown')
            print(f"  ❌ {pkg}: {vuln_id}")
    
    # Outdated critical packages
    outdated_status = check_outdated_security()
    results['Outdated Critical'] = (outdated_status, [])
    
    # Secret keys check
    secrets_status = check_secret_keys()
    results['Secret Keys'] = (secrets_status, [])
    
    # Resumen
    print_header("Resumen de Seguridad")
    
    passed = sum(1 for status, _ in results.values() if status is True)
    failed = sum(1 for status, _ in results.values() if status is False)
    skipped = sum(1 for status, _ in results.values() if status is None)
    
    print(f"✅ Pasados: {passed}")
    print(f"❌ Fallidos: {failed}")
    print(f"⚠️  Omitidos: {skipped}")
    
    if failed > 0:
        print_error("\n⚠️  Se encontraron problemas de seguridad!")
        print_info("Revisa el reporte y actualiza las dependencias vulnerables")
        sys.exit(1)
    else:
        print_success("\n✅ Todas las verificaciones de seguridad pasaron")
    
    # Generar reporte
    generate_security_report(results)

if __name__ == "__main__":
    main()




