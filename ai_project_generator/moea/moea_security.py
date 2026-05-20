"""
MOEA Security - Verificador de seguridad
========================================
Verifica y reporta problemas de seguridad en el sistema MOEA
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class MOEASecurityAudit:
    """Auditoría de seguridad MOEA"""
    
    def __init__(self):
        self.issues: List[Dict] = []
        self.severity_levels = {
            "critical": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "info": "ℹ️"
        }
    
    def check_env_files(self, project_dir: str) -> List[Dict]:
        """Verificar archivos .env"""
        issues = []
        project_path = Path(project_dir)
        
        for env_file in project_path.rglob('.env'):
            try:
                with open(env_file, 'r') as f:
                    content = f.read()
                    
                # Buscar credenciales hardcodeadas
                patterns = {
                    "password": r'password\s*=\s*["\']?[^"\'\s]+["\']?',
                    "secret": r'secret\s*=\s*["\']?[^"\'\s]+["\']?',
                    "api_key": r'api[_-]?key\s*=\s*["\']?[^"\'\s]+["\']?',
                    "token": r'token\s*=\s*["\']?[^"\'\s]+["\']?'
                }
                
                for pattern_name, pattern in patterns.items():
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append({
                            "file": str(env_file),
                            "type": "hardcoded_credentials",
                            "pattern": pattern_name,
                            "severity": "high",
                            "message": f"Posible credencial hardcodeada encontrada: {pattern_name}"
                        })
                
                # Verificar si está en .gitignore
                gitignore = project_path / '.gitignore'
                if gitignore.exists():
                    with open(gitignore, 'r') as f:
                        gitignore_content = f.read()
                    if '.env' not in gitignore_content:
                        issues.append({
                            "file": str(env_file),
                            "type": "gitignore_missing",
                            "severity": "medium",
                            "message": ".env no está en .gitignore"
                        })
                
            except Exception as e:
                issues.append({
                    "file": str(env_file),
                    "type": "read_error",
                    "severity": "low",
                    "message": f"Error leyendo archivo: {e}"
                })
        
        return issues
    
    def check_permissions(self, project_dir: str) -> List[Dict]:
        """Verificar permisos de archivos"""
        issues = []
        project_path = Path(project_dir)
        
        # Archivos que no deberían ser ejecutables
        sensitive_files = ['.env', 'config.json', '*.key', '*.pem']
        
        for file_path in project_path.rglob('*'):
            if file_path.is_file():
                # Verificar permisos en sistemas Unix
                try:
                    import stat
                    file_stat = file_path.stat()
                    mode = file_stat.st_mode
                    
                    # Verificar si es ejecutable y no debería serlo
                    if stat.S_IXUSR & mode or stat.S_IXGRP & mode or stat.S_IXOTH & mode:
                        if any(file_path.match(pattern) for pattern in sensitive_files):
                            issues.append({
                                "file": str(file_path),
                                "type": "executable_sensitive",
                                "severity": "medium",
                                "message": "Archivo sensible con permisos de ejecución"
                            })
                except:
                    pass  # Windows no tiene stat
        
        return issues
    
    def check_dependencies(self, requirements_file: str) -> List[Dict]:
        """Verificar dependencias"""
        issues = []
        req_path = Path(requirements_file)
        
        if not req_path.exists():
            return issues
        
        # Lista de paquetes conocidos con vulnerabilidades (ejemplo)
        vulnerable_packages = {
            "requests": "<2.28.0",
            "urllib3": "<1.26.0"
        }
        
        try:
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parsear dependencia
                    if '==' in line:
                        package, version = line.split('==', 1)
                        package = package.strip().lower()
                        version = version.strip()
                        
                        if package in vulnerable_packages:
                            min_version = vulnerable_packages[package]
                            issues.append({
                                "file": str(req_path),
                                "type": "vulnerable_dependency",
                                "package": package,
                                "version": version,
                                "severity": "high",
                                "message": f"Posible vulnerabilidad en {package} {version}"
                            })
        except Exception as e:
            issues.append({
                "file": str(req_path),
                "type": "read_error",
                "severity": "low",
                "message": f"Error leyendo archivo: {e}"
            })
        
        return issues
    
    def audit_project(self, project_dir: str) -> Dict:
        """Auditar proyecto completo"""
        print(f"🔍 Auditando proyecto: {project_dir}\n")
        
        all_issues = []
        
        # Verificar .env
        print("  Verificando archivos .env...")
        env_issues = self.check_env_files(project_dir)
        all_issues.extend(env_issues)
        
        # Verificar permisos
        print("  Verificando permisos...")
        perm_issues = self.check_permissions(project_dir)
        all_issues.extend(perm_issues)
        
        # Verificar dependencias
        print("  Verificando dependencias...")
        req_file = Path(project_dir) / "backend" / "requirements.txt"
        if req_file.exists():
            dep_issues = self.check_dependencies(str(req_file))
            all_issues.extend(dep_issues)
        
        # Agrupar por severidad
        by_severity = defaultdict(list)
        for issue in all_issues:
            by_severity[issue["severity"]].append(issue)
        
        report = {
            "project_dir": project_dir,
            "audited_at": datetime.now().isoformat(),
            "total_issues": len(all_issues),
            "by_severity": {
                severity: len(issues)
                for severity, issues in by_severity.items()
            },
            "issues": all_issues
        }
        
        return report
    
    def print_report(self, report: Dict):
        """Imprimir reporte"""
        print("\n" + "=" * 70)
        print("MOEA Security Audit Report".center(70))
        print("=" * 70)
        print(f"\nProyecto: {report['project_dir']}")
        print(f"Fecha: {report['audited_at']}")
        print(f"Total de issues: {report['total_issues']}\n")
        
        if report['total_issues'] == 0:
            print("✅ No se encontraron problemas de seguridad")
            return
        
        # Agrupar por severidad
        by_severity = defaultdict(list)
        for issue in report['issues']:
            by_severity[issue['severity']].append(issue)
        
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            if severity in by_severity:
                icon = self.severity_levels.get(severity, "ℹ️")
                print(f"\n{icon} {severity.upper()} ({len(by_severity[severity])}):")
                for issue in by_severity[severity]:
                    print(f"   • {issue['message']}")
                    print(f"     Archivo: {issue.get('file', 'N/A')}")
        
        print("\n" + "=" * 70)


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MOEA Security Audit")
    parser.add_argument(
        'project_dir',
        help='Directorio del proyecto a auditar'
    )
    parser.add_argument(
        '--output',
        help='Guardar reporte en archivo JSON'
    )
    
    args = parser.parse_args()
    
    auditor = MOEASecurityAudit()
    report = auditor.audit_project(args.project_dir)
    
    auditor.print_report(report)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n✅ Reporte guardado: {args.output}")


if __name__ == "__main__":
    main()

