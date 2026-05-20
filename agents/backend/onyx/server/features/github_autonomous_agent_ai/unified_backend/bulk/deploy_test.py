"""
Script de Deployment y Verificación
Verifica que todo esté listo para deployment
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Any

class DeploymentChecker:
    """Verifica preparación para deployment."""
    
    def __init__(self):
        self.checks: List[Dict[str, Any]] = []
        self.passed = 0
        self.failed = 0
    
    def check_file_exists(self, filepath: str, description: str) -> bool:
        """Verifica que un archivo existe."""
        exists = Path(filepath).exists()
        self.checks.append({
            "check": description,
            "status": "PASS" if exists else "FAIL",
            "file": filepath
        })
        
        if exists:
            self.passed += 1
            print(f"✅ {description}")
        else:
            self.failed += 1
            print(f"❌ {description} - Archivo no encontrado: {filepath}")
        
        return exists
    
    def check_python_syntax(self, filepath: str) -> bool:
        """Verifica sintaxis Python."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", filepath],
                capture_output=True,
                text=True
            )
            
            success = result.returncode == 0
            self.checks.append({
                "check": f"Sintaxis Python: {filepath}",
                "status": "PASS" if success else "FAIL"
            })
            
            if success:
                self.passed += 1
                print(f"✅ Sintaxis Python válida: {filepath}")
            else:
                self.failed += 1
                print(f"❌ Error de sintaxis en {filepath}")
                print(f"   {result.stderr}")
            
            return success
        except Exception as e:
            self.failed += 1
            print(f"❌ Error verificando sintaxis: {e}")
            return False
    
    def run_deployment_checks(self):
        """Ejecuta todas las verificaciones."""
        print("\n" + "="*70)
        print("  🚀 VERIFICACIÓN DE DEPLOYMENT")
        print("="*70 + "\n")
        
        print("📁 Verificando archivos esenciales...")
        self.check_file_exists("api_frontend_ready.py", "API principal")
        self.check_file_exists("requirements.txt", "Dependencias")
        self.check_file_exists("bul-api-client.ts", "Cliente TypeScript")
        self.check_file_exists("frontend_types.ts", "Tipos TypeScript")
        print()
        
        print("🐍 Verificando sintaxis Python...")
        self.check_python_syntax("api_frontend_ready.py")
        print()
        
        print("📚 Verificando documentación...")
        self.check_file_exists("README_FRONTEND.md", "README Frontend")
        self.check_file_exists("README_QUE_GENERA.md", "README Qué Genera")
        print()
        
        print("🧪 Verificando scripts de prueba...")
        self.check_file_exists("test_api_responses.py", "Tests básicos")
        self.check_file_exists("test_api_advanced.py", "Tests avanzados")
        self.check_file_exists("test_security.py", "Tests de seguridad")
        print()
        
        print("="*70)
        print(f"  RESUMEN")
        print("="*70)
        print(f"Verificaciones pasadas: {self.passed}")
        print(f"Verificaciones fallidas: {self.failed}")
        print(f"Total: {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n✅ Todo listo para deployment")
            return True
        else:
            print(f"\n⚠️ {self.failed} verificación(es) fallaron")
            print("   Revisa los errores antes de hacer deployment")
            return False

if __name__ == "__main__":
    checker = DeploymentChecker()
    success = checker.run_deployment_checks()
    sys.exit(0 if success else 1)
































