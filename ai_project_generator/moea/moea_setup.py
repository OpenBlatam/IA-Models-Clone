"""
MOEA Project Setup Script
=========================
Script para configurar y verificar el entorno del proyecto MOEA
"""
import os
import sys
import subprocess
from pathlib import Path
import json


class MOEASetup:
    """Configurador del proyecto MOEA"""
    
    def __init__(self, project_dir: str = "generated_projects/moea_optimization_system"):
        self.project_dir = Path(project_dir)
        self.backend_dir = self.project_dir / "backend"
        self.frontend_dir = self.project_dir / "frontend"
    
    def check_project_exists(self) -> bool:
        """Verificar que el proyecto exista"""
        if not self.project_dir.exists():
            print(f"❌ Proyecto no encontrado: {self.project_dir}")
            print("   Ejecuta primero: python quick_moea.py")
            return False
        print(f"✅ Proyecto encontrado: {self.project_dir}")
        return True
    
    def check_python(self) -> bool:
        """Verificar Python"""
        try:
            result = subprocess.run(
                ["python", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Python: {version}")
                return True
        except:
            pass
        print("❌ Python no encontrado")
        return False
    
    def check_node(self) -> bool:
        """Verificar Node.js"""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ Node.js: {version}")
                return True
        except:
            pass
        print("⚠️  Node.js no encontrado (necesario para frontend)")
        return False
    
    def check_npm(self) -> bool:
        """Verificar npm"""
        try:
            result = subprocess.run(
                ["npm", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip()
                print(f"✅ npm: {version}")
                return True
        except:
            pass
        print("⚠️  npm no encontrado (necesario para frontend)")
        return False
    
    def install_backend_deps(self) -> bool:
        """Instalar dependencias del backend"""
        if not self.backend_dir.exists():
            print("❌ Directorio backend no encontrado")
            return False
        
        requirements = self.backend_dir / "requirements.txt"
        if not requirements.exists():
            print("⚠️  requirements.txt no encontrado")
            return False
        
        print("\n📦 Instalando dependencias del backend...")
        try:
            result = subprocess.run(
                ["pip", "install", "-r", str(requirements)],
                cwd=str(self.backend_dir),
                timeout=300
            )
            if result.returncode == 0:
                print("✅ Dependencias del backend instaladas")
                return True
            else:
                print("❌ Error instalando dependencias del backend")
                return False
        except Exception as e:
            print(f"❌ Excepción: {e}")
            return False
    
    def install_frontend_deps(self) -> bool:
        """Instalar dependencias del frontend"""
        if not self.frontend_dir.exists():
            print("⚠️  Directorio frontend no encontrado")
            return False
        
        package_json = self.frontend_dir / "package.json"
        if not package_json.exists():
            print("⚠️  package.json no encontrado")
            return False
        
        print("\n📦 Instalando dependencias del frontend...")
        try:
            result = subprocess.run(
                ["npm", "install"],
                cwd=str(self.frontend_dir),
                timeout=300
            )
            if result.returncode == 0:
                print("✅ Dependencias del frontend instaladas")
                return True
            else:
                print("❌ Error instalando dependencias del frontend")
                return False
        except Exception as e:
            print(f"❌ Excepción: {e}")
            return False
    
    def create_env_file(self) -> bool:
        """Crear archivo .env para backend"""
        env_file = self.backend_dir / ".env"
        
        if env_file.exists():
            print("⚠️  .env ya existe, omitiendo creación")
            return True
        
        env_content = """# MOEA Backend Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# MOEA Default Parameters
MOEA_DEFAULT_POPULATION_SIZE=100
MOEA_DEFAULT_GENERATIONS=100
MOEA_DEFAULT_MUTATION_RATE=0.1
MOEA_DEFAULT_CROSSOVER_RATE=0.9
"""
        
        try:
            with open(env_file, "w") as f:
                f.write(env_content)
            print("✅ Archivo .env creado")
            return True
        except Exception as e:
            print(f"❌ Error creando .env: {e}")
            return False
    
    def show_project_info(self):
        """Mostrar información del proyecto"""
        info_file = self.project_dir / "project_info.json"
        
        if info_file.exists():
            try:
                with open(info_file, "r") as f:
                    info = json.load(f)
                print("\n📊 Información del Proyecto:")
                print(f"   Nombre: {info.get('project_name', 'N/A')}")
                print(f"   Autor: {info.get('author', 'N/A')}")
                print(f"   Versión: {info.get('version', 'N/A')}")
                print(f"   Tipo IA: {info.get('ai_type', 'N/A')}")
            except:
                pass
    
    def run_setup(self, install_backend=True, install_frontend=True):
        """Ejecutar setup completo"""
        print("=" * 60)
        print("MOEA Project Setup")
        print("=" * 60)
        print()
        
        # Verificaciones básicas
        if not self.check_project_exists():
            return False
        
        print("\n🔍 Verificando herramientas...")
        python_ok = self.check_python()
        node_ok = self.check_node()
        npm_ok = self.check_npm()
        
        if not python_ok:
            print("\n❌ Python es requerido. Instálalo desde python.org")
            return False
        
        # Instalación
        print("\n📦 Instalando dependencias...")
        
        backend_ok = True
        if install_backend:
            backend_ok = self.install_backend_deps()
        
        frontend_ok = True
        if install_frontend and node_ok and npm_ok:
            frontend_ok = self.install_frontend_deps()
        elif install_frontend:
            print("⚠️  Omitiendo frontend (Node.js/npm no disponible)")
        
        # Configuración
        print("\n⚙️  Configurando proyecto...")
        env_ok = self.create_env_file()
        
        # Información
        self.show_project_info()
        
        # Resumen
        print("\n" + "=" * 60)
        print("📋 Setup Summary")
        print("=" * 60)
        print(f"   Backend:  {'✅' if backend_ok else '❌'}")
        print(f"   Frontend: {'✅' if frontend_ok else '⚠️ '}")
        print(f"   Config:   {'✅' if env_ok else '❌'}")
        print("=" * 60)
        
        if backend_ok:
            print("\n✅ Setup completado!")
            print("\n🚀 Próximos pasos:")
            print("   1. Backend:  cd backend && uvicorn app.main:app --reload")
            if frontend_ok:
                print("   2. Frontend: cd frontend && npm run dev")
            print("   3. Probar:   python moea_test_api.py")
            return True
        else:
            print("\n❌ Setup incompleto. Revisa los errores arriba.")
            return False


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup MOEA project")
    parser.add_argument(
        "--project-dir",
        default="generated_projects/moea_optimization_system",
        help="Project directory"
    )
    parser.add_argument(
        "--no-backend",
        action="store_true",
        help="Skip backend installation"
    )
    parser.add_argument(
        "--no-frontend",
        action="store_true",
        help="Skip frontend installation"
    )
    
    args = parser.parse_args()
    
    setup = MOEASetup(args.project_dir)
    success = setup.run_setup(
        install_backend=not args.no_backend,
        install_frontend=not args.no_frontend
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

