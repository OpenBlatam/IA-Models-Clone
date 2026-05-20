#!/usr/bin/env python3
"""
Setup Script para Bulk TruthGPT
================================

Script para configurar el entorno y crear todos los directorios necesarios.
"""

import os
import sys
from pathlib import Path
import secrets

def create_directories():
    """Crear todos los directorios necesarios."""
    base_dir = Path(__file__).parent
    
    directories = [
        "storage",
        "templates",
        "models",
        "knowledge_base",
        "logs",
        "cache",
        "temp"
    ]
    
    print("📁 Creando directorios necesarios...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}/")
    
    return True

def create_env_file():
    """Crear archivo .env si no existe."""
    base_dir = Path(__file__).parent
    env_file = base_dir / ".env"
    env_example = base_dir / "env.example"
    
    if env_file.exists():
        print("  ✓ .env ya existe")
        return True
    
    if env_example.exists():
        print("📝 Creando archivo .env desde env.example...")
        content = env_example.read_text()
        
        # Generar SECRET_KEY seguro si está en el ejemplo
        if "your-secret-key-change-this-in-production" in content:
            secret_key = secrets.token_urlsafe(32)
            content = content.replace(
                "your-secret-key-change-this-in-production",
                secret_key
            )
        
        env_file.write_text(content)
        print("  ✓ .env creado")
        print("  ⚠️  IMPORTANTE: Revisa y configura las variables en .env")
        return True
    else:
        print("  ⚠️  env.example no encontrado, creando .env básico...")
        env_file.write_text("""# Bulk TruthGPT System Environment Variables
ENVIRONMENT=development
DEBUG=true
API_HOST=0.0.0.0
API_PORT=8000
SECRET_KEY={}
REDIS_URL=redis://localhost:6379/0
DATABASE_URL=postgresql://postgres:password@localhost:5432/bulk_truthgpt
""".format(secrets.token_urlsafe(32)))
        print("  ✓ .env básico creado")
        return True

def check_dependencies():
    """Verificar dependencias básicas."""
    print("🔍 Verificando dependencias...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "pydantic",
        "redis"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NO INSTALADO")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Dependencias faltantes: {', '.join(missing)}")
        print("   Ejecuta: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Función principal."""
    print("=" * 60)
    print("🚀 Configuración de Bulk TruthGPT System")
    print("=" * 60)
    print()
    
    # Crear directorios
    if not create_directories():
        print("❌ Error creando directorios")
        return 1
    
    print()
    
    # Crear .env
    if not create_env_file():
        print("❌ Error creando .env")
        return 1
    
    print()
    
    # Verificar dependencias
    deps_ok = check_dependencies()
    
    print()
    print("=" * 60)
    if deps_ok:
        print("✅ Configuración completada exitosamente!")
    else:
        print("⚠️  Configuración completada con advertencias")
    print("=" * 60)
    print()
    print("📋 Próximos pasos:")
    print("  1. Revisa y configura el archivo .env")
    print("  2. Instala dependencias: pip install -r requirements.txt")
    print("  3. Inicia Redis (si no está corriendo): docker run -d -p 6379:6379 redis")
    print("  4. Inicia el servidor: python -m bulk_truthgpt.main")
    print("     o usa: uvicorn bulk_truthgpt.main:app --reload")
    print()
    
    return 0 if deps_ok else 1

if __name__ == "__main__":
    sys.exit(main())
































