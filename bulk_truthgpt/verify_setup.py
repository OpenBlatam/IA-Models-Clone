#!/usr/bin/env python3
"""
Script de Verificación del Sistema
===================================

Verifica que todo esté configurado correctamente.
"""

import os
import sys
from pathlib import Path

def check_directories():
    """Verificar directorios necesarios."""
    print("📁 Verificando directorios...")
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
    
    all_ok = True
    for directory in directories:
        dir_path = base_dir / directory
        if dir_path.exists():
            print(f"  ✅ {directory}/")
        else:
            print(f"  ❌ {directory}/ - FALTA")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"     ✅ Creado")
            all_ok = False
    
    return all_ok

def check_env_file():
    """Verificar archivo .env."""
    print("\n📝 Verificando archivo .env...")
    base_dir = Path(__file__).parent
    env_file = base_dir / ".env"
    env_example = base_dir / "env.example"
    
    if env_file.exists():
        print("  ✅ .env existe")
        # Verificar que tenga contenido mínimo
        content = env_file.read_text()
        if "SECRET_KEY" in content and "your-secret-key" not in content:
            print("  ✅ SECRET_KEY configurado")
        else:
            print("  ⚠️  SECRET_KEY necesita ser configurado")
        return True
    else:
        print("  ❌ .env no existe")
        if env_example.exists():
            print("  📋 Creando desde env.example...")
            import secrets
            content = env_example.read_text()
            if "your-secret-key-change-this-in-production" in content:
                secret_key = secrets.token_urlsafe(32)
                content = content.replace(
                    "your-secret-key-change-this-in-production",
                    secret_key
                )
            env_file.write_text(content)
            print("  ✅ .env creado")
            return True
        else:
            print("  ❌ env.example no encontrado")
            return False

def check_imports():
    """Verificar imports críticos."""
    print("\n🔍 Verificando imports...")
    
    critical_modules = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
    ]
    
    all_ok = True
    for module_name, display_name in critical_modules:
        try:
            __import__(module_name)
            print(f"  ✅ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} - NO INSTALADO")
            all_ok = False
    
    return all_ok

def check_config():
    """Verificar configuración."""
    print("\n⚙️  Verificando configuración...")
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from config.settings import settings
        print("  ✅ Configuración cargada")
        print(f"     - Host: {settings.api_host}")
        print(f"     - Puerto: {settings.api_port}")
        print(f"     - Entorno: {settings.environment}")
        return True
    except Exception as e:
        print(f"  ⚠️  Error cargando configuración: {e}")
        return False

def check_main_app():
    """Verificar que main.py pueda importarse."""
    print("\n🚀 Verificando aplicación principal...")
    try:
        # Solo verificar que el archivo existe y es válido
        main_file = Path(__file__).parent / "main.py"
        if main_file.exists():
            print("  ✅ main.py existe")
            # Intentar leer sin ejecutar
            content = main_file.read_text(encoding='utf-8')
            if "FastAPI" in content or "app" in content:
                print("  ✅ main.py parece válido")
                return True
            else:
                print("  ⚠️  main.py puede tener problemas")
                return False
        else:
            print("  ❌ main.py no existe")
            return False
    except Exception as e:
        print(f"  ⚠️  Error verificando main.py: {e}")
        return False

def main():
    """Función principal."""
    print("=" * 60)
    print("🔍 Verificación del Sistema Bulk TruthGPT")
    print("=" * 60)
    print()
    
    results = {
        "directories": check_directories(),
        "env_file": check_env_file(),
        "imports": check_imports(),
        "config": check_config(),
        "main_app": check_main_app()
    }
    
    print()
    print("=" * 60)
    print("📊 Resumen de Verificación")
    print("=" * 60)
    
    all_passed = True
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False
    
    print()
    print("=" * 60)
    
    if all_passed:
        print("✅ ¡Sistema listo para usar!")
        print()
        print("📋 Próximos pasos:")
        print("  1. python start.py  - Iniciar servidor")
        print("  2. O: uvicorn bulk_truthgpt.main:app --reload")
        print("  3. Visita: http://localhost:8000/docs")
        return 0
    else:
        print("⚠️  Sistema necesita configuración adicional")
        print()
        print("📋 Acciones recomendadas:")
        if not results["imports"]:
            print("  1. pip install -r requirements.txt")
        if not results["env_file"]:
            print("  2. Configura el archivo .env")
        print("  3. Ejecuta: python setup.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
































