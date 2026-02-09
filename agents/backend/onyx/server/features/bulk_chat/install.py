#!/usr/bin/env python3
"""
Script de Instalación - Bulk Chat
==================================

Instala automáticamente las dependencias y configura el sistema.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Ejecutar comando y mostrar progreso."""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ {description} completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}:")
        print(f"   {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def check_python():
    """Verificar versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ requerido. Versión actual: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Instalar dependencias desde requirements.txt."""
    base = Path(__file__).parent
    requirements = base / 'requirements.txt'
    
    if not requirements.exists():
        print("⚠️  requirements.txt no encontrado")
        return False
    
    # Usar pip para instalar
    pip_cmd = f"{sys.executable} -m pip install -r {requirements}"
    return run_command(pip_cmd, "Instalando dependencias")

def create_directories():
    """Crear directorios necesarios."""
    base = Path(__file__).parent
    dirs = ['sessions', 'backups', 'logs']
    
    print("\n📁 Creando directorios...")
    for dir_name in dirs:
        dir_path = base / dir_name
        try:
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Directorio '{dir_name}' listo")
        except Exception as e:
            print(f"⚠️  No se pudo crear '{dir_name}': {e}")
    
    return True

def create_env_example():
    """Crear archivo .env.example si no existe."""
    base = Path(__file__).parent
    env_example = base / '.env.example'
    
    if env_example.exists():
        print("✅ .env.example ya existe")
        return True
    
    env_content = """# Bulk Chat - Configuración de Variables de Entorno
# Copia este archivo a .env y configura tus valores

# API Settings
CHAT_API_HOST=0.0.0.0
CHAT_API_PORT=8006

# LLM Settings
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=tu-api-key-aqui

# Chat Behavior
AUTO_CONTINUE=true
RESPONSE_INTERVAL=2.0
"""
    
    try:
        with open(env_example, 'w', encoding='utf-8') as f:
            f.write(env_content)
        print("✅ .env.example creado")
        return True
    except Exception as e:
        print(f"⚠️  No se pudo crear .env.example: {e}")
        return True  # No crítico

def verify_installation():
    """Verificar que la instalación fue exitosa."""
    print("\n🔍 Verificando instalación...")
    try:
        # Intentar importar módulos principales
        import fastapi
        import uvicorn
        import pydantic
        print("✅ Dependencias principales instaladas")
        return True
    except ImportError as e:
        print(f"❌ Error al importar: {e}")
        return False

def main():
    """Ejecutar instalación."""
    print("=" * 60)
    print("🚀 Instalación de Bulk Chat")
    print("=" * 60)
    
    # Verificar Python
    if not check_python():
        print("\n❌ Instalación cancelada")
        sys.exit(1)
    
    # Instalar dependencias
    if not install_dependencies():
        print("\n⚠️  Hubo problemas instalando dependencias")
        print("   Intenta manualmente: pip install -r requirements.txt")
    
    # Crear directorios
    create_directories()
    
    # Crear .env.example
    create_env_example()
    
    # Verificar
    if verify_installation():
        print("\n" + "=" * 60)
        print("✅ Instalación completada exitosamente!")
        print("=" * 60)
        print("\n📋 Próximos pasos:")
        print("   1. Ejecuta: python verify_setup.py")
        print("   2. (Opcional) Crea .env con tus API keys")
        print("   3. Inicia el servidor: python -m bulk_chat.main")
        print("\n💡 Para modo de prueba sin API key:")
        print("   python -m bulk_chat.main --llm-provider mock")
        return 0
    else:
        print("\n⚠️  La instalación puede tener problemas")
        print("   Ejecuta: python verify_setup.py para diagnosticar")
        return 1

if __name__ == "__main__":
    sys.exit(main())
















