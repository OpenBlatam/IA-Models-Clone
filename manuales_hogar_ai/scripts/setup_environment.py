"""
Setup Environment
=================

Script para configurar el entorno de desarrollo.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_environment():
    """Configurar entorno de desarrollo."""
    print("🚀 Configurando entorno de desarrollo...")
    
    # Crear directorios necesarios
    directories = [
        "models",
        "checkpoints",
        "outputs",
        "logs",
        "experiments",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directorio creado: {directory}")
    
    # Verificar variables de entorno
    required_vars = ["OPENROUTER_API_KEY"]
    optional_vars = [
        "DATABASE_URL",
        "WANDB_API_KEY",
        "NOTIFICATION_EMAIL_SENDER"
    ]
    
    print("\n📋 Verificando variables de entorno...")
    for var in required_vars:
        if not os.getenv(var):
            print(f"⚠️  Variable requerida no configurada: {var}")
        else:
            print(f"✅ {var} configurada")
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} configurada")
        else:
            print(f"ℹ️  {var} no configurada (opcional)")
    
    # Instalar dependencias
    print("\n📦 Instalando dependencias...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencias instaladas")
    except subprocess.CalledProcessError:
        print("❌ Error instalando dependencias")
    
    # Inicializar base de datos
    print("\n🗄️  Inicializando base de datos...")
    try:
        from scripts.init_db import main as init_db
        init_db()
        print("✅ Base de datos inicializada")
    except Exception as e:
        print(f"⚠️  Error inicializando BD: {str(e)}")
    
    print("\n✨ Configuración completada!")

if __name__ == "__main__":
    setup_environment()




