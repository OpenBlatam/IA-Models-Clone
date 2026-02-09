#!/usr/bin/env python3
"""
Script de Inicio Rápido para Bulk TruthGPT
==========================================

Script simple para iniciar el servidor con configuración por defecto.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_env_file():
    """Verificar que existe .env."""
    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print("⚠️  Archivo .env no encontrado")
        print("   Ejecutando setup.py para crear configuración...")
        subprocess.run([sys.executable, "setup.py"], cwd=Path(__file__).parent)
        print()

def start_server():
    """Iniciar el servidor."""
    base_dir = Path(__file__).parent
    
    # Cambiar al directorio del proyecto
    os.chdir(base_dir)
    
    # Verificar .env
    check_env_file()
    
    # Determinar el modo de ejecución
    import_arg = "bulk_truthgpt.main:app"
    
    # Obtener argumentos desde variables de entorno o usar defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("DEBUG", "false").lower() == "true"
    
    print("=" * 60)
    print("🚀 Iniciando Bulk TruthGPT Server")
    print("=" * 60)
    print(f"📍 Host: {host}")
    print(f"🔌 Puerto: {port}")
    print(f"🔄 Reload: {reload}")
    print("=" * 60)
    print()
    print("📖 Documentación disponible en:")
    print(f"   - Swagger UI: http://localhost:{port}/docs")
    print(f"   - ReDoc: http://localhost:{port}/redoc")
    print()
    print("💡 Para detener el servidor, presiona Ctrl+C")
    print()
    
    # Construir comando uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        import_arg,
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    # Ejecutar servidor
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n\n👋 Servidor detenido")
    except Exception as e:
        print(f"\n❌ Error iniciando servidor: {e}")
        print("\n💡 Alternativa: ejecuta manualmente:")
        print(f"   uvicorn {import_arg} --host {host} --port {port}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(start_server())
































