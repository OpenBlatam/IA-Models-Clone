"""
Script de inicio rápido para Analizador de Documentos
"""

import os
import sys
from pathlib import Path


def check_dependencies() -> bool:
    """Verificar dependencias principales"""
    print("Verificando dependencias...")
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "fastapi": "FastAPI",
        "uvicorn": "Uvicorn",
        "pydantic": "Pydantic"
    }
    
    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} instalado")
        except ImportError:
            print(f"✗ {name} no encontrado")
            missing.append(name)
    
    if missing:
        print(f"\n⚠ Dependencias faltantes: {', '.join(missing)}")
        print("Instala las dependencias con: pip install -r requirements.txt")
        return False
    
    return True


def check_structure() -> bool:
    """Verificar estructura de directorios"""
    print("\nVerificando estructura...")
    required_dirs = ["core", "api", "training", "config"]
    base_path = Path(__file__).parent
    missing_dirs = []
    
    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if dir_path.exists():
            print(f"✓ {dir_name}/ existe")
        else:
            print(f"✗ {dir_name}/ no encontrado")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\n⚠ Directorios faltantes: {', '.join(missing_dirs)}")
        return False
    
    return True


def check_environment() -> None:
    """Verificar variables de entorno y configuración"""
    print("\nVerificando configuración...")
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"✓ Host: {host}")
    print(f"✓ Port: {port}")
    
    # Verificar si hay archivo .env
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print("✓ Archivo .env encontrado")
    else:
        print("ℹ Archivo .env no encontrado (usando valores por defecto)")


def main():
    """Iniciar servidor"""
    print("=" * 60)
    print("Analizador de Documentos Inteligente v3.8.0")
    print("=" * 60)
    print()
    
    # Verificar dependencias
    if not check_dependencies():
        print("\n❌ Error: Dependencias faltantes. Por favor instálalas antes de continuar.")
        sys.exit(1)
    
    # Verificar estructura
    if not check_structure():
        print("\n⚠ Advertencia: Algunos directorios no existen. Puede haber problemas.")
        response = input("\n¿Continuar de todos modos? (s/N): ")
        if response.lower() != 's':
            sys.exit(1)
    
    # Verificar configuración
    check_environment()
    
    print("\n" + "=" * 60)
    print("Iniciando servidor...")
    print("=" * 60)
    print(f"📚 Documentación: http://localhost:{os.getenv('PORT', 8000)}/docs")
    print(f"📊 Dashboard: http://localhost:{os.getenv('PORT', 8000)}/dashboard")
    print(f"🏥 Health: http://localhost:{os.getenv('PORT', 8000)}/api/analizador-documentos/health")
    print()
    
    # Iniciar servidor
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()




