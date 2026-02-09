"""
Check System
============

Script para verificar el sistema.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def check_dependencies():
    """Verificar dependencias."""
    print("📦 Verificando dependencias...")
    
    required = [
        "torch",
        "transformers",
        "fastapi",
        "sqlalchemy",
        "sentence_transformers"
    ]
    
    optional = [
        "diffusers",
        "gradio",
        "wandb",
        "faiss",
        "onnxruntime"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (REQUERIDO)")
            missing_required.append(package)
    
    for package in optional:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✅ {package} (opcional)")
        except ImportError:
            print(f"  ⚠️  {package} (opcional, no instalado)")
            missing_optional.append(package)
    
    return len(missing_required) == 0

def check_gpu():
    """Verificar GPU."""
    print("\n🎮 Verificando GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA disponible")
            print(f"  ✅ Dispositivos: {torch.cuda.device_count()}")
            print(f"  ✅ Dispositivo actual: {torch.cuda.current_device()}")
            print(f"  ✅ Nombre: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ⚠️  CUDA no disponible (usando CPU)")
            return False
    except Exception as e:
        print(f"  ❌ Error verificando GPU: {str(e)}")
        return False

def check_database():
    """Verificar base de datos."""
    print("\n🗄️  Verificando base de datos...")
    
    try:
        from database.session import get_db_session
        from database.models import Base, engine
        
        # Intentar conectar
        print("  ✅ Conexión a base de datos OK")
        return True
    except Exception as e:
        print(f"  ❌ Error con base de datos: {str(e)}")
        return False

def check_config():
    """Verificar configuración."""
    print("\n⚙️  Verificando configuración...")
    
    try:
        from config.settings import get_settings
        settings = get_settings()
        
        if settings.openrouter_api_key:
            print("  ✅ OPENROUTER_API_KEY configurada")
        else:
            print("  ⚠️  OPENROUTER_API_KEY no configurada")
        
        print(f"  ✅ Modelo por defecto: {settings.default_model}")
        print(f"  ✅ Categorías soportadas: {len(settings.supported_categories)}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error verificando configuración: {str(e)}")
        return False

def main():
    """Verificar sistema completo."""
    print("=" * 60)
    print("VERIFICACIÓN DEL SISTEMA")
    print("=" * 60)
    
    results = {
        "dependencies": check_dependencies(),
        "gpu": check_gpu(),
        "database": check_database(),
        "config": check_config()
    }
    
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    
    all_ok = all(results.values())
    
    for check, status in results.items():
        status_str = "✅ OK" if status else "❌ FALLO"
        print(f"{check.upper()}: {status_str}")
    
    if all_ok:
        print("\n✨ Sistema listo para usar!")
    else:
        print("\n⚠️  Algunos checks fallaron. Revisa los errores arriba.")
    
    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)




