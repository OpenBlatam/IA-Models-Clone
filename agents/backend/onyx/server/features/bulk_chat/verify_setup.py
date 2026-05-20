#!/usr/bin/env python3
"""
Script de Verificación - Bulk Chat
===================================

Verifica que el sistema esté listo para usar.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    """Verificar versión de Python."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ requerido")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Verificar dependencias principales."""
    required = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic'),
    ]
    
    missing = []
    for module_name, display_name in required:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name:15} v{version}")
        except ImportError:
            print(f"❌ {display_name:15} - No instalado")
            missing.append(module_name)
    
    # Dependencias opcionales con información de versión
    optional = [
        ('openai', 'OpenAI'),
        ('anthropic', 'Anthropic'),
        ('redis', 'Redis'),
        ('dotenv', 'python-dotenv'),
    ]
    
    optional_missing = []
    for module_name, display_name in optional:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {display_name:15} v{version} (opcional)")
        except ImportError:
            print(f"⚠️  {display_name:15} (opcional) - No instalado")
            optional_missing.append(module_name)
    
    if missing:
        print(f"\n❌ Dependencias faltantes: {', '.join(missing)}")
        print("   Instala con: pip install -r requirements.txt")
        return False
    
    if optional_missing and len(optional_missing) == len(optional):
        print("\n💡 Tip: Algunas dependencias opcionales no están instaladas.")
        print("   El sistema funcionará, pero algunas características estarán limitadas.")
    
    return True

def check_directories():
    """Verificar directorios necesarios."""
    base = Path(__file__).parent
    dirs = ['sessions', 'backups']
    
    all_ok = True
    for dir_name in dirs:
        dir_path = base / dir_name
        if not dir_path.exists():
            try:
                dir_path.mkdir(exist_ok=True)
                print(f"✅ Directorio '{dir_name}' creado")
            except Exception as e:
                print(f"❌ No se pudo crear directorio '{dir_name}': {e}")
                all_ok = False
        else:
            print(f"✅ Directorio '{dir_name}' existe")
    
    return all_ok

def check_imports():
    """Verificar que los módulos se puedan importar."""
    modules_to_check = [
        ('bulk_chat', 'ContinuousChatEngine', 'ChatSession'),
        ('bulk_chat.core', 'chat_engine', 'chat_session'),
        ('bulk_chat.api', 'chat_api', 'ChatAPI'),
        ('bulk_chat.config', 'chat_config', 'ChatConfig'),
    ]
    
    all_ok = True
    for module_path, *items in modules_to_check:
        try:
            module = __import__(module_path, fromlist=items)
            for item in items:
                if not hasattr(module, item):
                    print(f"⚠️  {module_path}.{item} no encontrado")
                    all_ok = False
            if module_path == 'bulk_chat':
                print(f"✅ Módulo principal '{module_path}' importable")
        except ImportError as e:
            print(f"❌ Error al importar '{module_path}': {e}")
            all_ok = False
    
    return all_ok

def check_env_file():
    """Verificar archivo .env y variables de entorno."""
    base = Path(__file__).parent
    env_file = base / '.env'
    env_example = base / '.env.example'
    
    env_exists = env_file.exists()
    has_api_key = bool(os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY'))
    
    if env_exists:
        print("✅ Archivo .env existe")
        # Verificar si tiene API key
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if 'API_KEY' in content or 'OPENAI_API_KEY' in content:
                    print("✅ Archivo .env contiene configuración de API")
        except Exception:
            pass
        return True
    elif has_api_key:
        print("✅ Variables de entorno configuradas (OPENAI_API_KEY o ANTHROPIC_API_KEY)")
        return True
    elif env_example.exists():
        print("⚠️  Archivo .env no existe (usa .env.example como base)")
        print("💡 El sistema funcionará en modo 'mock' sin API key")
        return True
    else:
        print("⚠️  Archivo .env no existe (opcional)")
        print("💡 El sistema funcionará en modo 'mock' sin API key")
        return True

def check_port_availability():
    """Verificar si el puerto por defecto está disponible."""
    import socket
    
    default_port = 8006
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('localhost', default_port))
        sock.close()
        if result == 0:
            print(f"⚠️  Puerto {default_port} está en uso")
            print(f"💡 Usa --port para especificar otro puerto")
            return True  # No es crítico, solo advertencia
        else:
            print(f"✅ Puerto {default_port} disponible")
            return True
    except Exception:
        return True  # No crítico

def main():
    """Ejecutar todas las verificaciones."""
    print("=" * 60)
    print("🔍 Verificación de Setup - Bulk Chat")
    print("=" * 60)
    print()
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Dependencias", check_dependencies),
        ("Directorios", check_directories),
        ("Importaciones", check_imports),
        ("Configuración", check_env_file),
        ("Puerto", check_port_availability),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n📋 {name}:")
        print("-" * 60)
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 60)
    print("📊 Resumen:")
    print("=" * 60)
    
    all_ok = True
    critical_failed = False
    
    for name, result in results:
        status = "✅ OK" if result else "❌ FALLO"
        print(f"{status} - {name}")
        if not result:
            all_ok = False
            # Algunos checks no son críticos
            if name in ["Versión de Python", "Dependencias", "Importaciones"]:
                critical_failed = True
    
    print()
    if all_ok and not critical_failed:
        print("✅ ¡Todo está listo! Puedes iniciar el servidor con:")
        print()
        print("   python -m bulk_chat.main")
        print("   o")
        print("   python start.py")
        print()
        print("💡 Para usar con API real:")
        print("   export OPENAI_API_KEY=tu-api-key")
        print("   python -m bulk_chat.main --llm-provider openai")
        print()
        print("💡 Para modo de prueba (sin API key):")
        print("   python -m bulk_chat.main --llm-provider mock")
    elif critical_failed:
        print("❌ Hay problemas críticos que resolver antes de usar el sistema.")
        print("   Revisa los errores arriba y corrige los problemas.")
        print()
        print("💡 Comando útil:")
        print("   pip install -r requirements.txt")
    else:
        print("⚠️  Hay algunas advertencias, pero el sistema debería funcionar.")
        print("   Revisa las advertencias arriba.")
    
    return 0 if (all_ok and not critical_failed) else 1

if __name__ == "__main__":
    sys.exit(main())

