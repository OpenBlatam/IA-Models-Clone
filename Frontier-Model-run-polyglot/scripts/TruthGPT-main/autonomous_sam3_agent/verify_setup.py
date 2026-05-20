"""
Verification Script for Autonomous SAM3 Agent
==============================================

Script para verificar que todo está configurado correctamente.
"""

import sys
import os
from pathlib import Path

def check_imports():
    """Verificar que todos los imports funcionan."""
    print("🔍 Verificando imports...")
    
    try:
        import asyncio
        print("  ✅ asyncio")
    except ImportError:
        print("  ❌ asyncio - NO DISPONIBLE")
        return False
    
    try:
        import httpx
        print("  ✅ httpx")
    except ImportError:
        print("  ❌ httpx - Instalar con: pip install httpx")
        return False
    
    try:
        import yaml
        print("  ✅ yaml")
    except ImportError:
        print("  ❌ yaml - Instalar con: pip install pyyaml")
        return False
    
    try:
        import torch
        print(f"  ✅ torch (versión: {torch.__version__})")
    except ImportError:
        print("  ❌ torch - Instalar con: pip install torch")
        return False
    
    try:
        from PIL import Image
        print("  ✅ PIL/Pillow")
    except ImportError:
        print("  ❌ PIL/Pillow - Instalar con: pip install pillow")
        return False
    
    return True

def check_sam3():
    """Verificar que SAM3 está disponible."""
    print("\n🔍 Verificando SAM3...")
    
    sam3_path = Path(__file__).parent.parent / "sam3-main"
    if not sam3_path.exists():
        print(f"  ⚠️  sam3-main no encontrado en: {sam3_path}")
        print("     El agente intentará importar desde el path configurado")
        return False
    
    print(f"  ✅ sam3-main encontrado en: {sam3_path}")
    
    # Intentar importar
    sys.path.insert(0, str(sam3_path))
    try:
        from sam3.model_builder import build_sam3_image_model
        print("  ✅ SAM3 imports funcionando")
        return True
    except ImportError as e:
        print(f"  ⚠️  SAM3 imports fallaron: {e}")
        print("     Asegúrate de instalar las dependencias de sam3-main")
        return False

def check_openrouter():
    """Verificar configuración de OpenRouter."""
    print("\n🔍 Verificando OpenRouter...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ⚠️  OPENROUTER_API_KEY no configurado")
        print("     Configura con: export OPENROUTER_API_KEY='tu-api-key'")
        return False
    
    print(f"  ✅ OPENROUTER_API_KEY configurado (longitud: {len(api_key)})")
    return True

def check_structure():
    """Verificar estructura de archivos."""
    print("\n🔍 Verificando estructura de archivos...")
    
    base_path = Path(__file__).parent
    required_files = [
        "main.py",
        "config.yaml",
        "core/agent_core.py",
        "core/task_manager.py",
        "core/parallel_executor.py",
        "infrastructure/openrouter_client.py",
        "infrastructure/sam3_client.py",
        "system_prompts/system_prompt.txt",
        "system_prompts/system_prompt_iterative_checking.txt",
    ]
    
    all_ok = True
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - NO ENCONTRADO")
            all_ok = False
    
    return all_ok

def check_cuda():
    """Verificar disponibilidad de CUDA."""
    print("\n🔍 Verificando CUDA...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ CUDA disponible")
            print(f"     Dispositivo: {torch.cuda.get_device_name(0)}")
            print(f"     Versión CUDA: {torch.version.cuda}")
            return True
        else:
            print("  ⚠️  CUDA no disponible - El agente usará CPU (más lento)")
            return False
    except Exception as e:
        print(f"  ⚠️  Error verificando CUDA: {e}")
        return False

def main():
    """Ejecutar todas las verificaciones."""
    print("=" * 60)
    print("Verificación de Configuración - Autonomous SAM3 Agent")
    print("=" * 60)
    
    results = {
        "imports": check_imports(),
        "structure": check_structure(),
        "sam3": check_sam3(),
        "openrouter": check_openrouter(),
        "cuda": check_cuda(),
    }
    
    print("\n" + "=" * 60)
    print("Resumen:")
    print("=" * 60)
    
    for check, result in results.items():
        status = "✅ OK" if result else "⚠️  ADVERTENCIA"
        print(f"  {check}: {status}")
    
    all_critical = results["imports"] and results["structure"]
    
    if all_critical:
        print("\n✅ Configuración básica correcta. El agente debería funcionar.")
        if not results["openrouter"]:
            print("⚠️  Configura OPENROUTER_API_KEY antes de usar el agente.")
        if not results["sam3"]:
            print("⚠️  Verifica la instalación de SAM3 antes de procesar imágenes.")
    else:
        print("\n❌ Hay problemas críticos que deben resolverse antes de usar el agente.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
