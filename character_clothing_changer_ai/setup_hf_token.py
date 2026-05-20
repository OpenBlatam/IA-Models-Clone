"""
Script de Configuración Rápida - Token de HuggingFace
======================================================

Ayuda a configurar el token de HuggingFace para acceder al modelo Flux2.
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("🔧 Configuración de Token de HuggingFace")
    print("=" * 60)
    print()
    
    # Verificar si ya existe un token
    existing_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if existing_token:
        print(f"✅ Token encontrado: {existing_token[:10]}...{existing_token[-4:]}")
        response = input("\n¿Deseas configurar un nuevo token? (s/n): ").lower()
        if response != 's':
            print("\n✅ Usando token existente.")
            return
        print()
    
    print("📋 Pasos para obtener un token:")
    print()
    print("1. Ve a: https://huggingface.co/settings/tokens")
    print("2. Haz clic en 'New token'")
    print("3. Selecciona 'Read' como permiso")
    print("4. Copia el token generado")
    print()
    print("5. Acepta los términos del modelo:")
    print("   https://huggingface.co/black-forest-labs/flux2-dev")
    print("   (Haz clic en 'Agree and access repository')")
    print()
    
    token = input("Pega tu token aquí (o presiona Enter para salir): ").strip()
    
    if not token:
        print("\n❌ No se proporcionó token. Saliendo...")
        return
    
    print()
    print("🔧 Configurando token...")
    print()
    
    # Detectar sistema operativo
    is_windows = sys.platform == "win32"
    
    if is_windows:
        # Windows
        print("Para Windows, ejecuta este comando:")
        print(f'  set HUGGINGFACE_TOKEN={token}')
        print()
        print("O para configurarlo permanentemente:")
        print(f'  setx HUGGINGFACE_TOKEN "{token}"')
        print()
        print("Luego reinicia la terminal y ejecuta:")
        print("  python run_server.py")
    else:
        # Linux/Mac
        print("Para Linux/Mac, ejecuta este comando:")
        print(f'  export HUGGINGFACE_TOKEN={token}')
        print()
        print("O para configurarlo permanentemente, agrega a ~/.bashrc o ~/.zshrc:")
        print(f'  export HUGGINGFACE_TOKEN={token}')
        print()
        print("Luego ejecuta:")
        print("  source ~/.bashrc  # o source ~/.zshrc")
        print("  python run_server.py")
    
    print()
    print("=" * 60)
    print("✅ Configuración completada")
    print("=" * 60)
    print()
    print("💡 Tip: El token se guardará solo para esta sesión.")
    print("   Para guardarlo permanentemente, usa los comandos mostrados arriba.")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Operación cancelada por el usuario.")
        sys.exit(0)


