#!/usr/bin/env python3
"""
Script para ejecutar tests de la API TruthGPT
==============================================
"""

import subprocess
import sys
import os

def main():
    """Ejecutar tests."""
    print("=" * 60)
    print("🧪 Ejecutando Tests de TruthGPT API")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANTE: Asegúrate de que el servidor esté corriendo:")
    print("   python start_server.py")
    print()
    input("Presiona Enter cuando el servidor esté listo...")
    print()
    
    # Cambiar al directorio de tests
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Ejecutar pytest con todos los tests
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_dir, '-v', '--tb=short', '--maxfail=5'],
            cwd=os.path.dirname(__file__)
        )
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nTests interrumpidos por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error ejecutando tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

