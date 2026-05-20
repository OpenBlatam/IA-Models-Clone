#!/usr/bin/env python3
"""
Script para ejecutar tests E2E de la API TruthGPT
===================================================
"""

import subprocess
import sys
import os

def main():
    """Ejecutar tests E2E."""
    print("=" * 60)
    print("🧪 Ejecutando Tests E2E de TruthGPT API")
    print("=" * 60)
    print()
    print("⚠️  IMPORTANTE: Asegúrate de que el servidor esté corriendo:")
    print("   python start_server.py")
    print()
    print("📝 Los tests E2E verifican flujos completos:")
    print("   - Creación → Compilación → Entrenamiento → Evaluación → Predicción")
    print("   - Modelos CNN completos")
    print("   - Modelos RNN/LSTM completos")
    print("   - Múltiples modelos simultáneos")
    print("   - Persistencia de modelos")
    print("   - Recuperación de errores")
    print()
    input("Presiona Enter cuando el servidor esté listo...")
    print()
    
    # Cambiar al directorio de tests
    test_file = os.path.join(os.path.dirname(__file__), 'tests', 'test_e2e.py')
    
    # Ejecutar pytest con output verbose y prints
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_file, '-v', '-s', '--tb=short'],
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











