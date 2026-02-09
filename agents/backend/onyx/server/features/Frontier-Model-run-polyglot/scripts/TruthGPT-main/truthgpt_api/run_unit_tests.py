#!/usr/bin/env python3
"""
Script para ejecutar SOLO los tests unitarios
==============================================
Los tests unitarios NO requieren que el servidor esté corriendo
"""

import subprocess
import sys
import os

def main():
    """Ejecutar tests unitarios."""
    print("=" * 70)
    print("🧪 Ejecutando Tests Unitarios de TruthGPT API")
    print("=" * 70)
    print()
    print("ℹ️  Los tests unitarios NO requieren que el servidor esté corriendo")
    print("   Solo prueban componentes individuales (models, layers, utils, etc.)")
    print()
    
    # Directorio de tests
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    # Tests unitarios
    unit_tests = [
        'test_unit_models.py',
        'test_unit_utils.py',
        'test_unit_api_helpers.py'
    ]
    
    results = {}
    
    for test_file in unit_tests:
        test_path = os.path.join(test_dir, test_file)
        test_name = test_file.replace('test_', '').replace('.py', '')
        
        print("\n" + "=" * 70)
        print(f"🧪 Ejecutando: {test_file}")
        print("=" * 70)
        print()
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'],
                cwd=os.path.dirname(__file__),
                timeout=300  # 5 minutos máximo
            )
            results[test_name] = result.returncode == 0
            
            if result.returncode == 0:
                print(f"\n✅ {test_file}: PASÓ")
            else:
                print(f"\n❌ {test_file}: FALLÓ")
                
        except subprocess.TimeoutExpired:
            print(f"\n⏱️  {test_file}: TIMEOUT")
            results[test_name] = False
        except Exception as e:
            print(f"\n❌ {test_file}: ERROR - {e}")
            results[test_name] = False
    
    # Resumen
    print("\n" + "=" * 70)
    print("📊 Resumen de Tests Unitarios")
    print("=" * 70)
    print()
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"  {status}: {test_name}")
    
    print()
    print(f"Total: {total}")
    print(f"✅ Pasaron: {passed}")
    print(f"❌ Fallaron: {failed}")
    print()
    
    if failed == 0:
        print("🎉 ¡Todos los tests unitarios pasaron!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"⚠️  {failed} grupo(s) de tests fallaron")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()











