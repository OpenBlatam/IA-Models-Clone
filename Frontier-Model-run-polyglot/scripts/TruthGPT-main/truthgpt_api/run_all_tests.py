#!/usr/bin/env python3
"""
Script para ejecutar TODOS los tests de TruthGPT API
=====================================================
"""

import subprocess
import sys
import os
import time

def print_header(text):
    """Imprimir header con estilo."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")

def run_tests(test_type, test_path, description):
    """Ejecutar un grupo de tests."""
    print_header(f"🧪 {description}")
    print(f"Ejecutando: pytest {test_path} -v")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'],
            cwd=os.path.dirname(__file__),
            timeout=600  # 10 minutos máximo
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout ejecutando {test_type}")
        return False
    except Exception as e:
        print(f"❌ Error ejecutando {test_type}: {e}")
        return False

def main():
    """Ejecutar todos los tests."""
    print_header("🚀 Ejecutando TODOS los Tests de TruthGPT API")
    
    print("⚠️  IMPORTANTE:")
    print("   1. Asegúrate de que el servidor API esté corriendo:")
    print("      python start_server.py")
    print("   2. Los tests unitarios NO requieren servidor")
    print("   3. Los tests de integración y E2E SÍ requieren servidor")
    print()
    
    response = input("¿El servidor está corriendo? (s/n): ").lower()
    server_running = response == 's'
    
    if not server_running:
        print("\n⚠️  Ejecutando solo tests unitarios (sin servidor)...")
        print("   Los tests de integración y E2E se omitirán.\n")
    
    results = {}
    
    # 1. Tests unitarios (no requieren servidor)
    print_header("📦 Tests Unitarios (no requieren servidor)")
    results['unit_models'] = run_tests(
        'unit_models',
        'tests/test_unit_models.py',
        'Tests Unitarios - Models y Layers'
    )
    
    results['unit_utils'] = run_tests(
        'unit_utils',
        'tests/test_unit_utils.py',
        'Tests Unitarios - Utilidades'
    )
    
    results['unit_helpers'] = run_tests(
        'unit_helpers',
        'tests/test_unit_api_helpers.py',
        'Tests Unitarios - API Helpers'
    )
    
    if server_running:
        # 2. Tests de endpoints
        print_header("🌐 Tests de Endpoints API")
        results['endpoints'] = run_tests(
            'endpoints',
            'tests/test_api_endpoints.py',
            'Tests de Endpoints de la API'
        )
        
        # 3. Tests de layers
        print_header("🧩 Tests de Layers")
        results['layers'] = run_tests(
            'layers',
            'tests/test_layers.py',
            'Tests de Diferentes Tipos de Layers'
        )
        
        # 4. Tests de optimizers
        print_header("⚙️  Tests de Optimizers")
        results['optimizers'] = run_tests(
            'optimizers',
            'tests/test_optimizers.py',
            'Tests de Optimizers'
        )
        
        # 5. Tests de losses
        print_header("📉 Tests de Loss Functions")
        results['losses'] = run_tests(
            'losses',
            'tests/test_losses.py',
            'Tests de Loss Functions'
        )
        
        # 6. Tests de integración
        print_header("🔗 Tests de Integración")
        results['integration'] = run_tests(
            'integration',
            'tests/test_integration.py',
            'Tests de Integración entre Componentes'
        )
        
        # 7. Tests de validación
        print_header("✅ Tests de Validación")
        results['validation'] = run_tests(
            'validation',
            'tests/test_validation.py',
            'Tests de Validación de Datos'
        )
        
        # 8. Tests avanzados de layers
        print_header("🔬 Tests Avanzados de Layers")
        results['advanced_layers'] = run_tests(
            'advanced_layers',
            'tests/test_advanced_layers.py',
            'Tests Avanzados de Layers (Conv2D, RNN, Attention, etc.)'
        )
        
        # 9. Tests de tipos de datos
        print_header("📊 Tests de Tipos de Datos")
        results['data_types'] = run_tests(
            'data_types',
            'tests/test_data_types.py',
            'Tests para Diferentes Tipos de Datos'
        )
        
        # 10. Tests de manejo de errores
        print_header("⚠️  Tests de Manejo de Errores")
        results['error_handling'] = run_tests(
            'error_handling',
            'tests/test_error_handling.py',
            'Tests de Manejo de Errores'
        )
        
        # 11. Tests edge cases
        print_header("🎯 Tests de Casos Edge")
        results['edge_cases'] = run_tests(
            'edge_cases',
            'tests/test_edge_cases.py',
            'Tests de Casos Edge y Validaciones'
        )
        
        # 12. Tests de concurrencia
        print_header("⚡ Tests de Concurrencia")
        results['concurrency'] = run_tests(
            'concurrency',
            'tests/test_concurrency.py',
            'Tests de Concurrencia y Threads'
        )
        
        # 13. Tests de serialización
        print_header("💾 Tests de Serialización")
        results['serialization'] = run_tests(
            'serialization',
            'tests/test_model_serialization.py',
            'Tests de Serialización de Modelos'
        )
        
        # 14. Tests de seguridad
        print_header("🔒 Tests de Seguridad")
        results['security'] = run_tests(
            'security',
            'tests/test_security.py',
            'Tests de Seguridad y Validación'
        )
        
        # 15. Tests de compatibilidad
        print_header("🔌 Tests de Compatibilidad")
        results['compatibility'] = run_tests(
            'compatibility',
            'tests/test_compatibility.py',
            'Tests de Compatibilidad con Diferentes Clientes'
        )
        
        # 16. Tests E2E (pueden ser lentos)
        print_header("🔄 Tests End-to-End")
        print("⚠️  Los tests E2E pueden tomar varios minutos...")
        response = input("¿Ejecutar tests E2E? (s/n): ").lower()
        if response == 's':
            results['e2e'] = run_tests(
                'e2e',
                'tests/test_e2e.py',
                'Tests End-to-End Completos'
            )
        else:
            results['e2e'] = None
            print("⏭️  Tests E2E omitidos")
        
        # 17. Tests de rendimiento (opcional, pueden ser lentos)
        print_header("⚡ Tests de Rendimiento")
        print("⚠️  Los tests de rendimiento pueden tomar tiempo...")
        response = input("¿Ejecutar tests de rendimiento? (s/n): ").lower()
        if response == 's':
            results['performance'] = run_tests(
                'performance',
                'tests/test_performance.py',
                'Tests de Rendimiento y Latencia'
            )
        else:
            results['performance'] = None
            print("⏭️  Tests de rendimiento omitidos")
    
    # Resumen final
    print_header("📊 Resumen de Tests")
    
    total = 0
    passed = 0
    failed = 0
    skipped = 0
    
    for test_name, result in results.items():
        total += 1
        if result is True:
            passed += 1
            print(f"✅ {test_name}: PASÓ")
        elif result is False:
            failed += 1
            print(f"❌ {test_name}: FALLÓ")
        else:
            skipped += 1
            print(f"⏭️  {test_name}: OMITIDO")
    
    print()
    print(f"Total: {total}")
    print(f"✅ Pasaron: {passed}")
    print(f"❌ Fallaron: {failed}")
    print(f"⏭️  Omitidos: {skipped}")
    print()
    
    if failed == 0:
        print("🎉 ¡Todos los tests ejecutados pasaron!")
        sys.exit(0)
    else:
        print(f"⚠️  {failed} grupo(s) de tests fallaron")
        sys.exit(1)

if __name__ == "__main__":
    main()

