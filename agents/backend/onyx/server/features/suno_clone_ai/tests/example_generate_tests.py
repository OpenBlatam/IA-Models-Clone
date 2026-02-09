"""
Ejemplo de uso del generador de casos de prueba

Este script demuestra cómo usar el generador de casos de prueba
para generar tests automáticamente para funciones.
"""

import sys
from pathlib import Path

# Agregar el directorio raíz al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.test_case_generator import (
    generate_tests_for_function,
    TestCaseGenerator,
    TestCodeGenerator,
    FunctionAnalyzer,
    TestType
)

# Importar funciones a testear
try:
    from api.helpers import (
        generate_song_id,
        get_audio_file_path,
        create_song_info_from_request
    )
except ImportError:
    print("No se pudieron importar las funciones. Asegúrate de estar en el directorio correcto.")
    sys.exit(1)


def example_basic_usage():
    """Ejemplo básico de uso"""
    print("=" * 60)
    print("Ejemplo 1: Uso básico del generador")
    print("=" * 60)
    
    # Generar tests para generate_song_id
    test_cases, code = generate_tests_for_function(
        generate_song_id,
        num_cases=5
    )
    
    print(f"\nGenerados {len(test_cases)} casos de prueba:")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case.test_name}")
        print(f"   Tipo: {test_case.test_type.value}")
        print(f"   Descripción: {test_case.description}")
    
    print("\n" + "=" * 60)
    print("Código generado:")
    print("=" * 60)
    print(code[:500] + "..." if len(code) > 500 else code)


def example_custom_types():
    """Ejemplo con tipos de tests personalizados"""
    print("\n" + "=" * 60)
    print("Ejemplo 2: Tipos de tests personalizados")
    print("=" * 60)
    
    generator = TestCaseGenerator()
    
    # Solo generar happy path y error handling
    test_cases = generator.generate_test_cases(
        generate_song_id,
        num_cases=5,
        include_types=[TestType.HAPPY_PATH, TestType.ERROR_HANDLING]
    )
    
    print(f"\nGenerados {len(test_cases)} casos de prueba:")
    for test_case in test_cases:
        print(f"  - {test_case.test_name} ({test_case.test_type.value})")


def example_analyze_function():
    """Ejemplo de análisis de función"""
    print("\n" + "=" * 60)
    print("Ejemplo 3: Análisis de función")
    print("=" * 60)
    
    analyzer = FunctionAnalyzer()
    func_info = analyzer.analyze_function(generate_song_id)
    
    print(f"\nInformación de la función '{func_info.name}':")
    print(f"  - Es async: {func_info.is_async}")
    print(f"  - Es generator: {func_info.is_generator}")
    print(f"  - Parámetros: {list(func_info.parameters.keys())}")
    print(f"  - Tipo de retorno: {func_info.return_type}")
    print(f"  - Docstring: {func_info.docstring}")


def example_generate_to_file():
    """Ejemplo de generación a archivo"""
    print("\n" + "=" * 60)
    print("Ejemplo 4: Generar tests a archivo")
    print("=" * 60)
    
    output_file = Path(__file__).parent / "test_generated_example.py"
    
    test_cases, code = generate_tests_for_function(
        generate_song_id,
        num_cases=10,
        output_file=str(output_file)
    )
    
    print(f"\nTests generados en: {output_file}")
    print(f"Total de casos: {len(test_cases)}")


def example_multiple_functions():
    """Ejemplo con múltiples funciones"""
    print("\n" + "=" * 60)
    print("Ejemplo 5: Múltiples funciones")
    print("=" * 60)
    
    functions = [
        ("generate_song_id", generate_song_id),
        # Agregar más funciones aquí
    ]
    
    for name, func in functions:
        print(f"\nGenerando tests para: {name}")
        test_cases, code = generate_tests_for_function(func, num_cases=5)
        print(f"  - {len(test_cases)} casos generados")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generador de Casos de Prueba - Ejemplos")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_custom_types()
        example_analyze_function()
        # example_generate_to_file()  # Descomentar para generar archivo
        example_multiple_functions()
        
        print("\n" + "=" * 60)
        print("Ejemplos completados!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

