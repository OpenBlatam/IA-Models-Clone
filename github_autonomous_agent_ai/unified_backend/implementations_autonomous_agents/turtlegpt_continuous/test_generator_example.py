"""
Unit Test Generator - Usage Examples
=====================================

Ejemplos prácticos de uso del generador de tests unitarios.
Demuestra cómo usar el generador para crear tests comprehensivos.
"""

import asyncio
from pathlib import Path
from typing import Dict, Any, List

from .test_generator import (
    UnitTestGenerator,
    TestFramework,
    TestComplexity,
    TestObjective,
    create_test_generator
)


def example_basic_usage():
    """Ejemplo básico de uso del generador."""
    print("=" * 60)
    print("Example 1: Basic Test Generation")
    print("=" * 60)
    
    # Crear generador
    generator = create_test_generator(
        test_framework=TestFramework.PYTEST,
        test_complexity=TestComplexity.COMPREHENSIVE,
        target_coverage=0.80
    )
    
    # Ejemplo de función simple
    def calculate_sum(a: int, b: int) -> int:
        """Calculate sum of two numbers."""
        return a + b
    
    # Analizar función
    analysis = generator.analyze_function(calculate_sum)
    print(f"\nAnalyzed function: {analysis.name}")
    print(f"Parameters: {analysis.parameters}")
    print(f"Return type: {analysis.return_type}")
    print(f"Complexity: {analysis.complexity}")
    
    # Generar casos de prueba
    test_cases = generator.generate_test_cases(
        analysis,
        include_edge_cases=True,
        include_error_cases=True
    )
    
    print(f"\nGenerated {len(test_cases)} test cases:")
    for test_case in test_cases:
        print(f"  - {test_case.name}: {test_case.description}")
        print(f"    Tags: {', '.join(test_case.tags)}")
    
    # Generar suite de tests
    suite = generator.generate_test_suite(
        component_name="calculator",
        functions=[calculate_sum]
    )
    
    # Generar código de tests
    test_code = generator.generate_test_code(suite)
    
    print(f"\nGenerated test code ({len(test_code.split(chr(10)))} lines)")
    print("\n" + "-" * 60)
    print("Sample generated code:")
    print("-" * 60)
    print(test_code[:500] + "...\n")


def example_advanced_usage():
    """Ejemplo avanzado con objetivos personalizados."""
    print("=" * 60)
    print("Example 2: Advanced Usage with Custom Objectives")
    print("=" * 60)
    
    # Crear generador con alta complejidad
    generator = create_test_generator(
        test_framework=TestFramework.PYTEST,
        test_complexity=TestComplexity.EXHAUSTIVE,
        target_coverage=0.95
    )
    
    # Ejemplo de función más compleja
    def process_task(task: Dict[str, Any], priority: int = 5) -> Dict[str, Any]:
        """
        Process a task with validation.
        
        Args:
            task: Task dictionary
            priority: Task priority (1-10)
            
        Returns:
            Processed task dictionary
            
        Raises:
            ValueError: If priority is invalid
        """
        if priority < 1 or priority > 10:
            raise ValueError("Priority must be between 1 and 10")
        
        return {
            "task": task,
            "priority": priority,
            "processed": True,
            "timestamp": "2024-01-01T00:00:00"
        }
    
    # Objetivos personalizados
    custom_objectives = [
        TestObjective(
            description="Verify task processing with valid inputs",
            priority=10,
            category="functionality"
        ),
        TestObjective(
            description="Test priority validation edge cases",
            priority=9,
            category="edge_cases"
        ),
        TestObjective(
            description="Verify error handling for invalid priorities",
            priority=8,
            category="error_handling"
        ),
        TestObjective(
            description="Test with None and empty task dictionaries",
            priority=7,
            category="edge_cases"
        )
    ]
    
    # Generar suite con objetivos personalizados
    suite = generator.generate_test_suite(
        component_name="task_processor",
        functions=[process_task],
        objectives=custom_objectives
    )
    
    print(f"\nGenerated test suite with {len(suite.test_cases)} test cases")
    print(f"Test objectives: {len(suite.objectives)}")
    print(f"Test complexity: {suite.test_complexity.value}")
    
    print("\nTest Objectives:")
    for obj in suite.objectives:
        print(f"  [{obj.priority}] {obj.description} ({obj.category})")
    
    # Generar código
    test_code = generator.generate_test_code(suite)
    
    # Guardar en archivo (opcional)
    output_file = Path("generated_tests_example.py")
    output_file.write_text(test_code, encoding='utf-8')
    print(f"\n✅ Test code saved to: {output_file}")


def example_async_function():
    """Ejemplo con función asíncrona."""
    print("=" * 60)
    print("Example 3: Async Function Testing")
    print("=" * 60)
    
    generator = create_test_generator(
        test_framework=TestFramework.PYTEST,
        test_complexity=TestComplexity.COMPREHENSIVE
    )
    
    async def fetch_data(url: str, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Fetch data from URL asynchronously.
        
        Args:
            url: URL to fetch
            timeout: Request timeout in seconds
            
        Returns:
            Fetched data dictionary
        """
        # Simulación
        await asyncio.sleep(0.1)
        return {"url": url, "data": "mock_data", "timeout": timeout}
    
    # Analizar función async
    analysis = generator.analyze_function(fetch_data)
    print(f"\nAnalyzed async function: {analysis.name}")
    print(f"Parameters: {analysis.parameters}")
    
    # Generar tests
    test_cases = generator.generate_test_cases(analysis)
    print(f"\nGenerated {len(test_cases)} test cases for async function")
    
    # Generar suite
    suite = generator.generate_test_suite(
        component_name="async_data_fetcher",
        functions=[fetch_data]
    )
    
    # Generar código (incluirá pytest-asyncio)
    test_code = generator.generate_test_code(suite)
    
    print("\nGenerated test code includes async support:")
    if "pytest.mark.asyncio" in test_code or "async def" in test_code:
        print("  ✅ Async test support detected")
    else:
        print("  ⚠️  Async support may need manual addition")


def example_multiple_functions():
    """Ejemplo generando tests para múltiples funciones."""
    print("=" * 60)
    print("Example 4: Multiple Functions Test Suite")
    print("=" * 60)
    
    generator = create_test_generator(
        test_framework=TestFramework.PYTEST,
        test_complexity=TestComplexity.COMPREHENSIVE
    )
    
    # Múltiples funciones relacionadas
    def validate_email(email: str) -> bool:
        """Validate email format."""
        return "@" in email and "." in email.split("@")[1]
    
    def format_name(first: str, last: str) -> str:
        """Format full name."""
        return f"{first} {last}".strip()
    
    def calculate_age(birth_year: int) -> int:
        """Calculate age from birth year."""
        return 2024 - birth_year
    
    # Generar suite para todas las funciones
    suite = generator.generate_test_suite(
        component_name="user_utils",
        functions=[validate_email, format_name, calculate_age]
    )
    
    print(f"\nGenerated test suite for {len([validate_email, format_name, calculate_age])} functions")
    print(f"Total test cases: {len(suite.test_cases)}")
    
    # Agrupar por función
    by_function = {}
    for test_case in suite.test_cases:
        func_name = test_case.name.split("_")[1] if "_" in test_case.name else "unknown"
        if func_name not in by_function:
            by_function[func_name] = []
        by_function[func_name].append(test_case)
    
    print("\nTest cases by function:")
    for func_name, cases in by_function.items():
        print(f"  {func_name}: {len(cases)} test cases")
    
    # Generar código
    test_code = generator.generate_test_code(suite)
    print(f"\n✅ Generated {len(test_code.split(chr(10)))} lines of test code")


def example_save_to_file():
    """Ejemplo guardando tests en archivo."""
    print("=" * 60)
    print("Example 5: Save Generated Tests to File")
    print("=" * 60)
    
    generator = create_test_generator(
        test_framework=TestFramework.PYTEST,
        test_complexity=TestComplexity.COMPREHENSIVE
    )
    
    def example_function(x: int, y: int) -> int:
        """Example function for testing."""
        return x * y
    
    # Generar suite
    suite = generator.generate_test_suite(
        component_name="example_module",
        functions=[example_function]
    )
    
    # Generar código
    test_code = generator.generate_test_code(suite)
    
    # Guardar en archivo
    test_file = Path("tests") / "test_example_module.py"
    test_file.parent.mkdir(exist_ok=True)
    test_file.write_text(test_code, encoding='utf-8')
    
    print(f"✅ Test file created: {test_file}")
    print(f"   Lines: {len(test_code.split(chr(10)))}")
    print(f"   Test cases: {len(suite.test_cases)}")
    print(f"\nTo run tests:")
    print(f"   pytest {test_file}")


def main():
    """Ejecutar todos los ejemplos."""
    print("\n" + "=" * 60)
    print("Unit Test Generator - Comprehensive Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_usage()
        print("\n")
        
        example_advanced_usage()
        print("\n")
        
        example_async_function()
        print("\n")
        
        example_multiple_functions()
        print("\n")
        
        example_save_to_file()
        print("\n")
        
        print("=" * 60)
        print("✅ All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
