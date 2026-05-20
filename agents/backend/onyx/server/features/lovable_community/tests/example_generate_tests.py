"""
Ejemplo de uso del generador de tests
"""

from test_case_generator import TestCaseGenerator
from services import ChatService


def main():
    """Ejemplo de generación de tests"""
    generator = TestCaseGenerator()
    
    # Generar tests para publish_chat
    print("Generando tests para publish_chat...")
    tests = generator.generate_all_tests(ChatService.publish_chat)
    
    print(f"\nGenerados {len(tests)} tests:\n")
    
    for test in tests:
        print(f"✓ {test.name}")
        print(f"  Type: {test.test_type}")
        print(f"  Description: {test.description}")
        print()
    
    # Generar archivo de tests
    print("Generando archivo de tests...")
    generator.generate_test_file(
        ChatService.publish_chat,
        "tests/test_services/test_publish_chat_generated.py"
    )
    print("✓ Archivo generado: tests/test_services/test_publish_chat_generated.py")


if __name__ == "__main__":
    main()

