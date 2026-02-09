"""
Generador de casos de prueba usando IA para Lovable Community

Analiza funciones y genera casos de prueba automáticamente.
"""

import inspect
import ast
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class TestCase:
    """Caso de prueba generado"""
    name: str
    description: str
    test_type: str  # happy_path, edge_case, error_handling, boundary
    code: str
    expected_result: Optional[str] = None


class TestCaseGenerator:
    """Generador de casos de prueba"""
    
    def __init__(self):
        self.test_cases: List[TestCase] = []
    
    def analyze_function(self, func: Callable) -> Dict[str, Any]:
        """Analiza una función y extrae información"""
        sig = inspect.signature(func)
        doc = inspect.getdoc(func) or ""
        
        return {
            "name": func.__name__,
            "parameters": list(sig.parameters.keys()),
            "return_type": sig.return_annotation,
            "docstring": doc,
            "source": inspect.getsource(func)
        }
    
    def extract_validation_rules(self, docstring: str) -> List[str]:
        """Extrae reglas de validación del docstring"""
        rules = []
        lines = docstring.split("\n")
        
        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in ["must", "required", "cannot", "should", "max", "min"]):
                rules.append(line)
        
        return rules
    
    def extract_error_conditions(self, docstring: str) -> List[str]:
        """Extrae condiciones de error del docstring"""
        errors = []
        lines = docstring.split("\n")
        
        for line in lines:
            line = line.strip().lower()
            if any(keyword in line for keyword in ["raises", "error", "exception", "invalid", "not found"]):
                errors.append(line)
        
        return errors
    
    def generate_happy_path_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Genera tests de happy path"""
        tests = []
        func_name = func_info["name"]
        
        # Test básico de happy path
        test = TestCase(
            name=f"test_{func_name}_success",
            description=f"Test exitoso de {func_name}",
            test_type="happy_path",
            code=f"""
    def test_{func_name}_success(self, chat_service, sample_user_id):
        \"\"\"Test exitoso de {func_name}\"\"\"
        # TODO: Implementar con datos válidos
        result = chat_service.{func_name}(...)
        assert result is not None
"""
        )
        tests.append(test)
        
        return tests
    
    def generate_edge_case_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Genera tests de edge cases"""
        tests = []
        func_name = func_info["name"]
        params = func_info["parameters"]
        
        # Test con valores None
        if params:
            test = TestCase(
                name=f"test_{func_name}_none_values",
                description=f"Test de {func_name} con valores None",
                test_type="edge_case",
                code=f"""
    def test_{func_name}_none_values(self, chat_service):
        \"\"\"Test de {func_name} con valores None\"\"\"
        # TODO: Implementar con valores None donde sea apropiado
        with pytest.raises((ValueError, TypeError, InvalidChatError)):
            chat_service.{func_name}(None, ...)
"""
            )
            tests.append(test)
        
        # Test con strings vacíos
        test = TestCase(
            name=f"test_{func_name}_empty_strings",
            description=f"Test de {func_name} con strings vacíos",
            test_type="edge_case",
            code=f"""
    def test_{func_name}_empty_strings(self, chat_service):
        \"\"\"Test de {func_name} con strings vacíos\"\"\"
        # TODO: Implementar con strings vacíos
        with pytest.raises(InvalidChatError):
            chat_service.{func_name}("", ...)
"""
        )
        tests.append(test)
        
        return tests
    
    def generate_error_handling_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Genera tests de error handling"""
        tests = []
        func_name = func_info["name"]
        errors = self.extract_error_conditions(func_info["docstring"])
        
        for error in errors:
            if "not found" in error.lower():
                test = TestCase(
                    name=f"test_{func_name}_not_found",
                    description=f"Test de {func_name} cuando no se encuentra",
                    test_type="error_handling",
                    code=f"""
    def test_{func_name}_not_found(self, chat_service):
        \"\"\"Test de {func_name} cuando no se encuentra\"\"\"
        with pytest.raises(ChatNotFoundError):
            chat_service.{func_name}("nonexistent-id", ...)
"""
                )
                tests.append(test)
        
        return tests
    
    def generate_boundary_tests(self, func_info: Dict[str, Any]) -> List[TestCase]:
        """Genera tests de valores límite"""
        tests = []
        func_name = func_info["name"]
        rules = self.extract_validation_rules(func_info["docstring"])
        
        # Buscar límites en las reglas
        for rule in rules:
            if "max" in rule or "maximum" in rule:
                test = TestCase(
                    name=f"test_{func_name}_max_length",
                    description=f"Test de {func_name} con longitud máxima",
                    test_type="boundary",
                    code=f"""
    def test_{func_name}_max_length(self, chat_service):
        \"\"\"Test de {func_name} con longitud máxima\"\"\"
        # TODO: Implementar con valor en el límite máximo
        max_value = "A" * MAX_LENGTH
        result = chat_service.{func_name}(max_value, ...)
        assert result is not None
"""
                )
                tests.append(test)
        
        return tests
    
    def generate_all_tests(self, func: Callable) -> List[TestCase]:
        """Genera todos los tipos de tests para una función"""
        func_info = self.analyze_function(func)
        all_tests = []
        
        all_tests.extend(self.generate_happy_path_tests(func_info))
        all_tests.extend(self.generate_edge_case_tests(func_info))
        all_tests.extend(self.generate_error_handling_tests(func_info))
        all_tests.extend(self.generate_boundary_tests(func_info))
        
        return all_tests
    
    def generate_test_file(self, func: Callable, output_file: str) -> None:
        """Genera un archivo de tests para una función"""
        tests = self.generate_all_tests(func)
        
        imports = """
import pytest
from services import ChatService
from exceptions import ChatNotFoundError, InvalidChatError
"""
        
        class_name = f"Test{func.__name__.title()}"
        
        content = f'''"""
Tests generados automáticamente para {func.__name__}
"""
{imports}


class {class_name}:
    """Tests para {func.__name__}"""
    
'''
        
        for test in tests:
            content += test.code + "\n\n"
        
        with open(output_file, "w") as f:
            f.write(content)


# Ejemplo de uso
if __name__ == "__main__":
    from services import ChatService
    
    generator = TestCaseGenerator()
    
    # Generar tests para publish_chat
    tests = generator.generate_all_tests(ChatService.publish_chat)
    
    for test in tests:
        print(f"\n{test.name}: {test.description}")
        print(f"Type: {test.test_type}")
        print(f"Code:\n{test.code}")

