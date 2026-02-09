"""
Test Generator - Generación automática de tests para código mejorado
=====================================================================
"""

import logging
from typing import Dict, Any, Optional, List
import ast
import re

logger = logging.getLogger(__name__)


class TestGenerator:
    """
    Genera tests automáticos para código mejorado.
    """
    
    def __init__(self):
        """Inicializar generador de tests"""
        self.supported_frameworks = {
            "python": ["pytest", "unittest"],
            "javascript": ["jest", "mocha"],
            "typescript": ["jest", "mocha"]
        }
    
    def generate_tests(
        self,
        code: str,
        language: str = "python",
        framework: str = "pytest"
    ) -> Dict[str, Any]:
        """
        Genera tests para código dado.
        
        Args:
            code: Código a testear
            language: Lenguaje de programación
            framework: Framework de testing
            
        Returns:
            Tests generados
        """
        try:
            if language == "python":
                return self._generate_python_tests(code, framework)
            elif language in ["javascript", "typescript"]:
                return self._generate_js_tests(code, framework)
            else:
                return {
                    "tests": [],
                    "error": f"Lenguaje no soportado: {language}"
                }
        except Exception as e:
            logger.error(f"Error generando tests: {e}")
            return {
                "tests": [],
                "error": str(e)
            }
    
    def _generate_python_tests(self, code: str, framework: str) -> Dict[str, Any]:
        """Genera tests para código Python"""
        try:
            tree = ast.parse(code)
            
            tests = []
            functions_to_test = []
            
            # Encontrar funciones
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Omitir funciones privadas o especiales
                    if not node.name.startswith("_") or node.name == "__init__":
                        functions_to_test.append({
                            "name": node.name,
                            "args": [arg.arg for arg in node.args.args],
                            "lineno": node.lineno
                        })
            
            # Generar tests para cada función
            for func in functions_to_test:
                if framework == "pytest":
                    test_code = self._generate_pytest_test(func)
                else:
                    test_code = self._generate_unittest_test(func)
                
                tests.append({
                    "function": func["name"],
                    "test_code": test_code,
                    "framework": framework
                })
            
            return {
                "tests": tests,
                "framework": framework,
                "functions_tested": len(tests),
                "total_functions": len(functions_to_test)
            }
            
        except SyntaxError as e:
            logger.warning(f"Error de sintaxis: {e}")
            return {
                "tests": [],
                "error": f"Syntax error: {str(e)}"
            }
    
    def _generate_pytest_test(self, func: Dict[str, Any]) -> str:
        """Genera test usando pytest"""
        func_name = func["name"]
        args = func["args"]
        
        # Remover 'self' si existe
        if args and args[0] == "self":
            args = args[1:]
        
        test_code = f"""
def test_{func_name}():
    \"\"\"Test para función {func_name}\"\"\"
    # TODO: Implementar test real
    # Ejemplo básico:
"""
        
        if args:
            # Generar valores de ejemplo
            example_args = ", ".join([f"{arg}=None" for arg in args])
            test_code += f"    result = {func_name}({example_args})\n"
            test_code += "    # assert result is not None\n"
        else:
            test_code += f"    result = {func_name}()\n"
            test_code += "    # assert result is not None\n"
        
        return test_code.strip()
    
    def _generate_unittest_test(self, func: Dict[str, Any]) -> str:
        """Genera test usando unittest"""
        func_name = func["name"]
        args = func["args"]
        
        if args and args[0] == "self":
            args = args[1:]
        
        test_code = f"""
import unittest

class Test{func_name.capitalize()}(unittest.TestCase):
    \"\"\"Tests para función {func_name}\"\"\"
    
    def test_{func_name}(self):
        \"\"\"Test básico para {func_name}\"\"\"
        # TODO: Implementar test real
"""
        
        if args:
            example_args = ", ".join([f"{arg}=None" for arg in args])
            test_code += f"        result = {func_name}({example_args})\n"
        else:
            test_code += f"        result = {func_name}()\n"
        
        test_code += "        # self.assertIsNotNone(result)\n"
        
        return test_code.strip()
    
    def _generate_js_tests(self, code: str, framework: str) -> Dict[str, Any]:
        """Genera tests para código JavaScript/TypeScript"""
        tests = []
        
        # Encontrar funciones (regex básico)
        function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:\([^)]*\)\s*=>|function))'
        matches = re.finditer(function_pattern, code)
        
        functions = []
        for match in matches:
            func_name = match.group(1) or match.group(2)
            if func_name:
                functions.append(func_name)
        
        # Generar tests
        for func_name in functions:
            if framework == "jest":
                test_code = f"""
describe('{func_name}', () => {{
    test('should work correctly', () => {{
        // TODO: Implementar test real
        // const result = {func_name}();
        // expect(result).toBeDefined();
    }});
}});
"""
            else:  # mocha
                test_code = f"""
describe('{func_name}', function() {{
    it('should work correctly', function() {{
        // TODO: Implementar test real
        // const result = {func_name}();
        // assert(result !== undefined);
    }});
}});
"""
            
            tests.append({
                "function": func_name,
                "test_code": test_code.strip(),
                "framework": framework
            })
        
        return {
            "tests": tests,
            "framework": framework,
            "functions_tested": len(tests),
            "total_functions": len(functions)
        }
    
    def validate_tests(self, code: str, tests: str, language: str) -> Dict[str, Any]:
        """
        Valida que los tests sean correctos.
        
        Args:
            code: Código original
            tests: Código de tests
            language: Lenguaje
            
        Returns:
            Resultado de validación
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        try:
            if language == "python":
                # Intentar parsear tests
                ast.parse(tests)
                validation["valid"] = True
            elif language in ["javascript", "typescript"]:
                # Validación básica
                if "test" not in tests.lower() and "describe" not in tests.lower():
                    validation["warnings"].append("No se encontraron tests en el código")
        except SyntaxError as e:
            validation["valid"] = False
            validation["errors"].append(f"Error de sintaxis: {str(e)}")
        
        return validation




