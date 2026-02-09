"""
Generador de casos de prueba basado en análisis de funciones

Este módulo analiza funciones, sus firmas y docstrings para generar
casos de prueba únicos, diversos e intuitivos.
"""

import ast
import inspect
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class TestType(Enum):
    """Tipos de casos de prueba"""
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    BOUNDARY = "boundary"
    NULL_EMPTY = "null_empty"
    TYPE_VALIDATION = "type_validation"
    INTEGRATION = "integration"


@dataclass
class FunctionInfo:
    """Información extraída de una función"""
    name: str
    signature: inspect.Signature
    docstring: Optional[str]
    parameters: Dict[str, inspect.Parameter]
    return_type: Optional[Any]
    is_async: bool
    is_generator: bool
    source_code: Optional[str] = None


@dataclass
class TestCase:
    """Caso de prueba generado"""
    test_name: str
    test_type: TestType
    description: str
    parameters: Dict[str, Any]
    expected_result: Optional[Any] = None
    expected_exception: Optional[type] = None
    assertions: List[str] = None
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None


class FunctionAnalyzer:
    """Analiza funciones para extraer información útil"""
    
    def __init__(self):
        self.validators = {
            'min_length': self._extract_min_length,
            'max_length': self._extract_max_length,
            'ge': self._extract_min_value,
            'le': self._extract_max_value,
            'gt': self._extract_gt_value,
            'lt': self._extract_lt_value,
        }
    
    def analyze_function(self, func: Any) -> FunctionInfo:
        """Analiza una función y extrae información"""
        try:
            sig = inspect.signature(func)
            source = inspect.getsource(func)
            tree = ast.parse(source)
            
            return FunctionInfo(
                name=func.__name__,
                signature=sig,
                docstring=inspect.getdoc(func),
                parameters=dict(sig.parameters),
                return_type=sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None,
                is_async=inspect.iscoroutinefunction(func),
                is_generator=inspect.isgeneratorfunction(func),
                source_code=source
            )
        except Exception as e:
            raise ValueError(f"Error analyzing function {func.__name__}: {e}")
    
    def extract_parameter_constraints(self, param: inspect.Parameter) -> Dict[str, Any]:
        """Extrae restricciones de un parámetro desde el docstring o anotaciones"""
        constraints = {}
        
        # Buscar en docstring
        # Buscar en anotaciones de tipo
        if param.annotation != inspect.Parameter.empty:
            annotation_str = str(param.annotation)
            # Buscar validadores comunes
            for validator, extractor in self.validators.items():
                match = re.search(rf'{validator}\s*=\s*(\d+\.?\d*)', annotation_str)
                if match:
                    constraints[validator] = extractor(match.group(1))
        
        return constraints
    
    def _extract_min_length(self, value: str) -> int:
        return int(float(value))
    
    def _extract_max_length(self, value: str) -> int:
        return int(float(value))
    
    def _extract_min_value(self, value: str) -> float:
        return float(value)
    
    def _extract_max_value(self, value: str) -> float:
        return float(value)
    
    def _extract_gt_value(self, value: str) -> float:
        return float(value)
    
    def _extract_lt_value(self, value: str) -> float:
        return float(value)
    
    def extract_test_scenarios_from_docstring(self, docstring: Optional[str]) -> List[str]:
        """Extrae escenarios de prueba mencionados en el docstring"""
        if not docstring:
            return []
        
        scenarios = []
        # Buscar ejemplos, casos de uso, etc.
        example_pattern = r'(?:example|ejemplo|case|caso|scenario|escenario)[\s:]+(.+?)(?:\n|$)'
        matches = re.findall(example_pattern, docstring, re.IGNORECASE)
        scenarios.extend(matches)
        
        return scenarios
    
    def extract_validation_rules(self, docstring: Optional[str]) -> Dict[str, Any]:
        """Extrae reglas de validación del docstring"""
        if not docstring:
            return {}
        
        rules = {}
        
        # Buscar menciones de validación
        validation_patterns = {
            'required': r'(?:required|requerido|obligatorio)',
            'optional': r'(?:optional|opcional)',
            'min_length': r'min(?:imum)?\s*(?:length|longitud)?\s*[:=]\s*(\d+)',
            'max_length': r'max(?:imum)?\s*(?:length|longitud)?\s*[:=]\s*(\d+)',
            'min_value': r'min(?:imum)?\s*(?:value|valor)?\s*[:=]\s*(\d+)',
            'max_value': r'max(?:imum)?\s*(?:value|valor)?\s*[:=]\s*(\d+)',
        }
        
        for rule_name, pattern in validation_patterns.items():
            matches = re.findall(pattern, docstring, re.IGNORECASE)
            if matches:
                rules[rule_name] = matches
        
        return rules
    
    def extract_error_conditions(self, docstring: Optional[str]) -> List[str]:
        """Extrae condiciones de error mencionadas en el docstring"""
        if not docstring:
            return []
        
        error_patterns = [
            r'(?:raises?|lanza?|throws?)\s+(\w+)',
            r'(?:error|exception|excepción)\s+(?:when|si|cuando)\s+(.+?)(?:\.|$)',
            r'(?:fails?|falla?)\s+(?:when|si|cuando)\s+(.+?)(?:\.|$)',
        ]
        
        errors = []
        for pattern in error_patterns:
            matches = re.findall(pattern, docstring, re.IGNORECASE)
            errors.extend(matches)
        
        return errors


class TestCaseGenerator:
    """Genera casos de prueba únicos y diversos"""
    
    def __init__(self):
        self.analyzer = FunctionAnalyzer()
        self.test_counter = 0
    
    def generate_test_cases(
        self,
        func: Any,
        num_cases: int = 10,
        include_types: Optional[List[TestType]] = None
    ) -> List[TestCase]:
        """
        Genera casos de prueba para una función
        
        Args:
            func: Función a testear
            num_cases: Número aproximado de casos a generar
            include_types: Tipos de tests a incluir (None = todos)
        """
        func_info = self.analyzer.analyze_function(func)
        test_cases = []
        
        if include_types is None:
            include_types = list(TestType)
        
        # Generar casos happy path
        if TestType.HAPPY_PATH in include_types:
            test_cases.extend(self._generate_happy_path_cases(func_info))
        
        # Generar casos edge case
        if TestType.EDGE_CASE in include_types:
            test_cases.extend(self._generate_edge_cases(func_info))
        
        # Generar casos de error handling
        if TestType.ERROR_HANDLING in include_types:
            test_cases.extend(self._generate_error_cases(func_info))
        
        # Generar casos boundary
        if TestType.BOUNDARY in include_types:
            test_cases.extend(self._generate_boundary_cases(func_info))
        
        # Generar casos null/empty
        if TestType.NULL_EMPTY in include_types:
            test_cases.extend(self._generate_null_empty_cases(func_info))
        
        # Generar casos type validation
        if TestType.TYPE_VALIDATION in include_types:
            test_cases.extend(self._generate_type_validation_cases(func_info))
        
        # Limitar número de casos
        return test_cases[:num_cases]
    
    def _generate_happy_path_cases(self, func_info: FunctionInfo) -> List[TestCase]:
        """Genera casos de prueba para el camino feliz"""
        cases = []
        
        # Caso básico con valores normales
        params = self._generate_normal_parameters(func_info)
        cases.append(TestCase(
            test_name=f"test_{func_info.name}_happy_path_basic",
            test_type=TestType.HAPPY_PATH,
            description=f"Test básico de {func_info.name} con parámetros válidos",
            parameters=params,
            assertions=self._generate_basic_assertions(func_info)
        ))
        
        # Caso con todos los parámetros opcionales
        if any(p.default != inspect.Parameter.empty for p in func_info.parameters.values()):
            params_optional = self._generate_optional_parameters(func_info)
            cases.append(TestCase(
                test_name=f"test_{func_info.name}_happy_path_optional_params",
                test_type=TestType.HAPPY_PATH,
                description=f"Test con parámetros opcionales",
                parameters=params_optional,
                assertions=self._generate_basic_assertions(func_info)
            ))
        
        return cases
    
    def _generate_edge_cases(self, func_info: FunctionInfo) -> List[TestCase]:
        """Genera casos de prueba edge case"""
        cases = []
        
        # Valores extremos pero válidos
        params_extreme = self._generate_extreme_parameters(func_info)
        if params_extreme:
            cases.append(TestCase(
                test_name=f"test_{func_info.name}_edge_case_extreme_values",
                test_type=TestType.EDGE_CASE,
                description=f"Test con valores extremos pero válidos",
                parameters=params_extreme,
                assertions=self._generate_basic_assertions(func_info)
            ))
        
        return cases
    
    def _generate_error_cases(self, func_info: FunctionInfo) -> List[TestCase]:
        """Genera casos de prueba para manejo de errores"""
        cases = []
        
        # Parámetros inválidos
        for param_name, param in func_info.parameters.items():
            if param_name == 'self':
                continue
            
            # Tipo incorrecto
            cases.append(TestCase(
                test_name=f"test_{func_info.name}_error_invalid_type_{param_name}",
                test_type=TestType.ERROR_HANDLING,
                description=f"Test con tipo inválido para {param_name}",
                parameters={param_name: "invalid_type"},
                expected_exception=TypeError,
                assertions=[f"Should raise TypeError for invalid {param_name}"]
            ))
        
        return cases
    
    def _generate_boundary_cases(self, func_info: FunctionInfo) -> List[TestCase]:
        """Genera casos de prueba para valores límite"""
        cases = []
        
        for param_name, param in func_info.parameters.items():
            if param_name == 'self':
                continue
            
            constraints = self.analyzer.extract_parameter_constraints(param)
            
            # Valor mínimo
            if 'min_length' in constraints or 'ge' in constraints:
                min_val = constraints.get('min_length') or constraints.get('ge', 0)
                cases.append(TestCase(
                    test_name=f"test_{func_info.name}_boundary_min_{param_name}",
                    test_type=TestType.BOUNDARY,
                    description=f"Test con valor mínimo para {param_name}",
                    parameters={param_name: min_val},
                    assertions=[f"Should handle minimum value for {param_name}"]
                ))
            
            # Valor máximo
            if 'max_length' in constraints or 'le' in constraints:
                max_val = constraints.get('max_length') or constraints.get('le', 100)
                cases.append(TestCase(
                    test_name=f"test_{func_info.name}_boundary_max_{param_name}",
                    test_type=TestType.BOUNDARY,
                    description=f"Test con valor máximo para {param_name}",
                    parameters={param_name: max_val},
                    assertions=[f"Should handle maximum value for {param_name}"]
                ))
        
        return cases
    
    def _generate_null_empty_cases(self, func_info: FunctionInfo) -> List[TestCase]:
        """Genera casos de prueba para valores None/vacíos"""
        cases = []
        
        for param_name, param in func_info.parameters.items():
            if param_name == 'self' or param.default is not None:
                continue
            
            # None
            if param.annotation == Optional[str] or 'Optional' in str(param.annotation):
                cases.append(TestCase(
                    test_name=f"test_{func_info.name}_null_{param_name}",
                    test_type=TestType.NULL_EMPTY,
                    description=f"Test con None para {param_name}",
                    parameters={param_name: None},
                    assertions=[f"Should handle None for {param_name}"]
                ))
            
            # String vacío
            if param.annotation == str or 'str' in str(param.annotation):
                cases.append(TestCase(
                    test_name=f"test_{func_info.name}_empty_{param_name}",
                    test_type=TestType.NULL_EMPTY,
                    description=f"Test con string vacío para {param_name}",
                    parameters={param_name: ""},
                    expected_exception=ValueError,
                    assertions=[f"Should raise ValueError for empty {param_name}"]
                ))
        
        return cases
    
    def _generate_type_validation_cases(self, func_info: FunctionInfo) -> List[TestCase]:
        """Genera casos de prueba para validación de tipos"""
        cases = []
        
        for param_name, param in func_info.parameters.items():
            if param_name == 'self':
                continue
            
            # Tipo incorrecto
            wrong_type = self._get_wrong_type(param.annotation)
            if wrong_type:
                cases.append(TestCase(
                    test_name=f"test_{func_info.name}_type_validation_{param_name}",
                    test_type=TestType.TYPE_VALIDATION,
                    description=f"Test de validación de tipo para {param_name}",
                    parameters={param_name: wrong_type},
                    expected_exception=TypeError,
                    assertions=[f"Should raise TypeError for wrong type in {param_name}"]
                ))
        
        return cases
    
    def _generate_normal_parameters(self, func_info: FunctionInfo) -> Dict[str, Any]:
        """Genera parámetros normales para una función"""
        params = {}
        
        for param_name, param in func_info.parameters.items():
            if param_name == 'self':
                continue
            
            if param.default != inspect.Parameter.empty:
                params[param_name] = param.default
            else:
                params[param_name] = self._generate_default_value(param.annotation)
        
        return params
    
    def _generate_optional_parameters(self, func_info: FunctionInfo) -> Dict[str, Any]:
        """Genera parámetros solo con los requeridos"""
        params = {}
        
        for param_name, param in func_info.parameters.items():
            if param_name == 'self':
                continue
            
            if param.default == inspect.Parameter.empty:
                params[param_name] = self._generate_default_value(param.annotation)
        
        return params
    
    def _generate_extreme_parameters(self, func_info: FunctionInfo) -> Dict[str, Any]:
        """Genera parámetros con valores extremos"""
        params = {}
        
        for param_name, param in func_info.parameters.items():
            if param_name == 'self':
                continue
            
            constraints = self.analyzer.extract_parameter_constraints(param)
            
            if 'max_length' in constraints:
                params[param_name] = 'x' * constraints['max_length']
            elif 'le' in constraints:
                params[param_name] = constraints['le']
            else:
                params[param_name] = self._generate_default_value(param.annotation)
        
        return params
    
    def _generate_default_value(self, annotation: Any) -> Any:
        """Genera un valor por defecto basado en la anotación de tipo"""
        annotation_str = str(annotation)
        
        if 'str' in annotation_str:
            return "test_string"
        elif 'int' in annotation_str:
            return 42
        elif 'float' in annotation_str:
            return 3.14
        elif 'bool' in annotation_str:
            return True
        elif 'list' in annotation_str or 'List' in annotation_str:
            return []
        elif 'dict' in annotation_str or 'Dict' in annotation_str:
            return {}
        elif 'Optional' in annotation_str:
            return None
        else:
            return None
    
    def _get_wrong_type(self, annotation: Any) -> Any:
        """Obtiene un tipo incorrecto para testing"""
        annotation_str = str(annotation)
        
        if 'str' in annotation_str:
            return 123
        elif 'int' in annotation_str:
            return "not_an_int"
        elif 'float' in annotation_str:
            return "not_a_float"
        elif 'bool' in annotation_str:
            return "not_a_bool"
        elif 'list' in annotation_str or 'List' in annotation_str:
            return "not_a_list"
        elif 'dict' in annotation_str or 'Dict' in annotation_str:
            return "not_a_dict"
        else:
            return None
    
    def _generate_basic_assertions(self, func_info: FunctionInfo) -> List[str]:
        """Genera aserciones básicas para una función"""
        assertions = []
        
        if func_info.return_type:
            return_type_str = str(func_info.return_type)
            if 'None' not in return_type_str:
                assertions.append("assert result is not None")
            
            # Aserciones específicas por tipo
            if 'str' in return_type_str:
                assertions.append("assert isinstance(result, str)")
                assertions.append("assert len(result) > 0")
            elif 'int' in return_type_str:
                assertions.append("assert isinstance(result, int)")
            elif 'float' in return_type_str:
                assertions.append("assert isinstance(result, float)")
            elif 'bool' in return_type_str:
                assertions.append("assert isinstance(result, bool)")
            elif 'list' in return_type_str or 'List' in return_type_str:
                assertions.append("assert isinstance(result, list)")
            elif 'dict' in return_type_str or 'Dict' in return_type_str:
                assertions.append("assert isinstance(result, dict)")
        
        if func_info.is_async:
            assertions.append("# Async function - result is coroutine")
        
        return assertions
    
    def _generate_integration_cases(self, func_info: FunctionInfo) -> List[TestCase]:
        """Genera casos de integración"""
        cases = []
        
        # Caso de integración básico
        if len(func_info.parameters) > 1:
            cases.append(TestCase(
                test_name=f"test_{func_info.name}_integration_full_flow",
                test_type=TestType.INTEGRATION,
                description=f"Test de integración completo para {func_info.name}",
                parameters=self._generate_normal_parameters(func_info),
                assertions=self._generate_basic_assertions(func_info)
            ))
        
        return cases


class TestCodeGenerator:
    """Genera código de test a partir de casos de prueba"""
    
    def generate_test_code(
        self,
        func_info: FunctionInfo,
        test_cases: List[TestCase],
        module_name: str = "test_module"
    ) -> str:
        """Genera código Python de tests"""
        
        code_lines = [
            '"""',
            f'Tests generados automáticamente para {func_info.name}',
            '"""',
            '',
            'import pytest',
            'from unittest.mock import Mock, AsyncMock, patch',
            '',
            f'from {module_name} import {func_info.name.split(".")[-1]}',
            '',
        ]
        
        # Agregar cada caso de prueba
        for test_case in test_cases:
            code_lines.extend(self._generate_single_test(func_info, test_case))
            code_lines.append('')
        
        return '\n'.join(code_lines)
    
    def _generate_single_test(
        self,
        func_info: FunctionInfo,
        test_case: TestCase
    ) -> List[str]:
        """Genera código para un solo test"""
        lines = [
            f'@pytest.mark.{test_case.test_type.value}',
            f'def {test_case.test_name}():',
            f'    """{test_case.description}"""',
        ]
        
        # Setup code
        if test_case.setup_code:
            lines.append(f'    {test_case.setup_code}')
        
        # Parámetros
        params_str = ', '.join(f'{k}={repr(v)}' for k, v in test_case.parameters.items())
        
        # Async o sync
        if func_info.is_async:
            lines.append(f'    async def run_test():')
            lines.append(f'        result = await {func_info.name}({params_str})')
        else:
            lines.append(f'    result = {func_info.name}({params_str})')
        
        # Assertions o exception
        if test_case.expected_exception:
            if func_info.is_async:
                lines.append(f'    with pytest.raises({test_case.expected_exception.__name__}):')
                lines.append(f'        await run_test()')
            else:
                lines.append(f'    with pytest.raises({test_case.expected_exception.__name__}):')
                lines.append(f'        result')
        else:
            if func_info.is_async:
                lines.append(f'    result = await run_test()')
            
            if test_case.assertions:
                for assertion in test_case.assertions:
                    lines.append(f'    {assertion}')
            else:
                lines.append('    assert result is not None')
        
        return lines


# Función de conveniencia para uso directo
def generate_tests_for_function(
    func: Any,
    num_cases: int = 10,
    output_file: Optional[str] = None
) -> Tuple[List[TestCase], str]:
    """
    Genera casos de prueba y código para una función
    
    Returns:
        Tuple de (lista de casos de prueba, código generado)
    """
    generator = TestCaseGenerator()
    code_generator = TestCodeGenerator()
    
    test_cases = generator.generate_test_cases(func, num_cases)
    func_info = generator.analyzer.analyze_function(func)
    
    # Determinar module name
    module_name = func.__module__ if hasattr(func, '__module__') else "test_module"
    
    code = code_generator.generate_test_code(func_info, test_cases, module_name)
    
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(code)
    
    return test_cases, code

