"""
Unit Test Generator Module
==========================

Generador comprehensivo de tests unitarios para funciones y componentes.
Proporciona una interfaz estructurada para generar tests basados en análisis de código.
"""

import ast
import inspect
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TestFramework(Enum):
    """Framework de testing a usar."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    NOSE2 = "nose2"


class TestComplexity(Enum):
    """Nivel de complejidad de los tests."""
    BASIC = "basic"  # Tests básicos de funcionalidad
    COMPREHENSIVE = "comprehensive"  # Tests completos con edge cases
    EXHAUSTIVE = "exhaustive"  # Tests exhaustivos con todos los casos posibles


@dataclass
class TestObjective:
    """Objetivo de un test."""
    description: str
    priority: int = 5  # 1-10, mayor = más importante
    category: str = "functionality"  # functionality, edge_cases, error_handling, performance


@dataclass
class TestCase:
    """Caso de prueba individual."""
    name: str
    description: str
    test_input: Dict[str, Any]
    expected_output: Any
    expected_exception: Optional[type] = None
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class FunctionAnalysis:
    """Análisis de una función."""
    name: str
    signature: str
    parameters: List[str]
    return_type: Optional[str]
    docstring: Optional[str]
    complexity: int
    dependencies: List[str]
    raises: List[str]
    line_count: int


@dataclass
class TestSuite:
    """Suite de tests para un componente."""
    component_name: str
    test_framework: TestFramework
    test_complexity: TestComplexity
    objectives: List[TestObjective]
    test_cases: List[TestCase]
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None
    fixtures: List[str] = field(default_factory=list)
    mocks: List[str] = field(default_factory=list)


class UnitTestGenerator:
    """
    Generador comprehensivo de tests unitarios.
    
    Proporciona funcionalidades para:
    - Análisis de código fuente
    - Generación de casos de prueba
    - Creación de suites de tests
    - Generación de código de tests
    """
    
    def __init__(
        self,
        test_framework: TestFramework = TestFramework.PYTEST,
        test_complexity: TestComplexity = TestComplexity.COMPREHENSIVE,
        target_coverage: float = 0.80
    ):
        """
        Inicializar generador de tests.
        
        Args:
            test_framework: Framework de testing a usar
            test_complexity: Nivel de complejidad de los tests
            target_coverage: Cobertura objetivo (0.0-1.0)
        """
        self.test_framework = test_framework
        self.test_complexity = test_complexity
        self.target_coverage = target_coverage
        self._analyzed_functions: Dict[str, FunctionAnalysis] = {}
    
    def analyze_function(self, func: callable) -> FunctionAnalysis:
        """
        Analizar una función para generar tests.
        
        Args:
            func: Función a analizar
            
        Returns:
            Análisis de la función
        """
        sig = inspect.signature(func)
        source = inspect.getsource(func)
        
        # Parsear AST
        tree = ast.parse(source)
        func_node = tree.body[0] if tree.body else None
        
        # Extraer información
        parameters = list(sig.parameters.keys())
        return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else None
        
        # Analizar complejidad (simplificado: contar ramificaciones)
        complexity = self._calculate_complexity(func_node) if func_node else 1
        
        # Extraer dependencias
        dependencies = self._extract_dependencies(func_node) if func_node else []
        
        # Extraer excepciones que se lanzan
        raises = self._extract_raises(func_node) if func_node else []
        
        # Contar líneas
        line_count = len(source.split('\n'))
        
        analysis = FunctionAnalysis(
            name=func.__name__,
            signature=str(sig),
            parameters=parameters,
            return_type=str(return_type) if return_type else None,
            docstring=inspect.getdoc(func),
            complexity=complexity,
            dependencies=dependencies,
            raises=raises,
            line_count=line_count
        )
        
        self._analyzed_functions[func.__name__] = analysis
        logger.debug(f"Analyzed function: {func.__name__}")
        
        return analysis
    
    def generate_test_cases(
        self,
        func_analysis: FunctionAnalysis,
        include_edge_cases: bool = True,
        include_error_cases: bool = True
    ) -> List[TestCase]:
        """
        Generar casos de prueba para una función.
        
        Args:
            func_analysis: Análisis de la función
            include_edge_cases: Incluir casos límite
            include_error_cases: Incluir casos de error
            
        Returns:
            Lista de casos de prueba
        """
        test_cases = []
        
        # Casos básicos de funcionalidad
        test_cases.extend(self._generate_basic_cases(func_analysis))
        
        if include_edge_cases:
            test_cases.extend(self._generate_edge_cases(func_analysis))
        
        if include_error_cases and func_analysis.raises:
            test_cases.extend(self._generate_error_cases(func_analysis))
        
        logger.debug(f"Generated {len(test_cases)} test cases for {func_analysis.name}")
        
        return test_cases
    
    def generate_test_suite(
        self,
        component_name: str,
        functions: List[callable],
        objectives: Optional[List[TestObjective]] = None
    ) -> TestSuite:
        """
        Generar suite completa de tests para un componente.
        
        Args:
            component_name: Nombre del componente
            functions: Lista de funciones a testear
            objectives: Objetivos específicos de testing
            
        Returns:
            Suite de tests generada
        """
        if objectives is None:
            objectives = self._generate_default_objectives(component_name)
        
        all_test_cases = []
        
        # Analizar y generar tests para cada función
        for func in functions:
            analysis = self.analyze_function(func)
            test_cases = self.generate_test_cases(analysis)
            all_test_cases.extend(test_cases)
        
        suite = TestSuite(
            component_name=component_name,
            test_framework=self.test_framework,
            test_complexity=self.test_complexity,
            objectives=objectives,
            test_cases=all_test_cases,
            fixtures=self._generate_fixtures(functions),
            mocks=self._generate_mocks(functions)
        )
        
        logger.info(f"Generated test suite for {component_name} with {len(all_test_cases)} test cases")
        
        return suite
    
    def generate_test_code(self, test_suite: TestSuite) -> str:
        """
        Generar código de tests a partir de una suite.
        
        Args:
            test_suite: Suite de tests
            
        Returns:
            Código Python de tests
        """
        if self.test_framework == TestFramework.PYTEST:
            return self._generate_pytest_code(test_suite)
        elif self.test_framework == TestFramework.UNITTEST:
            return self._generate_unittest_code(test_suite)
        else:
            raise ValueError(f"Unsupported test framework: {self.test_framework}")
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calcular complejidad ciclomática (simplificada)."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_dependencies(self, node: ast.AST) -> List[str]:
        """Extraer dependencias de la función."""
        dependencies = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name):
                        dependencies.add(child.func.value.id)
        
        return sorted(list(dependencies))
    
    def _extract_raises(self, node: ast.AST) -> List[str]:
        """Extraer tipos de excepciones que se lanzan."""
        raises = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if child.exc:
                    if isinstance(child.exc, ast.Call):
                        if isinstance(child.exc.func, ast.Name):
                            raises.add(child.exc.func.id)
        
        return sorted(list(raises))
    
    def _generate_basic_cases(self, analysis: FunctionAnalysis) -> List[TestCase]:
        """Generar casos básicos de funcionalidad."""
        cases = []
        
        # Caso básico con valores por defecto
        basic_input = {}
        for param in analysis.parameters:
            if param != 'self':
                basic_input[param] = self._get_default_test_value(param)
        
        cases.append(TestCase(
            name=f"test_{analysis.name}_basic",
            description=f"Basic functionality test for {analysis.name}",
            test_input=basic_input,
            expected_output=None,  # Se determinará dinámicamente
            tags=["basic", "functionality"]
        ))
        
        return cases
    
    def _generate_edge_cases(self, analysis: FunctionAnalysis) -> List[TestCase]:
        """Generar casos límite."""
        cases = []
        
        # Casos con None
        if self.test_complexity in [TestComplexity.COMPREHENSIVE, TestComplexity.EXHAUSTIVE]:
            edge_input = {}
            for param in analysis.parameters:
                if param != 'self':
                    edge_input[param] = None
            
            cases.append(TestCase(
                name=f"test_{analysis.name}_none_values",
                description=f"Edge case: None values for {analysis.name}",
                test_input=edge_input,
                expected_output=None,
                tags=["edge_case", "none_values"]
            ))
        
        # Casos con valores vacíos
        if self.test_complexity == TestComplexity.EXHAUSTIVE:
            empty_input = {}
            for param in analysis.parameters:
                if param != 'self':
                    empty_input[param] = self._get_empty_value(param)
            
            cases.append(TestCase(
                name=f"test_{analysis.name}_empty_values",
                description=f"Edge case: Empty values for {analysis.name}",
                test_input=empty_input,
                expected_output=None,
                tags=["edge_case", "empty_values"]
            ))
        
        return cases
    
    def _generate_error_cases(self, analysis: FunctionAnalysis) -> List[TestCase]:
        """Generar casos de error."""
        cases = []
        
        for exception_type in analysis.raises:
            error_input = {}
            for param in analysis.parameters:
                if param != 'self':
                    error_input[param] = self._get_error_trigger_value(param)
            
            cases.append(TestCase(
                name=f"test_{analysis.name}_raises_{exception_type.lower()}",
                description=f"Error case: {analysis.name} should raise {exception_type}",
                test_input=error_input,
                expected_output=None,
                expected_exception=exception_type,
                tags=["error", "exception", exception_type.lower()]
            ))
        
        return cases
    
    def _generate_default_objectives(self, component_name: str) -> List[TestObjective]:
        """Generar objetivos por defecto."""
        return [
            TestObjective(
                description=f"Verify basic functionality of {component_name}",
                priority=10,
                category="functionality"
            ),
            TestObjective(
                description=f"Test error handling in {component_name}",
                priority=8,
                category="error_handling"
            ),
            TestObjective(
                description=f"Test edge cases in {component_name}",
                priority=7,
                category="edge_cases"
            ),
            TestObjective(
                description=f"Ensure {component_name} meets performance requirements",
                priority=5,
                category="performance"
            )
        ]
    
    def _generate_fixtures(self, functions: List[callable]) -> List[str]:
        """Generar fixtures necesarias."""
        fixtures = []
        
        # Fixture común para el componente
        fixtures.append("@pytest.fixture")
        fixtures.append("def component():")
        fixtures.append("    # Setup component")
        fixtures.append("    yield component_instance")
        fixtures.append("    # Teardown")
        
        return fixtures
    
    def _generate_mocks(self, functions: List[callable]) -> List[str]:
        """Generar mocks necesarios."""
        mocks = []
        
        # Mock común para dependencias externas
        mocks.append("from unittest.mock import Mock, patch")
        
        return mocks
    
    def _get_default_test_value(self, param_name: str) -> Any:
        """Obtener valor por defecto para un parámetro."""
        # Valores por defecto basados en el nombre del parámetro
        defaults = {
            "name": "test_name",
            "description": "test description",
            "priority": 5,
            "interval": 1.0,
            "enabled": True,
            "config": {},
            "metadata": {},
            "task_id": "test_task_id",
            "count": 5,
            "limit": 10,
            "timeout": 30.0
        }
        
        return defaults.get(param_name, "test_value")
    
    def _get_empty_value(self, param_name: str) -> Any:
        """Obtener valor vacío para un parámetro."""
        empty_values = {
            "name": "",
            "description": "",
            "config": {},
            "metadata": {},
            "tasks": [],
            "items": []
        }
        
        return empty_values.get(param_name, None)
    
    def _get_error_trigger_value(self, param_name: str) -> Any:
        """Obtener valor que puede disparar un error."""
        # Valores que típicamente causan errores
        error_values = {
            "name": None,
            "description": None,
            "priority": -1,
            "interval": -1.0,
            "timeout": -1.0
        }
        
        return error_values.get(param_name, "invalid_value")
    
    def _generate_pytest_code(self, test_suite: TestSuite) -> str:
        """Generar código de tests usando pytest con AAA pattern."""
        lines = [
            '"""',
            f'Unit tests for {test_suite.component_name}',
            f'Generated by UnitTestGenerator',
            '',
            'Test Objectives:',
        ]
        
        # Agregar objetivos
        for obj in test_suite.objectives:
            lines.append(f'  - {obj.description} (Priority: {obj.priority}, Category: {obj.category})')
        
        lines.extend([
            '',
            'Following AAA Pattern (Arrange, Act, Assert)',
            '"""',
            '',
            'import pytest',
            'from unittest.mock import Mock, patch, MagicMock',
            'import asyncio',
            '',
            f'# Test suite for {test_suite.component_name}',
            ''
        ])
        
        # Agregar fixtures
        if test_suite.fixtures:
            lines.extend(['# Fixtures', ''])
            lines.extend(test_suite.fixtures)
            lines.append('')
        
        # Agregar clase de tests
        class_name = f'Test{test_suite.component_name.replace("_", "").title()}'
        lines.append(f'class {class_name}:')
        lines.append(f'    """Tests for {test_suite.component_name}."""')
        lines.append('')
        
        # Agregar cada caso de prueba con AAA pattern
        for test_case in test_suite.test_cases:
            lines.append(f'    def {test_case.name}(self):')
            lines.append(f'        """{test_case.description}"""')
            lines.append('        # Arrange - Setup test data and mocks')
            
            if test_case.setup_code:
                lines.append(f'        {test_case.setup_code}')
            else:
                # Generar arrange section
                arrange_lines = []
                for key, value in test_case.test_input.items():
                    if isinstance(value, str):
                        arrange_lines.append(f'        {key} = "{value}"')
                    elif isinstance(value, (int, float, bool)):
                        arrange_lines.append(f'        {key} = {value}')
                    elif value is None:
                        arrange_lines.append(f'        {key} = None')
                    else:
                        arrange_lines.append(f'        {key} = {repr(value)}')
                
                if arrange_lines:
                    lines.extend(arrange_lines)
                else:
                    lines.append('        # No setup required')
            
            lines.append('')
            lines.append('        # Act - Execute function under test')
            
            # Generar código de test
            if test_case.expected_exception:
                lines.append(f'        # Assert - Verify exception is raised')
                lines.append(f'        with pytest.raises({test_case.expected_exception}):')
                lines.append(f'            # TODO: Call function that should raise {test_case.expected_exception}')
                lines.append(f'            pass')
            else:
                lines.append('        # TODO: Call function under test')
                lines.append('        result = None  # Replace with actual function call')
                lines.append('')
                lines.append('        # Assert - Verify results')
                if test_case.expected_output is not None:
                    lines.append(f'        assert result == {repr(test_case.expected_output)}')
                else:
                    lines.append('        assert result is not None')
                    lines.append('        # TODO: Add specific assertions based on expected behavior')
            
            if test_case.teardown_code:
                lines.append('')
                lines.append('        # Teardown')
                lines.append(f'        {test_case.teardown_code}')
            
            lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_unittest_code(self, test_suite: TestSuite) -> str:
        """Generar código de tests usando unittest."""
        lines = [
            '"""',
            f'Unit tests for {test_suite.component_name}',
            f'Generated by UnitTestGenerator',
            '"""',
            '',
            'import unittest',
            'from unittest.mock import Mock, patch, MagicMock',
            'import asyncio',
            '',
            f'# Test suite for {test_suite.component_name}',
            ''
        ]
        
        # Agregar objetivos como comentarios
        lines.append('# Test Objectives:')
        for obj in test_suite.objectives:
            lines.append(f'# - {obj.description} (Priority: {obj.priority}, Category: {obj.category})')
        lines.append('')
        
        # Agregar clase de tests
        class_name = f'Test{test_suite.component_name.replace("_", "").title()}'
        lines.append(f'class {class_name}(unittest.TestCase):')
        lines.append(f'    """Tests for {test_suite.component_name}."""')
        lines.append('')
        
        # Setup y teardown
        lines.append('    def setUp(self):')
        lines.append('        """Set up test fixtures."""')
        if test_suite.setup_code:
            lines.append(f'        {test_suite.setup_code}')
        else:
            lines.append('        pass')
        lines.append('')
        
        lines.append('    def tearDown(self):')
        lines.append('        """Clean up after tests."""')
        if test_suite.teardown_code:
            lines.append(f'        {test_suite.teardown_code}')
        else:
            lines.append('        pass')
        lines.append('')
        
        # Agregar cada caso de prueba
        for test_case in test_suite.test_cases:
            lines.append(f'    def {test_case.name}(self):')
            lines.append(f'        """{test_case.description}"""')
            
            if test_case.setup_code:
                lines.append(f'        {test_case.setup_code}')
            
            # Generar código de test
            if test_case.expected_exception:
                lines.append(f'        with self.assertRaises({test_case.expected_exception}):')
                lines.append(f'            # Test implementation')
            else:
                lines.append(f'        # Test implementation')
                lines.append(f'        result = None  # TODO: Implement test')
                lines.append(f'        self.assertIsNotNone(result)')
            
            if test_case.teardown_code:
                lines.append(f'        {test_case.teardown_code}')
            
            lines.append('')
        
        # Main block
        lines.append('')
        lines.append('if __name__ == "__main__":')
        lines.append('    unittest.main()')
        
        return '\n'.join(lines)


def create_test_generator(
    test_framework: TestFramework = TestFramework.PYTEST,
    test_complexity: TestComplexity = TestComplexity.COMPREHENSIVE,
    target_coverage: float = 0.80
) -> UnitTestGenerator:
    """
    Factory function para crear UnitTestGenerator.
    
    Args:
        test_framework: Framework de testing a usar
        test_complexity: Nivel de complejidad de los tests
        target_coverage: Cobertura objetivo
        
    Returns:
        Instancia de UnitTestGenerator
    """
    return UnitTestGenerator(
        test_framework=test_framework,
        test_complexity=test_complexity,
        target_coverage=target_coverage
    )
