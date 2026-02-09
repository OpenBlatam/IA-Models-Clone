"""
Test Generator
==============

Generador automático de tests para código generado.
"""

import logging
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """Caso de prueba."""
    name: str
    test_type: str  # 'unit', 'integration', 'performance'
    code: str
    description: str = ""


class TestGenerator:
    """
    Generador de tests automáticos.
    """
    
    def __init__(self):
        """Inicializar generador."""
        pass
    
    def generate_tests_for_file(
        self,
        file_path: Path,
        output_dir: Path
    ) -> List[TestCase]:
        """
        Generar tests para un archivo.
        
        Args:
            file_path: Archivo a testear
            output_dir: Directorio de salida
            
        Returns:
            Lista de casos de prueba generados
        """
        if not file_path.exists() or file_path.suffix != '.py':
            return []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            test_cases = []
            
            # Generar tests para clases
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_tests = self._generate_class_tests(node, file_path)
                    test_cases.extend(class_tests)
                
                elif isinstance(node, ast.FunctionDef):
                    if not self._is_private(node.name):
                        func_tests = self._generate_function_tests(node, file_path)
                        test_cases.extend(func_tests)
            
            # Escribir archivo de tests
            if test_cases:
                self._write_test_file(file_path, output_dir, test_cases)
            
            return test_cases
        except Exception as e:
            logger.warning(f"Error generando tests para {file_path}: {e}")
            return []
    
    def _generate_class_tests(
        self,
        class_node: ast.ClassDef,
        source_file: Path
    ) -> List[TestCase]:
        """Generar tests para una clase."""
        test_cases = []
        
        # Test de inicialización
        if '__init__' in [m.name for m in class_node.body if isinstance(m, ast.FunctionDef)]:
            test_code = f"""
def test_{class_node.name.lower()}_initialization():
    \"\"\"Test de inicialización de {class_node.name}.\"\"\"
    from {self._get_module_path(source_file)} import {class_node.name}
    
    # TODO: Agregar parámetros de inicialización
    instance = {class_node.name}()
    assert instance is not None
"""
            test_cases.append(TestCase(
                name=f"test_{class_node.name.lower()}_initialization",
                test_type="unit",
                code=test_code,
                description=f"Test de inicialización para {class_node.name}"
            ))
        
        # Tests para métodos públicos
        for method in class_node.body:
            if isinstance(method, ast.FunctionDef) and not self._is_private(method.name):
                test_code = f"""
def test_{class_node.name.lower()}_{method.name}():
    \"\"\"Test del método {method.name} de {class_node.name}.\"\"\"
    from {self._get_module_path(source_file)} import {class_node.name}
    
    instance = {class_node.name}()
    # TODO: Agregar test del método
    # result = instance.{method.name}()
    # assert result is not None
"""
                test_cases.append(TestCase(
                    name=f"test_{class_node.name.lower()}_{method.name}",
                    test_type="unit",
                    code=test_code,
                    description=f"Test del método {method.name}"
                ))
        
        return test_cases
    
    def _generate_function_tests(
        self,
        func_node: ast.FunctionDef,
        source_file: Path
    ) -> List[TestCase]:
        """Generar tests para una función."""
        test_code = f"""
def test_{func_node.name}():
    \"\"\"Test de la función {func_node.name}.\"\"\"
    from {self._get_module_path(source_file)} import {func_node.name}
    
    # TODO: Agregar test
    # result = {func_node.name}()
    # assert result is not None
"""
        return [TestCase(
            name=f"test_{func_node.name}",
            test_type="unit",
            code=test_code,
            description=f"Test de la función {func_node.name}"
        )]
    
    def _is_private(self, name: str) -> bool:
        """Verificar si es privado."""
        return name.startswith('_')
    
    def _get_module_path(self, file_path: Path) -> str:
        """Obtener ruta del módulo."""
        # Simplificado - se puede mejorar
        parts = file_path.parts
        if 'app' in parts:
            idx = parts.index('app')
            module_parts = parts[idx+1:-1] + (file_path.stem,)
            return '.'.join(module_parts)
        return file_path.stem
    
    def _write_test_file(
        self,
        source_file: Path,
        output_dir: Path,
        test_cases: List[TestCase]
    ) -> None:
        """Escribir archivo de tests."""
        test_file = output_dir / f"test_{source_file.name}"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        content = f'''"""
Tests automáticos para {source_file.name}
==========================================

Generado automáticamente por TestGenerator.
"""

import pytest
'''
        
        for test_case in test_cases:
            content += test_case.code + '\n'
        
        test_file.write_text(content, encoding='utf-8')
        logger.info(f"Archivo de tests generado: {test_file}")
    
    def generate_tests_for_project(
        self,
        project_dir: Path,
        output_dir: Optional[Path] = None
    ) -> Dict[str, List[TestCase]]:
        """
        Generar tests para todo el proyecto.
        
        Args:
            project_dir: Directorio del proyecto
            output_dir: Directorio de salida (opcional)
            
        Returns:
            Diccionario con tests por archivo
        """
        if output_dir is None:
            output_dir = project_dir / "tests"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        # Generar tests para archivos en app/
        app_dir = project_dir / "app"
        if app_dir.exists():
            for py_file in app_dir.rglob("*.py"):
                if py_file.name == "__init__.py":
                    continue
                
                test_cases = self.generate_tests_for_file(py_file, output_dir)
                if test_cases:
                    results[str(py_file.relative_to(project_dir))] = test_cases
        
        return results


# Instancia global
_global_test_generator: Optional[TestGenerator] = None


def get_test_generator() -> TestGenerator:
    """
    Obtener instancia global del generador de tests.
    
    Returns:
        Instancia del generador
    """
    global _global_test_generator
    
    if _global_test_generator is None:
        _global_test_generator = TestGenerator()
    
    return _global_test_generator

