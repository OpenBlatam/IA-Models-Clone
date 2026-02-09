"""
Test Protector
==============

Sistema que previene la modificación de tests a menos que se solicite explícitamente,
siguiendo las mejores prácticas de Devin de nunca modificar tests a menos que
la tarea lo requiera explícitamente.
"""

import logging
import re
from typing import Optional, Dict, Any, List, Set, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TestFile:
    """Archivo de test detectado"""
    file_path: str
    test_framework: str
    test_count: int = 0
    detected_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "file_path": self.file_path,
            "test_framework": self.test_framework,
            "test_count": self.test_count,
            "detected_at": self.detected_at.isoformat()
        }


@dataclass
class TestModificationAttempt:
    """Intento de modificación de test"""
    file_path: str
    task_id: str
    task_description: str
    allowed: bool = False
    reason: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "file_path": self.file_path,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "allowed": self.allowed,
            "reason": self.reason,
            "timestamp": self.timestamp.isoformat()
        }


class TestProtector:
    """
    Protector de tests.
    
    Previene la modificación de tests a menos que se solicite explícitamente,
    siguiendo las mejores prácticas de Devin.
    """
    
    TEST_FILE_PATTERNS = [
        r'test_.*\.py$',
        r'.*_test\.py$',
        r'.*\.test\.py$',
        r'.*\.spec\.py$',
    ]
    
    TEST_DIRECTORIES = [
        'tests', 'test', 'specs', 'spec', '__tests__'
    ]
    
    TEST_KEYWORDS = [
        'test', 'spec', 'assert', 'pytest', 'unittest', 'nose'
    ]
    
    def __init__(self, workspace_root: Optional[str] = None) -> None:
        """
        Inicializar protector de tests.
        
        Args:
            workspace_root: Raíz del workspace.
        """
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.test_files: Dict[str, TestFile] = {}
        self.modification_attempts: List[TestModificationAttempt] = []
        self.allowed_modifications: Set[str] = set()
        self._scan_test_files()
        logger.info("🛡️ Test protector initialized")
    
    def _scan_test_files(self) -> None:
        """Escanear y detectar archivos de test"""
        try:
            for pattern in self.TEST_FILE_PATTERNS:
                for test_file in self.workspace_root.rglob(pattern):
                    if test_file.is_file():
                        test_file_obj = self._analyze_test_file(str(test_file))
                        if test_file_obj:
                            self.test_files[str(test_file)] = test_file_obj
            
            for test_dir in self.TEST_DIRECTORIES:
                test_dir_path = self.workspace_root / test_dir
                if test_dir_path.exists() and test_dir_path.is_dir():
                    for py_file in test_dir_path.rglob("*.py"):
                        if py_file.is_file():
                            test_file_obj = self._analyze_test_file(str(py_file))
                            if test_file_obj:
                                self.test_files[str(py_file)] = test_file_obj
        except Exception as e:
            logger.error(f"Error scanning test files: {e}", exc_info=True)
    
    def _analyze_test_file(self, file_path: str) -> Optional[TestFile]:
        """Analizar archivo para determinar si es un test"""
        try:
            path = Path(file_path)
            if not path.exists():
                return None
            
            content = path.read_text(encoding='utf-8', errors='ignore')
            
            has_test_keywords = any(
                keyword in content.lower() 
                for keyword in self.TEST_KEYWORDS
            )
            
            if not has_test_keywords:
                return None
            
            test_framework = self._detect_test_framework(content)
            test_count = len(re.findall(r'def test_|def test |@pytest|@unittest', content))
            
            return TestFile(
                file_path=file_path,
                test_framework=test_framework,
                test_count=test_count
            )
        except Exception as e:
            logger.debug(f"Error analyzing test file {file_path}: {e}")
            return None
    
    def _detect_test_framework(self, content: str) -> str:
        """Detectar framework de test"""
        if 'import pytest' in content or 'from pytest' in content:
            return 'pytest'
        elif 'import unittest' in content or 'from unittest' in content:
            return 'unittest'
        elif 'import nose' in content or 'from nose' in content:
            return 'nose'
        else:
            return 'unknown'
    
    def is_test_file(self, file_path: str) -> bool:
        """
        Verificar si un archivo es un archivo de test.
        
        Args:
            file_path: Ruta del archivo.
        
        Returns:
            True si es un archivo de test.
        """
        path = Path(file_path)
        if not path.is_absolute():
            path = self.workspace_root / path
        
        file_str = str(path)
        
        if file_str in self.test_files:
            return True
        
        for pattern in self.TEST_FILE_PATTERNS:
            if re.search(pattern, file_str):
                return True
        
        for test_dir in self.TEST_DIRECTORIES:
            if test_dir in file_str.lower():
                return True
        
        return False
    
    def can_modify_test(
        self,
        file_path: str,
        task_id: str,
        task_description: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verificar si se puede modificar un archivo de test.
        
        Según las reglas de Devin:
        - Nunca modificar tests a menos que la tarea lo requiera explícitamente
        - Si la tarea menciona "modify tests", "update tests", "fix tests", etc., está permitido
        
        Args:
            file_path: Ruta del archivo de test.
            task_id: ID de la tarea.
            task_description: Descripción de la tarea.
        
        Returns:
            Tupla (puede_modificar, razón).
        """
        if not self.is_test_file(file_path):
            return True, "Not a test file"
        
        if file_path in self.allowed_modifications:
            return True, "Explicitly allowed"
        
        task_lower = task_description.lower()
        
        explicit_test_modification_keywords = [
            'modify test', 'update test', 'fix test', 'change test',
            'edit test', 'refactor test', 'improve test',
            'add test', 'remove test', 'delete test',
            'test modification', 'test update', 'test fix'
        ]
        
        has_explicit_permission = any(
            keyword in task_lower 
            for keyword in explicit_test_modification_keywords
        )
        
        if has_explicit_permission:
            self.allowed_modifications.add(file_path)
            attempt = TestModificationAttempt(
                file_path=file_path,
                task_id=task_id,
                task_description=task_description,
                allowed=True,
                reason="Task explicitly requests test modification"
            )
            self.modification_attempts.append(attempt)
            return True, "Task explicitly requests test modification"
        
        attempt = TestModificationAttempt(
            file_path=file_path,
            task_id=task_id,
            task_description=task_description,
            allowed=False,
            reason="Test modification not explicitly requested. Consider fixing the code instead of the test."
        )
        self.modification_attempts.append(attempt)
        
        logger.warning(
            f"⚠️ Attempt to modify test file {file_path} without explicit permission"
        )
        
        return False, "Test modification not explicitly requested. Consider fixing the code instead of the test."
    
    def allow_test_modification(self, file_path: str) -> None:
        """
        Permitir explícitamente la modificación de un archivo de test.
        
        Args:
            file_path: Ruta del archivo de test.
        """
        self.allowed_modifications.add(file_path)
        logger.info(f"✅ Test file {file_path} modification allowed")
    
    def get_test_files(self) -> List[Dict[str, Any]]:
        """Obtener todos los archivos de test detectados"""
        return [tf.to_dict() for tf in self.test_files.values()]
    
    def get_modification_attempts(self) -> List[Dict[str, Any]]:
        """Obtener todos los intentos de modificación"""
        return [ma.to_dict() for ma in self.modification_attempts]

