"""
Validator - Sistema de validación avanzada
============================================
"""

import logging
import re
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class AdvancedValidator:
    """
    Sistema de validación avanzada para código y datos.
    """
    
    def __init__(self):
        """Inicializar validador"""
        self.validation_rules: Dict[str, List[Callable]] = {}
    
    def validate_code(
        self,
        code: str,
        language: str = "python",
        rules: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Valida código según reglas.
        
        Args:
            code: Código a validar
            language: Lenguaje de programación
            rules: Reglas específicas a aplicar (opcional)
            
        Returns:
            Resultado de validación
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        
        # Validaciones básicas
        if not code or not code.strip():
            validation_result["valid"] = False
            validation_result["errors"].append("Código vacío")
            return validation_result
        
        # Validaciones por lenguaje
        if language == "python":
            python_errors = self._validate_python(code)
            validation_result["errors"].extend(python_errors["errors"])
            validation_result["warnings"].extend(python_errors["warnings"])
        elif language in ["javascript", "typescript"]:
            js_errors = self._validate_javascript(code)
            validation_result["errors"].extend(js_errors["errors"])
            validation_result["warnings"].extend(js_errors["warnings"])
        
        # Validaciones generales
        general_errors = self._validate_general(code)
        validation_result["errors"].extend(general_errors["errors"])
        validation_result["warnings"].extend(general_errors["warnings"])
        
        # Determinar si es válido
        validation_result["valid"] = len(validation_result["errors"]) == 0
        
        return validation_result
    
    def _validate_python(self, code: str) -> Dict[str, List[str]]:
        """Valida código Python"""
        errors = []
        warnings = []
        
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Error de sintaxis: {str(e)}")
        
        # Verificar imports peligrosos
        dangerous_imports = ["os.system", "subprocess.call", "eval", "exec"]
        for dangerous in dangerous_imports:
            if dangerous in code:
                warnings.append(f"Uso de {dangerous} detectado - puede ser inseguro")
        
        # Verificar complejidad
        lines = code.split("\n")
        if len(lines) > 1000:
            warnings.append("Código muy largo - considerar dividir en módulos")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_javascript(self, code: str) -> Dict[str, List[str]]:
        """Valida código JavaScript/TypeScript"""
        errors = []
        warnings = []
        
        # Verificar uso de eval
        if "eval(" in code:
            warnings.append("Uso de eval() detectado - puede ser inseguro")
        
        # Verificar uso de var
        if "var " in code:
            warnings.append("Considerar usar let o const en lugar de var")
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_general(self, code: str) -> Dict[str, List[str]]:
        """Validaciones generales"""
        errors = []
        warnings = []
        
        # Verificar líneas muy largas
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            if len(line) > 200:
                warnings.append(f"Línea {i} muy larga ({len(line)} caracteres)")
        
        # Verificar caracteres especiales problemáticos
        if "\t" in code:
            warnings.append("Se detectaron tabs - considerar usar espacios")
        
        return {"errors": errors, "warnings": warnings}
    
    def validate_improvement_result(
        self,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Valida resultado de mejora.
        
        Args:
            result: Resultado de mejora
            
        Returns:
            Resultado de validación
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validar campos requeridos
        required_fields = ["original_code", "improved_code"]
        for field in required_fields:
            if field not in result:
                validation["valid"] = False
                validation["errors"].append(f"Campo requerido faltante: {field}")
        
        # Validar que el código mejorado sea diferente
        if "original_code" in result and "improved_code" in result:
            if result["original_code"] == result["improved_code"]:
                validation["warnings"].append("Código mejorado es idéntico al original")
        
        # Validar sugerencias
        if "suggestions" in result:
            if not isinstance(result["suggestions"], list):
                validation["errors"].append("Suggestions debe ser una lista")
        
        return validation
    
    def validate_paper_data(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Valida datos de paper.
        
        Args:
            paper_data: Datos del paper
            
        Returns:
            Resultado de validación
        """
        validation = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Validar campos requeridos
        if not paper_data.get("title"):
            validation["warnings"].append("Título del paper no encontrado")
        
        if not paper_data.get("content"):
            validation["errors"].append("Contenido del paper vacío")
            validation["valid"] = False
        
        # Validar tamaño
        content_length = len(paper_data.get("content", ""))
        if content_length < 100:
            validation["warnings"].append("Contenido del paper muy corto")
        elif content_length > 1000000:
            validation["warnings"].append("Contenido del paper muy largo")
        
        return validation




