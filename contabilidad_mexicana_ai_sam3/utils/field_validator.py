"""
Field validation utilities for validators.

Refactored to consolidate field validation patterns.
"""

from typing import Dict, Any, Optional, Tuple, List


class FieldValidator:
    """
    Validates fields in data dictionaries.
    
    Responsibilities:
    - Validate required fields
    - Validate field types
    - Validate field values
    
    Single Responsibility: Handle all field validation operations.
    """
    
    @staticmethod
    def validate_required_fields(
        datos: Dict[str, Any],
        required_fields: List[str]
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate that all required fields are present.
        
        Args:
            datos: Data dictionary
            required_fields: List of required field names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        for field in required_fields:
            if field not in datos:
                return False, f"Campo requerido faltante: {field}"
        return True, None
    
    @staticmethod
    def validate_numeric_field(
        datos: Dict[str, Any],
        field: str,
        allow_negative: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a numeric field.
        
        Args:
            datos: Data dictionary
            field: Field name
            allow_negative: Whether negative values are allowed
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if field not in datos:
            return True, None  # Optional field
        
        value = datos[field]
        
        if not isinstance(value, (int, float)):
            return False, f"Campo {field} debe ser numérico"
        
        if not allow_negative and value < 0:
            return False, f"Campo {field} no puede ser negativo"
        
        if min_value is not None and value < min_value:
            return False, f"Campo {field} debe ser mayor o igual a {min_value}"
        
        if max_value is not None and value > max_value:
            return False, f"Campo {field} debe ser menor o igual a {max_value}"
        
        return True, None
    
    @staticmethod
    def validate_field_comparison(
        datos: Dict[str, Any],
        field1: str,
        field2: str,
        comparison: str = "less_than_or_equal"
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate comparison between two fields.
        
        Args:
            datos: Data dictionary
            field1: First field name
            field2: Second field name
            comparison: Type of comparison ("less_than_or_equal", "greater_than_or_equal", "equal")
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if field1 not in datos or field2 not in datos:
            return True, None  # Skip if fields not present
        
        val1 = datos[field1]
        val2 = datos[field2]
        
        if comparison == "less_than_or_equal":
            if val1 > val2:
                return False, f"{field1} no puede ser mayor que {field2}"
        elif comparison == "greater_than_or_equal":
            if val1 < val2:
                return False, f"{field1} no puede ser menor que {field2}"
        elif comparison == "equal":
            if val1 != val2:
                return False, f"{field1} debe ser igual a {field2}"
        
        return True, None

