"""
Validators for Physical Store Designer AI
"""

from typing import Any, Callable, Optional
from ..core.exceptions import ValidationError


class Validator:
    """Base validator class"""
    
    @staticmethod
    def validate_store_type(value: Any) -> bool:
        """Validate store type"""
        from .models import StoreType
        if isinstance(value, str):
            try:
                StoreType(value)
                return True
            except ValueError:
                return False
        return isinstance(value, StoreType)
    
    @staticmethod
    def validate_design_style(value: Any) -> bool:
        """Validate design style"""
        from .models import DesignStyle
        if isinstance(value, str):
            try:
                DesignStyle(value)
                return True
            except ValueError:
                return False
        return isinstance(value, DesignStyle)
    
    @staticmethod
    def validate_dimensions(dimensions: dict) -> bool:
        """Validate store dimensions"""
        required = {"width", "length", "height"}
        if not all(key in dimensions for key in required):
            return False
        return all(
            isinstance(dimensions[key], (int, float)) and dimensions[key] > 0
            for key in required
        )
    
    @staticmethod
    def validate_session_id(session_id: str) -> bool:
        """Validate session ID format"""
        import uuid
        try:
            uuid.UUID(session_id)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_store_id(store_id: str) -> bool:
        """Validate store ID format"""
        return bool(store_id and len(store_id) > 0 and len(store_id) <= 100)
    
    @staticmethod
    def validate_budget_range(budget: str) -> bool:
        """Validate budget range"""
        valid_budgets = ["bajo", "medio", "alto", "premium"]
        return budget.lower() in valid_budgets if isinstance(budget, str) else False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format (simple)"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email)) if isinstance(email, str) else False
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format (simple)"""
        import re
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return bool(re.match(pattern, url)) if isinstance(url, str) else False
    
    @staticmethod
    def validate_positive_number(value: Any) -> bool:
        """Validate that value is a positive number"""
        try:
            num = float(value)
            return num > 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_range(value: Any, min_val: float, max_val: float) -> bool:
        """Validate that value is within range"""
        try:
            num = float(value)
            return min_val <= num <= max_val
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_string_length(value: str, min_len: int = 1, max_len: int = 1000) -> bool:
        """Validate string length"""
        if not isinstance(value, str):
            return False
        return min_len <= len(value) <= max_len
    
    @staticmethod
    def validate_uuid(value: str) -> bool:
        """Validate UUID format"""
        import uuid
        try:
            uuid.UUID(value)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_iso_date(date_str: str) -> bool:
        """Validate ISO date format (YYYY-MM-DD)"""
        from datetime import datetime
        try:
            datetime.fromisoformat(date_str)
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_list_not_empty(value: list) -> bool:
        """Validate that list is not empty"""
        return isinstance(value, list) and len(value) > 0
    
    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """Validate phone number format (international format)"""
        import re
        # Basic international phone validation (digits, +, spaces, dashes, parentheses)
        pattern = r'^\+?[\d\s\-\(\)]{7,20}$'
        return bool(re.match(pattern, phone)) if isinstance(phone, str) else False
    
    @staticmethod
    def validate_postal_code(postal_code: str) -> bool:
        """Validate postal code format (alphanumeric, 3-10 chars)"""
        import re
        pattern = r'^[A-Z0-9\s\-]{3,10}$'
        return bool(re.match(pattern, postal_code.upper())) if isinstance(postal_code, str) else False
    
    @staticmethod
    def validate_percentage(value: Any) -> bool:
        """Validate percentage value (0-100)"""
        try:
            num = float(value)
            return 0 <= num <= 100
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_currency_amount(value: Any, min_amount: float = 0.0) -> bool:
        """Validate currency amount (positive number with 2 decimal places)"""
        try:
            num = float(value)
            if num < min_amount:
                return False
            # Check if it has at most 2 decimal places
            return round(num, 2) == num
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_color_hex(color: str) -> bool:
        """Validate hex color code (#RRGGBB or #RRGGBBAA)"""
        import re
        pattern = r'^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{8})$'
        return bool(re.match(pattern, color)) if isinstance(color, str) else False
    
    @staticmethod
    def validate_coordinates(lat: Any, lon: Any) -> bool:
        """Validate latitude and longitude coordinates"""
        try:
            lat_num = float(lat)
            lon_num = float(lon)
            return -90 <= lat_num <= 90 and -180 <= lon_num <= 180
        except (ValueError, TypeError):
            return False


def validate_and_raise(validator: Callable[[Any], bool], value: Any, error_message: str):
    """Validate value and raise ValidationError if invalid"""
    if not validator(value):
        raise ValidationError(error_message, details={"value": value})

