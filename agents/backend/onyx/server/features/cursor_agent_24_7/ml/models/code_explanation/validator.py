"""
Input validation for code explanation model
"""

from typing import List, Optional


class InputValidator:
    """Validates inputs for code explanation"""
    
    @staticmethod
    def validate_code(code: str, max_length: Optional[int] = None) -> str:
        """
        Validate code input.
        
        Args:
            code: Code to validate
            max_length: Maximum length in tokens (for character validation)
            
        Returns:
            Validated and stripped code
            
        Raises:
            ValueError: If code is invalid
        """
        if not isinstance(code, str):
            raise ValueError(f"code must be a string, got {type(code)}")
        
        code = code.strip()
        if not code:
            raise ValueError("code cannot be empty")
        
        # Validar longitud aproximada (4 caracteres por token)
        if max_length is not None:
            max_chars = max_length * 4
            if len(code) > max_chars:
                raise ValueError(
                    f"code too long: {len(code)} characters. "
                    f"Maximum: {max_chars} characters"
                )
        
        return code
    
    @staticmethod
    def validate_generation_params(
        max_length: Optional[int] = None,
        temperature: float = 0.7,
        num_beams: int = 4
    ) -> None:
        """Validate generation parameters
        
        Args:
            max_length: Maximum generation length
            temperature: Sampling temperature
            num_beams: Number of beams for beam search
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if max_length is not None and (not isinstance(max_length, int) or max_length <= 0):
            raise ValueError("max_length must be a positive integer")
        if not isinstance(temperature, (int, float)) or temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not isinstance(num_beams, int) or num_beams < 1:
            raise ValueError("num_beams must be a positive integer")
    
    @staticmethod
    def validate_config(config: dict) -> None:
        """Validate model configuration
        
        Args:
            config: Configuration dictionary
            
        Raises:
            ValueError: If configuration is invalid
        """
        model_name = config.get("model_name", "t5-small")
        max_length = config.get("max_length", 512)
        max_target_length = config.get("max_target_length", 128)
        
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if not isinstance(max_target_length, int) or max_target_length <= 0:
            raise ValueError("max_target_length must be a positive integer")
    
    @staticmethod
    def validate_batch_codes(codes: List[str], min_valid: int = 1) -> List[str]:
        """
        Validate and filter batch codes.
        
        Args:
            codes: List of codes to validate
            min_valid: Minimum number of valid codes required
            
        Returns:
            List of valid codes
            
        Raises:
            ValueError: If codes list is invalid or has insufficient valid codes
        """
        if not isinstance(codes, list):
            raise ValueError("codes must be a list")
        if not codes:
            raise ValueError("codes list cannot be empty")
        
        valid_codes = []
        for code in codes:
            if isinstance(code, str) and code.strip():
                valid_codes.append(code.strip())
        
        if len(valid_codes) < min_valid:
            raise ValueError(
                f"Insufficient valid codes: {len(valid_codes)} valid out of {len(codes)} total. "
                f"Minimum required: {min_valid}"
            )
        
        return valid_codes

