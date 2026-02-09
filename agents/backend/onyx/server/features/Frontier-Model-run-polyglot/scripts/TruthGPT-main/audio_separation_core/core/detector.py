"""
Separator Detector - Auto-detection of available separators.

Single Responsibility: Detect which separators are available in the system.
This separates detection logic from factory creation logic.
"""

from __future__ import annotations

from typing import List, Optional


class SeparatorDetector:
    """
    Detect available separators in the system.
    
    Single Responsibility: Determine which separators can be used.
    Separated from factory to improve testability and maintainability.
    
    Example:
        detector = SeparatorDetector()
        best = detector.detect_best()  # "demucs"
        available = detector.list_available()  # ["demucs", "spleeter"]
    """
    
    # Priority order for separator selection (best to worst)
    PRIORITY: List[str] = ["demucs", "spleeter", "lalal"]
    
    @classmethod
    def detect_best(cls) -> str:
        """
        Detect the best available separator.
        
        Returns:
            Name of the best available separator
            
        Note:
            Falls back to "spleeter" if no separators are available.
            This ensures the factory always returns a valid type.
        """
        for separator_type in cls.PRIORITY:
            if cls.is_available(separator_type):
                return separator_type
        
        # Fallback to spleeter (most common)
        return "spleeter"
    
    # Mapeo de separadores a módulos a importar
    SEPARATOR_IMPORTS: Dict[str, Optional[str]] = {
        "demucs": "demucs",
        "spleeter": "spleeter",
        "lalal": None,  # LALAL es API-based, no requiere import
    }
    
    @classmethod
    def is_available(cls, separator_type: str) -> bool:
        """
        Check if a separator is available.
        
        Args:
            separator_type: Type of separator to check
        
        Returns:
            True if available, False otherwise
        """
        separator_type = separator_type.lower()
        
        import_name = cls.SEPARATOR_IMPORTS.get(separator_type)
        
        # LALAL siempre disponible (puede requerir API key)
        if import_name is None:
            return separator_type == "lalal"
        
        # Intentar importar módulo
        try:
            __import__(import_name)
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all available separators.
        
        Returns:
            List of available separator names, in priority order
        """
        return [s for s in cls.PRIORITY if cls.is_available(s)]
    
    @classmethod
    def get_priority(cls) -> List[str]:
        """
        Get the priority order of separators.
        
        Returns:
            List of separator types in priority order
        """
        return cls.PRIORITY.copy()

