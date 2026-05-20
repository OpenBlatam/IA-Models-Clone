"""
Gradio Utilities for HeyGen AI
================================

Implements Gradio interfaces following best practices:
- User-friendly interfaces
- Proper error handling
- Input validation
- Model inference integration
"""

import logging
from typing import Any, Callable, Dict, List, Optional

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    logging.warning("Gradio not available. Install with: pip install gradio")

logger = logging.getLogger(__name__)


class GradioInterface:
    """Base class for Gradio interfaces.
    
    Features:
    - Error handling
    - Input validation
    - Progress tracking
    - Model integration
    """
    
    def __init__(
        self,
        title: str = "HeyGen AI",
        description: str = "",
    ):
        """Initialize Gradio interface.
        
        Args:
            title: Interface title
            description: Interface description
        """
        if not GRADIO_AVAILABLE:
            raise RuntimeError("Gradio not available. Install with: pip install gradio")
        
        self.title = title
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.GradioInterface")
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface - to be implemented by subclasses.
        
        Returns:
            Gradio Blocks interface
        """
        raise NotImplementedError("Subclasses must implement create_interface")
    
    def launch(
        self,
        server_name: str = "0.0.0.0",
        server_port: int = 7860,
        share: bool = False,
    ) -> None:
        """Launch Gradio interface.
        
        Args:
            server_name: Server hostname
            server_port: Server port
            share: Create public link
        """
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True,
        )


def validate_input(
    input_value: Any,
    input_type: type,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    required: bool = True,
) -> tuple[bool, Optional[str]]:
    """Validate input value.
    
    Args:
        input_value: Value to validate
        input_type: Expected type
        min_value: Minimum value (for numbers)
        max_value: Maximum value (for numbers)
        required: Whether input is required
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if required and input_value is None:
        return False, "Input is required"
    
    if input_value is None:
        return True, None
    
    if not isinstance(input_value, input_type):
        return False, f"Expected {input_type.__name__}, got {type(input_value).__name__}"
    
    if isinstance(input_value, (int, float)):
        if min_value is not None and input_value < min_value:
            return False, f"Value must be >= {min_value}"
        if max_value is not None and input_value > max_value:
            return False, f"Value must be <= {max_value}"
    
    return True, None


def handle_errors(func: Callable) -> Callable:
    """Decorator for error handling in Gradio functions.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function with error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logging.error(f"Gradio function error: {e}", exc_info=True)
            return error_msg
    
    return wrapper



