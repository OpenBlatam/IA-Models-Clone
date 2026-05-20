"""
Gradio module for Blaze AI interactive demos and visualization.
"""

from .interface import BlazeAIGradioInterface, create_blaze_ai_interface
from .demos import (
    InteractiveModelDemos,
    create_text_generation_demo,
    create_image_generation_demo,
    create_model_comparison_demo,
    create_training_visualization_demo,
    create_performance_analysis_demo,
    create_error_analysis_demo
)
from .launcher import GradioLauncher, create_launcher
from .validation import (
    GradioValidator,
    TextGenerationValidator,
    ImageGenerationValidator,
    TrainingValidator,
    GradioErrorHandler,
    SafeGradioExecutor,
    get_gradio_validator,
    get_text_generation_validator,
    get_image_generation_validator,
    get_training_validator,
    get_gradio_error_handler,
    get_safe_gradio_executor
)

__all__ = [
    # Main interface
    "BlazeAIGradioInterface",
    "create_blaze_ai_interface",
    
    # Demo classes
    "InteractiveModelDemos",
    
    # Demo factory functions
    "create_text_generation_demo",
    "create_image_generation_demo",
    "create_model_comparison_demo",
    "create_training_visualization_demo",
    "create_performance_analysis_demo",
    "create_error_analysis_demo",
    
    # Launcher
    "GradioLauncher",
    "create_launcher",
    
    # Validation and error handling
    "GradioValidator",
    "TextGenerationValidator",
    "ImageGenerationValidator",
    "TrainingValidator",
    "GradioErrorHandler",
    "SafeGradioExecutor",
    "get_gradio_validator",
    "get_text_generation_validator",
    "get_image_generation_validator",
    "get_training_validator",
    "get_gradio_error_handler",
    "get_safe_gradio_executor"
]
