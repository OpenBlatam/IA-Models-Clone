"""AI processor for surgery visualization."""

import base64
from io import BytesIO
from PIL import Image
from typing import Optional, List

from api.schemas.visualization import SurgeryType
from config.settings import settings
from core.exceptions import AIProcessingError
from core.interfaces import IAProcessor
from core.services.model_registry import (
    ModelRegistry,
    ModelProvider,
    OpenAIModel,
    AnthropicModel,
    LocalModel
)
from utils.logger import get_logger

logger = get_logger(__name__)


class AIProcessor(IAProcessor):
    """Processes images using AI models for surgery visualization."""
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        self.registry = model_registry or self._initialize_registry()
        self.model_provider = ModelProvider(settings.model_provider.lower())
    
    def _initialize_registry(self) -> ModelRegistry:
        """Initialize model registry with available models."""
        registry = ModelRegistry()
        
        # Register OpenAI model if API key is provided
        if settings.api_key and settings.model_provider.lower() == "openai":
            registry.register_model(
                ModelProvider.OPENAI,
                OpenAIModel(settings.api_key, settings.model_name)
            )
        
        # Register Anthropic model if configured
        if settings.model_provider.lower() == "anthropic":
            registry.register_model(
                ModelProvider.ANTHROPIC,
                AnthropicModel(settings.api_key, settings.model_name)
            )
        
        # Register local model
        registry.register_model(
            ModelProvider.LOCAL,
            LocalModel()
        )
        
        return registry
    
    async def process_surgery_visualization(
        self,
        image: Image.Image,
        surgery_type: SurgeryType,
        intensity: float,
        target_areas: Optional[List[str]] = None
    ) -> Image.Image:
        """
        Process image to show surgery visualization.
        
        Args:
            image: Input image
            surgery_type: Type of surgery
            intensity: Intensity of the effect (0.0 to 1.0)
            target_areas: Specific areas to modify
            
        Returns:
            Processed image with surgery visualization
            
        Raises:
            AIProcessingError: If processing fails
        """
        try:
            logger.info(
                f"Processing {surgery_type} visualization "
                f"with intensity {intensity} using {self.model_provider}"
            )
            
            # Get model from registry
            model = self.registry.get_model(self.model_provider)
            if not model:
                # Fallback to default model
                model = self.registry.get_default_model()
                if not model:
                    raise AIProcessingError("No AI model available")
            
            # Process image
            processed_image = await model.process_surgery_visualization(
                image=image,
                surgery_type=surgery_type,
                intensity=intensity,
                target_areas=target_areas
            )
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Error processing image with AI: {e}")
            raise AIProcessingError(f"Failed to process image: {str(e)}")
    
    def _prepare_image_for_ai(self, image: Image.Image) -> str:
        """Convert image to base64 for AI API."""
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64

