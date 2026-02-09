"""Model registry for different AI providers."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional
from PIL import Image

from api.schemas.visualization import SurgeryType
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelProvider(str, Enum):
    """Supported AI model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class AIModel(ABC):
    """Abstract base class for AI models."""
    
    @abstractmethod
    async def process_surgery_visualization(
        self,
        image: Image.Image,
        surgery_type: SurgeryType,
        intensity: float,
        target_areas: Optional[list] = None
    ) -> Image.Image:
        """Process image for surgery visualization."""
        pass


class OpenAIModel(AIModel):
    """OpenAI GPT-4 Vision model implementation."""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4-vision-preview"):
        self.api_key = api_key
        self.model_name = model_name
    
    async def process_surgery_visualization(
        self,
        image: Image.Image,
        surgery_type: SurgeryType,
        intensity: float,
        target_areas: Optional[list] = None
    ) -> Image.Image:
        """Process using OpenAI Vision API."""
        # TODO: Implement OpenAI Vision API integration
        logger.info(f"Processing with OpenAI {self.model_name}")
        return image


class AnthropicModel(AIModel):
    """Anthropic Claude Vision model implementation."""
    
    def __init__(self, api_key: str, model_name: str = "claude-3-opus-20240229"):
        self.api_key = api_key
        self.model_name = model_name
    
    async def process_surgery_visualization(
        self,
        image: Image.Image,
        surgery_type: SurgeryType,
        intensity: float,
        target_areas: Optional[list] = None
    ) -> Image.Image:
        """Process using Anthropic Claude Vision API."""
        # TODO: Implement Anthropic Claude Vision API integration
        logger.info(f"Processing with Anthropic {self.model_name}")
        return image


class LocalModel(AIModel):
    """Local model implementation (for future ML models)."""
    
    async def process_surgery_visualization(
        self,
        image: Image.Image,
        surgery_type: SurgeryType,
        intensity: float,
        target_areas: Optional[list] = None
    ) -> Image.Image:
        """Process using local ML model."""
        # TODO: Implement local model inference
        logger.info("Processing with local model")
        return image


class ModelRegistry:
    """Registry for AI models."""
    
    def __init__(self):
        self.models: dict[ModelProvider, AIModel] = {}
    
    def register_model(self, provider: ModelProvider, model: AIModel):
        """Register a model."""
        self.models[provider] = model
        logger.info(f"Registered model: {provider}")
    
    def get_model(self, provider: ModelProvider) -> Optional[AIModel]:
        """Get model by provider."""
        return self.models.get(provider)
    
    def get_default_model(self) -> Optional[AIModel]:
        """Get default model."""
        if self.models:
            return next(iter(self.models.values()))
        return None

