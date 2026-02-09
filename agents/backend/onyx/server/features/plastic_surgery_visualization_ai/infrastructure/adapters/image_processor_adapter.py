"""Adapter for ImageProcessor to implement IImageProcessor interface."""

from core.interfaces import IImageProcessor
from core.services.image_processor import ImageProcessor

# Type alias for backward compatibility
ImageProcessorAdapter = ImageProcessor

