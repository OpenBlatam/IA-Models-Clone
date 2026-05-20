"""
Metrics Calculator
=================

Handles quality metrics calculation.
"""

import logging
from typing import Optional, Dict, Any, Union
from PIL import Image
import numpy as np

from ...models.quality_metrics import QualityMetrics

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates quality metrics for clothing changes."""
    
    def __init__(self):
        """Initialize Metrics Calculator."""
        self.quality_metrics = QualityMetrics()
    
    def calculate_metrics(
        self,
        original_image: Image.Image,
        changed_image: Image.Image,
        mask: Optional[Union[Image.Image, np.ndarray]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate quality metrics.
        
        Args:
            original_image: Original image
            changed_image: Changed image
            mask: Optional mask
            
        Returns:
            Metrics dictionary or None if calculation fails
        """
        try:
            metrics = self.quality_metrics.calculate_metrics(
                original_image=original_image,
                changed_image=changed_image,
                mask=mask,
            )
            logger.info(f"Quality metrics: overall={metrics.get('overall_quality', 0):.3f}")
            return metrics
        except Exception as e:
            logger.warning(f"Error calculating metrics: {e}")
            return None


