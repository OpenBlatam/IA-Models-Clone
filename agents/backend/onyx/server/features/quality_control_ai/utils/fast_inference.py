"""
Fast Inference Utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Union
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class FastPreprocessor:
    """Optimized image preprocessing"""
    
    def __init__(self, size=(224, 224), device='cpu'):
        self.size = size
        self.device = torch.device(device)
        self._cache = {}
    
    @lru_cache(maxsize=100)
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Fast preprocessing with caching"""
        if isinstance(image, str):
            img = cv2.imread(image)
        else:
            img = image.copy()
        
        # Fast resize
        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        
        # Convert to tensor
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        img = img[:, :, ::-1]  # BGR to RGB
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0).to(self.device)


class BatchProcessor:
    """Process multiple images in batches"""
    
    def __init__(self, model, batch_size=32, device='cpu'):
        self.model = model
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.preprocessor = FastPreprocessor(device=device)
    
    def process_batch(self, images: List[np.ndarray]) -> List[torch.Tensor]:
        """Process batch of images"""
        batch = []
        results = []
        
        for img in images:
            preprocessed = self.preprocessor.preprocess(img)
            batch.append(preprocessed)
            
            if len(batch) >= self.batch_size:
                batch_tensor = torch.cat(batch, dim=0)
                with torch.no_grad():
                    output = self.model(batch_tensor)
                results.extend([o.cpu() for o in output])
                batch = []
        
        # Process remaining
        if batch:
            batch_tensor = torch.cat(batch, dim=0)
            with torch.no_grad():
                output = self.model(batch_tensor)
            results.extend([o.cpu() for o in output])
        
        return results


class AsyncInference:
    """Asynchronous inference for non-blocking processing"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = torch.device(device)
        self.queue = []
    
    async def infer_async(self, image: np.ndarray):
        """Async inference"""
        import asyncio
        preprocessed = FastPreprocessor(device=self.device).preprocess(image)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self._infer_sync, preprocessed
        )
        return result
    
    def _infer_sync(self, tensor):
        with torch.no_grad():
            return self.model(tensor)


def optimize_model_for_mobile(model: nn.Module):
    """Optimize model for mobile deployment"""
    model.eval()
    
    # Fuse operations
    try:
        torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=True)
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.warning(f"Could not fuse modules: {e}")
    
    # Quantization
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
        )
        return quantized
    except (AttributeError, ValueError, RuntimeError) as e:
        logger.warning(f"Could not quantize model: {e}")
        return model

