"""
SAM3 Client
===========

Client for SAM3 model inference, adapted from sam3.agent.client_sam3
but with async support for parallel execution.
"""

import asyncio
import logging
import json
import os
from typing import Optional, Dict, Any
from pathlib import Path
import torch
from PIL import Image

from .sam3_output_processor import (
    process_sam3_outputs,
    build_output_paths,
    filter_outputs_by_indices
)

logger = logging.getLogger(__name__)

# Import SAM3 components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "sam3-main"))

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model.box_ops import box_xyxy_to_xywh
    from sam3.train.masks_ops import rle_encode
    from sam3.agent.helpers.mask_overlap_removal import remove_overlapping_masks
    from sam3.agent.viz import visualize
    SAM3_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SAM3 not available: {e}")
    SAM3_AVAILABLE = False


class SAM3Client:
    """
    Client for SAM3 model inference.
    
    Features:
    - Async inference support
    - Model loading and caching
    - Image processing
    - Result formatting
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize SAM3 client.
        
        Args:
            model_path: Path to SAM3 checkpoint (optional, will download if not provided)
            device: Device to run model on ("cuda" or "cpu")
        """
        if not SAM3_AVAILABLE:
            raise ImportError("SAM3 components not available. Please install sam3-main dependencies.")
        
        self.model_path = model_path
        self.device = device
        self._model = None
        self._processor = None
        self._lock = asyncio.Lock()
        
        logger.info(f"Initialized SAM3Client (device: {device})")
    
    async def _ensure_model_loaded(self):
        """Ensure SAM3 model is loaded (thread-safe)."""
        if self._model is None or self._processor is None:
            async with self._lock:
                if self._model is None or self._processor is None:
                    logger.info("Loading SAM3 model...")
                    
                    # Run model loading in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    self._model, self._processor = await loop.run_in_executor(
                        None,
                        self._load_model
                    )
                    
                    logger.info("SAM3 model loaded successfully")
    
    def _load_model(self) -> tuple:
        """Load SAM3 model and processor (synchronous)."""
        model = build_sam3_image_model(
            checkpoint_path=self.model_path,
            device=self.device,
            eval_mode=True,
            load_from_HF=True if self.model_path is None else False,
        )
        processor = Sam3Processor(model)
        return model, processor
    
    async def call_sam_service(
        self,
        image_path: str,
        text_prompt: str,
        output_folder_path: str = "sam3_output",
    ) -> str:
        """
        Call SAM3 service for image segmentation.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for segmentation
            output_folder_path: Output folder for results
            
        Returns:
            Path to output JSON file
        """
        await self._ensure_model_loaded()
        
        # Run inference in executor
        loop = asyncio.get_event_loop()
        output_json_path = await loop.run_in_executor(
            None,
            self._sam3_inference_sync,
            image_path,
            text_prompt,
            output_folder_path,
        )
        
        return output_json_path
    
    def _sam3_inference_sync(
        self,
        image_path: str,
        text_prompt: str,
        output_folder_path: str,
    ) -> str:
        """Synchronous SAM3 inference."""
        logger.info(f"Running SAM3 inference: {text_prompt}")
        
        # Prepare output paths
        output_json_path, output_image_path = build_output_paths(
            image_path=image_path,
            text_prompt=text_prompt,
            output_folder_path=output_folder_path
        )
        
        # Load and process image
        image = Image.open(image_path)
        orig_img_w, orig_img_h = image.size
        
        # Run inference
        inference_state = self._processor.set_image(image)
        inference_state = self._processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt
        )
        
        # Format outputs
        pred_boxes_xyxy = torch.stack([
            inference_state["boxes"][:, 0] / orig_img_w,
            inference_state["boxes"][:, 1] / orig_img_h,
            inference_state["boxes"][:, 2] / orig_img_w,
            inference_state["boxes"][:, 3] / orig_img_h,
        ], dim=-1)
        
        pred_boxes_xywh = box_xyxy_to_xywh(pred_boxes_xyxy).tolist()
        pred_masks = rle_encode(inference_state["masks"].squeeze(1))
        pred_masks = [m["counts"] for m in pred_masks]
        
        outputs = {
            "original_image_path": image_path,
            "orig_img_h": orig_img_h,
            "orig_img_w": orig_img_w,
            "pred_boxes": pred_boxes_xywh,
            "pred_masks": pred_masks,
            "pred_scores": inference_state["scores"].tolist(),
        }
        
        # Remove overlapping masks
        outputs = remove_overlapping_masks(outputs)
        
        # Process outputs: sort by scores and filter invalid masks
        outputs = process_sam3_outputs(outputs)
        outputs["output_image_path"] = output_image_path
        
        # Save JSON
        with open(output_json_path, "w") as f:
            json.dump(outputs, f, indent=4)
        
        # Render visualization
        viz_image = visualize(outputs)
        os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
        viz_image.save(output_image_path)
        
        logger.info(f"SAM3 inference completed: {len(outputs['pred_masks'])} masks")
        
        return output_json_path
    
    async def close(self):
        """Close SAM3 client and cleanup resources."""
        if self._model is not None:
            # Clear model from memory
            del self._model
            del self._processor
            self._model = None
            self._processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("SAM3Client closed")

