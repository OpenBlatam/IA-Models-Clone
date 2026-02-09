"""
Pipeline Optimizer Utilities
============================

Utilities for applying optimizations to diffusers pipelines.
"""

import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PipelineOptimizer:
    """Handles pipeline optimizations for memory and speed."""
    
    @staticmethod
    def apply_optimizations(pipeline, device: torch.device) -> None:
        """
        Apply memory and speed optimizations to a pipeline.
        
        Args:
            pipeline: Diffusers pipeline to optimize
            device: Device the pipeline is on
        """
        if device.type != "cuda":
            logger.debug("Skipping CUDA optimizations for non-CUDA device")
            return
        
        try:
            PipelineOptimizer._enable_attention_slicing(pipeline)
            PipelineOptimizer._enable_xformers(pipeline)
            PipelineOptimizer._compile_transformer(pipeline)
        except Exception as e:
            logger.warning(f"Some optimizations failed: {e}")
    
    @staticmethod
    def _enable_attention_slicing(pipeline) -> None:
        """Enable attention slicing for memory efficiency."""
        if hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing(1)
            logger.debug("Attention slicing enabled")
    
    @staticmethod
    def _enable_xformers(pipeline) -> None:
        """Enable xformers memory efficient attention."""
        try:
            if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                pipeline.enable_xformers_memory_efficient_attention()
                logger.info("XFormers memory efficient attention enabled")
        except Exception as e:
            logger.warning(f"XFormers not available: {e}")
    
    @staticmethod
    def _compile_transformer(pipeline) -> None:
        """Compile transformer with torch.compile (PyTorch 2.0+)."""
        try:
            if hasattr(torch, "compile") and hasattr(pipeline, "transformer"):
                try:
                    pipeline.transformer = torch.compile(
                        pipeline.transformer,
                        mode="reduce-overhead"
                    )
                    logger.info("Transformer compiled with torch.compile")
                except Exception as e:
                    logger.warning(f"Could not compile transformer: {e}")
        except Exception as e:
            logger.warning(f"torch.compile not available or failed: {e}")


