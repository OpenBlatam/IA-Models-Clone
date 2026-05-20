"""
Model Visualizer
Visualize model architecture and statistics
"""

from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.tensorboard import SummaryWriter
    TORCH_AVAILABLE = True
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TENSORBOARD_AVAILABLE = False
    logger.warning("PyTorch or TensorBoard not available")


class ModelVisualizer:
    """Visualize model"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        self.writer = None
    
    def visualize_model_graph(
        self,
        model: nn.Module,
        input_shape: tuple,
        log_dir: Optional[str] = None
    ):
        """Visualize model graph in TensorBoard"""
        if not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available")
            return
        
        log_dir = log_dir or self.log_dir
        writer = SummaryWriter(log_dir)
        
        try:
            dummy_input = torch.randn(input_shape)
            writer.add_graph(model, dummy_input)
            logger.info(f"Model graph saved to {log_dir}")
        except Exception as e:
            logger.error(f"Error visualizing model graph: {str(e)}")
        finally:
            writer.close()
    
    def visualize_weights(
        self,
        model: nn.Module,
        step: int = 0,
        log_dir: Optional[str] = None
    ):
        """Visualize model weights in TensorBoard"""
        if not TENSORBOARD_AVAILABLE:
            logger.warning("TensorBoard not available")
            return
        
        log_dir = log_dir or self.log_dir
        writer = SummaryWriter(log_dir)
        
        try:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f"weights/{name}", param, step)
                    if param.grad is not None:
                        writer.add_histogram(f"gradients/{name}", param.grad, step)
            
            logger.info(f"Weights visualization saved to {log_dir}")
        except Exception as e:
            logger.error(f"Error visualizing weights: {str(e)}")
        finally:
            writer.close()



