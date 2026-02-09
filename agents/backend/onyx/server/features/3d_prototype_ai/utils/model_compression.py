"""
Model Compression System - Sistema de compresión de modelos
============================================================
Pruning, quantization, y knowledge distillation
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
import copy

try:
    import torch_pruning as tp
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False
    logging.warning("torch_pruning not available")

logger = logging.getLogger(__name__)


class ModelCompressor:
    """Sistema de compresión de modelos"""
    
    def __init__(self):
        self.compressed_models: Dict[str, nn.Module] = {}
    
    def quantize_model(
        self,
        model: nn.Module,
        quantization_type: str = "dynamic"
    ) -> nn.Module:
        """Cuantiza modelo"""
        if quantization_type == "dynamic":
            quantized_model = torch.quantization.quantize_dynamic(
                model, {nn.Linear, nn.LSTM, nn.GRU}, dtype=torch.qint8
            )
        elif quantization_type == "static":
            # Preparar modelo para cuantización estática
            model.eval()
            model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
            torch.quantization.prepare(model, inplace=True)
            # Calibrar con datos dummy
            # torch.quantization.convert(model, inplace=True)
            quantized_model = model
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        logger.info(f"Model quantized with {quantization_type} quantization")
        return quantized_model
    
    def prune_model(
        self,
        model: nn.Module,
        pruning_ratio: float = 0.3,
        method: str = "magnitude"
    ) -> nn.Module:
        """Prune modelo"""
        if not PRUNING_AVAILABLE:
            logger.warning("Pruning not available, returning original model")
            return model
        
        pruned_model = copy.deepcopy(model)
        pruned_model.eval()
        
        if method == "magnitude":
            # Magnitude-based pruning
            for module in pruned_model.modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    # Prune weights con menor magnitud
                    threshold = torch.quantile(
                        torch.abs(module.weight.data),
                        pruning_ratio
                    )
                    mask = torch.abs(module.weight.data) > threshold
                    module.weight.data *= mask.float()
        
        logger.info(f"Model pruned with {method} method, ratio: {pruning_ratio}")
        return pruned_model
    
    def distill_knowledge(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.5
    ) -> nn.Module:
        """Knowledge distillation"""
        teacher_model.eval()
        student_model.train()
        
        def distillation_loss(student_logits, teacher_logits, targets, temperature, alpha):
            # Soft targets del teacher
            soft_targets = F.softmax(teacher_logits / temperature, dim=1)
            soft_prob = F.log_softmax(student_logits / temperature, dim=1)
            
            # Soft loss
            soft_loss = F.kl_div(soft_prob, soft_targets, reduction="batchmean") * (temperature ** 2)
            
            # Hard loss
            hard_loss = F.cross_entropy(student_logits, targets)
            
            # Combined
            return alpha * soft_loss + (1 - alpha) * hard_loss
        
        # Retornar función de pérdida y modelo estudiante
        return student_model, distillation_loss
    
    def compress_model(
        self,
        model: nn.Module,
        compression_config: Dict[str, Any]
    ) -> nn.Module:
        """Compresión completa del modelo"""
        compressed = model
        
        if compression_config.get("quantize", False):
            quant_type = compression_config.get("quantization_type", "dynamic")
            compressed = self.quantize_model(compressed, quant_type)
        
        if compression_config.get("prune", False):
            prune_ratio = compression_config.get("pruning_ratio", 0.3)
            prune_method = compression_config.get("pruning_method", "magnitude")
            compressed = self.prune_model(compressed, prune_ratio, prune_method)
        
        logger.info("Model compressed successfully")
        return compressed
    
    def get_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """Obtiene tamaño del modelo"""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        
        return {
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "size_bytes": total_size,
            "size_mb": total_size / (1024 * 1024)
        }




