"""
Model Pruning and Distillation
Techniques for creating smaller, faster models
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def prune_model(
    model: nn.Module,
    pruning_method: str = "l1_unstructured",
    amount: float = 0.2,
    modules_to_prune: Optional[List[str]] = None
) -> nn.Module:
    """
    Prune model to reduce size and speed up inference
    
    Args:
        model: PyTorch model
        pruning_method: "l1_unstructured", "l1_structured", "ln_structured", "random_unstructured"
        amount: Fraction of parameters to prune (0.0 to 1.0)
        modules_to_prune: List of module names to prune (None = all linear/conv)
        
    Returns:
        Pruned model
    """
    model.eval()
    
    # Default modules to prune
    if modules_to_prune is None:
        modules_to_prune = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
    
    # Apply pruning
    if pruning_method == "l1_unstructured":
        for module, param_name in modules_to_prune:
            prune.l1_unstructured(module, param_name, amount=amount)
    elif pruning_method == "l1_structured":
        for module, param_name in modules_to_prune:
            prune.ln_structured(module, param_name, amount=amount, n=1, dim=0)
    elif pruning_method == "random_unstructured":
        for module, param_name in modules_to_prune:
            prune.random_unstructured(module, param_name, amount=amount)
    else:
        raise ValueError(f"Unknown pruning method: {pruning_method}")
    
    # Make pruning permanent
    for module, param_name in modules_to_prune:
        prune.remove(module, param_name)
    
    logger.info(f"Model pruned with {pruning_method}, amount={amount}")
    return model


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Get model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    
    return {
        'total_mb': total_size / (1024 ** 2),
        'params_mb': param_size / (1024 ** 2),
        'buffers_mb': buffer_size / (1024 ** 2),
        'num_params': sum(p.numel() for p in model.parameters()),
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }


class KnowledgeDistillation:
    """
    Knowledge distillation for model compression
    Train smaller student model to mimic larger teacher model
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.5
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for distillation loss vs hard target loss
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        temperature: Optional[float] = None
    ) -> torch.Tensor:
        """
        Compute distillation loss
        
        Args:
            student_logits: Student model logits
            teacher_logits: Teacher model logits
            targets: Hard targets
            temperature: Softmax temperature (None = use self.temperature)
            
        Returns:
            Combined loss
        """
        temp = temperature or self.temperature
        
        # Soft targets (distillation)
        student_soft = nn.functional.log_softmax(student_logits / temp, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / temp, dim=1)
        distillation_loss = nn.functional.kl_div(
            student_soft, teacher_soft, reduction='batchmean'
        ) * (temp ** 2)
        
        # Hard targets
        hard_loss = nn.functional.cross_entropy(student_logits, targets)
        
        # Combined
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * hard_loss
        
        return total_loss


def create_quantized_model(
    model: nn.Module,
    quantization_type: str = "dynamic",
    calibration_data: Optional[Any] = None
) -> nn.Module:
    """
    Create quantized version of model
    
    Args:
        model: Original model
        quantization_type: "dynamic", "static", "qat" (quantization-aware training)
        calibration_data: Data for static quantization calibration
        
    Returns:
        Quantized model
    """
    model.eval()
    
    if quantization_type == "dynamic":
        # Dynamic quantization (no calibration needed)
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        logger.info("Model quantized with dynamic quantization")
    
    elif quantization_type == "static":
        # Static quantization (requires calibration)
        if calibration_data is None:
            raise ValueError("calibration_data required for static quantization")
        
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(model, inplace=True)
        
        # Calibrate
        with torch.no_grad():
            for data in calibration_data:
                _ = model(data)
        
        quantized = torch.quantization.convert(model, inplace=True)
        logger.info("Model quantized with static quantization")
    
    elif quantization_type == "qat":
        # Quantization-aware training (requires training)
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        logger.info("Model prepared for quantization-aware training")
        quantized = model
    
    else:
        raise ValueError(f"Unknown quantization type: {quantization_type}")
    
    return quantized


def compare_models(
    original_model: nn.Module,
    optimized_model: nn.Module,
    test_input: torch.Tensor
) -> Dict[str, Any]:
    """
    Compare original and optimized models
    
    Returns:
        Dictionary with comparison metrics
    """
    original_model.eval()
    optimized_model.eval()
    
    # Size comparison
    original_size = get_model_size(original_model)
    optimized_size = get_model_size(optimized_model)
    
    # Speed comparison
    import time
    
    # Warmup
    with torch.no_grad():
        _ = original_model(test_input)
        _ = optimized_model(test_input)
    
    if test_input.device.type == "cuda":
        torch.cuda.synchronize()
    
    # Original speed
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = original_model(test_input)
    if test_input.device.type == "cuda":
        torch.cuda.synchronize()
    original_time = (time.time() - start) / 10
    
    # Optimized speed
    start = time.time()
    with torch.no_grad():
        for _ in range(10):
            _ = optimized_model(test_input)
    if test_input.device.type == "cuda":
        torch.cuda.synchronize()
    optimized_time = (time.time() - start) / 10
    
    # Accuracy comparison (if possible)
    with torch.no_grad():
        original_output = original_model(test_input)
        optimized_output = optimized_model(test_input)
    
    # Calculate similarity
    if original_output.shape == optimized_output.shape:
        similarity = torch.nn.functional.cosine_similarity(
            original_output.flatten(),
            optimized_output.flatten(),
            dim=0
        ).item()
    else:
        similarity = None
    
    return {
        'original_size_mb': original_size['total_mb'],
        'optimized_size_mb': optimized_size['total_mb'],
        'size_reduction': 1 - (optimized_size['total_mb'] / original_size['total_mb']),
        'original_time_ms': original_time * 1000,
        'optimized_time_ms': optimized_time * 1000,
        'speedup': original_time / optimized_time,
        'similarity': similarity
    }













