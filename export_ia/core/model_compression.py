"""
Advanced Model Compression Engine for Export IA
State-of-the-art model compression techniques for efficient deployment
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.quantization as quantization
from torch.jit import script, trace
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import math
import copy
from collections import defaultdict
import onnx
import onnxruntime as ort
from onnxconverter_common import float16
import tensorrt as trt
import coremltools as ct
import tflite

logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration for model compression"""
    # Pruning configuration
    pruning_method: str = "magnitude"  # magnitude, gradient, lottery_ticket
    pruning_ratio: float = 0.3
    pruning_structured: bool = False
    pruning_global: bool = True
    
    # Quantization configuration
    quantization_method: str = "dynamic"  # dynamic, static, qat
    quantization_bits: int = 8
    quantization_symmetric: bool = True
    quantization_per_channel: bool = True
    
    # Knowledge distillation
    distillation_enabled: bool = True
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.7
    
    # Low-rank factorization
    low_rank_enabled: bool = False
    low_rank_ratio: float = 0.5
    
    # Neural architecture search for compression
    nas_compression: bool = False
    target_flops: float = 1e9  # Target FLOPS
    
    # Export formats
    export_onnx: bool = True
    export_tensorrt: bool = False
    export_coreml: bool = False
    export_tflite: bool = False

class ModelPruner:
    """Advanced model pruning with multiple strategies"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.pruning_history = []
        self.mask_history = []
        
    def prune_model(self, model: nn.Module, 
                   pruning_ratio: float = None) -> nn.Module:
        """Prune model using specified method"""
        
        if pruning_ratio is None:
            pruning_ratio = self.config.pruning_ratio
            
        if self.config.pruning_method == "magnitude":
            return self._magnitude_pruning(model, pruning_ratio)
        elif self.config.pruning_method == "gradient":
            return self._gradient_pruning(model, pruning_ratio)
        elif self.config.pruning_method == "lottery_ticket":
            return self._lottery_ticket_pruning(model, pruning_ratio)
        else:
            raise ValueError(f"Unknown pruning method: {self.config.pruning_method}")
            
    def _magnitude_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Magnitude-based pruning"""
        pruned_model = copy.deepcopy(model)
        
        # Get all linear and conv layers
        modules_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                modules_to_prune.append((module, 'weight'))
                
        if self.config.pruning_global:
            # Global pruning across all layers
            prune.global_unstructured(
                modules_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio
            )
        else:
            # Layer-wise pruning
            for module, param_name in modules_to_prune:
                prune.l1_unstructured(module, param_name, amount=pruning_ratio)
                
        # Make pruning permanent
        for module, param_name in modules_to_prune:
            prune.remove(module, param_name)
            
        return pruned_model
        
    def _gradient_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Gradient-based pruning (requires training data)"""
        # This would require access to training data and gradients
        # For now, fall back to magnitude pruning
        logger.warning("Gradient pruning requires training data, falling back to magnitude pruning")
        return self._magnitude_pruning(model, pruning_ratio)
        
    def _lottery_ticket_pruning(self, model: nn.Module, pruning_ratio: float) -> nn.Module:
        """Lottery ticket hypothesis pruning"""
        # Initialize model and find winning ticket
        original_state = copy.deepcopy(model.state_dict())
        
        # Apply magnitude pruning
        pruned_model = self._magnitude_pruning(model, pruning_ratio)
        
        # Store the mask (winning ticket)
        mask = {}
        for name, param in pruned_model.named_parameters():
            if param.requires_grad:
                mask[name] = (param != 0).float()
                
        self.mask_history.append(mask)
        
        return pruned_model
        
    def iterative_pruning(self, model: nn.Module, 
                         target_ratio: float,
                         steps: int = 5) -> List[nn.Module]:
        """Iterative pruning for better performance"""
        models = []
        current_model = model
        current_ratio = 0.0
        step_ratio = target_ratio / steps
        
        for step in range(steps):
            current_ratio += step_ratio
            pruned_model = self.prune_model(current_model, step_ratio)
            models.append(pruned_model)
            current_model = pruned_model
            
            # Log pruning statistics
            sparsity = self._calculate_sparsity(pruned_model)
            self.pruning_history.append({
                'step': step,
                'ratio': current_ratio,
                'sparsity': sparsity
            })
            
        return models
        
    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate model sparsity"""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
            
        return zero_params / total_params if total_params > 0 else 0.0

class ModelQuantizer:
    """Advanced model quantization with multiple strategies"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.quantization_history = []
        
    def quantize_model(self, model: nn.Module, 
                      calibration_data: List[torch.Tensor] = None) -> nn.Module:
        """Quantize model using specified method"""
        
        if self.config.quantization_method == "dynamic":
            return self._dynamic_quantization(model)
        elif self.config.quantization_method == "static":
            return self._static_quantization(model, calibration_data)
        elif self.config.quantization_method == "qat":
            return self._quantization_aware_training(model)
        else:
            raise ValueError(f"Unknown quantization method: {self.config.quantization_method}")
            
    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Dynamic quantization (post-training)"""
        quantized_model = quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )
        
        self.quantization_history.append({
            'method': 'dynamic',
            'dtype': 'qint8'
        })
        
        return quantized_model
        
    def _static_quantization(self, model: nn.Module, 
                           calibration_data: List[torch.Tensor]) -> nn.Module:
        """Static quantization with calibration data"""
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
            
        # Set model to evaluation mode
        model.eval()
        
        # Fuse modules for better quantization
        fused_model = self._fuse_modules(model)
        
        # Set quantization configuration
        quantization_config = quantization.QConfig(
            activation=quantization.observer.MinMaxObserver.with_args(
                dtype=torch.quint8,
                qscheme=torch.per_tensor_affine
            ),
            weight=quantization.default_weight_observer
        )
        
        # Prepare model for quantization
        prepared_model = quantization.prepare(fused_model)
        
        # Calibrate with data
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)
                
        # Convert to quantized model
        quantized_model = quantization.convert(prepared_model)
        
        self.quantization_history.append({
            'method': 'static',
            'dtype': 'quint8',
            'calibration_samples': len(calibration_data)
        })
        
        return quantized_model
        
    def _quantization_aware_training(self, model: nn.Module) -> nn.Module:
        """Quantization-aware training"""
        # Set quantization configuration
        quantization_config = quantization.QConfig(
            activation=quantization.observer.MovingAverageMinMaxObserver.with_args(
                dtype=torch.quint8
            ),
            weight=quantization.default_weight_observer
        )
        
        # Prepare model for QAT
        qat_model = quantization.prepare_qat(model, qconfig=quantization_config)
        
        self.quantization_history.append({
            'method': 'qat',
            'dtype': 'quint8'
        })
        
        return qat_model
        
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse modules for better quantization"""
        # This is a simplified version - in practice, you'd need to handle
        # specific module combinations based on your model architecture
        return model

class KnowledgeDistiller:
    """Knowledge distillation for model compression"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.distillation_losses = []
        
    def distill_knowledge(self, teacher_model: nn.Module, 
                         student_model: nn.Module,
                         train_loader: torch.utils.data.DataLoader,
                         optimizer: torch.optim.Optimizer,
                         num_epochs: int = 10) -> nn.Module:
        """Distill knowledge from teacher to student"""
        
        teacher_model.eval()
        student_model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = teacher_model(data)
                    
                # Get student predictions
                student_outputs = student_model(data)
                
                # Calculate distillation loss
                distillation_loss = self._calculate_distillation_loss(
                    student_outputs, teacher_outputs, target
                )
                
                # Backward pass
                distillation_loss.backward()
                optimizer.step()
                
                epoch_loss += distillation_loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            self.distillation_losses.append(avg_loss)
            
            logger.info(f"Distillation Epoch {epoch}: Loss = {avg_loss:.4f}")
            
        return student_model
        
    def _calculate_distillation_loss(self, student_outputs: torch.Tensor,
                                   teacher_outputs: torch.Tensor,
                                   target: torch.Tensor) -> torch.Tensor:
        """Calculate knowledge distillation loss"""
        
        # Soft targets from teacher
        teacher_soft = torch.softmax(teacher_outputs / self.config.distillation_temperature, dim=1)
        student_log_soft = torch.log_softmax(student_outputs / self.config.distillation_temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(
            student_log_soft, teacher_soft
        ) * (self.config.distillation_temperature ** 2)
        
        # Hard targets
        student_soft = torch.softmax(student_outputs, dim=1)
        hard_loss = nn.CrossEntropyLoss()(student_soft, target)
        
        # Combined loss
        total_loss = (self.config.distillation_alpha * distillation_loss + 
                     (1 - self.config.distillation_alpha) * hard_loss)
        
        return total_loss

class LowRankFactorization:
    """Low-rank factorization for model compression"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        
    def factorize_model(self, model: nn.Module) -> nn.Module:
        """Apply low-rank factorization to model"""
        factorized_model = copy.deepcopy(model)
        
        for name, module in factorized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Factorize linear layers
                factorized_module = self._factorize_linear(module)
                # Replace module (this is simplified - in practice you'd need proper replacement)
                
        return factorized_model
        
    def _factorize_linear(self, linear_layer: nn.Linear) -> nn.Module:
        """Factorize linear layer into two smaller layers"""
        input_dim, output_dim = linear_layer.weight.shape
        
        # Calculate rank
        rank = max(1, int(min(input_dim, output_dim) * self.config.low_rank_ratio))
        
        # Create factorized layers
        first_layer = nn.Linear(input_dim, rank, bias=False)
        second_layer = nn.Linear(rank, output_dim, bias=linear_layer.bias is not None)
        
        # Initialize weights using SVD
        U, S, V = torch.svd(linear_layer.weight)
        first_layer.weight.data = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
        second_layer.weight.data = torch.diag(torch.sqrt(S[:rank])) @ V[:, :rank].T
        
        if linear_layer.bias is not None:
            second_layer.bias.data = linear_layer.bias.data
            
        # Create sequential module
        factorized = nn.Sequential(first_layer, second_layer)
        
        return factorized

class ModelExporter:
    """Export models to various formats for deployment"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        
    def export_model(self, model: nn.Module, 
                    input_shape: Tuple[int, ...],
                    output_path: str,
                    export_format: str = "onnx") -> str:
        """Export model to specified format"""
        
        if export_format == "onnx":
            return self._export_onnx(model, input_shape, output_path)
        elif export_format == "tensorrt":
            return self._export_tensorrt(model, input_shape, output_path)
        elif export_format == "coreml":
            return self._export_coreml(model, input_shape, output_path)
        elif export_format == "tflite":
            return self._export_tflite(model, input_shape, output_path)
        else:
            raise ValueError(f"Unknown export format: {export_format}")
            
    def _export_onnx(self, model: nn.Module, 
                    input_shape: Tuple[int, ...],
                    output_path: str) -> str:
        """Export model to ONNX format"""
        
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Optimize ONNX model
        onnx_model = onnx.load(output_path)
        optimized_model = self._optimize_onnx(onnx_model)
        
        # Save optimized model
        optimized_path = output_path.replace('.onnx', '_optimized.onnx')
        onnx.save(optimized_model, optimized_path)
        
        return optimized_path
        
    def _optimize_onnx(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """Optimize ONNX model"""
        # Apply ONNX optimizations
        from onnx import optimizer
        
        passes = [
            'eliminate_identity',
            'eliminate_nop_transpose',
            'fuse_consecutive_transposes',
            'fuse_transpose_into_gemm'
        ]
        
        optimized_model = optimizer.optimize(model, passes)
        return optimized_model
        
    def _export_tensorrt(self, model: nn.Module,
                        input_shape: Tuple[int, ...],
                        output_path: str) -> str:
        """Export model to TensorRT format"""
        # First export to ONNX
        onnx_path = output_path.replace('.trt', '.onnx')
        self._export_onnx(model, input_shape, onnx_path)
        
        # Convert ONNX to TensorRT
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        
        with open(onnx_path, 'rb') as model_file:
            parser.parse(model_file.read())
            
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(output_path, 'wb') as f:
            f.write(engine.serialize())
            
        return output_path
        
    def _export_coreml(self, model: nn.Module,
                      input_shape: Tuple[int, ...],
                      output_path: str) -> str:
        """Export model to CoreML format"""
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        # Trace model
        traced_model = torch.jit.trace(model, dummy_input)
        
        # Convert to CoreML
        coreml_model = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=dummy_input.shape)]
        )
        
        # Save model
        coreml_model.save(output_path)
        
        return output_path
        
    def _export_tflite(self, model: nn.Module,
                      input_shape: Tuple[int, ...],
                      output_path: str) -> str:
        """Export model to TensorFlow Lite format"""
        # This would require conversion through TensorFlow
        # For now, return a placeholder
        logger.warning("TensorFlow Lite export not fully implemented")
        return output_path

class ModelCompressionEngine:
    """Main model compression engine combining all techniques"""
    
    def __init__(self, config: CompressionConfig):
        self.config = config
        self.pruner = ModelPruner(config)
        self.quantizer = ModelQuantizer(config)
        self.distiller = KnowledgeDistiller(config)
        self.factorizer = LowRankFactorization(config)
        self.exporter = ModelExporter(config)
        
    def compress_model(self, model: nn.Module,
                      compression_ratio: float = 0.5,
                      calibration_data: List[torch.Tensor] = None) -> Dict[str, Any]:
        """Comprehensive model compression"""
        
        results = {}
        compressed_model = model
        
        # 1. Pruning
        if self.config.pruning_ratio > 0:
            logger.info("Applying model pruning...")
            compressed_model = self.pruner.prune_model(compressed_model)
            results['pruning'] = {
                'sparsity': self.pruner._calculate_sparsity(compressed_model),
                'method': self.config.pruning_method
            }
            
        # 2. Quantization
        if self.config.quantization_method != "none":
            logger.info("Applying model quantization...")
            compressed_model = self.quantizer.quantize_model(compressed_model, calibration_data)
            results['quantization'] = {
                'method': self.config.quantization_method,
                'bits': self.config.quantization_bits
            }
            
        # 3. Low-rank factorization
        if self.config.low_rank_enabled:
            logger.info("Applying low-rank factorization...")
            compressed_model = self.factorizer.factorize_model(compressed_model)
            results['low_rank'] = {
                'ratio': self.config.low_rank_ratio
            }
            
        # 4. Calculate compression metrics
        original_size = self._calculate_model_size(model)
        compressed_size = self._calculate_model_size(compressed_model)
        
        results['compression_metrics'] = {
            'original_size_mb': original_size,
            'compressed_size_mb': compressed_size,
            'compression_ratio': compressed_size / original_size,
            'size_reduction': (original_size - compressed_size) / original_size
        }
        
        return {
            'compressed_model': compressed_model,
            'results': results
        }
        
    def _calculate_model_size(self, model: nn.Module) -> float:
        """Calculate model size in MB"""
        total_params = sum(p.numel() for p in model.parameters())
        # Assuming float32 (4 bytes per parameter)
        size_mb = total_params * 4 / (1024 * 1024)
        return size_mb

# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test model compression
    print("Testing Model Compression Engine...")
    
    # Create a simple model for testing
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * 32 * 32, 512)
            self.fc2 = nn.Linear(512, 10)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Create test model
    model = TestModel()
    
    # Create compression config
    config = CompressionConfig(
        pruning_method="magnitude",
        pruning_ratio=0.3,
        quantization_method="dynamic",
        quantization_bits=8
    )
    
    # Create compression engine
    engine = ModelCompressionEngine(config)
    
    # Compress model
    print("Compressing model...")
    results = engine.compress_model(model)
    
    print(f"Compression results:")
    print(f"Original size: {results['results']['compression_metrics']['original_size_mb']:.2f} MB")
    print(f"Compressed size: {results['results']['compression_metrics']['compressed_size_mb']:.2f} MB")
    print(f"Compression ratio: {results['results']['compression_metrics']['compression_ratio']:.2f}")
    print(f"Size reduction: {results['results']['compression_metrics']['size_reduction']:.2%}")
    
    # Test individual components
    print("\nTesting individual compression components...")
    
    # Test pruning
    pruned_model = engine.pruner.prune_model(model, 0.2)
    sparsity = engine.pruner._calculate_sparsity(pruned_model)
    print(f"Pruning sparsity: {sparsity:.2%}")
    
    # Test quantization
    quantized_model = engine.quantizer.quantize_model(model)
    print(f"Quantization method: {engine.quantizer.quantization_history[-1]['method']}")
    
    print("\nModel compression engine initialized successfully!")
























