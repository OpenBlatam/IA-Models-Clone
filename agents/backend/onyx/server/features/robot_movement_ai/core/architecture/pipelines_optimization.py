"""
Model Optimization Module
==========================

Sistema profesional de optimización de modelos.
Incluye pruning avanzado, knowledge distillation, y optimizaciones de arquitectura.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    logging.warning("PyTorch not available. Optimization disabled.")

logger = logging.getLogger(__name__)


class KnowledgeDistillation:
    """
    Knowledge Distillation para transferir conocimiento de teacher a student.
    
    Permite comprimir modelos grandes en modelos más pequeños manteniendo performance.
    """
    
    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module,
        temperature: float = 3.0,
        alpha: float = 0.7
    ):
        """
        Inicializar knowledge distillation.
        
        Args:
            teacher_model: Modelo teacher (grande, pre-entrenado)
            student_model: Modelo student (pequeño, a entrenar)
            temperature: Temperature para softmax
            alpha: Peso entre loss de teacher y ground truth
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        self.teacher_model.eval()
        logger.info("KnowledgeDistillation initialized")
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        hard_loss_fn: Callable = nn.CrossEntropyLoss(),
        soft_loss_fn: Callable = nn.KLDivLoss(reduction='batchmean')
    ) -> torch.Tensor:
        """
        Calcular loss de distillation.
        
        Args:
            student_logits: Logits del student
            teacher_logits: Logits del teacher
            targets: Targets reales
            hard_loss_fn: Función de pérdida para hard targets
            soft_loss_fn: Función de pérdida para soft targets
            
        Returns:
            Loss total
        """
        # Soft targets (distillation loss)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = soft_loss_fn(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Hard targets (student loss)
        hard_loss = hard_loss_fn(student_logits, targets)
        
        # Combined loss
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return total_loss
    
    def train_step(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """
        Un paso de entrenamiento de distillation.
        
        Args:
            inputs: Inputs
            targets: Targets
            optimizer: Optimizador
            
        Returns:
            Dict con losses
        """
        self.student_model.train()
        self.teacher_model.eval()
        
        optimizer.zero_grad()
        
        # Forward pass
        with torch.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        student_logits = self.student_model(inputs)
        
        # Compute loss
        loss = self.compute_distillation_loss(student_logits, teacher_logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {
            "distillation_loss": loss.item(),
            "student_loss": F.cross_entropy(student_logits, targets).item()
        }


class NeuralArchitectureSearch:
    """
    Búsqueda de Arquitectura Neural (NAS) simplificada.
    
    Permite buscar arquitecturas óptimas automáticamente.
    """
    
    def __init__(
        self,
        search_space: Dict[str, List[Any]],
        evaluation_fn: Callable
    ):
        """
        Inicializar NAS.
        
        Args:
            search_space: Espacio de búsqueda de arquitecturas
            evaluation_fn: Función de evaluación (arquitectura -> score)
        """
        self.search_space = search_space
        self.evaluation_fn = evaluation_fn
        self.results: List[Dict[str, Any]] = []
        logger.info("NeuralArchitectureSearch initialized")
    
    def random_search(self, n_trials: int = 20) -> Dict[str, Any]:
        """
        Búsqueda aleatoria de arquitecturas.
        
        Args:
            n_trials: Número de trials
            
        Returns:
            Mejor arquitectura encontrada
        """
        best_score = float('-inf')
        best_architecture = None
        
        for i in range(n_trials):
            # Sample architecture
            architecture = {
                key: np.random.choice(values)
                for key, values in self.search_space.items()
            }
            
            # Evaluate
            score = self.evaluation_fn(architecture)
            self.results.append({
                'architecture': architecture,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_architecture = architecture
            
            logger.info(f"Trial {i+1}/{n_trials}: Score = {score:.4f}")
        
        return {
            'architecture': best_architecture,
            'score': best_score,
            'all_results': self.results
        }


class ModelCompression:
    """
    Compresión avanzada de modelos.
    
    Combina múltiples técnicas de compresión.
    """
    
    def __init__(self, model: nn.Module):
        """
        Inicializar compresor.
        
        Args:
            model: Modelo a comprimir
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        
        self.model = model
        self.original_size = self._get_model_size()
        logger.info(f"ModelCompression initialized. Original size: {self.original_size:.2f} MB")
    
    def _get_model_size(self) -> float:
        """Obtener tamaño del modelo en MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)
    
    def compress(
        self,
        methods: List[str] = ["quantization", "pruning"],
        quantization_config: Optional[Dict] = None,
        pruning_amount: float = 0.3
    ) -> nn.Module:
        """
        Comprimir modelo usando múltiples métodos.
        
        Args:
            methods: Lista de métodos a aplicar
            quantization_config: Configuración de cuantización
            pruning_amount: Cantidad de pruning
            
        Returns:
            Modelo comprimido
        """
        compressed_model = self.model
        
        if "pruning" in methods:
            from .pipelines_export import ModelPruner
            pruner = ModelPruner(compressed_model)
            compressed_model = pruner.magnitude_pruning(amount=pruning_amount)
            logger.info("Applied pruning")
        
        if "quantization" in methods:
            from .pipelines_export import ModelQuantizer
            quantizer = ModelQuantizer(compressed_model)
            compressed_model = quantizer.dynamic_quantize()
            logger.info("Applied quantization")
        
        final_size = self._get_model_size()
        compression_ratio = (1 - final_size / self.original_size) * 100
        
        logger.info(
            f"Compression complete: {self.original_size:.2f} MB -> {final_size:.2f} MB "
            f"({compression_ratio:.1f}% reduction)"
        )
        
        return compressed_model


class GradientBasedOptimization:
    """
    Optimizaciones basadas en gradientes.
    
    Incluye gradient checkpointing, gradient accumulation avanzado, etc.
    """
    
    @staticmethod
    def enable_gradient_checkpointing(model: nn.Module):
        """
        Habilitar gradient checkpointing para ahorrar memoria.
        
        Args:
            model: Modelo a optimizar
        """
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    @staticmethod
    def optimize_for_inference(model: nn.Module) -> nn.Module:
        """
        Optimizar modelo para inferencia.
        
        Args:
            model: Modelo a optimizar
            
        Returns:
            Modelo optimizado
        """
        model.eval()
        
        # Fuse operations si es posible
        try:
            if hasattr(torch.quantization, 'fuse_modules'):
                # Ejemplo para modelos comunes
                if hasattr(model, 'features'):
                    torch.quantization.fuse_modules(
                        model.features,
                        [['conv', 'bn', 'relu']],
                        inplace=True
                    )
                logger.info("Fused modules for inference")
        except Exception as e:
            logger.warning(f"Could not fuse modules: {e}")
        
        # Compilar si está disponible
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("Model compiled for inference")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model

