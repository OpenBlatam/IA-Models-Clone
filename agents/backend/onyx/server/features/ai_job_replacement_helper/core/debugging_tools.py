"""
Debugging Tools - Herramientas de debugging
===========================================

Sistema de herramientas para debugging de modelos y entrenamiento.
Sigue mejores prácticas de debugging en PyTorch.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DebuggingConfig:
    """Configuración de debugging"""
    enable_anomaly_detection: bool = False
    check_gradients: bool = True
    check_weights: bool = True
    check_activations: bool = False
    log_interval: int = 10
    verbose: bool = False


class DebuggingTools:
    """Herramientas de debugging"""
    
    def __init__(self, config: Optional[DebuggingConfig] = None):
        """
        Inicializar herramientas de debugging.
        
        Args:
            config: Configuración de debugging
        """
        self.config = config or DebuggingConfig()
        self.hooks = []
        self.activation_stats: Dict[str, List[float]] = {}
        
        if self.config.enable_anomaly_detection:
            torch.autograd.set_detect_anomaly(True)
            logger.warning("Anomaly detection enabled. This will slow down training.")
    
    def check_model_health(
        self,
        model: nn.Module,
        input_shape: tuple,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Verificar salud del modelo.
        
        Args:
            model: Modelo a verificar
            input_shape: Forma del input (sin batch dimension)
            device: Dispositivo (None = auto)
        
        Returns:
            Diccionario con información de salud
        """
        health_info = {
            "has_nan_weights": False,
            "has_inf_weights": False,
            "has_zero_weights": False,
            "num_parameters": 0,
            "num_trainable_parameters": 0,
            "forward_pass_ok": False,
            "backward_pass_ok": False,
            "issues": [],
        }
        
        try:
            # Check weights
            for name, param in model.named_parameters():
                if param.requires_grad:
                    health_info["num_trainable_parameters"] += param.numel()
                
                health_info["num_parameters"] += param.numel()
                
                if torch.isnan(param).any():
                    health_info["has_nan_weights"] = True
                    health_info["issues"].append(f"NaN weights in {name}")
                
                if torch.isinf(param).any():
                    health_info["has_inf_weights"] = True
                    health_info["issues"].append(f"Inf weights in {name}")
                
                if (param == 0).all():
                    health_info["has_zero_weights"] = True
                    health_info["issues"].append(f"All zero weights in {name}")
            
            # Test forward pass
            device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            dummy_input = torch.randn(1, *input_shape).to(device)
            
            try:
                with torch.no_grad():
                    output = model(dummy_input)
                    if not (torch.isnan(output).any() or torch.isinf(output).any()):
                        health_info["forward_pass_ok"] = True
                    else:
                        health_info["issues"].append("NaN/Inf in forward pass output")
            except Exception as e:
                health_info["issues"].append(f"Forward pass failed: {str(e)}")
            
            # Test backward pass (if trainable)
            if model.training:
                try:
                    output = model(dummy_input)
                    if isinstance(output, torch.Tensor):
                        loss = output.mean()
                    else:
                        loss = output["logits"].mean() if isinstance(output, dict) else output[0].mean()
                    
                    loss.backward()
                    
                    # Check gradients
                    has_nan_grad = False
                    has_inf_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any():
                                has_nan_grad = True
                                health_info["issues"].append(f"NaN gradient in {name}")
                            if torch.isinf(param.grad).any():
                                has_inf_grad = True
                                health_info["issues"].append(f"Inf gradient in {name}")
                    
                    if not (has_nan_grad or has_inf_grad):
                        health_info["backward_pass_ok"] = True
                    
                    # Clear gradients
                    model.zero_grad()
                
                except Exception as e:
                    health_info["issues"].append(f"Backward pass failed: {str(e)}")
        
        except Exception as e:
            health_info["issues"].append(f"Health check failed: {str(e)}")
            logger.error(f"Error in model health check: {e}", exc_info=True)
        
        return health_info
    
    def register_activation_hooks(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> None:
        """
        Registrar hooks para monitorear activaciones.
        
        Args:
            model: Modelo
            layer_names: Nombres de capas a monitorear (None = todas)
        """
        def hook_fn(name: str):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    stats = {
                        "mean": output.mean().item(),
                        "std": output.std().item(),
                        "min": output.min().item(),
                        "max": output.max().item(),
                        "has_nan": torch.isnan(output).any().item(),
                        "has_inf": torch.isinf(output).any().item(),
                    }
                    
                    if name not in self.activation_stats:
                        self.activation_stats[name] = []
                    
                    self.activation_stats[name].append(stats)
            
            return hook
        
        for name, module in model.named_modules():
            if layer_names is None or name in layer_names:
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
        
        logger.info(f"Registered activation hooks for {len(self.hooks)} layers")
    
    def remove_hooks(self) -> None:
        """Remover todos los hooks registrados"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activation_stats.clear()
        logger.info("All hooks removed")
    
    def get_activation_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas de activaciones.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.activation_stats:
            return {}
        
        summary = {}
        for name, stats_list in self.activation_stats.items():
            if stats_list:
                summary[name] = {
                    "mean": np.mean([s["mean"] for s in stats_list]),
                    "std": np.mean([s["std"] for s in stats_list]),
                    "min": np.min([s["min"] for s in stats_list]),
                    "max": np.max([s["max"] for s in stats_list]),
                    "has_nan": any(s["has_nan"] for s in stats_list),
                    "has_inf": any(s["has_inf"] for s in stats_list),
                    "num_samples": len(stats_list),
                }
        
        return summary
    
    def diagnose_training_issue(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Diagnosticar problemas en entrenamiento.
        
        Args:
            model: Modelo
            loss: Pérdida actual
            gradients: Gradientes (opcional, se calculan si None)
        
        Returns:
            Diagnóstico
        """
        diagnosis = {
            "loss_ok": True,
            "gradients_ok": True,
            "weights_ok": True,
            "issues": [],
            "recommendations": [],
        }
        
        # Check loss
        if torch.isnan(loss) or torch.isinf(loss):
            diagnosis["loss_ok"] = False
            diagnosis["issues"].append("Loss is NaN or Inf")
            diagnosis["recommendations"].append("Check learning rate, reduce if too high")
            diagnosis["recommendations"].append("Check data for NaN/Inf values")
        
        # Check gradients
        if gradients is None:
            gradients = {name: param.grad for name, param in model.named_parameters() if param.grad is not None}
        
        for name, grad in gradients.items():
            if grad is not None:
                if torch.isnan(grad).any():
                    diagnosis["gradients_ok"] = False
                    diagnosis["issues"].append(f"NaN gradient in {name}")
                    diagnosis["recommendations"].append("Use gradient clipping")
                    diagnosis["recommendations"].append("Reduce learning rate")
                
                if torch.isinf(grad).any():
                    diagnosis["gradients_ok"] = False
                    diagnosis["issues"].append(f"Inf gradient in {name}")
                    diagnosis["recommendations"].append("Use gradient clipping")
                
                grad_norm = grad.norm().item()
                if grad_norm > 100:
                    diagnosis["issues"].append(f"Large gradient norm in {name}: {grad_norm}")
                    diagnosis["recommendations"].append("Use gradient clipping")
                elif grad_norm < 1e-6:
                    diagnosis["issues"].append(f"Very small gradient norm in {name}: {grad_norm}")
                    diagnosis["recommendations"].append("Check if layer is frozen")
                    diagnosis["recommendations"].append("Check learning rate")
        
        # Check weights
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                diagnosis["weights_ok"] = False
                diagnosis["issues"].append(f"NaN weights in {name}")
                diagnosis["recommendations"].append("Reinitialize model")
            
            if torch.isinf(param).any():
                diagnosis["weights_ok"] = False
                diagnosis["issues"].append(f"Inf weights in {name}")
                diagnosis["recommendations"].append("Reinitialize model")
        
        return diagnosis




