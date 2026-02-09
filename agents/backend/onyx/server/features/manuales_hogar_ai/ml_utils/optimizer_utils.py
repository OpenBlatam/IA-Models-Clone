"""
Optimizer Utils - Optimizadores Avanzados
===========================================

Optimizadores personalizados y avanzados.
"""

import logging
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from typing import Optional, Dict, Any, Callable
import math

logger = logging.getLogger(__name__)


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper.
    
    Paper: https://arxiv.org/abs/1907.08610
    """
    
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        """
        Inicializar Lookahead.
        
        Args:
            optimizer: Optimizador base
            k: Frecuencia de actualización
            alpha: Tasa de interpolación
        """
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Guardar parámetros lentos
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                state['slow_buffer'] = torch.empty_like(p.data)
                state['slow_buffer'].copy_(p.data)
    
    def step(self, closure: Optional[Callable] = None):
        """
        Realizar paso de optimización.
        
        Args:
            closure: Closure opcional
        """
        loss = self.optimizer.step(closure)
        self.step_count += 1
        
        if self.step_count % self.k == 0:
            # Actualizar parámetros lentos
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.optimizer.state[p]
                    p.data.mul_(self.alpha).add_(state['slow_buffer'], alpha=1 - self.alpha)
                    state['slow_buffer'].copy_(p.data)
        
        return loss
    
    def state_dict(self):
        """Obtener state dict."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Cargar state dict."""
        self.optimizer.load_state_dict(state_dict)
    
    def zero_grad(self):
        """Resetear gradientes."""
        self.optimizer.zero_grad()


class RAdam(Optimizer):
    """
    Rectified Adam optimizer.
    
    Paper: https://arxiv.org/abs/1908.03265
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        """
        Inicializar RAdam.
        
        Args:
            params: Parámetros del modelo
            lr: Learning rate
            betas: Coeficientes de momentum
            eps: Término epsilon
            weight_decay: Weight decay
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        Realizar paso de optimización.
        
        Args:
            closure: Closure opcional
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')
                
                state = self.state[p]
                
                # Estado inicial
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Actualizar momentos
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Calcular términos de corrección
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Calcular rho
                rho_inf = 2 / (1 - beta2) - 1
                rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / (1 - beta2 ** state['step'])
                
                if rho_t > 4:
                    # Calcular varianza rectificada
                    r_t = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                    
                    # Actualizar parámetros
                    exp_avg_hat = exp_avg / bias_correction1
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                    
                    p.data.addcdiv_(
                        exp_avg_hat,
                        (exp_avg_sq_hat.sqrt() + group['eps']),
                        value=-group['lr'] * r_t
                    )
                else:
                    # Usar Adam sin rectificación
                    exp_avg_hat = exp_avg / bias_correction1
                    exp_avg_sq_hat = exp_avg_sq / bias_correction2
                    
                    p.data.addcdiv_(
                        exp_avg_hat,
                        (exp_avg_sq_hat.sqrt() + group['eps']),
                        value=-group['lr']
                    )
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        
        return loss


class AdaBound(Optimizer):
    """
    AdaBound optimizer.
    
    Paper: https://arxiv.org/abs/1902.09843
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        final_lr: float = 0.1,
        gamma: float = 1e-3,
        eps: float = 1e-8,
        weight_decay: float = 0
    ):
        """
        Inicializar AdaBound.
        
        Args:
            params: Parámetros del modelo
            lr: Learning rate inicial
            betas: Coeficientes de momentum
            final_lr: Learning rate final
            gamma: Tasa de convergencia
            eps: Término epsilon
            weight_decay: Weight decay
        """
        defaults = dict(
            lr=lr,
            betas=betas,
            final_lr=final_lr,
            gamma=gamma,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        """
        Realizar paso de optimización.
        
        Args:
            closure: Closure opcional
        """
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdaBound does not support sparse gradients')
                
                state = self.state[p]
                
                # Estado inicial
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Actualizar momentos
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Calcular learning rate acotado
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                final_lr = group['final_lr'] * group['lr'] / group['lr']
                
                # Calcular límites
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                
                # Aplicar límites
                step_size = torch.clamp(step_size, lower_bound, upper_bound)
                
                # Actualizar parámetros
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2
                
                p.data.addcdiv_(
                    exp_avg_hat,
                    (exp_avg_sq_hat.sqrt() + group['eps']),
                    value=-step_size
                )
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'] * group['lr'])
        
        return loss


def create_optimizer(
    model: torch.nn.Module,
    optimizer_name: str = "adam",
    lr: float = 1e-3,
    **kwargs
) -> Optimizer:
    """
    Crear optimizador.
    
    Args:
        model: Modelo PyTorch
        optimizer_name: Nombre del optimizador
        lr: Learning rate
        **kwargs: Argumentos adicionales
        
    Returns:
        Optimizador
    """
    params = model.parameters()
    
    if optimizer_name.lower() == "adam":
        return optim.Adam(params, lr=lr, **kwargs)
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(params, lr=lr, **kwargs)
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(params, lr=lr, **kwargs)
    elif optimizer_name.lower() == "radam":
        return RAdam(params, lr=lr, **kwargs)
    elif optimizer_name.lower() == "adabound":
        return AdaBound(params, lr=lr, **kwargs)
    elif optimizer_name.lower() == "rmsprop":
        return optim.RMSprop(params, lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")




