"""
Code Templates - Plantillas de código para Deep Learning (optimizado)
======================================================================

Plantillas de código siguiendo mejores prácticas de PyTorch, Transformers,
Diffusers, y Deep Learning en general.
"""

from typing import Dict, Any, Optional
from enum import Enum


class TemplateType(str, Enum):
    """Tipos de plantillas disponibles"""
    TRANSFORMER_MODEL = "transformer_model"
    DIFFUSION_PIPELINE = "diffusion_pipeline"
    TRAINING_LOOP = "training_loop"
    DATALOADER = "dataloader"
    GRADIO_INTERFACE = "gradio_interface"
    CONFIG_YAML = "config_yaml"


def get_transformer_model_template(framework: str = "pytorch") -> str:
    """
    Obtener plantilla de modelo Transformer (optimizado).
    
    Args:
        framework: Framework a usar (pytorch, tensorflow, jax)
        
    Returns:
        Código de plantilla
    """
    if framework == "pytorch":
        return '''"""
Transformer Model Architecture (optimizado)
===========================================

Arquitectura de modelo Transformer siguiendo mejores prácticas de PyTorch.
Incluye atención multi-head, normalización de capas, y dropout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from transformers import PreTrainedModel, PretrainedConfig


class TransformerConfig(PretrainedConfig):
    """Configuración para modelo Transformer"""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        n_positions: int = 1024,
        n_embd: int = 768,
        n_layer: int = 12,
        n_head: int = 12,
        n_inner: Optional[int] = None,
        activation_function: str = "gelu",
        resid_pdrop: float = 0.1,
        embd_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        layer_norm_epsilon: float = 1e-5,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner if n_inner is not None else 4 * n_embd
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention mechanism (optimizado)"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = nn.Dropout(config.attn_pdrop)
        
        # Q, K, V projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        
        self._init_weights(config)
    
    def _init_weights(self, config: TransformerConfig):
        """Inicializar pesos siguiendo mejores prácticas"""
        nn.init.normal_(self.c_attn.weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.c_attn.bias)
        nn.init.normal_(self.c_proj.weight, mean=0.0, std=config.initializer_range)
        nn.init.zeros_(self.c_proj.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (optimizado).
        
        Args:
            x: Input tensor (batch_size, seq_len, n_embd)
            attention_mask: Attention mask (batch_size, seq_len, seq_len)
            
        Returns:
            Output tensor y attention weights
        """
        batch_size, seq_len, _ = x.size()
        
        # Q, K, V projections
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # Reshape para multi-head
        q = q.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.n_embd // self.n_head).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_output, attn_weights = self._scaled_dot_product_attention(
            q, k, v, attention_mask
        )
        
        # Concatenar heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.n_embd
        )
        
        # Output projection
        output = self.c_proj(attn_output)
        output = self.dropout(output)
        
        return output, attn_weights
    
    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention"""
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Transformer block con residual connections (optimizado)"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, config.n_inner),
            self._get_activation(config.activation_function),
            nn.Dropout(config.resid_pdrop),
            nn.Linear(config.n_inner, config.n_embd),
            nn.Dropout(config.resid_pdrop)
        )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Obtener función de activación"""
        if activation == "gelu":
            return nn.GELU()
        elif activation == "relu":
            return nn.ReLU()
        elif activation == "swish":
            return nn.SiLU()
        else:
            return nn.GELU()
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass con residual connections"""
        # Self-attention
        attn_output, _ = self.attn(self.ln_1(x), attention_mask)
        x = x + attn_output
        
        # Feed-forward
        x = x + self.mlp(self.ln_2(x))
        
        return x


class TransformerModel(PreTrainedModel):
    """
    Modelo Transformer completo (optimizado).
    
    Implementa arquitectura Transformer estándar con mejores prácticas.
    """
    
    config_class = TransformerConfig
    
    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Layer norm final
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Inicializar pesos
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Inicializar pesos del módulo"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass (optimizado).
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            position_ids: Position IDs (batch_size, seq_len)
            
        Returns:
            Hidden states (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len = input_ids.size()
        
        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        
        x = self.drop(token_embeddings + position_embeddings)
        
        # Transformer blocks
        for block in self.h:
            x = block(x, attention_mask)
        
        # Layer norm final
        x = self.ln_f(x)
        
        return x
'''
    else:
        # Placeholder para otros frameworks
        return "# Template for other frameworks not yet implemented"


def get_training_loop_template(
    mixed_precision: bool = True,
    gradient_accumulation: bool = True,
    multi_gpu: bool = False
) -> str:
    """
    Obtener plantilla de training loop (optimizado).
    
    Args:
        mixed_precision: Si usar mixed precision training
        gradient_accumulation: Si usar gradient accumulation
        multi_gpu: Si usar multi-GPU training
        
    Returns:
        Código de plantilla
    """
    return f'''"""
Training Loop (optimizado)
===========================

Loop de entrenamiento siguiendo mejores prácticas de PyTorch.
Incluye mixed precision, gradient accumulation, y manejo de errores.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TrainingLoop:
    """
    Loop de entrenamiento optimizado (optimizado).
    
    Características:
    - Mixed precision training (FP16/BF16)
    - Gradient accumulation
    - Multi-GPU support
    - Early stopping
    - Learning rate scheduling
    - Gradient clipping
    - NaN/Inf detection
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        mixed_precision: bool = {str(mixed_precision).lower()},
        gradient_accumulation_steps: int = {4 if gradient_accumulation else 1},
        max_grad_norm: float = 1.0,
        multi_gpu: bool = {str(multi_gpu).lower()}
    ):
        """
        Inicializar training loop (optimizado).
        
        Args:
            model: Modelo PyTorch
            optimizer: Optimizador
            criterion: Función de pérdida
            device: Dispositivo (CPU/GPU)
            mixed_precision: Si usar mixed precision
            gradient_accumulation_steps: Pasos de acumulación de gradientes
            max_grad_norm: Norma máxima de gradientes
            multi_gpu: Si usar multi-GPU
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Mixed precision scaler
        self.scaler = GradScaler() if mixed_precision else None
        
        # Multi-GPU setup
        if multi_gpu and torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            logger.info(f"Using {{torch.cuda.device_count()}} GPUs")
        
        self.model.to(device)
        self.model.train()
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        global_step: int = 0
    ) -> Dict[str, float]:
        """
        Un paso de entrenamiento (optimizado).
        
        Args:
            batch: Batch de datos
            global_step: Paso global actual
            
        Returns:
            Diccionario con métricas
        """
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        
        # Forward pass con mixed precision
        if self.mixed_precision and self.scaler:
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss = loss / self.gradient_accumulation_steps
        else:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss = loss / self.gradient_accumulation_steps
        
        # Detectar NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"NaN/Inf detected in loss at step {{global_step}}")
            raise RuntimeError(f"Invalid loss value: {{loss.item()}}")
        
        # Backward pass
        if self.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (global_step + 1) % self.gradient_accumulation_steps == 0:
            if self.mixed_precision and self.scaler:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        return {{
            "loss": loss.item() * self.gradient_accumulation_steps,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }}
    
    def validate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Validar modelo (optimizado).
        
        Args:
            val_loader: DataLoader de validación
            
        Returns:
            Diccionario con métricas de validación
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)
                
                if self.mixed_precision:
                    with autocast():
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        self.model.train()
        
        return {{"val_loss": avg_loss}}
'''


def get_dataloader_template(
    pin_memory: bool = True,
    num_workers: int = 4,
    prefetch_factor: int = 2
) -> str:
    """
    Obtener plantilla de DataLoader (optimizado).
    
    Args:
        pin_memory: Si usar pin_memory
        num_workers: Número de workers
        prefetch_factor: Factor de prefetch
        
    Returns:
        Código de plantilla
    """
    return f'''"""
DataLoader Configuration (optimizado)
=====================================

Configuración de DataLoader siguiendo mejores prácticas de PyTorch.
Optimizado para rendimiento con pin_memory, prefetching, y workers.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = {num_workers},
    pin_memory: bool = {str(pin_memory).lower()},
    prefetch_factor: int = {prefetch_factor},
    persistent_workers: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Crear DataLoader optimizado (optimizado).
    
    Args:
        dataset: Dataset de PyTorch
        batch_size: Tamaño de batch
        shuffle: Si mezclar datos
        num_workers: Número de workers
        pin_memory: Si usar pin_memory (mejor para GPU)
        prefetch_factor: Factor de prefetch
        persistent_workers: Si mantener workers persistentes
        drop_last: Si descartar último batch incompleto
        
    Returns:
        DataLoader optimizado
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=drop_last
    )
'''


def get_gradio_interface_template() -> str:
    """
    Obtener plantilla de interfaz Gradio (optimizado).
    
    Returns:
        Código de plantilla
    """
    return '''"""
Gradio Interface (optimizado)
=============================

Interfaz Gradio para inferencia y visualización de modelos.
Incluye validación de inputs, manejo de errores, y mejoras de UX.
"""

import gradio as gr
import torch
from typing import Optional, Tuple, Any
import logging
import traceback

logger = logging.getLogger(__name__)


class GradioInterface:
    """
    Interfaz Gradio optimizada (optimizado).
    
    Características:
    - Validación de inputs
    - Manejo de errores robusto
    - Mejoras de UX
    - Soporte para diferentes tipos de modelos
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Optional[Any] = None,
        device: torch.device = None
    ):
        """
        Inicializar interfaz Gradio (optimizado).
        
        Args:
            model: Modelo PyTorch
            tokenizer: Tokenizer (opcional, para modelos de texto)
            device: Dispositivo (CPU/GPU)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
    
    def predict(
        self,
        input_text: str,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> Tuple[str, Optional[str]]:
        """
        Realizar predicción (optimizado).
        
        Args:
            input_text: Texto de entrada
            max_length: Longitud máxima
            temperature: Temperatura para sampling
            
        Returns:
            Tupla con resultado y error (si hay)
        """
        try:
            # Validar input
            if not input_text or not input_text.strip():
                return "", "Input text cannot be empty"
            
            if len(input_text) > 10000:
                return "", "Input text too long (max 10000 characters)"
            
            # Tokenizar si hay tokenizer
            if self.tokenizer:
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
            else:
                # Procesar input directamente
                inputs = {"input": input_text}
            
            # Inferencia
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Procesar output
            if self.tokenizer:
                result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                result = str(outputs)
            
            return result, None
            
        except Exception as e:
            error_msg = f"Error during prediction: {{str(e)}}"
            logger.error(error_msg, exc_info=True)
            return "", error_msg
    
    def create_interface(self) -> gr.Blocks:
        """
        Crear interfaz Gradio (optimizado).
        
        Returns:
            Interfaz Gradio
        """
        with gr.Blocks(title="Deep Learning Model Interface") as interface:
            gr.Markdown("# Deep Learning Model Interface")
            gr.Markdown("Interfaz para inferencia de modelos de Deep Learning")
            
            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter your text here...",
                        lines=5
                    )
                    max_length = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Max Length"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature"
                    )
                    submit_btn = gr.Button("Generate", variant="primary")
                
                with gr.Column():
                    output_text = gr.Textbox(
                        label="Output",
                        lines=10,
                        interactive=False
                    )
                    error_text = gr.Textbox(
                        label="Error",
                        lines=3,
                        interactive=False,
                        visible=False
                    )
            
            submit_btn.click(
                fn=self.predict,
                inputs=[input_text, max_length, temperature],
                outputs=[output_text, error_text]
            )
            
            gr.Examples(
                examples=[
                    ["Example input 1"],
                    ["Example input 2"],
                ],
                inputs=input_text
            )
        
        return interface
    
    def launch(self, share: bool = False, server_name: str = "0.0.0.0", server_port: int = 7860):
        """
        Lanzar interfaz Gradio (optimizado).
        
        Args:
            share: Si crear link público
            server_name: Nombre del servidor
            server_port: Puerto del servidor
        """
        interface = self.create_interface()
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port
        )


if __name__ == "__main__":
    # Ejemplo de uso
    # model = YourModel()
    # tokenizer = YourTokenizer()
    # interface = GradioInterface(model, tokenizer)
    # interface.launch()
    pass
'''


def get_config_yaml_template(features: Dict[str, Any]) -> str:
    """
    Obtener plantilla de configuración YAML (optimizado).
    
    Args:
        features: Características detectadas
        
    Returns:
        Código de plantilla YAML
    """
    return f'''# Training Configuration (optimizado)
# ===========================================
# Configuración siguiendo mejores prácticas de Deep Learning

# Model Configuration
model:
  type: "{features.get('model_type', 'base')}"
  framework: "{features.get('framework', 'pytorch')}"
  fine_tuning_technique: "{features.get('fine_tuning_technique', 'full_finetuning')}"
  
  # Transformer specific
  vocab_size: 50257
  n_positions: 1024
  n_embd: 768
  n_layer: 12
  n_head: 12
  activation_function: "gelu"
  
  # Diffusion specific
  unet_dim: 320
  num_timesteps: 1000
  scheduler: "ddpm"

# Training Configuration
training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 1e-4
  weight_decay: 0.01
  
  # Optimizations
  mixed_precision: {str(features.get('requires_mixed_precision', True)).lower()}
  gradient_accumulation_steps: {4 if features.get('requires_gradient_accumulation', True) else 1}
  max_grad_norm: 1.0
  
  # Multi-GPU
  multi_gpu: {str(features.get('requires_multi_gpu', False)).lower()}
  
  # Early Stopping
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 1e-6
  
  # Learning Rate Scheduling
  lr_scheduler:
    type: "cosine"
    warmup_steps: 1000
    T_max: 100

# Data Configuration
data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  
  # DataLoader
  num_workers: 4
  pin_memory: true
  prefetch_factor: 2
  persistent_workers: true

# Experiment Tracking
experiment_tracking:
  enabled: {str(features.get('requires_experiment_tracking', False)).lower()}
  tool: "wandb"  # wandb, tensorboard, mlflow
  project_name: "deep_learning_project"
  run_name: "experiment_1"

# Logging
logging:
  level: "INFO"
  log_dir: "./logs"
  log_file: "training.log"
'''


def _validate_template_type(template_type: TemplateType) -> None:
    """
    Valida el tipo de plantilla (función pura).
    
    Args:
        template_type: Tipo de plantilla
        
    Raises:
        ValueError: Si el tipo de plantilla es inválido
        TypeError: Si no es un TemplateType
    """
    if not isinstance(template_type, TemplateType):
        raise TypeError("template_type must be a TemplateType enum")


def get_template(template_type: TemplateType, **kwargs) -> str:
    """
    Obtener plantilla por tipo.
    
    Args:
        template_type: Tipo de plantilla
        **kwargs: Argumentos adicionales
        
    Returns:
        Código de plantilla
        
    Raises:
        ValueError: Si el tipo de plantilla es inválido
        TypeError: Si template_type no es un TemplateType
    """
    _validate_template_type(template_type)
    
    if template_type == TemplateType.TRANSFORMER_MODEL:
        framework = kwargs.get("framework", "pytorch")
        return get_transformer_model_template(framework)
    elif template_type == TemplateType.TRAINING_LOOP:
        return get_training_loop_template(
            mixed_precision=kwargs.get("mixed_precision", True),
            gradient_accumulation=kwargs.get("gradient_accumulation", True),
            multi_gpu=kwargs.get("multi_gpu", False)
        )
    elif template_type == TemplateType.DATALOADER:
        return get_dataloader_template(
            pin_memory=kwargs.get("pin_memory", True),
            num_workers=kwargs.get("num_workers", 4),
            prefetch_factor=kwargs.get("prefetch_factor", 2)
        )
    elif template_type == TemplateType.GRADIO_INTERFACE:
        return get_gradio_interface_template()
    elif template_type == TemplateType.CONFIG_YAML:
        return get_config_yaml_template(kwargs.get("features", {}))
    else:
        raise ValueError(f"Unknown template type: {template_type}")

