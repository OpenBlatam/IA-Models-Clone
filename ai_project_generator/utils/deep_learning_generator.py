"""
Deep Learning Project Generator
================================

Sistema de generación de proyectos de deep learning con PyTorch, Transformers y Diffusers.
Refactorizado con estructura modular.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .deep_learning.templates import TransformerTemplate, DiffusionTemplate
from .deep_learning.project_structure import ProjectStructure
from .deep_learning.evaluation import generate_evaluation_code
from .deep_learning.utils import generate_utils_code
from .deep_learning.data_processing import generate_data_processing_code
from .deep_learning.config_loader import generate_config_loader_code
from .deep_learning.gradio_enhanced import generate_enhanced_gradio_code
from .deep_learning.notebook_template import generate_notebook_template
from .deep_learning.experiment_tracking import generate_experiment_tracking_code
from .deep_learning.performance import generate_performance_code
from .deep_learning.visualization import generate_visualization_code
from .deep_learning.testing import generate_testing_code

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Tipos de modelos de deep learning."""
    TRANSFORMER = "transformer"
    DIFFUSION = "diffusion"
    LLM = "llm"
    VISION_TRANSFORMER = "vision_transformer"
    CNN = "cnn"
    LSTM = "lstm"
    GAN = "gan"


class TrainingConfig(Enum):
    """Configuraciones de entrenamiento."""
    STANDARD = "standard"
    MIXED_PRECISION = "mixed_precision"
    DISTRIBUTED = "distributed"
    GRADIENT_ACCUMULATION = "gradient_accumulation"


@dataclass
class ModelArchitecture:
    """Arquitectura de modelo."""
    model_type: ModelType
    config: Dict[str, Any]
    layers: List[Dict[str, Any]]
    activation: str = "gelu"
    normalization: str = "layer_norm"
    dropout: float = 0.1


@dataclass
class TrainingSetup:
    """Configuración de entrenamiento."""
    optimizer: str
    learning_rate: float
    scheduler: Optional[str]
    loss_function: str
    batch_size: int
    num_epochs: int
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    distributed: bool = False


class DeepLearningGenerator:
    """Generador de proyectos de deep learning."""
    
    def __init__(self):
        self.model_templates = self._initialize_templates()
        self.training_templates = self._initialize_training_templates()
    
    def generate_transformer_model(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ) -> str:
        """Genera código para modelo Transformer usando template."""
        config = {
            'vocab_size': vocab_size,
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dim_feedforward': dim_feedforward,
            'max_seq_length': max_seq_length,
            'dropout': dropout
        }
        return TransformerTemplate.generate_model_code(config)
    
    def _generate_transformer_model_old(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        nhead: int = 12,
        num_layers: int = 12,
        dim_feedforward: int = 3072,
        max_seq_length: int = 512,
        dropout: float = 0.1
    ) -> str:
        """Genera código para modelo Transformer (método legacy)."""
        return f'''import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


class PositionalEncoding(nn.Module):
    """Codificación posicional para Transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """Modelo Transformer completo."""
    
    def __init__(
        self,
        vocab_size: int = {vocab_size},
        d_model: int = {d_model},
        nhead: int = {nhead},
        num_layers: int = {num_layers},
        dim_feedforward: int = {dim_feedforward},
        max_seq_length: int = {max_seq_length},
        dropout: float = {dropout}
    ):
        super().__init__()
        self.d_model = d_model
        
        # Embedding de tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Codificación posicional
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Capas de encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # Capa de salida
        self.decoder = nn.Linear(d_model, vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa pesos del modelo."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: Tensor de entrada, shape [seq_len, batch_size]
            src_mask: Máscara de atención opcional
        Returns:
            Tensor de salida, shape [seq_len, batch_size, vocab_size]
        """
        # Embedding + posicional
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer encoder
        output = self.transformer_encoder(src, src_mask)
        
        # Decoder
        output = self.decoder(output)
        
        return output
    
    def generate_mask(self, sz: int) -> torch.Tensor:
        """Genera máscara causal para autoregresión."""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
'''
    
    def generate_diffusion_model(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.0
    ) -> str:
        """Genera código para modelo de difusión usando template."""
        config = {
            'in_channels': in_channels,
            'out_channels': out_channels,
            'model_channels': model_channels,
            'num_res_blocks': num_res_blocks,
            'attention_resolutions': attention_resolutions,
            'dropout': dropout
        }
        return DiffusionTemplate.generate_model_code(config)
    
    def _generate_diffusion_model_old(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 128,
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.0
    ) -> str:
        """Genera código para modelo de difusión (método legacy)."""
        return f'''import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResBlock(nn.Module):
    """Bloque residual para modelo de difusión."""
    
    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return x + h


class AttentionBlock(nn.Module):
    """Bloque de atención para modelo de difusión."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.norm(x)
        x = x.view(B, C, H * W)
        
        q, k, v = self.qkv(x).chunk(3, dim=1)
        scale = (C // 3) ** -0.5
        
        attn = torch.softmax(q.transpose(-2, -1) @ k * scale, dim=-1)
        h = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        
        return x.view(B, C, H, W) + self.proj(h)


class DiffusionUNet(nn.Module):
    """UNet para modelo de difusión."""
    
    def __init__(
        self,
        in_channels: int = {in_channels},
        out_channels: int = {out_channels},
        model_channels: int = {model_channels},
        num_res_blocks: int = {num_res_blocks},
        attention_resolutions: List[int] = {attention_resolutions},
        dropout: float = {dropout}
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Embedding de tiempo
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels * 4),
            nn.SiLU(),
            nn.Linear(model_channels * 4, model_channels)
        )
        
        # Entrada
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Downsampling
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        for i, res in enumerate([32, 16, 8, 4]):
            self.down_blocks.append(self._make_down_block(ch, ch * 2, num_res_blocks, dropout, res in attention_resolutions))
            ch *= 2
        
        # Middle
        self.middle_block = ResBlock(ch, dropout)
        self.middle_attn = AttentionBlock(ch)
        
        # Upsampling
        self.up_blocks = nn.ModuleList()
        for i, res in enumerate([4, 8, 16, 32]):
            self.up_blocks.append(self._make_up_block(ch, ch // 2, num_res_blocks, dropout, res in attention_resolutions))
            ch //= 2
        
        # Salida
        self.output_norm = nn.GroupNorm(32, model_channels)
        self.output_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
    
    def _make_down_block(self, in_ch: int, out_ch: int, num_res: int, dropout: float, use_attn: bool):
        blocks = [nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)]
        for _ in range(num_res):
            blocks.append(ResBlock(out_ch, dropout))
        if use_attn:
            blocks.append(AttentionBlock(out_ch))
        return nn.Sequential(*blocks)
    
    def _make_up_block(self, in_ch: int, out_ch: int, num_res: int, dropout: float, use_attn: bool):
        blocks = []
        for _ in range(num_res):
            blocks.append(ResBlock(in_ch, dropout))
        if use_attn:
            blocks.append(AttentionBlock(in_ch))
        blocks.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
        return nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de entrada [B, C, H, W]
            timestep: Tensor de timesteps [B]
        """
        # Time embedding
        t_emb = self.time_embed(timestep.float()[:, None])
        
        # Downsampling
        h = self.input_conv(x)
        skip_connections = []
        for block in self.down_blocks:
            h = block(h)
            skip_connections.append(h)
        
        # Middle
        h = self.middle_block(h)
        h = self.middle_attn(h)
        
        # Upsampling
        for block in self.up_blocks:
            h = torch.cat([h, skip_connections.pop()], dim=1)
            h = block(h)
        
        # Output
        h = self.output_norm(h)
        h = F.silu(h)
        return self.output_conv(h)
'''
    
    def generate_training_script(
        self,
        model_type: ModelType,
        training_config: TrainingConfig,
        use_mixed_precision: bool = True,
        use_distributed: bool = False,
        gradient_accumulation_steps: int = 1,
        use_lora: bool = False,
        use_8bit: bool = False
    ) -> str:
        """Genera script de entrenamiento con mejores prácticas."""
        
        # Configuración de Accelerate
        accelerate_setup = ""
        if use_distributed or use_mixed_precision:
            accelerate_setup = '''
from accelerate import Accelerator
from accelerate.utils import set_seed

# Initialize accelerator
accelerator = Accelerator(
    mixed_precision="fp16" if use_mixed_precision else "no",
    gradient_accumulation_steps=gradient_accumulation_steps,
    log_with="wandb" if use_wandb else None,
    project_dir="./logs"
)
'''
        
        # Configuración de LoRA/PEFT
        lora_setup = ""
        if use_lora:
            lora_setup = '''
from peft import LoraConfig, get_peft_model, TaskType

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
'''
        
        # Configuración de 8-bit
        bitsandbytes_setup = ""
        if use_8bit:
            bitsandbytes_setup = '''
from transformers import BitsAndBytesConfig
import torch

# 8-bit quantization config
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)
'''
        
        # Código de entrenamiento con Accelerate
        training_loop = '''
        # Training loop with Accelerate
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs.logits, batch['labels'])
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            
            # Logging
            if accelerator.is_main_process and (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                
                if use_wandb:
                    accelerator.log({"train_loss": avg_loss, "learning_rate": scheduler.get_last_lr()[0]})
        '''
        
        # Código sin Accelerate (fallback)
        training_loop_simple = '''
        # Training loop (simple)
        model.train()
        total_loss = 0.0
        
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            if use_mixed_precision and scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs.logits, batch['labels'])
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
            else:
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs.logits, batch['labels'])
                loss = loss / gradient_accumulation_steps
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if use_mixed_precision and scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": f"{total_loss / (batch_idx + 1):.4f}"})
        '''
        
        use_accelerate = use_distributed or use_mixed_precision
        training_code = training_loop if use_accelerate else training_loop_simple
        
        return f'''"""
Training script with best practices for Deep Learning.
Uses Accelerate, Mixed Precision, LoRA, and proper experiment tracking.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
import os
import logging
from pathlib import Path
import json
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# WandB (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available, skipping experiment tracking")

{accelerate_setup if use_accelerate else ""}
{lora_setup if use_lora else ""}
{bitsandbytes_setup if use_8bit else ""}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 1000,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = {gradient_accumulation_steps},
    use_mixed_precision: bool = {str(use_mixed_precision).lower()},
    use_distributed: bool = {str(use_distributed).lower()},
    use_wandb: bool = WANDB_AVAILABLE,
    log_interval: int = 100,
    save_dir: str = "./checkpoints",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Train model with best practices.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for optimizer
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        gradient_accumulation_steps: Steps for gradient accumulation
        use_mixed_precision: Use mixed precision training
        use_distributed: Use distributed training
        use_wandb: Use Weights & Biases for tracking
        log_interval: Logging interval
        save_dir: Directory to save checkpoints
        device: Device to use
    """
    set_seed(42)  # For reproducibility
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize accelerator if using distributed/mixed precision
    if use_accelerate:
        accelerator = Accelerator(
            mixed_precision="fp16" if use_mixed_precision else "no",
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="wandb" if use_wandb else None,
            project_dir=str(save_path / "logs")
        )
        
        # Prepare model, optimizer, scheduler, and data loaders
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
        
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
        
        if use_wandb and accelerator.is_main_process:
            accelerator.init_trackers(
                project_name="deep-learning-project",
                config={{
                    "learning_rate": learning_rate,
                    "num_epochs": num_epochs,
                    "batch_size": train_loader.batch_size,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                }}
            )
    else:
        # Simple setup without Accelerate
        model = model.to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    logger.info(f"Starting training for {{num_epochs}} epochs")
    logger.info(f"Device: {{device}}, Mixed Precision: {{use_mixed_precision}}")
    
    for epoch in range(num_epochs):
        logger.info(f"\\nEpoch {{epoch+1}}/{{num_epochs}}")
        
        {training_code}
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=use_accelerate and not accelerator.is_local_main_process):
                if use_accelerate:
                    outputs = model(**batch)
                else:
                    batch = {{k: v.to(device) for k, v in batch.items()}}
                    outputs = model(**batch)
                
                loss = outputs.loss if hasattr(outputs, 'loss') else criterion(outputs.logits, batch['labels'])
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        if use_accelerate:
            avg_val_loss = accelerator.gather(torch.tensor(avg_val_loss)).mean().item()
        
        logger.info(f"Validation Loss: {{avg_val_loss:.4f}}")
        
        if use_wandb and (not use_accelerate or accelerator.is_main_process):
            if use_accelerate:
                accelerator.log({{"val_loss": avg_val_loss, "epoch": epoch+1}})
            else:
                wandb.log({{"val_loss": avg_val_loss, "epoch": epoch+1}})
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
            checkpoint = {{
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict() if not use_accelerate else accelerator.get_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'config': {{
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                }}
            }}
            
            checkpoint_path = save_path / f"checkpoint_epoch_{{epoch+1}}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {{checkpoint_path}}")
            
            # Save best model
            best_path = save_path / "best_model.pt"
            torch.save(checkpoint, best_path)
    
    logger.info(f"\\nTraining completed! Best validation loss: {{best_val_loss:.4f}} at epoch {{best_epoch}}")
    
    if use_accelerate:
        accelerator.end_training()
    
    return model


if __name__ == "__main__":
    # Example usage
    from model import TransformerModel  # Import your model
    
    # Initialize model
    model = TransformerModel(
        vocab_size=50257,
        d_model=768,
        nhead=12,
        num_layers=12
    )
    
    # Create dummy data loaders (replace with your actual data)
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(
        torch.randint(0, 50257, (1000, 512)),
        torch.randint(0, 50257, (1000, 512))
    )
    val_dataset = TensorDataset(
        torch.randint(0, 50257, (100, 512)),
        torch.randint(0, 50257, (100, 512))
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Train
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        learning_rate=1e-4,
        use_mixed_precision=True,
        use_wandb=True
    )
'''
    
    def generate_gradio_interface(
        self,
        model_type: ModelType,
        task: str = "text_generation"
    ) -> str:
        """Genera interfaz Gradio."""
        if task == "text_generation":
            return '''import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_path: str):
    """Carga modelo y tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def generate_text(prompt: str, max_length: int = 100, temperature: float = 0.7):
    """Genera texto a partir de un prompt."""
    if not prompt:
        return "Por favor, ingresa un prompt."
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


# Cargar modelo
model, tokenizer = load_model("path/to/model")

# Crear interfaz Gradio
interface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Escribe tu prompt aquí..."),
        gr.Slider(minimum=50, maximum=500, value=100, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="Text Generation Demo",
    description="Genera texto usando un modelo de lenguaje"
)

if __name__ == "__main__":
    interface.launch()
'''
        elif task == "image_generation":
            return '''import gradio as gr
import torch
from diffusers import StableDiffusionPipeline


def load_diffusion_model(model_id: str = "runwayml/stable-diffusion-v1-5"):
    """Carga pipeline de difusión."""
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe


def generate_image(prompt: str, num_inference_steps: int = 50, guidance_scale: float = 7.5):
    """Genera imagen a partir de un prompt."""
    if not prompt:
        return None
    
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    ).images[0]
    
    return image


# Cargar modelo
pipe = load_diffusion_model()

# Crear interfaz Gradio
interface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Describe la imagen que quieres generar..."),
        gr.Slider(minimum=20, maximum=100, value=50, label="Inference Steps"),
        gr.Slider(minimum=1.0, maximum=20.0, value=7.5, step=0.5, label="Guidance Scale")
    ],
    outputs=gr.Image(label="Generated Image"),
    title="Image Generation Demo",
    description="Genera imágenes usando Stable Diffusion"
)

if __name__ == "__main__":
    interface.launch()
'''
    
    def generate_project_structure(self, project_type: ModelType) -> Dict[str, str]:
        """Genera estructura completa de proyecto usando ProjectStructure."""
        base_structure = ProjectStructure.generate_structure(project_type.value)
        
        # Agregar modelos específicos
        if project_type == ModelType.TRANSFORMER:
            base_structure["src/models/transformer.py"] = self.generate_transformer_model()
            base_structure["src/training/train.py"] = self.generate_training_script(
                ModelType.TRANSFORMER, TrainingConfig.STANDARD
            )
            base_structure["src/evaluation/evaluate.py"] = generate_evaluation_code()
            base_structure["src/utils/utils.py"] = generate_utils_code()
            base_structure["src/data/dataset.py"] = generate_data_processing_code()
            base_structure["src/utils/config_loader.py"] = generate_config_loader_code()
            base_structure["src/utils/experiment_tracking.py"] = generate_experiment_tracking_code()
            base_structure["src/utils/performance.py"] = generate_performance_code()
            base_structure["src/utils/visualization.py"] = generate_visualization_code()
            base_structure["tests/test_models.py"] = generate_testing_code()
            base_structure["src/demo.py"] = generate_enhanced_gradio_code("text_generation")
            base_structure["notebooks/exploratory_analysis.ipynb"] = generate_notebook_template()
        elif project_type == ModelType.DIFFUSION:
            base_structure["src/models/diffusion.py"] = self.generate_diffusion_model()
            base_structure["src/training/train.py"] = self.generate_training_script(
                ModelType.DIFFUSION, TrainingConfig.MIXED_PRECISION
            )
            base_structure["src/evaluation/evaluate.py"] = generate_evaluation_code()
            base_structure["src/utils/utils.py"] = generate_utils_code()
            base_structure["src/data/dataset.py"] = generate_data_processing_code()
            base_structure["src/utils/config_loader.py"] = generate_config_loader_code()
            base_structure["src/utils/experiment_tracking.py"] = generate_experiment_tracking_code()
            base_structure["src/utils/performance.py"] = generate_performance_code()
            base_structure["src/utils/visualization.py"] = generate_visualization_code()
            base_structure["tests/test_models.py"] = generate_testing_code()
            base_structure["src/demo.py"] = generate_enhanced_gradio_code("image_generation")
            base_structure["notebooks/exploratory_analysis.ipynb"] = generate_notebook_template()
        
        return base_structure
    
    def _generate_requirements(self) -> str:
        """Genera requirements.txt usando ProjectStructure."""
        return ProjectStructure._generate_requirements()
    
    def _generate_requirements_old(self) -> str:
        """Genera requirements.txt con las mejores librerías (método legacy)."""
        return '''# Core Deep Learning
torch>=2.1.0
torchvision>=0.16.0
torchaudio>=2.1.0

# Transformers & LLMs
transformers>=4.35.0
tokenizers>=0.15.0
sentencepiece>=0.1.99

# Diffusion Models
diffusers>=0.24.0
xformers>=0.0.23  # Optimized attention
safetensors>=0.4.0

# Efficient Training & Fine-tuning
accelerate>=0.25.0  # Multi-GPU, mixed precision
peft>=0.7.0  # LoRA, P-tuning, etc.
bitsandbytes>=0.41.0  # Quantization
trl>=0.7.0  # Reinforcement Learning from Human Feedback

# Data & Datasets
datasets>=2.14.0
huggingface-hub>=0.19.0
pillow>=10.1.0
opencv-python>=4.8.0

# Visualization & UI
gradio>=4.7.0
streamlit>=1.28.0
matplotlib>=3.8.0
seaborn>=0.13.0
plotly>=5.18.0

# Experiment Tracking
wandb>=0.16.0
tensorboard>=2.15.0
mlflow>=2.8.0

# Utilities
numpy>=1.26.0
pandas>=2.1.0
scipy>=1.11.0
tqdm>=4.66.0
pyyaml>=6.0.1
omegaconf>=2.3.0  # Better config management

# Performance & Optimization
ninja>=1.11.0  # Faster compilation
flash-attn>=2.3.0  # Flash attention (optional, requires CUDA)
optimum>=1.14.0  # Model optimization

# Monitoring & Profiling
psutil>=5.9.0
gpustat>=1.1.0
py3nvml>=0.2.7

# Testing & Quality
pytest>=7.4.0
black>=23.11.0
flake8>=6.1.0
mypy>=1.7.0

# Jupyter & Development
jupyter>=1.0.0
ipywidgets>=8.1.0
rich>=13.7.0  # Beautiful terminal output
'''
    
    def _generate_config_yaml(self, project_type: ModelType) -> str:
        """Genera config.yaml usando ProjectStructure."""
        return ProjectStructure._generate_config_yaml(project_type.value)
    
    def _generate_config_yaml_old(self, project_type: ModelType) -> str:
        """Genera config.yaml (método legacy)."""
        return f'''# Configuración del proyecto
project_type: {project_type.value}

# Modelo
model:
  type: {project_type.value}
  vocab_size: 50257
  d_model: 768
  nhead: 12
  num_layers: 12

# Entrenamiento
training:
  batch_size: 32
  learning_rate: 1e-4
  num_epochs: 10
  gradient_accumulation_steps: 1
  mixed_precision: true
  distributed: false

# Datos
data:
  train_path: "data/train"
  val_path: "data/val"
  test_path: "data/test"
'''
    
    def _generate_readme(self, project_type: ModelType) -> str:
        """Genera README.md usando ProjectStructure."""
        return ProjectStructure._generate_readme(project_type.value)
    
    def _generate_readme_old(self, project_type: ModelType) -> str:
        """Genera README.md (método legacy)."""
        return f'''# {project_type.value.upper()} Deep Learning Project

Proyecto de deep learning usando PyTorch, Transformers y Diffusers.

## Instalación

```bash
pip install -r requirements.txt
```

## Entrenamiento

```bash
python train.py --config config.yaml
```

## Demo

```bash
python demo.py
```

## Estructura

- `model.py`: Arquitectura del modelo
- `train.py`: Script de entrenamiento
- `demo.py`: Interfaz Gradio para inferencia
- `config.yaml`: Configuración del proyecto
'''
    
    def _initialize_templates(self) -> Dict[str, Any]:
        """Inicializa plantillas de modelos."""
        return {}
    
    def _initialize_training_templates(self) -> Dict[str, Any]:
        """Inicializa plantillas de entrenamiento."""
        return {}


# Factory function
_deep_learning_generator = None

def get_deep_learning_generator() -> DeepLearningGenerator:
    """Obtiene instancia global del generador."""
    global _deep_learning_generator
    if _deep_learning_generator is None:
        _deep_learning_generator = DeepLearningGenerator()
    return _deep_learning_generator

