"""
Training Script Generator
=========================

Generador de scripts de entrenamiento optimizados.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    framework: str = "pytorch"
    mixed_precision: bool = True
    gradient_accumulation: int = 1
    early_stopping: bool = True
    checkpointing: bool = True
    wandb_enabled: bool = False
    tensorboard_enabled: bool = True
    distributed: bool = False
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4


class TrainingScriptGenerator:
    """
    Generador de scripts de entrenamiento.
    """
    
    def __init__(self):
        """Inicializar generador."""
        pass
    
    def generate_training_script(
        self,
        project_dir: Path,
        config: Optional[TrainingConfig] = None
    ) -> str:
        """
        Generar script de entrenamiento.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Contenido del script de entrenamiento
        """
        if config is None:
            config = TrainingConfig()
        
        if config.framework == "pytorch":
            return self._generate_pytorch_script(project_dir, config)
        elif config.framework == "tensorflow":
            return self._generate_tensorflow_script(project_dir, config)
        else:
            return self._generate_pytorch_script(project_dir, config)
    
    def _generate_pytorch_script(
        self,
        project_dir: Path,
        config: TrainingConfig
    ) -> str:
        """Generar script PyTorch."""
        script_content = f""""""
Script de Entrenamiento Optimizado - PyTorch
============================================

Generado automáticamente por DeepLearningGenerator
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from app.models import get_model
from app.data import get_dataset, get_data_loaders
from app.training import EarlyStopping, save_checkpoint, load_checkpoint
from app.evaluation import evaluate_model

"""
        
        if config.wandb_enabled:
            script_content += """import wandb
"""
        
        if config.tensorboard_enabled:
            script_content += """from torch.utils.tensorboard import SummaryWriter
"""
        
        script_content += f"""
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, criterion, optimizer, device, scaler=None, gradient_accumulation={config.gradient_accumulation}):
    \"\"\"Entrenar una época.\"\"\"
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        # Forward pass con mixed precision
"""
        
        if config.mixed_precision:
            script_content += """        with autocast():
            output = model(data)
            loss = criterion(output, target) / gradient_accumulation
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % gradient_accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
"""
        else:
            script_content += """        output = model(data)
        loss = criterion(output, target) / gradient_accumulation
        loss.backward()
        
        if (batch_idx + 1) % gradient_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()
"""
        
        script_content += f"""
        total_loss += loss.item() * gradient_accumulation
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def validate(model, val_loader, criterion, device):
    \"\"\"Validar modelo.\"\"\"
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc="Validation"):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    return avg_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--config', type=str, default='app/config/training_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--epochs', type=int, default={config.num_epochs})
    parser.add_argument('--batch-size', type=int, default={config.batch_size})
    parser.add_argument('--lr', type=float, default={config.learning_rate})
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {{device}}")
    
    # Inicializar experiment tracking
"""
        
        if config.wandb_enabled:
            script_content += """    wandb.init(project="deep-learning-project", config=vars(args))
"""
        
        if config.tensorboard_enabled:
            script_content += """    writer = SummaryWriter(log_dir=f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")
"""
        
        script_content += f"""
    # Cargar datos
    train_loader, val_loader = get_data_loaders(
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Crear modelo
    model = get_model().to(device)
    
    # Loss y optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Mixed precision
"""
        
        if config.mixed_precision:
            script_content += """    scaler = GradScaler()
"""
        else:
            script_content += """    scaler = None
"""
        
        script_content += f"""
    # Early stopping
"""
        
        if config.early_stopping:
            script_content += """    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
"""
        
        script_content += f"""
    # Resume training
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Resumed from epoch {{start_epoch}}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {{epoch+1}}/{{args.epochs}}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        logger.info(f"Train Loss: {{train_loss:.4f}}, Val Loss: {{val_loss:.4f}}, Val Acc: {{val_accuracy:.2f}}%")
"""
        
        if config.wandb_enabled:
            script_content += """        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })
"""
        
        if config.tensorboard_enabled:
            script_content += """        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_accuracy, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
"""
        
        script_content += f"""
        # Checkpointing
"""
        
        if config.checkpointing:
            script_content += """        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'val_accuracy': val_accuracy
            }, f"checkpoints/best_model_epoch_{epoch+1}.pth")
            logger.info(f"Saved checkpoint with val_loss: {val_loss:.4f}")
"""
        
        script_content += f"""
        # Early stopping
"""
        
        if config.early_stopping:
            script_content += """        if early_stopping(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
"""
        
        script_content += """
    logger.info("Training completed!")
    
    # Final evaluation
    final_loss, final_accuracy = evaluate_model(model, val_loader, device)
    logger.info(f"Final Validation - Loss: {final_loss:.4f}, Accuracy: {final_accuracy:.2f}%")
"""

        if config.tensorboard_enabled:
            script_content += """    writer.close()
"""
        
        script_content += """
if __name__ == "__main__":
    main()
"""
        
        return script_content
    
    def _generate_tensorflow_script(
        self,
        project_dir: Path,
        config: TrainingConfig
    ) -> str:
        """Generar script TensorFlow."""
        # Implementación básica para TensorFlow
        return """# TensorFlow training script\n# TODO: Implement\n"""
    
    def generate_all(
        self,
        project_dir: Path,
        config: Optional[TrainingConfig] = None
    ) -> Dict[str, str]:
        """
        Generar todos los archivos de entrenamiento.
        
        Args:
            project_dir: Directorio del proyecto
            config: Configuración (opcional)
            
        Returns:
            Diccionario con archivos generados
        """
        if config is None:
            config = TrainingConfig()
        
        files = {}
        scripts_dir = project_dir / "scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Training script
        script_content = self.generate_training_script(project_dir, config)
        script_path = scripts_dir / "train.py"
        script_path.write_text(script_content, encoding='utf-8')
        files['scripts/train.py'] = script_content
        
        logger.info(f"Script de entrenamiento generado en {script_path}")
        
        return files


# Instancia global
_global_training_generator: Optional[TrainingScriptGenerator] = None


def get_training_generator() -> TrainingScriptGenerator:
    """
    Obtener instancia global del generador de scripts de entrenamiento.
    
    Returns:
        Instancia del generador
    """
    global _global_training_generator
    
    if _global_training_generator is None:
        _global_training_generator = TrainingScriptGenerator()
    
    return _global_training_generator

