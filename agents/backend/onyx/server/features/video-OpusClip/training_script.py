import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
import os
from tqdm import tqdm
import logging

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str):
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_experiment_tracking(config):
    """Configurar experiment tracking con TensorBoard y wandb"""
    # TensorBoard
    writer = SummaryWriter(log_dir=f"runs/{config['experiment']['name']}")
    
    # wandb
    if config['logging']['use_wandb']: wandb.init(project=config['experiment']['name'], config=config)
    
    return writer

def save_checkpoint(model, optimizer, epoch, loss, config, output_dir):
    """Guardar checkpoint del modelo"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint guardado: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    """Cargar checkpoint del modelo"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Checkpoint cargado desde epoch {checkpoint['epoch']}")
    return start_epoch

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, config):
    """Entrenar una época"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # Mover datos a dispositivo
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config['training']['gradient_clipping'] > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping'])
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Logging
        if batch_idx % config['logging']['log_interval'] == 0:
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
            if config['logging']['use_wandb']: wandb.log({'train_loss': loss.item(), 'epoch': epoch, 'batch': batch_idx})
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, writer, config):
    """Validar modelo"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    
    # Logging
    writer.add_scalar('Loss/val', avg_loss, epoch)
    if config['logging']['use_wandb']: wandb.log({'val_loss': avg_loss, 'epoch': epoch})
    
    return avg_loss

def main():
    # Cargar configuración
    config = load_config('config.yaml')
    
    # Configurar dispositivo
    device = torch.device(config['model']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Configurar experiment tracking
    writer = setup_experiment_tracking(config)
    
    # Crear directorio de salida
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    
    # Inicializar modelo (ejemplo genérico)
    model = nn.Linear(10, 1)  # Reemplazar con tu modelo
    model = model.to(device)
    
    # Configurar optimizador y loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    criterion = nn.MSELoss()
    
    # Configurar dataloader (ejemplo genérico)
    # dataset = YourDataset()
    # dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
    
    # Entrenamiento
    start_epoch = 0
    best_loss = float('inf')
    
    for epoch in range(start_epoch, config['training']['epochs']):
        # Entrenar
        train_loss = train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, config)
        
        # Validar
        val_loss = validate(model, val_dataloader, criterion, device, epoch, writer, config)
        
        # Guardar checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, config, config['experiment']['output_dir'])
        
        # Guardar mejor modelo
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['experiment']['output_dir'], 'best_model.pt'))
        
        logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    writer.close()
    if config['logging']['use_wandb']: wandb.finish()

if __name__ == "__main__":
    main() 