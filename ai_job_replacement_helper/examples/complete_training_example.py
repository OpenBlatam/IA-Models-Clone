"""
Complete Training Example - Ejemplo completo de entrenamiento
=============================================================

Ejemplo completo que demuestra el uso de todas las utilidades refactorizadas.
Sigue mejores prácticas de deep learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Imports de utilidades refactorizadas
from core.base_model_service import BaseModelService, DeviceConfig
from core.utils.model_utils import (
    initialize_weights,
    get_model_size,
    count_parameters,
    clip_gradients,
)
from core.utils.training_utils import (
    create_optimizer,
    create_scheduler,
    train_one_epoch,
    validate_one_epoch,
    EarlyStopping,
)
from core.utils.data_utils import (
    create_data_splits,
    create_dataloader,
    normalize_tensor,
)
from core.utils.validation_utils import (
    validate_model_config,
    validate_training_config,
    validate_model_output,
)
from core.utils.performance_utils import (
    timer,
    profile_model,
    get_memory_usage,
)
from core.config.model_config import TrainingConfig, OptimizerConfig, SchedulerConfig


class SimpleClassifier(nn.Module):
    """Clasificador simple de ejemplo"""
    
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TrainingService(BaseModelService):
    """Servicio de entrenamiento usando utilidades refactorizadas"""
    
    def __init__(self):
        """Inicializar servicio"""
        super().__init__()
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss()
    
    def create_model(
        self,
        input_size: int,
        num_classes: int,
        config: dict = None
    ) -> nn.Module:
        """
        Crear y configurar modelo.
        
        Args:
            input_size: Tamaño de entrada
            num_classes: Número de clases
            config: Configuración adicional
        
        Returns:
            Modelo configurado
        """
        # Validar configuración
        model_config = {
            "input_size": input_size,
            "output_size": num_classes,
            **config or {}
        }
        is_valid, errors = validate_model_config(model_config)
        if not is_valid:
            raise ValueError(f"Invalid model config: {errors}")
        
        # Crear modelo
        model = SimpleClassifier(input_size, num_classes)
        
        # Mover a dispositivo
        model = self.move_to_device(model)
        
        # Inicializar pesos
        initialize_weights(model, method="xavier_uniform")
        
        # Validar modelo
        is_valid, error_msg, output = validate_model_output(
            model,
            input_shape=(input_size,),
            expected_output_shape=(num_classes,)
        )
        if not is_valid:
            raise ValueError(f"Model validation failed: {error_msg}")
        
        self.model = model
        
        # Log información del modelo
        model_info = self.get_model_info(model)
        print(f"Model created: {model_info}")
        
        return model
    
    def setup_training(
        self,
        learning_rate: float = 1e-3,
        optimizer_type: str = "adamw",
        scheduler_type: str = "cosine"
    ):
        """Configurar optimizador y scheduler"""
        if self.model is None:
            raise ValueError("Model must be created first")
        
        # Validar configuración de entrenamiento
        train_config = {
            "learning_rate": learning_rate,
            "num_epochs": 10,
            "batch_size": 32,
        }
        is_valid, errors = validate_training_config(train_config)
        if not is_valid:
            raise ValueError(f"Invalid training config: {errors}")
        
        # Crear optimizador
        self.optimizer = create_optimizer(
            self.model,
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            weight_decay=1e-5
        )
        
        # Crear scheduler
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            T_max=10
        )
        
        print(f"Training setup: optimizer={optimizer_type}, scheduler={scheduler_type}")
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        use_mixed_precision: bool = True,
        max_grad_norm: float = 1.0
    ):
        """Entrenar modelo completo"""
        if self.model is None or self.optimizer is None:
            raise ValueError("Model and optimizer must be setup first")
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=5,
            min_delta=0.001,
            mode="min"
        )
        
        # Mixed precision scaler
        scaler = None
        if use_mixed_precision and self.device_config.use_mixed_precision:
            scaler = torch.cuda.amp.GradScaler()
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Entrenar
            with timer(f"Training epoch {epoch + 1}"):
                train_loss, train_acc = train_one_epoch(
                    self.model,
                    train_loader,
                    self.criterion,
                    self.optimizer,
                    device=self.device_config.device,
                    use_mixed_precision=use_mixed_precision,
                    max_grad_norm=max_grad_norm,
                    scaler=scaler
                )
            
            # Validar
            with timer(f"Validation epoch {epoch + 1}"):
                val_loss, val_acc = validate_one_epoch(
                    self.model,
                    val_loader,
                    self.criterion,
                    device=self.device_config.device,
                    use_mixed_precision=use_mixed_precision
                )
            
            # Scheduler step
            if self.scheduler:
                self.scheduler.step()
            
            # Logging
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f if train_acc else 'N/A'}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f if val_acc else 'N/A'}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                print("Early stopping triggered")
                break
            
            # Guardar mejor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch=epoch,
                    loss=val_loss,
                    filepath="best_model.pt"
                )
        
        print(f"\nTraining completed. Best val loss: {best_val_loss:.4f}")
    
    def profile_inference(self, input_shape: tuple, num_runs: int = 100):
        """Perfilar modelo para inferencia"""
        if self.model is None:
            raise ValueError("Model must be created first")
        
        print("\nProfiling model inference...")
        stats = profile_model(self.model, input_shape, num_runs=num_runs)
        
        print(f"Mean inference time: {stats['mean_time']*1000:.2f} ms")
        print(f"Throughput: {stats['throughput']:.2f} samples/sec")
        
        return stats


# Ejemplo de uso
if __name__ == "__main__":
    # Crear servicio
    service = TrainingService()
    
    # Crear modelo
    model = service.create_model(input_size=784, num_classes=10)
    
    # Setup entrenamiento
    service.setup_training(learning_rate=1e-3, optimizer_type="adamw")
    
    # Crear datos de ejemplo (en producción, usar datasets reales)
    # train_loader, val_loader = create_dataloaders(...)
    
    # Entrenar
    # service.train_model(train_loader, val_loader, num_epochs=10)
    
    # Perfilar
    # service.profile_inference(input_shape=(784,), num_runs=100)
    
    print("Example completed!")




