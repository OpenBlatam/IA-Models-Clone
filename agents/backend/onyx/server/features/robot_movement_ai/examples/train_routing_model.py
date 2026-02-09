"""
Example: Training a Routing Model
=================================

Ejemplo completo de cómo entrenar un modelo de enrutamiento
usando la arquitectura modular.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import logging
from typing import List

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports de la arquitectura modular
from core.routing_config import load_config, save_config
from core.routing_models import ModelFactory, ModelConfig
from core.routing_data import RouteDataset, RouteDataLoader, RoutePreprocessor, FeatureExtractor
from core.routing_training import RouteTrainer, TrainingConfig, EarlyStopping, ModelCheckpoint
from core.experiment_tracker import ExperimentTracker


def generate_synthetic_data(n_samples: int = 1000) -> tuple:
    """
    Generar datos sintéticos para ejemplo.
    
    Args:
        n_samples: Número de muestras
        
    Returns:
        (features, targets)
    """
    features = []
    targets = []
    
    for _ in range(n_samples):
        # Features aleatorias
        feat = np.random.randn(20).astype(np.float32)
        features.append(feat)
        
        # Targets (simulados)
        target = np.array([
            abs(feat[0]) * 10,  # predicted_time
            abs(feat[1]) * 5,   # predicted_cost
            abs(feat[2]),       # predicted_load
            min(1.0, abs(feat[3]) * 0.5 + 0.5)  # success_probability
        ], dtype=np.float32)
        targets.append(target)
    
    return features, targets


def main():
    """Función principal de ejemplo."""
    logger.info("=== Ejemplo de Entrenamiento de Modelo de Enrutamiento ===")
    
    # 1. Cargar configuración
    config_path = "../config/default_config.yaml"
    config = load_config(config_path)
    logger.info("Configuración cargada")
    
    # 2. Generar datos sintéticos (en producción, cargar desde archivos)
    logger.info("Generando datos sintéticos...")
    features, targets = generate_synthetic_data(n_samples=1000)
    
    # 3. Crear preprocesador
    preprocessor = RoutePreprocessor()
    preprocessor.fit(features, targets)
    logger.info("Preprocesador ajustado")
    
    # 4. Crear dataset
    dataset = RouteDataset(features, targets, preprocessor=preprocessor)
    train_dataset, val_dataset, test_dataset = dataset.split(train_ratio=0.8, val_ratio=0.1)
    logger.info(f"Dataset dividido: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 5. Crear data loaders
    train_loader, val_loader = RouteDataLoader.create_train_val_loaders(
        train_dataset,
        val_dataset,
        batch_size=config["training"]["batch_size"],
        num_workers=0
    )
    logger.info("Data loaders creados")
    
    # 6. Crear modelo usando factory
    model_config = ModelConfig(**config["model"])
    model = ModelFactory.create_model(config["model"]["model_type"], model_config)
    logger.info(f"Modelo creado: {model.__class__.__name__}")
    logger.info(f"Número de parámetros: {model.count_parameters():,}")
    
    # 7. Crear experiment tracker
    tracker = ExperimentTracker(
        project_name=config["experiment"]["project_name"],
        use_wandb=config["experiment"]["use_wandb"],
        use_tensorboard=config["experiment"]["use_tensorboard"],
        log_dir=config["experiment"]["log_dir"]
    )
    tracker.log_config(config)
    logger.info("Experiment tracker inicializado")
    
    # 8. Crear callbacks
    callbacks = [
        EarlyStopping(
            patience=config["training"]["early_stopping_patience"],
            min_delta=config["training"]["early_stopping_min_delta"]
        ),
        ModelCheckpoint(
            checkpoint_dir=config["training"]["checkpoint_dir"],
            save_best=True
        )
    ]
    
    # 9. Crear entrenador
    training_config = TrainingConfig(**config["training"])
    trainer = RouteTrainer(
        model=model,
        config=training_config,
        train_loader=train_loader,
        val_loader=val_loader,
        callbacks=callbacks
    )
    logger.info("Trainer creado")
    
    # 10. Entrenar
    logger.info("Iniciando entrenamiento...")
    history = trainer.train()
    
    # 11. Loggear métricas finales
    if tracker:
        for epoch_metrics in history["history"]:
            tracker.log_metrics(epoch_metrics, step=epoch_metrics["epoch"])
        tracker.finish()
    
    logger.info("=== Entrenamiento Completado ===")
    logger.info(f"Mejor pérdida de validación: {history['best_val_loss']:.4f}")
    
    # 12. Evaluar en test set (opcional)
    logger.info("Evaluando en test set...")
    model.eval()
    test_losses = []
    
    import torch
    from torch.nn import MSELoss
    criterion = MSELoss()
    
    with torch.no_grad():
        for batch_features, batch_targets, _ in RouteDataLoader.create(
            test_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False
        ):
            batch_features = batch_features.to(model.device)
            batch_targets = batch_targets.to(model.device)
            
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            test_losses.append(loss.item())
    
    avg_test_loss = np.mean(test_losses)
    logger.info(f"Pérdida en test set: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()


