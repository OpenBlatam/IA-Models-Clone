"""
Advanced Training Example
=========================

Ejemplo de uso de funcionalidades avanzadas:
- Distributed Training
- LoRA Fine-tuning
- Hyperparameter Optimization
- Model Ensembling
- Advanced Callbacks
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import logging
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.routing_models import ModelFactory, ModelConfig, apply_lora_to_model, count_lora_parameters
from core.routing_models.ensemble import ModelEnsemble, EnsembleBuilder
from core.routing_data import RouteDataset, RouteDataLoader, RoutePreprocessor
from core.routing_training import (
    RouteTrainer, TrainingConfig,
    DistributedTrainer, GradientAccumulator,
    HyperparameterOptimizer, create_objective_function,
    LearningRateFinder, GradientMonitor, ModelEMA
)


def example_lora_finetuning():
    """Ejemplo de fine-tuning con LoRA."""
    logger.info("=== Ejemplo: Fine-tuning con LoRA ===")
    
    # Crear modelo base
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Contar parámetros originales
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Parámetros totales: {total_params:,}")
    logger.info(f"Parámetros entrenables: {trainable_params:,}")
    
    # Aplicar LoRA
    model = apply_lora_to_model(model, rank=8, alpha=16.0)
    
    # Contar parámetros LoRA
    lora_stats = count_lora_parameters(model)
    logger.info(f"Después de LoRA:")
    logger.info(f"  Parámetros totales: {lora_stats['total_parameters']:,}")
    logger.info(f"  Parámetros entrenables: {lora_stats['trainable_parameters']:,}")
    logger.info(f"  Parámetros LoRA: {lora_stats['lora_parameters']:,}")
    logger.info(f"  Ratio entrenable: {lora_stats['trainable_ratio']:.2%}")


def example_distributed_training():
    """Ejemplo de entrenamiento distribuido."""
    logger.info("=== Ejemplo: Entrenamiento Distribuido ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Configurar entrenamiento distribuido
    if torch.cuda.device_count() > 1:
        distributed_trainer = DistributedTrainer(
            model=model,
            use_ddp=True,  # Usar DistributedDataParallel
            use_dp=False
        )
        logger.info(f"Modelo configurado para {torch.cuda.device_count()} GPUs")
    else:
        logger.info("Solo hay 1 GPU disponible, entrenamiento normal")


def example_gradient_accumulation():
    """Ejemplo de acumulación de gradientes."""
    logger.info("=== Ejemplo: Acumulación de Gradientes ===")
    
    accumulator = GradientAccumulator(accumulation_steps=4)
    
    # Simular entrenamiento
    for step in range(10):
        # Forward y backward (sin optimizer.step())
        # ...
        
        if accumulator.should_update():
            # optimizer.step()
            logger.info(f"Actualizar pesos en paso {step + 1}")
        
        accumulator.step()


def example_hyperparameter_optimization():
    """Ejemplo de optimización de hiperparámetros."""
    logger.info("=== Ejemplo: Optimización de Hiperparámetros ===")
    
    # Generar datos sintéticos
    features = [np.random.randn(20).astype(np.float32) for _ in range(100)]
    targets = [np.random.randn(4).astype(np.float32) for _ in range(100)]
    
    preprocessor = RoutePreprocessor()
    preprocessor.fit(features, targets)
    
    dataset = RouteDataset(features, targets, preprocessor=preprocessor)
    train_dataset, val_dataset, _ = dataset.split()
    
    train_loader, val_loader = RouteDataLoader.create_train_val_loaders(
        train_dataset, val_dataset, batch_size=32
    )
    
    # Crear optimizador
    optimizer = HyperparameterOptimizer(
        study_name="routing_optimization",
        n_trials=10  # Reducido para ejemplo
    )
    
    # Función objetivo
    def model_factory(config):
        return ModelFactory.create_model("mlp", ModelConfig(**config))
    
    objective = create_objective_function(
        train_loader,
        val_loader,
        model_factory,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Optimizar (comentado para no ejecutar en ejemplo)
    # study = optimizer.optimize(objective, n_trials=10)
    # logger.info(f"Mejores parámetros: {optimizer.get_best_params()}")


def example_model_ensemble():
    """Ejemplo de ensamblaje de modelos."""
    logger.info("=== Ejemplo: Ensamblaje de Modelos ===")
    
    # Crear múltiples modelos
    config1 = ModelConfig(input_dim=20, hidden_dims=[128, 128], output_dim=4)
    config2 = ModelConfig(input_dim=20, hidden_dims=[256, 256], output_dim=4)
    config3 = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    
    model1 = ModelFactory.create_model("mlp", config1)
    model2 = ModelFactory.create_model("mlp", config2)
    model3 = ModelFactory.create_model("mlp", config3)
    
    # Crear ensamblaje
    ensemble = ModelEnsemble(
        models=[model1, model2, model3],
        weights=[0.4, 0.3, 0.3],  # Pesos personalizados
        voting_method="weighted_average"
    )
    
    logger.info(f"Ensamblaje creado con {len(ensemble.models)} modelos")
    logger.info(f"Diversidad del ensamblaje: {ensemble.get_model_diversity():.4f}")
    
    # Predecir con incertidumbre
    dummy_input = torch.randn(1, 20)
    mean, std = ensemble.predict_with_uncertainty(dummy_input, return_std=True)
    logger.info(f"Predicción: {mean.squeeze().tolist()}")
    logger.info(f"Incertidumbre: {std.squeeze().tolist()}")


def example_advanced_callbacks():
    """Ejemplo de callbacks avanzados."""
    logger.info("=== Ejemplo: Callbacks Avanzados ===")
    
    # Learning Rate Finder
    lr_finder = LearningRateFinder(min_lr=1e-8, max_lr=1.0, num_iterations=50)
    logger.info("LearningRateFinder creado")
    
    # Gradient Monitor
    grad_monitor = GradientMonitor(log_frequency=5, max_grad_norm=10.0)
    logger.info("GradientMonitor creado")
    
    # Model EMA
    ema = ModelEMA(decay=0.9999)
    logger.info("ModelEMA creado")
    
    # Uso en entrenador
    # trainer = RouteTrainer(..., callbacks=[lr_finder, grad_monitor, ema])


def main():
    """Ejecutar todos los ejemplos."""
    logger.info("Ejecutando ejemplos de funcionalidades avanzadas...\n")
    
    example_lora_finetuning()
    print()
    
    example_distributed_training()
    print()
    
    example_gradient_accumulation()
    print()
    
    example_model_ensemble()
    print()
    
    example_advanced_callbacks()
    print()
    
    # example_hyperparameter_optimization()  # Comentado para no ejecutar
    print()
    
    logger.info("=== Todos los ejemplos completados ===")


if __name__ == "__main__":
    main()

