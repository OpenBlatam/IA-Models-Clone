"""
Evaluation and Debugging Example
=================================

Ejemplo de evaluación avanzada y debugging.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from core.routing_models import ModelFactory, ModelConfig
from core.routing_data import RouteDataset, RouteDataLoader, RoutePreprocessor
from core.routing_evaluation import (
    ModelEvaluator, EvaluationConfig,
    KFoldCrossValidator, ModelComparator,
    EvaluationVisualizer, plot_training_curves
)
from core.routing_debugging import (
    ModelDebugger, DebugConfig,
    analyze_gradients, analyze_activations,
    detect_nans
)


def example_model_evaluation():
    """Ejemplo de evaluación de modelo."""
    logger.info("=== Model Evaluation ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Generar datos
    features = [np.random.randn(20).astype(np.float32) for _ in range(200)]
    targets = [np.random.randn(4).astype(np.float32) for _ in range(200)]
    
    preprocessor = RoutePreprocessor()
    preprocessor.fit(features, targets)
    
    dataset = RouteDataset(features, targets, preprocessor=preprocessor)
    _, val_dataset, _ = dataset.split()
    
    val_loader = RouteDataLoader.create(
        val_dataset,
        batch_size=32,
        shuffle=False
    )
    
    # Evaluar
    evaluator = ModelEvaluator(EvaluationConfig(compute_confidence_intervals=True))
    metrics = evaluator.evaluate(model, val_loader)
    
    logger.info(f"R²: {metrics.r2:.4f}")
    logger.info(f"MSE: {metrics.mse:.4f}")
    logger.info(f"MAE: {metrics.mae:.4f}")
    logger.info(f"Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")


def example_cross_validation():
    """Ejemplo de validación cruzada."""
    logger.info("=== Cross Validation ===")
    
    # Crear modelo factory
    def model_factory():
        config = ModelConfig(input_dim=20, hidden_dims=[128, 128], output_dim=4)
        return ModelFactory.create_model("mlp", config)
    
    # Generar datos
    features = [np.random.randn(20).astype(np.float32) for _ in range(500)]
    targets = [np.random.randn(4).astype(np.float32) for _ in range(500)]
    
    preprocessor = RoutePreprocessor()
    preprocessor.fit(features, targets)
    
    dataset = RouteDataset(features, targets, preprocessor=preprocessor)
    
    # K-Fold CV
    from core.routing_training import TrainingConfig
    
    cv = KFoldCrossValidator(n_splits=5)
    results = cv.validate(
        model_factory,
        dataset,
        TrainingConfig(epochs=10, batch_size=32)
    )
    
    logger.info(f"Mean R²: {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
    logger.info(f"Best fold: {results['best_fold']}")


def example_model_comparison():
    """Ejemplo de comparación de modelos."""
    logger.info("=== Model Comparison ===")
    
    # Crear múltiples modelos
    models = {
        "small": ModelFactory.create_model("mlp", ModelConfig(input_dim=20, hidden_dims=[64, 64], output_dim=4)),
        "medium": ModelFactory.create_model("mlp", ModelConfig(input_dim=20, hidden_dims=[128, 128], output_dim=4)),
        "large": ModelFactory.create_model("mlp", ModelConfig(input_dim=20, hidden_dims=[256, 256], output_dim=4))
    }
    
    # Generar datos
    features = [np.random.randn(20).astype(np.float32) for _ in range(100)]
    targets = [np.random.randn(4).astype(np.float32) for _ in range(100)]
    
    preprocessor = RoutePreprocessor()
    preprocessor.fit(features, targets)
    
    dataset = RouteDataset(features, targets, preprocessor=preprocessor)
    _, val_dataset, _ = dataset.split()
    
    val_loader = RouteDataLoader.create(val_dataset, batch_size=32, shuffle=False)
    
    # Comparar
    comparator = ModelComparator()
    results = comparator.compare_models(models, val_loader, metric="r2")
    
    summary = comparator.get_comparison_summary()
    logger.info(f"Mejor modelo: {summary['best_model']}")
    logger.info(f"Mejor score: {summary['best_score']:.4f}")


def example_debugging():
    """Ejemplo de debugging."""
    logger.info("=== Model Debugging ===")
    
    # Crear modelo
    config = ModelConfig(input_dim=20, hidden_dims=[128, 256, 128], output_dim=4)
    model = ModelFactory.create_model("mlp", config)
    
    # Debugger
    debugger = ModelDebugger(model, DebugConfig())
    debugger.enable_anomaly_detection()
    debugger.register_hooks()
    
    # Diagnóstico
    input_tensor = torch.randn(1, 20)
    diagnosis = debugger.diagnose(input_tensor)
    
    logger.info("Diagnóstico completado")
    logger.info(f"Forward pass: {diagnosis['forward_pass']['success']}")
    
    # Análisis de gradientes
    model.train()
    output = model(input_tensor)
    loss = output.mean()
    loss.backward()
    
    grad_analysis = analyze_gradients(model)
    logger.info(f"Gradient norm: {grad_analysis['total_norm']:.4f}")
    
    # Detección de NaN
    nan_check = detect_nans(model, input_tensor)
    logger.info(f"NaN en pesos: {any(nan_check['weights'].values())}")
    
    debugger.remove_hooks()
    debugger.disable_anomaly_detection()


def example_visualization():
    """Ejemplo de visualización."""
    logger.info("=== Evaluation Visualization ===")
    
    # Generar datos de ejemplo
    predictions = np.random.randn(100, 4)
    targets = predictions + np.random.randn(100, 4) * 0.1
    
    # Visualizar
    try:
        viz = EvaluationVisualizer()
        
        # Predicciones vs targets
        fig1 = viz.plot_predictions_vs_targets(
            predictions, targets,
            output_names=["time", "cost", "load", "probability"],
            save_path="predictions_vs_targets.png"
        )
        
        # Residuos
        fig2 = viz.plot_residuals(
            predictions, targets,
            save_path="residuals.png"
        )
        
        # Distribución de errores
        fig3 = viz.plot_error_distribution(
            predictions, targets,
            save_path="error_distribution.png"
        )
        
        logger.info("Visualizaciones guardadas")
    except Exception as e:
        logger.warning(f"Error en visualización: {e}")


def main():
    """Ejecutar ejemplos."""
    logger.info("Ejecutando ejemplos de evaluación y debugging...\n")
    
    example_model_evaluation()
    print()
    
    # example_cross_validation()  # Comentado para no ejecutar entrenamiento
    print()
    
    example_model_comparison()
    print()
    
    example_debugging()
    print()
    
    example_visualization()
    print()
    
    logger.info("=== Todos los ejemplos completados ===")


if __name__ == "__main__":
    main()

