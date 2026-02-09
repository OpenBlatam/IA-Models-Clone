#!/usr/bin/env python3
import time
import numpy as np
import asyncio
from evaluation_metrics_optimized import (
    OptimizedSEOModelEvaluator, SEOMetricsConfig, 
    ClassificationMetricsConfig, RegressionMetricsConfig
)

async def test_optimization():
    print("🚀 Testing SEO Evaluation Optimization")
    
    # Configuración optimizada
    seo_config = SEOMetricsConfig(
        use_vectorization=True,
        use_caching=True,
        batch_size=10000,
        max_workers=4
    )
    
    evaluator = OptimizedSEOModelEvaluator(
        seo_config=seo_config,
        classification_config=ClassificationMetricsConfig(use_vectorization=True),
        regression_config=RegressionMetricsConfig(use_vectorization=True)
    )
    
    # Datos de prueba
    test_data = {
        'y_true': np.random.randint(0, 2, 50000),
        'y_pred': np.random.randint(0, 2, 50000),
        'y_prob': np.random.random(50000),
        'content_data': {
            'content_length': np.random.randint(200, 2000, 50000),
            'keyword_density': np.random.random(50000) * 0.05,
            'readability_score': np.random.random(50000) * 100
        }
    }
    
    # Test de rendimiento
    start_time = time.time()
    results = await evaluator.evaluate_seo_model(
        model=None, test_data=test_data, task_type="classification"
    )
    execution_time = time.time() - start_time
    
    print(f"✅ Evaluación completada en {execution_time:.4f}s")
    print(f"📊 Resultados: {len(results)} métricas calculadas")
    print(f"🎯 F1 Score: {results.get('f1_score', 0):.4f}")
    
    evaluator.cleanup()

if __name__ == "__main__":
    asyncio.run(test_optimization())
