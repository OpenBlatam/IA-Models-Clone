"""
Ejemplo de uso de NLP y ML en Robot Maintenance Teaching AI.
"""

import numpy as np
from robot_maintenance_teaching_ai.core.nlp_processor import MaintenanceNLPProcessor
from robot_maintenance_teaching_ai.core.ml_predictor import MaintenancePredictor
from robot_maintenance_teaching_ai.config.maintenance_config import NLPConfig, MLConfig

def nlp_example():
    """Ejemplo de procesamiento NLP."""
    print("📝 Ejemplo de Procesamiento NLP\n")
    print("=" * 60)
    
    nlp_config = NLPConfig()
    nlp = MaintenanceNLPProcessor(nlp_config)
    
    text = """
    El robot industrial necesita mantenimiento preventivo. 
    Se deben revisar los engranajes, lubricar las juntas y 
    verificar el sistema de control. Hay ruidos extraños 
    en el reductor de velocidad.
    """
    
    print("Texto de entrada:")
    print(text)
    print("\n" + "-" * 60)
    
    entities = nlp.extract_entities(text)
    print("\nEntidades extraídas:")
    for entity_type, values in entities.items():
        if values:
            print(f"  {entity_type}: {values}")
    
    keywords = nlp.extract_keywords(text, top_n=5)
    print("\nPalabras clave:")
    for keyword, score in keywords:
        print(f"  {keyword}: {score:.4f}")
    
    sentiment = nlp.analyze_sentiment(text)
    print(f"\nAnálisis de sentimiento: {sentiment['label']} ({sentiment['score']:.4f})")
    
    steps = nlp.extract_maintenance_steps(text)
    print("\nPasos de mantenimiento detectados:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")


def ml_example():
    """Ejemplo de predicción ML."""
    print("\n\n🤖 Ejemplo de Predicción ML\n")
    print("=" * 60)
    
    ml_config = MLConfig()
    predictor = MaintenancePredictor(ml_config)
    
    print("\nGenerando datos de ejemplo para entrenamiento...")
    np.random.seed(42)
    n_samples = 1000
    
    features = np.random.rand(n_samples, 6)
    labels = np.random.randint(0, 2, n_samples)
    
    print("Entrenando modelo...")
    metrics = predictor.train(features, labels)
    
    print("\nMétricas de entrenamiento:")
    print(f"  Precisión en entrenamiento: {metrics['train_accuracy']:.4f}")
    print(f"  Precisión en prueba: {metrics['test_accuracy']:.4f}")
    print(f"  CV Score (media): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    
    print("\nPrediciendo necesidad de mantenimiento...")
    prediction = predictor.predict_maintenance_need(
        robot_type="industrial_robot",
        operating_hours=5000.0,
        error_count=3,
        temperature=45.0,
        vibration_level=0.8,
        last_maintenance_hours=200.0
    )
    
    print("\nResultado de la predicción:")
    print(f"  ¿Necesita mantenimiento?: {prediction['needs_maintenance']}")
    print(f"  Confianza: {prediction['confidence']:.4f}")
    print(f"  Probabilidades:")
    print(f"    Sin mantenimiento: {prediction['probabilities']['no_maintenance']:.4f}")
    print(f"    Con mantenimiento: {prediction['probabilities']['maintenance_needed']:.4f}")
    print(f"  Recomendación: {prediction['recommendation']}")


if __name__ == "__main__":
    try:
        nlp_example()
        ml_example()
        print("\n✅ Ejemplos completados!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()






