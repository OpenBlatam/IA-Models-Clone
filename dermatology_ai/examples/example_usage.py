"""
Ejemplos de uso de Dermatology AI
"""

import numpy as np
from PIL import Image
import requests
import io

from core.skin_analyzer import SkinAnalyzer
from services.image_processor import ImageProcessor
from services.skincare_recommender import SkincareRecommender


def example_direct_analysis():
    """Ejemplo de análisis directo sin API"""
    print("=== Ejemplo: Análisis Directo ===\n")
    
    # Inicializar componentes
    analyzer = SkinAnalyzer()
    processor = ImageProcessor()
    
    # Crear imagen de prueba (en producción, cargaría una imagen real)
    test_image = np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
    
    # Procesar y analizar
    enhanced = processor.enhance_for_analysis(test_image)
    result = analyzer.analyze_image(enhanced)
    
    # Mostrar resultados
    print("Puntuaciones de Calidad:")
    for key, value in result["quality_scores"].items():
        print(f"  {key}: {value:.2f}")
    
    print(f"\nTipo de Piel: {result['skin_type']}")
    print(f"Áreas Prioritarias: {', '.join(result['recommendations_priority'])}")
    
    if result["conditions"]:
        print("\nCondiciones Detectadas:")
        for condition in result["conditions"]:
            print(f"  - {condition['name']}: {condition['severity']} "
                  f"(confianza: {condition['confidence']:.2f})")
    else:
        print("\nNo se detectaron condiciones específicas")
    
    return result


def example_recommendations():
    """Ejemplo de generación de recomendaciones"""
    print("\n=== Ejemplo: Recomendaciones ===\n")
    
    # Simular resultado de análisis
    analysis_result = {
        "quality_scores": {
            "overall_score": 65.0,
            "hydration_score": 45.0,
            "texture_score": 70.0,
            "pigmentation_score": 60.0
        },
        "conditions": [
            {
                "name": "dryness",
                "severity": "moderate",
                "confidence": 0.7
            }
        ],
        "skin_type": "dry",
        "recommendations_priority": ["hydration", "texture"]
    }
    
    # Generar recomendaciones
    recommender = SkincareRecommender()
    recommendations = recommender.generate_recommendations(analysis_result)
    
    # Mostrar rutina
    print("Rutina de la Mañana:")
    for i, product in enumerate(recommendations["routine"]["morning"], 1):
        print(f"  {i}. {product['name']}")
        print(f"     Categoría: {product['category']}")
        print(f"     Uso: {product['usage_frequency']}")
        print(f"     Ingredientes clave: {', '.join(product['key_ingredients'])}")
        print()
    
    print("Rutina de la Noche:")
    for i, product in enumerate(recommendations["routine"]["evening"], 1):
        print(f"  {i}. {product['name']}")
        print(f"     Categoría: {product['category']}")
        print(f"     Uso: {product['usage_frequency']}")
        print()
    
    print("Tratamientos Semanales:")
    for i, product in enumerate(recommendations["routine"]["weekly"], 1):
        print(f"  {i}. {product['name']}")
        print(f"     Uso: {product['usage_frequency']}")
        print()
    
    print("Tips:")
    for tip in recommendations["tips"]:
        print(f"  • {tip}")
    
    return recommendations


def example_api_usage():
    """Ejemplo de uso de la API"""
    print("\n=== Ejemplo: Uso de API ===\n")
    
    base_url = "http://localhost:8006"
    
    # Crear imagen de prueba
    test_image = Image.new('RGB', (400, 400), color='rgb(200, 180, 160)')
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Enviar a API
    try:
        response = requests.post(
            f"{base_url}/dermatology/analyze-image",
            files={"file": ("test.jpg", img_bytes, "image/jpeg")},
            data={"enhance": True}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("Análisis exitoso!")
            print(f"Score general: {result['analysis']['quality_scores']['overall_score']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except requests.exceptions.ConnectionError:
        print("Error: No se pudo conectar al servidor.")
        print("Asegúrese de que el servidor esté ejecutándose:")
        print("  python main.py")


def example_video_analysis():
    """Ejemplo de análisis de video"""
    print("\n=== Ejemplo: Análisis de Video ===\n")
    
    from services.video_processor import VideoProcessor
    analyzer = SkinAnalyzer()
    video_processor = VideoProcessor(target_fps=1, max_frames=10)
    
    # Crear frames de prueba
    frames = [
        np.random.randint(100, 200, (400, 400, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    # Analizar
    result = analyzer.analyze_video(frames)
    
    print(f"Frames analizados: {result['analysis_frames']}")
    print(f"Score general: {result['quality_scores']['overall_score']:.2f}")
    print(f"Tipo de piel: {result['skin_type']}")
    
    return result


if __name__ == "__main__":
    # Ejecutar ejemplos
    example_direct_analysis()
    example_recommendations()
    example_video_analysis()
    
    # Descomentar para probar API (requiere servidor corriendo)
    # example_api_usage()






