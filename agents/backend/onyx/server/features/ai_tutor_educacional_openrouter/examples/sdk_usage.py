"""
Ejemplo de uso del SDK de Python.
"""

from sdk import TutorClient


def ejemplo_basico():
    """Ejemplo básico de uso del SDK."""
    print("=== Ejemplo: Uso Básico del SDK ===\n")
    
    client = TutorClient(base_url="http://localhost:8000")
    
    try:
        # Hacer una pregunta
        response = client.ask_question(
            question="¿Qué es la fotosíntesis?",
            subject="ciencias",
            difficulty="intermedio"
        )
        
        if response.get("success"):
            print(f"Pregunta: ¿Qué es la fotosíntesis?")
            print(f"Respuesta: {response['data']['answer'][:200]}...\n")
        
        # Explicar un concepto
        explanation = client.explain_concept(
            concept="derivadas",
            subject="matematicas",
            difficulty="avanzado"
        )
        
        if explanation.get("success"):
            print(f"Concepto: Derivadas")
            print(f"Explicación: {explanation['data']['answer'][:200]}...\n")
        
        # Generar ejercicios
        exercises = client.generate_exercises(
            topic="ecuaciones cuadráticas",
            subject="matematicas",
            difficulty="intermedio",
            num_exercises=3
        )
        
        if exercises.get("success"):
            print(f"Ejercicios generados sobre: ecuaciones cuadráticas")
            print(f"Contenido: {exercises['data']['answer'][:200]}...\n")
        
        # Verificar salud del sistema
        health = client.get_health()
        print(f"Estado del sistema: {health.get('status', 'unknown')}\n")
        
    finally:
        client.close()


def ejemplo_con_errores():
    """Ejemplo manejando errores."""
    print("=== Ejemplo: Manejo de Errores ===\n")
    
    client = TutorClient(base_url="http://localhost:8000")
    
    try:
        response = client.ask_question(
            question="Test question",
            subject="invalid_subject"
        )
        print(f"Respuesta: {response}")
    except Exception as e:
        print(f"Error capturado: {e}\n")
    finally:
        client.close()


def ejemplo_metricas():
    """Ejemplo obteniendo métricas."""
    print("=== Ejemplo: Obtener Métricas ===\n")
    
    client = TutorClient(base_url="http://localhost:8000")
    
    try:
        metrics = client.get_metrics()
        
        if metrics.get("success"):
            data = metrics.get("data", {})
            metrics_data = data.get("metrics", {})
            
            print("Métricas del Sistema:")
            print(f"  Total de preguntas: {metrics_data.get('total_questions', 0)}")
            print(f"  Total de explicaciones: {metrics_data.get('total_explanations', 0)}")
            print(f"  Tiempo promedio de respuesta: {metrics_data.get('average_response_time', 0):.3f}s")
            print(f"  Tasa de aciertos de cache: {data.get('cache', {}).get('hit_rate', 0):.1%}\n")
        
    finally:
        client.close()


if __name__ == "__main__":
    print("=" * 60)
    print("AI Tutor Educacional - Ejemplos de SDK")
    print("=" * 60)
    print()
    print("⚠️  Asegúrate de que el servidor esté corriendo en http://localhost:8000")
    print()
    
    try:
        ejemplo_basico()
        print("\n" + "-" * 60 + "\n")
        
        ejemplo_metricas()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()






