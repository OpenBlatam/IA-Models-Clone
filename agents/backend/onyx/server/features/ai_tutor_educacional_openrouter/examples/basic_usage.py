"""
Ejemplo básico de uso del AI Tutor Educacional.
"""

import asyncio
import os
from pathlib import Path

# Agregar el directorio padre al path para importar el módulo
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tutor import AITutor
from config.tutor_config import TutorConfig
from core.conversation_manager import ConversationManager
from core.learning_analyzer import LearningAnalyzer


async def ejemplo_pregunta_simple():
    """Ejemplo de hacer una pregunta simple."""
    print("=== Ejemplo: Pregunta Simple ===\n")
    
    config = TutorConfig()
    tutor = AITutor(config)
    
    try:
        response = await tutor.ask_question(
            question="¿Qué es la fotosíntesis?",
            subject="ciencias",
            difficulty="intermedio"
        )
        
        print(f"Pregunta: ¿Qué es la fotosíntesis?")
        print(f"\nRespuesta:\n{response['answer']}\n")
        print(f"Modelo usado: {response['model']}")
        print(f"Timestamp: {response['timestamp']}\n")
        
    finally:
        await tutor.close()


async def ejemplo_explicacion_concepto():
    """Ejemplo de explicar un concepto."""
    print("=== Ejemplo: Explicar Concepto ===\n")
    
    config = TutorConfig()
    tutor = AITutor(config)
    
    try:
        response = await tutor.explain_concept(
            concept="derivadas",
            subject="matematicas",
            difficulty="avanzado"
        )
        
        print(f"Concepto: Derivadas")
        print(f"\nExplicación:\n{response['answer']}\n")
        
    finally:
        await tutor.close()


async def ejemplo_generar_ejercicios():
    """Ejemplo de generar ejercicios."""
    print("=== Ejemplo: Generar Ejercicios ===\n")
    
    config = TutorConfig()
    tutor = AITutor(config)
    
    try:
        response = await tutor.generate_exercise(
            topic="ecuaciones cuadráticas",
            subject="matematicas",
            difficulty="intermedio",
            num_exercises=3
        )
        
        print(f"Tema: Ecuaciones Cuadráticas")
        print(f"\nEjercicios:\n{response['answer']}\n")
        
    finally:
        await tutor.close()


async def ejemplo_con_conversacion():
    """Ejemplo usando el gestor de conversaciones."""
    print("=== Ejemplo: Conversación con Contexto ===\n")
    
    config = TutorConfig()
    tutor = AITutor(config)
    conversation_manager = ConversationManager()
    
    conversation_id = "estudiante_001"
    
    try:
        # Primera pregunta
        pregunta1 = "¿Qué es una variable en programación?"
        response1 = await tutor.ask_question(
            question=pregunta1,
            subject="programacion",
            difficulty="basico"
        )
        
        conversation_manager.add_message(conversation_id, "user", pregunta1)
        conversation_manager.add_message(conversation_id, "assistant", response1["answer"])
        
        print(f"Pregunta 1: {pregunta1}")
        print(f"Respuesta 1: {response1['answer'][:200]}...\n")
        
        # Segunda pregunta con contexto
        pregunta2 = "¿Puedes darme un ejemplo?"
        context = conversation_manager.get_context(conversation_id, last_n=2)
        context_str = "\n".join([msg["content"] for msg in context])
        
        response2 = await tutor.ask_question(
            question=pregunta2,
            subject="programacion",
            difficulty="basico",
            context=context_str
        )
        
        conversation_manager.add_message(conversation_id, "user", pregunta2)
        conversation_manager.add_message(conversation_id, "assistant", response2["answer"])
        
        print(f"Pregunta 2: {pregunta2}")
        print(f"Respuesta 2: {response2['answer'][:200]}...\n")
        
        # Guardar conversación
        conversation_manager.save_conversation(conversation_id)
        print(f"Conversación guardada con ID: {conversation_id}\n")
        
    finally:
        await tutor.close()


async def ejemplo_analisis_aprendizaje():
    """Ejemplo usando el analizador de aprendizaje."""
    print("=== Ejemplo: Análisis de Aprendizaje ===\n")
    
    learning_analyzer = LearningAnalyzer()
    student_id = "estudiante_001"
    
    # Simular progreso del estudiante
    learning_analyzer.update_student_profile(
        student_id=student_id,
        subject="matematicas",
        topic="algebra",
        performance=0.8,
        difficulty="intermedio"
    )
    
    learning_analyzer.update_student_profile(
        student_id=student_id,
        subject="matematicas",
        topic="geometria",
        performance=0.5,
        difficulty="intermedio"
    )
    
    learning_analyzer.update_student_profile(
        student_id=student_id,
        subject="ciencias",
        topic="biologia",
        performance=0.9,
        difficulty="avanzado"
    )
    
    # Obtener recomendaciones
    recommended_difficulty = learning_analyzer.get_recommended_difficulty(
        student_id=student_id,
        subject="matematicas",
        topic="geometria"
    )
    
    print(f"Dificultad recomendada para Geometría: {recommended_difficulty}\n")
    
    # Obtener fortalezas y debilidades
    analysis = learning_analyzer.get_strengths_and_weaknesses(student_id)
    
    print("Fortalezas:")
    for subject, topics in analysis["strengths"].items():
        print(f"  {subject}: {', '.join(topics)}")
    
    print("\nDebilidades:")
    for subject, topics in analysis["weaknesses"].items():
        print(f"  {subject}: {', '.join(topics)}")
    
    print()


async def main():
    """Ejecutar todos los ejemplos."""
    print("=" * 60)
    print("AI Tutor Educacional - Ejemplos de Uso")
    print("=" * 60)
    print()
    
    # Verificar que la API key esté configurada
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  ADVERTENCIA: OPENROUTER_API_KEY no está configurada")
        print("   Configura la variable de entorno antes de ejecutar los ejemplos\n")
        return
    
    try:
        await ejemplo_pregunta_simple()
        print("\n" + "-" * 60 + "\n")
        
        await ejemplo_explicacion_concepto()
        print("\n" + "-" * 60 + "\n")
        
        await ejemplo_generar_ejercicios()
        print("\n" + "-" * 60 + "\n")
        
        await ejemplo_con_conversacion()
        print("\n" + "-" * 60 + "\n")
        
        await ejemplo_analisis_aprendizaje()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())






