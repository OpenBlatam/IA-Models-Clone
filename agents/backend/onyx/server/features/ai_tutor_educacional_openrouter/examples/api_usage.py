"""
Ejemplo de uso de la API REST del AI Tutor Educacional.
"""

import requests
import json

BASE_URL = "http://localhost:8000/api/tutor"


def ejemplo_hacer_pregunta():
    """Ejemplo de hacer una pregunta a través de la API."""
    print("=== Ejemplo: Hacer Pregunta vía API ===\n")
    
    url = f"{BASE_URL}/ask"
    payload = {
        "question": "¿Qué es la fotosíntesis?",
        "subject": "ciencias",
        "difficulty": "intermedio",
        "conversation_id": "estudiante_001"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data["success"]:
            print(f"Pregunta: {payload['question']}")
            print(f"\nRespuesta:\n{data['data']['answer']}\n")
            print(f"Modelo: {data['data']['model']}")
        else:
            print(f"Error: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la petición: {e}")


def ejemplo_explicar_concepto():
    """Ejemplo de explicar un concepto a través de la API."""
    print("=== Ejemplo: Explicar Concepto vía API ===\n")
    
    url = f"{BASE_URL}/explain"
    payload = {
        "concept": "derivadas",
        "subject": "matematicas",
        "difficulty": "avanzado"
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data["success"]:
            print(f"Concepto: {payload['concept']}")
            print(f"\nExplicación:\n{data['data']['answer']}\n")
        else:
            print(f"Error: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la petición: {e}")


def ejemplo_generar_ejercicios():
    """Ejemplo de generar ejercicios a través de la API."""
    print("=== Ejemplo: Generar Ejercicios vía API ===\n")
    
    url = f"{BASE_URL}/exercises"
    payload = {
        "topic": "ecuaciones cuadráticas",
        "subject": "matematicas",
        "difficulty": "intermedio",
        "num_exercises": 3
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if data["success"]:
            print(f"Tema: {payload['topic']}")
            print(f"\nEjercicios:\n{data['data']['answer']}\n")
        else:
            print(f"Error: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la petición: {e}")


def ejemplo_obtener_conversacion():
    """Ejemplo de obtener el historial de conversación."""
    print("=== Ejemplo: Obtener Conversación ===\n")
    
    conversation_id = "estudiante_001"
    url = f"{BASE_URL}/conversation/{conversation_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data["success"]:
            messages = data["data"]["messages"]
            print(f"Conversación ID: {conversation_id}")
            print(f"Total de mensajes: {len(messages)}\n")
            
            for i, msg in enumerate(messages, 1):
                role = msg["role"]
                content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
                print(f"{i}. [{role.upper()}]: {content}\n")
        else:
            print(f"Error: {data}")
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la petición: {e}")


def ejemplo_health_check():
    """Ejemplo de verificar el estado del servicio."""
    print("=== Ejemplo: Health Check ===\n")
    
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        print(f"Estado: {data['status']}")
        print(f"Servicio: {data['service']}\n")
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la petición: {e}")


def main():
    """Ejecutar todos los ejemplos de API."""
    print("=" * 60)
    print("AI Tutor Educacional - Ejemplos de API REST")
    print("=" * 60)
    print()
    print("⚠️  Asegúrate de que el servidor esté corriendo en http://localhost:8000")
    print()
    
    try:
        ejemplo_health_check()
        print("\n" + "-" * 60 + "\n")
        
        ejemplo_hacer_pregunta()
        print("\n" + "-" * 60 + "\n")
        
        ejemplo_explicar_concepto()
        print("\n" + "-" * 60 + "\n")
        
        ejemplo_generar_ejercicios()
        print("\n" + "-" * 60 + "\n")
        
        ejemplo_obtener_conversacion()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()






