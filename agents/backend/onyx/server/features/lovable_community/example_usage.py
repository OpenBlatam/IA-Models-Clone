"""
Ejemplo de uso de la API de Lovable Community

Este script muestra cómo usar los diferentes endpoints de la API.
"""

import requests
import json

BASE_URL = "http://localhost:8007/lovable/community"

def publish_chat():
    """Ejemplo: Publicar un chat"""
    data = {
        "title": "Mi primera conversación sobre IA",
        "description": "Una conversación interesante sobre inteligencia artificial",
        "chat_content": json.dumps({
            "messages": [
                {"role": "user", "content": "¿Qué es la IA?"},
                {"role": "assistant", "content": "La IA es..."}
            ]
        }),
        "tags": ["ai", "conversation", "learning"],
        "is_public": True
    }
    
    response = requests.post(f"{BASE_URL}/publish", json=data)
    print("Publicar chat:")
    print(json.dumps(response.json(), indent=2))
    return response.json()["id"]


def list_chats():
    """Ejemplo: Listar chats"""
    params = {
        "page": 1,
        "page_size": 10,
        "sort_by": "score",
        "order": "desc"
    }
    
    response = requests.get(f"{BASE_URL}/chats", params=params)
    print("\nListar chats:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def get_top_chats():
    """Ejemplo: Obtener chats más populares"""
    params = {"limit": 10}
    
    response = requests.get(f"{BASE_URL}/top", params=params)
    print("\nTop chats:")
    print(json.dumps(response.json(), indent=2))
    return response.json()


def remix_chat(original_chat_id: str):
    """Ejemplo: Remixar un chat"""
    data = {
        "original_chat_id": original_chat_id,
        "title": "Mi remix mejorado",
        "description": "Versión mejorada del chat original",
        "chat_content": json.dumps({
            "messages": [
                {"role": "user", "content": "¿Qué es la IA? (versión mejorada)"},
                {"role": "assistant", "content": "La IA es una tecnología..."}
            ]
        }),
        "tags": ["remix", "improved", "ai"]
    }
    
    response = requests.post(f"{BASE_URL}/chats/{original_chat_id}/remix", json=data)
    print("\nRemixar chat:")
    print(json.dumps(response.json(), indent=2))
    return response.json()["remix_chat_id"]


def vote_chat(chat_id: str):
    """Ejemplo: Votar un chat"""
    data = {
        "chat_id": chat_id,
        "vote_type": "upvote"
    }
    
    response = requests.post(f"{BASE_URL}/chats/{chat_id}/vote", json=data)
    print("\nVotar chat:")
    print(json.dumps(response.json(), indent=2))


def search_chats():
    """Ejemplo: Buscar chats"""
    params = {
        "query": "IA",
        "tags": "ai,conversation",
        "sort_by": "score",
        "order": "desc",
        "page": 1,
        "page_size": 10
    }
    
    response = requests.get(f"{BASE_URL}/search", params=params)
    print("\nBuscar chats:")
    print(json.dumps(response.json(), indent=2))


def get_chat_stats(chat_id: str):
    """Ejemplo: Obtener estadísticas de un chat"""
    response = requests.get(f"{BASE_URL}/chats/{chat_id}/stats")
    print("\nEstadísticas del chat:")
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    print("Ejemplos de uso de la API de Lovable Community")
    print("=" * 50)
    
    # Asegúrate de que el servidor esté corriendo en http://localhost:8007
    
    try:
        # Publicar un chat
        chat_id = publish_chat()
        
        # Listar chats
        list_chats()
        
        # Obtener top chats
        get_top_chats()
        
        # Votar el chat
        vote_chat(chat_id)
        
        # Remixar el chat
        remix_id = remix_chat(chat_id)
        
        # Buscar chats
        search_chats()
        
        # Obtener estadísticas
        get_chat_stats(chat_id)
        
    except requests.exceptions.ConnectionError:
        print("\nError: No se pudo conectar al servidor.")
        print("Asegúrate de que el servidor esté corriendo:")
        print("  uvicorn features.lovable_community.main:app --host 0.0.0.0 --port 8007")
    except Exception as e:
        print(f"\nError: {e}")

