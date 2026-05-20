#!/usr/bin/env python3
"""
Ejemplo de uso de la API REST de Bulk Chat
==========================================

Este script muestra cómo usar la API REST del sistema.
"""

import requests
import json
import time
from typing import Optional

BASE_URL = "http://localhost:8006"


def create_session(initial_message: str, auto_continue: bool = True) -> Optional[str]:
    """Crear una nueva sesión de chat."""
    url = f"{BASE_URL}/api/v1/chat/sessions"
    data = {
        "initial_message": initial_message,
        "auto_continue": auto_continue
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        result = response.json()
        session_id = result.get("session_id")
        print(f"✅ Sesión creada: {session_id}")
        return session_id
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al crear sesión: {e}")
        return None


def get_session(session_id: str):
    """Obtener información de una sesión."""
    url = f"{BASE_URL}/api/v1/chat/sessions/{session_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al obtener sesión: {e}")
        return None


def get_messages(session_id: str):
    """Obtener mensajes de una sesión."""
    url = f"{BASE_URL}/api/v1/chat/sessions/{session_id}/messages"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al obtener mensajes: {e}")
        return None


def send_message(session_id: str, message: str):
    """Enviar un mensaje a una sesión."""
    url = f"{BASE_URL}/api/v1/chat/sessions/{session_id}/messages"
    data = {"message": message}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al enviar mensaje: {e}")
        return None


def pause_session(session_id: str, reason: str = "Usuario pausó"):
    """Pausar una sesión."""
    url = f"{BASE_URL}/api/v1/chat/sessions/{session_id}/pause"
    data = {"action": "pause", "reason": reason}
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print(f"✅ Sesión pausada: {reason}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al pausar sesión: {e}")
        return None


def resume_session(session_id: str):
    """Reanudar una sesión."""
    url = f"{BASE_URL}/api/v1/chat/sessions/{session_id}/resume"
    
    try:
        response = requests.post(url)
        response.raise_for_status()
        print("✅ Sesión reanudada")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al reanudar sesión: {e}")
        return None


def stop_session(session_id: str):
    """Detener una sesión."""
    url = f"{BASE_URL}/api/v1/chat/sessions/{session_id}/stop"
    
    try:
        response = requests.post(url)
        response.raise_for_status()
        print("✅ Sesión detenida")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al detener sesión: {e}")
        return None


def check_health():
    """Verificar que el servidor esté corriendo."""
    url = f"{BASE_URL}/health"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        print("✅ Servidor está corriendo")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Servidor no está disponible: {e}")
        print(f"   Asegúrate de que el servidor esté corriendo en {BASE_URL}")
        return False


def main():
    """Ejemplo principal."""
    print("=" * 60)
    print("🚀 Ejemplo de uso de la API REST - Bulk Chat")
    print("=" * 60)
    print()
    
    # Verificar que el servidor esté corriendo
    if not check_health():
        print("\n💡 Para iniciar el servidor:")
        print("   python -m bulk_chat.main --llm-provider mock")
        return
    
    print()
    
    # Crear sesión
    session_id = create_session(
        initial_message="Hola, explícame sobre Python",
        auto_continue=True
    )
    
    if not session_id:
        return
    
    print()
    
    # Esperar un poco para que genere respuestas
    print("⏳ Esperando respuestas automáticas (5 segundos)...")
    time.sleep(5)
    
    # Obtener mensajes
    print("\n📨 Mensajes generados:")
    messages = get_messages(session_id)
    if messages:
        for msg in messages.get("messages", [])[:5]:  # Mostrar primeros 5
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:80]
            emoji = "👤" if role == "user" else "🤖"
            print(f"  {emoji} [{role}]: {content}...")
    
    # Pausar
    print("\n⏸️  Pausando sesión...")
    pause_session(session_id, "Ejemplo de pausa")
    
    time.sleep(2)
    
    # Reanudar
    print("\n▶️  Reanudando sesión...")
    resume_session(session_id)
    
    time.sleep(3)
    
    # Obtener estado final
    print("\n📊 Estado final de la sesión:")
    session = get_session(session_id)
    if session:
        print(f"  ID: {session.get('session_id')}")
        print(f"  Estado: {session.get('state')}")
        print(f"  Pausado: {session.get('is_paused')}")
        print(f"  Mensajes: {session.get('message_count')}")
    
    # Detener
    print("\n🛑 Deteniendo sesión...")
    stop_session(session_id)
    
    print("\n✨ Ejemplo completado!")
    print(f"\n💡 Puedes ver más información en: {BASE_URL}/docs")


if __name__ == "__main__":
    main()
















