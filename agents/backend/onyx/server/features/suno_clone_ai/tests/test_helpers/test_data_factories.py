"""
Factories para crear datos de prueba
"""

import pytest
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import random


def generate_uuid() -> str:
    """Genera un UUID aleatorio"""
    return str(uuid.uuid4())


def generate_timestamp(offset_days: int = 0) -> str:
    """
    Genera un timestamp ISO.
    
    Args:
        offset_days: Días de offset desde ahora
        
    Returns:
        Timestamp en formato ISO
    """
    dt = datetime.now() + timedelta(days=offset_days)
    return dt.isoformat()


def generate_song_data(
    song_id: Optional[str] = None,
    user_id: Optional[str] = None,
    prompt: Optional[str] = None,
    status: str = "completed",
    duration: float = 30.0
) -> Dict[str, Any]:
    """
    Genera datos de canción de prueba.
    
    Args:
        song_id: ID de la canción (se genera si no se proporciona)
        user_id: ID del usuario (se genera si no se proporciona)
        prompt: Prompt de la canción
        status: Estado de la canción
        duration: Duración en segundos
        
    Returns:
        Diccionario con datos de canción
    """
    return {
        "song_id": song_id or generate_uuid(),
        "user_id": user_id or generate_uuid(),
        "prompt": prompt or f"Test song {random.randint(1, 1000)}",
        "status": status,
        "file_path": f"/tmp/{song_id or generate_uuid()}.wav",
        "duration": duration,
        "created_at": generate_timestamp(),
        "genre": random.choice(["pop", "rock", "jazz", "electronic"]),
        "mood": random.choice(["happy", "sad", "energetic", "calm"]),
        "bpm": random.randint(60, 180),
        "rating": round(random.uniform(0, 5), 1)
    }


def generate_playlist_data(
    playlist_id: Optional[str] = None,
    user_id: Optional[str] = None,
    name: Optional[str] = None,
    song_count: int = 5
) -> Dict[str, Any]:
    """
    Genera datos de playlist de prueba.
    
    Args:
        playlist_id: ID de la playlist (se genera si no se proporciona)
        user_id: ID del usuario (se genera si no se proporciona)
        name: Nombre de la playlist
        song_count: Número de canciones
        
    Returns:
        Diccionario con datos de playlist
    """
    return {
        "playlist_id": playlist_id or generate_uuid(),
        "user_id": user_id or generate_uuid(),
        "name": name or f"Test Playlist {random.randint(1, 1000)}",
        "songs": [generate_uuid() for _ in range(song_count)],
        "created_at": generate_timestamp(),
        "description": f"Test playlist description",
        "is_public": random.choice([True, False])
    }


def generate_user_data(
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    role: str = "user"
) -> Dict[str, Any]:
    """
    Genera datos de usuario de prueba.
    
    Args:
        user_id: ID del usuario (se genera si no se proporciona)
        email: Email del usuario
        role: Rol del usuario
        
    Returns:
        Diccionario con datos de usuario
    """
    uid = user_id or generate_uuid()
    return {
        "user_id": uid,
        "email": email or f"user_{uid[:8]}@example.com",
        "role": role,
        "sub": uid,
        "created_at": generate_timestamp(),
        "is_active": True
    }


def generate_analytics_event(
    event_type: str = "song_generated",
    user_id: Optional[str] = None,
    song_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Genera un evento de analytics de prueba.
    
    Args:
        event_type: Tipo de evento
        user_id: ID del usuario
        song_id: ID de la canción
        
    Returns:
        Diccionario con datos de evento
    """
    return {
        "event_id": generate_uuid(),
        "event_type": event_type,
        "user_id": user_id or generate_uuid(),
        "song_id": song_id or generate_uuid(),
        "timestamp": generate_timestamp(),
        "metadata": {
            "duration": random.uniform(10, 300),
            "genre": random.choice(["pop", "rock", "jazz"])
        }
    }


def generate_batch_data(
    batch_id: Optional[str] = None,
    item_count: int = 10,
    status: str = "pending"
) -> Dict[str, Any]:
    """
    Genera datos de batch de prueba.
    
    Args:
        batch_id: ID del batch (se genera si no se proporciona)
        item_count: Número de items
        status: Estado del batch
        
    Returns:
        Diccionario con datos de batch
    """
    return {
        "batch_id": batch_id or generate_uuid(),
        "status": status,
        "items_count": item_count,
        "completed_count": 0 if status == "pending" else item_count,
        "failed_count": 0,
        "created_at": generate_timestamp(),
        "items": [generate_uuid() for _ in range(item_count)]
    }


def generate_multiple_songs(count: int = 10) -> List[Dict[str, Any]]:
    """
    Genera múltiples canciones de prueba.
    
    Args:
        count: Número de canciones a generar
        
    Returns:
        Lista de diccionarios con datos de canciones
    """
    return [generate_song_data() for _ in range(count)]


def generate_multiple_playlists(count: int = 5, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Genera múltiples playlists de prueba.
    
    Args:
        count: Número de playlists a generar
        user_id: ID del usuario (compartido para todas)
        
    Returns:
        Lista de diccionarios con datos de playlists
    """
    uid = user_id or generate_uuid()
    return [generate_playlist_data(user_id=uid) for _ in range(count)]


@pytest.fixture
def song_data_factory():
    """Factory para generar datos de canción"""
    return generate_song_data


@pytest.fixture
def playlist_data_factory():
    """Factory para generar datos de playlist"""
    return generate_playlist_data


@pytest.fixture
def user_data_factory():
    """Factory para generar datos de usuario"""
    return generate_user_data


@pytest.fixture
def analytics_event_factory():
    """Factory para generar eventos de analytics"""
    return generate_analytics_event


@pytest.fixture
def batch_data_factory():
    """Factory para generar datos de batch"""
    return generate_batch_data

