"""
Servicio para gestionar favoritos/guardados
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FavoritesService:
    """Servicio para gestionar canciones favoritas"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/favorites")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.favorites_file = self.storage_path / "favorites.json"
        self.logger = logger
        self._load_favorites()
    
    def _load_favorites(self) -> None:
        """Carga los favoritos desde el archivo"""
        if self.favorites_file.exists():
            try:
                with open(self.favorites_file, "r", encoding="utf-8") as f:
                    self.favorites = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading favorites: {e}")
                self.favorites = {}
        else:
            self.favorites = {}
    
    def _save_favorites(self) -> None:
        """Guarda los favoritos en el archivo"""
        try:
            with open(self.favorites_file, "w", encoding="utf-8") as f:
                json.dump(self.favorites, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error saving favorites: {e}")
    
    def add_favorite(self, user_id: str, track_id: str, 
                    track_name: str, artists: List[str],
                    notes: Optional[str] = None) -> bool:
        """Agrega una canción a favoritos"""
        if user_id not in self.favorites:
            self.favorites[user_id] = []
        
        # Verificar si ya existe
        existing = next(
            (f for f in self.favorites[user_id] if f.get("track_id") == track_id),
            None
        )
        
        if existing:
            return False  # Ya existe
        
        favorite = {
            "track_id": track_id,
            "track_name": track_name,
            "artists": artists,
            "notes": notes,
            "added_at": datetime.now().isoformat()
        }
        
        self.favorites[user_id].append(favorite)
        self._save_favorites()
        
        self.logger.info(f"Favorite added: {track_id} for user {user_id}")
        return True
    
    def remove_favorite(self, user_id: str, track_id: str) -> bool:
        """Elimina una canción de favoritos"""
        if user_id not in self.favorites:
            return False
        
        original_count = len(self.favorites[user_id])
        self.favorites[user_id] = [
            f for f in self.favorites[user_id]
            if f.get("track_id") != track_id
        ]
        
        if len(self.favorites[user_id]) < original_count:
            self._save_favorites()
            self.logger.info(f"Favorite removed: {track_id} for user {user_id}")
            return True
        
        return False
    
    def get_favorites(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene los favoritos de un usuario"""
        return self.favorites.get(user_id, [])
    
    def is_favorite(self, user_id: str, track_id: str) -> bool:
        """Verifica si una canción está en favoritos"""
        if user_id not in self.favorites:
            return False
        
        return any(f.get("track_id") == track_id for f in self.favorites[user_id])
    
    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Obtiene estadísticas de favoritos"""
        favorites = self.get_favorites(user_id)
        
        return {
            "total_favorites": len(favorites),
            "oldest_favorite": favorites[-1].get("added_at") if favorites else None,
            "newest_favorite": favorites[0].get("added_at") if favorites else None
        }

