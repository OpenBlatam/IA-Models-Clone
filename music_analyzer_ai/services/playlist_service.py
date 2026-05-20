"""
Servicio para gestionar playlists personalizadas
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PlaylistService:
    """Servicio para gestionar playlists"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/playlists")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.playlists_file = self.storage_path / "playlists.json"
        self.logger = logger
        self._load_playlists()
    
    def _load_playlists(self) -> None:
        """Carga playlists desde archivo"""
        if self.playlists_file.exists():
            try:
                with open(self.playlists_file, "r", encoding="utf-8") as f:
                    self.playlists = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading playlists: {e}")
                self.playlists = {}
        else:
            self.playlists = {}
    
    def _save_playlists(self) -> None:
        """Guarda playlists en archivo"""
        try:
            with open(self.playlists_file, "w", encoding="utf-8") as f:
                json.dump(self.playlists, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error saving playlists: {e}")
    
    def create_playlist(self, user_id: str, name: str, 
                       description: Optional[str] = None,
                       is_public: bool = False) -> str:
        """Crea una nueva playlist"""
        playlist_id = f"playlist_{datetime.now().timestamp()}_{len(self.playlists)}"
        
        self.playlists[playlist_id] = {
            "playlist_id": playlist_id,
            "user_id": user_id,
            "name": name,
            "description": description,
            "is_public": is_public,
            "tracks": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        self._save_playlists()
        self.logger.info(f"Playlist created: {playlist_id}")
        return playlist_id
    
    def add_track_to_playlist(self, playlist_id: str, track_id: str,
                              track_name: str, artists: List[str]) -> bool:
        """Agrega una canción a una playlist"""
        if playlist_id not in self.playlists:
            return False
        
        # Verificar si ya existe
        existing = next(
            (t for t in self.playlists[playlist_id]["tracks"] if t.get("track_id") == track_id),
            None
        )
        
        if existing:
            return False
        
        track = {
            "track_id": track_id,
            "track_name": track_name,
            "artists": artists,
            "added_at": datetime.now().isoformat()
        }
        
        self.playlists[playlist_id]["tracks"].append(track)
        self.playlists[playlist_id]["updated_at"] = datetime.now().isoformat()
        self._save_playlists()
        
        return True
    
    def remove_track_from_playlist(self, playlist_id: str, track_id: str) -> bool:
        """Elimina una canción de una playlist"""
        if playlist_id not in self.playlists:
            return False
        
        original_count = len(self.playlists[playlist_id]["tracks"])
        self.playlists[playlist_id]["tracks"] = [
            t for t in self.playlists[playlist_id]["tracks"]
            if t.get("track_id") != track_id
        ]
        
        if len(self.playlists[playlist_id]["tracks"]) < original_count:
            self.playlists[playlist_id]["updated_at"] = datetime.now().isoformat()
            self._save_playlists()
            return True
        
        return False
    
    def get_playlist(self, playlist_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene una playlist"""
        return self.playlists.get(playlist_id)
    
    def get_user_playlists(self, user_id: str) -> List[Dict[str, Any]]:
        """Obtiene todas las playlists de un usuario"""
        return [
            {
                "playlist_id": p["playlist_id"],
                "name": p["name"],
                "description": p.get("description"),
                "is_public": p.get("is_public", False),
                "track_count": len(p.get("tracks", [])),
                "created_at": p.get("created_at"),
                "updated_at": p.get("updated_at")
            }
            for p in self.playlists.values()
            if p.get("user_id") == user_id
        ]
    
    def delete_playlist(self, playlist_id: str, user_id: str) -> bool:
        """Elimina una playlist"""
        playlist = self.playlists.get(playlist_id)
        
        if not playlist:
            return False
        
        if playlist.get("user_id") != user_id:
            return False
        
        del self.playlists[playlist_id]
        self._save_playlists()
        self.logger.info(f"Playlist deleted: {playlist_id}")
        return True
    
    def get_public_playlists(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Obtiene playlists públicas"""
        public = [
            {
                "playlist_id": p["playlist_id"],
                "name": p["name"],
                "description": p.get("description"),
                "user_id": p.get("user_id"),
                "track_count": len(p.get("tracks", [])),
                "created_at": p.get("created_at")
            }
            for p in self.playlists.values()
            if p.get("is_public", False)
        ]
        
        # Ordenar por fecha de creación (más recientes primero)
        public.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return public[:limit]

