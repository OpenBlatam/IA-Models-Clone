"""
Servicio para gestionar historial de análisis
"""

import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class HistoryService:
    """Servicio para gestionar historial de análisis musicales"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./data/history")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_file = self.storage_path / "history.json"
        self.logger = logger
        self._load_history()
    
    def _load_history(self) -> None:
        """Carga el historial desde el archivo"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    self.history = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading history: {e}")
                self.history = []
        else:
            self.history = []
    
    def _save_history(self) -> None:
        """Guarda el historial en el archivo"""
        try:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")
    
    def add_analysis(self, track_id: str, track_name: str, 
                    artists: List[str], analysis: Dict[str, Any],
                    user_id: Optional[str] = None) -> str:
        """Agrega un análisis al historial"""
        entry = {
            "id": f"{track_id}_{datetime.now().timestamp()}",
            "track_id": track_id,
            "track_name": track_name,
            "artists": artists,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "analysis_summary": {
                "key_signature": analysis.get("musical_analysis", {}).get("key_signature"),
                "tempo": analysis.get("musical_analysis", {}).get("tempo", {}).get("bpm"),
                "difficulty": analysis.get("coaching", {}).get("overview", {}).get("difficulty_level") if "coaching" in analysis else None
            }
        }
        
        self.history.append(entry)
        self._save_history()
        
        self.logger.info(f"Analysis added to history: {track_id}")
        return entry["id"]
    
    def get_history(self, user_id: Optional[str] = None, 
                   limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene el historial de análisis"""
        history = self.history
        
        if user_id:
            history = [h for h in history if h.get("user_id") == user_id]
        
        # Ordenar por timestamp (más reciente primero)
        history.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return history[:limit]
    
    def get_analysis_by_track_id(self, track_id: str,
                                user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Obtiene el análisis más reciente de un track"""
        history = self.history
        
        if user_id:
            history = [h for h in history if h.get("user_id") == user_id]
        
        # Buscar por track_id
        matches = [h for h in history if h.get("track_id") == track_id]
        
        if matches:
            # Retornar el más reciente
            matches.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return matches[0]
        
        return None
    
    def delete_analysis(self, analysis_id: str,
                      user_id: Optional[str] = None) -> bool:
        """Elimina un análisis del historial"""
        for i, entry in enumerate(self.history):
            if entry.get("id") == analysis_id:
                if user_id and entry.get("user_id") != user_id:
                    return False
                self.history.pop(i)
                self._save_history()
                self.logger.info(f"Analysis deleted from history: {analysis_id}")
                return True
        return False
    
    def get_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Obtiene estadísticas del historial"""
        history = self.history
        
        if user_id:
            history = [h for h in history if h.get("user_id") == user_id]
        
        if not history:
            return {
                "total_analyses": 0,
                "unique_tracks": 0,
                "most_analyzed_genre": None
            }
        
        unique_tracks = len(set(h.get("track_id") for h in history))
        
        # Contar géneros (si están disponibles)
        genres = []
        for h in history:
            genre = h.get("analysis_summary", {}).get("genre")
            if genre:
                genres.append(genre)
        
        most_common_genre = max(set(genres), key=genres.count) if genres else None
        
        return {
            "total_analyses": len(history),
            "unique_tracks": unique_tracks,
            "most_analyzed_genre": most_common_genre,
            "first_analysis": history[-1].get("timestamp") if history else None,
            "last_analysis": history[0].get("timestamp") if history else None
        }

