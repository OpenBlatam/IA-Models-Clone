"""
Servicio de análisis de colaboraciones entre artistas
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)


class CollaborationAnalyzer:
    """Analiza colaboraciones y relaciones entre artistas"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.logger = logger
    
    def analyze_track_collaborations(self, track_id: str) -> Dict[str, Any]:
        """Analiza las colaboraciones en un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            artists = track_info.get("artists", [])
            
            if len(artists) < 2:
                return {
                    "track_id": track_id,
                    "is_collaboration": False,
                    "artist_count": len(artists),
                    "message": "Este track no es una colaboración"
                }
            
            # Obtener información de cada artista
            artists_data = []
            for artist in artists:
                artist_id = artist.get("id")
                artist_name = artist.get("name", "Unknown")
                
                # Buscar otros tracks del artista para análisis
                try:
                    artist_tracks = self.spotify.search_tracks(f"artist:{artist_name}", limit=5)
                    
                    # Analizar género y emoción promedio
                    genres = []
                    emotions = []
                    
                    for track in artist_tracks[:3]:
                        try:
                            features = self.spotify.get_track_audio_features(track.get("id"))
                            if features:
                                genre = self.genre_detector.detect_genre(features)
                                emotion = self.emotion_analyzer.analyze_emotions(features)
                                genres.append(genre.get("primary_genre", "Unknown"))
                                emotions.append(emotion.get("primary_emotion", "Unknown"))
                        except:
                            continue
                    
                    # Género más común
                    genre_counter = Counter(genres)
                    primary_genre = genre_counter.most_common(1)[0][0] if genre_counter else "Unknown"
                    
                    # Emoción más común
                    emotion_counter = Counter(emotions)
                    primary_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else "Unknown"
                    
                    artists_data.append({
                        "artist_id": artist_id,
                        "artist_name": artist_name,
                        "primary_genre": primary_genre,
                        "primary_emotion": primary_emotion,
                        "tracks_analyzed": len(artist_tracks)
                    })
                except:
                    artists_data.append({
                        "artist_id": artist_id,
                        "artist_name": artist_name,
                        "primary_genre": "Unknown",
                        "primary_emotion": "Unknown"
                    })
            
            # Analizar compatibilidad
            compatibility = self._analyze_artist_compatibility(artists_data)
            
            return {
                "track_id": track_id,
                "track_name": track_info.get("name", "Unknown"),
                "is_collaboration": True,
                "artist_count": len(artists),
                "artists": artists_data,
                "compatibility": compatibility,
                "collaboration_type": self._identify_collaboration_type(artists_data)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing collaborations: {e}")
            return {"error": str(e)}
    
    def analyze_artist_network(self, artist_ids: List[str]) -> Dict[str, Any]:
        """Analiza la red de colaboraciones de artistas"""
        try:
            if len(artist_ids) > 10:
                artist_ids = artist_ids[:10]  # Limitar a 10
            
            network = defaultdict(list)
            artist_info = {}
            
            for artist_id in artist_ids:
                try:
                    # Buscar tracks del artista
                    tracks = self.spotify.search_tracks(f"artist:{artist_id}", limit=20)
                    
                    collaborators = set()
                    genres = []
                    
                    for track in tracks:
                        track_artists = track.get("artists", [])
                        for artist in track_artists:
                            collab_id = artist.get("id")
                            if collab_id != artist_id:
                                collaborators.add(collab_id)
                        
                        # Analizar género
                        try:
                            features = self.spotify.get_track_audio_features(track.get("id"))
                            if features:
                                genre = self.genre_detector.detect_genre(features)
                                genres.append(genre.get("primary_genre", "Unknown"))
                        except:
                            continue
                    
                    # Género principal
                    genre_counter = Counter(genres)
                    primary_genre = genre_counter.most_common(1)[0][0] if genre_counter else "Unknown"
                    
                    artist_info[artist_id] = {
                        "primary_genre": primary_genre,
                        "collaboration_count": len(collaborators),
                        "tracks_analyzed": len(tracks)
                    }
                    
                    network[artist_id] = list(collaborators)
                except:
                    continue
            
            # Encontrar conexiones entre artistas
            connections = []
            artist_list = list(network.keys())
            
            for i, artist1 in enumerate(artist_list):
                for artist2 in artist_list[i+1:]:
                    if artist2 in network[artist1] or artist1 in network[artist2]:
                        connections.append({
                            "artist1": artist1,
                            "artist2": artist2,
                            "type": "Direct Collaboration"
                        })
            
            return {
                "artists_analyzed": len(artist_info),
                "network": dict(network),
                "connections": connections,
                "artist_info": artist_info,
                "most_connected": max(artist_info.items(), key=lambda x: x[1]["collaboration_count"])[0] if artist_info else None
            }
        except Exception as e:
            self.logger.error(f"Error analyzing artist network: {e}")
            return {"error": str(e)}
    
    def compare_versions(self, track_ids: List[str]) -> Dict[str, Any]:
        """Compara diferentes versiones de una canción (original, cover, remix)"""
        try:
            if len(track_ids) < 2:
                return {"error": "Se necesitan al menos 2 tracks para comparar"}
            
            if len(track_ids) > 10:
                track_ids = track_ids[:10]
            
            versions_data = []
            
            for track_id in track_ids:
                try:
                    track_info = self.spotify.get_track(track_id)
                    audio_features = self.spotify.get_track_audio_features(track_id)
                    
                    genre = self.genre_detector.detect_genre(audio_features)
                    emotion = self.emotion_analyzer.analyze_emotions(audio_features)
                    
                    versions_data.append({
                        "track_id": track_id,
                        "track_name": track_info.get("name", "Unknown"),
                        "artists": [a.get("name") for a in track_info.get("artists", [])],
                        "duration_ms": track_info.get("duration_ms", 0),
                        "popularity": track_info.get("popularity", 0),
                        "genre": genre.get("primary_genre", "Unknown"),
                        "emotion": emotion.get("primary_emotion", "Unknown"),
                        "tempo": audio_features.get("tempo", 120),
                        "energy": audio_features.get("energy", 0.5),
                        "danceability": audio_features.get("danceability", 0.5),
                        "key": audio_features.get("key", -1),
                        "mode": audio_features.get("mode", 0)
                    })
                except:
                    continue
            
            if len(versions_data) < 2:
                return {"error": "No se pudieron analizar suficientes tracks"}
            
            # Análisis comparativo
            similarities = []
            differences = []
            
            # Comparar con el primero (asumido como original)
            original = versions_data[0]
            
            for version in versions_data[1:]:
                similarity_score = self._calculate_version_similarity(original, version)
                
                similarities.append({
                    "track_id": version["track_id"],
                    "track_name": version["track_name"],
                    "similarity": round(similarity_score, 3)
                })
                
                # Identificar diferencias
                diff = {
                    "track_id": version["track_id"],
                    "differences": []
                }
                
                if abs(original["tempo"] - version["tempo"]) > 10:
                    diff["differences"].append(f"Tempo: {original['tempo']:.1f} vs {version['tempo']:.1f} BPM")
                
                if abs(original["energy"] - version["energy"]) > 0.2:
                    diff["differences"].append(f"Energy: {original['energy']:.2f} vs {version['energy']:.2f}")
                
                if original["genre"] != version["genre"]:
                    diff["differences"].append(f"Genre: {original['genre']} vs {version['genre']}")
                
                if original["key"] != version["key"]:
                    diff["differences"].append(f"Key: {original['key']} vs {version['key']}")
                
                differences.append(diff)
            
            # Identificar tipo de versión
            version_types = []
            for version in versions_data[1:]:
                version_type = self._identify_version_type(original, version)
                version_types.append({
                    "track_id": version["track_id"],
                    "type": version_type
                })
            
            return {
                "original": original,
                "versions": versions_data[1:],
                "similarities": similarities,
                "differences": differences,
                "version_types": version_types,
                "most_similar": max(similarities, key=lambda x: x["similarity"]) if similarities else None,
                "most_different": min(similarities, key=lambda x: x["similarity"]) if similarities else None
            }
        except Exception as e:
            self.logger.error(f"Error comparing versions: {e}")
            return {"error": str(e)}
    
    def _analyze_artist_compatibility(self, artists_data: List[Dict]) -> Dict[str, Any]:
        """Analiza la compatibilidad entre artistas"""
        genres = [a.get("primary_genre", "Unknown") for a in artists_data]
        emotions = [a.get("primary_emotion", "Unknown") for a in artists_data]
        
        # Compatibilidad de género
        unique_genres = set(genres)
        genre_compatibility = "High" if len(unique_genres) <= 2 else "Medium" if len(unique_genres) <= 3 else "Low"
        
        # Compatibilidad de emoción
        unique_emotions = set(emotions)
        emotion_compatibility = "High" if len(unique_emotions) <= 2 else "Medium" if len(unique_emotions) <= 3 else "Low"
        
        # Score general
        if genre_compatibility == "High" and emotion_compatibility == "High":
            overall = "High"
            score = 0.8
        elif genre_compatibility == "High" or emotion_compatibility == "High":
            overall = "Medium-High"
            score = 0.6
        elif genre_compatibility == "Medium" and emotion_compatibility == "Medium":
            overall = "Medium"
            score = 0.4
        else:
            overall = "Low"
            score = 0.2
        
        return {
            "overall": overall,
            "score": score,
            "genre_compatibility": genre_compatibility,
            "emotion_compatibility": emotion_compatibility,
            "genres": genres,
            "emotions": emotions
        }
    
    def _identify_collaboration_type(self, artists_data: List[Dict]) -> str:
        """Identifica el tipo de colaboración"""
        genres = [a.get("primary_genre", "Unknown") for a in artists_data]
        unique_genres = set(genres)
        
        if len(unique_genres) == 1:
            return "Same Genre Collaboration"
        elif len(unique_genres) == 2:
            return "Cross-Genre Collaboration"
        else:
            return "Multi-Genre Collaboration"
    
    def _calculate_version_similarity(self, original: Dict, version: Dict) -> float:
        """Calcula similitud entre versiones"""
        # Factores de similitud
        tempo_diff = abs(original["tempo"] - version["tempo"]) / 200  # Normalizar
        energy_diff = abs(original["energy"] - version["energy"])
        danceability_diff = abs(original["danceability"] - version["danceability"])
        key_same = 1.0 if original["key"] == version["key"] else 0.0
        mode_same = 1.0 if original["mode"] == version["mode"] else 0.0
        
        similarity = (
            (1.0 - tempo_diff) * 0.3 +
            (1.0 - energy_diff) * 0.2 +
            (1.0 - danceability_diff) * 0.2 +
            key_same * 0.15 +
            mode_same * 0.15
        )
        
        return max(0.0, min(1.0, similarity))
    
    def _identify_version_type(self, original: Dict, version: Dict) -> str:
        """Identifica el tipo de versión (cover, remix, etc.)"""
        tempo_diff = abs(original["tempo"] - version["tempo"])
        energy_diff = abs(original["energy"] - version["energy"])
        genre_diff = original["genre"] != version["genre"]
        
        if tempo_diff > 20 or energy_diff > 0.3:
            if genre_diff:
                return "Remix (Genre Change)"
            else:
                return "Remix"
        elif genre_diff:
            return "Cover (Genre Change)"
        elif tempo_diff > 10 or energy_diff > 0.15:
            return "Cover (Arrangement)"
        else:
            return "Cover (Similar)"

