"""
Servicio de descubrimiento musical avanzado
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer
from .intelligent_recommender import IntelligentRecommender

logger = logging.getLogger(__name__)


class DiscoveryService:
    """Servicio de descubrimiento musical avanzado"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.recommender = IntelligentRecommender()
        self.logger = logger
    
    def discover_similar_artists(self, artist_name: str, limit: int = 10) -> Dict[str, Any]:
        """Descubre artistas similares"""
        try:
            # Buscar tracks del artista
            tracks = self.spotify.search_tracks(f"artist:{artist_name}", limit=5)
            
            if not tracks:
                return {"error": "Artista no encontrado"}
            
            # Analizar tracks del artista
            artist_features = []
            for track in tracks:
                try:
                    features = self.spotify.get_track_audio_features(track.get("id"))
                    if features:
                        artist_features.append(features)
                except:
                    continue
            
            if not artist_features:
                return {"error": "No se pudieron analizar tracks del artista"}
            
            # Calcular características promedio del artista
            avg_features = self._calculate_average_features(artist_features)
            
            # Buscar artistas similares usando recomendaciones
            similar_artists = []
            seen_artists = {artist_name.lower()}
            
            for track in tracks[:3]:  # Usar primeros 3 tracks
                try:
                    recommendations = self.spotify.get_recommendations(track.get("id"), limit=20)
                    
                    for rec in recommendations:
                        rec_artists = [a.get("name") for a in rec.get("artists", [])]
                        
                        for rec_artist in rec_artists:
                            if rec_artist.lower() not in seen_artists:
                                seen_artists.add(rec_artist.lower())
                                
                                # Analizar tracks del artista recomendado
                                rec_tracks = self.spotify.search_tracks(f"artist:{rec_artist}", limit=3)
                                
                                if rec_tracks:
                                    rec_features = []
                                    for rt in rec_tracks:
                                        try:
                                            rf = self.spotify.get_track_audio_features(rt.get("id"))
                                            if rf:
                                                rec_features.append(rf)
                                        except:
                                            continue
                                    
                                    if rec_features:
                                        rec_avg = self._calculate_average_features(rec_features)
                                        similarity = self._calculate_feature_similarity(avg_features, rec_avg)
                                        
                                        similar_artists.append({
                                            "artist_name": rec_artist,
                                            "similarity": round(similarity, 3),
                                            "tracks_analyzed": len(rec_features)
                                        })
                except:
                    continue
            
            # Ordenar por similitud
            similar_artists.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "source_artist": artist_name,
                "similar_artists": similar_artists[:limit],
                "total_found": len(similar_artists)
            }
        except Exception as e:
            self.logger.error(f"Error discovering similar artists: {e}")
            return {"error": str(e)}
    
    def discover_underground_tracks(self, genre: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        """Descubre tracks underground (baja popularidad pero alta calidad)"""
        try:
            # Buscar tracks por género o aleatorios
            if genre:
                query = f"genre:{genre}"
            else:
                query = "year:2020-2024"  # Tracks recientes
            
            tracks = self.spotify.search_tracks(query, limit=100)
            
            if not tracks:
                return {"error": "No se encontraron tracks"}
            
            # Filtrar por popularidad baja pero características interesantes
            underground_tracks = []
            
            for track in tracks:
                try:
                    popularity = track.get("popularity", 0)
                    track_info = self.spotify.get_track(track.get("id"))
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    
                    if not audio_features:
                        continue
                    
                    # Criterios: popularidad < 40 pero características interesantes
                    if popularity < 40:
                        # Calcular score de calidad
                        quality_score = (
                            audio_features.get("danceability", 0.5) * 0.2 +
                            audio_features.get("energy", 0.5) * 0.2 +
                            audio_features.get("valence", 0.5) * 0.2 +
                            (1.0 - abs(audio_features.get("acousticness", 0.5) - 0.5)) * 0.2 +
                            (1.0 - abs(audio_features.get("instrumentalness", 0.5) - 0.3)) * 0.2
                        )
                        
                        if quality_score > 0.6:  # Alta calidad
                            genre_analysis = self.genre_detector.detect_genre(audio_features)
                            emotion_analysis = self.emotion_analyzer.analyze_emotions(audio_features)
                            
                            underground_tracks.append({
                                "track_id": track.get("id"),
                                "track_name": track_info.get("name", "Unknown"),
                                "artists": [a.get("name") for a in track_info.get("artists", [])],
                                "popularity": popularity,
                                "quality_score": round(quality_score, 3),
                                "genre": genre_analysis.get("primary_genre", "Unknown"),
                                "emotion": emotion_analysis.get("primary_emotion", "Unknown"),
                                "preview_url": track_info.get("preview_url")
                            })
                except:
                    continue
            
            # Ordenar por quality score
            underground_tracks.sort(key=lambda x: x["quality_score"], reverse=True)
            
            return {
                "genre_filter": genre,
                "underground_tracks": underground_tracks[:limit],
                "total_found": len(underground_tracks)
            }
        except Exception as e:
            self.logger.error(f"Error discovering underground tracks: {e}")
            return {"error": str(e)}
    
    def discover_by_mood_transition(self, start_mood: str, target_mood: str, limit: int = 10) -> Dict[str, Any]:
        """Descubre tracks que transicionan entre moods"""
        try:
            mood_mapping = {
                "sad": {"valence_range": [0.0, 0.4], "energy_range": [0.2, 0.6]},
                "happy": {"valence_range": [0.6, 1.0], "energy_range": [0.5, 1.0]},
                "energetic": {"valence_range": [0.5, 1.0], "energy_range": [0.7, 1.0]},
                "calm": {"valence_range": [0.4, 0.7], "energy_range": [0.2, 0.5]},
                "romantic": {"valence_range": [0.5, 0.8], "energy_range": [0.3, 0.6]},
                "angry": {"valence_range": [0.0, 0.4], "energy_range": [0.7, 1.0]}
            }
            
            start_range = mood_mapping.get(start_mood.lower(), mood_mapping["calm"])
            target_range = mood_mapping.get(target_mood.lower(), mood_mapping["happy"])
            
            # Buscar tracks que estén en el rango intermedio
            mid_valence = (start_range["valence_range"][0] + target_range["valence_range"][1]) / 2
            mid_energy = (start_range["energy_range"][0] + target_range["energy_range"][1]) / 2
            
            # Buscar tracks
            tracks = self.spotify.search_tracks("year:2020-2024", limit=100)
            
            transition_tracks = []
            
            for track in tracks:
                try:
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    if not audio_features:
                        continue
                    
                    valence = audio_features.get("valence", 0.5)
                    energy = audio_features.get("energy", 0.5)
                    
                    # Verificar si está en rango de transición
                    if (abs(valence - mid_valence) < 0.2 and abs(energy - mid_energy) < 0.2):
                        track_info = self.spotify.get_track(track.get("id"))
                        transition_tracks.append({
                            "track_id": track.get("id"),
                            "track_name": track_info.get("name", "Unknown"),
                            "artists": [a.get("name") for a in track_info.get("artists", [])],
                            "valence": round(valence, 3),
                            "energy": round(energy, 3),
                            "transition_score": round(1.0 - abs(valence - mid_valence) - abs(energy - mid_energy), 3)
                        })
                except:
                    continue
            
            transition_tracks.sort(key=lambda x: x["transition_score"], reverse=True)
            
            return {
                "start_mood": start_mood,
                "target_mood": target_mood,
                "transition_tracks": transition_tracks[:limit],
                "total_found": len(transition_tracks)
            }
        except Exception as e:
            self.logger.error(f"Error discovering mood transitions: {e}")
            return {"error": str(e)}
    
    def discover_fresh_tracks(self, genre: Optional[str] = None, days_old: int = 30, limit: int = 20) -> Dict[str, Any]:
        """Descubre tracks frescos (recientes)"""
        try:
            # Buscar tracks recientes
            query = "year:2024" if not genre else f"genre:{genre} year:2024"
            tracks = self.spotify.search_tracks(query, limit=100)
            
            if not tracks:
                return {"error": "No se encontraron tracks recientes"}
            
            fresh_tracks = []
            
            for track in tracks:
                try:
                    track_info = self.spotify.get_track(track.get("id"))
                    release_date = track_info.get("album", {}).get("release_date", "")
                    
                    if not release_date:
                        continue
                    
                    # Verificar si es reciente (simplificado - en producción usar fecha real)
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    if not audio_features:
                        continue
                    
                    popularity = track_info.get("popularity", 0)
                    
                    # Incluir tracks con popularidad moderada o alta
                    if popularity > 30:
                        genre_analysis = self.genre_detector.detect_genre(audio_features)
                        emotion_analysis = self.emotion_analyzer.analyze_emotions(audio_features)
                        
                        fresh_tracks.append({
                            "track_id": track.get("id"),
                            "track_name": track_info.get("name", "Unknown"),
                            "artists": [a.get("name") for a in track_info.get("artists", [])],
                            "release_date": release_date,
                            "popularity": popularity,
                            "genre": genre_analysis.get("primary_genre", "Unknown"),
                            "emotion": emotion_analysis.get("primary_emotion", "Unknown"),
                            "preview_url": track_info.get("preview_url")
                        })
                except:
                    continue
            
            # Ordenar por popularidad
            fresh_tracks.sort(key=lambda x: x["popularity"], reverse=True)
            
            return {
                "genre_filter": genre,
                "days_old": days_old,
                "fresh_tracks": fresh_tracks[:limit],
                "total_found": len(fresh_tracks)
            }
        except Exception as e:
            self.logger.error(f"Error discovering fresh tracks: {e}")
            return {"error": str(e)}
    
    def _calculate_average_features(self, features_list: List[Dict]) -> Dict[str, float]:
        """Calcula características promedio"""
        if not features_list:
            return {}
        
        avg = {}
        keys = ["energy", "danceability", "valence", "tempo", "acousticness", "instrumentalness"]
        
        for key in keys:
            values = [f.get(key, 0) for f in features_list if f.get(key) is not None]
            if values:
                avg[key] = sum(values) / len(values)
        
        return avg
    
    def _calculate_feature_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calcula similitud entre características"""
        if not features1 or not features2:
            return 0.0
        
        keys = ["energy", "danceability", "valence", "acousticness", "instrumentalness"]
        similarities = []
        
        for key in keys:
            if key in features1 and key in features2:
                if key == "tempo":
                    # Normalizar tempo
                    diff = abs(features1[key] - features2[key]) / 200
                    sim = 1.0 - min(diff, 1.0)
                else:
                    diff = abs(features1[key] - features2[key])
                    sim = 1.0 - diff
                
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

