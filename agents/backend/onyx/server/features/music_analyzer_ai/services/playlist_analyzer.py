"""
Servicio de análisis inteligente de playlists
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer
from .intelligent_recommender import IntelligentRecommender

logger = logging.getLogger(__name__)


class PlaylistAnalyzer:
    """Analiza playlists de manera inteligente"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.recommender = IntelligentRecommender()
        self.logger = logger
    
    def analyze_playlist(self, track_ids: List[str]) -> Dict[str, Any]:
        """Analiza una playlist completa"""
        try:
            if len(track_ids) > 100:
                track_ids = track_ids[:100]  # Limitar a 100 tracks
            
            if not track_ids:
                return {"error": "Playlist vacía"}
            
            # Analizar cada track
            tracks_data = []
            genres = []
            emotions = []
            artists = []
            years = []
            
            for track_id in track_ids:
                try:
                    track_info = self.spotify.get_track(track_id)
                    audio_features = self.spotify.get_track_audio_features(track_id)
                    
                    if not audio_features:
                        continue
                    
                    genre = self.genre_detector.detect_genre(audio_features)
                    emotion = self.emotion_analyzer.analyze_emotions(audio_features)
                    
                    tracks_data.append({
                        "track_id": track_id,
                        "track_name": track_info.get("name", "Unknown"),
                        "artists": [a.get("name") for a in track_info.get("artists", [])],
                        "genre": genre.get("primary_genre", "Unknown"),
                        "emotion": emotion.get("primary_emotion", "Unknown"),
                        "popularity": track_info.get("popularity", 0),
                        "duration_ms": track_info.get("duration_ms", 0),
                        "audio_features": audio_features
                    })
                    
                    genres.append(genre.get("primary_genre", "Unknown"))
                    emotions.append(emotion.get("primary_emotion", "Unknown"))
                    artists.extend([a.get("name") for a in track_info.get("artists", [])])
                    
                    # Extraer año si está disponible
                    release_date = track_info.get("album", {}).get("release_date", "")
                    if release_date:
                        try:
                            year = int(release_date.split("-")[0])
                            years.append(year)
                        except:
                            pass
                except:
                    continue
            
            if not tracks_data:
                return {"error": "No se pudieron analizar tracks"}
            
            # Calcular estadísticas
            total_duration = sum(t["duration_ms"] for t in tracks_data) / 1000 / 60  # minutos
            avg_popularity = sum(t["popularity"] for t in tracks_data) / len(tracks_data)
            
            # Análisis de diversidad
            diversity = self._calculate_diversity(tracks_data)
            
            # Análisis de coherencia
            coherence = self._calculate_coherence(tracks_data)
            
            # Análisis de flujo
            flow = self._analyze_flow(tracks_data)
            
            # Estadísticas de género y emoción
            genre_stats = Counter(genres)
            emotion_stats = Counter(emotions)
            artist_stats = Counter(artists)
            
            # Análisis temporal
            year_range = None
            if years:
                year_range = {
                    "min": min(years),
                    "max": max(years),
                    "span": max(years) - min(years),
                    "average": int(sum(years) / len(years))
                }
            
            return {
                "playlist_stats": {
                    "total_tracks": len(tracks_data),
                    "total_duration_minutes": round(total_duration, 2),
                    "average_popularity": round(avg_popularity, 2),
                    "year_range": year_range
                },
                "diversity": diversity,
                "coherence": coherence,
                "flow": flow,
                "genre_distribution": dict(genre_stats.most_common(10)),
                "emotion_distribution": dict(emotion_stats.most_common(8)),
                "top_artists": dict(artist_stats.most_common(10)),
                "tracks": tracks_data[:20]  # Primeros 20 para respuesta
            }
        except Exception as e:
            self.logger.error(f"Error analyzing playlist: {e}")
            return {"error": str(e)}
    
    def suggest_playlist_improvements(self, track_ids: List[str]) -> Dict[str, Any]:
        """Sugiere mejoras para una playlist"""
        try:
            analysis = self.analyze_playlist(track_ids)
            
            if "error" in analysis:
                return analysis
            
            suggestions = []
            
            # Sugerencias basadas en diversidad
            if analysis["diversity"]["score"] < 0.3:
                suggestions.append({
                    "type": "diversity",
                    "priority": "high",
                    "message": "Playlist muy homogénea - agregar más variedad de géneros",
                    "action": "Considerar agregar tracks de diferentes géneros"
                })
            
            # Sugerencias basadas en coherencia
            if analysis["coherence"]["score"] < 0.4:
                suggestions.append({
                    "type": "coherence",
                    "priority": "medium",
                    "message": "Playlist poco coherente - tracks muy diferentes",
                    "action": "Reorganizar tracks para mejor flujo"
                })
            
            # Sugerencias basadas en flujo
            if analysis["flow"]["score"] < 0.5:
                suggestions.append({
                    "type": "flow",
                    "priority": "medium",
                    "message": "Flujo de energía irregular",
                    "action": "Reordenar tracks para progresión más suave"
                })
            
            # Sugerencias basadas en duración
            if analysis["playlist_stats"]["total_duration_minutes"] < 30:
                suggestions.append({
                    "type": "length",
                    "priority": "low",
                    "message": "Playlist corta - considerar agregar más tracks",
                    "action": "Agregar más tracks para mejor experiencia"
                })
            
            return {
                "suggestions": suggestions,
                "analysis": analysis
            }
        except Exception as e:
            self.logger.error(f"Error suggesting improvements: {e}")
            return {"error": str(e)}
    
    def optimize_playlist_order(self, track_ids: List[str]) -> Dict[str, Any]:
        """Optimiza el orden de una playlist"""
        try:
            if len(track_ids) > 100:
                track_ids = track_ids[:100]
            
            # Analizar tracks
            tracks_data = []
            for track_id in track_ids:
                try:
                    track_info = self.spotify.get_track(track_id)
                    audio_features = self.spotify.get_track_audio_features(track_id)
                    
                    if audio_features:
                        tracks_data.append({
                            "track_id": track_id,
                            "energy": audio_features.get("energy", 0.5),
                            "tempo": audio_features.get("tempo", 120),
                            "valence": audio_features.get("valence", 0.5),
                            "danceability": audio_features.get("danceability", 0.5)
                        })
                except:
                    continue
            
            if len(tracks_data) < 2:
                return {"error": "No hay suficientes tracks para optimizar"}
            
            # Ordenar por energía (progresión ascendente)
            sorted_tracks = sorted(tracks_data, key=lambda x: x["energy"])
            
            # Crear orden optimizado con variación
            optimized_order = []
            low_energy = [t for t in sorted_tracks if t["energy"] < 0.5]
            high_energy = [t for t in sorted_tracks if t["energy"] >= 0.5]
            
            # Intercalar para mejor flujo
            max_len = max(len(low_energy), len(high_energy))
            for i in range(max_len):
                if i < len(low_energy):
                    optimized_order.append(low_energy[i])
                if i < len(high_energy):
                    optimized_order.append(high_energy[i])
            
            return {
                "original_count": len(track_ids),
                "optimized_count": len(optimized_order),
                "optimized_order": [t["track_id"] for t in optimized_order],
                "strategy": "energy_progression_with_variation"
            }
        except Exception as e:
            self.logger.error(f"Error optimizing playlist order: {e}")
            return {"error": str(e)}
    
    def _calculate_diversity(self, tracks_data: List[Dict]) -> Dict[str, Any]:
        """Calcula la diversidad de la playlist"""
        genres = [t["genre"] for t in tracks_data]
        emotions = [t["emotion"] for t in tracks_data]
        artists = []
        for t in tracks_data:
            artists.extend(t["artists"])
        
        genre_diversity = len(set(genres)) / len(genres) if genres else 0
        emotion_diversity = len(set(emotions)) / len(emotions) if emotions else 0
        artist_diversity = len(set(artists)) / len(artists) if artists else 0
        
        score = (genre_diversity * 0.4 + emotion_diversity * 0.3 + artist_diversity * 0.3)
        
        return {
            "score": round(score, 3),
            "level": "High" if score > 0.6 else "Medium" if score > 0.3 else "Low",
            "genre_diversity": round(genre_diversity, 3),
            "emotion_diversity": round(emotion_diversity, 3),
            "artist_diversity": round(artist_diversity, 3)
        }
    
    def _calculate_coherence(self, tracks_data: List[Dict]) -> Dict[str, Any]:
        """Calcula la coherencia de la playlist"""
        if len(tracks_data) < 2:
            return {"score": 1.0, "level": "High"}
        
        # Calcular similitud promedio entre tracks consecutivos
        similarities = []
        for i in range(len(tracks_data) - 1):
            try:
                features1 = tracks_data[i]["audio_features"]
                features2 = tracks_data[i+1]["audio_features"]
                similarity = self.recommender._calculate_similarity(features1, features2)
                similarities.append(similarity)
            except:
                continue
        
        if not similarities:
            return {"score": 0.5, "level": "Medium"}
        
        avg_similarity = sum(similarities) / len(similarities)
        
        return {
            "score": round(avg_similarity, 3),
            "level": "High" if avg_similarity > 0.7 else "Medium" if avg_similarity > 0.5 else "Low",
            "average_similarity": round(avg_similarity, 3)
        }
    
    def _analyze_flow(self, tracks_data: List[Dict]) -> Dict[str, Any]:
        """Analiza el flujo de energía de la playlist"""
        if len(tracks_data) < 2:
            return {"score": 1.0, "level": "Smooth"}
        
        energies = []
        for t in tracks_data:
            try:
                energy = t["audio_features"].get("energy", 0.5)
                energies.append(energy)
            except:
                continue
        
        if len(energies) < 2:
            return {"score": 0.5, "level": "Unknown"}
        
        # Calcular variación de energía
        energy_changes = []
        for i in range(1, len(energies)):
            change = abs(energies[i] - energies[i-1])
            energy_changes.append(change)
        
        avg_change = sum(energy_changes) / len(energy_changes) if energy_changes else 0
        
        # Score: menos cambios bruscos = mejor flujo
        flow_score = 1.0 / (1.0 + avg_change * 2)
        
        if flow_score > 0.7:
            level = "Smooth"
        elif flow_score > 0.5:
            level = "Moderate"
        else:
            level = "Irregular"
        
        return {
            "score": round(flow_score, 3),
            "level": level,
            "average_energy_change": round(avg_change, 3)
        }

