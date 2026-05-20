"""
Servicio de predicción de tendencias musicales
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter
from datetime import datetime, timedelta

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)


class TrendPredictor:
    """Predice tendencias musicales futuras"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.logger = logger
    
    def predict_genre_trends(self, time_horizon: str = "6months") -> Dict[str, Any]:
        """Predice tendencias de géneros"""
        try:
            # Buscar tracks recientes
            recent_tracks = self.spotify.search_tracks("year:2024", limit=100)
            
            if not recent_tracks:
                return {"error": "No se encontraron tracks recientes"}
            
            # Analizar géneros de tracks recientes
            genres = []
            genre_popularities = {}
            
            for track in recent_tracks[:50]:  # Limitar para rendimiento
                try:
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    if not audio_features:
                        continue
                    
                    genre_analysis = self.genre_detector.detect_genre(audio_features)
                    genre = genre_analysis.get("primary_genre", "Unknown")
                    popularity = track.get("popularity", 0)
                    
                    genres.append(genre)
                    
                    if genre not in genre_popularities:
                        genre_popularities[genre] = []
                    genre_popularities[genre].append(popularity)
                except:
                    continue
            
            if not genres:
                return {"error": "No se pudieron analizar géneros"}
            
            # Calcular tendencias
            genre_counter = Counter(genres)
            trending_genres = []
            
            for genre, count in genre_counter.most_common(10):
                avg_popularity = sum(genre_popularities[genre]) / len(genre_popularities[genre]) if genre in genre_popularities else 0
                trend_score = (count / len(genres)) * 0.6 + (avg_popularity / 100) * 0.4
                
                trending_genres.append({
                    "genre": genre,
                    "frequency": count,
                    "average_popularity": round(avg_popularity, 2),
                    "trend_score": round(trend_score, 3),
                    "prediction": "Rising" if trend_score > 0.5 else "Stable" if trend_score > 0.3 else "Declining"
                })
            
            trending_genres.sort(key=lambda x: x["trend_score"], reverse=True)
            
            return {
                "time_horizon": time_horizon,
                "tracks_analyzed": len(recent_tracks),
                "trending_genres": trending_genres,
                "top_predicted_genre": trending_genres[0] if trending_genres else None
            }
        except Exception as e:
            self.logger.error(f"Error predicting genre trends: {e}")
            return {"error": str(e)}
    
    def predict_emotion_trends(self, time_horizon: str = "6months") -> Dict[str, Any]:
        """Predice tendencias de emociones"""
        try:
            # Buscar tracks recientes
            recent_tracks = self.spotify.search_tracks("year:2024", limit=100)
            
            if not recent_tracks:
                return {"error": "No se encontraron tracks recientes"}
            
            # Analizar emociones de tracks recientes
            emotions = []
            emotion_popularities = {}
            
            for track in recent_tracks[:50]:
                try:
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    if not audio_features:
                        continue
                    
                    emotion_analysis = self.emotion_analyzer.analyze_emotions(audio_features)
                    emotion = emotion_analysis.get("primary_emotion", "Unknown")
                    popularity = track.get("popularity", 0)
                    
                    emotions.append(emotion)
                    
                    if emotion not in emotion_popularities:
                        emotion_popularities[emotion] = []
                    emotion_popularities[emotion].append(popularity)
                except:
                    continue
            
            if not emotions:
                return {"error": "No se pudieron analizar emociones"}
            
            # Calcular tendencias
            emotion_counter = Counter(emotions)
            trending_emotions = []
            
            for emotion, count in emotion_counter.most_common(8):
                avg_popularity = sum(emotion_popularities[emotion]) / len(emotion_popularities[emotion]) if emotion in emotion_popularities else 0
                trend_score = (count / len(emotions)) * 0.6 + (avg_popularity / 100) * 0.4
                
                trending_emotions.append({
                    "emotion": emotion,
                    "frequency": count,
                    "average_popularity": round(avg_popularity, 2),
                    "trend_score": round(trend_score, 3),
                    "prediction": "Rising" if trend_score > 0.5 else "Stable" if trend_score > 0.3 else "Declining"
                })
            
            trending_emotions.sort(key=lambda x: x["trend_score"], reverse=True)
            
            return {
                "time_horizon": time_horizon,
                "tracks_analyzed": len(recent_tracks),
                "trending_emotions": trending_emotions,
                "top_predicted_emotion": trending_emotions[0] if trending_emotions else None
            }
        except Exception as e:
            self.logger.error(f"Error predicting emotion trends: {e}")
            return {"error": str(e)}
    
    def predict_feature_trends(self, time_horizon: str = "6months") -> Dict[str, Any]:
        """Predice tendencias de características musicales"""
        try:
            # Buscar tracks recientes
            recent_tracks = self.spotify.search_tracks("year:2024", limit=100)
            
            if not recent_tracks:
                return {"error": "No se encontraron tracks recientes"}
            
            # Analizar características
            energies = []
            danceabilities = []
            valences = []
            tempos = []
            
            for track in recent_tracks[:50]:
                try:
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    if not audio_features:
                        continue
                    
                    energies.append(audio_features.get("energy", 0.5))
                    danceabilities.append(audio_features.get("danceability", 0.5))
                    valences.append(audio_features.get("valence", 0.5))
                    tempos.append(audio_features.get("tempo", 120))
                except:
                    continue
            
            if not energies:
                return {"error": "No se pudieron analizar características"}
            
            # Calcular promedios y tendencias
            avg_energy = sum(energies) / len(energies)
            avg_danceability = sum(danceabilities) / len(danceabilities)
            avg_valence = sum(valences) / len(valences)
            avg_tempo = sum(tempos) / len(tempos)
            
            # Comparar con valores históricos (simplificado)
            historical_energy = 0.6  # Valor de referencia
            historical_danceability = 0.6
            historical_valence = 0.5
            historical_tempo = 120
            
            trends = {
                "energy": {
                    "current": round(avg_energy, 3),
                    "historical": historical_energy,
                    "trend": "Increasing" if avg_energy > historical_energy + 0.1 else "Decreasing" if avg_energy < historical_energy - 0.1 else "Stable",
                    "change": round(avg_energy - historical_energy, 3)
                },
                "danceability": {
                    "current": round(avg_danceability, 3),
                    "historical": historical_danceability,
                    "trend": "Increasing" if avg_danceability > historical_danceability + 0.1 else "Decreasing" if avg_danceability < historical_danceability - 0.1 else "Stable",
                    "change": round(avg_danceability - historical_danceability, 3)
                },
                "valence": {
                    "current": round(avg_valence, 3),
                    "historical": historical_valence,
                    "trend": "Increasing" if avg_valence > historical_valence + 0.1 else "Decreasing" if avg_valence < historical_valence - 0.1 else "Stable",
                    "change": round(avg_valence - historical_valence, 3)
                },
                "tempo": {
                    "current": round(avg_tempo, 2),
                    "historical": historical_tempo,
                    "trend": "Increasing" if avg_tempo > historical_tempo + 10 else "Decreasing" if avg_tempo < historical_tempo - 10 else "Stable",
                    "change": round(avg_tempo - historical_tempo, 2)
                }
            }
            
            return {
                "time_horizon": time_horizon,
                "tracks_analyzed": len(recent_tracks),
                "trends": trends,
                "summary": {
                    "overall_mood": "More Positive" if avg_valence > 0.6 else "More Negative" if avg_valence < 0.4 else "Balanced",
                    "energy_level": "Higher" if avg_energy > 0.7 else "Lower" if avg_energy < 0.5 else "Moderate",
                    "danceability": "More Danceable" if avg_danceability > 0.7 else "Less Danceable" if avg_danceability < 0.5 else "Moderate"
                }
            }
        except Exception as e:
            self.logger.error(f"Error predicting feature trends: {e}")
            return {"error": str(e)}
    
    def predict_next_big_thing(self, genre: Optional[str] = None) -> Dict[str, Any]:
        """Predice el próximo gran éxito"""
        try:
            # Buscar tracks recientes con características prometedoras
            query = f"genre:{genre}" if genre else "year:2024"
            tracks = self.spotify.search_tracks(query, limit=100)
            
            if not tracks:
                return {"error": "No se encontraron tracks"}
            
            candidates = []
            
            for track in tracks:
                try:
                    track_info = self.spotify.get_track(track.get("id"))
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    
                    if not audio_features:
                        continue
                    
                    popularity = track_info.get("popularity", 0)
                    
                    # Criterios para "próximo gran éxito"
                    # Popularidad moderada pero características comerciales fuertes
                    if 30 < popularity < 70:
                        danceability = audio_features.get("danceability", 0.5)
                        energy = audio_features.get("energy", 0.5)
                        valence = audio_features.get("valence", 0.5)
                        tempo = audio_features.get("tempo", 120)
                        
                        # Score de potencial
                        potential_score = (
                            (danceability * 0.3) +
                            (energy * 0.3) +
                            (valence * 0.2) +
                            (popularity / 100 * 0.2)
                        )
                        
                        # Bonus por tempo comercial
                        if 100 <= tempo <= 140:
                            potential_score += 0.1
                        
                        if potential_score > 0.7:
                            genre_analysis = self.genre_detector.detect_genre(audio_features)
                            emotion_analysis = self.emotion_analyzer.analyze_emotions(audio_features)
                            
                            candidates.append({
                                "track_id": track.get("id"),
                                "track_name": track_info.get("name", "Unknown"),
                                "artists": [a.get("name") for a in track_info.get("artists", [])],
                                "popularity": popularity,
                                "potential_score": round(potential_score, 3),
                                "genre": genre_analysis.get("primary_genre", "Unknown"),
                                "emotion": emotion_analysis.get("primary_emotion", "Unknown"),
                                "features": {
                                    "danceability": round(danceability, 3),
                                    "energy": round(energy, 3),
                                    "valence": round(valence, 3),
                                    "tempo": round(tempo, 2)
                                }
                            })
                except:
                    continue
            
            # Ordenar por potencial
            candidates.sort(key=lambda x: x["potential_score"], reverse=True)
            
            return {
                "genre_filter": genre,
                "candidates": candidates[:10],
                "top_candidate": candidates[0] if candidates else None,
                "total_found": len(candidates)
            }
        except Exception as e:
            self.logger.error(f"Error predicting next big thing: {e}")
            return {"error": str(e)}
