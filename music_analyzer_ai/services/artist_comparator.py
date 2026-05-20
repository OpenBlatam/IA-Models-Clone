"""
Servicio de comparación avanzada de artistas
"""

import logging
from typing import Dict, List, Any, Optional
from collections import Counter, defaultdict

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)


class ArtistComparator:
    """Compara artistas de manera avanzada"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.logger = logger
    
    def compare_artists(self, artist_names: List[str], limit_per_artist: int = 10) -> Dict[str, Any]:
        """Compara múltiples artistas"""
        try:
            if len(artist_names) > 5:
                artist_names = artist_names[:5]  # Máximo 5 artistas
            
            if len(artist_names) < 2:
                return {"error": "Se necesitan al menos 2 artistas"}
            
            artists_data = []
            
            for artist_name in artist_names:
                try:
                    # Buscar tracks del artista
                    tracks = self.spotify.search_tracks(f"artist:{artist_name}", limit=limit_per_artist)
                    
                    if not tracks:
                        continue
                    
                    # Analizar tracks
                    genres = []
                    emotions = []
                    features_avg = {
                        "energy": [],
                        "danceability": [],
                        "valence": [],
                        "tempo": [],
                        "acousticness": []
                    }
                    popularities = []
                    
                    for track in tracks:
                        try:
                            track_info = self.spotify.get_track(track.get("id"))
                            audio_features = self.spotify.get_track_audio_features(track.get("id"))
                            
                            if audio_features:
                                genre = self.genre_detector.detect_genre(audio_features)
                                emotion = self.emotion_analyzer.analyze_emotions(audio_features)
                                
                                genres.append(genre.get("primary_genre", "Unknown"))
                                emotions.append(emotion.get("primary_emotion", "Unknown"))
                                
                                features_avg["energy"].append(audio_features.get("energy", 0.5))
                                features_avg["danceability"].append(audio_features.get("danceability", 0.5))
                                features_avg["valence"].append(audio_features.get("valence", 0.5))
                                features_avg["tempo"].append(audio_features.get("tempo", 120))
                                features_avg["acousticness"].append(audio_features.get("acousticness", 0.5))
                                
                                popularities.append(track_info.get("popularity", 0))
                        except:
                            continue
                    
                    if not genres:
                        continue
                    
                    # Calcular promedios
                    avg_features = {}
                    for key, values in features_avg.items():
                        if values:
                            avg_features[key] = sum(values) / len(values)
                    
                    # Género y emoción más comunes
                    genre_counter = Counter(genres)
                    emotion_counter = Counter(emotions)
                    
                    artists_data.append({
                        "artist_name": artist_name,
                        "tracks_analyzed": len(tracks),
                        "primary_genre": genre_counter.most_common(1)[0][0] if genre_counter else "Unknown",
                        "primary_emotion": emotion_counter.most_common(1)[0][0] if emotion_counter else "Unknown",
                        "genre_distribution": dict(genre_counter.most_common(5)),
                        "emotion_distribution": dict(emotion_counter.most_common(5)),
                        "average_features": {k: round(v, 3) for k, v in avg_features.items()},
                        "average_popularity": round(sum(popularities) / len(popularities), 2) if popularities else 0
                    })
                except:
                    continue
            
            if len(artists_data) < 2:
                return {"error": "No se pudieron analizar suficientes artistas"}
            
            # Comparaciones
            comparisons = self._compare_artists_features(artists_data)
            
            # Similitud entre artistas
            similarities = self._calculate_artist_similarities(artists_data)
            
            return {
                "artists_compared": len(artists_data),
                "artists": artists_data,
                "comparisons": comparisons,
                "similarities": similarities,
                "most_similar_pair": max(similarities, key=lambda x: x["similarity"]) if similarities else None
            }
        except Exception as e:
            self.logger.error(f"Error comparing artists: {e}")
            return {"error": str(e)}
    
    def analyze_artist_evolution(self, artist_name: str, limit: int = 20) -> Dict[str, Any]:
        """Analiza la evolución musical de un artista"""
        try:
            # Buscar tracks del artista
            tracks = self.spotify.search_tracks(f"artist:{artist_name}", limit=limit)
            
            if not tracks:
                return {"error": "No se encontraron tracks del artista"}
            
            # Analizar tracks con fechas
            tracks_with_dates = []
            
            for track in tracks:
                try:
                    track_info = self.spotify.get_track(track.get("id"))
                    audio_features = self.spotify.get_track_audio_features(track.get("id"))
                    
                    if not audio_features:
                        continue
                    
                    release_date = track_info.get("album", {}).get("release_date", "")
                    year = None
                    if release_date:
                        try:
                            year = int(release_date.split("-")[0])
                        except:
                            pass
                    
                    if year:
                        genre = self.genre_detector.detect_genre(audio_features)
                        emotion = self.emotion_analyzer.analyze_emotions(audio_features)
                        
                        tracks_with_dates.append({
                            "track_id": track.get("id"),
                            "track_name": track_info.get("name", "Unknown"),
                            "year": year,
                            "genre": genre.get("primary_genre", "Unknown"),
                            "emotion": emotion.get("primary_emotion", "Unknown"),
                            "energy": audio_features.get("energy", 0.5),
                            "popularity": track_info.get("popularity", 0)
                        })
                except:
                    continue
            
            if not tracks_with_dates:
                return {"error": "No se pudieron analizar tracks con fechas"}
            
            # Ordenar por año
            tracks_with_dates.sort(key=lambda x: x["year"])
            
            # Analizar evolución
            evolution = self._analyze_evolution(tracks_with_dates)
            
            return {
                "artist_name": artist_name,
                "tracks_analyzed": len(tracks_with_dates),
                "year_range": {
                    "start": tracks_with_dates[0]["year"],
                    "end": tracks_with_dates[-1]["year"],
                    "span": tracks_with_dates[-1]["year"] - tracks_with_dates[0]["year"]
                },
                "evolution": evolution,
                "tracks": tracks_with_dates
            }
        except Exception as e:
            self.logger.error(f"Error analyzing artist evolution: {e}")
            return {"error": str(e)}
    
    def _compare_artists_features(self, artists_data: List[Dict]) -> Dict[str, Any]:
        """Compara características de artistas"""
        comparisons = {}
        
        # Comparar energía
        energies = [a["average_features"].get("energy", 0.5) for a in artists_data]
        comparisons["energy"] = {
            "values": {a["artist_name"]: round(a["average_features"].get("energy", 0.5), 3) for a in artists_data},
            "highest": max(artists_data, key=lambda x: x["average_features"].get("energy", 0.5))["artist_name"],
            "lowest": min(artists_data, key=lambda x: x["average_features"].get("energy", 0.5))["artist_name"],
            "range": round(max(energies) - min(energies), 3)
        }
        
        # Comparar popularidad
        popularities = [a["average_popularity"] for a in artists_data]
        comparisons["popularity"] = {
            "values": {a["artist_name"]: a["average_popularity"] for a in artists_data},
            "highest": max(artists_data, key=lambda x: x["average_popularity"])["artist_name"],
            "lowest": min(artists_data, key=lambda x: x["average_popularity"])["artist_name"]
        }
        
        # Comparar géneros
        all_genres = set()
        for artist in artists_data:
            all_genres.update(artist["genre_distribution"].keys())
        
        genre_overlap = {}
        for genre in all_genres:
            artists_with_genre = [
                a["artist_name"] for a in artists_data
                if genre in a["genre_distribution"]
            ]
            if len(artists_with_genre) > 1:
                genre_overlap[genre] = artists_with_genre
        
        comparisons["genre_overlap"] = genre_overlap
        
        return comparisons
    
    def _calculate_artist_similarities(self, artists_data: List[Dict]) -> List[Dict]:
        """Calcula similitud entre artistas"""
        similarities = []
        
        for i, artist1 in enumerate(artists_data):
            for artist2 in artists_data[i+1:]:
                # Calcular similitud basada en características
                features1 = artist1["average_features"]
                features2 = artist2["average_features"]
                
                # Similitud de energía
                energy_sim = 1.0 - abs(features1.get("energy", 0.5) - features2.get("energy", 0.5))
                
                # Similitud de género
                genre_sim = 1.0 if artist1["primary_genre"] == artist2["primary_genre"] else 0.5
                
                # Similitud de emoción
                emotion_sim = 1.0 if artist1["primary_emotion"] == artist2["primary_emotion"] else 0.5
                
                # Score combinado
                similarity = (energy_sim * 0.4 + genre_sim * 0.3 + emotion_sim * 0.3)
                
                similarities.append({
                    "artist1": artist1["artist_name"],
                    "artist2": artist2["artist_name"],
                    "similarity": round(similarity, 3)
                })
        
        return similarities
    
    def _analyze_evolution(self, tracks: List[Dict]) -> Dict[str, Any]:
        """Analiza la evolución del artista"""
        if len(tracks) < 2:
            return {"error": "No hay suficientes tracks para analizar evolución"}
        
        # Dividir en períodos
        total_years = tracks[-1]["year"] - tracks[0]["year"]
        if total_years < 5:
            # Período único
            periods = [tracks]
        else:
            # Dividir en 3 períodos
            period_size = len(tracks) // 3
            periods = [
                tracks[:period_size],
                tracks[period_size:period_size*2],
                tracks[period_size*2:]
            ]
        
        period_analysis = []
        for i, period in enumerate(periods):
            if not period:
                continue
            
            genres = [t["genre"] for t in period]
            emotions = [t["emotion"] for t in period]
            energies = [t["energy"] for t in period]
            
            genre_counter = Counter(genres)
            emotion_counter = Counter(emotions)
            
            period_analysis.append({
                "period": i + 1,
                "year_range": f"{period[0]['year']}-{period[-1]['year']}",
                "tracks_count": len(period),
                "primary_genre": genre_counter.most_common(1)[0][0] if genre_counter else "Unknown",
                "primary_emotion": emotion_counter.most_common(1)[0][0] if emotion_counter else "Unknown",
                "average_energy": round(sum(energies) / len(energies), 3) if energies else 0
            })
        
        # Detectar cambios
        changes = []
        if len(period_analysis) > 1:
            for i in range(1, len(period_analysis)):
                prev = period_analysis[i-1]
                curr = period_analysis[i]
                
                if prev["primary_genre"] != curr["primary_genre"]:
                    changes.append({
                        "type": "genre_change",
                        "from": prev["primary_genre"],
                        "to": curr["primary_genre"],
                        "period": curr["period"]
                    })
                
                if abs(prev["average_energy"] - curr["average_energy"]) > 0.2:
                    changes.append({
                        "type": "energy_change",
                        "from": prev["average_energy"],
                        "to": curr["average_energy"],
                        "period": curr["period"]
                    })
        
        return {
            "periods": period_analysis,
            "changes": changes,
            "evolution_trend": self._identify_evolution_trend(period_analysis)
        }
    
    def _identify_evolution_trend(self, periods: List[Dict]) -> str:
        """Identifica la tendencia de evolución"""
        if len(periods) < 2:
            return "Insufficient Data"
        
        energies = [p["average_energy"] for p in periods]
        
        if energies[-1] > energies[0] + 0.1:
            return "Increasing Energy"
        elif energies[-1] < energies[0] - 0.1:
            return "Decreasing Energy"
        else:
            return "Stable"

