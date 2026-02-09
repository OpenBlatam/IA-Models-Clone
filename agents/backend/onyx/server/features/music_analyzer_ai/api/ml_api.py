"""
API endpoints para funcionalidades de Machine Learning
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any
import logging

from ..services.spotify_service import SpotifyService
from ..core.music_analyzer import MusicAnalyzer
from ..services.intelligent_recommender import IntelligentRecommender
from ..services.emotion_analyzer import EmotionAnalyzer
from ..services.genre_detector import GenreDetector
from ..utils.exceptions import TrackNotFoundException, InvalidTrackIDException

logger = logging.getLogger(__name__)

ml_router = APIRouter(prefix="/music/ml", tags=["Machine Learning"])

# Inicializar servicios
spotify_service = SpotifyService()
music_analyzer = MusicAnalyzer()
intelligent_recommender = IntelligentRecommender()
emotion_analyzer = EmotionAnalyzer()
genre_detector = GenreDetector()


@ml_router.get("/")
async def ml_root():
    """Endpoint raíz de ML"""
    return {
        "service": "Music Analyzer AI - ML Features",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Intelligent Recommendations",
            "Emotion Analysis",
            "Genre Detection",
            "Harmonic Analysis",
            "Similarity Matching"
        ]
    }


@ml_router.post("/predict/genre", response_model=dict)
async def predict_genre(track_id: str):
    """Predice el género de una canción usando ML"""
    try:
        audio_features = spotify_service.get_track_audio_features(track_id)
        genre_analysis = genre_detector.detect_genre(audio_features)
        
        return {
            "success": True,
            "track_id": track_id,
            "prediction": genre_analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting genre: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir género: {str(e)}")


@ml_router.post("/predict/emotion", response_model=dict)
async def predict_emotion(track_id: str):
    """Predice las emociones de una canción usando ML"""
    try:
        audio_features = spotify_service.get_track_audio_features(track_id)
        emotion_analysis = emotion_analyzer.analyze_emotions(audio_features)
        
        return {
            "success": True,
            "track_id": track_id,
            "prediction": emotion_analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting emotion: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir emoción: {str(e)}")


@ml_router.post("/similarity/batch", response_model=dict)
async def batch_similarity(
    track_ids: List[str],
    target_track_id: Optional[str] = None
):
    """Calcula similitud entre múltiples tracks"""
    try:
        if not track_ids:
            raise HTTPException(status_code=400, detail="Debe proporcionar al menos un track_id")
        
        if len(track_ids) > 50:
            raise HTTPException(status_code=400, detail="Máximo 50 tracks a la vez")
        
        # Obtener características del track objetivo si se proporciona
        target_features = None
        if target_track_id:
            target_data = spotify_service.get_track_full_analysis(target_track_id)
            target_features = target_data.get("audio_features", {})
        
        # Obtener características de todos los tracks
        tracks_with_features = []
        for track_id in track_ids:
            try:
                features = spotify_service.get_track_audio_features(track_id)
                track_info = spotify_service.get_track(track_id)
                tracks_with_features.append({
                    "track_id": track_id,
                    "track_name": track_info.get("name", "Unknown"),
                    "audio_features": features
                })
            except:
                continue
        
        # Calcular similitudes
        similarities = []
        if target_features:
            # Comparar todos con el target
            for track_data in tracks_with_features:
                similarity = intelligent_recommender._calculate_similarity(
                    target_features, track_data["audio_features"]
                )
                similarities.append({
                    "track_id": track_data["track_id"],
                    "track_name": track_data["track_name"],
                    "similarity_score": round(similarity, 3)
                })
        else:
            # Comparar todos entre sí
            for i, track1 in enumerate(tracks_with_features):
                for track2 in tracks_with_features[i+1:]:
                    similarity = intelligent_recommender._calculate_similarity(
                        track1["audio_features"], track2["audio_features"]
                    )
                    similarities.append({
                        "track1_id": track1["track_id"],
                        "track1_name": track1["track_name"],
                        "track2_id": track2["track_id"],
                        "track2_name": track2["track_name"],
                        "similarity_score": round(similarity, 3)
                    })
        
        # Ordenar por similitud
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return {
            "success": True,
            "target_track_id": target_track_id,
            "similarities": similarities,
            "total": len(similarities)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating batch similarity: {e}")
        raise HTTPException(status_code=500, detail=f"Error al calcular similitud: {str(e)}")


@ml_router.post("/cluster/genres", response_model=dict)
async def cluster_by_genre(track_ids: List[str]):
    """Agrupa tracks por género usando ML"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        # Analizar géneros de todos los tracks
        genre_clusters = {}
        
        for track_id in track_ids:
            try:
                features = spotify_service.get_track_audio_features(track_id)
                track_info = spotify_service.get_track(track_id)
                genre_analysis = genre_detector.detect_genre(features)
                
                genre = genre_analysis["primary_genre"]
                
                if genre not in genre_clusters:
                    genre_clusters[genre] = []
                
                genre_clusters[genre].append({
                    "track_id": track_id,
                    "track_name": track_info.get("name", "Unknown"),
                    "confidence": genre_analysis["confidence"]
                })
            except:
                continue
        
        # Ordenar por confianza dentro de cada cluster
        for genre in genre_clusters:
            genre_clusters[genre].sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "clusters": genre_clusters,
            "total_clusters": len(genre_clusters),
            "total_tracks": sum(len(tracks) for tracks in genre_clusters.values())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clustering by genre: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agrupar por género: {str(e)}")


@ml_router.post("/cluster/emotions", response_model=dict)
async def cluster_by_emotion(track_ids: List[str]):
    """Agrupa tracks por emoción usando ML"""
    try:
        if len(track_ids) > 100:
            raise HTTPException(status_code=400, detail="Máximo 100 tracks")
        
        # Analizar emociones de todos los tracks
        emotion_clusters = {}
        
        for track_id in track_ids:
            try:
                features = spotify_service.get_track_audio_features(track_id)
                track_info = spotify_service.get_track(track_id)
                emotion_analysis = emotion_analyzer.analyze_emotions(features)
                
                emotion = emotion_analysis["primary_emotion"]
                
                if emotion not in emotion_clusters:
                    emotion_clusters[emotion] = []
                
                emotion_clusters[emotion].append({
                    "track_id": track_id,
                    "track_name": track_info.get("name", "Unknown"),
                    "confidence": emotion_analysis["confidence"]
                })
            except:
                continue
        
        # Ordenar por confianza dentro de cada cluster
        for emotion in emotion_clusters:
            emotion_clusters[emotion].sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "clusters": emotion_clusters,
            "total_clusters": len(emotion_clusters),
            "total_tracks": sum(len(tracks) for tracks in emotion_clusters.values())
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error clustering by emotion: {e}")
        raise HTTPException(status_code=500, detail=f"Error al agrupar por emoción: {str(e)}")


@ml_router.post("/analyze/advanced", response_model=dict)
async def advanced_ml_analysis(
    track_id: str,
    include_predictions: bool = True,
    include_clustering: bool = False
):
    """Análisis ML avanzado de una canción"""
    try:
        # Obtener datos completos
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        analysis = music_analyzer.analyze_track(spotify_data)
        
        result = {
            "success": True,
            "track_id": track_id,
            "basic_analysis": {
                "track_basic_info": analysis["track_basic_info"],
                "musical_analysis": analysis["musical_analysis"],
                "technical_analysis": analysis["technical_analysis"]
            }
        }
        
        if include_predictions:
            result["ml_predictions"] = {
                "genre": analysis.get("genre_analysis", {}),
                "emotion": analysis.get("emotion_analysis", {}),
                "harmonic": analysis.get("harmonic_analysis", {})
            }
        
        if include_clustering:
            # Obtener recomendaciones similares para clustering
            try:
                recommendations = spotify_service.get_recommendations(track_id, limit=20)
                rec_track_ids = [r.get("id") for r in recommendations if r.get("id")]
                
                if rec_track_ids:
                    # Realizar clustering directamente
                    all_track_ids = [track_id] + rec_track_ids[:10]
                    
                    # Clustering por género
                    genre_clusters_data = {}
                    for tid in all_track_ids:
                        try:
                            features = spotify_service.get_track_audio_features(tid)
                            track_info = spotify_service.get_track(tid)
                            genre_analysis = genre_detector.detect_genre(features)
                            genre = genre_analysis["primary_genre"]
                            
                            if genre not in genre_clusters_data:
                                genre_clusters_data[genre] = []
                            genre_clusters_data[genre].append({
                                "track_id": tid,
                                "track_name": track_info.get("name", "Unknown")
                            })
                        except:
                            continue
                    
                    # Clustering por emoción
                    emotion_clusters_data = {}
                    for tid in all_track_ids:
                        try:
                            features = spotify_service.get_track_audio_features(tid)
                            track_info = spotify_service.get_track(tid)
                            emotion_analysis = emotion_analyzer.analyze_emotions(features)
                            emotion = emotion_analysis["primary_emotion"]
                            
                            if emotion not in emotion_clusters_data:
                                emotion_clusters_data[emotion] = []
                            emotion_clusters_data[emotion].append({
                                "track_id": tid,
                                "track_name": track_info.get("name", "Unknown")
                            })
                        except:
                            continue
                    
                    result["clustering"] = {
                        "by_genre": genre_clusters_data,
                        "by_emotion": emotion_clusters_data
                    }
            except:
                pass  # Si falla el clustering, continuar sin él
        
        return result
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in advanced ML analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis ML avanzado: {str(e)}")

