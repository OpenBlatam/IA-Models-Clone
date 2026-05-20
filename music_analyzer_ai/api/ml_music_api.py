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
from ..services.advanced_ml_service import AdvancedMLService
from ..utils.exceptions import TrackNotFoundException, InvalidTrackIDException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/music/ml", tags=["Machine Learning"])

# Inicializar servicios
spotify_service = SpotifyService()
music_analyzer = MusicAnalyzer()
intelligent_recommender = IntelligentRecommender()
emotion_analyzer = EmotionAnalyzer()
genre_detector = GenreDetector()
advanced_ml_service = AdvancedMLService()


@router.get("/")
async def ml_root():
    """Endpoint raíz de ML"""
    return {
        "service": "Music Analyzer AI - ML Features",
        "version": "2.1.0",
        "status": "running",
        "features": [
            "Intelligent Recommendations",
            "Emotion Analysis",
            "Genre Detection",
            "Harmonic Analysis",
            "Similarity Matching"
        ]
    }


@router.post("/predict/genre", response_model=dict)
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


@router.post("/predict/emotion", response_model=dict)
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


@router.post("/similarity/batch", response_model=dict)
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


@router.post("/cluster/genres", response_model=dict)
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


@router.post("/cluster/emotions", response_model=dict)
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


@router.post("/analyze/advanced", response_model=dict)
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


@router.post("/analyze-comprehensive", response_model=dict)
async def analyze_comprehensive(track_id: str):
    """Análisis ML comprehensivo de una canción con todas las predicciones"""
    try:
        # Obtener datos completos
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        analysis = music_analyzer.analyze_track(spotify_data)
        
        # Obtener recomendaciones similares
        recommendations = spotify_service.get_recommendations(track_id, limit=10)
        
        # Análisis comparativo con recomendaciones
        comparison_data = []
        for rec in recommendations[:5]:
            try:
                rec_features = spotify_service.get_track_audio_features(rec.get("id"))
                similarity = intelligent_recommender._calculate_similarity(
                    spotify_data.get("audio_features", {}),
                    rec_features
                )
                comparison_data.append({
                    "track_id": rec.get("id"),
                    "track_name": rec.get("name"),
                    "similarity": round(similarity, 3)
                })
            except:
                continue
        
        return {
            "success": True,
            "track_id": track_id,
            "comprehensive_analysis": {
                "basic_info": analysis["track_basic_info"],
                "musical": analysis["musical_analysis"],
                "technical": analysis["technical_analysis"],
                "composition": analysis["composition_analysis"],
                "performance": analysis["performance_analysis"],
                "genre": analysis.get("genre_analysis", {}),
                "emotion": analysis.get("emotion_analysis", {}),
                "harmonic": analysis.get("harmonic_analysis", {}),
                "educational": analysis["educational_insights"]
            },
            "similar_tracks": comparison_data,
            "insights": {
                "primary_genre": analysis.get("genre_analysis", {}).get("primary_genre", "Unknown"),
                "primary_emotion": analysis.get("emotion_analysis", {}).get("primary_emotion", "Unknown"),
                "key_signature": analysis["musical_analysis"].get("key_signature", "Unknown"),
                "tempo": analysis["musical_analysis"].get("tempo", {}).get("bpm", 0),
                "complexity": analysis["composition_analysis"].get("complexity", {}).get("level", "Unknown")
            }
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Error en análisis comprehensivo: {str(e)}")


@router.post("/compare-tracks", response_model=dict)
async def compare_tracks_ml(
    track_ids: List[str],
    comparison_type: str = Query("all", regex="^(all|genre|emotion|harmonic|technical)$")
):
    """Compara múltiples tracks usando ML"""
    try:
        if len(track_ids) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tracks")
        
        if len(track_ids) > 20:
            raise HTTPException(status_code=400, detail="Máximo 20 tracks")
        
        tracks_data = []
        
        for track_id in track_ids:
            try:
                spotify_data = spotify_service.get_track_full_analysis(track_id)
                analysis = music_analyzer.analyze_track(spotify_data)
                
                tracks_data.append({
                    "track_id": track_id,
                    "track_name": analysis["track_basic_info"]["name"],
                    "analysis": analysis
                })
            except:
                continue
        
        if len(tracks_data) < 2:
            raise HTTPException(status_code=400, detail="No se pudieron analizar suficientes tracks")
        
        # Comparaciones
        comparisons = {}
        
        if comparison_type in ["all", "genre"]:
            genres = [t["analysis"].get("genre_analysis", {}).get("primary_genre", "Unknown") for t in tracks_data]
            comparisons["genres"] = {
                "values": genres,
                "unique_count": len(set(genres)),
                "most_common": max(set(genres), key=genres.count) if genres else None
            }
        
        if comparison_type in ["all", "emotion"]:
            emotions = [t["analysis"].get("emotion_analysis", {}).get("primary_emotion", "Unknown") for t in tracks_data]
            comparisons["emotions"] = {
                "values": emotions,
                "unique_count": len(set(emotions)),
                "most_common": max(set(emotions), key=emotions.count) if emotions else None
            }
        
        if comparison_type in ["all", "harmonic"]:
            keys = [t["analysis"]["musical_analysis"].get("key_signature", "Unknown") for t in tracks_data]
            comparisons["keys"] = {
                "values": keys,
                "unique_count": len(set(keys)),
                "most_common": max(set(keys), key=keys.count) if keys else None
            }
        
        if comparison_type in ["all", "technical"]:
            energies = [t["analysis"]["technical_analysis"]["energy"]["value"] for t in tracks_data]
            tempos = [t["analysis"]["musical_analysis"]["tempo"]["bpm"] for t in tracks_data]
            
            comparisons["technical"] = {
                "energy": {
                    "values": energies,
                    "average": sum(energies) / len(energies) if energies else 0,
                    "min": min(energies) if energies else 0,
                    "max": max(energies) if energies else 0
                },
                "tempo": {
                    "values": tempos,
                    "average": sum(tempos) / len(tempos) if tempos else 0,
                    "min": min(tempos) if tempos else 0,
                    "max": max(tempos) if tempos else 0
                }
            }
        
        # Calcular similitud entre todos los pares
        similarity_matrix = []
        for i, track1 in enumerate(tracks_data):
            for track2 in tracks_data[i+1:]:
                features1 = spotify_service.get_track_audio_features(track1["track_id"])
                features2 = spotify_service.get_track_audio_features(track2["track_id"])
                similarity = intelligent_recommender._calculate_similarity(features1, features2)
                
                similarity_matrix.append({
                    "track1_id": track1["track_id"],
                    "track1_name": track1["track_name"],
                    "track2_id": track2["track_id"],
                    "track2_name": track2["track_name"],
                    "similarity": round(similarity, 3)
                })
        
        similarity_matrix.sort(key=lambda x: x["similarity"], reverse=True)
        
        return {
            "success": True,
            "tracks_compared": len(tracks_data),
            "comparisons": comparisons,
            "similarity_matrix": similarity_matrix,
            "most_similar_pair": similarity_matrix[0] if similarity_matrix else None,
            "least_similar_pair": similarity_matrix[-1] if similarity_matrix else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing tracks: {e}")
        raise HTTPException(status_code=500, detail=f"Error al comparar tracks: {str(e)}")


@router.post("/predict/multi-task", response_model=dict)
async def predict_multi_task(track_id: str):
    """Predicción multi-tarea: género, emoción, complejidad, etc."""
    try:
        spotify_data = spotify_service.get_track_full_analysis(track_id)
        analysis = music_analyzer.analyze_track(spotify_data)
        
        # Extraer todas las predicciones
        predictions = {
            "genre": {
                "primary": analysis.get("genre_analysis", {}).get("primary_genre", "Unknown"),
                "confidence": analysis.get("genre_analysis", {}).get("confidence", 0),
                "top_3": analysis.get("genre_analysis", {}).get("top_3", [])
            },
            "emotion": {
                "primary": analysis.get("emotion_analysis", {}).get("primary_emotion", "Unknown"),
                "confidence": analysis.get("emotion_analysis", {}).get("confidence", 0),
                "top_3": analysis.get("emotion_analysis", {}).get("top_3", []),
                "profile": analysis.get("emotion_analysis", {}).get("emotional_profile", {})
            },
            "complexity": {
                "level": analysis["composition_analysis"].get("complexity", {}).get("level", "Unknown"),
                "score": analysis["composition_analysis"].get("complexity", {}).get("score", 0),
                "factors": analysis["composition_analysis"].get("complexity", {}).get("factors", {})
            },
            "harmonic": {
                "progressions_found": len(analysis.get("harmonic_analysis", {}).get("progressions", [])),
                "cadences_found": len(analysis.get("harmonic_analysis", {}).get("cadences", [])),
                "complexity": analysis.get("harmonic_analysis", {}).get("harmonic_complexity", {})
            },
            "difficulty": {
                "level": "Unknown"  # Se calculará basado en múltiples factores
            }
        }
        
        # Calcular dificultad basada en múltiples factores
        complexity_score = predictions["complexity"]["score"]
        harmonic_complexity = predictions["harmonic"]["complexity"].get("score", 0) if predictions["harmonic"]["complexity"] else 0
        tempo = analysis["musical_analysis"]["tempo"]["bpm"]
        
        difficulty_score = (
            complexity_score * 0.4 +
            harmonic_complexity * 0.3 +
            (tempo / 200) * 0.3  # Normalizar tempo
        )
        
        if difficulty_score < 0.3:
            difficulty_level = "Beginner"
        elif difficulty_score < 0.6:
            difficulty_level = "Intermediate"
        else:
            difficulty_level = "Advanced"
        
        predictions["difficulty"]["level"] = difficulty_level
        predictions["difficulty"]["score"] = round(difficulty_score, 3)
        
        return {
            "success": True,
            "track_id": track_id,
            "track_name": analysis["track_basic_info"]["name"],
            "multi_task_predictions": predictions,
            "summary": {
                "genre": predictions["genre"]["primary"],
                "emotion": predictions["emotion"]["primary"],
                "complexity": predictions["complexity"]["level"],
                "difficulty": predictions["difficulty"]["level"]
            }
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in multi-task prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error en predicción multi-tarea: {str(e)}")


@router.get("/pipeline/info", response_model=dict)
async def get_pipeline_info():
    """Obtiene información sobre el pipeline de ML"""
    return {
        "success": True,
        "pipeline": {
            "name": "Music Analyzer AI ML Pipeline",
            "version": "2.1.0",
            "stages": [
                {
                    "stage": 1,
                    "name": "Data Collection",
                    "description": "Recopilación de datos de Spotify API",
                    "output": "Audio features, audio analysis, track info"
                },
                {
                    "stage": 2,
                    "name": "Feature Extraction",
                    "description": "Extracción de características musicales",
                    "output": "Tonalidad, tempo, energía, valencia, etc."
                },
                {
                    "stage": 3,
                    "name": "Genre Detection",
                    "description": "Clasificación de género usando características de audio",
                    "output": "Género primario y top 3 géneros con confianza"
                },
                {
                    "stage": 4,
                    "name": "Emotion Analysis",
                    "description": "Análisis de emociones basado en valence y energy",
                    "output": "Emoción primaria y perfil emocional"
                },
                {
                    "stage": 5,
                    "name": "Harmonic Analysis",
                    "description": "Análisis de progresiones armónicas y cadencias",
                    "output": "Progresiones, cadencias, patrones armónicos"
                },
                {
                    "stage": 6,
                    "name": "Similarity Calculation",
                    "description": "Cálculo de similitud entre tracks",
                    "output": "Scores de similitud basados en características"
                },
                {
                    "stage": 7,
                    "name": "Recommendation Generation",
                    "description": "Generación de recomendaciones inteligentes",
                    "output": "Tracks recomendados con scores"
                }
            ],
            "models": {
                "genre_detector": {
                    "type": "Rule-based classifier",
                    "genres_supported": 12,
                    "accuracy": "High (based on audio features)"
                },
                "emotion_analyzer": {
                    "type": "Rule-based classifier",
                    "emotions_supported": 8,
                    "accuracy": "High (based on valence/energy)"
                },
                "harmonic_analyzer": {
                    "type": "Pattern matching",
                    "features": "Progression detection, cadence analysis"
                },
                "similarity_calculator": {
                    "type": "Distance-based",
                    "method": "Multi-attribute similarity"
                }
            },
            "capabilities": [
                "Genre prediction",
                "Emotion detection",
                "Harmonic analysis",
                "Similarity matching",
                "Clustering",
                "Recommendation generation",
                "Multi-task prediction",
                "Style analysis",
                "Era prediction",
                "Influence detection",
                "Diversity calculation"
            ]
        }
    }


@router.post("/analyze/style", response_model=dict)
async def analyze_music_style(track_id: str):
    """Analiza el estilo musical completo"""
    try:
        audio_features = spotify_service.get_track_audio_features(track_id)
        style_analysis = advanced_ml_service.analyze_music_style(audio_features)
        
        return {
            "success": True,
            "track_id": track_id,
            "style_analysis": style_analysis
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing style: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar estilo: {str(e)}")


@router.post("/predict/era", response_model=dict)
async def predict_era(track_id: str):
    """Predice la era musical de una canción"""
    try:
        track_info = spotify_service.get_track(track_id)
        audio_features = spotify_service.get_track_audio_features(track_id)
        
        era_prediction = advanced_ml_service.predict_musical_era(audio_features, track_info)
        
        return {
            "success": True,
            "track_id": track_id,
            "era_prediction": era_prediction
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error predicting era: {e}")
        raise HTTPException(status_code=500, detail=f"Error al predecir era: {str(e)}")


@router.post("/analyze/influences", response_model=dict)
async def analyze_influences(track_id: str):
    """Analiza posibles influencias musicales"""
    try:
        audio_features = spotify_service.get_track_audio_features(track_id)
        genre_analysis = genre_detector.detect_genre(audio_features)
        
        influences = advanced_ml_service.analyze_musical_influences(audio_features, genre_analysis)
        
        return {
            "success": True,
            "track_id": track_id,
            "influences": influences
        }
    except TrackNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error analyzing influences: {e}")
        raise HTTPException(status_code=500, detail=f"Error al analizar influencias: {str(e)}")


@router.post("/diversity/calculate", response_model=dict)
async def calculate_diversity(track_ids: List[str]):
    """Calcula la diversidad musical de un conjunto de tracks"""
    try:
        if len(track_ids) < 2:
            raise HTTPException(status_code=400, detail="Se necesitan al menos 2 tracks")
        
        if len(track_ids) > 50:
            raise HTTPException(status_code=400, detail="Máximo 50 tracks")
        
        # Analizar todos los tracks
        tracks_data = []
        for track_id in track_ids:
            try:
                spotify_data = spotify_service.get_track_full_analysis(track_id)
                analysis = music_analyzer.analyze_track(spotify_data)
                tracks_data.append({
                    "track_id": track_id,
                    "analysis": analysis
                })
            except:
                continue
        
        if len(tracks_data) < 2:
            raise HTTPException(status_code=400, detail="No se pudieron analizar suficientes tracks")
        
        diversity = advanced_ml_service.calculate_musical_diversity(tracks_data)
        
        return {
            "success": True,
            "tracks_analyzed": len(tracks_data),
            "diversity_analysis": diversity
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating diversity: {e}")
        raise HTTPException(status_code=500, detail=f"Error al calcular diversidad: {str(e)}")


@router.get("/performance/stats")
async def get_performance_stats():
    """Obtiene estadísticas de rendimiento del sistema"""
    try:
        from ..performance.profiler import PerformanceMonitor
        
        stats = {
            "system_info": PerformanceMonitor.get_system_info(),
            "gpu_memory": PerformanceMonitor.get_gpu_memory(),
            "cpu_usage": PerformanceMonitor.get_cpu_usage()
        }
        
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.post("/performance/benchmark")
async def benchmark_model(
    model_name: str,
    num_runs: int = Query(100, ge=1, le=1000)
):
    """Benchmark de un modelo"""
    try:
        from ..performance.profiler import Benchmark
        from ..core.deep_models import get_deep_analyzer
        
        analyzer = get_deep_analyzer()
        model = analyzer.models.get(model_name)
        
        if model is None:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        # Benchmark
        result = Benchmark.benchmark_model(
            model,
            input_shape=(1, 169),
            num_runs=num_runs
        )
        
        return {
            "success": True,
            "model_name": model_name,
            "benchmark": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error benchmarking model: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

