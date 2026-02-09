"""
Servicio de análisis detallado de covers y remixes
"""

import logging
from typing import Dict, List, Any, Optional
from difflib import SequenceMatcher

from .spotify_service import SpotifyService
from .genre_detector import GenreDetector
from .emotion_analyzer import EmotionAnalyzer

logger = logging.getLogger(__name__)


class CoverRemixAnalyzer:
    """Analiza covers y remixes en detalle"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.genre_detector = GenreDetector()
        self.emotion_analyzer = EmotionAnalyzer()
        self.logger = logger
    
    def analyze_cover(self, original_track_id: str, cover_track_id: str) -> Dict[str, Any]:
        """Analiza un cover comparándolo con el original"""
        try:
            # Obtener información de ambos tracks
            original_info = self.spotify.get_track(original_track_id)
            cover_info = self.spotify.get_track(cover_track_id)
            
            original_features = self.spotify.get_track_audio_features(original_track_id)
            cover_features = self.spotify.get_track_audio_features(cover_track_id)
            
            original_analysis = self.spotify.get_track_audio_analysis(original_track_id)
            cover_analysis = self.spotify.get_track_audio_analysis(cover_track_id)
            
            # Comparar características
            comparison = self._compare_tracks_detailed(
                original_info, original_features, original_analysis,
                cover_info, cover_features, cover_analysis
            )
            
            # Determinar tipo de cover
            cover_type = self._identify_cover_type(original_features, cover_features, comparison)
            
            # Análisis de fidelidad
            fidelity = self._calculate_fidelity(original_features, cover_features, comparison)
            
            return {
                "original": {
                    "track_id": original_track_id,
                    "track_name": original_info.get("name", "Unknown"),
                    "artists": [a.get("name") for a in original_info.get("artists", [])]
                },
                "cover": {
                    "track_id": cover_track_id,
                    "track_name": cover_info.get("name", "Unknown"),
                    "artists": [a.get("name") for a in cover_info.get("artists", [])]
                },
                "cover_type": cover_type,
                "fidelity_score": fidelity["score"],
                "fidelity_level": fidelity["level"],
                "comparison": comparison,
                "changes": self._identify_changes(original_features, cover_features, comparison)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing cover: {e}")
            return {"error": str(e)}
    
    def analyze_remix(self, original_track_id: str, remix_track_id: str) -> Dict[str, Any]:
        """Analiza un remix comparándolo con el original"""
        try:
            # Obtener información de ambos tracks
            original_info = self.spotify.get_track(original_track_id)
            remix_info = self.spotify.get_track(remix_track_id)
            
            original_features = self.spotify.get_track_audio_features(original_track_id)
            remix_features = self.spotify.get_track_audio_features(remix_track_id)
            
            original_analysis = self.spotify.get_track_audio_analysis(original_track_id)
            remix_analysis = self.spotify.get_track_audio_analysis(remix_track_id)
            
            # Comparar características
            comparison = self._compare_tracks_detailed(
                original_info, original_features, original_analysis,
                remix_info, remix_features, remix_analysis
            )
            
            # Determinar tipo de remix
            remix_type = self._identify_remix_type(original_features, remix_features, comparison)
            
            # Análisis de transformación
            transformation = self._analyze_transformation(original_features, remix_features, comparison)
            
            return {
                "original": {
                    "track_id": original_track_id,
                    "track_name": original_info.get("name", "Unknown"),
                    "artists": [a.get("name") for a in original_info.get("artists", [])]
                },
                "remix": {
                    "track_id": remix_track_id,
                    "track_name": remix_info.get("name", "Unknown"),
                    "artists": [a.get("name") for a in remix_info.get("artists", [])]
                },
                "remix_type": remix_type,
                "transformation_score": transformation["score"],
                "transformation_level": transformation["level"],
                "comparison": comparison,
                "modifications": self._identify_modifications(original_features, remix_features, comparison)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing remix: {e}")
            return {"error": str(e)}
    
    def find_covers_and_remixes(self, track_id: str, limit: int = 20) -> Dict[str, Any]:
        """Encuentra covers y remixes de un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            track_name = track_info.get("name", "")
            
            # Buscar por nombre del track
            search_results = self.spotify.search_tracks(track_name, limit=limit * 2)
            
            covers = []
            remixes = []
            
            original_features = self.spotify.get_track_audio_features(track_id)
            
            for result in search_results:
                result_id = result.get("id")
                if result_id == track_id:
                    continue
                
                try:
                    result_features = self.spotify.get_track_audio_features(result_id)
                    if not result_features:
                        continue
                    
                    # Comparar características
                    similarity = self._calculate_similarity(original_features, result_features)
                    
                    # Determinar si es cover o remix
                    result_name = result.get("name", "").lower()
                    is_remix = "remix" in result_name or "remix" in result.get("artists", [{}])[0].get("name", "").lower()
                    
                    if is_remix:
                        remixes.append({
                            "track_id": result_id,
                            "track_name": result.get("name", "Unknown"),
                            "artists": [a.get("name") for a in result.get("artists", [])],
                            "similarity": round(similarity, 3),
                            "popularity": result.get("popularity", 0)
                        })
                    elif similarity > 0.6:  # Alta similitud = posible cover
                        covers.append({
                            "track_id": result_id,
                            "track_name": result.get("name", "Unknown"),
                            "artists": [a.get("name") for a in result.get("artists", [])],
                            "similarity": round(similarity, 3),
                            "popularity": result.get("popularity", 0)
                        })
                except:
                    continue
            
            # Ordenar por similitud
            covers.sort(key=lambda x: x["similarity"], reverse=True)
            remixes.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "original_track_id": track_id,
                "original_track_name": track_name,
                "covers": covers[:limit],
                "remixes": remixes[:limit],
                "total_covers": len(covers),
                "total_remixes": len(remixes)
            }
        except Exception as e:
            self.logger.error(f"Error finding covers and remixes: {e}")
            return {"error": str(e)}
    
    def _compare_tracks_detailed(
        self,
        info1: Dict, features1: Dict, analysis1: Dict,
        info2: Dict, features2: Dict, analysis2: Dict
    ) -> Dict[str, Any]:
        """Compara dos tracks en detalle"""
        comparison = {
            "tempo": {
                "original": features1.get("tempo", 120),
                "modified": features2.get("tempo", 120),
                "change": round(features2.get("tempo", 120) - features1.get("tempo", 120), 2),
                "change_percent": round(((features2.get("tempo", 120) - features1.get("tempo", 120)) / features1.get("tempo", 120) * 100), 2) if features1.get("tempo", 120) > 0 else 0
            },
            "energy": {
                "original": round(features1.get("energy", 0.5), 3),
                "modified": round(features2.get("energy", 0.5), 3),
                "change": round(features2.get("energy", 0.5) - features1.get("energy", 0.5), 3)
            },
            "danceability": {
                "original": round(features1.get("danceability", 0.5), 3),
                "modified": round(features2.get("danceability", 0.5), 3),
                "change": round(features2.get("danceability", 0.5) - features1.get("danceability", 0.5), 3)
            },
            "valence": {
                "original": round(features1.get("valence", 0.5), 3),
                "modified": round(features2.get("valence", 0.5), 3),
                "change": round(features2.get("valence", 0.5) - features1.get("valence", 0.5), 3)
            },
            "key": {
                "original": features1.get("key", -1),
                "modified": features2.get("key", -1),
                "changed": features1.get("key", -1) != features2.get("key", -1)
            },
            "duration": {
                "original": info1.get("duration_ms", 0) / 1000,
                "modified": info2.get("duration_ms", 0) / 1000,
                "change": round((info2.get("duration_ms", 0) - info1.get("duration_ms", 0)) / 1000, 2)
            }
        }
        
        return comparison
    
    def _identify_cover_type(self, original_features: Dict, cover_features: Dict, comparison: Dict) -> str:
        """Identifica el tipo de cover"""
        tempo_change = abs(comparison["tempo"]["change_percent"])
        energy_change = abs(comparison["energy"]["change"])
        key_changed = comparison["key"]["changed"]
        
        if tempo_change > 20 or energy_change > 0.3:
            if key_changed:
                return "Radical Cover (Tempo/Energy/Key Changed)"
            else:
                return "Arrangement Cover (Tempo/Energy Changed)"
        elif key_changed:
            return "Transposed Cover (Key Changed)"
        elif tempo_change > 10 or energy_change > 0.15:
            return "Arrangement Cover (Moderate Changes)"
        else:
            return "Faithful Cover (Minimal Changes)"
    
    def _identify_remix_type(self, original_features: Dict, remix_features: Dict, comparison: Dict) -> str:
        """Identifica el tipo de remix"""
        tempo_change = abs(comparison["tempo"]["change_percent"])
        energy_change = abs(comparison["energy"]["change"])
        danceability_change = abs(comparison["danceability"]["change"])
        
        if tempo_change > 30 and energy_change > 0.4:
            return "High Energy Remix"
        elif danceability_change > 0.3:
            return "Dance Remix"
        elif tempo_change > 20:
            return "Tempo Remix"
        elif energy_change > 0.3:
            return "Energy Remix"
        else:
            return "Subtle Remix"
    
    def _calculate_fidelity(self, original_features: Dict, cover_features: Dict, comparison: Dict) -> Dict[str, Any]:
        """Calcula la fidelidad del cover al original"""
        # Factores de fidelidad
        tempo_fidelity = 1.0 - min(abs(comparison["tempo"]["change_percent"]) / 50, 1.0)
        energy_fidelity = 1.0 - abs(comparison["energy"]["change"])
        key_fidelity = 1.0 if not comparison["key"]["changed"] else 0.5
        
        score = (tempo_fidelity * 0.4 + energy_fidelity * 0.4 + key_fidelity * 0.2)
        
        if score > 0.8:
            level = "Very Faithful"
        elif score > 0.6:
            level = "Faithful"
        elif score > 0.4:
            level = "Moderate Changes"
        else:
            level = "Significant Changes"
        
        return {
            "score": round(score, 3),
            "level": level
        }
    
    def _analyze_transformation(self, original_features: Dict, remix_features: Dict, comparison: Dict) -> Dict[str, Any]:
        """Analiza la transformación en el remix"""
        # Factores de transformación
        tempo_transformation = min(abs(comparison["tempo"]["change_percent"]) / 50, 1.0)
        energy_transformation = abs(comparison["energy"]["change"])
        danceability_transformation = abs(comparison["danceability"]["change"])
        
        score = (tempo_transformation * 0.4 + energy_transformation * 0.4 + danceability_transformation * 0.2)
        
        if score > 0.7:
            level = "High Transformation"
        elif score > 0.4:
            level = "Moderate Transformation"
        else:
            level = "Subtle Transformation"
        
        return {
            "score": round(score, 3),
            "level": level
        }
    
    def _identify_changes(self, original_features: Dict, cover_features: Dict, comparison: Dict) -> List[str]:
        """Identifica cambios específicos en el cover"""
        changes = []
        
        if abs(comparison["tempo"]["change_percent"]) > 10:
            changes.append(f"Tempo: {comparison['tempo']['change']:.1f} BPM ({comparison['tempo']['change_percent']:.1f}%)")
        
        if abs(comparison["energy"]["change"]) > 0.2:
            changes.append(f"Energy: {comparison['energy']['change']:.2f}")
        
        if comparison["key"]["changed"]:
            changes.append("Key changed")
        
        if abs(comparison["duration"]["change"]) > 10:
            changes.append(f"Duration: {comparison['duration']['change']:.1f}s")
        
        return changes
    
    def _identify_modifications(self, original_features: Dict, remix_features: Dict, comparison: Dict) -> List[str]:
        """Identifica modificaciones específicas en el remix"""
        modifications = []
        
        if abs(comparison["tempo"]["change_percent"]) > 15:
            modifications.append(f"Tempo modified: {comparison['tempo']['change']:.1f} BPM")
        
        if abs(comparison["energy"]["change"]) > 0.3:
            modifications.append(f"Energy modified: {comparison['energy']['change']:.2f}")
        
        if abs(comparison["danceability"]["change"]) > 0.2:
            modifications.append(f"Danceability modified: {comparison['danceability']['change']:.2f}")
        
        if abs(comparison["valence"]["change"]) > 0.2:
            modifications.append(f"Mood modified: {comparison['valence']['change']:.2f}")
        
        return modifications
    
    def _calculate_similarity(self, features1: Dict, features2: Dict) -> float:
        """Calcula similitud entre tracks"""
        keys = ["energy", "danceability", "valence", "tempo"]
        similarities = []
        
        for key in keys:
            if key in features1 and key in features2:
                if key == "tempo":
                    diff = abs(features1[key] - features2[key]) / 200
                    sim = 1.0 - min(diff, 1.0)
                else:
                    diff = abs(features1[key] - features2[key])
                    sim = 1.0 - diff
                
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0

