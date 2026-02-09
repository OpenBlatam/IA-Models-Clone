"""
Servicio para comparar canciones
"""

import logging
from typing import Dict, List, Any, Optional

from ..core.music_analyzer import MusicAnalyzer
from ..services.spotify_service import SpotifyService
from ..utils.exceptions import TrackNotFoundException

logger = logging.getLogger(__name__)


class ComparisonService:
    """Servicio para comparar múltiples canciones"""
    
    def __init__(self):
        self.analyzer = MusicAnalyzer()
        self.spotify = SpotifyService()
        self.logger = logger
    
    def compare_tracks(self, track_ids: List[str]) -> Dict[str, Any]:
        """Compara múltiples canciones"""
        if len(track_ids) < 2:
            raise ValueError("Se necesitan al menos 2 canciones para comparar")
        
        if len(track_ids) > 10:
            raise ValueError("No se pueden comparar más de 10 canciones a la vez")
        
        analyses = []
        for track_id in track_ids:
            try:
                spotify_data = self.spotify.get_track_full_analysis(track_id)
                analysis = self.analyzer.analyze_track(spotify_data)
                analyses.append({
                    "track_id": track_id,
                    "track_name": analysis["track_basic_info"]["name"],
                    "analysis": analysis
                })
            except Exception as e:
                self.logger.error(f"Error analizando track {track_id}: {e}")
                raise TrackNotFoundException(f"Error al analizar track {track_id}: {str(e)}")
        
        return {
            "comparison": self._generate_comparison(analyses),
            "tracks": analyses,
            "similarities": self._find_similarities(analyses),
            "differences": self._find_differences(analyses),
            "recommendations": self._generate_comparison_recommendations(analyses)
        }
    
    def _generate_comparison(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Genera un análisis comparativo"""
        if not analyses:
            return {}
        
        # Comparar tonalidades
        keys = [a["analysis"]["musical_analysis"]["key_signature"] for a in analyses]
        key_comparison = {
            "all_same": len(set(keys)) == 1,
            "keys": keys,
            "most_common": max(set(keys), key=keys.count) if keys else None
        }
        
        # Comparar tempos
        tempos = [a["analysis"]["musical_analysis"]["tempo"]["bpm"] for a in analyses]
        tempo_comparison = {
            "average": sum(tempos) / len(tempos) if tempos else 0,
            "min": min(tempos) if tempos else 0,
            "max": max(tempos) if tempos else 0,
            "range": max(tempos) - min(tempos) if tempos else 0,
            "tempos": tempos
        }
        
        # Comparar energía
        energies = [a["analysis"]["technical_analysis"]["energy"]["value"] for a in analyses]
        energy_comparison = {
            "average": sum(energies) / len(energies) if energies else 0,
            "min": min(energies) if energies else 0,
            "max": max(energies) if energies else 0,
            "energies": energies
        }
        
        # Comparar bailabilidad
        danceabilities = [a["analysis"]["technical_analysis"]["danceability"]["value"] for a in analyses]
        danceability_comparison = {
            "average": sum(danceabilities) / len(danceabilities) if danceabilities else 0,
            "min": min(danceabilities) if danceabilities else 0,
            "max": max(danceabilities) if danceabilities else 0,
            "danceabilities": danceabilities
        }
        
        # Comparar complejidad
        complexities = [a["analysis"]["composition_analysis"]["complexity"]["level"] for a in analyses]
        complexity_comparison = {
            "levels": complexities,
            "most_common": max(set(complexities), key=complexities.count) if complexities else None
        }
        
        return {
            "key_signatures": key_comparison,
            "tempos": tempo_comparison,
            "energy": energy_comparison,
            "danceability": danceability_comparison,
            "complexity": complexity_comparison,
            "total_tracks": len(analyses)
        }
    
    def _find_similarities(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra similitudes entre las canciones"""
        similarities = []
        
        if len(analyses) < 2:
            return similarities
        
        # Similitud en tonalidad
        keys = [a["analysis"]["musical_analysis"]["key_signature"] for a in analyses]
        if len(set(keys)) == 1:
            similarities.append({
                "aspect": "Tonalidad",
                "description": f"Todas las canciones están en {keys[0]}",
                "tracks": [a["track_name"] for a in analyses]
            })
        
        # Similitud en tempo (dentro de 10 BPM)
        tempos = [a["analysis"]["musical_analysis"]["tempo"]["bpm"] for a in analyses]
        if max(tempos) - min(tempos) <= 10:
            similarities.append({
                "aspect": "Tempo",
                "description": f"Tempos similares (rango: {min(tempos):.1f}-{max(tempos):.1f} BPM)",
                "tracks": [a["track_name"] for a in analyses]
            })
        
        # Similitud en energía (dentro de 0.2)
        energies = [a["analysis"]["technical_analysis"]["energy"]["value"] for a in analyses]
        if max(energies) - min(energies) <= 0.2:
            similarities.append({
                "aspect": "Energía",
                "description": "Niveles de energía similares",
                "tracks": [a["track_name"] for a in analyses]
            })
        
        return similarities
    
    def _find_differences(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Encuentra diferencias significativas entre las canciones"""
        differences = []
        
        if len(analyses) < 2:
            return differences
        
        # Diferencia en tempo
        tempos = [a["analysis"]["musical_analysis"]["tempo"]["bpm"] for a in analyses]
        if max(tempos) - min(tempos) > 30:
            fastest = max(analyses, key=lambda x: x["analysis"]["musical_analysis"]["tempo"]["bpm"])
            slowest = min(analyses, key=lambda x: x["analysis"]["musical_analysis"]["tempo"]["bpm"])
            differences.append({
                "aspect": "Tempo",
                "description": f"Gran diferencia de tempo: {slowest['track_name']} ({slowest['analysis']['musical_analysis']['tempo']['bpm']:.1f} BPM) vs {fastest['track_name']} ({fastest['analysis']['musical_analysis']['tempo']['bpm']:.1f} BPM)",
                "difference": max(tempos) - min(tempos)
            })
        
        # Diferencia en energía
        energies = [a["analysis"]["technical_analysis"]["energy"]["value"] for a in analyses]
        if max(energies) - min(energies) > 0.4:
            highest = max(analyses, key=lambda x: x["analysis"]["technical_analysis"]["energy"]["value"])
            lowest = min(analyses, key=lambda x: x["analysis"]["technical_analysis"]["energy"]["value"])
            differences.append({
                "aspect": "Energía",
                "description": f"Gran diferencia de energía: {lowest['track_name']} (baja) vs {highest['track_name']} (alta)",
                "difference": max(energies) - min(energies)
            })
        
        return differences
    
    def _generate_comparison_recommendations(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Genera recomendaciones basadas en la comparación"""
        recommendations = []
        
        comparison = self._generate_comparison(analyses)
        
        # Recomendación basada en tonalidad
        if not comparison["key_signatures"]["all_same"]:
            recommendations.append(
                "Las canciones están en diferentes tonalidades. "
                "Practica modulaciones entre estas tonalidades."
            )
        
        # Recomendación basada en tempo
        tempo_range = comparison["tempos"]["range"]
        if tempo_range > 30:
            recommendations.append(
                f"Las canciones tienen un rango amplio de tempos ({comparison['tempos']['min']:.1f}-{comparison['tempos']['max']:.1f} BPM). "
                "Practica con metrónomo en diferentes velocidades."
            )
        
        # Recomendación basada en complejidad
        complexities = comparison["complexity"]["levels"]
        if len(set(complexities)) > 1:
            recommendations.append(
                "Las canciones tienen diferentes niveles de complejidad. "
                "Empieza con las más simples y progresa gradualmente."
            )
        
        return recommendations

