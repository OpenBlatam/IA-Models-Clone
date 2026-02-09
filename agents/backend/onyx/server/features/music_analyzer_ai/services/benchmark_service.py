"""
Servicio de benchmarking musical
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from .spotify_service import SpotifyService
from .music_analyzer import MusicAnalyzer

logger = logging.getLogger(__name__)


class BenchmarkService:
    """Servicio para benchmarking de tracks y análisis"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.music_analyzer = MusicAnalyzer()
        self.logger = logger
    
    def benchmark_track(self, track_id: str, reference_tracks: List[str]) -> Dict[str, Any]:
        """Compara un track con tracks de referencia"""
        try:
            # Obtener análisis del track
            track_data = self.spotify.get_track_full_analysis(track_id)
            track_analysis = self.music_analyzer.analyze_track(track_data)
            
            # Obtener análisis de tracks de referencia
            reference_analyses = []
            for ref_id in reference_tracks[:10]:  # Limitar a 10
                try:
                    ref_data = self.spotify.get_track_full_analysis(ref_id)
                    ref_analysis = self.music_analyzer.analyze_track(ref_data)
                    reference_analyses.append({
                        "track_id": ref_id,
                        "analysis": ref_analysis
                    })
                except:
                    continue
            
            if not reference_analyses:
                return {"error": "No se pudieron analizar tracks de referencia"}
            
            # Comparar con referencias
            comparison = self._compare_with_references(track_analysis, reference_analyses)
            
            return {
                "track_id": track_id,
                "reference_count": len(reference_analyses),
                "benchmark": comparison,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Error benchmarking track: {e}")
            return {"error": str(e)}
    
    def _compare_with_references(self, track_analysis: Dict, reference_analyses: List[Dict]) -> Dict[str, Any]:
        """Compara análisis con referencias"""
        # Extraer características del track
        track_features = self._extract_features(track_analysis)
        
        # Extraer características de referencias
        reference_features = [self._extract_features(ref["analysis"]) for ref in reference_analyses]
        
        # Calcular promedios de referencia
        avg_features = {}
        for key in track_features.keys():
            values = [rf.get(key, 0) for rf in reference_features if key in rf]
            if values:
                avg_features[key] = sum(values) / len(values)
        
        # Comparar
        comparison = {}
        for key, track_value in track_features.items():
            ref_value = avg_features.get(key, track_value)
            difference = track_value - ref_value
            percentage_diff = (difference / ref_value * 100) if ref_value > 0 else 0
            
            comparison[key] = {
                "track_value": round(track_value, 3),
                "reference_average": round(ref_value, 3),
                "difference": round(difference, 3),
                "percentage_difference": round(percentage_diff, 2),
                "status": "Above" if difference > 0.1 else "Below" if difference < -0.1 else "Similar"
            }
        
        # Score general
        above_count = sum(1 for c in comparison.values() if c["status"] == "Above")
        total_count = len(comparison)
        benchmark_score = above_count / total_count if total_count > 0 else 0
        
        return {
            "comparison": comparison,
            "benchmark_score": round(benchmark_score, 3),
            "benchmark_level": "High" if benchmark_score > 0.6 else "Medium" if benchmark_score > 0.4 else "Low",
            "summary": {
                "above_reference": above_count,
                "similar_to_reference": sum(1 for c in comparison.values() if c["status"] == "Similar"),
                "below_reference": sum(1 for c in comparison.values() if c["status"] == "Below")
            }
        }
    
    def _extract_features(self, analysis: Dict) -> Dict[str, float]:
        """Extrae características numéricas del análisis"""
        features = {}
        
        # Características técnicas
        technical = analysis.get("technical_analysis", {})
        if technical:
            energy = technical.get("energy", {}).get("value", 0.5)
            danceability = technical.get("danceability", {}).get("value", 0.5)
            valence = technical.get("valence", {}).get("value", 0.5)
            
            features["energy"] = energy
            features["danceability"] = danceability
            features["valence"] = valence
        
        # Características musicales
        musical = analysis.get("musical_analysis", {})
        if musical:
            tempo = musical.get("tempo", {}).get("bpm", 120)
            features["tempo"] = tempo / 200.0  # Normalizar
        
        return features
    
    def create_benchmark_set(self, genre: str, limit: int = 20) -> Dict[str, Any]:
        """Crea un conjunto de referencia para benchmarking"""
        try:
            # Buscar tracks del género
            tracks = self.spotify.search_tracks(f"genre:{genre}", limit=limit)
            
            if not tracks:
                return {"error": f"No se encontraron tracks del género {genre}"}
            
            # Filtrar por popularidad
            popular_tracks = [t for t in tracks if t.get("popularity", 0) > 50]
            
            track_ids = [t.get("id") for t in popular_tracks[:10]]
            
            return {
                "genre": genre,
                "track_ids": track_ids,
                "track_count": len(track_ids),
                "description": f"Benchmark set for {genre} genre with {len(track_ids)} popular tracks"
            }
        except Exception as e:
            self.logger.error(f"Error creating benchmark set: {e}")
            return {"error": str(e)}

