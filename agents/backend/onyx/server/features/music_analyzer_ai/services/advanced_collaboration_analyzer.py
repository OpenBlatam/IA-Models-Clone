"""
Servicio de análisis avanzado de colaboraciones
"""

import logging
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from .spotify_service import SpotifyService
from .collaboration_analyzer import CollaborationAnalyzer

logger = logging.getLogger(__name__)


class AdvancedCollaborationAnalyzer:
    """Análisis avanzado de colaboraciones musicales"""
    
    def __init__(self):
        self.spotify = SpotifyService()
        self.collaboration_analyzer = CollaborationAnalyzer()
        self.logger = logger
    
    def analyze_advanced_collaborations(self, track_id: str) -> Dict[str, Any]:
        """Análisis avanzado de colaboraciones en un track"""
        try:
            track_info = self.spotify.get_track(track_id)
            
            if not track_info:
                return {"error": "No hay datos del track disponibles"}
            
            artists = track_info.get("artists", [])
            
            if len(artists) < 2:
                return {
                    "track_id": track_id,
                    "is_collaboration": False,
                    "message": "Track no es una colaboración"
                }
            
            # Análisis avanzado
            analysis = {
                "collaboration_info": self._analyze_collaboration_info(artists),
                "artist_compatibility": self._analyze_artist_compatibility(artists),
                "collaboration_history": self._analyze_collaboration_history(artists),
                "genre_alignment": self._analyze_genre_alignment(artists),
                "popularity_analysis": self._analyze_popularity_balance(artists, track_info)
            }
            
            return {
                "track_id": track_id,
                "track_name": track_info.get("name", "Unknown"),
                "is_collaboration": True,
                "artist_count": len(artists),
                "advanced_analysis": analysis
            }
        except Exception as e:
            self.logger.error(f"Error analyzing advanced collaborations: {e}")
            return {"error": str(e)}
    
    def _analyze_collaboration_info(self, artists: List[Dict]) -> Dict[str, Any]:
        """Analiza información de la colaboración"""
        artist_names = [a.get("name", "Unknown") for a in artists]
        
        return {
            "artists": artist_names,
            "collaboration_type": "Duo" if len(artists) == 2 else "Group" if len(artists) <= 5 else "Large Group",
            "primary_artist": artist_names[0] if artist_names else "Unknown",
            "featured_artists": artist_names[1:] if len(artist_names) > 1 else []
        }
    
    def _analyze_artist_compatibility(self, artists: List[Dict]) -> Dict[str, Any]:
        """Analiza compatibilidad entre artistas"""
        if len(artists) < 2:
            return {"error": "Se requieren al menos 2 artistas"}
        
        # Obtener información de artistas
        artist_data = []
        for artist in artists[:5]:  # Limitar a 5
            try:
                artist_id = artist.get("id")
                if artist_id:
                    artist_info = self.spotify.get_artist(artist_id)
                    if artist_info:
                        artist_data.append({
                            "id": artist_id,
                            "name": artist_info.get("name", "Unknown"),
                            "popularity": artist_info.get("popularity", 0),
                            "genres": artist_info.get("genres", [])
                        })
            except:
                continue
        
        if len(artist_data) < 2:
            return {"error": "No se pudo obtener información suficiente de los artistas"}
        
        # Analizar compatibilidad
        compatibilities = []
        for i in range(len(artist_data)):
            for j in range(i+1, len(artist_data)):
                artist1 = artist_data[i]
                artist2 = artist_data[j]
                
                # Calcular compatibilidad basada en géneros
                genres1 = set(artist1.get("genres", []))
                genres2 = set(artist2.get("genres", []))
                common_genres = genres1.intersection(genres2)
                
                genre_compatibility = len(common_genres) / max(len(genres1), len(genres2), 1)
                
                # Compatibilidad de popularidad (artistas de niveles similares)
                pop1 = artist1.get("popularity", 0)
                pop2 = artist2.get("popularity", 0)
                pop_diff = abs(pop1 - pop2)
                pop_compatibility = 1.0 - (pop_diff / 100.0) if pop_diff < 100 else 0.0
                
                total_compatibility = (genre_compatibility * 0.6 + pop_compatibility * 0.4)
                
                compatibilities.append({
                    "artist1": artist1["name"],
                    "artist2": artist2["name"],
                    "compatibility_score": round(total_compatibility, 3),
                    "common_genres": list(common_genres),
                    "genre_compatibility": round(genre_compatibility, 3),
                    "popularity_compatibility": round(pop_compatibility, 3)
                })
        
        avg_compatibility = sum(c["compatibility_score"] for c in compatibilities) / len(compatibilities) if compatibilities else 0
        
        return {
            "pair_compatibilities": compatibilities,
            "average_compatibility": round(avg_compatibility, 3),
            "compatibility_level": "High" if avg_compatibility > 0.7 else "Medium" if avg_compatibility > 0.4 else "Low"
        }
    
    def _analyze_collaboration_history(self, artists: List[Dict]) -> Dict[str, Any]:
        """Analiza historial de colaboraciones"""
        # Simplificado: buscar tracks de los artistas juntos
        artist_names = [a.get("name", "") for a in artists[:2]]
        
        if len(artist_names) < 2:
            return {"error": "Se requieren al menos 2 artistas"}
        
        # Buscar colaboraciones previas
        query = f"{artist_names[0]} {artist_names[1]}"
        previous_tracks = self.spotify.search_tracks(query, limit=10)
        
        collaboration_count = len([t for t in previous_tracks if len(t.get("artists", [])) >= 2])
        
        return {
            "previous_collaborations_found": collaboration_count > 1,  # Excluyendo el track actual
            "estimated_collaboration_count": max(0, collaboration_count - 1),
            "collaboration_frequency": "Frequent" if collaboration_count > 3 else "Occasional" if collaboration_count > 1 else "First Time"
        }
    
    def _analyze_genre_alignment(self, artists: List[Dict]) -> Dict[str, Any]:
        """Analiza alineación de géneros"""
        artist_genres = []
        for artist in artists[:5]:
            try:
                artist_id = artist.get("id")
                if artist_id:
                    artist_info = self.spotify.get_artist(artist_id)
                    if artist_info:
                        artist_genres.append(set(artist_info.get("genres", [])))
            except:
                continue
        
        if not artist_genres:
            return {"error": "No se pudieron obtener géneros"}
        
        # Encontrar géneros comunes
        if len(artist_genres) > 1:
            common_genres = set.intersection(*artist_genres)
            all_genres = set.union(*artist_genres)
        else:
            common_genres = artist_genres[0] if artist_genres else set()
            all_genres = artist_genres[0] if artist_genres else set()
        
        alignment_score = len(common_genres) / len(all_genres) if all_genres else 0
        
        return {
            "common_genres": list(common_genres),
            "all_genres": list(all_genres),
            "alignment_score": round(alignment_score, 3),
            "alignment_level": "High" if alignment_score > 0.5 else "Medium" if alignment_score > 0.2 else "Low"
        }
    
    def _analyze_popularity_balance(self, artists: List[Dict], track_info: Dict) -> Dict[str, Any]:
        """Analiza balance de popularidad"""
        artist_popularities = []
        for artist in artists:
            try:
                artist_id = artist.get("id")
                if artist_id:
                    artist_info = self.spotify.get_artist(artist_id)
                    if artist_info:
                        artist_popularities.append({
                            "name": artist_info.get("name", "Unknown"),
                            "popularity": artist_info.get("popularity", 0)
                        })
            except:
                continue
        
        if not artist_popularities:
            return {"error": "No se pudieron obtener popularidades"}
        
        popularities = [a["popularity"] for a in artist_popularities]
        avg_popularity = sum(popularities) / len(popularities)
        max_popularity = max(popularities)
        min_popularity = min(popularities)
        popularity_range = max_popularity - min_popularity
        
        track_popularity = track_info.get("popularity", 0)
        
        return {
            "artist_popularities": artist_popularities,
            "average_artist_popularity": round(avg_popularity, 2),
            "popularity_range": round(popularity_range, 2),
            "track_popularity": track_popularity,
            "popularity_balance": "Balanced" if popularity_range < 30 else "Unbalanced",
            "track_vs_artists": "Above Average" if track_popularity > avg_popularity else "Below Average"
        }
    
    def analyze_collaboration_network(self, artist_names: List[str], depth: int = 2) -> Dict[str, Any]:
        """Analiza red de colaboraciones"""
        try:
            if len(artist_names) < 1:
                return {"error": "Se requiere al menos un artista"}
            
            # Buscar colaboraciones del artista principal
            network = {
                "central_artists": artist_names,
                "collaborations": [],
                "network_depth": depth
            }
            
            for artist_name in artist_names[:3]:  # Limitar a 3 artistas centrales
                # Buscar tracks del artista
                tracks = self.spotify.search_tracks(f"artist:{artist_name}", limit=50)
                
                collaborations = defaultdict(int)
                for track in tracks:
                    track_artists = [a.get("name") for a in track.get("artists", [])]
                    if len(track_artists) > 1:
                        for other_artist in track_artists:
                            if other_artist != artist_name:
                                collaborations[other_artist] += 1
                
                # Top colaboradores
                top_collaborators = sorted(collaborations.items(), key=lambda x: x[1], reverse=True)[:10]
                
                network["collaborations"].append({
                    "artist": artist_name,
                    "top_collaborators": [{"name": name, "count": count} for name, count in top_collaborators],
                    "total_collaborations": sum(collaborations.values())
                })
            
            return network
        except Exception as e:
            self.logger.error(f"Error analyzing collaboration network: {e}")
            return {"error": str(e)}

