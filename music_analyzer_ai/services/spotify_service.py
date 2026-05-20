"""
Servicio de integración con Spotify API
"""

import base64
import requests
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import logging

from ..config.settings import settings
from ..utils.exceptions import SpotifyAPIException, TrackNotFoundException, InvalidTrackIDException
from ..utils.cache import cache_manager
from ..utils.validators import validate_spotify_track_id, validate_search_query, sanitize_search_query

logger = logging.getLogger(__name__)


class SpotifyService:
    """Servicio para interactuar con la API de Spotify"""
    
    def __init__(self):
        self.client_id = settings.SPOTIFY_CLIENT_ID
        self.client_secret = settings.SPOTIFY_CLIENT_SECRET
        self.redirect_uri = settings.SPOTIFY_REDIRECT_URI
        self.base_url = "https://api.spotify.com/v1"
        self.auth_url = "https://accounts.spotify.com/api/token"
        self.access_token: Optional[str] = None
        self.token_expires_at: Optional[datetime] = None
        
    def _get_access_token(self) -> str:
        """Obtiene un token de acceso usando Client Credentials Flow"""
        if self.access_token and self.token_expires_at and datetime.now() < self.token_expires_at:
            return self.access_token
            
        if not self.client_id or not self.client_secret:
            raise ValueError("Spotify Client ID y Client Secret deben estar configurados")
        
        # Codificar credenciales
        credentials = f"{self.client_id}:{self.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            "Authorization": f"Basic {encoded_credentials}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        data = {
            "grant_type": "client_credentials"
        }
        
        try:
            response = requests.post(self.auth_url, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)  # 1 minuto de margen
            
            logger.info("Token de Spotify obtenido exitosamente")
            return self.access_token
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener token de Spotify: {e}")
            raise SpotifyAPIException(
                f"No se pudo obtener el token de acceso de Spotify: {str(e)}",
                status_code=getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            )
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, 
                     use_cache: bool = True, cache_key: Optional[str] = None) -> Dict[str, Any]:
        """Realiza una petición autenticada a la API de Spotify"""
        # Verificar cache
        if use_cache and cache_key:
            cached_data = cache_manager.get("spotify", cache_key)
            if cached_data:
                return cached_data
        
        token = self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}"
        }
        
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Guardar en cache
            if use_cache and cache_key:
                cache_manager.set("spotify", cache_key, data)
            
            return data
        except requests.exceptions.Timeout:
            logger.error(f"Timeout en petición a Spotify: {endpoint}")
            raise SpotifyAPIException(f"Timeout al consultar Spotify API: {endpoint}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            if status_code == 404:
                raise TrackNotFoundException(f"Recurso no encontrado: {endpoint}")
            logger.error(f"Error HTTP en petición a Spotify: {e}")
            raise SpotifyAPIException(
                f"Error al consultar Spotify API: {str(e)}",
                status_code=status_code
            )
        except requests.exceptions.RequestException as e:
            logger.error(f"Error en petición a Spotify: {e}")
            raise SpotifyAPIException(f"Error al consultar Spotify API: {str(e)}")
    
    def search_track(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Busca una canción en Spotify"""
        # Validar y sanitizar query
        validate_search_query(query)
        query = sanitize_search_query(query)
        
        # Limitar el límite
        limit = min(max(1, limit), 50)
        
        cache_key = f"search:{query}:{limit}"
        params = {
            "q": query,
            "type": "track",
            "limit": limit
        }
        
        response = self._make_request("/search", params, cache_key=cache_key)
        tracks = response.get("tracks", {}).get("items", [])
        
        return tracks
    
    def get_track(self, track_id: str) -> Dict[str, Any]:
        """Obtiene información detallada de una canción"""
        validate_spotify_track_id(track_id)
        return self._make_request(f"/tracks/{track_id}", cache_key=f"track:{track_id}")
    
    def get_track_audio_features(self, track_id: str) -> Dict[str, Any]:
        """Obtiene las características de audio de una canción"""
        validate_spotify_track_id(track_id)
        return self._make_request(f"/audio-features/{track_id}", cache_key=f"features:{track_id}")
    
    def get_track_audio_analysis(self, track_id: str) -> Dict[str, Any]:
        """Obtiene el análisis de audio detallado de una canción"""
        validate_spotify_track_id(track_id)
        # El análisis de audio no se cachea porque puede cambiar
        return self._make_request(f"/audio-analysis/{track_id}", use_cache=False)
    
    def get_artist(self, artist_id: str) -> Dict[str, Any]:
        """Obtiene información de un artista"""
        return self._make_request(f"/artists/{artist_id}", cache_key=f"artist:{artist_id}")
    
    def get_album(self, album_id: str) -> Dict[str, Any]:
        """Obtiene información de un álbum"""
        return self._make_request(f"/albums/{album_id}", cache_key=f"album:{album_id}")
    
    def get_track_full_analysis(self, track_id: str) -> Dict[str, Any]:
        """Obtiene análisis completo de una canción (track info + audio features + audio analysis)"""
        validate_spotify_track_id(track_id)
        
        # Verificar cache para análisis completo
        cache_key = f"full_analysis:{track_id}"
        cached = cache_manager.get("spotify", cache_key)
        if cached:
            return cached
        
        track_info = self.get_track(track_id)
        audio_features = self.get_track_audio_features(track_id)
        audio_analysis = self.get_track_audio_analysis(track_id)
        
        result = {
            "track_info": track_info,
            "audio_features": audio_features,
            "audio_analysis": audio_analysis
        }
        
        # Cachear resultado completo (con TTL más corto porque incluye audio_analysis)
        cache_manager.set("spotify", cache_key, result, ttl=1800)  # 30 minutos
        
        return result
    
    def search_and_analyze(self, query: str) -> Dict[str, Any]:
        """Busca una canción y obtiene su análisis completo"""
        validate_search_query(query)
        tracks = self.search_track(query, limit=1)
        
        if not tracks:
            raise TrackNotFoundException(f"No se encontró la canción: {query}")
        
        track_id = tracks[0]["id"]
        return self.get_track_full_analysis(track_id)
    
    def get_recommendations(self, track_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Obtiene recomendaciones de canciones similares basadas en una canción"""
        validate_spotify_track_id(track_id)
        limit = min(max(1, limit), 100)
        
        # Obtener audio features para usar en recomendaciones
        audio_features = self.get_track_audio_features(track_id)
        
        params = {
            "seed_tracks": track_id,
            "limit": limit,
            "target_energy": audio_features.get("energy", 0.5),
            "target_danceability": audio_features.get("danceability", 0.5),
            "target_valence": audio_features.get("valence", 0.5)
        }
        
        cache_key = f"recommendations:{track_id}:{limit}"
        response = self._make_request("/recommendations", params, cache_key=cache_key)
        
        return response.get("tracks", [])

