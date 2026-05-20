"""
Excepciones personalizadas para el sistema de análisis musical
"""


class MusicAnalyzerException(Exception):
    """Excepción base para el sistema de análisis musical"""
    pass


class SpotifyAPIException(MusicAnalyzerException):
    """Excepción relacionada con la API de Spotify"""
    def __init__(self, message: str, status_code: int = None, response_data: dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class TrackNotFoundException(MusicAnalyzerException):
    """Excepción cuando no se encuentra una canción"""
    pass


class InvalidTrackIDException(MusicAnalyzerException):
    """Excepción cuando el ID de track es inválido"""
    pass


class AnalysisException(MusicAnalyzerException):
    """Excepción durante el análisis de música"""
    pass


class ConfigurationException(MusicAnalyzerException):
    """Excepción relacionada con la configuración"""
    pass


class CacheException(MusicAnalyzerException):
    """Excepción relacionada con el cache"""
    pass

