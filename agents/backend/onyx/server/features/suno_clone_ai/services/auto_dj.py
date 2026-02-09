"""
Sistema de DJ Automático

Proporciona:
- Mix automático de canciones
- Transiciones suaves
- Detección de BPM y key matching
- Playlists inteligentes
- Efectos automáticos
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import librosa
    import numpy as np
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("Librosa not available, auto DJ limited")


@dataclass
class TrackInfo:
    """Información de una pista"""
    track_id: str
    path: str
    bpm: float = 0.0
    key: str = "unknown"
    energy: float = 0.0
    duration: float = 0.0


@dataclass
class MixTransition:
    """Transición entre pistas"""
    from_track: str
    to_track: str
    transition_type: str  # "fade", "crossfade", "beat_match"
    duration: float = 2.0
    start_time: float = 0.0


@dataclass
class DJSet:
    """Set de DJ"""
    set_id: str
    tracks: List[TrackInfo]
    transitions: List[MixTransition]
    total_duration: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class AutoDJService:
    """Servicio de DJ automático"""
    
    def __init__(self):
        logger.info("AutoDJService initialized")
    
    def analyze_track(self, track_path: str) -> TrackInfo:
        """
        Analiza una pista para DJ
        
        Args:
            track_path: Ruta del archivo
        
        Returns:
            TrackInfo
        """
        if not LIBROSA_AVAILABLE:
            return TrackInfo(track_id="", path=track_path)
        
        try:
            import os
            from services.audio_analysis import get_audio_analyzer
            
            analyzer = get_audio_analyzer()
            analysis = analyzer.analyze(track_path)
            
            # Obtener duración
            y, sr = librosa.load(track_path, sr=None)
            duration = len(y) / sr
            
            track_id = os.path.basename(track_path)
            
            return TrackInfo(
                track_id=track_id,
                path=track_path,
                bpm=analysis.bpm,
                key=analysis.key,
                energy=analysis.energy,
                duration=duration
            )
        
        except Exception as e:
            logger.error(f"Error analyzing track: {e}")
            return TrackInfo(track_id="", path=track_path)
    
    def create_mix(
        self,
        track_paths: List[str],
        transition_type: str = "crossfade",
        transition_duration: float = 2.0
    ) -> DJSet:
        """
        Crea un mix automático de pistas
        
        Args:
            track_paths: Lista de rutas de pistas
            transition_type: Tipo de transición
            transition_duration: Duración de transición (segundos)
        
        Returns:
            DJSet
        """
        import uuid
        
        # Analizar todas las pistas
        tracks = [self.analyze_track(path) for path in track_paths]
        
        # Ordenar por BPM para transiciones suaves
        tracks.sort(key=lambda t: t.bpm)
        
        # Crear transiciones
        transitions = []
        total_duration = 0.0
        
        for i in range(len(tracks) - 1):
            from_track = tracks[i]
            to_track = tracks[i + 1]
            
            transition = MixTransition(
                from_track=from_track.track_id,
                to_track=to_track.track_id,
                transition_type=transition_type,
                duration=transition_duration,
                start_time=total_duration + from_track.duration - transition_duration
            )
            
            transitions.append(transition)
            total_duration += from_track.duration
        
        # Agregar duración de la última pista
        if tracks:
            total_duration += tracks[-1].duration
        
        set_id = str(uuid.uuid4())
        
        dj_set = DJSet(
            set_id=set_id,
            tracks=tracks,
            transitions=transitions,
            total_duration=total_duration
        )
        
        logger.info(f"DJ set created: {set_id} with {len(tracks)} tracks")
        return dj_set
    
    def generate_playlist(
        self,
        seed_track_path: str,
        num_tracks: int = 10,
        bpm_tolerance: float = 5.0,
        key_compatible: bool = True
    ) -> List[TrackInfo]:
        """
        Genera una playlist basada en una pista semilla
        
        Args:
            seed_track_path: Pista semilla
            num_tracks: Número de pistas
            bpm_tolerance: Tolerancia de BPM
            key_compatible: Solo keys compatibles
        
        Returns:
            Lista de TrackInfo
        """
        # Analizar pista semilla
        seed_track = self.analyze_track(seed_track_path)
        
        # En producción, esto buscaría en una base de datos
        # Por ahora, retornamos solo la pista semilla
        return [seed_track]
    
    def create_beat_matched_mix(
        self,
        track_paths: List[str],
        target_bpm: Optional[float] = None
    ) -> DJSet:
        """
        Crea un mix con beat matching
        
        Args:
            track_paths: Lista de rutas
            target_bpm: BPM objetivo (opcional)
        
        Returns:
            DJSet
        """
        # Analizar pistas
        tracks = [self.analyze_track(path) for path in track_paths]
        
        # Si hay BPM objetivo, ajustar todas las pistas
        if target_bpm:
            for track in tracks:
                if track.bpm != target_bpm:
                    # En producción, esto ajustaría el tempo
                    track.bpm = target_bpm
        
        # Crear mix con beat matching
        return self.create_mix(
            [t.path for t in tracks],
            transition_type="beat_match",
            transition_duration=4.0
        )
    
    def get_mix_recommendations(
        self,
        current_track: TrackInfo,
        available_tracks: List[TrackInfo]
    ) -> List[TrackInfo]:
        """
        Obtiene recomendaciones de pistas para mix
        
        Args:
            current_track: Pista actual
            available_tracks: Pistas disponibles
        
        Returns:
            Lista de pistas recomendadas
        """
        recommendations = []
        
        for track in available_tracks:
            if track.track_id == current_track.track_id:
                continue
            
            # Calcular score de compatibilidad
            bpm_diff = abs(track.bpm - current_track.bpm)
            key_match = track.key == current_track.key or self._are_keys_compatible(
                current_track.key,
                track.key
            )
            
            score = 0.0
            
            # BPM cercano es mejor
            if bpm_diff <= 5:
                score += 0.5
            elif bpm_diff <= 10:
                score += 0.3
            
            # Key matching
            if key_match:
                score += 0.3
            
            # Energy similar
            energy_diff = abs(track.energy - current_track.energy)
            if energy_diff < 0.2:
                score += 0.2
            
            if score > 0.3:
                recommendations.append((track, score))
        
        # Ordenar por score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in recommendations[:5]]
    
    def _are_keys_compatible(self, key1: str, key2: str) -> bool:
        """Verifica si dos keys son compatibles"""
        # Simplificado: keys relativas (mayor/menor)
        compatible_pairs = [
            ("C major", "A minor"),
            ("A minor", "C major"),
            ("G major", "E minor"),
            ("E minor", "G major"),
            ("D major", "B minor"),
            ("B minor", "D major")
        ]
        
        return (key1, key2) in compatible_pairs or key1 == key2


# Instancia global
_auto_dj_service: Optional[AutoDJService] = None


def get_auto_dj_service() -> AutoDJService:
    """Obtiene la instancia global del servicio de DJ automático"""
    global _auto_dj_service
    if _auto_dj_service is None:
        _auto_dj_service = AutoDJService()
    return _auto_dj_service

