"""
Servicio de análisis de audio en tiempo real
"""

import logging
from typing import Dict, List, Any, Optional
from collections import deque
import time

logger = logging.getLogger(__name__)


class RealtimeAudioAnalyzer:
    """Análisis de audio en tiempo real (simulado)"""
    
    def __init__(self):
        self.logger = logger
        self.analysis_buffer = deque(maxlen=100)
    
    def analyze_realtime_stream(self, audio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza stream de audio en tiempo real"""
        try:
            # Simulación de análisis en tiempo real
            timestamp = time.time()
            
            # Extraer características del stream
            energy = audio_data.get("energy", 0.5)
            tempo = audio_data.get("tempo", 120)
            loudness = audio_data.get("loudness", -10)
            
            # Análisis en tiempo real
            analysis = {
                "timestamp": timestamp,
                "current_energy": round(energy, 3),
                "current_tempo": round(tempo, 2),
                "current_loudness": round(loudness, 2),
                "trends": self._calculate_trends(energy, tempo, loudness),
                "alerts": self._check_alerts(energy, tempo, loudness)
            }
            
            # Agregar al buffer
            self.analysis_buffer.append(analysis)
            
            return {
                "success": True,
                "realtime_analysis": analysis,
                "buffer_size": len(self.analysis_buffer)
            }
        except Exception as e:
            self.logger.error(f"Error analyzing realtime stream: {e}")
            return {"error": str(e)}
    
    def _calculate_trends(self, energy: float, tempo: float, loudness: float) -> Dict[str, Any]:
        """Calcula tendencias en tiempo real"""
        if len(self.analysis_buffer) < 2:
            return {
                "energy_trend": "Stable",
                "tempo_trend": "Stable",
                "loudness_trend": "Stable"
            }
        
        # Obtener valores anteriores
        prev_analysis = self.analysis_buffer[-2] if len(self.analysis_buffer) >= 2 else None
        
        if not prev_analysis:
            return {
                "energy_trend": "Stable",
                "tempo_trend": "Stable",
                "loudness_trend": "Stable"
            }
        
        prev_energy = prev_analysis.get("current_energy", energy)
        prev_tempo = prev_analysis.get("current_tempo", tempo)
        prev_loudness = prev_analysis.get("current_loudness", loudness)
        
        # Calcular tendencias
        energy_trend = "Increasing" if energy > prev_energy + 0.05 else "Decreasing" if energy < prev_energy - 0.05 else "Stable"
        tempo_trend = "Increasing" if tempo > prev_tempo + 5 else "Decreasing" if tempo < prev_tempo - 5 else "Stable"
        loudness_trend = "Increasing" if loudness > prev_loudness + 2 else "Decreasing" if loudness < prev_loudness - 2 else "Stable"
        
        return {
            "energy_trend": energy_trend,
            "tempo_trend": tempo_trend,
            "loudness_trend": loudness_trend,
            "energy_change": round(energy - prev_energy, 3),
            "tempo_change": round(tempo - prev_tempo, 2),
            "loudness_change": round(loudness - prev_loudness, 2)
        }
    
    def _check_alerts(self, energy: float, tempo: float, loudness: float) -> List[Dict[str, Any]]:
        """Verifica alertas en tiempo real"""
        alerts = []
        
        if energy > 0.9:
            alerts.append({
                "type": "High Energy",
                "level": "Warning",
                "message": "Energy level muy alto detectado"
            })
        
        if loudness > 0:
            alerts.append({
                "type": "High Loudness",
                "level": "Warning",
                "message": "Loudness muy alto - riesgo de clipping"
            })
        
        if tempo > 200:
            alerts.append({
                "type": "Very Fast Tempo",
                "level": "Info",
                "message": "Tempo muy rápido detectado"
            })
        
        return alerts
    
    def get_analysis_history(self, limit: int = 50) -> Dict[str, Any]:
        """Obtiene historial de análisis"""
        history = list(self.analysis_buffer)[-limit:]
        
        return {
            "history": history,
            "count": len(history),
            "time_range": {
                "oldest": history[0].get("timestamp") if history else None,
                "newest": history[-1].get("timestamp") if history else None
            }
        }
    
    def clear_buffer(self):
        """Limpia el buffer"""
        self.analysis_buffer.clear()
        return {"success": True, "message": "Buffer cleared"}

