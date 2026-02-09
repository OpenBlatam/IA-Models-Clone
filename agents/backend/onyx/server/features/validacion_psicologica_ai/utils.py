"""
Utilidades para Validación Psicológica AI
==========================================
Funciones helper y utilidades generales
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from uuid import UUID
import hashlib
import json
import structlog
from functools import lru_cache
import re

from .models import SocialMediaPlatform, ValidationStatus

logger = structlog.get_logger()


class TokenEncryption:
    """Utilidad para encriptación de tokens"""
    
    @staticmethod
    def encrypt_token(token: str, key: str) -> str:
        """
        Encriptar token (simplificado - usar Fernet en producción)
        
        Args:
            token: Token a encriptar
            key: Clave de encriptación
            
        Returns:
            Token encriptado
        """
        # En producción, usar cryptography.fernet
        # Por ahora, usar hash simple para demostración
        combined = f"{token}{key}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    @staticmethod
    def decrypt_token(encrypted_token: str, key: str, original_token: str) -> bool:
        """
        Verificar token encriptado
        
        Args:
            encrypted_token: Token encriptado
            key: Clave de encriptación
            original_token: Token original para verificar
            
        Returns:
            True si el token coincide
        """
        expected = TokenEncryption.encrypt_token(original_token, key)
        return encrypted_token == expected


class TextProcessor:
    """Procesador de texto para análisis"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Limpiar texto para análisis
        
        Args:
            text: Texto a limpiar
            
        Returns:
            Texto limpio
        """
        if not text:
            return ""
        
        # Remover URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remover menciones y hashtags (opcional)
        # text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remover caracteres especiales excesivos
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    @staticmethod
    def extract_keywords(text: str, top_n: int = 10) -> List[str]:
        """
        Extraer palabras clave de un texto
        
        Args:
            text: Texto a analizar
            top_n: Número de palabras clave a extraer
            
        Returns:
            Lista de palabras clave
        """
        if not text:
            return []
        
        # Palabras comunes a ignorar
        stop_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we',
            'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all',
            'would', 'there', 'their', 'el', 'la', 'de', 'que', 'y', 'a',
            'en', 'un', 'ser', 'se', 'no', 'haber', 'por', 'con', 'su',
            'para', 'como', 'estar', 'tener', 'le', 'lo', 'todo', 'pero',
            'más', 'hacer', 'o', 'poder', 'decir', 'este', 'ir', 'otro',
            'ese', 'la', 'si', 'me', 'ya', 'ver', 'porque', 'dar', 'cuando'
        }
        
        # Limpiar y tokenizar
        text = TextProcessor.clean_text(text.lower())
        words = re.findall(r'\b\w+\b', text)
        
        # Filtrar stop words y contar frecuencia
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Ordenar por frecuencia
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        return [word for word, _ in sorted_words[:top_n]]
    
    @staticmethod
    def calculate_readability_score(text: str) -> float:
        """
        Calcular score de legibilidad (simplificado)
        
        Args:
            text: Texto a analizar
            
        Returns:
            Score de legibilidad (0-1)
        """
        if not text:
            return 0.0
        
        sentences = re.split(r'[.!?]+', text)
        words = re.findall(r'\b\w+\b', text)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Score simplificado (más alto = más legible)
        score = 1.0 / (1.0 + avg_sentence_length / 20.0 + avg_word_length / 10.0)
        
        return max(0.0, min(1.0, score))


class ValidationComparator:
    """Comparador de validaciones temporales"""
    
    @staticmethod
    def compare_validations(
        validation1: Dict[str, Any],
        validation2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comparar dos validaciones
        
        Args:
            validation1: Primera validación
            validation2: Segunda validación
            
        Returns:
            Comparación detallada
        """
        profile1 = validation1.get("profile", {})
        profile2 = validation2.get("profile", {})
        
        # Comparar rasgos de personalidad
        traits1 = profile1.get("personality_traits", {})
        traits2 = profile2.get("personality_traits", {})
        
        trait_changes = {}
        for trait in traits1:
            if trait in traits2:
                change = traits2[trait] - traits1[trait]
                trait_changes[trait] = {
                    "previous": traits1[trait],
                    "current": traits2[trait],
                    "change": change,
                    "change_percent": (change / traits1[trait] * 100) if traits1[trait] > 0 else 0
                }
        
        # Comparar estado emocional
        emotional1 = profile1.get("emotional_state", {})
        emotional2 = profile2.get("emotional_state", {})
        
        emotional_changes = {}
        for key in emotional1:
            if key in emotional2:
                if isinstance(emotional1[key], (int, float)):
                    change = emotional2[key] - emotional1[key]
                    emotional_changes[key] = {
                        "previous": emotional1[key],
                        "current": emotional2[key],
                        "change": change
                    }
        
        # Comparar confianza
        confidence1 = profile1.get("confidence_score", 0.0)
        confidence2 = profile2.get("confidence_score", 0.0)
        
        return {
            "trait_changes": trait_changes,
            "emotional_changes": emotional_changes,
            "confidence_change": {
                "previous": confidence1,
                "current": confidence2,
                "change": confidence2 - confidence1
            },
            "time_difference_days": (
                datetime.fromisoformat(validation2.get("created_at", datetime.utcnow().isoformat())) -
                datetime.fromisoformat(validation1.get("created_at", datetime.utcnow().isoformat()))
            ).days
        }
    
    @staticmethod
    def detect_trends(validations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detectar tendencias en múltiples validaciones
        
        Args:
            validations: Lista de validaciones ordenadas por fecha
            
        Returns:
            Tendencias detectadas
        """
        if len(validations) < 2:
            return {"trends": [], "insufficient_data": True}
        
        trends = []
        
        # Analizar tendencia de confianza
        confidence_scores = [
            v.get("profile", {}).get("confidence_score", 0.0)
            for v in validations
        ]
        
        if len(confidence_scores) >= 2:
            if confidence_scores[-1] > confidence_scores[0]:
                trends.append({
                    "metric": "confidence_score",
                    "trend": "increasing",
                    "change": confidence_scores[-1] - confidence_scores[0]
                })
            elif confidence_scores[-1] < confidence_scores[0]:
                trends.append({
                    "metric": "confidence_score",
                    "trend": "decreasing",
                    "change": confidence_scores[-1] - confidence_scores[0]
                })
        
        # Analizar tendencia de sentimientos
        sentiments = [
            v.get("profile", {}).get("emotional_state", {}).get("overall_sentiment", "neutral")
            for v in validations
        ]
        
        positive_count = sentiments.count("positive")
        negative_count = sentiments.count("negative")
        
        if positive_count > negative_count:
            trends.append({
                "metric": "sentiment",
                "trend": "predominantly_positive",
                "positive_ratio": positive_count / len(sentiments)
            })
        elif negative_count > positive_count:
            trends.append({
                "metric": "sentiment",
                "trend": "predominantly_negative",
                "negative_ratio": negative_count / len(sentiments)
            })
        
        return {
            "trends": trends,
            "total_validations": len(validations),
            "time_span_days": (
                datetime.fromisoformat(validations[-1].get("created_at", datetime.utcnow().isoformat())) -
                datetime.fromisoformat(validations[0].get("created_at", datetime.utcnow().isoformat()))
            ).days if len(validations) > 1 else 0
        }


class CacheManager:
    """Gestor de caché simple"""
    
    def __init__(self, ttl: int = 3600, max_size: int = 1000):
        """
        Inicializar gestor de caché
        
        Args:
            ttl: Tiempo de vida en segundos
            max_size: Tamaño máximo del caché
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        self.max_size = max_size
    
    def _generate_key(self, prefix: str, *args) -> str:
        """Generar clave de caché"""
        key_parts = [prefix] + [str(arg) for arg in args]
        return ":".join(key_parts)
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtener valor del caché
        
        Args:
            key: Clave del caché
            
        Returns:
            Valor o None si no existe o expiró
        """
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        expires_at = entry.get("expires_at")
        
        if expires_at and datetime.utcnow() > expires_at:
            del self._cache[key]
            return None
        
        return entry.get("value")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Guardar valor en caché
        
        Args:
            key: Clave del caché
            value: Valor a guardar
            ttl: Tiempo de vida en segundos (opcional)
        """
        # Limpiar si el caché está lleno
        if len(self._cache) >= self.max_size:
            # Eliminar entrada más antigua
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k].get("expires_at", datetime.max)
            )
            del self._cache[oldest_key]
        
        expires_at = datetime.utcnow() + timedelta(seconds=ttl or self.ttl)
        
        self._cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": datetime.utcnow()
        }
    
    def delete(self, key: str) -> None:
        """Eliminar entrada del caché"""
        if key in self._cache:
            del self._cache[key]
    
    def clear(self) -> None:
        """Limpiar todo el caché"""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del caché"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }


class MetricsCollector:
    """Colector de métricas"""
    
    def __init__(self):
        """Inicializar colector"""
        self._metrics: Dict[str, Any] = {
            "validations_created": 0,
            "validations_completed": 0,
            "validations_failed": 0,
            "profiles_generated": 0,
            "reports_generated": 0,
            "api_calls": 0,
            "api_errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
        self._start_time = datetime.utcnow()
    
    def increment(self, metric: str, value: int = 1) -> None:
        """Incrementar métrica"""
        if metric in self._metrics:
            self._metrics[metric] += value
        else:
            self._metrics[metric] = value
    
    def set(self, metric: str, value: Any) -> None:
        """Establecer valor de métrica"""
        self._metrics[metric] = value
    
    def get(self, metric: str) -> Any:
        """Obtener valor de métrica"""
        return self._metrics.get(metric, 0)
    
    def get_all(self) -> Dict[str, Any]:
        """Obtener todas las métricas"""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()
        
        return {
            **self._metrics,
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "cache_hit_rate": (
                self._metrics["cache_hits"] / 
                (self._metrics["cache_hits"] + self._metrics["cache_misses"])
                if (self._metrics["cache_hits"] + self._metrics["cache_misses"]) > 0
                else 0.0
            ),
            "success_rate": (
                self._metrics["validations_completed"] /
                (self._metrics["validations_created"])
                if self._metrics["validations_created"] > 0
                else 0.0
            )
        }
    
    def reset(self) -> None:
        """Resetear métricas"""
        self._metrics = {k: 0 for k in self._metrics if isinstance(self._metrics[k], (int, float))}
        self._start_time = datetime.utcnow()


# Instancia global de métricas
metrics = MetricsCollector()




