#!/usr/bin/env python3
"""
Advanced Features - Funcionalidades Avanzadas
Implementación de funcionalidades avanzadas para el sistema de comparación de historial de IA
"""

import asyncio
import json
import base64
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import redis.asyncio as redis
from cryptography.fernet import Fernet
import pyotp
import qrcode
from io import BytesIO
import aiohttp
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Resultado de análisis"""
    content_id: str
    analysis_type: str
    results: Dict[str, Any]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = None

class SemanticAnalyzer:
    """Analizador semántico avanzado"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Inicializar analizador semántico"""
        self.model = SentenceTransformer(model_name)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    
    async def analyze_semantic_similarity(self, content1: str, content2: str) -> float:
        """Análisis de similitud semántica"""
        try:
            # Generar embeddings
            embeddings1 = self.model.encode([content1])
            embeddings2 = self.model.encode([content2])
            
            # Calcular similitud coseno
            similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
            
            logger.info(f"Semantic similarity calculated: {similarity:.4f}")
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0
    
    async def extract_key_concepts(self, content: str) -> List[str]:
        """Extraer conceptos clave"""
        try:
            # Usar TF-IDF para extraer conceptos
            tfidf_matrix = self.vectorizer.fit_transform([content])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Obtener scores más altos
            scores = tfidf_matrix.toarray()[0]
            top_indices = np.argsort(scores)[-10:][::-1]  # Top 10 conceptos
            
            concepts = [feature_names[i] for i in top_indices if scores[i] > 0]
            
            logger.info(f"Extracted {len(concepts)} key concepts")
            return concepts
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {str(e)}")
            return []
    
    async def analyze_topic_evolution(self, contents: List[str]) -> Dict[str, Any]:
        """Analizar evolución de temas"""
        try:
            if len(contents) < 2:
                return {"error": "Need at least 2 contents for topic evolution"}
            
            # Generar embeddings para todos los contenidos
            embeddings = self.model.encode(contents)
            
            # Calcular similitud entre contenidos consecutivos
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
                similarities.append(sim)
            
            # Análisis de evolución
            evolution_analysis = {
                "average_similarity": float(np.mean(similarities)),
                "similarity_trend": similarities,
                "topic_stability": float(np.std(similarities)),
                "evolution_direction": "stable" if np.std(similarities) < 0.1 else "changing"
            }
            
            logger.info(f"Topic evolution analyzed: {evolution_analysis['evolution_direction']}")
            return evolution_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing topic evolution: {str(e)}")
            return {"error": str(e)}

class AdvancedSentimentAnalyzer:
    """Analizador de sentimiento avanzado"""
    
    def __init__(self):
        """Inicializar analizador de sentimiento"""
        self.emotion_keywords = {
            "joy": ["happy", "joyful", "excited", "pleased", "delighted"],
            "sadness": ["sad", "depressed", "melancholy", "gloomy", "sorrowful"],
            "anger": ["angry", "furious", "irritated", "annoyed", "rage"],
            "fear": ["afraid", "scared", "terrified", "anxious", "worried"],
            "surprise": ["surprised", "amazed", "astonished", "shocked", "stunned"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "nauseated"]
        }
    
    async def analyze_emotions(self, content: str) -> Dict[str, float]:
        """Análisis de emociones específicas"""
        try:
            content_lower = content.lower()
            emotions = {}
            
            for emotion, keywords in self.emotion_keywords.items():
                # Contar ocurrencias de palabras clave
                count = sum(content_lower.count(keyword) for keyword in keywords)
                # Normalizar por longitud del contenido
                emotions[emotion] = count / len(content.split()) if content.split() else 0
            
            # Normalizar scores
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            
            logger.info(f"Emotions analyzed: {emotions}")
            return emotions
            
        except Exception as e:
            logger.error(f"Error analyzing emotions: {str(e)}")
            return {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
    
    async def analyze_sentiment_intensity(self, content: str) -> float:
        """Análisis de intensidad del sentimiento"""
        try:
            # Palabras intensificadoras
            intensifiers = ["very", "extremely", "incredibly", "absolutely", "completely"]
            content_lower = content.lower()
            
            # Contar intensificadores
            intensity_count = sum(content_lower.count(intensifier) for intensifier in intensifiers)
            
            # Normalizar por longitud
            intensity = intensity_count / len(content.split()) if content.split() else 0
            
            # Escalar a 0-1
            intensity = min(intensity * 10, 1.0)
            
            logger.info(f"Sentiment intensity: {intensity:.4f}")
            return intensity
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment intensity: {str(e)}")
            return 0.0

class ContentQualityAnalyzer:
    """Analizador de calidad de contenido"""
    
    def __init__(self):
        """Inicializar analizador de calidad"""
        self.quality_metrics = {
            "clarity": self._analyze_clarity,
            "coherence": self._analyze_coherence,
            "completeness": self._analyze_completeness,
            "accuracy": self._analyze_accuracy,
            "relevance": self._analyze_relevance,
            "originality": self._analyze_originality
        }
    
    async def analyze_quality_dimensions(self, content: str) -> Dict[str, float]:
        """Análisis de dimensiones de calidad"""
        try:
            quality_scores = {}
            
            for metric, analyzer in self.quality_metrics.items():
                score = await analyzer(content)
                quality_scores[metric] = score
            
            logger.info(f"Quality analysis completed: {quality_scores}")
            return quality_scores
            
        except Exception as e:
            logger.error(f"Error analyzing quality: {str(e)}")
            return {metric: 0.0 for metric in self.quality_metrics.keys()}
    
    async def _analyze_clarity(self, content: str) -> float:
        """Analizar claridad"""
        # Métricas de claridad: longitud promedio de oraciones, uso de palabras complejas
        sentences = content.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        # Score basado en longitud promedio (ideal: 15-20 palabras)
        if 15 <= avg_sentence_length <= 20:
            return 1.0
        elif 10 <= avg_sentence_length <= 25:
            return 0.8
        else:
            return 0.6
    
    async def _analyze_coherence(self, content: str) -> float:
        """Analizar coherencia"""
        # Métricas de coherencia: conectores, repetición de palabras clave
        connectors = ["however", "therefore", "moreover", "furthermore", "consequently"]
        content_lower = content.lower()
        
        connector_count = sum(content_lower.count(connector) for connector in connectors)
        coherence_score = min(connector_count / 10, 1.0)  # Normalizar
        
        return coherence_score
    
    async def _analyze_completeness(self, content: str) -> float:
        """Analizar completitud"""
        # Métricas de completitud: longitud, estructura
        word_count = len(content.split())
        
        if word_count >= 100:
            return 1.0
        elif word_count >= 50:
            return 0.8
        elif word_count >= 20:
            return 0.6
        else:
            return 0.4
    
    async def _analyze_accuracy(self, content: str) -> float:
        """Analizar precisión"""
        # Métricas de precisión: uso de datos, referencias
        accuracy_indicators = ["according to", "research shows", "studies indicate", "data suggests"]
        content_lower = content.lower()
        
        indicator_count = sum(content_lower.count(indicator) for indicator in accuracy_indicators)
        accuracy_score = min(indicator_count / 5, 1.0)  # Normalizar
        
        return accuracy_score
    
    async def _analyze_relevance(self, content: str) -> float:
        """Analizar relevancia"""
        # Métricas de relevancia: palabras clave específicas del tema
        # Por simplicidad, asumimos que el contenido es relevante si tiene suficiente longitud
        word_count = len(content.split())
        relevance_score = min(word_count / 100, 1.0)
        
        return relevance_score
    
    async def _analyze_originality(self, content: str) -> float:
        """Analizar originalidad"""
        # Métricas de originalidad: diversidad de vocabulario
        words = content.lower().split()
        unique_words = set(words)
        
        if len(words) > 0:
            originality_score = len(unique_words) / len(words)
        else:
            originality_score = 0.0
        
        return originality_score

class DistributedCache:
    """Caché distribuido con Redis"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """Inicializar caché distribuido"""
        self.redis_url = redis_url
        self.redis = None
    
    async def connect(self):
        """Conectar a Redis"""
        try:
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.error(f"Error connecting to Redis: {str(e)}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Obtener del caché"""
        try:
            if not self.redis:
                await self.connect()
            
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
            
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Establecer en caché"""
        try:
            if not self.redis:
                await self.connect()
            
            await self.redis.setex(key, ttl, json.dumps(value))
            logger.info(f"Value cached with key: {key}")
            
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")
    
    async def delete(self, key: str):
        """Eliminar del caché"""
        try:
            if not self.redis:
                await self.connect()
            
            await self.redis.delete(key)
            logger.info(f"Cache key deleted: {key}")
            
        except Exception as e:
            logger.error(f"Error deleting from cache: {str(e)}")

class TrendAnalyzer:
    """Analizador de tendencias temporales"""
    
    def __init__(self):
        """Inicializar analizador de tendencias"""
        self.semantic_analyzer = SemanticAnalyzer()
        self.sentiment_analyzer = AdvancedSentimentAnalyzer()
        self.quality_analyzer = ContentQualityAnalyzer()
    
    async def analyze_temporal_trends(self, content_ids: List[str], time_range: str = "30d") -> Dict[str, Any]:
        """Analizar tendencias temporales"""
        try:
            # Simular datos históricos (en implementación real, obtener de BD)
            historical_data = await self._get_historical_data(content_ids, time_range)
            
            if len(historical_data) < 2:
                return {"error": "Insufficient data for trend analysis"}
            
            # Analizar tendencias
            trends = {
                "sentiment_trend": await self._analyze_sentiment_trend(historical_data),
                "topic_evolution": await self._analyze_topic_evolution(historical_data),
                "quality_trend": await self._analyze_quality_trend(historical_data),
                "complexity_trend": await self._analyze_complexity_trend(historical_data),
                "readability_trend": await self._analyze_readability_trend(historical_data)
            }
            
            logger.info("Temporal trends analyzed successfully")
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing temporal trends: {str(e)}")
            return {"error": str(e)}
    
    async def _get_historical_data(self, content_ids: List[str], time_range: str) -> List[Dict]:
        """Obtener datos históricos"""
        # Simular datos históricos
        return [
            {
                "content_id": content_id,
                "content": f"Sample content {i}",
                "timestamp": datetime.now() - timedelta(days=i),
                "sentiment": 0.5 + (i * 0.1),
                "quality": 0.7 + (i * 0.05)
            }
            for i, content_id in enumerate(content_ids)
        ]
    
    async def _analyze_sentiment_trend(self, data: List[Dict]) -> List[float]:
        """Analizar tendencia de sentimiento"""
        return [item["sentiment"] for item in data]
    
    async def _analyze_topic_evolution(self, data: List[Dict]) -> List[str]:
        """Analizar evolución de temas"""
        return [f"topic_{i}" for i in range(len(data))]
    
    async def _analyze_quality_trend(self, data: List[Dict]) -> List[float]:
        """Analizar tendencia de calidad"""
        return [item["quality"] for item in data]
    
    async def _analyze_complexity_trend(self, data: List[Dict]) -> List[float]:
        """Analizar tendencia de complejidad"""
        return [0.5 + (i * 0.1) for i in range(len(data))]
    
    async def _analyze_readability_trend(self, data: List[Dict]) -> List[float]:
        """Analizar tendencia de legibilidad"""
        return [0.6 + (i * 0.05) for i in range(len(data))]

class PlagiarismDetector:
    """Detector de plagio avanzado"""
    
    def __init__(self):
        """Inicializar detector de plagio"""
        self.semantic_analyzer = SemanticAnalyzer()
    
    async def detect_plagiarism(self, content: str, reference_corpus: List[str]) -> Dict[str, Any]:
        """Detectar plagio"""
        try:
            similarities = []
            
            for reference in reference_corpus:
                similarity = await self.semantic_analyzer.analyze_semantic_similarity(content, reference)
                similarities.append(similarity)
            
            max_similarity = max(similarities) if similarities else 0.0
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            plagiarism_report = {
                "similarity_score": float(max_similarity),
                "average_similarity": float(avg_similarity),
                "suspicious_sections": await self._find_suspicious_sections(content, reference_corpus),
                "source_matches": await self._find_source_matches(content, reference_corpus),
                "confidence": float(max_similarity * 0.9)  # Ajustar confianza
            }
            
            logger.info(f"Plagiarism detection completed: {plagiarism_report['similarity_score']:.4f}")
            return plagiarism_report
            
        except Exception as e:
            logger.error(f"Error detecting plagiarism: {str(e)}")
            return {"error": str(e)}
    
    async def _find_suspicious_sections(self, content: str, references: List[str]) -> List[Dict]:
        """Encontrar secciones sospechosas"""
        # Implementación simplificada
        return [
            {
                "start": 0,
                "end": 100,
                "similarity": 0.8,
                "source": "reference_1"
            }
        ]
    
    async def _find_source_matches(self, content: str, references: List[str]) -> List[Dict]:
        """Encontrar coincidencias con fuentes"""
        # Implementación simplificada
        return [
            {
                "source": "reference_1",
                "similarity": 0.8,
                "matched_sections": ["section_1", "section_2"]
            }
        ]

class DataEncryption:
    """Cifrado de datos sensibles"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """Inicializar cifrado"""
        if encryption_key:
            self.cipher = Fernet(encryption_key.encode())
        else:
            # Generar clave si no se proporciona
            key = Fernet.generate_key()
            self.cipher = Fernet(key)
            logger.info(f"Generated encryption key: {key.decode()}")
    
    def encrypt_content(self, content: str) -> str:
        """Cifrar contenido"""
        try:
            encrypted_data = self.cipher.encrypt(content.encode())
            return base64.b64encode(encrypted_data).decode()
        except Exception as e:
            logger.error(f"Error encrypting content: {str(e)}")
            return content
    
    def decrypt_content(self, encrypted_content: str) -> str:
        """Descifrar contenido"""
        try:
            encrypted_data = base64.b64decode(encrypted_content.encode())
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return decrypted_data.decode()
        except Exception as e:
            logger.error(f"Error decrypting content: {str(e)}")
            return encrypted_content

class MultiFactorAuth:
    """Autenticación multi-factor"""
    
    def __init__(self):
        """Inicializar MFA"""
        self.totp_service = pyotp.TOTP
    
    async def setup_mfa(self, user_id: str) -> Dict[str, str]:
        """Configurar MFA para usuario"""
        try:
            # Generar secreto
            secret = pyotp.random_base32()
            totp = self.totp_service(secret)
            
            # Generar QR code
            qr_code = totp.provisioning_uri(
                name=user_id,
                issuer_name="AI History Comparison System"
            )
            
            # Crear imagen QR
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data(qr_code)
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            
            # Convertir a base64
            buffer = BytesIO()
            img.save(buffer, format='PNG')
            qr_code_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            result = {
                "secret": secret,
                "qr_code": qr_code_base64,
                "manual_entry_key": secret
            }
            
            logger.info(f"MFA setup completed for user: {user_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error setting up MFA: {str(e)}")
            return {"error": str(e)}
    
    async def verify_mfa(self, user_id: str, token: str, secret: str) -> bool:
        """Verificar token MFA"""
        try:
            totp = self.totp_service(secret)
            is_valid = totp.verify(token, valid_window=1)
            
            logger.info(f"MFA verification for user {user_id}: {'success' if is_valid else 'failed'}")
            return is_valid
            
        except Exception as e:
            logger.error(f"Error verifying MFA: {str(e)}")
            return False

class CustomMetrics:
    """Métricas personalizadas"""
    
    def __init__(self):
        """Inicializar métricas"""
        self.metrics = {
            "analysis_requests": 0,
            "analysis_duration": [],
            "active_connections": 0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0
        }
    
    def record_analysis_request(self, analysis_type: str):
        """Registrar request de análisis"""
        self.metrics["analysis_requests"] += 1
        logger.info(f"Analysis request recorded: {analysis_type}")
    
    def record_analysis_duration(self, duration: float):
        """Registrar duración de análisis"""
        self.metrics["analysis_duration"].append(duration)
        logger.info(f"Analysis duration recorded: {duration:.4f}s")
    
    def update_active_connections(self, count: int):
        """Actualizar conexiones activas"""
        self.metrics["active_connections"] = count
        logger.info(f"Active connections updated: {count}")
    
    def update_cache_hit_rate(self, hit_rate: float):
        """Actualizar tasa de acierto de caché"""
        self.metrics["cache_hit_rate"] = hit_rate
        logger.info(f"Cache hit rate updated: {hit_rate:.4f}")
    
    def update_error_rate(self, error_rate: float):
        """Actualizar tasa de error"""
        self.metrics["error_rate"] = error_rate
        logger.info(f"Error rate updated: {error_rate:.4f}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Obtener métricas"""
        return self.metrics.copy()

class IntelligentAlerts:
    """Sistema de alertas inteligentes"""
    
    def __init__(self):
        """Inicializar alertas"""
        self.alert_rules = {
            "high_error_rate": {"threshold": 0.05, "severity": "critical"},
            "high_latency": {"threshold": 2.0, "severity": "warning"},
            "low_cache_hit_rate": {"threshold": 0.7, "severity": "warning"},
            "high_memory_usage": {"threshold": 0.8, "severity": "critical"}
        }
    
    async def check_anomalies(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar anomalías en métricas"""
        try:
            anomalies = []
            
            # Verificar tasa de error alta
            if metrics.get("error_rate", 0) > self.alert_rules["high_error_rate"]["threshold"]:
                anomalies.append({
                    "type": "high_error_rate",
                    "severity": self.alert_rules["high_error_rate"]["severity"],
                    "message": f"Error rate {metrics['error_rate']:.2%} exceeds threshold",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Verificar latencia alta
            avg_duration = np.mean(metrics.get("analysis_duration", [0]))
            if avg_duration > self.alert_rules["high_latency"]["threshold"]:
                anomalies.append({
                    "type": "high_latency",
                    "severity": self.alert_rules["high_latency"]["severity"],
                    "message": f"Average latency {avg_duration:.2f}s exceeds threshold",
                    "timestamp": datetime.now().isoformat()
                })
            
            # Verificar tasa de acierto de caché baja
            if metrics.get("cache_hit_rate", 1.0) < self.alert_rules["low_cache_hit_rate"]["threshold"]:
                anomalies.append({
                    "type": "low_cache_hit_rate",
                    "severity": self.alert_rules["low_cache_hit_rate"]["severity"],
                    "message": f"Cache hit rate {metrics['cache_hit_rate']:.2%} below threshold",
                    "timestamp": datetime.now().isoformat()
                })
            
            if anomalies:
                logger.warning(f"Anomalies detected: {len(anomalies)}")
                for anomaly in anomalies:
                    logger.warning(f"Alert: {anomaly['type']} - {anomaly['message']}")
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error checking anomalies: {str(e)}")
            return []

# Función principal para demostrar funcionalidades
async def main():
    """Función principal para demostrar funcionalidades avanzadas"""
    print("🚀 AI History Comparison System - Advanced Features Demo")
    print("=" * 60)
    
    # Inicializar componentes
    semantic_analyzer = SemanticAnalyzer()
    sentiment_analyzer = AdvancedSentimentAnalyzer()
    quality_analyzer = ContentQualityAnalyzer()
    cache = DistributedCache()
    trend_analyzer = TrendAnalyzer()
    plagiarism_detector = PlagiarismDetector()
    encryption = DataEncryption()
    mfa = MultiFactorAuth()
    metrics = CustomMetrics()
    alerts = IntelligentAlerts()
    
    # Conectar caché
    await cache.connect()
    
    # Contenido de ejemplo
    content1 = "This is a sample content for analysis. It contains various emotions and concepts."
    content2 = "This is another sample content that might be similar to the first one."
    
    print("\n📊 Análisis Semántico:")
    similarity = await semantic_analyzer.analyze_semantic_similarity(content1, content2)
    print(f"  Similitud semántica: {similarity:.4f}")
    
    concepts = await semantic_analyzer.extract_key_concepts(content1)
    print(f"  Conceptos clave: {concepts[:5]}")
    
    print("\n😊 Análisis de Sentimiento:")
    emotions = await sentiment_analyzer.analyze_emotions(content1)
    print(f"  Emociones: {emotions}")
    
    intensity = await sentiment_analyzer.analyze_sentiment_intensity(content1)
    print(f"  Intensidad: {intensity:.4f}")
    
    print("\n📈 Análisis de Calidad:")
    quality = await quality_analyzer.analyze_quality_dimensions(content1)
    print(f"  Calidad: {quality}")
    
    print("\n💾 Caché Distribuido:")
    await cache.set("test_key", {"data": "test_value"}, ttl=60)
    cached_value = await cache.get("test_key")
    print(f"  Valor en caché: {cached_value}")
    
    print("\n📊 Análisis de Tendencias:")
    trends = await trend_analyzer.analyze_temporal_trends(["content1", "content2"])
    print(f"  Tendencias: {trends}")
    
    print("\n🔍 Detección de Plagio:")
    plagiarism = await plagiarism_detector.detect_plagiarism(content1, [content2])
    print(f"  Plagio detectado: {plagiarism['similarity_score']:.4f}")
    
    print("\n🔐 Cifrado de Datos:")
    encrypted = encryption.encrypt_content("sensitive data")
    decrypted = encryption.decrypt_content(encrypted)
    print(f"  Datos cifrados y descifrados: {decrypted}")
    
    print("\n🔑 Autenticación Multi-Factor:")
    mfa_setup = await mfa.setup_mfa("user123")
    print(f"  MFA configurado: {bool(mfa_setup.get('secret'))}")
    
    print("\n📊 Métricas Personalizadas:")
    metrics.record_analysis_request("semantic")
    metrics.record_analysis_duration(1.5)
    metrics.update_active_connections(10)
    metrics.update_cache_hit_rate(0.85)
    metrics.update_error_rate(0.02)
    
    current_metrics = metrics.get_metrics()
    print(f"  Métricas actuales: {current_metrics}")
    
    print("\n🚨 Alertas Inteligentes:")
    anomalies = await alerts.check_anomalies(current_metrics)
    print(f"  Anomalías detectadas: {len(anomalies)}")
    for anomaly in anomalies:
        print(f"    - {anomaly['type']}: {anomaly['message']}")
    
    print("\n✅ Demo completado exitosamente!")

if __name__ == "__main__":
    asyncio.run(main())