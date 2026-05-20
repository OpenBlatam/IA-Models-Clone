"""
Servicio de Análisis de Redes Sociales Avanzado - Sistema completo de análisis social
"""

from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class SocialMediaAnalysisService:
    """Servicio de análisis de redes sociales avanzado"""
    
    def __init__(self):
        """Inicializa el servicio de análisis social"""
        pass
    
    def analyze_social_activity(
        self,
        user_id: str,
        social_posts: List[Dict],
        platform: str = "general"
    ) -> Dict:
        """
        Analiza actividad en redes sociales
        
        Args:
            user_id: ID del usuario
            social_posts: Lista de publicaciones
            platform: Plataforma social
        
        Returns:
            Análisis de actividad social
        """
        if not social_posts:
            return {
                "user_id": user_id,
                "analysis": "no_data"
            }
        
        analysis = {
            "user_id": user_id,
            "platform": platform,
            "total_posts": len(social_posts),
            "sentiment_analysis": self._analyze_sentiment(social_posts),
            "topic_analysis": self._analyze_topics(social_posts),
            "engagement_metrics": self._calculate_engagement(social_posts),
            "risk_indicators": self._detect_social_risk_indicators(social_posts),
            "generated_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def detect_social_triggers(
        self,
        user_id: str,
        social_content: List[Dict]
    ) -> Dict:
        """
        Detecta triggers en contenido social
        
        Args:
            user_id: ID del usuario
            social_content: Contenido social
        
        Returns:
            Triggers detectados
        """
        triggers = []
        
        trigger_keywords = ["alcohol", "cigarrillo", "droga", "fiesta", "bar"]
        
        for content in social_content:
            text = content.get("text", "").lower()
            for keyword in trigger_keywords:
                if keyword in text:
                    triggers.append({
                        "content_id": content.get("id"),
                        "trigger_keyword": keyword,
                        "severity": "medium",
                        "detected_at": datetime.now().isoformat()
                    })
        
        return {
            "user_id": user_id,
            "triggers_detected": triggers,
            "total_triggers": len(triggers),
            "recommendations": self._generate_trigger_recommendations(triggers),
            "generated_at": datetime.now().isoformat()
        }
    
    def analyze_social_support_network(
        self,
        user_id: str,
        social_connections: List[Dict]
    ) -> Dict:
        """
        Analiza red de apoyo social
        
        Args:
            user_id: ID del usuario
            social_connections: Conexiones sociales
        
        Returns:
            Análisis de red de apoyo
        """
        return {
            "user_id": user_id,
            "total_connections": len(social_connections),
            "supportive_connections": sum(1 for c in social_connections if c.get("supportive", False)),
            "risk_connections": sum(1 for c in social_connections if c.get("risk_factor", False)),
            "network_health_score": 7.5,
            "recommendations": [
                "Fortalecer conexiones con personas de apoyo",
                "Reducir exposición a conexiones de riesgo"
            ],
            "generated_at": datetime.now().isoformat()
        }
    
    def _analyze_sentiment(self, posts: List[Dict]) -> Dict:
        """Analiza sentimiento en publicaciones"""
        positive_count = sum(1 for p in posts if p.get("sentiment") == "positive")
        negative_count = sum(1 for p in posts if p.get("sentiment") == "negative")
        
        return {
            "positive": positive_count,
            "negative": negative_count,
            "neutral": len(posts) - positive_count - negative_count,
            "overall_sentiment": "positive" if positive_count > negative_count else "negative"
        }
    
    def _analyze_topics(self, posts: List[Dict]) -> List[Dict]:
        """Analiza temas en publicaciones"""
        topics = defaultdict(int)
        
        topic_keywords = {
            "recovery": ["recuperación", "sobriedad", "progreso"],
            "support": ["apoyo", "ayuda", "familia"],
            "challenges": ["desafío", "difícil", "lucha"]
        }
        
        for post in posts:
            text = post.get("text", "").lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in text for keyword in keywords):
                    topics[topic] += 1
        
        return [{"topic": k, "count": v} for k, v in topics.items()]
    
    def _calculate_engagement(self, posts: List[Dict]) -> Dict:
        """Calcula métricas de engagement"""
        total_likes = sum(p.get("likes", 0) for p in posts)
        total_comments = sum(p.get("comments", 0) for p in posts)
        
        return {
            "total_likes": total_likes,
            "total_comments": total_comments,
            "average_engagement": (total_likes + total_comments) / len(posts) if posts else 0
        }
    
    def _detect_social_risk_indicators(self, posts: List[Dict]) -> List[str]:
        """Detecta indicadores de riesgo en publicaciones"""
        indicators = []
        
        risk_keywords = ["recaída", "tentación", "desesperado"]
        
        for post in posts:
            text = post.get("text", "").lower()
            for keyword in risk_keywords:
                if keyword in text:
                    indicators.append(keyword)
                    break
        
        return indicators
    
    def _generate_trigger_recommendations(self, triggers: List[Dict]) -> List[str]:
        """Genera recomendaciones basadas en triggers"""
        if triggers:
            return [
                "Considera filtrar contenido relacionado con adicciones",
                "Limita exposición a contenido que pueda desencadenar cravings"
            ]
        return []

