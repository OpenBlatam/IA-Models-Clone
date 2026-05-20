"""
Servicio de Análisis de Texto Avanzado con NLP - Procesamiento de lenguaje natural
"""

from typing import Dict, List, Optional
from datetime import datetime
import re


class NLPAnalysisService:
    """Servicio de análisis de texto con NLP"""
    
    def __init__(self):
        """Inicializa el servicio de análisis NLP"""
        pass
    
    def analyze_text(
        self,
        text: str,
        analysis_type: str = "comprehensive"
    ) -> Dict:
        """
        Analiza texto con NLP
        
        Args:
            text: Texto a analizar
            analysis_type: Tipo de análisis (comprehensive, sentiment, keywords, topics)
        
        Returns:
            Análisis del texto
        """
        analysis = {
            "text": text,
            "analysis_type": analysis_type,
            "word_count": len(text.split()),
            "character_count": len(text),
            "sentiment": self._analyze_sentiment(text),
            "keywords": self._extract_keywords(text),
            "topics": self._extract_topics(text),
            "entities": self._extract_entities(text),
            "readability_score": self._calculate_readability(text),
            "analyzed_at": datetime.now().isoformat()
        }
        
        return analysis
    
    def extract_insights(
        self,
        text: str,
        context: Optional[Dict] = None
    ) -> Dict:
        """
        Extrae insights del texto
        
        Args:
            text: Texto a analizar
            context: Contexto adicional (opcional)
        
        Returns:
            Insights extraídos
        """
        insights = {
            "text": text,
            "key_insights": [],
            "risk_indicators": self._detect_risk_indicators(text),
            "positive_indicators": self._detect_positive_indicators(text),
            "recommendations": [],
            "extracted_at": datetime.now().isoformat()
        }
        
        # Generar insights basados en análisis
        if self._detect_risk_indicators(text):
            insights["key_insights"].append("Se detectaron indicadores de riesgo en el texto")
            insights["recommendations"].append("Considera contactar a tu sistema de apoyo")
        
        if self._detect_positive_indicators(text):
            insights["key_insights"].append("Se detectaron indicadores positivos de progreso")
        
        return insights
    
    def compare_texts(
        self,
        text1: str,
        text2: str
    ) -> Dict:
        """
        Compara dos textos
        
        Args:
            text1: Primer texto
            text2: Segundo texto
        
        Returns:
            Comparación de textos
        """
        analysis1 = self.analyze_text(text1)
        analysis2 = self.analyze_text(text2)
        
        return {
            "text1_analysis": analysis1,
            "text2_analysis": analysis2,
            "similarity_score": self._calculate_similarity(text1, text2),
            "sentiment_change": self._compare_sentiment(analysis1["sentiment"], analysis2["sentiment"]),
            "topic_evolution": self._compare_topics(analysis1["topics"], analysis2["topics"]),
            "compared_at": datetime.now().isoformat()
        }
    
    def analyze_journal_entry(
        self,
        entry_text: str,
        entry_date: str
    ) -> Dict:
        """
        Analiza entrada de diario
        
        Args:
            entry_text: Texto de la entrada
            entry_date: Fecha de la entrada
        
        Returns:
            Análisis de entrada de diario
        """
        analysis = self.analyze_text(entry_text, "comprehensive")
        insights = self.extract_insights(entry_text)
        
        return {
            "entry_date": entry_date,
            "analysis": analysis,
            "insights": insights,
            "progress_indicators": self._extract_progress_indicators(entry_text),
            "challenges_mentioned": self._extract_challenges(entry_text),
            "analyzed_at": datetime.now().isoformat()
        }
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analiza sentimiento del texto"""
        positive_words = ["feliz", "bien", "mejor", "progreso", "éxito", "orgulloso"]
        negative_words = ["triste", "mal", "difícil", "fracaso", "ansiedad", "miedo"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
        elif negative_count > positive_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "positive_score": positive_count,
            "negative_score": negative_count,
            "confidence": abs(positive_count - negative_count) / max(len(text.split()), 1)
        }
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extrae palabras clave"""
        # Palabras comunes a excluir
        stop_words = {"el", "la", "de", "que", "y", "a", "en", "un", "es", "se", "no", "te", "lo", "le", "da", "su", "por", "son", "con", "para"}
        
        words = re.findall(r'\b\w+\b', text.lower())
        words = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Contar frecuencia
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Obtener top N
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extrae temas del texto"""
        topics = []
        
        # Detectar temas comunes
        topic_keywords = {
            "recuperación": ["recuperación", "progreso", "mejora", "avance"],
            "desafíos": ["difícil", "desafío", "problema", "obstáculo"],
            "apoyo": ["apoyo", "ayuda", "familia", "amigos", "terapeuta"],
            "salud": ["salud", "ejercicio", "sueño", "energía"],
            "emociones": ["feliz", "triste", "ansiedad", "miedo", "esperanza"]
        }
        
        text_lower = text.lower()
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extrae entidades nombradas"""
        entities = []
        
        # Detectar fechas
        date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
        dates = re.findall(date_pattern, text)
        for date in dates:
            entities.append({"type": "date", "value": date})
        
        # Detectar números
        number_pattern = r'\b\d+\b'
        numbers = re.findall(number_pattern, text)
        for number in numbers:
            entities.append({"type": "number", "value": number})
        
        return entities
    
    def _calculate_readability(self, text: str) -> float:
        """Calcula puntuación de legibilidad"""
        sentences = text.split('.')
        words = text.split()
        
        if len(sentences) == 0:
            return 0.0
        
        avg_sentence_length = len(words) / len(sentences)
        readability = max(0, min(100, 100 - (avg_sentence_length * 2)))
        
        return round(readability, 2)
    
    def _detect_risk_indicators(self, text: str) -> List[str]:
        """Detecta indicadores de riesgo"""
        risk_keywords = ["recaída", "tentación", "desesperado", "sin esperanza", "rendirse"]
        text_lower = text.lower()
        
        detected = []
        for keyword in risk_keywords:
            if keyword in text_lower:
                detected.append(keyword)
        
        return detected
    
    def _detect_positive_indicators(self, text: str) -> List[str]:
        """Detecta indicadores positivos"""
        positive_keywords = ["progreso", "mejor", "orgulloso", "logro", "éxito", "determinación"]
        text_lower = text.lower()
        
        detected = []
        for keyword in positive_keywords:
            if keyword in text_lower:
                detected.append(keyword)
        
        return detected
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcula similitud entre textos"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union) if union else 0.0
        return round(similarity, 2)
    
    def _compare_sentiment(self, sentiment1: Dict, sentiment2: Dict) -> Dict:
        """Compara sentimientos"""
        return {
            "from": sentiment1["sentiment"],
            "to": sentiment2["sentiment"],
            "change": "improved" if sentiment2["sentiment"] == "positive" and sentiment1["sentiment"] != "positive" else "declined"
        }
    
    def _compare_topics(self, topics1: List[str], topics2: List[str]) -> Dict:
        """Compara temas"""
        return {
            "common_topics": list(set(topics1).intersection(set(topics2))),
            "new_topics": list(set(topics2) - set(topics1)),
            "removed_topics": list(set(topics1) - set(topics2))
        }
    
    def _extract_progress_indicators(self, text: str) -> List[str]:
        """Extrae indicadores de progreso"""
        progress_keywords = ["día", "semana", "mes", "progreso", "mejor", "avance"]
        text_lower = text.lower()
        
        indicators = []
        for keyword in progress_keywords:
            if keyword in text_lower:
                indicators.append(keyword)
        
        return indicators
    
    def _extract_challenges(self, text: str) -> List[str]:
        """Extrae desafíos mencionados"""
        challenge_keywords = ["difícil", "desafío", "problema", "obstáculo", "lucha"]
        text_lower = text.lower()
        
        challenges = []
        for keyword in challenge_keywords:
            if keyword in text_lower:
                challenges.append(keyword)
        
        return challenges

