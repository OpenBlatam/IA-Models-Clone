"""
PDF Variantes Text Utilities
Utilidades de texto para el sistema PDF Variantes
"""

import re
import string
from typing import Dict, List, Any, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Procesador de texto"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Descargar recursos de NLTK si no están disponibles
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
    
    def clean_text(self, text: str) -> str:
        """Limpiar texto"""
        try:
            # Convertir a minúsculas
            text = text.lower()
            
            # Eliminar caracteres especiales
            text = re.sub(r'[^\w\s]', '', text)
            
            # Eliminar espacios extra
            text = re.sub(r'\s+', ' ', text)
            
            # Eliminar espacios al inicio y final
            text = text.strip()
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {e}")
            return text
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenizar texto"""
        try:
            tokens = word_tokenize(text)
            return tokens
            
        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Eliminar palabras vacías"""
        try:
            filtered_tokens = [token for token in tokens if token not in self.stop_words]
            return filtered_tokens
            
        except Exception as e:
            logger.error(f"Error removing stopwords: {e}")
            return tokens
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Aplicar stemming a tokens"""
        try:
            stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
            return stemmed_tokens
            
        except Exception as e:
            logger.error(f"Error stemming tokens: {e}")
            return tokens
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extraer palabras clave"""
        try:
            # Limpiar y tokenizar texto
            cleaned_text = self.clean_text(text)
            tokens = self.tokenize_text(cleaned_text)
            
            # Eliminar palabras vacías
            filtered_tokens = self.remove_stopwords(tokens)
            
            # Aplicar stemming
            stemmed_tokens = self.stem_tokens(filtered_tokens)
            
            # Contar frecuencia
            word_freq = {}
            for token in stemmed_tokens:
                word_freq[token] = word_freq.get(token, 0) + 1
            
            # Ordenar por frecuencia
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Retornar palabras clave
            keywords = [word for word, freq in sorted_words[:num_keywords]]
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extraer entidades"""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            
            entities = []
            current_entity = ""
            current_type = ""
            
            for token, pos in pos_tags:
                if pos in ['NNP', 'NNPS']:  # Proper nouns
                    if current_type == pos:
                        current_entity += " " + token
                    else:
                        if current_entity:
                            entities.append({
                                "text": current_entity.strip(),
                                "type": current_type,
                                "start": text.find(current_entity),
                                "end": text.find(current_entity) + len(current_entity)
                            })
                        current_entity = token
                        current_type = pos
                else:
                    if current_entity:
                        entities.append({
                            "text": current_entity.strip(),
                            "type": current_type,
                            "start": text.find(current_entity),
                            "end": text.find(current_entity) + len(current_entity)
                        })
                        current_entity = ""
                        current_type = ""
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return []
    
    def calculate_readability(self, text: str) -> Dict[str, float]:
        """Calcular legibilidad"""
        try:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            
            # Contar sílabas (aproximación)
            syllables = 0
            for word in words:
                syllables += self._count_syllables(word)
            
            # Fórmula de Flesch Reading Ease
            if len(sentences) > 0 and len(words) > 0:
                flesch_score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
            else:
                flesch_score = 0
            
            # Fórmula de Flesch-Kincaid Grade Level
            if len(sentences) > 0 and len(words) > 0:
                fk_grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59
            else:
                fk_grade = 0
            
            return {
                "flesch_score": flesch_score,
                "flesch_kincaid_grade": fk_grade,
                "sentence_count": len(sentences),
                "word_count": len(words),
                "syllable_count": syllables
            }
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return {}
    
    def _count_syllables(self, word: str) -> int:
        """Contar sílabas en una palabra"""
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Ajustar para palabras que terminan en 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        # Mínimo 1 sílaba
        return max(1, syllable_count)
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analizar sentimiento"""
        try:
            # Implementación básica de análisis de sentimiento
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'worst']
            
            words = word_tokenize(text.lower())
            
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_words = len(words)
            
            if total_words > 0:
                positive_ratio = positive_count / total_words
                negative_ratio = negative_count / total_words
                
                if positive_ratio > negative_ratio:
                    sentiment = "positive"
                    confidence = positive_ratio
                elif negative_ratio > positive_ratio:
                    sentiment = "negative"
                    confidence = negative_ratio
                else:
                    sentiment = "neutral"
                    confidence = 0.5
            else:
                sentiment = "neutral"
                confidence = 0.5
            
            return {
                "sentiment": sentiment,
                "confidence": confidence,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "total_words": total_words
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
    
    def extract_topics(self, text: str, num_topics: int = 5) -> List[Dict[str, Any]]:
        """Extraer temas"""
        try:
            # Implementación básica de extracción de temas
            keywords = self.extract_keywords(text, num_topics * 2)
            
            topics = []
            for i, keyword in enumerate(keywords[:num_topics]):
                topics.append({
                    "topic": keyword,
                    "relevance_score": 1.0 - (i * 0.1),
                    "mentions": text.lower().count(keyword),
                    "context": self._get_context(text, keyword)
                })
            
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    def _get_context(self, text: str, keyword: str, context_length: int = 50) -> List[str]:
        """Obtener contexto de una palabra clave"""
        try:
            contexts = []
            text_lower = text.lower()
            keyword_lower = keyword.lower()
            
            start = 0
            while True:
                pos = text_lower.find(keyword_lower, start)
                if pos == -1:
                    break
                
                context_start = max(0, pos - context_length)
                context_end = min(len(text), pos + len(keyword) + context_length)
                context = text[context_start:context_end]
                contexts.append(context.strip())
                
                start = pos + 1
            
            return contexts[:3]  # Máximo 3 contextos
            
        except Exception as e:
            logger.error(f"Error getting context: {e}")
        return []
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generar resumen"""
        try:
            sentences = sent_tokenize(text)
            
            if len(sentences) <= max_sentences:
                return text
            
            # Calcular puntuación de oraciones
            sentence_scores = []
            for sentence in sentences:
                score = 0
                words = word_tokenize(sentence.lower())
                
                # Puntuación basada en longitud
                score += len(words) * 0.1
                
                # Puntuación basada en palabras clave
                keywords = self.extract_keywords(text, 10)
    for word in words:
                    if word in keywords:
                        score += 1
                
                sentence_scores.append((sentence, score))
            
            # Ordenar por puntuación
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Seleccionar mejores oraciones
            selected_sentences = [sentence for sentence, score in sentence_scores[:max_sentences]]
            
            return " ".join(selected_sentences)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:200] + "..." if len(text) > 200 else text

class TextAnalyzer:
    """Analizador de texto"""
    
    def __init__(self):
        self.processor = TextProcessor()
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analizar texto completo"""
        try:
            analysis = {
                "text_length": len(text),
                "word_count": len(word_tokenize(text)),
                "sentence_count": len(sent_tokenize(text)),
                "keywords": self.processor.extract_keywords(text),
                "entities": self.processor.extract_entities(text),
                "readability": self.processor.calculate_readability(text),
                "sentiment": self.processor.analyze_sentiment(text),
                "topics": self.processor.extract_topics(text),
                "summary": self.processor.generate_summary(text)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {}
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """Comparar dos textos"""
        try:
            analysis1 = self.analyze_text(text1)
            analysis2 = self.analyze_text(text2)
            
            # Calcular similitud
            keywords1 = set(analysis1.get("keywords", []))
            keywords2 = set(analysis2.get("keywords", []))
            
            common_keywords = keywords1.intersection(keywords2)
            total_keywords = keywords1.union(keywords2)
            
            similarity = len(common_keywords) / len(total_keywords) if total_keywords else 0
            
            return {
                "similarity_score": similarity,
                "common_keywords": list(common_keywords),
                "unique_to_text1": list(keywords1 - keywords2),
                "unique_to_text2": list(keywords2 - keywords1),
                "analysis1": analysis1,
                "analysis2": analysis2
            }
            
        except Exception as e:
            logger.error(f"Error comparing texts: {e}")
            return {}

# Factory functions
def create_text_processor() -> TextProcessor:
    """Crear procesador de texto"""
    return TextProcessor()

def create_text_analyzer() -> TextAnalyzer:
    """Crear analizador de texto"""
    return TextAnalyzer()