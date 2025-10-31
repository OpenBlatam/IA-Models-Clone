from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import re
import string
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
🔧 NLP Utilities for Facebook Posts
===================================

Utilidades NLP específicas para análisis de Facebook posts.
"""



def extract_features(text: str) -> Dict[str, float]:
    """Extraer características del texto para análisis."""
    features = {}
    
    # Basic text statistics
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['sentence_count'] = len(re.findall(r'[.!?]+', text))
    
    # Engagement features
    features['question_count'] = text.count('?')
    features['exclamation_count'] = text.count('!')
    features['emoji_count'] = len(re.findall(r'[😀-🿿]', text))
    features['hashtag_count'] = len(re.findall(r'#\w+', text))
    features['mention_count'] = len(re.findall(r'@\w+', text))
    features['url_count'] = len(re.findall(r'http[s]?://', text))
    
    # Language features
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    features['punct_ratio'] = sum(1 for c in text if c in string.punctuation) / len(text) if text else 0
    
    # Readability features
    words = text.split()
    if words:
        features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        features['avg_sentence_length'] = len(words) / max(features['sentence_count'], 1)
    else:
        features['avg_word_length'] = 0
        features['avg_sentence_length'] = 0
    
    return features


def calculate_sentiment_lexicon(text: str) -> Dict[str, float]:
    """Calcular sentimiento usando lexicon simple."""
    positive_words = {
        'amazing': 0.9, 'awesome': 0.8, 'excellent': 0.9, 'fantastic': 0.8,
        'great': 0.7, 'good': 0.6, 'love': 0.8, 'wonderful': 0.8,
        'perfect': 0.9, 'best': 0.8, 'brilliant': 0.8, 'outstanding': 0.9
    }
    
    negative_words = {
        'terrible': -0.9, 'awful': -0.8, 'horrible': -0.9, 'bad': -0.6,
        'worst': -0.9, 'hate': -0.8, 'disappointing': -0.7, 'sad': -0.6,
        'angry': -0.7, 'frustrated': -0.6, 'annoying': -0.6, 'poor': -0.5
    }
    
    words = text.lower().split()
    scores = []
    
    for word in words:
        word_clean = re.sub(r'[^\w]', '', word)
        if word_clean in positive_words:
            scores.append(positive_words[word_clean])
        elif word_clean in negative_words:
            scores.append(negative_words[word_clean])
    
    if not scores:
        return {'polarity': 0.0, 'intensity': 0.0}
    
    polarity = sum(scores) / len(scores)
    intensity = sum(abs(score) for score in scores) / len(scores)
    
    return {'polarity': polarity, 'intensity': intensity}


def detect_content_type(text: str) -> str:
    """Detectar tipo de contenido."""
    text_lower = text.lower()
    
    # Question post
    if '?' in text:
        return 'question'
    
    # Promotional content
    promo_keywords = ['buy', 'sale', 'discount', 'offer', 'deal', 'shop']
    if any(keyword in text_lower for keyword in promo_keywords):
        return 'promotional'
    
    # Educational content
    edu_keywords = ['learn', 'how to', 'guide', 'tutorial', 'tip', 'advice']
    if any(keyword in text_lower for keyword in edu_keywords):
        return 'educational'
    
    # News/informational
    news_keywords = ['breaking', 'news', 'update', 'report', 'announce']
    if any(keyword in text_lower for keyword in news_keywords):
        return 'news'
    
    # Personal/lifestyle
    personal_keywords = ['my', 'i am', 'personal', 'life', 'experience']
    if any(keyword in text_lower for keyword in personal_keywords):
        return 'personal'
    
    return 'general'


def extract_n_grams(text: str, n: int = 2) -> List[str]:
    """Extraer n-gramas del texto."""
    words = text.lower().split()
    words = [re.sub(r'[^\w]', '', word) for word in words if word]
    
    if len(words) < n:
        return []
    
    n_grams = []
    for i in range(len(words) - n + 1):
        n_gram = ' '.join(words[i:i + n])
        n_grams.append(n_gram)
    
    return n_grams


def calculate_text_diversity(text: str) -> float:
    """Calcular diversidad del vocabulario."""
    words = text.lower().split()
    words = [re.sub(r'[^\w]', '', word) for word in words if word]
    
    if not words:
        return 0.0
    
    unique_words = set(words)
    return len(unique_words) / len(words)


def identify_call_to_action(text: str) -> List[str]:
    """Identificar call-to-actions en el texto."""
    cta_patterns = [
        r'click\s+here',
        r'visit\s+our',
        r'follow\s+us',
        r'share\s+this',
        r'comment\s+below',
        r'tell\s+us',
        r'let\s+us\s+know',
        r'check\s+out',
        r'learn\s+more',
        r'sign\s+up',
        r'subscribe',
        r'download',
        r'get\s+started'
    ]
    
    ctas_found = []
    text_lower = text.lower()
    
    for pattern in cta_patterns:
        matches = re.findall(pattern, text_lower)
        ctas_found.extend(matches)
    
    return ctas_found


def analyze_urgency_indicators(text: str) -> Dict[str, int]:
    """Analizar indicadores de urgencia."""
    urgency_patterns = {
        'time_limited': [r'limited\s+time', r'expires\s+soon', r'only\s+today'],
        'scarcity': [r'limited\s+supply', r'few\s+left', r'running\s+out'],
        'immediate': [r'now', r'right\s+now', r'immediately', r'instantly'],
        'deadline': [r'deadline', r'last\s+chance', r'final\s+day', r'ends\s+soon']
    }
    
    urgency_scores = defaultdict(int)
    text_lower = text.lower()
    
    for category, patterns in urgency_patterns.items():
        for pattern in patterns:
            matches = len(re.findall(pattern, text_lower))
            urgency_scores[category] += matches
    
    return dict(urgency_scores)


def extract_social_proof_signals(text: str) -> Dict[str, List[str]]:
    """Extraer señales de prueba social."""
    social_proof = {
        'numbers': re.findall(r'\d+(?:,\d+)*\s*(?:people|users|customers|clients)', text, re.IGNORECASE),
        'testimonials': re.findall(r'customers?\s+(?:say|love|think|believe)', text, re.IGNORECASE),
        'popularity': re.findall(r'(?:popular|trending|viral|bestselling)', text, re.IGNORECASE),
        'authority': re.findall(r'(?:expert|professional|certified|award)', text, re.IGNORECASE)
    }
    
    return {k: v for k, v in social_proof.items() if v}


def calculate_emotional_intensity(text: str) -> float:
    """Calcular intensidad emocional del texto."""
    # Intensifiers
    intensifiers = ['very', 'extremely', 'incredibly', 'absolutely', 'totally', 'completely']
    
    # Emotional words
    emotional_words = [
        'love', 'hate', 'amazing', 'terrible', 'fantastic', 'awful',
        'excited', 'devastated', 'thrilled', 'disappointed', 'furious', 'delighted'
    ]
    
    words = text.lower().split()
    
    intensifier_count = sum(1 for word in words if word in intensifiers)
    emotional_count = sum(1 for word in words if word in emotional_words)
    caps_words = sum(1 for word in text.split() if word.isupper() and len(word) > 1)
    exclamation_count = text.count('!')
    
    # Normalize by text length
    text_length = len(words)
    if text_length == 0:
        return 0.0
    
    intensity = (
        (intensifier_count * 0.3) +
        (emotional_count * 0.4) +
        (caps_words * 0.2) +
        (exclamation_count * 0.1)
    ) / text_length
    
    return min(intensity * 10, 1.0)  # Scale and cap at 1.0


def suggest_improvements(text: str, target_metrics: Dict[str, float]) -> List[str]:
    """Sugerir mejoras basadas en métricas objetivo."""
    improvements = []
    current_features = extract_features(text)
    
    # Length suggestions
    word_count = current_features['word_count']
    target_length = target_metrics.get('optimal_length', 100)
    
    if word_count < target_length * 0.7:
        improvements.append(f"Consider expanding content (current: {word_count} words, target: ~{target_length})")
    elif word_count > target_length * 1.5:
        improvements.append(f"Consider shortening content for better engagement")
    
    # Engagement suggestions
    if current_features['question_count'] == 0:
        improvements.append("Add a question to encourage user interaction")
    
    if current_features['emoji_count'] == 0:
        improvements.append("Add relevant emojis to increase visual appeal")
    
    if current_features['hashtag_count'] < 3:
        improvements.append("Add more relevant hashtags for better discoverability")
    
    # CTA suggestions
    ctas = identify_call_to_action(text)
    if not ctas:
        improvements.append("Include a clear call-to-action")
    
    # Urgency suggestions
    urgency = analyze_urgency_indicators(text)
    if not any(urgency.values()):
        improvements.append("Consider adding urgency indicators if appropriate")
    
    return improvements


def optimize_hashtags(hashtags: List[str], text: str) -> List[str]:
    """Optimizar lista de hashtags."""
    # Extract topics from text
    text_words = set(text.lower().split())
    
    # Score hashtags based on relevance
    scored_hashtags = []
    for hashtag in hashtags:
        relevance_score = 0
        hashtag_words = hashtag.lower().split()
        
        # Check if hashtag words appear in text
        for word in hashtag_words:
            if word in text_words:
                relevance_score += 1
        
        scored_hashtags.append((hashtag, relevance_score))
    
    # Sort by relevance and return top hashtags
    scored_hashtags.sort(key=lambda x: x[1], reverse=True)
    return [hashtag for hashtag, score in scored_hashtags[:10]] 