"""
Brand voice analysis evaluation metrics.

This module provides comprehensive evaluation metrics for brand voice analysis
including tone consistency, vocabulary analysis, sentiment analysis, and brand alignment metrics.
"""

from __future__ import annotations

import re
import math
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter, defaultdict
import string

import numpy as np
from textblob import TextBlob

try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.tokenize import word_tokenize, sent_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def analyze_tone_consistency(texts: List[str], brand_guidelines: Dict[str, Any]) -> Dict[str, float]:
    """Analyze tone consistency across multiple texts."""
    tone_metrics = {}
    
    # Extract tone guidelines
    target_tone = brand_guidelines.get("tone", {})
    formal_threshold = target_tone.get("formality", 0.5)
    friendly_threshold = target_tone.get("friendliness", 0.7)
    professional_threshold = target_tone.get("professionalism", 0.8)
    
    # Analyze each text
    tone_scores = []
    for text in texts:
        text_tone = analyze_single_text_tone(text)
        tone_scores.append(text_tone)
    
    # Calculate consistency metrics
    if tone_scores:
        # Formality consistency
        formality_scores = [score["formality"] for score in tone_scores]
        formality_std = np.std(formality_scores)
        formality_consistency = max(0, 1 - (formality_std / 0.5))  # Normalize to 0-1
        tone_metrics["formality_consistency"] = round(formality_consistency, 3)
        
        # Friendliness consistency
        friendliness_scores = [score["friendliness"] for score in tone_scores]
        friendliness_std = np.std(friendliness_scores)
        friendliness_consistency = max(0, 1 - (friendliness_std / 0.5))
        tone_metrics["friendliness_consistency"] = round(friendliness_consistency, 3)
        
        # Professionalism consistency
        professionalism_scores = [score["professionalism"] for score in tone_scores]
        professionalism_std = np.std(professionalism_scores)
        professionalism_consistency = max(0, 1 - (professionalism_std / 0.5))
        tone_metrics["professionalism_consistency"] = round(professionalism_consistency, 3)
        
        # Overall tone consistency
        overall_consistency = (formality_consistency + friendliness_consistency + professionalism_consistency) / 3
        tone_metrics["overall_tone_consistency"] = round(overall_consistency, 3)
        
        # Average tone scores
        tone_metrics["avg_formality"] = round(np.mean(formality_scores), 3)
        tone_metrics["avg_friendliness"] = round(np.mean(friendliness_scores), 3)
        tone_metrics["avg_professionalism"] = round(np.mean(professionalism_scores), 3)
    
    return tone_metrics


def analyze_single_text_tone(text: str) -> Dict[str, float]:
    """Analyze tone characteristics of a single text."""
    tone_analysis = {}
    
    # Convert to lowercase for analysis
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    # Formality analysis
    formal_indicators = [
        'therefore', 'consequently', 'furthermore', 'moreover', 'thus', 'hence',
        'accordingly', 'subsequently', 'nevertheless', 'nonetheless', 'whereas',
        'whilst', 'whom', 'whose', 'wherein', 'whereby'
    ]
    
    informal_indicators = [
        'gonna', 'wanna', 'gotta', 'lemme', 'gimme', 'y\'all', 'ain\'t',
        'dunno', 'gonna', 'wanna', 'gotta', 'lemme', 'gimme', 'y\'all'
    ]
    
    formal_count = sum(1 for word in words if word in formal_indicators)
    informal_count = sum(1 for word in words if word in informal_indicators)
    
    total_indicators = formal_count + informal_count
    if total_indicators > 0:
        formality = formal_count / total_indicators
    else:
        # Default to neutral formality
        formality = 0.5
    
    tone_analysis["formality"] = round(formality, 3)
    
    # Friendliness analysis
    friendly_indicators = [
        'great', 'awesome', 'amazing', 'wonderful', 'fantastic', 'excellent',
        'love', 'like', 'enjoy', 'happy', 'glad', 'pleased', 'excited',
        'welcome', 'hello', 'hi', 'thanks', 'thank you', 'appreciate'
    ]
    
    unfriendly_indicators = [
        'terrible', 'awful', 'horrible', 'bad', 'hate', 'dislike', 'angry',
        'upset', 'disappointed', 'frustrated', 'annoyed', 'irritated'
    ]
    
    friendly_count = sum(1 for word in words if word in friendly_indicators)
    unfriendly_count = sum(1 for word in words if word in unfriendly_indicators)
    
    total_friendliness_indicators = friendly_count + unfriendly_count
    if total_friendliness_indicators > 0:
        friendliness = friendly_count / total_friendliness_indicators
    else:
        # Default to neutral friendliness
        friendliness = 0.5
    
    tone_analysis["friendliness"] = round(friendliness, 3)
    
    # Professionalism analysis
    professional_indicators = [
        'professional', 'expertise', 'experience', 'quality', 'reliable',
        'trustworthy', 'confidential', 'secure', 'efficient', 'effective',
        'innovative', 'strategic', 'comprehensive', 'thorough', 'precise'
    ]
    
    unprofessional_indicators = [
        'slang', 'casual', 'informal', 'relaxed', 'chill', 'cool', 'awesome',
        'gonna', 'wanna', 'gotta', 'lemme', 'gimme', 'y\'all', 'ain\'t'
    ]
    
    professional_count = sum(1 for word in words if word in professional_indicators)
    unprofessional_count = sum(1 for word in words if word in unprofessional_indicators)
    
    total_professional_indicators = professional_count + unprofessional_count
    if total_professional_indicators > 0:
        professionalism = professional_count / total_professional_indicators
    else:
        # Default to neutral professionalism
        professionalism = 0.5
    
    tone_analysis["professionalism"] = round(professionalism, 3)
    
    return tone_analysis


def analyze_vocabulary_consistency(texts: List[str], brand_guidelines: Dict[str, Any]) -> Dict[str, float]:
    """Analyze vocabulary consistency and brand-specific terminology usage."""
    vocab_metrics = {}
    
    # Extract brand vocabulary guidelines
    brand_terms = brand_guidelines.get("vocabulary", {}).get("brand_terms", [])
    preferred_terms = brand_guidelines.get("vocabulary", {}).get("preferred_terms", {})
    avoided_terms = brand_guidelines.get("vocabulary", {}).get("avoided_terms", [])
    
    # Analyze vocabulary across all texts
    all_words = []
    brand_term_usage = defaultdict(int)
    preferred_term_usage = defaultdict(int)
    avoided_term_usage = defaultdict(int)
    
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        all_words.extend(words)
        
        # Count brand term usage
        for term in brand_terms:
            term_lower = term.lower()
            count = len(re.findall(rf'\b{re.escape(term_lower)}\b', text.lower()))
            brand_term_usage[term] += count
        
        # Count preferred term usage
        for preferred, alternatives in preferred_terms.items():
            preferred_lower = preferred.lower()
            count = len(re.findall(rf'\b{re.escape(preferred_lower)}\b', text.lower()))
            preferred_term_usage[preferred] += count
        
        # Count avoided term usage
        for term in avoided_terms:
            term_lower = term.lower()
            count = len(re.findall(rf'\b{re.escape(term_lower)}\b', text.lower()))
            avoided_term_usage[term] += count
    
    # Calculate vocabulary metrics
    total_words = len(all_words)
    unique_words = len(set(all_words))
    
    # Vocabulary diversity (Type-Token Ratio)
    vocab_metrics["vocabulary_diversity"] = round(unique_words / max(total_words, 1), 3)
    
    # Brand term consistency
    if brand_terms:
        total_brand_usage = sum(brand_term_usage.values())
        brand_term_density = total_brand_usage / max(total_words, 1)
        vocab_metrics["brand_term_density"] = round(brand_term_density, 4)
        
        # Brand term distribution consistency
        brand_term_counts = list(brand_term_usage.values())
        if brand_term_counts:
            brand_term_std = np.std(brand_term_counts)
            brand_term_mean = np.mean(brand_term_counts)
            if brand_term_mean > 0:
                brand_term_cv = brand_term_std / brand_term_mean
                brand_term_consistency = max(0, 1 - brand_term_cv)
            else:
                brand_term_consistency = 0
            vocab_metrics["brand_term_consistency"] = round(brand_term_consistency, 3)
        else:
            vocab_metrics["brand_term_consistency"] = 0.0
    else:
        vocab_metrics["brand_term_density"] = 0.0
        vocab_metrics["brand_term_consistency"] = 0.0
    
    # Preferred term usage
    if preferred_terms:
        total_preferred_usage = sum(preferred_term_usage.values())
        preferred_term_density = total_preferred_usage / max(total_words, 1)
        vocab_metrics["preferred_term_density"] = round(preferred_term_density, 4)
    else:
        vocab_metrics["preferred_term_density"] = 0.0
    
    # Avoided term usage (penalty)
    if avoided_terms:
        total_avoided_usage = sum(avoided_term_usage.values())
        avoided_term_density = total_avoided_usage / max(total_words, 1)
        vocab_metrics["avoided_term_density"] = round(avoided_term_density, 4)
        
        # Penalty score (lower is better)
        avoided_term_penalty = min(1.0, total_avoided_usage / max(total_words / 100, 1))
        vocab_metrics["avoided_term_penalty"] = round(avoided_term_penalty, 3)
    else:
        vocab_metrics["avoided_term_density"] = 0.0
        vocab_metrics["avoided_term_penalty"] = 0.0
    
    return vocab_metrics


def analyze_sentiment_consistency(texts: List[str]) -> Dict[str, float]:
    """Analyze sentiment consistency across multiple texts."""
    sentiment_metrics = {}
    
    if not NLTK_AVAILABLE:
        # Fallback to TextBlob
        sentiment_scores = []
        for text in texts:
            blob = TextBlob(text)
            sentiment_scores.append(blob.sentiment.polarity)
    else:
        # Use NLTK VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = []
        for text in texts:
            scores = sia.polarity_scores(text)
            sentiment_scores.append(scores['compound'])
    
    if sentiment_scores:
        # Sentiment consistency (lower std = higher consistency)
        sentiment_std = np.std(sentiment_scores)
        sentiment_consistency = max(0, 1 - (sentiment_std / 2.0))  # Normalize to 0-1
        sentiment_metrics["sentiment_consistency"] = round(sentiment_consistency, 3)
        
        # Average sentiment
        sentiment_metrics["avg_sentiment"] = round(np.mean(sentiment_scores), 3)
        
        # Sentiment range
        sentiment_metrics["sentiment_range"] = round(max(sentiment_scores) - min(sentiment_scores), 3)
        
        # Sentiment stability (inverse of range)
        sentiment_stability = max(0, 1 - (sentiment_metrics["sentiment_range"] / 2.0))
        sentiment_metrics["sentiment_stability"] = round(sentiment_stability, 3)
    
    return sentiment_metrics


def analyze_brand_alignment(texts: List[str], brand_guidelines: Dict[str, Any]) -> Dict[str, float]:
    """Analyze overall brand alignment and consistency."""
    alignment_metrics = {}
    
    # Get individual analysis results
    tone_consistency = analyze_tone_consistency(texts, brand_guidelines)
    vocab_consistency = analyze_vocabulary_consistency(texts, brand_guidelines)
    sentiment_consistency = analyze_sentiment_consistency(texts)
    
    # Calculate overall brand alignment score
    alignment_components = []
    
    # Tone alignment (30% weight)
    if "overall_tone_consistency" in tone_consistency:
        tone_alignment = tone_consistency["overall_tone_consistency"]
        alignment_components.append(("tone", tone_alignment, 0.3))
    
    # Vocabulary alignment (35% weight)
    if "brand_term_consistency" in vocab_consistency:
        vocab_alignment = vocab_consistency["brand_term_consistency"]
        # Penalize avoided term usage
        avoided_penalty = vocab_consistency.get("avoided_term_penalty", 0)
        vocab_alignment = max(0, vocab_alignment - avoided_penalty)
        alignment_components.append(("vocabulary", vocab_alignment, 0.35))
    
    # Sentiment alignment (20% weight)
    if "sentiment_consistency" in sentiment_consistency:
        sentiment_alignment = sentiment_consistency["sentiment_consistency"]
        alignment_components.append(("sentiment", sentiment_alignment, 0.2))
    
    # Content structure alignment (15% weight)
    structure_alignment = analyze_content_structure_alignment(texts, brand_guidelines)
    alignment_components.append(("structure", structure_alignment, 0.15))
    
    # Calculate weighted overall alignment
    total_weight = sum(weight for _, _, weight in alignment_components)
    if total_weight > 0:
        overall_alignment = sum(score * weight for _, score, weight in alignment_components) / total_weight
        alignment_metrics["overall_brand_alignment"] = round(overall_alignment, 3)
        
        # Individual component scores
        for component, score, _ in alignment_components:
            alignment_metrics[f"{component}_alignment"] = round(score, 3)
    else:
        alignment_metrics["overall_brand_alignment"] = 0.0
    
    # Brand alignment grade
    overall_score = alignment_metrics["overall_brand_alignment"]
    if overall_score >= 0.9:
        grade = "A+"
    elif overall_score >= 0.8:
        grade = "A"
    elif overall_score >= 0.7:
        grade = "B+"
    elif overall_score >= 0.6:
        grade = "B"
    elif overall_score >= 0.5:
        grade = "C"
    else:
        grade = "D"
    
    alignment_metrics["brand_alignment_grade"] = grade
    
    return alignment_metrics


def analyze_content_structure_alignment(texts: List[str], brand_guidelines: Dict[str, Any]) -> float:
    """Analyze content structure alignment with brand guidelines."""
    structure_guidelines = brand_guidelines.get("content_structure", {})
    
    if not structure_guidelines:
        return 0.5  # Default neutral score
    
    alignment_scores = []
    
    for text in texts:
        text_score = 0
        max_possible = 0
        
        # Paragraph length guidelines
        if "paragraph_length" in structure_guidelines:
            target_length = structure_guidelines["paragraph_length"]
            paragraphs = text.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            if paragraphs:
                avg_length = np.mean([len(p.split()) for p in paragraphs])
                if target_length - 50 <= avg_length <= target_length + 50:
                    text_score += 1
                max_possible += 1
        
        # Heading structure guidelines
        if "heading_structure" in structure_guidelines:
            heading_pattern = structure_guidelines["heading_structure"]
            headings = re.findall(r'^#{1,6}\s+(.+)$', text, re.MULTILINE)
            
            if headings:
                heading_levels = [len(h.split()[0]) for h in headings]
                if heading_levels == sorted(heading_levels):  # Proper hierarchy
                    text_score += 1
                max_possible += 1
        
        # Content length guidelines
        if "content_length" in structure_guidelines:
            target_length = structure_guidelines["content_length"]
            word_count = len(re.findall(r'\b\w+\b', text))
            
            if target_length - 100 <= word_count <= target_length + 100:
                text_score += 1
            max_possible += 1
        
        # Calculate score for this text
        if max_possible > 0:
            text_alignment = text_score / max_possible
            alignment_scores.append(text_alignment)
    
    # Return average alignment score
    if alignment_scores:
        return np.mean(alignment_scores)
    else:
        return 0.5


def evaluate_brand_voice(texts: List[str], brand_guidelines: Dict[str, Any]) -> Dict[str, Any]:
    """
    Comprehensive evaluation of brand voice consistency and alignment.
    
    Args:
        texts: List of text samples to evaluate
        brand_guidelines: Dictionary containing brand voice guidelines
    
    Returns:
        Dictionary containing all brand voice evaluation metrics
    """
    evaluation = {}
    
    # Individual analysis components
    evaluation["tone_consistency"] = analyze_tone_consistency(texts, brand_guidelines)
    evaluation["vocabulary_consistency"] = analyze_vocabulary_consistency(texts, brand_guidelines)
    evaluation["sentiment_consistency"] = analyze_sentiment_consistency(texts)
    evaluation["brand_alignment"] = analyze_brand_alignment(texts, brand_guidelines)
    
    # Overall assessment
    overall_alignment = evaluation["brand_alignment"].get("overall_brand_alignment", 0)
    
    # Generate recommendations
    recommendations = []
    
    # Tone recommendations
    tone_consistency = evaluation["tone_consistency"]
    if tone_consistency.get("overall_tone_consistency", 0) < 0.7:
        recommendations.append("Improve tone consistency across content pieces")
    
    # Vocabulary recommendations
    vocab_consistency = evaluation["vocabulary_consistency"]
    if vocab_consistency.get("brand_term_consistency", 0) < 0.6:
        recommendations.append("Increase consistent usage of brand-specific terminology")
    
    if vocab_consistency.get("avoided_term_penalty", 0) > 0.1:
        recommendations.append("Reduce usage of terms that don't align with brand guidelines")
    
    # Sentiment recommendations
    sentiment_consistency = evaluation["sentiment_consistency"]
    if sentiment_consistency.get("sentiment_consistency", 0) < 0.6:
        recommendations.append("Maintain more consistent emotional tone across content")
    
    # Overall recommendations
    if overall_alignment < 0.7:
        recommendations.append("Overall brand alignment needs improvement - review brand guidelines")
    elif overall_alignment < 0.8:
        recommendations.append("Brand alignment is good but could be improved")
    else:
        recommendations.append("Excellent brand alignment - maintain current approach")
    
    evaluation["recommendations"] = recommendations
    
    return evaluation
