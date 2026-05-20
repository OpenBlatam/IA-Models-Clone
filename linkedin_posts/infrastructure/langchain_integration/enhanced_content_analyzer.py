from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import re
from collections import Counter
import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat
from readability import Readability
import spacy
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from ...shared.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
"""
Enhanced Content Analyzer
=========================

Advanced content analysis using multiple NLP libraries for comprehensive
LinkedIn post optimization and analysis.
"""




logger = get_logger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except Exception as e:
    logger.warning(f"Could not download NLTK data: {e}")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None


class EnhancedContentAnalyzer:
    """
    Enhanced content analyzer using multiple NLP libraries.
    
    Provides comprehensive content analysis including:
    - Sentiment analysis (multiple approaches)
    - Readability metrics
    - Content structure analysis
    - Keyword analysis
    - Engagement prediction
    - SEO optimization
    - Content quality scoring
    """
    
    def __init__(self) -> Any:
        """Initialize the enhanced content analyzer."""
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize transformers pipeline for advanced analysis
        try:
            self.text_classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased",
                return_all_scores=True
            )
        except Exception as e:
            logger.warning(f"Could not load transformers pipeline: {e}")
            self.text_classifier = None
    
    async def comprehensive_analysis(self, content: str, target_audience: str = "general") -> Dict[str, Any]:
        """
        Perform comprehensive content analysis.
        
        Args:
            content: Text content to analyze
            target_audience: Target audience for analysis
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Run all analyses concurrently
            tasks = [
                self._analyze_sentiment(content),
                self._analyze_readability(content),
                self._analyze_content_structure(content),
                self._analyze_keywords(content),
                self._analyze_engagement_potential(content, target_audience),
                self._analyze_seo_metrics(content),
                self._analyze_content_quality(content),
                self._generate_content_insights(content),
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            analysis = {
                "sentiment": results[0] if not isinstance(results[0], Exception) else {},
                "readability": results[1] if not isinstance(results[1], Exception) else {},
                "structure": results[2] if not isinstance(results[2], Exception) else {},
                "keywords": results[3] if not isinstance(results[3], Exception) else {},
                "engagement": results[4] if not isinstance(results[4], Exception) else {},
                "seo": results[5] if not isinstance(results[5], Exception) else {},
                "quality": results[6] if not isinstance(results[6], Exception) else {},
                "insights": results[7] if not isinstance(results[7], Exception) else {},
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Calculate overall score
            analysis["overall_score"] = self._calculate_overall_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {"error": str(e)}
    
    async def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze content sentiment using multiple approaches."""
        try:
            # VADER sentiment analysis
            vader_scores = self.sentiment_analyzer.polarity_scores(content)
            
            # TextBlob sentiment analysis
            blob = TextBlob(content)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # spaCy sentiment analysis (if available)
            spacy_sentiment = 0.0
            if nlp:
                doc = nlp(content)
                spacy_sentiment = doc.sentiment
            
            # Determine overall sentiment
            sentiment_scores = [
                vader_scores['compound'],
                textblob_polarity,
                spacy_sentiment
            ]
            avg_sentiment = np.mean([s for s in sentiment_scores if s != 0])
            
            # Sentiment classification
            if avg_sentiment > 0.1:
                sentiment_label = "positive"
            elif avg_sentiment < -0.1:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return {
                "overall_sentiment": sentiment_label,
                "sentiment_score": avg_sentiment,
                "vader_scores": vader_scores,
                "textblob_polarity": textblob_polarity,
                "textblob_subjectivity": textblob_subjectivity,
                "spacy_sentiment": spacy_sentiment,
                "confidence": 0.85,  # Mock confidence score
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {}
    
    async def _analyze_readability(self, content: str) -> Dict[str, Any]:
        """Analyze content readability using multiple metrics."""
        try:
            # TextStat readability metrics
            flesch_reading_ease = textstat.flesch_reading_ease(content)
            flesch_kincaid_grade = textstat.flesch_kincaid_grade(content)
            gunning_fog = textstat.gunning_fog(content)
            smog_index = textstat.smog_index(content)
            automated_readability_index = textstat.automated_readability_index(content)
            coleman_liau_index = textstat.coleman_liau_index(content)
            linsear_write_formula = textstat.linsear_write_formula(content)
            dale_chall_readability_score = textstat.dale_chall_readability_score(content)
            
            # Readability library metrics
            readability = Readability(content)
            ari = readability.ari()
            cli = readability.cli()
            dale_chall = readability.dale_chall()
            flesch = readability.flesch()
            gunning_fog_readability = readability.gunning_fog()
            smog = readability.smog()
            
            # Calculate average readability score
            readability_scores = [
                flesch_reading_ease,
                ari.score if ari else 0,
                cli.score if cli else 0,
                dale_chall.score if dale_chall else 0,
                flesch.score if flesch else 0,
                gunning_fog_readability.score if gunning_fog_readability else 0,
                smog.score if smog else 0,
            ]
            
            avg_readability = np.mean([s for s in readability_scores if s > 0])
            
            # Readability level classification
            if avg_readability >= 80:
                level = "very_easy"
            elif avg_readability >= 60:
                level = "easy"
            elif avg_readability >= 40:
                level = "moderate"
            elif avg_readability >= 20:
                level = "difficult"
            else:
                level = "very_difficult"
            
            return {
                "overall_readability": avg_readability,
                "readability_level": level,
                "flesch_reading_ease": flesch_reading_ease,
                "flesch_kincaid_grade": flesch_kincaid_grade,
                "gunning_fog": gunning_fog,
                "smog_index": smog_index,
                "automated_readability_index": automated_readability_index,
                "coleman_liau_index": coleman_liau_index,
                "linsear_write_formula": linsear_write_formula,
                "dale_chall_readability_score": dale_chall_readability_score,
                "readability_library_scores": {
                    "ari": ari.score if ari else None,
                    "cli": cli.score if cli else None,
                    "dale_chall": dale_chall.score if dale_chall else None,
                    "flesch": flesch.score if flesch else None,
                    "gunning_fog": gunning_fog_readability.score if gunning_fog_readability else None,
                    "smog": smog.score if smog else None,
                }
            }
            
        except Exception as e:
            logger.error(f"Error in readability analysis: {e}")
            return {}
    
    async def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze content structure and formatting."""
        try:
            # Tokenization
            sentences = sent_tokenize(content)
            words = word_tokenize(content.lower())
            
            # Basic statistics
            word_count = len(words)
            sentence_count = len(sentences)
            paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
            
            # Average sentence length
            avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
            avg_words_per_sentence = avg_sentence_length
            
            # Paragraph analysis
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            avg_paragraph_length = np.mean([len(p.split()) for p in paragraphs]) if paragraphs else 0
            
            # Structure analysis
            has_bullet_points = bool(re.search(r'[•\-\*]\s', content))
            has_numbers = bool(re.search(r'\d+', content))
            has_questions = bool(re.search(r'\?', content))
            has_exclamations = bool(re.search(r'!', content))
            has_quotes = bool(re.search(r'["\']', content))
            has_hashtags = bool(re.search(r'#\w+', content))
            has_mentions = bool(re.search(r'@\w+', content))
            has_links = bool(re.search(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
            
            # Call-to-action analysis
            cta_patterns = [
                r'\b(comment|share|like|follow|connect|learn|discover|explore|try|visit|click|read|watch|listen)\b',
                r'\b(join|subscribe|sign up|register|download|get|find|search|contact|reach out|message)\b',
                r'\b(what do you think|share your thoughts|let me know|tell us|drop a comment|tag someone)\b'
            ]
            
            cta_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in cta_patterns)
            has_strong_cta = cta_count > 0
            
            return {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_sentence_length": avg_sentence_length,
                "avg_paragraph_length": avg_paragraph_length,
                "structure_elements": {
                    "has_bullet_points": has_bullet_points,
                    "has_numbers": has_numbers,
                    "has_questions": has_questions,
                    "has_exclamations": has_exclamations,
                    "has_quotes": has_quotes,
                    "has_hashtags": has_hashtags,
                    "has_mentions": has_mentions,
                    "has_links": has_links,
                },
                "call_to_action": {
                    "has_cta": has_strong_cta,
                    "cta_count": cta_count,
                    "cta_strength": "strong" if cta_count > 2 else "moderate" if cta_count > 0 else "weak"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in content structure analysis: {e}")
            return {}
    
    async def _analyze_keywords(self, content: str) -> Dict[str, Any]:
        """Analyze keywords and key phrases."""
        try:
            # Tokenization and preprocessing
            words = word_tokenize(content.lower())
            
            # Remove stop words and lemmatize
            filtered_words = [
                self.lemmatizer.lemmatize(word) 
                for word in words 
                if word.isalnum() and word not in self.stop_words and len(word) > 2
            ]
            
            # Word frequency analysis
            word_freq = Counter(filtered_words)
            top_keywords = word_freq.most_common(10)
            
            # Keyword density
            total_words = len(filtered_words)
            keyword_density = {
                word: (count / total_words) * 100 
                for word, count in top_keywords
            }
            
            # Key phrase extraction (bigrams and trigrams)
            bigrams = list(zip(filtered_words, filtered_words[1:]))
            trigrams = list(zip(filtered_words, filtered_words[1:], filtered_words[2:]))
            
            bigram_freq = Counter(bigrams)
            trigram_freq = Counter(trigrams)
            
            top_bigrams = bigram_freq.most_common(5)
            top_trigrams = trigram_freq.most_common(5)
            
            # Named entity recognition (if spaCy is available)
            entities = []
            if nlp:
                doc = nlp(content)
                entities = [
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char
                    }
                    for ent in doc.ents
                ]
            
            return {
                "top_keywords": top_keywords,
                "keyword_density": keyword_density,
                "top_bigrams": top_bigrams,
                "top_trigrams": top_trigrams,
                "entities": entities,
                "total_unique_words": len(set(filtered_words)),
                "lexical_diversity": len(set(filtered_words)) / total_words if total_words > 0 else 0,
            }
            
        except Exception as e:
            logger.error(f"Error in keyword analysis: {e}")
            return {}
    
    async def _analyze_engagement_potential(self, content: str, target_audience: str) -> Dict[str, Any]:
        """Analyze content engagement potential."""
        try:
            # Engagement factors scoring
            engagement_factors = {
                "emotional_appeal": 0,
                "storytelling": 0,
                "call_to_action": 0,
                "interactivity": 0,
                "relevance": 0,
                "uniqueness": 0,
                "clarity": 0,
                "actionability": 0,
            }
            
            # Emotional appeal analysis
            sentiment_scores = self.sentiment_analyzer.polarity_scores(content)
            emotional_intensity = abs(sentiment_scores['compound'])
            engagement_factors["emotional_appeal"] = min(emotional_intensity * 100, 100)
            
            # Storytelling analysis
            personal_pronouns = len(re.findall(r'\b(I|me|my|mine|we|us|our|ours)\b', content, re.IGNORECASE))
            storytelling_score = min((personal_pronouns / len(content.split())) * 1000, 100)
            engagement_factors["storytelling"] = storytelling_score
            
            # Call-to-action analysis
            cta_patterns = [
                r'\b(comment|share|like|follow|connect)\b',
                r'\b(what do you think|share your thoughts|let me know)\b',
                r'\b(tag someone|drop a comment|join the conversation)\b'
            ]
            cta_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in cta_patterns)
            engagement_factors["call_to_action"] = min(cta_count * 25, 100)
            
            # Interactivity analysis
            questions = len(re.findall(r'\?', content))
            engagement_factors["interactivity"] = min(questions * 20, 100)
            
            # Relevance analysis (mock - would use audience analysis)
            engagement_factors["relevance"] = 75  # Mock score
            
            # Uniqueness analysis
            common_phrases = [
                "in today's world", "as we all know", "it's important to",
                "moving forward", "at the end of the day", "to be honest"
            ]
            common_phrase_count = sum(content.lower().count(phrase) for phrase in common_phrases)
            uniqueness_score = max(100 - (common_phrase_count * 10), 0)
            engagement_factors["uniqueness"] = uniqueness_score
            
            # Clarity analysis
            readability_score = textstat.flesch_reading_ease(content)
            clarity_score = max(min(readability_score, 100), 0)
            engagement_factors["clarity"] = clarity_score
            
            # Actionability analysis
            action_words = len(re.findall(r'\b(do|make|take|get|find|use|try|learn|start|begin)\b', content, re.IGNORECASE))
            engagement_factors["actionability"] = min(action_words * 10, 100)
            
            # Calculate overall engagement score
            overall_engagement = np.mean(list(engagement_factors.values()))
            
            return {
                "overall_engagement_score": overall_engagement,
                "engagement_factors": engagement_factors,
                "engagement_level": self._classify_engagement_level(overall_engagement),
                "recommendations": self._generate_engagement_recommendations(engagement_factors),
            }
            
        except Exception as e:
            logger.error(f"Error in engagement analysis: {e}")
            return {}
    
    async def _analyze_seo_metrics(self, content: str) -> Dict[str, Any]:
        """Analyze SEO metrics and optimization."""
        try:
            # SEO factors
            seo_factors = {
                "keyword_optimization": 0,
                "content_length": 0,
                "readability": 0,
                "structure": 0,
                "internal_linking": 0,
                "external_linking": 0,
                "meta_optimization": 0,
            }
            
            # Content length analysis
            word_count = len(content.split())
            if 100 <= word_count <= 300:
                seo_factors["content_length"] = 100
            elif 50 <= word_count < 100 or 300 < word_count <= 500:
                seo_factors["content_length"] = 80
            elif word_count < 50 or word_count > 500:
                seo_factors["content_length"] = 60
            
            # Readability for SEO
            readability_score = textstat.flesch_reading_ease(content)
            seo_factors["readability"] = max(min(readability_score, 100), 0)
            
            # Structure analysis
            has_headings = bool(re.search(r'^#+\s', content, re.MULTILINE))
            has_bullets = bool(re.search(r'[•\-\*]\s', content))
            has_numbers = bool(re.search(r'\d+', content))
            
            structure_score = 0
            if has_headings: structure_score += 30
            if has_bullets: structure_score += 30
            if has_numbers: structure_score += 20
            if len(content.split('\n\n')) > 2: structure_score += 20
            
            seo_factors["structure"] = min(structure_score, 100)
            
            # Link analysis
            internal_links = len(re.findall(r'\[.*?\]\(.*?\)', content))
            external_links = len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content))
            
            seo_factors["internal_linking"] = min(internal_links * 25, 100)
            seo_factors["external_linking"] = min(external_links * 20, 100)
            
            # Keyword optimization (mock - would use target keywords)
            seo_factors["keyword_optimization"] = 75  # Mock score
            
            # Meta optimization (mock - would analyze meta tags)
            seo_factors["meta_optimization"] = 80  # Mock score
            
            # Calculate overall SEO score
            overall_seo = np.mean(list(seo_factors.values()))
            
            return {
                "overall_seo_score": overall_seo,
                "seo_factors": seo_factors,
                "seo_level": self._classify_seo_level(overall_seo),
                "recommendations": self._generate_seo_recommendations(seo_factors),
            }
            
        except Exception as e:
            logger.error(f"Error in SEO analysis: {e}")
            return {}
    
    async def _analyze_content_quality(self, content: str) -> Dict[str, Any]:
        """Analyze overall content quality."""
        try:
            # Quality factors
            quality_factors = {
                "originality": 0,
                "accuracy": 0,
                "completeness": 0,
                "coherence": 0,
                "grammar": 0,
                "style": 0,
                "value": 0,
            }
            
            # Originality analysis (mock - would use plagiarism detection)
            quality_factors["originality"] = 85  # Mock score
            
            # Accuracy analysis (mock - would use fact-checking)
            quality_factors["accuracy"] = 90  # Mock score
            
            # Completeness analysis
            has_intro = len(content.split('\n')[0]) > 20
            has_body = len(content.split('\n')) > 3
            has_conclusion = content.strip().endswith(('.', '!', '?'))
            
            completeness_score = 0
            if has_intro: completeness_score += 30
            if has_body: completeness_score += 40
            if has_conclusion: completeness_score += 30
            
            quality_factors["completeness"] = completeness_score
            
            # Coherence analysis
            sentences = sent_tokenize(content)
            if len(sentences) > 1:
                coherence_score = min(len(sentences) * 5, 100)
            else:
                coherence_score = 50
            
            quality_factors["coherence"] = coherence_score
            
            # Grammar analysis (mock - would use grammar checking)
            quality_factors["grammar"] = 88  # Mock score
            
            # Style analysis
            style_score = 0
            if textstat.flesch_reading_ease(content) > 60: style_score += 25
            if len(re.findall(r'[•\-\*]\s', content)) > 0: style_score += 25
            if len(re.findall(r'\?', content)) > 0: style_score += 25
            if len(re.findall(r'["\']', content)) > 0: style_score += 25
            
            quality_factors["style"] = style_score
            
            # Value analysis
            value_score = 0
            if len(content.split()) > 100: value_score += 30
            if len(re.findall(r'\b(insight|learn|discover|understand|know)\b', content, re.IGNORECASE)) > 0: value_score += 30
            if len(re.findall(r'\b(tip|advice|strategy|method|approach)\b', content, re.IGNORECASE)) > 0: value_score += 40
            
            quality_factors["value"] = value_score
            
            # Calculate overall quality score
            overall_quality = np.mean(list(quality_factors.values()))
            
            return {
                "overall_quality_score": overall_quality,
                "quality_factors": quality_factors,
                "quality_level": self._classify_quality_level(overall_quality),
                "recommendations": self._generate_quality_recommendations(quality_factors),
            }
            
        except Exception as e:
            logger.error(f"Error in content quality analysis: {e}")
            return {}
    
    async def _generate_content_insights(self, content: str) -> Dict[str, Any]:
        """Generate actionable content insights."""
        try:
            insights = {
                "strengths": [],
                "weaknesses": [],
                "opportunities": [],
                "recommendations": [],
                "trends": [],
            }
            
            # Analyze strengths
            word_count = len(content.split())
            if word_count > 150:
                insights["strengths"].append("Comprehensive content length")
            
            readability = textstat.flesch_reading_ease(content)
            if readability > 70:
                insights["strengths"].append("High readability score")
            
            if len(re.findall(r'\?', content)) > 0:
                insights["strengths"].append("Engaging questions included")
            
            if len(re.findall(r'[•\-\*]\s', content)) > 0:
                insights["strengths"].append("Well-structured with bullet points")
            
            # Analyze weaknesses
            if word_count < 100:
                insights["weaknesses"].append("Content may be too short")
            
            if readability < 50:
                insights["weaknesses"].append("Low readability score")
            
            if len(re.findall(r'\b(very|really|quite|extremely)\b', content, re.IGNORECASE)) > 3:
                insights["weaknesses"].append("Overuse of intensifiers")
            
            # Generate opportunities
            if len(re.findall(r'#\w+', content)) < 3:
                insights["opportunities"].append("Add more relevant hashtags")
            
            if len(re.findall(r'\b(comment|share|like|follow)\b', content, re.IGNORECASE)) == 0:
                insights["opportunities"].append("Include call-to-action")
            
            # Generate recommendations
            if readability < 60:
                insights["recommendations"].append("Simplify sentence structure for better readability")
            
            if len(content.split('\n')) < 3:
                insights["recommendations"].append("Break content into shorter paragraphs")
            
            if len(re.findall(r'\?', content)) == 0:
                insights["recommendations"].append("Add questions to encourage engagement")
            
            # Identify trends (mock - would use trend analysis)
            insights["trends"] = [
                "Video content is trending",
                "Personal stories perform well",
                "Industry insights are highly engaging"
            ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating content insights: {e}")
            return {}
    
    def _calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall content score."""
        try:
            scores = []
            
            # Sentiment score (normalized to 0-100)
            if "sentiment" in analysis and "sentiment_score" in analysis["sentiment"]:
                sentiment_score = (analysis["sentiment"]["sentiment_score"] + 1) * 50
                scores.append(sentiment_score)
            
            # Readability score
            if "readability" in analysis and "overall_readability" in analysis["readability"]:
                scores.append(analysis["readability"]["overall_readability"])
            
            # Engagement score
            if "engagement" in analysis and "overall_engagement_score" in analysis["engagement"]:
                scores.append(analysis["engagement"]["overall_engagement_score"])
            
            # SEO score
            if "seo" in analysis and "overall_seo_score" in analysis["seo"]:
                scores.append(analysis["seo"]["overall_seo_score"])
            
            # Quality score
            if "quality" in analysis and "overall_quality_score" in analysis["quality"]:
                scores.append(analysis["quality"]["overall_quality_score"])
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def _classify_engagement_level(self, score: float) -> str:
        """Classify engagement level based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "average"
        elif score >= 20:
            return "below_average"
        else:
            return "poor"
    
    def _classify_seo_level(self, score: float) -> str:
        """Classify SEO level based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "average"
        elif score >= 20:
            return "below_average"
        else:
            return "poor"
    
    def _classify_quality_level(self, score: float) -> str:
        """Classify quality level based on score."""
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"
        elif score >= 40:
            return "average"
        elif score >= 20:
            return "below_average"
        else:
            return "poor"
    
    def _generate_engagement_recommendations(self, factors: Dict[str, float]) -> List[str]:
        """Generate engagement improvement recommendations."""
        recommendations = []
        
        if factors.get("emotional_appeal", 0) < 50:
            recommendations.append("Increase emotional appeal with personal stories or examples")
        
        if factors.get("call_to_action", 0) < 50:
            recommendations.append("Add clear call-to-action to encourage engagement")
        
        if factors.get("interactivity", 0) < 50:
            recommendations.append("Include questions to encourage comments and discussion")
        
        if factors.get("clarity", 0) < 60:
            recommendations.append("Improve clarity by simplifying language and structure")
        
        return recommendations
    
    def _generate_seo_recommendations(self, factors: Dict[str, float]) -> List[str]:
        """Generate SEO improvement recommendations."""
        recommendations = []
        
        if factors.get("content_length", 0) < 70:
            recommendations.append("Increase content length for better SEO performance")
        
        if factors.get("structure", 0) < 70:
            recommendations.append("Improve content structure with headings and bullet points")
        
        if factors.get("readability", 0) < 60:
            recommendations.append("Improve readability for better user experience")
        
        return recommendations
    
    def _generate_quality_recommendations(self, factors: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        if factors.get("completeness", 0) < 70:
            recommendations.append("Ensure content has introduction, body, and conclusion")
        
        if factors.get("style", 0) < 70:
            recommendations.append("Improve writing style with better formatting and variety")
        
        if factors.get("value", 0) < 70:
            recommendations.append("Add more actionable insights and valuable information")
        
        return recommendations 