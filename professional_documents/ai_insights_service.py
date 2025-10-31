"""
AI Insights Service
==================

Advanced AI-powered insights and content analysis for documents.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from uuid import uuid4
import re
import json
from collections import Counter
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

logger = logging.getLogger(__name__)


class InsightType(str, Enum):
    """Insight type."""
    READABILITY = "readability"
    SENTIMENT = "sentiment"
    TOPICS = "topics"
    KEYWORDS = "keywords"
    ENTITIES = "entities"
    GRAMMAR = "grammar"
    STYLE = "style"
    STRUCTURE = "structure"
    QUALITY = "quality"
    RECOMMENDATIONS = "recommendations"


class QualityLevel(str, Enum):
    """Quality level."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class ContentInsight:
    """Content insight."""
    insight_id: str
    document_id: str
    insight_type: InsightType
    title: str
    description: str
    score: float
    quality_level: QualityLevel
    details: Dict[str, Any]
    recommendations: List[str]
    created_at: datetime
    confidence: float


@dataclass
class DocumentAnalysis:
    """Document analysis result."""
    analysis_id: str
    document_id: str
    content: str
    insights: List[ContentInsight]
    overall_score: float
    overall_quality: QualityLevel
    summary: str
    created_at: datetime
    processing_time: float


class AIInsightsService:
    """AI-powered insights service for documents."""
    
    def __init__(self):
        self.nlp = None
        self.sentiment_analyzer = None
        self.topic_modeler = None
        self.grammar_checker = None
        self.quality_analyzer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models."""
        try:
            # Initialize spaCy model
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            
            # Initialize sentiment analyzer
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
            
            # Initialize topic modeling
            try:
                self.topic_modeler = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli"
                )
            except Exception as e:
                logger.warning(f"Could not load topic modeler: {e}")
            
            # Initialize grammar checker
            try:
                self.grammar_checker = pipeline(
                    "text2text-generation",
                    model="grammarly/coedit-large"
                )
            except Exception as e:
                logger.warning(f"Could not load grammar checker: {e}")
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing AI models: {str(e)}")
    
    async def analyze_document(
        self,
        document_id: str,
        content: str,
        title: str = "",
        document_type: str = "general"
    ) -> DocumentAnalysis:
        """Analyze document and generate insights."""
        
        start_time = datetime.now()
        
        try:
            insights = []
            
            # Generate different types of insights
            if self.nlp:
                insights.extend(await self._analyze_readability(content))
                insights.extend(await self._analyze_entities(content))
                insights.extend(await self._analyze_structure(content))
            
            if self.sentiment_analyzer:
                insights.extend(await self._analyze_sentiment(content))
            
            if self.topic_modeler:
                insights.extend(await self._analyze_topics(content, document_type))
            
            insights.extend(await self._analyze_keywords(content))
            insights.extend(await self._analyze_style(content))
            insights.extend(await self._analyze_grammar(content))
            
            # Calculate overall quality
            overall_score, overall_quality = self._calculate_overall_quality(insights)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(insights, content)
            
            # Create summary
            summary = await self._generate_summary(insights, overall_quality)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            analysis = DocumentAnalysis(
                analysis_id=str(uuid4()),
                document_id=document_id,
                content=content,
                insights=insights,
                overall_score=overall_score,
                overall_quality=overall_quality,
                summary=summary,
                created_at=datetime.now(),
                processing_time=processing_time
            )
            
            logger.info(f"Document analysis completed for {document_id} in {processing_time:.2f}s")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            raise
    
    async def _analyze_readability(self, content: str) -> List[ContentInsight]:
        """Analyze document readability."""
        
        insights = []
        
        try:
            # Calculate readability scores
            flesch_score = flesch_reading_ease(content)
            fk_grade = flesch_kincaid_grade(content)
            
            # Determine quality level
            if flesch_score >= 80:
                quality = QualityLevel.EXCELLENT
            elif flesch_score >= 60:
                quality = QualityLevel.GOOD
            elif flesch_score >= 40:
                quality = QualityLevel.FAIR
            elif flesch_score >= 20:
                quality = QualityLevel.POOR
            else:
                quality = QualityLevel.CRITICAL
            
            # Generate recommendations
            recommendations = []
            if flesch_score < 60:
                recommendations.append("Consider using shorter sentences and simpler words")
            if fk_grade > 12:
                recommendations.append("Reduce sentence complexity for better readability")
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.READABILITY,
                title="Readability Analysis",
                description=f"Document readability score: {flesch_score:.1f} (Grade level: {fk_grade:.1f})",
                score=flesch_score,
                quality_level=quality,
                details={
                    "flesch_score": flesch_score,
                    "flesch_kincaid_grade": fk_grade,
                    "reading_level": self._get_reading_level(flesch_score)
                },
                recommendations=recommendations,
                created_at=datetime.now(),
                confidence=0.9
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing readability: {str(e)}")
        
        return insights
    
    async def _analyze_sentiment(self, content: str) -> List[ContentInsight]:
        """Analyze document sentiment."""
        
        insights = []
        
        try:
            if not self.sentiment_analyzer:
                return insights
            
            # Analyze sentiment
            results = self.sentiment_analyzer(content[:512])  # Limit length for API
            
            # Get primary sentiment
            primary_sentiment = max(results[0], key=lambda x: x['score'])
            
            # Determine quality level
            if primary_sentiment['label'] == 'POSITIVE':
                quality = QualityLevel.EXCELLENT
            elif primary_sentiment['label'] == 'NEUTRAL':
                quality = QualityLevel.GOOD
            else:
                quality = QualityLevel.POOR
            
            # Generate recommendations
            recommendations = []
            if primary_sentiment['label'] == 'NEGATIVE':
                recommendations.append("Consider using more positive language")
            if primary_sentiment['score'] < 0.7:
                recommendations.append("Sentiment is mixed - consider clarifying tone")
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.SENTIMENT,
                title="Sentiment Analysis",
                description=f"Document sentiment: {primary_sentiment['label']} (confidence: {primary_sentiment['score']:.2f})",
                score=primary_sentiment['score'] * 100,
                quality_level=quality,
                details={
                    "primary_sentiment": primary_sentiment['label'],
                    "confidence": primary_sentiment['score'],
                    "all_scores": results[0]
                },
                recommendations=recommendations,
                created_at=datetime.now(),
                confidence=primary_sentiment['score']
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
        
        return insights
    
    async def _analyze_topics(self, content: str, document_type: str) -> List[ContentInsight]:
        """Analyze document topics."""
        
        insights = []
        
        try:
            if not self.topic_modeler:
                return insights
            
            # Define topic candidates based on document type
            topic_candidates = self._get_topic_candidates(document_type)
            
            # Analyze topics
            results = self.topic_modeler(content[:512], topic_candidates)
            
            # Get top topics
            top_topics = results['labels'][:3]
            top_scores = results['scores'][:3]
            
            # Determine quality level
            if max(top_scores) > 0.8:
                quality = QualityLevel.EXCELLENT
            elif max(top_scores) > 0.6:
                quality = QualityLevel.GOOD
            elif max(top_scores) > 0.4:
                quality = QualityLevel.FAIR
            else:
                quality = QualityLevel.POOR
            
            # Generate recommendations
            recommendations = []
            if max(top_scores) < 0.6:
                recommendations.append("Consider focusing on a single main topic")
            if len([s for s in top_scores if s > 0.3]) > 3:
                recommendations.append("Document covers many topics - consider splitting")
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.TOPICS,
                title="Topic Analysis",
                description=f"Main topics: {', '.join(top_topics)}",
                score=max(top_scores) * 100,
                quality_level=quality,
                details={
                    "topics": list(zip(top_topics, top_scores)),
                    "topic_distribution": dict(zip(top_topics, top_scores))
                },
                recommendations=recommendations,
                created_at=datetime.now(),
                confidence=max(top_scores)
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing topics: {str(e)}")
        
        return insights
    
    async def _analyze_keywords(self, content: str) -> List[ContentInsight]:
        """Analyze document keywords."""
        
        insights = []
        
        try:
            if not self.nlp:
                return insights
            
            # Process content
            doc = self.nlp(content)
            
            # Extract keywords (nouns, adjectives, proper nouns)
            keywords = []
            for token in doc:
                if (token.pos_ in ['NOUN', 'ADJ', 'PROPN'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_.lower())
            
            # Count keyword frequency
            keyword_counts = Counter(keywords)
            top_keywords = keyword_counts.most_common(10)
            
            # Calculate keyword density
            total_words = len([token for token in doc if not token.is_stop and not token.is_punct])
            keyword_density = sum(count for _, count in top_keywords) / total_words if total_words > 0 else 0
            
            # Determine quality level
            if keyword_density > 0.3:
                quality = QualityLevel.EXCELLENT
            elif keyword_density > 0.2:
                quality = QualityLevel.GOOD
            elif keyword_density > 0.1:
                quality = QualityLevel.FAIR
            else:
                quality = QualityLevel.POOR
            
            # Generate recommendations
            recommendations = []
            if keyword_density < 0.1:
                recommendations.append("Add more relevant keywords to improve SEO")
            if len(top_keywords) < 5:
                recommendations.append("Document lacks keyword variety")
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.KEYWORDS,
                title="Keyword Analysis",
                description=f"Top keywords: {', '.join([kw for kw, _ in top_keywords[:5]])}",
                score=keyword_density * 100,
                quality_level=quality,
                details={
                    "top_keywords": top_keywords,
                    "keyword_density": keyword_density,
                    "total_keywords": len(keywords),
                    "unique_keywords": len(set(keywords))
                },
                recommendations=recommendations,
                created_at=datetime.now(),
                confidence=0.8
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing keywords: {str(e)}")
        
        return insights
    
    async def _analyze_entities(self, content: str) -> List[ContentInsight]:
        """Analyze named entities."""
        
        insights = []
        
        try:
            if not self.nlp:
                return insights
            
            # Process content
            doc = self.nlp(content)
            
            # Extract entities
            entities = {}
            for ent in doc.ents:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                entities[ent.label_].append(ent.text)
            
            # Count entity types
            entity_counts = {label: len(ents) for label, ents in entities.items()}
            total_entities = sum(entity_counts.values())
            
            # Determine quality level
            if total_entities > 10:
                quality = QualityLevel.EXCELLENT
            elif total_entities > 5:
                quality = QualityLevel.GOOD
            elif total_entities > 2:
                quality = QualityLevel.FAIR
            else:
                quality = QualityLevel.POOR
            
            # Generate recommendations
            recommendations = []
            if total_entities < 3:
                recommendations.append("Add more specific names, places, or organizations")
            if 'PERSON' not in entities:
                recommendations.append("Consider adding personal names or references")
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.ENTITIES,
                title="Entity Analysis",
                description=f"Found {total_entities} named entities across {len(entities)} categories",
                score=min(100, total_entities * 10),
                quality_level=quality,
                details={
                    "entity_counts": entity_counts,
                    "entities": entities,
                    "total_entities": total_entities
                },
                recommendations=recommendations,
                created_at=datetime.now(),
                confidence=0.9
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing entities: {str(e)}")
        
        return insights
    
    async def _analyze_structure(self, content: str) -> List[ContentInsight]:
        """Analyze document structure."""
        
        insights = []
        
        try:
            # Analyze structure elements
            paragraphs = content.split('\n\n')
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            # Count structure elements
            headings = len(re.findall(r'^#+\s', content, re.MULTILINE))
            lists = len(re.findall(r'^\s*[-*+]\s', content, re.MULTILINE))
            links = len(re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content))
            
            # Calculate structure score
            structure_score = 0
            if len(paragraphs) > 1:
                structure_score += 20
            if headings > 0:
                structure_score += 30
            if lists > 0:
                structure_score += 20
            if links > 0:
                structure_score += 10
            if len(sentences) > 5:
                structure_score += 20
            
            # Determine quality level
            if structure_score >= 80:
                quality = QualityLevel.EXCELLENT
            elif structure_score >= 60:
                quality = QualityLevel.GOOD
            elif structure_score >= 40:
                quality = QualityLevel.FAIR
            else:
                quality = QualityLevel.POOR
            
            # Generate recommendations
            recommendations = []
            if headings == 0:
                recommendations.append("Add headings to improve document structure")
            if len(paragraphs) < 3:
                recommendations.append("Break content into more paragraphs")
            if lists == 0 and len(words) > 100:
                recommendations.append("Consider using lists for better readability")
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.STRUCTURE,
                title="Structure Analysis",
                description=f"Document structure score: {structure_score}/100",
                score=structure_score,
                quality_level=quality,
                details={
                    "paragraphs": len(paragraphs),
                    "sentences": len(sentences),
                    "words": len(words),
                    "headings": headings,
                    "lists": lists,
                    "links": links,
                    "average_sentence_length": len(words) / len(sentences) if sentences else 0
                },
                recommendations=recommendations,
                created_at=datetime.now(),
                confidence=0.9
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing structure: {str(e)}")
        
        return insights
    
    async def _analyze_style(self, content: str) -> List[ContentInsight]:
        """Analyze writing style."""
        
        insights = []
        
        try:
            # Analyze style elements
            sentences = re.split(r'[.!?]+', content)
            words = content.split()
            
            # Calculate style metrics
            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            
            # Count style elements
            passive_voice = len(re.findall(r'\b(was|were|been|being)\s+\w+ed\b', content, re.IGNORECASE))
            contractions = len(re.findall(r"\b\w+'\w+\b", content))
            exclamations = len(re.findall(r'!', content))
            questions = len(re.findall(r'\?', content))
            
            # Calculate style score
            style_score = 50  # Base score
            
            # Adjust based on metrics
            if 10 <= avg_sentence_length <= 20:
                style_score += 20
            elif avg_sentence_length > 30:
                style_score -= 20
            
            if 4 <= avg_word_length <= 6:
                style_score += 15
            elif avg_word_length > 8:
                style_score -= 15
            
            if passive_voice / len(sentences) < 0.1:
                style_score += 15
            else:
                style_score -= 10
            
            # Determine quality level
            if style_score >= 80:
                quality = QualityLevel.EXCELLENT
            elif style_score >= 60:
                quality = QualityLevel.GOOD
            elif style_score >= 40:
                quality = QualityLevel.FAIR
            else:
                quality = QualityLevel.POOR
            
            # Generate recommendations
            recommendations = []
            if avg_sentence_length > 25:
                recommendations.append("Use shorter sentences for better readability")
            if avg_word_length > 7:
                recommendations.append("Consider using simpler words")
            if passive_voice / len(sentences) > 0.2:
                recommendations.append("Reduce passive voice usage")
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.STYLE,
                title="Style Analysis",
                description=f"Writing style score: {style_score}/100",
                score=style_score,
                quality_level=quality,
                details={
                    "average_sentence_length": avg_sentence_length,
                    "average_word_length": avg_word_length,
                    "passive_voice_count": passive_voice,
                    "contractions": contractions,
                    "exclamations": exclamations,
                    "questions": questions
                },
                recommendations=recommendations,
                created_at=datetime.now(),
                confidence=0.8
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing style: {str(e)}")
        
        return insights
    
    async def _analyze_grammar(self, content: str) -> List[ContentInsight]:
        """Analyze grammar and language quality."""
        
        insights = []
        
        try:
            # Basic grammar checks
            grammar_issues = []
            
            # Check for common issues
            if re.search(r'\b(its|it's)\b', content, re.IGNORECASE):
                grammar_issues.append("Check 'its' vs 'it's' usage")
            
            if re.search(r'\b(there|their|they're)\b', content, re.IGNORECASE):
                grammar_issues.append("Check 'there/their/they're' usage")
            
            if re.search(r'\b(your|you're)\b', content, re.IGNORECASE):
                grammar_issues.append("Check 'your' vs 'you're' usage")
            
            # Check for double spaces
            if '  ' in content:
                grammar_issues.append("Remove double spaces")
            
            # Check for missing periods
            sentences = re.split(r'[.!?]+', content)
            for sentence in sentences:
                if sentence.strip() and not sentence.strip().endswith(('.', '!', '?')):
                    grammar_issues.append("Add proper sentence endings")
                    break
            
            # Calculate grammar score
            grammar_score = max(0, 100 - len(grammar_issues) * 10)
            
            # Determine quality level
            if grammar_score >= 90:
                quality = QualityLevel.EXCELLENT
            elif grammar_score >= 70:
                quality = QualityLevel.GOOD
            elif grammar_score >= 50:
                quality = QualityLevel.FAIR
            else:
                quality = QualityLevel.POOR
            
            insight = ContentInsight(
                insight_id=str(uuid4()),
                document_id="",
                insight_type=InsightType.GRAMMAR,
                title="Grammar Analysis",
                description=f"Grammar score: {grammar_score}/100 ({len(grammar_issues)} issues found)",
                score=grammar_score,
                quality_level=quality,
                details={
                    "issues_found": grammar_issues,
                    "total_issues": len(grammar_issues)
                },
                recommendations=grammar_issues,
                created_at=datetime.now(),
                confidence=0.7
            )
            
            insights.append(insight)
            
        except Exception as e:
            logger.error(f"Error analyzing grammar: {str(e)}")
        
        return insights
    
    def _calculate_overall_quality(self, insights: List[ContentInsight]) -> Tuple[float, QualityLevel]:
        """Calculate overall document quality."""
        
        if not insights:
            return 0.0, QualityLevel.POOR
        
        # Weight different insight types
        weights = {
            InsightType.READABILITY: 0.25,
            InsightType.STRUCTURE: 0.20,
            InsightType.STYLE: 0.15,
            InsightType.GRAMMAR: 0.15,
            InsightType.KEYWORDS: 0.10,
            InsightType.SENTIMENT: 0.10,
            InsightType.TOPICS: 0.05
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for insight in insights:
            weight = weights.get(insight.insight_type, 0.05)
            weighted_score += insight.score * weight
            total_weight += weight
        
        overall_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Determine quality level
        if overall_score >= 80:
            quality = QualityLevel.EXCELLENT
        elif overall_score >= 60:
            quality = QualityLevel.GOOD
        elif overall_score >= 40:
            quality = QualityLevel.FAIR
        else:
            quality = QualityLevel.POOR
        
        return overall_score, quality
    
    async def _generate_recommendations(
        self,
        insights: List[ContentInsight],
        content: str
    ) -> List[str]:
        """Generate overall recommendations."""
        
        recommendations = []
        
        # Collect recommendations from insights
        for insight in insights:
            recommendations.extend(insight.recommendations)
        
        # Add content-specific recommendations
        if len(content) < 100:
            recommendations.append("Consider expanding content for better detail")
        elif len(content) > 5000:
            recommendations.append("Consider breaking into multiple sections")
        
        # Remove duplicates and limit
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:10]
    
    async def _generate_summary(self, insights: List[ContentInsight], quality: QualityLevel) -> str:
        """Generate analysis summary."""
        
        if not insights:
            return "No analysis available."
        
        # Count insights by quality level
        quality_counts = {}
        for insight in insights:
            quality_counts[insight.quality_level] = quality_counts.get(insight.quality_level, 0) + 1
        
        # Generate summary
        summary_parts = [
            f"Document analysis completed with {len(insights)} insights.",
            f"Overall quality: {quality.value.title()}",
        ]
        
        if quality_counts:
            quality_summary = ", ".join([
                f"{count} {level.value}" for level, count in quality_counts.items()
            ])
            summary_parts.append(f"Insight breakdown: {quality_summary}")
        
        return " ".join(summary_parts)
    
    def _get_reading_level(self, flesch_score: float) -> str:
        """Get reading level description."""
        
        if flesch_score >= 90:
            return "Very Easy"
        elif flesch_score >= 80:
            return "Easy"
        elif flesch_score >= 70:
            return "Fairly Easy"
        elif flesch_score >= 60:
            return "Standard"
        elif flesch_score >= 50:
            return "Fairly Difficult"
        elif flesch_score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"
    
    def _get_topic_candidates(self, document_type: str) -> List[str]:
        """Get topic candidates based on document type."""
        
        topic_map = {
            "business": [
                "strategy", "marketing", "finance", "operations", "management",
                "leadership", "growth", "innovation", "customer service"
            ],
            "technical": [
                "software", "development", "programming", "architecture",
                "database", "security", "performance", "testing"
            ],
            "academic": [
                "research", "analysis", "methodology", "findings", "conclusions",
                "literature", "hypothesis", "data", "results"
            ],
            "legal": [
                "contract", "agreement", "liability", "compliance", "regulation",
                "terms", "conditions", "rights", "obligations"
            ],
            "general": [
                "information", "details", "overview", "summary", "description",
                "guidelines", "instructions", "procedures", "requirements"
            ]
        }
        
        return topic_map.get(document_type, topic_map["general"])
    
    async def get_insight_trends(
        self,
        document_ids: List[str],
        insight_type: InsightType,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get insight trends across multiple documents."""
        
        # This would typically query a database
        # For now, return mock data
        
        return {
            "insight_type": insight_type.value,
            "period_days": days,
            "total_documents": len(document_ids),
            "average_score": 75.5,
            "trend": "improving",
            "top_issues": [
                "Readability needs improvement",
                "Grammar errors detected",
                "Structure could be better"
            ],
            "recommendations": [
                "Focus on sentence length",
                "Add more headings",
                "Improve keyword usage"
            ]
        }



























