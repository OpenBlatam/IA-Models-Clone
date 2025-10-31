"""
Advanced Document Analysis and Processing
========================================

This module provides advanced document analysis capabilities for the workflow chain engine,
including sentiment analysis, topic extraction, entity recognition, and comprehensive
quality assessment.

Features:
- Comprehensive document analysis
- Sentiment analysis
- Topic extraction
- Entity recognition
- Readability metrics
- Coherence analysis
- Structure analysis
- Performance tracking
"""

import re
import logging
from typing import Dict, List, Any, Tuple
from datetime import datetime
from collections import deque

from .workflow_chain_engine import DocumentProcessor, ModelContextLimits

logger = logging.getLogger(__name__)


class DocumentAnalyzer:
    """Advanced document analysis capabilities"""
    
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_extractor = TopicExtractor()
        self.entity_recognizer = EntityRecognizer()
    
    async def analyze_document_comprehensive(self, content: str) -> Dict[str, Any]:
        """
        Perform comprehensive document analysis
        
        Args:
            content: Document content to analyze
            
        Returns:
            Dict containing comprehensive analysis results
        """
        try:
            analysis = {
                'basic_stats': DocumentProcessor.get_document_statistics(content),
                'sentiment': await self.sentiment_analyzer.analyze(content),
                'topics': await self.topic_extractor.extract_topics(content),
                'entities': await self.entity_recognizer.extract_entities(content),
                'readability': self._calculate_readability_metrics(content),
                'coherence': self._analyze_coherence(content),
                'structure': self._analyze_structure(content)
            }
            
            # Calculate overall quality score
            analysis['overall_quality'] = self._calculate_overall_quality(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in comprehensive document analysis: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_readability_metrics(self, content: str) -> Dict[str, float]:
        """Calculate various readability metrics"""
        try:
            words = content.split()
            sentences = re.split(r'[.!?]+\s*', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return {'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0, 'gunning_fog': 0}
            
            # Flesch Reading Ease
            avg_sentence_length = len(words) / len(sentences)
            avg_syllables_per_word = sum(self._count_syllables(word) for word in words) / len(words)
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            
            # Flesch-Kincaid Grade Level
            fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
            
            # Gunning Fog Index
            complex_words = sum(1 for word in words if self._count_syllables(word) > 2)
            fog_index = 0.4 * (avg_sentence_length + (100 * complex_words / len(words)))
            
            return {
                'flesch_reading_ease': max(0, min(100, flesch_score)),
                'flesch_kincaid_grade': max(0, fk_grade),
                'gunning_fog': max(0, fog_index)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating readability metrics: {str(e)}")
            return {'flesch_reading_ease': 50, 'flesch_kincaid_grade': 10, 'gunning_fog': 12}
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximation)"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
            
        return max(1, syllable_count)
    
    def _analyze_coherence(self, content: str) -> Dict[str, Any]:
        """Analyze document coherence"""
        try:
            sentences = re.split(r'[.!?]+\s*', content)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return {'coherence_score': 0.5, 'transition_density': 0, 'repetition_score': 0.5}
            
            # Transition words analysis
            transition_words = [
                'however', 'therefore', 'moreover', 'furthermore', 'consequently',
                'additionally', 'in contrast', 'on the other hand', 'meanwhile',
                'subsequently', 'nevertheless', 'thus', 'hence', 'accordingly'
            ]
            
            transition_count = 0
            for sentence in sentences:
                for transition in transition_words:
                    if transition in sentence.lower():
                        transition_count += 1
            
            transition_density = transition_count / len(sentences)
            
            # Repetition analysis
            words = content.lower().split()
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Only consider meaningful words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Calculate repetition score (lower is better)
            total_words = len(words)
            unique_words = len(word_freq)
            repetition_score = 1 - (unique_words / total_words) if total_words > 0 else 0
            
            # Overall coherence score
            coherence_score = min(1.0, (transition_density * 0.6) + ((1 - repetition_score) * 0.4))
            
            return {
                'coherence_score': coherence_score,
                'transition_density': transition_density,
                'repetition_score': repetition_score,
                'unique_word_ratio': unique_words / total_words if total_words > 0 else 0
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing coherence: {str(e)}")
            return {'coherence_score': 0.5, 'transition_density': 0, 'repetition_score': 0.5}
    
    def _analyze_structure(self, content: str) -> Dict[str, Any]:
        """Analyze document structure"""
        try:
            # Paragraph analysis
            paragraphs = content.split('\n\n')
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
            
            # Heading analysis (simple heuristic)
            lines = content.split('\n')
            potential_headings = []
            for line in lines:
                line = line.strip()
                if line and (line.isupper() or line.endswith(':') or len(line.split()) <= 8):
                    potential_headings.append(line)
            
            # List analysis
            list_items = content.count('\n-') + content.count('\n*') + content.count('\n1.')
            
            # Quote analysis
            quotes = content.count('"') + content.count("'")
            
            return {
                'paragraph_count': len(paragraphs),
                'avg_paragraph_length': sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0,
                'potential_headings': len(potential_headings),
                'list_items': list_items,
                'quotes': quotes,
                'structure_score': self._calculate_structure_score(len(paragraphs), len(potential_headings), list_items)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing structure: {str(e)}")
            return {'paragraph_count': 1, 'avg_paragraph_length': 0, 'potential_headings': 0, 'list_items': 0, 'quotes': 0, 'structure_score': 0.5}
    
    def _calculate_structure_score(self, paragraphs: int, headings: int, lists: int) -> float:
        """Calculate structure quality score"""
        score = 0.0
        
        # Paragraph score (optimal: 3-8 paragraphs)
        if 3 <= paragraphs <= 8:
            score += 0.4
        elif 2 <= paragraphs <= 12:
            score += 0.3
        else:
            score += 0.1
        
        # Heading score (optimal: 1-3 headings)
        if 1 <= headings <= 3:
            score += 0.3
        elif headings <= 5:
            score += 0.2
        else:
            score += 0.1
        
        # List score (some lists are good)
        if 1 <= lists <= 5:
            score += 0.3
        elif lists == 0:
            score += 0.2
        else:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_overall_quality(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall document quality score"""
        try:
            scores = []
            
            # Basic stats quality
            if 'basic_stats' in analysis:
                word_count = analysis['basic_stats'].get('word_count', 0)
                if 200 <= word_count <= 2000:
                    scores.append(0.9)
                elif 100 <= word_count <= 3000:
                    scores.append(0.7)
                else:
                    scores.append(0.4)
            
            # Readability quality
            if 'readability' in analysis:
                flesch_score = analysis['readability'].get('flesch_reading_ease', 50)
                if 60 <= flesch_score <= 80:
                    scores.append(0.9)
                elif 40 <= flesch_score <= 90:
                    scores.append(0.7)
                else:
                    scores.append(0.5)
            
            # Coherence quality
            if 'coherence' in analysis:
                coherence_score = analysis['coherence'].get('coherence_score', 0.5)
                scores.append(coherence_score)
            
            # Structure quality
            if 'structure' in analysis:
                structure_score = analysis['structure'].get('structure_score', 0.5)
                scores.append(structure_score)
            
            return sum(scores) / len(scores) if scores else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating overall quality: {str(e)}")
            return 0.5


class SentimentAnalyzer:
    """Simple sentiment analysis (can be enhanced with actual NLP libraries)"""
    
    async def analyze(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of content"""
        try:
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'perfect', 'outstanding']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing', 'poor', 'fail', 'problem']
            
            words = content.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}
            
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
            
            if sentiment_score > 0.2:
                sentiment = 'positive'
            elif sentiment_score < -0.2:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            confidence = min(1.0, total_sentiment_words / 10)  # Simple confidence based on word count
            
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'confidence': confidence,
                'positive_words': positive_count,
                'negative_words': negative_count
            }
            
        except Exception as e:
            logger.warning(f"Error in sentiment analysis: {str(e)}")
            return {'sentiment': 'neutral', 'score': 0.0, 'confidence': 0.0}


class TopicExtractor:
    """Simple topic extraction (can be enhanced with actual NLP libraries)"""
    
    async def extract_topics(self, content: str) -> List[Dict[str, Any]]:
        """Extract topics from content"""
        try:
            # Simple keyword frequency analysis
            words = re.findall(r'\b\w+\b', content.lower())
            word_freq = {}
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
            
            for word in words:
                if len(word) > 3 and word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get top topics
            topics = []
            for word, freq in sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]:
                topics.append({
                    'topic': word,
                    'frequency': freq,
                    'relevance': freq / len(words) if words else 0
                })
            
            return topics
            
        except Exception as e:
            logger.warning(f"Error extracting topics: {str(e)}")
            return []


class EntityRecognizer:
    """Simple entity recognition (can be enhanced with actual NLP libraries)"""
    
    async def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract entities from content"""
        try:
            entities = {
                'people': [],
                'organizations': [],
                'locations': [],
                'dates': [],
                'numbers': []
            }
            
            # Simple regex-based entity extraction
            # People (capitalized words that might be names)
            people_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            entities['people'] = re.findall(people_pattern, content)
            
            # Organizations (words with common org suffixes)
            org_pattern = r'\b[A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd|Company|Corporation|Organization|Institute|University|College)\b'
            entities['organizations'] = re.findall(org_pattern, content)
            
            # Locations (common location indicators)
            location_pattern = r'\b[A-Z][a-z]+ (City|State|Country|Street|Avenue|Road|Boulevard)\b'
            entities['locations'] = re.findall(location_pattern, content)
            
            # Dates (various date formats)
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2},? \d{4}\b'
            entities['dates'] = re.findall(date_pattern, content)
            
            # Numbers (monetary amounts, percentages, etc.)
            number_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:\.\d+)?%|\b\d+(?:,\d{3})*\b'
            entities['numbers'] = re.findall(number_pattern, content)
            
            return entities
            
        except Exception as e:
            logger.warning(f"Error extracting entities: {str(e)}")
            return {'people': [], 'organizations': [], 'locations': [], 'dates': [], 'numbers': []}


class PerformanceMetrics:
    """Track and analyze performance metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_session = {
            'start_time': datetime.now(),
            'documents_processed': 0,
            'total_tokens_used': 0,
            'total_generation_time': 0.0,
            'quality_scores': []
        }
    
    def record_generation(self, tokens_used: int, generation_time: float, quality_score: float):
        """Record a document generation event"""
        self.current_session['documents_processed'] += 1
        self.current_session['total_tokens_used'] += tokens_used
        self.current_session['total_generation_time'] += generation_time
        self.current_session['quality_scores'].append(quality_score)
        
        # Add to history
        self.metrics_history.append({
            'timestamp': datetime.now(),
            'tokens_used': tokens_used,
            'generation_time': generation_time,
            'quality_score': quality_score
        })
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary"""
        session = self.current_session
        avg_quality = sum(session['quality_scores']) / len(session['quality_scores']) if session['quality_scores'] else 0
        
        return {
            'session_duration': (datetime.now() - session['start_time']).total_seconds(),
            'documents_processed': session['documents_processed'],
            'total_tokens_used': session['total_tokens_used'],
            'total_generation_time': session['total_generation_time'],
            'average_quality': avg_quality,
            'tokens_per_minute': session['total_tokens_used'] / max((datetime.now() - session['start_time']).total_seconds() / 60, 0.1),
            'documents_per_hour': session['documents_processed'] / max((datetime.now() - session['start_time']).total_seconds() / 3600, 0.1)
        }
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends over time"""
        if len(self.metrics_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Analyze trends over last 10 generations
        recent_metrics = self.metrics_history[-10:]
        
        quality_trend = self._calculate_trend([m['quality_score'] for m in recent_metrics])
        speed_trend = self._calculate_trend([m['generation_time'] for m in recent_metrics])
        efficiency_trend = self._calculate_trend([m['tokens_used'] / max(m['generation_time'], 0.1) for m in recent_metrics])
        
        return {
            'quality_trend': quality_trend,
            'speed_trend': speed_trend,
            'efficiency_trend': efficiency_trend,
            'recent_avg_quality': sum(m['quality_score'] for m in recent_metrics) / len(recent_metrics),
            'recent_avg_speed': sum(m['generation_time'] for m in recent_metrics) / len(recent_metrics)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return 'stable'
        
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return 'improving'
        elif second_avg < first_avg * 0.9:
            return 'declining'
        else:
            return 'stable'


class AdvancedWorkflowChainEngine:
    """Enhanced workflow chain engine with advanced document analysis"""
    
    def __init__(self, base_engine):
        self.base_engine = base_engine
        self.document_analyzer = DocumentAnalyzer()
        self.performance_metrics = PerformanceMetrics()
    
    async def create_workflow_chain_with_analysis(
        self, 
        name: str, 
        description: str,
        initial_prompt: str,
        settings: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        enable_analysis: bool = True
    ):
        """
        Create a workflow chain with comprehensive document analysis
        
        Args:
            name: Name of the workflow chain
            description: Description of the workflow purpose
            initial_prompt: Initial prompt to start the chain
            settings: Optional settings for the workflow
            user_id: Optional user ID for multi-user support
            enable_analysis: Whether to enable comprehensive analysis
            
        Returns:
            WorkflowChain: The created workflow chain with analysis
        """
        try:
            # Create the basic workflow chain
            chain = await self.base_engine.create_workflow_chain(
                name=name,
                description=description,
                initial_prompt=initial_prompt,
                settings=settings,
                user_id=user_id
            )
            
            # Perform comprehensive analysis if enabled
            if enable_analysis and chain.root_node_id:
                root_node = chain.nodes[chain.root_node_id]
                analysis = await self.document_analyzer.analyze_document_comprehensive(root_node.content)
                root_node.metadata['comprehensive_analysis'] = analysis
                
                # Update quality score with comprehensive analysis
                if 'overall_quality' in analysis:
                    root_node.metadata['quality_score'] = analysis['overall_quality']
                
                # Record performance metrics
                self.performance_metrics.record_generation(
                    root_node.metadata.get('tokens_used', 0),
                    root_node.metadata.get('generation_time', 0.0),
                    root_node.metadata.get('quality_score', 0.5)
                )
                
                logger.info(f"Performed comprehensive analysis on root document: {root_node.id}")
            
            return chain
            
        except Exception as e:
            logger.error(f"Error creating workflow chain with analysis: {str(e)}")
            raise
    
    async def continue_workflow_chain_with_analysis(
        self, 
        chain_id: str, 
        continuation_prompt: Optional[str] = None,
        enable_analysis: bool = True
    ):
        """
        Continue a workflow chain with comprehensive analysis
        
        Args:
            chain_id: ID of the workflow chain to continue
            continuation_prompt: Optional custom prompt
            enable_analysis: Whether to enable comprehensive analysis
            
        Returns:
            DocumentNode: The newly generated document node with analysis
        """
        try:
            # Continue the basic workflow chain
            new_node = await self.base_engine.continue_workflow_chain(chain_id, continuation_prompt)
            
            # Perform comprehensive analysis if enabled
            if enable_analysis:
                analysis = await self.document_analyzer.analyze_document_comprehensive(new_node.content)
                new_node.metadata['comprehensive_analysis'] = analysis
                
                # Update quality score with comprehensive analysis
                if 'overall_quality' in analysis:
                    new_node.metadata['quality_score'] = analysis['overall_quality']
                
                # Record performance metrics
                self.performance_metrics.record_generation(
                    new_node.metadata.get('tokens_used', 0),
                    new_node.metadata.get('generation_time', 0.0),
                    new_node.metadata.get('quality_score', 0.5)
                )
                
                logger.info(f"Performed comprehensive analysis on new document: {new_node.id}")
            
            return new_node
            
        except Exception as e:
            logger.error(f"Error continuing workflow chain with analysis: {str(e)}")
            raise
    
    def get_chain_analytics(self, chain_id: str) -> Dict[str, Any]:
        """
        Get comprehensive analytics for a workflow chain
        
        Args:
            chain_id: ID of the workflow chain
            
        Returns:
            Dict containing comprehensive analytics
        """
        try:
            chain = self.base_engine.get_workflow_chain(chain_id)
            if not chain:
                return {'error': 'Chain not found'}
            
            nodes = list(chain.nodes.values())
            
            if not nodes:
                return {'error': 'No documents in chain'}
            
            # Calculate chain-level metrics
            total_words = sum(node.metadata.get('word_count', 0) for node in nodes)
            total_tokens = sum(node.metadata.get('token_count', 0) for node in nodes)
            avg_quality = sum(node.metadata.get('quality_score', 0) for node in nodes) / len(nodes)
            
            # Analyze quality trends
            quality_scores = [node.metadata.get('quality_score', 0) for node in nodes]
            quality_trend = self._calculate_trend(quality_scores)
            
            # Analyze content evolution
            content_evolution = self._analyze_content_evolution(nodes)
            
            return {
                'chain_id': chain_id,
                'total_documents': len(nodes),
                'total_words': total_words,
                'total_tokens': total_tokens,
                'average_quality': avg_quality,
                'quality_trend': quality_trend,
                'content_evolution': content_evolution,
                'estimated_reading_time': total_words / 200,  # 200 words per minute
                'estimated_pages': total_words // 375,  # 375 words per page
                'generation_efficiency': self._calculate_generation_efficiency(nodes)
            }
            
        except Exception as e:
            logger.error(f"Error getting chain analytics: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return 'stable'
        
        # Simple trend calculation
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        if second_avg > first_avg * 1.1:
            return 'improving'
        elif second_avg < first_avg * 0.9:
            return 'declining'
        else:
            return 'stable'
    
    def _analyze_content_evolution(self, nodes) -> Dict[str, Any]:
        """Analyze how content evolves through the chain"""
        try:
            if len(nodes) < 2:
                return {'evolution_type': 'single_document', 'complexity_change': 0}
            
            # Analyze word count evolution
            word_counts = [node.metadata.get('word_count', 0) for node in nodes]
            word_count_trend = self._calculate_trend(word_counts)
            
            # Analyze quality evolution
            quality_scores = [node.metadata.get('quality_score', 0) for node in nodes]
            quality_trend = self._calculate_trend(quality_scores)
            
            # Analyze topic consistency
            topic_consistency = self._analyze_topic_consistency(nodes)
            
            return {
                'evolution_type': self._determine_evolution_type(word_count_trend, quality_trend),
                'word_count_trend': word_count_trend,
                'quality_trend': quality_trend,
                'topic_consistency': topic_consistency,
                'complexity_change': self._calculate_complexity_change(nodes)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing content evolution: {str(e)}")
            return {'evolution_type': 'unknown', 'complexity_change': 0}
    
    def _analyze_topic_consistency(self, nodes) -> float:
        """Analyze topic consistency across documents"""
        try:
            if len(nodes) < 2:
                return 1.0
            
            # Extract keywords from each document
            all_keywords = []
            for node in nodes:
                if 'comprehensive_analysis' in node.metadata:
                    topics = node.metadata['comprehensive_analysis'].get('topics', [])
                    keywords = [topic['topic'] for topic in topics[:5]]  # Top 5 topics
                    all_keywords.append(set(keywords))
            
            # Calculate overlap between consecutive documents
            overlaps = []
            for i in range(len(all_keywords) - 1):
                overlap = len(all_keywords[i].intersection(all_keywords[i + 1]))
                total_unique = len(all_keywords[i].union(all_keywords[i + 1]))
                if total_unique > 0:
                    overlaps.append(overlap / total_unique)
            
            return sum(overlaps) / len(overlaps) if overlaps else 0.0
            
        except Exception as e:
            logger.warning(f"Error analyzing topic consistency: {str(e)}")
            return 0.5
    
    def _determine_evolution_type(self, word_trend: str, quality_trend: str) -> str:
        """Determine the type of content evolution"""
        if word_trend == 'improving' and quality_trend == 'improving':
            return 'progressive_enhancement'
        elif word_trend == 'declining' and quality_trend == 'declining':
            return 'progressive_degradation'
        elif word_trend == 'stable' and quality_trend == 'stable':
            return 'consistent_quality'
        elif word_trend == 'improving' and quality_trend == 'stable':
            return 'expanding_content'
        elif word_trend == 'stable' and quality_trend == 'improving':
            return 'refining_content'
        else:
            return 'mixed_evolution'
    
    def _calculate_complexity_change(self, nodes) -> float:
        """Calculate how complexity changes through the chain"""
        try:
            if len(nodes) < 2:
                return 0.0
            
            complexities = []
            for node in nodes:
                if 'comprehensive_analysis' in node.metadata:
                    analysis = node.metadata['comprehensive_analysis']
                    readability = analysis.get('readability', {})
                    flesch_score = readability.get('flesch_reading_ease', 50)
                    # Lower Flesch score = higher complexity
                    complexity = 100 - flesch_score
                    complexities.append(complexity)
                else:
                    # Fallback: use word count as complexity proxy
                    word_count = node.metadata.get('word_count', 0)
                    complexities.append(min(100, word_count / 20))  # Normalize to 0-100
            
            # Calculate change from first to last
            if len(complexities) >= 2:
                return complexities[-1] - complexities[0]
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating complexity change: {str(e)}")
            return 0.0
    
    def _calculate_generation_efficiency(self, nodes) -> Dict[str, float]:
        """Calculate generation efficiency metrics"""
        try:
            if not nodes:
                return {'avg_generation_time': 0, 'tokens_per_second': 0, 'quality_per_token': 0}
            
            total_generation_time = sum(node.metadata.get('generation_time', 0) for node in nodes)
            total_tokens = sum(node.metadata.get('tokens_used', 0) for node in nodes)
            total_quality = sum(node.metadata.get('quality_score', 0) for node in nodes)
            
            avg_generation_time = total_generation_time / len(nodes)
            tokens_per_second = total_tokens / max(total_generation_time, 0.1)  # Avoid division by zero
            quality_per_token = total_quality / max(total_tokens, 1)  # Avoid division by zero
            
            return {
                'avg_generation_time': avg_generation_time,
                'tokens_per_second': tokens_per_second,
                'quality_per_token': quality_per_token,
                'efficiency_score': min(1.0, (tokens_per_second / 100) * (quality_per_token * 1000))  # Normalized efficiency
            }
            
        except Exception as e:
            logger.warning(f"Error calculating generation efficiency: {str(e)}")
            return {'avg_generation_time': 0, 'tokens_per_second': 0, 'quality_per_token': 0}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'session_summary': self.performance_metrics.get_session_summary(),
            'performance_trends': self.performance_metrics.get_performance_trends()
        }






