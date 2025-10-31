"""
PDF Topic Extractor
===================

Extractor for identifying topics in PDF documents.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter
import re
from datetime import datetime
import fitz  # PyMuPDF
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Topic:
    """A topic extracted from the document."""
    topic: str
    category: str = "main"
    relevance_score: float = 0.0
    mentions: int = 0
    context: List[str] = field(default_factory=list)
    related_topics: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "topic": self.topic,
            "category": self.category,
            "relevance_score": self.relevance_score,
            "mentions": self.mentions,
            "context": self.context,
            "related_topics": self.related_topics
        }


class PDFTopicExtractor:
    """Extractor for topics in PDF documents."""
    
    def __init__(self, upload_dir: Optional[Path] = None):
        self.upload_dir = upload_dir or Path("./uploads/pdf_variantes")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Initialized PDF topic extractor")
    
    async def extract_topics(
        self,
        file_id: str,
        min_relevance: float = 0.5,
        max_topics: int = 50
    ) -> List[Topic]:
        """
        Extract topics from PDF.
        
        Args:
            file_id: File ID of the PDF
            min_relevance: Minimum relevance score
            max_topics: Maximum number of topics to extract
            
        Returns:
            List of extracted topics
        """
        file_path = self.upload_dir / f"{file_id}.pdf"
        
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_id}")
        
        # Extract text from PDF
        text = await self._extract_text(file_path)
        
        if not text:
            return []
    
        # Extract topics using various methods
        topics = []
        
        # Method 1: Key phrases and noun phrases
        topics.extend(await self._extract_key_phrases(text))
        
        # Method 2: Named entities
        topics.extend(await self._extract_named_entities(text))
        
        # Method 3: Technical terms
        topics.extend(await self._extract_technical_terms(text))
        
        # Method 4: Subject matter
        topics.extend(await self._extract_subject_matter(text))
        
        # Merge and deduplicate
        merged_topics = self._merge_topics(topics)
        
        # Calculate relevance scores
        for topic in merged_topics:
            topic.relevance_score = await self._calculate_relevance(topic, text)
        
        # Filter by relevance
        filtered_topics = [
            topic for topic in merged_topics
            if topic.relevance_score >= min_relevance
        ]
        
        # Sort by relevance and limit
        filtered_topics.sort(key=lambda x: x.relevance_score, reverse=True)
        filtered_topics = filtered_topics[:max_topics]
        
        # Find related topics
        for topic in filtered_topics:
            topic.related_topics = self._find_related_topics(topic, filtered_topics)
        
        logger.info(f"Extracted {len(filtered_topics)} topics from {file_id}")
        
        return filtered_topics
    
    async def extract_main_topic(self, file_id: str) -> Optional[str]:
        """Extract the main topic of the document."""
        topics = await self.extract_topics(file_id, min_relevance=0.7, max_topics=1)
        
        if topics:
            return topics[0].topic
        
        return None
    
    async def get_topic_categories(
        self,
        file_id: str
    ) -> Dict[str, List[Topic]]:
        """Get topics organized by category."""
        topics = await self.extract_topics(file_id)
        
        categories: Dict[str, List[Topic]] = {}
        
        for topic in topics:
            if topic.category not in categories:
                categories[topic.category] = []
            categories[topic.category].append(topic)
        
        return categories
    
    async def _extract_text(self, file_path: Path) -> str:
        """Extract text from PDF."""
        try:
            doc = fitz.open(str(file_path))
            text_parts = []
            
            for page in doc:
                text_parts.append(page.get_text())
            
            doc.close()
            return "\n\n".join(text_parts)
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    async def _extract_key_phrases(self, text: str) -> List[Topic]:
        """Extract key phrases from text."""
        topics = []
        
        # Extract noun phrases (simple approach)
        words = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', text)
        
        # Count frequency
        phrase_counts = Counter(words)
        
        # Create topics from most frequent phrases
        for phrase, count in phrase_counts.most_common(20):
            if len(phrase) > 3:  # Filter short words
                topics.append(Topic(
                    topic=phrase,
                    category="main",
                    mentions=count,
                    relevance_score=count / max(phrase_counts.values()) if phrase_counts else 0
                ))
        
        return topics
    
    async def _extract_named_entities(self, text: str) -> List[Topic]:
        """Extract named entities."""
        topics = []
        
        # Extract capitalized words/phrases (likely named entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        
        entity_counts = Counter(capitalized)
        
        for entity, count in entity_counts.most_common(15):
            if len(entity.split()) <= 3:  # Limit to 3-word entities
                topics.append(Topic(
                    topic=entity,
                    category="entity",
                    mentions=count
                ))
        
        return topics
    
    async def _extract_technical_terms(self, text: str) -> List[Topic]:
        """Extract technical terms."""
        topics = []
        
        # Look for patterns that indicate technical terms
        technical_patterns = [
            r'\b[A-Z]{2,}\b',  # Acronyms
            r'\b\w+ology\b',  # Fields of study
            r'\b\w+ing\s+\w+',  # Actions
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, text)
            term_counts = Counter(matches)
            
            for term, count in term_counts.most_common(10):
                topics.append(Topic(
                    topic=term,
                    category="technical",
                    mentions=count
                ))
        
        return topics
    
    async def _extract_subject_matter(self, text: str) -> List[Topic]:
        """Extract subject matter topics."""
        topics = []
        
        # Common subject indicators
        subject_keywords = [
            ("reports", "reporting"),
            ("analysis", "analyze", "analytical"),
            ("research", "study", "investigation"),
            ("project", "initiative", "program"),
            ("strategy", "strategic", "planning"),
            ("implementation", "execution"),
            ("assessment", "evaluation"),
            ("development", "progress", "growth")
        ]
        
        text_lower = text.lower()
        
        for subject, variants in subject_keywords:
            if any(variant in text_lower for variant in variants):
                topics.append(Topic(
                    topic=subject,
                    category="subject",
                    mentions=text_lower.count(subject)
                ))
        
        return topics
    
    def _merge_topics(self, topics: List[Topic]) -> List[Topic]:
        """Merge similar topics."""
        if not topics:
            return []
        
        # Group by topic name (case-insensitive)
        topic_groups: Dict[str, Topic] = {}
        
        for topic in topics:
            topic_key = topic.topic.lower()
            
            if topic_key in topic_groups:
                # Merge with existing topic
                existing = topic_groups[topic_key]
                existing.mentions += topic.mentions
                existing.relevance_score = max(existing.relevance_score, topic.relevance_score)
            else:
                topic_groups[topic_key] = topic
        
        return list(topic_groups.values())
    
    async def _calculate_relevance(
        self,
        topic: Topic,
        text: str
    ) -> float:
        """Calculate relevance score for a topic."""
        text_lower = text.lower()
        topic_lower = topic.topic.lower()
        
        # Count occurrences
        occurrences = text_lower.count(topic_lower)
        
        # Calculate frequency
        total_words = len(text.split())
        frequency = occurrences / total_words if total_words > 0 else 0
        
        # Base score on frequency and mentions
        score = min(1.0, (occurrences * 0.1 + topic.mentions * 0.2 + frequency * 100))
        
        return score
    
    def _find_related_topics(
        self,
        topic: Topic,
        all_topics: List[Topic]
    ) -> List[str]:
        """Find related topics."""
        related = []
        topic_words = set(topic.topic.lower().split())
        
        for other in all_topics:
            if other == topic:
                continue
            
            other_words = set(other.topic.lower().split())
            
            # Calculate word overlap
            overlap = len(topic_words & other_words)
            
            if overlap > 0:
                related.append(other.topic)
        
        return related[:5]  # Limit to 5 related topics
