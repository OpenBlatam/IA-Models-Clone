"""
Script Processor Service
Processes and segments scripts for video generation
"""

import re
from typing import List, Dict, Any, Optional
from ..core.models import VideoScript
import logging

logger = logging.getLogger(__name__)


class ScriptProcessor:
    """Processes scripts into segments for video generation"""

    def __init__(self):
        self.min_segment_length = 3  # Minimum words per segment
        self.max_segment_length = 20  # Maximum words per segment
        self.pause_markers = ['.', '!', '?', ';', ',']

    def process_script(self, script: VideoScript) -> List[Dict[str, Any]]:
        """
        Process script text into segments
        
        Args:
            script: VideoScript object with text content
            
        Returns:
            List of segment dictionaries with text, timing, and metadata
        """
        if script.segments:
            logger.info("Using provided segments from script")
            return script.segments

        text = script.text
        language = script.language

        # Clean and normalize text
        text = self._clean_text(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text, language)
        
        # Group sentences into segments
        segments = self._create_segments(sentences)
        
        # Add timing estimates
        segments = self._add_timing_estimates(segments)
        
        # Add metadata
        segments = self._add_metadata(segments, language)
        
        logger.info(f"Processed script into {len(segments)} segments")
        return segments

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters that might interfere
        text = text.strip()
        return text

    def _split_into_sentences(self, text: str, language: str) -> List[str]:
        """Split text into sentences based on language"""
        # Basic sentence splitting
        # For Spanish and English, use common punctuation
        sentences = re.split(r'[.!?]+\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _create_segments(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Group sentences into video segments"""
        segments = []
        current_segment = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            
            # If adding this sentence would exceed max length, start new segment
            if current_word_count + word_count > self.max_segment_length and current_segment:
                segments.append({
                    "text": " ".join(current_segment),
                    "word_count": current_word_count,
                    "index": len(segments),
                })
                current_segment = [sentence]
                current_word_count = word_count
            else:
                current_segment.append(sentence)
                current_word_count += word_count

        # Add remaining segment
        if current_segment:
            segments.append({
                "text": " ".join(current_segment),
                "word_count": current_word_count,
                "index": len(segments),
            })

        return segments

    def _add_timing_estimates(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add timing estimates to segments"""
        # Average reading speed: ~150 words per minute
        words_per_second = 150 / 60  # ~2.5 words per second
        
        start_time = 0.0
        for segment in segments:
            word_count = segment.get("word_count", len(segment["text"].split()))
            duration = word_count / words_per_second
            # Add pause between segments
            duration += 0.5
            
            segment["start_time"] = start_time
            segment["duration"] = duration
            segment["end_time"] = start_time + duration
            
            start_time += duration

        return segments

    def _add_metadata(self, segments: List[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
        """Add metadata to segments"""
        for segment in segments:
            segment["language"] = language
            segment["char_count"] = len(segment["text"])
            # Generate keywords for image generation
            segment["keywords"] = self._extract_keywords(segment["text"])
        
        return segments

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from text for image generation"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = text.lower().split()
        # Remove common stop words
        stop_words = {
            'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas',
            'de', 'del', 'en', 'a', 'al', 'con', 'por', 'para',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were',
        }
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        # Return most frequent or important keywords
        return keywords[:max_keywords]

    def estimate_total_duration(self, segments: List[Dict[str, Any]]) -> float:
        """Estimate total video duration from segments"""
        if not segments:
            return 0.0
        last_segment = segments[-1]
        return last_segment.get("end_time", 0.0)

