"""
Search Service for Social Video Transcriber AI
Provides semantic search within transcriptions
"""

import json
import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from ..config.settings import get_settings
from .openrouter_client import get_openrouter_client

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result"""
    job_id: UUID
    video_title: str
    relevance_score: float
    matched_text: str
    context: str
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    highlight_positions: List[tuple] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": str(self.job_id),
            "video_title": self.video_title,
            "relevance_score": self.relevance_score,
            "matched_text": self.matched_text,
            "context": self.context,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
        }


@dataclass 
class TranscriptionIndex:
    """Index entry for a transcription"""
    job_id: UUID
    video_title: str
    video_author: str
    full_text: str
    segments: List[Dict[str, Any]]
    keywords: List[str] = field(default_factory=list)
    indexed_at: datetime = field(default_factory=datetime.utcnow)


class SearchService:
    """Service for searching within transcriptions"""
    
    SEMANTIC_SEARCH_PROMPT = """Analiza si el siguiente texto es relevante para la consulta de búsqueda.

CONSULTA: {query}

TEXTO:
{text}

Responde en JSON:
{{
    "is_relevant": true/false,
    "relevance_score": 0.0-1.0,
    "matched_segments": ["fragmento relevante 1", "fragmento relevante 2"],
    "explanation": "breve explicación de por qué es o no relevante"
}}"""

    EXTRACT_HIGHLIGHTS_PROMPT = """Del siguiente texto, extrae los fragmentos más relevantes para la consulta.

CONSULTA: {query}

TEXTO:
{text}

Devuelve los 3-5 fragmentos más relevantes en JSON:
{{
    "highlights": [
        {{"text": "fragmento relevante", "reason": "por qué es relevante"}}
    ]
}}"""

    def __init__(self):
        self.settings = get_settings()
        self.client = get_openrouter_client()
        self._index: Dict[UUID, TranscriptionIndex] = {}
    
    def index_transcription(
        self,
        job_id: UUID,
        video_title: str,
        video_author: str,
        full_text: str,
        segments: List[Dict[str, Any]],
        keywords: Optional[List[str]] = None,
    ):
        """
        Index a transcription for searching
        
        Args:
            job_id: Job identifier
            video_title: Video title
            video_author: Video author
            full_text: Full transcription text
            segments: List of segments with timestamps
            keywords: Optional pre-extracted keywords
        """
        self._index[job_id] = TranscriptionIndex(
            job_id=job_id,
            video_title=video_title,
            video_author=video_author,
            full_text=full_text,
            segments=segments,
            keywords=keywords or [],
        )
        
        logger.debug(f"Indexed transcription: {job_id}")
    
    def remove_from_index(self, job_id: UUID):
        """Remove a transcription from the index"""
        if job_id in self._index:
            del self._index[job_id]
            logger.debug(f"Removed from index: {job_id}")
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        job_ids: Optional[List[UUID]] = None,
        use_semantic: bool = True,
    ) -> List[SearchResult]:
        """
        Search across indexed transcriptions
        
        Args:
            query: Search query
            limit: Maximum results
            job_ids: Optional list of job IDs to search within
            use_semantic: Use AI-powered semantic search
            
        Returns:
            List of SearchResult
        """
        if not query or len(query.strip()) < 2:
            return []
        
        results = []
        
        search_scope = self._index.values()
        if job_ids:
            search_scope = [
                idx for idx in self._index.values()
                if idx.job_id in job_ids
            ]
        
        for index_entry in search_scope:
            keyword_results = self._keyword_search(query, index_entry)
            results.extend(keyword_results)
        
        if use_semantic and len(results) < limit:
            for index_entry in search_scope:
                if any(r.job_id == index_entry.job_id for r in results):
                    continue
                
                semantic_result = await self._semantic_search(query, index_entry)
                if semantic_result:
                    results.append(semantic_result)
        
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:limit]
    
    def _keyword_search(
        self,
        query: str,
        index_entry: TranscriptionIndex,
    ) -> List[SearchResult]:
        """Perform keyword-based search"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        if query_lower in index_entry.full_text.lower():
            positions = []
            text_lower = index_entry.full_text.lower()
            start = 0
            
            while True:
                pos = text_lower.find(query_lower, start)
                if pos == -1:
                    break
                positions.append((pos, pos + len(query_lower)))
                start = pos + 1
            
            context_start = max(0, positions[0][0] - 100)
            context_end = min(len(index_entry.full_text), positions[0][1] + 100)
            context = index_entry.full_text[context_start:context_end]
            
            segment = self._find_segment_for_position(positions[0][0], index_entry.segments)
            
            results.append(SearchResult(
                job_id=index_entry.job_id,
                video_title=index_entry.video_title,
                relevance_score=0.95,
                matched_text=query,
                context=f"...{context}...",
                timestamp_start=segment.get("start_time") if segment else None,
                timestamp_end=segment.get("end_time") if segment else None,
                highlight_positions=positions[:5],
            ))
        
        for segment in index_entry.segments:
            segment_text = segment.get("text", "").lower()
            if any(word in segment_text for word in query_words):
                word_matches = sum(1 for word in query_words if word in segment_text)
                relevance = word_matches / len(query_words)
                
                if relevance > 0.3:
                    results.append(SearchResult(
                        job_id=index_entry.job_id,
                        video_title=index_entry.video_title,
                        relevance_score=relevance * 0.8,
                        matched_text=segment.get("text", ""),
                        context=segment.get("text", ""),
                        timestamp_start=segment.get("start_time"),
                        timestamp_end=segment.get("end_time"),
                    ))
        
        return results
    
    async def _semantic_search(
        self,
        query: str,
        index_entry: TranscriptionIndex,
    ) -> Optional[SearchResult]:
        """Perform AI-powered semantic search"""
        text_sample = index_entry.full_text[:2000]
        
        try:
            response = await self.client.complete(
                prompt=self.SEMANTIC_SEARCH_PROMPT.format(
                    query=query,
                    text=text_sample,
                ),
                system_prompt="Eres un experto en búsqueda semántica. Analiza la relevancia del texto para la consulta.",
                max_tokens=500,
                temperature=0.2,
            )
            
            data = self._parse_json(response)
            
            if data.get("is_relevant") and data.get("relevance_score", 0) > 0.5:
                matched_segments = data.get("matched_segments", [])
                
                return SearchResult(
                    job_id=index_entry.job_id,
                    video_title=index_entry.video_title,
                    relevance_score=float(data.get("relevance_score", 0.5)),
                    matched_text=matched_segments[0] if matched_segments else "",
                    context=data.get("explanation", ""),
                )
            
            return None
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")
            return None
    
    async def extract_highlights(
        self,
        query: str,
        text: str,
        max_highlights: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Extract highlighted fragments relevant to query
        
        Args:
            query: Search query
            text: Text to search in
            max_highlights: Maximum highlights
            
        Returns:
            List of highlight dicts with text and reason
        """
        try:
            response = await self.client.complete(
                prompt=self.EXTRACT_HIGHLIGHTS_PROMPT.format(
                    query=query,
                    text=text[:3000],
                ),
                system_prompt="Extrae los fragmentos más relevantes del texto.",
                max_tokens=1000,
                temperature=0.3,
            )
            
            data = self._parse_json(response)
            return data.get("highlights", [])[:max_highlights]
            
        except Exception as e:
            logger.error(f"Failed to extract highlights: {e}")
            return []
    
    def _find_segment_for_position(
        self,
        char_position: int,
        segments: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Find which segment contains a character position"""
        current_pos = 0
        
        for segment in segments:
            segment_text = segment.get("text", "")
            segment_end = current_pos + len(segment_text) + 1
            
            if current_pos <= char_position < segment_end:
                return segment
            
            current_pos = segment_end
        
        return None
    
    def _parse_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from response"""
        response = response.strip()
        
        if response.startswith('```'):
            lines = response.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            response = '\n'.join(lines)
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except:
                    pass
            return {}
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get search index statistics"""
        return {
            "total_indexed": len(self._index),
            "total_segments": sum(len(idx.segments) for idx in self._index.values()),
            "total_words": sum(len(idx.full_text.split()) for idx in self._index.values()),
        }


_search_service: Optional[SearchService] = None


def get_search_service() -> SearchService:
    """Get search service singleton"""
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service












