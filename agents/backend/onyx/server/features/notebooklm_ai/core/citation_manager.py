from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import logging
import time
import re
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
from urllib.parse import urlparse, quote
import aiohttp
from datetime import datetime
import pandas as pd
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
import aiofiles
from .entities import Citation, Document, Analysis
from ..nlp import NLPEngine
                    import xml.etree.ElementTree as ET
from typing import Any, List, Dict, Optional
"""
Advanced Citation and Reference Management System
================================================

A comprehensive system for automatic citation generation, validation,
formatting, and reference management with support for multiple formats
and academic databases.
"""



# Core imports

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CitationConfig:
    """Configuration for citation processing"""
    enable_auto_detection: bool = True
    enable_validation: bool = True
    enable_formatting: bool = True
    enable_database_lookup: bool = True
    enable_doi_resolution: bool = True
    enable_arxiv_lookup: bool = True
    enable_google_scholar: bool = True
    enable_crossref: bool = True
    max_citations_per_doc: int = 100
    confidence_threshold: float = 0.7
    cache_ttl: int = 86400  # 24 hours
    request_timeout: int = 30


class CitationFormat(BaseModel):
    """Citation format specification"""
    name: str
    description: str
    format_string: str
    examples: List[str] = Field(default_factory=list)
    fields: List[str] = Field(default_factory=list)


class CitationDatabase(BaseModel):
    """Citation database configuration"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per minute
    timeout: int = 30
    enabled: bool = True


class CitationManager:
    """
    Advanced Citation and Reference Management System
    
    Features:
    - Automatic citation detection and extraction
    - Multi-format citation generation (APA, MLA, Chicago, etc.)
    - Database integration (CrossRef, arXiv, Google Scholar)
    - DOI resolution and validation
    - Citation clustering and deduplication
    - Reference list generation
    - Academic database lookups
    """
    
    def __init__(
        self,
        config: CitationConfig = None,
        redis_url: str = "redis://localhost:6379",
        db_session: AsyncSession = None
    ):
        
    """__init__ function."""
self.config = config or CitationConfig()
        self.redis_url = redis_url
        self.db_session = db_session
        
        # Initialize components
        self.nlp_engine = NLPEngine()
        
        # Citation patterns and formats
        self.citation_patterns = self._load_citation_patterns()
        self.citation_formats = self._load_citation_formats()
        self.databases = self._load_databases()
        
        # Cache and state
        self._cache = {}
        self._session = None
        
        # Performance metrics
        self.metrics = {
            'citations_detected': 0,
            'citations_validated': 0,
            'citations_formatted': 0,
            'database_lookups': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        
        logger.info("Citation Manager initialized")
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        await self.startup()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self.shutdown()
    
    async def startup(self) -> Any:
        """Initialize components"""
        try:
            # Initialize Redis connection
            self.redis = redis.from_url(self.redis_url)
            await self.redis.ping()
            
            # Initialize HTTP session
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.request_timeout)
            )
            
            logger.info("Citation Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start Citation Manager: {e}")
            raise
    
    async def shutdown(self) -> Any:
        """Cleanup and shutdown"""
        try:
            if hasattr(self, 'redis'):
                await self.redis.close()
            
            if self._session:
                await self._session.close()
            
            logger.info("Citation Manager shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    async def extract_citations(
        self,
        text: str,
        document: Document = None
    ) -> List[Citation]:
        """
        Extract citations from text content
        
        Args:
            text: Text content to analyze
            document: Source document (optional)
            
        Returns:
            List of detected citations
        """
        if not self.config.enable_auto_detection:
            return []
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(text, "extract_citations")
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                return [Citation(**c) for c in cached_result]
            
            self.metrics['cache_misses'] += 1
            
            citations = []
            
            # Extract citations using multiple methods
            pattern_citations = await self._extract_by_patterns(text)
            nlp_citations = await self._extract_by_nlp(text)
            url_citations = await self._extract_by_urls(text)
            
            # Combine and deduplicate
            all_citations = pattern_citations + nlp_citations + url_citations
            citations = await self._deduplicate_citations(all_citations)
            
            # Limit number of citations
            citations = citations[:self.config.max_citations_per_doc]
            
            # Cache result
            await self._cache_result(cache_key, [c.dict() for c in citations])
            
            self.metrics['citations_detected'] += len(citations)
            
            return citations
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error extracting citations: {e}")
            return []
    
    async def validate_citations(
        self,
        citations: List[Citation]
    ) -> List[Citation]:
        """
        Validate and enrich citations with database lookups
        
        Args:
            citations: List of citations to validate
            
        Returns:
            List of validated citations
        """
        if not self.config.enable_validation:
            return citations
        
        try:
            validated_citations = []
            
            for citation in citations:
                # Skip if already validated
                if citation.confidence >= self.config.confidence_threshold:
                    validated_citations.append(citation)
                    continue
                
                # Validate citation
                validated_citation = await self._validate_single_citation(citation)
                if validated_citation:
                    validated_citations.append(validated_citation)
                    self.metrics['citations_validated'] += 1
            
            return validated_citations
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error validating citations: {e}")
            return citations
    
    async def format_citations(
        self,
        citations: List[Citation],
        format_name: str = "APA"
    ) -> List[str]:
        """
        Format citations in specified style
        
        Args:
            citations: List of citations to format
            format_name: Citation format (APA, MLA, Chicago, etc.)
            
        Returns:
            List of formatted citation strings
        """
        if not self.config.enable_formatting:
            return [str(c) for c in citations]
        
        try:
            format_spec = self.citation_formats.get(format_name.lower())
            if not format_spec:
                logger.warning(f"Unknown citation format: {format_name}")
                return [str(c) for c in citations]
            
            formatted_citations = []
            
            for citation in citations:
                formatted = await self._format_single_citation(citation, format_spec)
                formatted_citations.append(formatted)
                self.metrics['citations_formatted'] += 1
            
            return formatted_citations
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error formatting citations: {e}")
            return [str(c) for c in citations]
    
    async def generate_reference_list(
        self,
        citations: List[Citation],
        format_name: str = "APA",
        sort_by: str = "authors"
    ) -> str:
        """
        Generate a complete reference list
        
        Args:
            citations: List of citations
            format_name: Citation format
            sort_by: Sort criteria (authors, year, title)
            
        Returns:
            Formatted reference list
        """
        try:
            # Sort citations
            sorted_citations = await self._sort_citations(citations, sort_by)
            
            # Format citations
            formatted_citations = await self.format_citations(sorted_citations, format_name)
            
            # Generate reference list
            reference_list = "\n\n".join([
                f"{i+1}. {citation}" for i, citation in enumerate(formatted_citations)
            ])
            
            return reference_list
            
        except Exception as e:
            logger.error(f"Error generating reference list: {e}")
            return ""
    
    async def lookup_citation(
        self,
        query: str,
        database: str = "crossref"
    ) -> Optional[Citation]:
        """
        Look up citation information from academic databases
        
        Args:
            query: Search query (title, author, DOI, etc.)
            database: Database to search (crossref, arxiv, scholar)
            
        Returns:
            Citation object if found
        """
        if not self.config.enable_database_lookup:
            return None
        
        try:
            db_config = self.databases.get(database.lower())
            if not db_config or not db_config.enabled:
                logger.warning(f"Database {database} not available or disabled")
                return None
            
            # Check cache first
            cache_key = self._generate_cache_key(f"{database}:{query}", "lookup")
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                return Citation(**cached_result)
            
            self.metrics['cache_misses'] += 1
            
            # Perform database lookup
            citation = await self._perform_database_lookup(query, db_config)
            
            if citation:
                # Cache result
                await self._cache_result(cache_key, citation.dict())
                self.metrics['database_lookups'] += 1
            
            return citation
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error looking up citation: {e}")
            return None
    
    async def resolve_doi(self, doi: str) -> Optional[Citation]:
        """
        Resolve DOI to citation information
        
        Args:
            doi: DOI string
            
        Returns:
            Citation object if resolved
        """
        if not self.config.enable_doi_resolution:
            return None
        
        try:
            # Normalize DOI
            doi = doi.strip().lower()
            if not doi.startswith('10.'):
                return None
            
            # Check cache
            cache_key = self._generate_cache_key(f"doi:{doi}", "resolve")
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                self.metrics['cache_hits'] += 1
                return Citation(**cached_result)
            
            self.metrics['cache_misses'] += 1
            
            # Resolve DOI using CrossRef
            citation = await self._resolve_doi_crossref(doi)
            
            if citation:
                # Cache result
                await self._cache_result(cache_key, citation.dict())
            
            return citation
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"Error resolving DOI {doi}: {e}")
            return None
    
    async def _extract_by_patterns(self, text: str) -> List[Citation]:
        """Extract citations using regex patterns"""
        citations = []
        
        for pattern_name, pattern_info in self.citation_patterns.items():
            pattern = pattern_info['pattern']
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                citation_data = self._parse_pattern_match(match, pattern_info)
                if citation_data:
                    citation = Citation(**citation_data)
                    citations.append(citation)
        
        return citations
    
    async def _extract_by_nlp(self, text: str) -> List[Citation]:
        """Extract citations using NLP techniques"""
        try:
            # Use NLP engine to identify citation patterns
            citation_patterns = await self.nlp_engine.extract_citation_patterns(text)
            
            citations = []
            for pattern in citation_patterns:
                citation = Citation(
                    id=f"nlp_citation_{len(citations)}",
                    source=pattern.get('source', ''),
                    title=pattern.get('title', ''),
                    authors=pattern.get('authors', []),
                    year=pattern.get('year', ''),
                    url=pattern.get('url', ''),
                    confidence=pattern.get('confidence', 0.5),
                    context=pattern.get('context', '')
                )
                citations.append(citation)
            
            return citations
            
        except Exception as e:
            logger.error(f"Error in NLP citation extraction: {e}")
            return []
    
    async def _extract_by_urls(self, text: str) -> List[Citation]:
        """Extract citations from URLs in text"""
        citations = []
        
        # URL patterns
        url_patterns = [
            r'https?://(?:www\.)?(arxiv\.org|doi\.org|scholar\.google\.com|researchgate\.net)/[^\s]+',
            r'https?://(?:www\.)?([a-zA-Z0-9.-]+\.(?:com|org|edu|gov))/[^\s]+'
        ]
        
        for pattern in url_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                url = match.group(0)
                domain = urlparse(url).netloc
                
                citation = Citation(
                    id=f"url_citation_{len(citations)}",
                    source=domain,
                    title="",
                    authors=[],
                    year="",
                    url=url,
                    confidence=0.3,
                    context=match.group(0)
                )
                citations.append(citation)
        
        return citations
    
    async def _deduplicate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Remove duplicate citations"""
        seen = set()
        unique_citations = []
        
        for citation in citations:
            # Create unique identifier
            identifier = self._create_citation_identifier(citation)
            
            if identifier not in seen:
                seen.add(identifier)
                unique_citations.append(citation)
        
        return unique_citations
    
    async def _validate_single_citation(self, citation: Citation) -> Optional[Citation]:
        """Validate a single citation"""
        try:
            # Try to resolve DOI if present
            if citation.url and 'doi.org' in citation.url:
                doi = citation.url.split('doi.org/')[-1]
                resolved = await self.resolve_doi(doi)
                if resolved:
                    return resolved
            
            # Try database lookup
            if citation.title:
                looked_up = await self.lookup_citation(citation.title)
                if looked_up:
                    return looked_up
            
            # Return original if no validation possible
            return citation
            
        except Exception as e:
            logger.error(f"Error validating citation: {e}")
            return citation
    
    async def _format_single_citation(
        self,
        citation: Citation,
        format_spec: CitationFormat
    ) -> str:
        """Format a single citation"""
        try:
            # Extract format fields
            format_fields = {}
            
            # Authors
            if citation.authors:
                if len(citation.authors) == 1:
                    format_fields['authors'] = citation.authors[0]
                elif len(citation.authors) == 2:
                    format_fields['authors'] = f"{citation.authors[0]} & {citation.authors[1]}"
                else:
                    format_fields['authors'] = f"{citation.authors[0]} et al."
            else:
                format_fields['authors'] = "Unknown"
            
            # Title
            format_fields['title'] = citation.title or "Untitled"
            
            # Year
            format_fields['year'] = citation.year or "n.d."
            
            # Source
            format_fields['source'] = citation.source or "Unknown"
            
            # URL
            format_fields['url'] = citation.url or ""
            
            # Apply format string
            formatted = format_spec.format_string
            for field, value in format_fields.items():
                formatted = formatted.replace(f"{{{field}}}", str(value))
            
            return formatted
            
        except Exception as e:
            logger.error(f"Error formatting citation: {e}")
            return str(citation)
    
    async def _sort_citations(
        self,
        citations: List[Citation],
        sort_by: str
    ) -> List[Citation]:
        """Sort citations by specified criteria"""
        try:
            match sort_by:
    case "authors":
                return sorted(citations, key=lambda x: x.authors[0] if x.authors else "")
            elif sort_by == "year":
                return sorted(citations, key=lambda x: x.year or "0000")
            elif sort_by == "title":
                return sorted(citations, key=lambda x: x.title or "")
            else:
                return citations
                
        except Exception as e:
            logger.error(f"Error sorting citations: {e}")
            return citations
    
    async def _perform_database_lookup(
        self,
        query: str,
        db_config: CitationDatabase
    ) -> Optional[Citation]:
        """Perform lookup in academic database"""
        try:
            if db_config.name.lower() == "crossref":
                return await self._lookup_crossref(query, db_config)
            elif db_config.name.lower() == "arxiv":
                return await self._lookup_arxiv(query, db_config)
            elif db_config.name.lower() == "scholar":
                return await self._lookup_google_scholar(query, db_config)
            else:
                logger.warning(f"Unknown database: {db_config.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error in database lookup: {e}")
            return None
    
    async def _resolve_doi_crossref(self, doi: str) -> Optional[Citation]:
        """Resolve DOI using CrossRef API"""
        try:
            url = f"https://api.crossref.org/works/{doi}"
            
            async with self._session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    work = data.get('message', {})
                    
                    citation = Citation(
                        id=f"doi_{doi}",
                        source="CrossRef",
                        title=work.get('title', [''])[0] if work.get('title') else '',
                        authors=[author.get('given', '') + ' ' + author.get('family', '') 
                                for author in work.get('author', [])],
                        year=str(work.get('published-print', {}).get('date-parts', [['']])[0][0]),
                        url=f"https://doi.org/{doi}",
                        confidence=0.9,
                        context=f"DOI: {doi}"
                    )
                    
                    return citation
                
                return None
                
        except Exception as e:
            logger.error(f"Error resolving DOI {doi}: {e}")
            return None
    
    async def _lookup_crossref(
        self,
        query: str,
        db_config: CitationDatabase
    ) -> Optional[Citation]:
        """Lookup in CrossRef database"""
        try:
            url = "https://api.crossref.org/works"
            params = {
                'query': query,
                'rows': 1,
                'select': 'DOI,title,author,published-print'
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get('message', {}).get('items', [])
                    
                    if items:
                        work = items[0]
                        doi = work.get('DOI', '')
                        
                        citation = Citation(
                            id=f"crossref_{doi}",
                            source="CrossRef",
                            title=work.get('title', [''])[0] if work.get('title') else '',
                            authors=[author.get('given', '') + ' ' + author.get('family', '') 
                                    for author in work.get('author', [])],
                            year=str(work.get('published-print', {}).get('date-parts', [['']])[0][0]),
                            url=f"https://doi.org/{doi}",
                            confidence=0.8,
                            context=query
                        )
                        
                        return citation
                
                return None
                
        except Exception as e:
            logger.error(f"Error in CrossRef lookup: {e}")
            return None
    
    async def _lookup_arxiv(
        self,
        query: str,
        db_config: CitationDatabase
    ) -> Optional[Citation]:
        """Lookup in arXiv database"""
        try:
            url = "http://export.arxiv.org/api/query"
            params = {
                'search_query': f'all:"{query}"',
                'start': 0,
                'max_results': 1
            }
            
            async with self._session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.text()
                    
                    # Parse XML response (simplified)
                    root = ET.fromstring(data)
                    
                    entry = root.find('.//{http://www.w3.org/2005/Atom}entry')
                    if entry:
                        title = entry.find('.//{http://www.w3.org/2005/Atom}title').text
                        authors = [author.find('.//{http://www.w3.org/2005/Atom}name').text 
                                 for author in entry.findall('.//{http://www.w3.org/2005/Atom}author')]
                        published = entry.find('.//{http://www.w3.org/2005/Atom}published').text[:4]
                        arxiv_id = entry.find('.//{http://www.w3.org/2005/Atom}id').text.split('/')[-1]
                        
                        citation = Citation(
                            id=f"arxiv_{arxiv_id}",
                            source="arXiv",
                            title=title,
                            authors=authors,
                            year=published,
                            url=f"https://arxiv.org/abs/{arxiv_id}",
                            confidence=0.8,
                            context=query
                        )
                        
                        return citation
                
                return None
                
        except Exception as e:
            logger.error(f"Error in arXiv lookup: {e}")
            return None
    
    async def _lookup_google_scholar(
        self,
        query: str,
        db_config: CitationDatabase
    ) -> Optional[Citation]:
        """Lookup in Google Scholar (simplified)"""
        # Note: Google Scholar doesn't have a public API
        # This is a simplified implementation
        try:
            # For now, return a basic citation
            citation = Citation(
                id=f"scholar_{hashlib.md5(query.encode()).hexdigest()[:8]}",
                source="Google Scholar",
                title=query,
                authors=[],
                year="",
                url="",
                confidence=0.3,
                context=query
            )
            
            return citation
            
        except Exception as e:
            logger.error(f"Error in Google Scholar lookup: {e}")
            return None
    
    def _load_citation_patterns(self) -> Dict[str, Dict]:
        """Load citation detection patterns"""
        return {
            'apa_inline': {
                'pattern': r'\(([^)]+),\s*(\d{4})\)',
                'groups': ['authors', 'year']
            },
            'apa_reference': {
                'pattern': r'^([^,]+),\s*([^,]+),\s*(\d{4})\.\s*(.+)$',
                'groups': ['authors', 'title', 'year', 'source']
            },
            'doi': {
                'pattern': r'10\.\d{4,}/[-._;()/:\w]+',
                'groups': ['doi']
            },
            'url': {
                'pattern': r'https?://[^\s]+',
                'groups': ['url']
            }
        }
    
    def _load_citation_formats(self) -> Dict[str, CitationFormat]:
        """Load citation format specifications"""
        return {
            'apa': CitationFormat(
                name="APA",
                description="American Psychological Association",
                format_string="{authors} ({year}). {title}. {source}.",
                fields=['authors', 'year', 'title', 'source']
            ),
            'mla': CitationFormat(
                name="MLA",
                description="Modern Language Association",
                format_string="{authors}. \"{title}.\" {source}, {year}.",
                fields=['authors', 'title', 'source', 'year']
            ),
            'chicago': CitationFormat(
                name="Chicago",
                description="Chicago Manual of Style",
                format_string="{authors}. \"{title}.\" {source} ({year}).",
                fields=['authors', 'title', 'source', 'year']
            ),
            'harvard': CitationFormat(
                name="Harvard",
                description="Harvard Referencing Style",
                format_string="{authors} {year}, {title}, {source}.",
                fields=['authors', 'year', 'title', 'source']
            )
        }
    
    def _load_databases(self) -> Dict[str, CitationDatabase]:
        """Load database configurations"""
        return {
            'crossref': CitationDatabase(
                name="CrossRef",
                base_url="https://api.crossref.org",
                rate_limit=100,
                timeout=30,
                enabled=True
            ),
            'arxiv': CitationDatabase(
                name="arXiv",
                base_url="http://export.arxiv.org/api",
                rate_limit=50,
                timeout=30,
                enabled=True
            ),
            'scholar': CitationDatabase(
                name="Google Scholar",
                base_url="",
                rate_limit=10,
                timeout=30,
                enabled=False  # No public API
            )
        }
    
    def _parse_pattern_match(self, match, pattern_info: Dict) -> Optional[Dict]:
        """Parse regex match into citation data"""
        try:
            groups = pattern_info.get('groups', [])
            group_values = match.groups()
            
            citation_data = {
                'id': f"pattern_{hashlib.md5(match.group(0).encode()).hexdigest()[:8]}",
                'source': 'Pattern Detection',
                'title': '',
                'authors': [],
                'year': '',
                'url': '',
                'confidence': 0.5,
                'context': match.group(0)
            }
            
            for i, group in enumerate(groups):
                if i < len(group_values):
                    if group == 'authors':
                        citation_data['authors'] = [group_values[i]]
                    elif group == 'year':
                        citation_data['year'] = group_values[i]
                    elif group == 'title':
                        citation_data['title'] = group_values[i]
                    elif group == 'source':
                        citation_data['source'] = group_values[i]
                    elif group == 'doi':
                        citation_data['url'] = f"https://doi.org/{group_values[i]}"
                    elif group == 'url':
                        citation_data['url'] = group_values[i]
            
            return citation_data
            
        except Exception as e:
            logger.error(f"Error parsing pattern match: {e}")
            return None
    
    def _create_citation_identifier(self, citation: Citation) -> str:
        """Create unique identifier for citation"""
        # Combine key fields for deduplication
        key_parts = [
            citation.title or '',
            citation.authors[0] if citation.authors else '',
            citation.year or '',
            citation.url or ''
        ]
        
        return hashlib.md5(''.join(key_parts).encode()).hexdigest()
    
    def _generate_cache_key(self, content: str, operation: str) -> str:
        """Generate cache key"""
        return f"citation:{operation}:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        try:
            if hasattr(self, 'redis'):
                cached_data = await self.redis.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
            return None
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: Any):
        """Cache result"""
        try:
            if hasattr(self, 'redis'):
                await self.redis.setex(
                    cache_key,
                    self.config.cache_ttl,
                    json.dumps(result)
                )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            **self.metrics,
            'cache_size': len(self._cache),
            'uptime': time.time() - getattr(self, '_start_time', time.time())
        }
    
    async def clear_cache(self) -> Any:
        """Clear all caches"""
        self._cache.clear()
        if hasattr(self, 'redis'):
            await self.redis.flushdb()
        logger.info("Citation cache cleared")


# Example usage
async def main():
    """Example usage of Citation Manager"""
    
    # Initialize manager
    config = CitationConfig(
        enable_auto_detection=True,
        enable_validation=True,
        enable_formatting=True,
        enable_database_lookup=True,
        enable_doi_resolution=True
    )
    
    async with CitationManager(config) as manager:
        # Sample text with citations
        text = """
        Recent studies have shown significant improvements in AI performance (Smith et al., 2023).
        The research by Johnson and Brown (2022) demonstrated novel approaches to machine learning.
        For more information, see https://doi.org/10.1038/s41586-023-06184-4
        """
        
        # Extract citations
        citations = await manager.extract_citations(text)
        print(f"Extracted {len(citations)} citations")
        
        # Validate citations
        validated = await manager.validate_citations(citations)
        print(f"Validated {len(validated)} citations")
        
        # Format citations
        formatted = await manager.format_citations(validated, "APA")
        for citation in formatted:
            print(f"- {citation}")
        
        # Generate reference list
        reference_list = await manager.generate_reference_list(validated, "APA")
        print(f"\nReference List:\n{reference_list}")
        
        # Look up specific citation
        lookup_result = await manager.lookup_citation("Deep Learning", "crossref")
        if lookup_result:
            print(f"\nLookup result: {lookup_result}")
        
        # Get metrics
        metrics = await manager.get_metrics()
        print(f"\nMetrics: {metrics}")


match __name__:
    case "__main__":
    asyncio.run(main()) 