"""
Knowledge Base
==============

Advanced knowledge base system for TruthGPT.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
import aiofiles
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class KnowledgeEntry:
    """Knowledge base entry."""
    id: str
    content: str
    category: str
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    quality_score: float = 0.0

class KnowledgeBase:
    """
    Advanced knowledge base system.
    
    Features:
    - Knowledge storage and retrieval
    - Semantic search
    - Category organization
    - Quality assessment
    - Access tracking
    - Knowledge evolution
    """
    
    def __init__(self, storage_path: str = "./knowledge_base"):
        self.storage_path = Path(storage_path)
        self.entries = {}
        self.categories = defaultdict(list)
        self.tags = defaultdict(list)
        self.search_index = {}
        self.access_stats = defaultdict(int)
        
    async def initialize(self):
        """Initialize knowledge base."""
        logger.info("Initializing Knowledge Base...")
        
        try:
            # Create storage directory
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Load existing knowledge
            await self._load_knowledge()
            
            # Start background tasks
            asyncio.create_task(self._update_access_stats())
            asyncio.create_task(self._cleanup_old_entries())
            
            logger.info("Knowledge Base initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Base: {str(e)}")
            raise
    
    async def _load_knowledge(self):
        """Load existing knowledge entries."""
        try:
            knowledge_file = self.storage_path / "knowledge.json"
            
            if knowledge_file.exists():
                async with aiofiles.open(knowledge_file, 'r', encoding='utf-8') as f:
                    data = json.loads(await f.read())
                
                # Load entries
                for entry_data in data.get('entries', []):
                    entry = KnowledgeEntry(
                        id=entry_data['id'],
                        content=entry_data['content'],
                        category=entry_data['category'],
                        tags=entry_data['tags'],
                        metadata=entry_data['metadata'],
                        created_at=datetime.fromisoformat(entry_data['created_at']),
                        updated_at=datetime.fromisoformat(entry_data['updated_at']),
                        access_count=entry_data.get('access_count', 0),
                        quality_score=entry_data.get('quality_score', 0.0)
                    )
                    
                    self.entries[entry.id] = entry
                    self.categories[entry.category].append(entry.id)
                    
                    for tag in entry.tags:
                        self.tags[tag].append(entry.id)
                
                logger.info(f"Loaded {len(self.entries)} knowledge entries")
            
        except Exception as e:
            logger.error(f"Failed to load knowledge: {str(e)}")
    
    async def add_knowledge(
        self,
        content: str,
        category: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add knowledge entry."""
        try:
            # Generate entry ID
            entry_id = hashlib.md5(content.encode()).hexdigest()[:16]
            
            # Check if entry already exists
            if entry_id in self.entries:
                # Update existing entry
                entry = self.entries[entry_id]
                entry.content = content
                entry.category = category
                entry.tags = tags or []
                entry.metadata = metadata or {}
                entry.updated_at = datetime.utcnow()
                
                logger.info(f"Updated knowledge entry: {entry_id}")
            else:
                # Create new entry
                entry = KnowledgeEntry(
                    id=entry_id,
                    content=content,
                    category=category,
                    tags=tags or [],
                    metadata=metadata or {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                
                self.entries[entry_id] = entry
                self.categories[category].append(entry_id)
                
                for tag in entry.tags:
                    self.tags[tag].append(entry_id)
                
                logger.info(f"Added knowledge entry: {entry_id}")
            
            # Save to disk
            await self._save_knowledge()
            
            return entry_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge: {str(e)}")
            raise
    
    async def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[KnowledgeEntry]:
        """Search knowledge base."""
        try:
            # Get candidate entries
            candidates = set()
            
            if category:
                candidates.update(self.categories.get(category, []))
            elif tags:
                for tag in tags:
                    candidates.update(self.tags.get(tag, []))
            else:
                candidates = set(self.entries.keys())
            
            # Filter and score entries
            results = []
            query_lower = query.lower()
            
            for entry_id in candidates:
                entry = self.entries[entry_id]
                
                # Calculate relevance score
                relevance_score = self._calculate_relevance_score(entry, query_lower)
                
                if relevance_score > 0:
                    results.append((entry, relevance_score))
            
            # Sort by relevance score
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            top_results = [entry for entry, score in results[:limit]]
            
            # Update access counts
            for entry in top_results:
                entry.access_count += 1
                self.access_stats[entry.id] += 1
            
            return top_results
            
        except Exception as e:
            logger.error(f"Failed to search knowledge: {str(e)}")
            return []
    
    def _calculate_relevance_score(self, entry: KnowledgeEntry, query: str) -> float:
        """Calculate relevance score for an entry."""
        try:
            score = 0.0
            content_lower = entry.content.lower()
            
            # Exact phrase match
            if query in content_lower:
                score += 10.0
            
            # Word matches
            query_words = query.split()
            content_words = content_lower.split()
            
            for word in query_words:
                if word in content_words:
                    score += 1.0
            
            # Tag matches
            for tag in entry.tags:
                if any(word in tag.lower() for word in query_words):
                    score += 2.0
            
            # Quality score bonus
            score += entry.quality_score * 2.0
            
            # Access count bonus (popular entries)
            score += min(entry.access_count * 0.1, 5.0)
            
            return score
            
        except Exception as e:
            logger.error(f"Failed to calculate relevance score: {str(e)}")
            return 0.0
    
    async def get_knowledge_by_category(self, category: str) -> List[KnowledgeEntry]:
        """Get all knowledge entries in a category."""
        try:
            entry_ids = self.categories.get(category, [])
            return [self.entries[entry_id] for entry_id in entry_ids if entry_id in self.entries]
            
        except Exception as e:
            logger.error(f"Failed to get knowledge by category: {str(e)}")
            return []
    
    async def get_knowledge_by_tags(self, tags: List[str]) -> List[KnowledgeEntry]:
        """Get knowledge entries by tags."""
        try:
            entry_ids = set()
            for tag in tags:
                entry_ids.update(self.tags.get(tag, []))
            
            return [self.entries[entry_id] for entry_id in entry_ids if entry_id in self.entries]
            
        except Exception as e:
            logger.error(f"Failed to get knowledge by tags: {str(e)}")
            return []
    
    async def update_knowledge_quality(
        self,
        entry_id: str,
        quality_score: float
    ):
        """Update knowledge entry quality score."""
        try:
            if entry_id in self.entries:
                self.entries[entry_id].quality_score = quality_score
                self.entries[entry_id].updated_at = datetime.utcnow()
                
                # Save to disk
                await self._save_knowledge()
                
                logger.info(f"Updated quality score for entry {entry_id}: {quality_score}")
            
        except Exception as e:
            logger.error(f"Failed to update knowledge quality: {str(e)}")
    
    async def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        try:
            total_entries = len(self.entries)
            total_categories = len(self.categories)
            total_tags = len(self.tags)
            
            # Category distribution
            category_distribution = {
                category: len(entry_ids) 
                for category, entry_ids in self.categories.items()
            }
            
            # Tag distribution
            tag_distribution = {
                tag: len(entry_ids) 
                for tag, entry_ids in self.tags.items()
            }
            
            # Quality distribution
            quality_scores = [entry.quality_score for entry in self.entries.values()]
            quality_distribution = {
                'excellent': len([s for s in quality_scores if s >= 0.9]),
                'good': len([s for s in quality_scores if 0.7 <= s < 0.9]),
                'fair': len([s for s in quality_scores if 0.5 <= s < 0.7]),
                'poor': len([s for s in quality_scores if s < 0.5])
            }
            
            # Access statistics
            most_accessed = sorted(
                self.access_stats.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                'total_entries': total_entries,
                'total_categories': total_categories,
                'total_tags': total_tags,
                'category_distribution': category_distribution,
                'tag_distribution': dict(sorted(tag_distribution.items(), key=lambda x: x[1], reverse=True)[:20]),
                'quality_distribution': quality_distribution,
                'most_accessed_entries': most_accessed,
                'average_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get knowledge stats: {str(e)}")
            return {}
    
    async def _save_knowledge(self):
        """Save knowledge base to disk."""
        try:
            knowledge_file = self.storage_path / "knowledge.json"
            
            data = {
                'entries': [
                    {
                        'id': entry.id,
                        'content': entry.content,
                        'category': entry.category,
                        'tags': entry.tags,
                        'metadata': entry.metadata,
                        'created_at': entry.created_at.isoformat(),
                        'updated_at': entry.updated_at.isoformat(),
                        'access_count': entry.access_count,
                        'quality_score': entry.quality_score
                    }
                    for entry in self.entries.values()
                ],
                'categories': dict(self.categories),
                'tags': dict(self.tags),
                'access_stats': dict(self.access_stats),
                'last_updated': datetime.utcnow().isoformat()
            }
            
            async with aiofiles.open(knowledge_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
            
        except Exception as e:
            logger.error(f"Failed to save knowledge: {str(e)}")
    
    async def _update_access_stats(self):
        """Update access statistics."""
        while True:
            try:
                await asyncio.sleep(3600)  # Update every hour
                
                # Update access statistics
                for entry in self.entries.values():
                    if entry.access_count > 0:
                        # Decay old access counts
                        entry.access_count = int(entry.access_count * 0.95)
                
                # Save updated stats
                await self._save_knowledge()
                
            except Exception as e:
                logger.error(f"Error updating access stats: {str(e)}")
    
    async def _cleanup_old_entries(self):
        """Cleanup old, low-quality entries."""
        while True:
            try:
                await asyncio.sleep(86400)  # Cleanup daily
                
                cutoff_time = datetime.utcnow() - timedelta(days=30)
                entries_to_remove = []
                
                for entry_id, entry in self.entries.items():
                    # Remove old, low-quality, rarely accessed entries
                    if (entry.updated_at < cutoff_time and 
                        entry.quality_score < 0.3 and 
                        entry.access_count < 2):
                        entries_to_remove.append(entry_id)
                
                # Remove entries
                for entry_id in entries_to_remove:
                    entry = self.entries[entry_id]
                    
                    # Remove from categories
                    if entry_id in self.categories[entry.category]:
                        self.categories[entry.category].remove(entry_id)
                    
                    # Remove from tags
                    for tag in entry.tags:
                        if entry_id in self.tags[tag]:
                            self.tags[tag].remove(entry_id)
                    
                    # Remove from entries
                    del self.entries[entry_id]
                
                if entries_to_remove:
                    logger.info(f"Cleaned up {len(entries_to_remove)} old knowledge entries")
                    await self._save_knowledge()
                
            except Exception as e:
                logger.error(f"Error cleaning up old entries: {str(e)}")
    
    async def export_knowledge(self, format: str = "json") -> str:
        """Export knowledge base."""
        try:
            if format == "json":
                return json.dumps({
                    'entries': [
                        {
                            'id': entry.id,
                            'content': entry.content,
                            'category': entry.category,
                            'tags': entry.tags,
                            'metadata': entry.metadata,
                            'created_at': entry.created_at.isoformat(),
                            'updated_at': entry.updated_at.isoformat(),
                            'access_count': entry.access_count,
                            'quality_score': entry.quality_score
                        }
                        for entry in self.entries.values()
                    ],
                    'categories': dict(self.categories),
                    'tags': dict(self.tags),
                    'exported_at': datetime.utcnow().isoformat()
                }, indent=2, ensure_ascii=False)
            else:
                return str(self.entries)
                
        except Exception as e:
            logger.error(f"Failed to export knowledge: {str(e)}")
            return ""
    
    async def cleanup(self):
        """Cleanup knowledge base."""
        try:
            # Save final state
            await self._save_knowledge()
            
            logger.info("Knowledge Base cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Failed to cleanup Knowledge Base: {str(e)}")











