"""
Core AI History System
=====================

Single file containing all core functionality.
Optimized for maximum efficiency and minimal complexity.
"""

import sqlite3
import json
import re
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import uuid


@dataclass
class HistoryEntry:
    """AI History Entry - optimized data structure."""
    id: str
    content: str
    model: str
    timestamp: str
    quality: float
    words: int
    readability: float
    sentiment: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComparisonResult:
    """Model Comparison Result - optimized data structure."""
    id: str
    model_a: str
    model_b: str
    similarity: float
    quality_diff: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AIHistorySystem:
    """
    Complete AI History Comparison System in a single class.
    Optimized for maximum efficiency and minimal complexity.
    """
    
    def __init__(self, db_path: str = "ai_history.db"):
        """Initialize the system."""
        self.db_path = db_path
        self._init_db()
        
        # Analysis constants
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent'
        }
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor',
            'worst', 'dreadful', 'atrocious', 'appalling', 'deplorable'
        }
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    model TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    quality REAL NOT NULL,
                    words INTEGER NOT NULL,
                    readability REAL NOT NULL,
                    sentiment REAL NOT NULL,
                    metadata TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comparisons (
                    id TEXT PRIMARY KEY,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    quality_diff REAL NOT NULL,
                    timestamp TEXT NOT NULL
                )
            """)
            conn.commit()
    
    def analyze_content(self, content: str, model: str, metadata: Optional[Dict] = None) -> HistoryEntry:
        """
        Analyze content and create history entry.
        
        Args:
            content: Content to analyze
            model: AI model version
            metadata: Optional metadata
            
        Returns:
            HistoryEntry with analysis results
        """
        # Extract words and sentences
        words = re.findall(r'\b[a-zA-Z]+\b', content.lower())
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate metrics
        word_count = len(words)
        sentence_count = len(sentences)
        
        # Readability score (simplified Flesch)
        readability = self._calculate_readability(word_count, sentence_count, content)
        
        # Sentiment score
        sentiment = self._calculate_sentiment(words)
        
        # Quality score (weighted average)
        quality = (readability * 0.4 + sentiment * 0.3 + 
                  min(1.0, word_count / 100) * 0.3)
        
        # Create entry
        entry = HistoryEntry(
            id=str(uuid.uuid4()),
            content=content,
            model=model,
            timestamp=datetime.utcnow().isoformat(),
            quality=quality,
            words=word_count,
            readability=readability,
            sentiment=sentiment,
            metadata=metadata or {}
        )
        
        # Save to database
        self._save_entry(entry)
        
        return entry
    
    def compare_models(self, entry1_id: str, entry2_id: str) -> ComparisonResult:
        """
        Compare two model entries.
        
        Args:
            entry1_id: First entry ID
            entry2_id: Second entry ID
            
        Returns:
            ComparisonResult
        """
        # Get entries
        entry1 = self._get_entry(entry1_id)
        entry2 = self._get_entry(entry2_id)
        
        if not entry1 or not entry2:
            raise ValueError("One or both entries not found")
        
        # Calculate similarity (Jaccard index)
        words1 = set(re.findall(r'\b[a-zA-Z]+\b', entry1.content.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]+\b', entry2.content.lower()))
        
        if words1 and words2:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            similarity = intersection / union if union > 0 else 0.0
        else:
            similarity = 0.0
        
        # Calculate quality difference
        quality_diff = abs(entry1.quality - entry2.quality)
        
        # Create result
        result = ComparisonResult(
            id=str(uuid.uuid4()),
            model_a=entry1.model,
            model_b=entry2.model,
            similarity=similarity,
            quality_diff=quality_diff,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Save to database
        self._save_comparison(result)
        
        return result
    
    def get_entries(self, model: Optional[str] = None, days: int = 7, limit: int = 100) -> List[HistoryEntry]:
        """
        Get history entries with optional filtering.
        
        Args:
            model: Filter by model
            days: Number of recent days
            limit: Maximum number of entries
            
        Returns:
            List of HistoryEntry objects
        """
        with sqlite3.connect(self.db_path) as conn:
            if model:
                query = "SELECT * FROM entries WHERE model = ? ORDER BY timestamp DESC LIMIT ?"
                params = (model, limit)
            else:
                cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
                query = "SELECT * FROM entries WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?"
                params = (cutoff, limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_entry(row) for row in rows]
    
    def get_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Get single entry by ID."""
        return self._get_entry(entry_id)
    
    def get_comparisons(self, model_a: Optional[str] = None, model_b: Optional[str] = None, 
                       days: int = 7, limit: int = 50) -> List[ComparisonResult]:
        """
        Get comparison results with optional filtering.
        
        Args:
            model_a: Filter by model A
            model_b: Filter by model B
            days: Number of recent days
            limit: Maximum number of results
            
        Returns:
            List of ComparisonResult objects
        """
        with sqlite3.connect(self.db_path) as conn:
            if model_a and model_b:
                query = """SELECT * FROM comparisons 
                          WHERE (model_a = ? AND model_b = ?) OR (model_a = ? AND model_b = ?)
                          ORDER BY timestamp DESC LIMIT ?"""
                params = (model_a, model_b, model_b, model_a, limit)
            else:
                cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
                query = "SELECT * FROM comparisons WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?"
                params = (cutoff, limit)
            
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            return [self._row_to_comparison(row) for row in rows]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            # Entry counts by model
            cursor = conn.execute("SELECT model, COUNT(*) FROM entries GROUP BY model")
            entry_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Average quality by model
            cursor = conn.execute("SELECT model, AVG(quality) FROM entries GROUP BY model")
            avg_quality = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Total comparisons
            cursor = conn.execute("SELECT COUNT(*) FROM comparisons")
            total_comparisons = cursor.fetchone()[0]
            
            return {
                "entries_by_model": entry_counts,
                "avg_quality_by_model": avg_quality,
                "total_entries": sum(entry_counts.values()),
                "total_comparisons": total_comparisons
            }
    
    def delete_entry(self, entry_id: str) -> bool:
        """Delete entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def _calculate_readability(self, word_count: int, sentence_count: int, content: str) -> float:
        """Calculate readability score (simplified Flesch Reading Ease)."""
        if sentence_count == 0 or word_count == 0:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = word_count / sentence_count
        
        # Count syllables (simplified)
        syllables = self._count_syllables(content)
        avg_syllables_per_word = syllables / word_count if word_count > 0 else 0
        
        # Simplified Flesch Reading Ease formula
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Normalize to 0-1 scale
        return max(0.0, min(1.0, readability / 100))
    
    def _calculate_sentiment(self, words: List[str]) -> float:
        """Calculate sentiment score."""
        if not words:
            return 0.5  # Neutral
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return (sentiment + 1) / 2  # Normalize to 0-1
    
    def _count_syllables(self, text: str) -> int:
        """Count syllables in text (simplified)."""
        vowels = 'aeiouy'
        count = 0
        
        for word in text.lower().split():
            word_count = 0
            prev_was_vowel = False
            
            for char in word:
                if char in vowels:
                    if not prev_was_vowel:
                        word_count += 1
                    prev_was_vowel = True
                else:
                    prev_was_vowel = False
            
            # Adjust for silent 'e'
            if word.endswith('e'):
                word_count -= 1
            
            # Every word has at least one syllable
            if word_count == 0:
                word_count = 1
            
            count += word_count
        
        return count
    
    def _save_entry(self, entry: HistoryEntry):
        """Save entry to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO entries 
                (id, content, model, timestamp, quality, words, readability, sentiment, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.id, entry.content, entry.model, entry.timestamp,
                entry.quality, entry.words, entry.readability, entry.sentiment,
                json.dumps(entry.metadata)
            ))
            conn.commit()
    
    def _save_comparison(self, result: ComparisonResult):
        """Save comparison to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO comparisons 
                (id, model_a, model_b, similarity, quality_diff, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                result.id, result.model_a, result.model_b,
                result.similarity, result.quality_diff, result.timestamp
            ))
            conn.commit()
    
    def _get_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        """Get entry by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_entry(row)
        
        return None
    
    def _row_to_entry(self, row) -> HistoryEntry:
        """Convert database row to HistoryEntry."""
        return HistoryEntry(
            id=row[0],
            content=row[1],
            model=row[2],
            timestamp=row[3],
            quality=row[4],
            words=row[5],
            readability=row[6],
            sentiment=row[7],
            metadata=json.loads(row[8]) if row[8] else {}
        )
    
    def _row_to_comparison(self, row) -> ComparisonResult:
        """Convert database row to ComparisonResult."""
        return ComparisonResult(
            id=row[0],
            model_a=row[1],
            model_b=row[2],
            similarity=row[3],
            quality_diff=row[4],
            timestamp=row[5]
        )




