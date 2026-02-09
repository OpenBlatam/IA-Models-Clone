from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from typing import List, Optional
import re
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Content Value Object
===================

Value object for post content with validation and business rules.
"""



@dataclass(frozen=True)
class Content:
    """
    Content value object with validation and business rules.
    
    This value object encapsulates content validation logic and ensures
    content meets business requirements.
    """
    
    value: str
    
    def __post_init__(self) -> Any:
        """Validate content after initialization."""
        self._validate_content()
    
    def _validate_content(self) -> None:
        """Validate content according to business rules."""
        if not self.value:
            raise ValueError("Content cannot be empty")
        
        if len(self.value.strip()) < 10:
            raise ValueError("Content must be at least 10 characters long")
        
        if len(self.value) > 3000:
            raise ValueError("Content cannot exceed 3000 characters")
        
        # Check for profanity (basic implementation)
        if self._contains_profanity():
            raise ValueError("Content contains inappropriate language")
    
    def _contains_profanity(self) -> bool:
        """Check if content contains profanity."""
        # Basic profanity check - in production, use a proper profanity filter
        profanity_patterns = [
            r'\b(bad_word1|bad_word2)\b',  # Add actual profanity patterns
        ]
        
        for pattern in profanity_patterns:
            if re.search(pattern, self.value.lower()):
                return True
        
        return False
    
    def is_valid_for_publishing(self) -> bool:
        """Check if content is valid for publishing."""
        try:
            self._validate_content()
            return True
        except ValueError:
            return False
    
    def get_word_count(self) -> int:
        """Get the word count of the content."""
        return len(self.value.split())
    
    def get_character_count(self) -> int:
        """Get the character count of the content."""
        return len(self.value)
    
    def get_hashtags(self) -> List[str]:
        """Extract hashtags from content."""
        hashtag_pattern = r'#\w+'
        return re.findall(hashtag_pattern, self.value)
    
    def get_mentions(self) -> List[str]:
        """Extract mentions from content."""
        mention_pattern = r'@\w+'
        return re.findall(mention_pattern, self.value)
    
    def get_links(self) -> List[str]:
        """Extract links from content."""
        url_pattern = r'https?://[^\s]+'
        return re.findall(url_pattern, self.value)
    
    def get_readability_score(self) -> float:
        """Calculate readability score."""
        # Simple Flesch Reading Ease approximation
        words = self.get_word_count()
        sentences = len(re.split(r'[.!?]+', self.value))
        syllables = self._count_syllables()
        
        if words == 0 or sentences == 0:
            return 0.0
        
        return 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
    
    def _count_syllables(self) -> int:
        """Count syllables in content."""
        # Simplified syllable counting
        vowels = 'aeiouy'
        content = self.value.lower()
        count = 0
        
        for char in content:
            if char in vowels:
                count += 1
        
        return count
    
    def get_sentiment_score(self) -> float:
        """Calculate basic sentiment score."""
        # Simple sentiment analysis based on positive/negative words
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic'}
        negative_words = {'bad', 'terrible', 'awful', 'horrible', 'disappointing'}
        
        words = set(self.value.lower().split())
        positive_count = len(words.intersection(positive_words))
        negative_count = len(words.intersection(negative_words))
        
        total_words = len(words)
        if total_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_words
    
    def optimize_for_linkedin(self) -> str:
        """Optimize content for LinkedIn platform."""
        optimized = self.value
        
        # Add line breaks for better readability
        optimized = re.sub(r'\. ', '.\n\n', optimized)
        
        # Ensure proper hashtag formatting
        optimized = re.sub(r'#(\w+)', r'#\1', optimized)
        
        # Add call-to-action if missing
        if not re.search(r'\b(comment|share|like|follow|connect)\b', optimized.lower()):
            optimized += '\n\nWhat are your thoughts? Share in the comments below! 👇'
        
        return optimized
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "value": self.value,
            "word_count": self.get_word_count(),
            "character_count": self.get_character_count(),
            "hashtags": self.get_hashtags(),
            "mentions": self.get_mentions(),
            "links": self.get_links(),
            "readability_score": self.get_readability_score(),
            "sentiment_score": self.get_sentiment_score()
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Content':
        """Create from dictionary."""
        return cls(data["value"])
    
    def __str__(self) -> str:
        return self.value
    
    def __len__(self) -> int:
        return len(self.value) 