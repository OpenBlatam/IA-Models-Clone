"""
Text processing utilities for blog posts
"""

import re
from typing import List
from unidecode import unidecode


def generate_slug(text: str) -> str:
    """Generate a URL-friendly slug from text."""
    # Convert to lowercase and remove accents
    text = unidecode(text.lower())
    
    # Replace spaces and special characters with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    
    # Remove leading/trailing hyphens
    return text.strip('-')


def calculate_reading_time(content: str, words_per_minute: int = 200) -> int:
    """Calculate estimated reading time in minutes."""
    # Remove HTML tags
    import re
    plain_text = re.sub(r'<[^>]+>', '', content)
    
    # Count words
    word_count = len(plain_text.split())
    
    # Calculate reading time
    reading_time = max(1, round(word_count / words_per_minute))
    
    return reading_time


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis."""
    import re
    from collections import Counter
    
    # Remove HTML tags and convert to lowercase
    plain_text = re.sub(r'<[^>]+>', '', text.lower())
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he',
        'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Extract words (alphanumeric only)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', plain_text)
    
    # Filter out stop words and count frequency
    filtered_words = [word for word in words if word not in stop_words]
    word_counts = Counter(filtered_words)
    
    # Return most common keywords
    return [word for word, count in word_counts.most_common(max_keywords)]


def clean_html(content: str) -> str:
    """Clean HTML content by removing tags and normalizing whitespace."""
    import re
    
    # Remove HTML tags
    clean_text = re.sub(r'<[^>]+>', '', content)
    
    # Normalize whitespace
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    return clean_text.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if len(text) <= max_length:
        return text
    
    # Find the last complete word within the limit
    truncated = text[:max_length - len(suffix)]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:  # If we can find a good break point
        return truncated[:last_space] + suffix
    
    return truncated + suffix


def extract_meta_description(content: str, max_length: int = 160) -> str:
    """Extract or generate meta description from content."""
    # Clean HTML
    clean_content = clean_html(content)
    
    # Try to find a good paragraph for description
    paragraphs = [p.strip() for p in clean_content.split('\n') if p.strip()]
    
    for paragraph in paragraphs:
        if 50 <= len(paragraph) <= max_length:
            return paragraph
    
    # If no good paragraph found, truncate the beginning
    return truncate_text(clean_content, max_length)


def validate_content_length(content: str, min_length: int = 100, max_length: int = 50000) -> bool:
    """Validate content length."""
    clean_content = clean_html(content)
    return min_length <= len(clean_content) <= max_length


def count_words(text: str) -> int:
    """Count words in text."""
    clean_text = clean_html(text)
    return len(clean_text.split())


def count_characters(text: str) -> int:
    """Count characters in text (excluding HTML)."""
    clean_text = clean_html(text)
    return len(clean_text)


def extract_links(text: str) -> List[str]:
    """Extract URLs from text."""
    import re
    
    # URL pattern
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    return re.findall(url_pattern, text)


def extract_images(text: str) -> List[str]:
    """Extract image URLs from HTML content."""
    import re
    
    # Image src pattern
    img_pattern = r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
    
    return re.findall(img_pattern, text, re.IGNORECASE)






























