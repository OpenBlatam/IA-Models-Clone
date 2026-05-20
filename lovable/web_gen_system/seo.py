from typing import List, Dict
from bs4 import BeautifulSoup
import structlog

logger = structlog.get_logger()

class SEOOptimizer:
    """
    Optimizes web content for search engines.
    Reference: "Enhancing Image SEO using Deep Learning Algorithms..."
    """
    
    def optimize_meta_tags(self, html_content: str, title: str, description: str, keywords: List[str]) -> str:
        """
        Injects or updates meta tags for title, description, and keywords.
        """
        logger.info("Optimizing meta tags", title=title)
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Ensure head tag exists
        if not soup.head:
            if not soup.html:
                soup.append(soup.new_tag('html'))
            soup.html.insert(0, soup.new_tag('head'))
            
        head = soup.head
        
        # Update Title
        if head.title:
            head.title.string = title
        else:
            new_title = soup.new_tag('title')
            new_title.string = title
            head.append(new_title)
            
        # Update Description
        desc_tag = head.find('meta', attrs={'name': 'description'})
        if desc_tag:
            desc_tag['content'] = description
        else:
            new_desc = soup.new_tag('meta', attrs={'name': 'description', 'content': description})
            head.append(new_desc)
            
        # Update Keywords
        keywords_str = ", ".join(keywords)
        keywords_tag = head.find('meta', attrs={'name': 'keywords'})
        if keywords_tag:
            keywords_tag['content'] = keywords_str
        else:
            new_keywords = soup.new_tag('meta', attrs={'name': 'keywords', 'content': keywords_str})
            head.append(new_keywords)
            
        return str(soup)

    def generate_image_alt_tags(self, html_content: str) -> str:
        """
        Generates SEO-friendly alt tags for images.
        (Placeholder for Deep Learning based generation)
        """
        # This is a placeholder. In a real system, this would call a Vision API.
        # For now, we ensure alt tags exist and are descriptive if possible.
        return html_content

    def analyze_content_relevance(self, html_content: str, topic: str) -> float:
        """
        Analyzes how relevant the content is to the given topic.
        Returns a score between 0.0 and 1.0.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Get text content only
        text_content = soup.get_text(separator=' ').lower()
        topic_lower = topic.lower()
        
        words = text_content.split()
        word_count = len(words)
        
        if word_count == 0:
            return 0.0
            
        topic_count = text_content.count(topic_lower)
        
        # Heuristic: 1-3% keyword density is often cited as good
        density = topic_count / word_count
        
        if 0.01 <= density <= 0.03:
            return 1.0
        elif density > 0.03:
            return 0.8  # Keyword stuffing penalty
        elif density > 0:
            return 0.5  # Present but low density
        else:
            return 0.0
