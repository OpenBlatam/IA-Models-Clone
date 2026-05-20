"""AI-powered suggestions system"""
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class AISuggestionEngine:
    """AI-powered suggestion engine"""
    
    def __init__(self):
        self.suggestion_rules = []
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default suggestion rules"""
        self.suggestion_rules = [
            {
                "type": "format",
                "condition": lambda data: len(data.get("tables", [])) > 3,
                "suggestion": "Consider using Excel format for better table handling",
                "priority": "high"
            },
            {
                "type": "chart",
                "condition": lambda data: len(data.get("tables", [])) > 0 and any(len(t.get("rows", [])) > 5 for t in data.get("tables", [])),
                "suggestion": "Add charts to visualize table data",
                "priority": "medium"
            },
            {
                "type": "structure",
                "condition": lambda data: len(data.get("headings", [])) < 2,
                "suggestion": "Consider adding more headings for better document structure",
                "priority": "low"
            },
            {
                "type": "images",
                "condition": lambda data: len(data.get("images", [])) == 0 and len(data.get("tables", [])) > 0,
                "suggestion": "Consider adding images or diagrams to enhance visual appeal",
                "priority": "low"
            }
        ]
    
    def analyze_content(
        self,
        parsed_content: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze content and generate suggestions
        
        Args:
            parsed_content: Parsed Markdown content
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        for rule in self.suggestion_rules:
            try:
                if rule["condition"](parsed_content):
                    suggestions.append({
                        "type": rule["type"],
                        "message": rule["suggestion"],
                        "priority": rule.get("priority", "medium"),
                        "category": rule.get("category", "general")
                    })
            except Exception as e:
                logger.error(f"Error evaluating suggestion rule: {e}")
        
        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda x: priority_order.get(x["priority"], 1))
        
        return suggestions
    
    def suggest_format(
        self,
        parsed_content: Dict[str, Any]
    ) -> Optional[str]:
        """
        Suggest best output format
        
        Args:
            parsed_content: Parsed Markdown content
            
        Returns:
            Suggested format or None
        """
        tables_count = len(parsed_content.get("tables", []))
        images_count = len(parsed_content.get("images", []))
        headings_count = len(parsed_content.get("headings", []))
        code_blocks_count = len(parsed_content.get("code_blocks", []))
        
        # Excel for many tables
        if tables_count > 3:
            return "excel"
        
        # PowerPoint for presentations (many headings)
        if headings_count > 5:
            return "pptx"
        
        # PDF for documents with images
        if images_count > 2:
            return "pdf"
        
        # HTML for interactive content
        if code_blocks_count > 0:
            return "html"
        
        # Default to PDF
        return "pdf"
    
    def suggest_improvements(
        self,
        parsed_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Suggest document improvements
        
        Args:
            parsed_content: Parsed Markdown content
            
        Returns:
            Improvement suggestions
        """
        improvements = {
            "format_suggestion": self.suggest_format(parsed_content),
            "suggestions": self.analyze_content(parsed_content),
            "score": self._calculate_quality_score(parsed_content)
        }
        
        return improvements
    
    def _calculate_quality_score(
        self,
        parsed_content: Dict[str, Any]
    ) -> float:
        """
        Calculate document quality score
        
        Args:
            parsed_content: Parsed Markdown content
            
        Returns:
            Quality score (0-100)
        """
        score = 50.0  # Base score
        
        # Add points for structure
        headings = parsed_content.get("headings", [])
        if len(headings) >= 2:
            score += 10
        if len(headings) >= 5:
            score += 10
        
        # Add points for content
        paragraphs = parsed_content.get("paragraphs", [])
        if len(paragraphs) >= 3:
            score += 10
        
        # Add points for tables
        tables = parsed_content.get("tables", [])
        if len(tables) > 0:
            score += 10
        
        # Add points for images
        images = parsed_content.get("images", [])
        if len(images) > 0:
            score += 10
        
        return min(100.0, score)
    
    def generate_summary(
        self,
        parsed_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate content summary
        
        Args:
            parsed_content: Parsed Markdown content
            
        Returns:
            Content summary
        """
        return {
            "headings_count": len(parsed_content.get("headings", [])),
            "paragraphs_count": len(parsed_content.get("paragraphs", [])),
            "tables_count": len(parsed_content.get("tables", [])),
            "images_count": len(parsed_content.get("images", [])),
            "code_blocks_count": len(parsed_content.get("code_blocks", [])),
            "links_count": len(parsed_content.get("links", [])),
            "estimated_reading_time": self._estimate_reading_time(parsed_content),
            "suggested_format": self.suggest_format(parsed_content)
        }
    
    def _estimate_reading_time(
        self,
        parsed_content: Dict[str, Any]
    ) -> int:
        """Estimate reading time in minutes"""
        total_words = 0
        
        for paragraph in parsed_content.get("paragraphs", []):
            total_words += len(paragraph.get("text", "").split())
        
        # Average reading speed: 200 words per minute
        reading_time = max(1, total_words // 200)
        
        return reading_time


# Global AI engine
_ai_engine: Optional[AISuggestionEngine] = None


def get_ai_engine() -> AISuggestionEngine:
    """Get global AI engine"""
    global _ai_engine
    if _ai_engine is None:
        _ai_engine = AISuggestionEngine()
    return _ai_engine

