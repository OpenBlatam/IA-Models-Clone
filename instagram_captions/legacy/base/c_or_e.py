from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from collections import Counter
from datetime import datetime
from models import CaptionStyle, InstagramTarget, ContentType, HashtagStrategy
from typing import Any, List, Dict, Optional
"""
Instagram Captions Core Engine.

Consolidated system for high-quality Instagram caption generation with
integrated optimization, hashtag intelligence, and GMT scheduling.
"""



logger = logging.getLogger(__name__)

# ==================== QUALITY SYSTEM ====================

class QualityGrade(str, Enum):
    """Quality grading system."""
    EXCELLENT = "A+"    # 95-100%
    VERY_GOOD = "A"     # 90-94%
    GOOD = "B"          # 80-89%
    AVERAGE = "C"       # 70-79%
    POOR = "D"          # 60-69%
    FAILING = "F"       # <60%

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics."""
    overall_score: float
    hook_strength: float
    engagement_potential: float
    readability: float
    cta_effectiveness: float
    specificity: float
    grade: QualityGrade
    issues: List[str]
    suggestions: List[str]

class QualityAnalyzer:
    """Analyze and score caption quality."""
    
    def __init__(self) -> Any:
        self.hook_patterns = [
            r'^(What if|Did you know|Here\'s|Ready to|Want to|Ever wondered)',
            r'^(This changed|Stop doing|Never again|The truth about)',
            r'^[0-9]+ (things|ways|tips|secrets|reasons)',
            r'^\w+:', r'\?', r'^(Plot twist|Real talk|Truth bomb)'
        ]
        
        self.engagement_words = [
            'you', 'your', 'we', 'us', 'how', 'why', 'what', '?', '!',
            'comment', 'share', 'tag', 'save', 'follow', 'thoughts'
        ]
        
        self.weak_words = {
            'very', 'really', 'quite', 'pretty', 'somewhat', 'maybe',
            'thing', 'stuff', 'nice', 'good', 'bad', 'amazing'
        }
    
    def analyze(self, caption: str) -> QualityMetrics:
        """Analyze caption quality comprehensively."""
        
        # Individual metrics
        hook_strength = self._analyze_hook(caption)
        engagement_potential = self._analyze_engagement(caption)
        readability = self._analyze_readability(caption)
        cta_effectiveness = self._analyze_cta(caption)
        specificity = self._analyze_specificity(caption)
        
        # Overall score (weighted average)
        overall_score = (
            hook_strength * 0.25 +
            engagement_potential * 0.25 +
            readability * 0.20 +
            cta_effectiveness * 0.15 +
            specificity * 0.15
        )
        
        # Grade calculation
        grade = self._calculate_grade(overall_score)
        
        # Issues and suggestions
        issues, suggestions = self._generate_feedback(
            caption, hook_strength, engagement_potential, cta_effectiveness, specificity
        )
        
        return QualityMetrics(
            overall_score=overall_score * 100,
            hook_strength=hook_strength * 100,
            engagement_potential=engagement_potential * 100,
            readability=readability * 100,
            cta_effectiveness=cta_effectiveness * 100,
            specificity=specificity * 100,
            grade=grade,
            issues=issues,
            suggestions=suggestions
        )
    
    def _analyze_hook(self, caption: str) -> float:
        """Analyze hook effectiveness."""
        first_line = caption.split('\n')[0] if '\n' in caption else caption[:80]
        score = 0.2
        
        # Pattern matching
        for pattern in self.hook_patterns:
            if re.search(pattern, first_line, re.IGNORECASE):
                score += 0.4
                break
        
        # Engagement elements
        if '?' in first_line or '!' in first_line:
            score += 0.2
        if any(word in first_line.lower() for word in ['you', 'your']):
            score += 0.2
        if re.search(r'\d+', first_line):
            score += 0.1
        
        return min(1.0, score)
    
    def _analyze_engagement(self, caption: str) -> float:
        """Analyze engagement potential."""
        caption_lower = caption.lower()
        score = 0.2
        
        # Count engagement triggers
        engagement_count = sum(1 for word in self.engagement_words if word in caption_lower)
        score += min(0.4, engagement_count * 0.04)
        
        # Questions boost engagement
        question_count = caption.count('?')
        score += min(0.2, question_count * 0.1)
        
        # Personal pronouns
        personal_count = caption_lower.count('you') + caption_lower.count('your')
        score += min(0.2, personal_count * 0.05)
        
        return min(1.0, score)
    
    def _analyze_readability(self, caption: str) -> float:
        """Analyze readability for mobile."""
        words = caption.split()
        if not words:
            return 0.0
        
        sentences = caption.count('.') + caption.count('!') + caption.count('?') + 1
        avg_words_per_sentence = len(words) / sentences
        
        # Optimal: 10-20 words per sentence
        sentence_score = max(0, 1 - abs(avg_words_per_sentence - 15) / 20)
        
        # Line breaks for mobile
        line_breaks = caption.count('\n')
        if len(caption) > 200:
            structure_score = min(1.0, line_breaks / (len(caption) / 150))
        else:
            structure_score = 0.8
        
        return (sentence_score + structure_score) / 2
    
    def _analyze_cta(self, caption: str) -> float:
        """Analyze call-to-action effectiveness."""
        caption_lower = caption.lower()
        score = 0.0
        
        # CTA words
        cta_words = ['comment', 'share', 'tag', 'save', 'follow', 'dm', 'click', 'try']
        cta_count = sum(1 for word in cta_words if word in caption_lower)
        score += min(0.6, cta_count * 0.2)
        
        # Questions as implicit CTA
        if '?' in caption:
            score += 0.3
        
        # Direction words
        if any(word in caption_lower for word in ['below', 'comments', 'bio', 'stories']):
            score += 0.2
        
        return min(1.0, score)
    
    def _analyze_specificity(self, caption: str) -> float:
        """Analyze content specificity."""
        score = 0.5
        
        # Numbers and data
        numbers = re.findall(r'\d+', caption)
        score += min(0.2, len(numbers) * 0.05)
        
        # Specific indicators
        specific_words = ['example', 'specifically', 'exactly', 'step', 'method']
        specific_count = sum(1 for word in specific_words if word in caption.lower())
        score += min(0.2, specific_count * 0.1)
        
        # Weak word penalty
        weak_count = sum(1 for word in self.weak_words if word in caption.lower().split())
        score -= min(0.3, weak_count * 0.05)
        
        return max(0.0, min(1.0, score))
    
    def _calculate_grade(self, score: float) -> QualityGrade:
        """Calculate grade from score."""
        percentage = score * 100
        if percentage >= 95:
            return QualityGrade.EXCELLENT
        elif percentage >= 90:
            return QualityGrade.VERY_GOOD
        elif percentage >= 80:
            return QualityGrade.GOOD
        elif percentage >= 70:
            return QualityGrade.AVERAGE
        elif percentage >= 60:
            return QualityGrade.POOR
        else:
            return QualityGrade.FAILING
    
    def _generate_feedback(self, caption: str, hook: float, engagement: float, 
                          cta: float, specificity: float) -> Tuple[List[str], List[str]]:
        """Generate issues and suggestions."""
        issues = []
        suggestions = []
        
        if hook < 0.5:
            issues.append("Weak opening hook")
            suggestions.append("Start with a question, bold statement, or intriguing fact")
        
        if engagement < 0.5:
            issues.append("Low engagement potential")
            suggestions.append("Use more 'you/your' and ask questions to involve audience")
        
        if cta < 0.3:
            issues.append("Missing or weak call-to-action")
            suggestions.append("End with clear request: 'What's your experience? Share below!'")
        
        if specificity < 0.4:
            issues.append("Too generic or vague")
            suggestions.append("Add specific examples, numbers, or detailed steps")
        
        return issues, suggestions

# ==================== HASHTAG INTELLIGENCE ====================

@dataclass
class HashtagData:
    """Hashtag performance data."""
    tag: str
    popularity: float        # 0-1
    competition: float       # 0-1
    relevance: float        # 0-1
    estimated_reach: int

class HashtagIntelligence:
    """Intelligent hashtag generation and optimization."""
    
    def __init__(self) -> Any:
        self.hashtag_db = self._build_hashtag_database()
        self.audience_hashtags = self._build_audience_hashtags()
        self.style_hashtags = self._build_style_hashtags()
    
    def _build_hashtag_database(self) -> Dict[str, HashtagData]:
        """Build optimized hashtag database."""
        return {
            # High-performance general
            "love": HashtagData("love", 0.95, 0.9, 0.7, 2000000000),
            "instagood": HashtagData("instagood", 0.92, 0.88, 0.75, 1800000000),
            "photooftheday": HashtagData("photooftheday", 0.88, 0.85, 0.7, 900000000),
            
            # Business & Professional
            "entrepreneur": HashtagData("entrepreneur", 0.7, 0.6, 0.9, 50000000),
            "business": HashtagData("business", 0.75, 0.65, 0.85, 80000000),
            "success": HashtagData("success", 0.72, 0.62, 0.88, 60000000),
            "leadership": HashtagData("leadership", 0.65, 0.55, 0.9, 30000000),
            
            # Lifestyle & Personal
            "lifestyle": HashtagData("lifestyle", 0.8, 0.75, 0.8, 120000000),
            "wellness": HashtagData("wellness", 0.72, 0.67, 0.88, 60000000),
            "selfcare": HashtagData("selfcare", 0.75, 0.7, 0.85, 80000000),
            
            # Tech & Innovation
            "technology": HashtagData("technology", 0.68, 0.58, 0.85, 40000000),
            "innovation": HashtagData("innovation", 0.62, 0.52, 0.9, 25000000),
            "ai": HashtagData("ai", 0.72, 0.7, 0.88, 35000000),
            
            # Creative & Content
            "creative": HashtagData("creative", 0.75, 0.7, 0.85, 70000000),
            "contentcreator": HashtagData("contentcreator", 0.68, 0.63, 0.9, 40000000),
            "art": HashtagData("art", 0.82, 0.78, 0.8, 200000000),
        }
    
    def _build_audience_hashtags(self) -> Dict[InstagramTarget, List[str]]:
        """Audience-specific hashtags."""
        return {
            InstagramTarget.GEN_Z: ["authentic", "real", "vibe", "aesthetic", "mindful"],
            InstagramTarget.MILLENNIALS: ["growth", "balance", "journey", "experience", "lifestyle"],
            InstagramTarget.BUSINESS: ["strategy", "success", "leadership", "growth", "professional"],
            InstagramTarget.CREATORS: ["creative", "content", "behindthescenes", "process", "community"],
            InstagramTarget.LIFESTYLE: ["wellness", "selfcare", "mindfulness", "inspiration", "goals"]
        }
    
    def _build_style_hashtags(self) -> Dict[CaptionStyle, List[str]]:
        """Style-specific hashtags."""
        return {
            CaptionStyle.CASUAL: ["daily", "real", "authentic", "relatable"],
            CaptionStyle.PROFESSIONAL: ["tips", "strategy", "insights", "professional"],
            CaptionStyle.PLAYFUL: ["fun", "creative", "playful", "colorful"],
            CaptionStyle.INSPIRATIONAL: ["motivation", "inspiration", "mindset", "growth"],
            CaptionStyle.EDUCATIONAL: ["learn", "tips", "knowledge", "educational"]
        }
    
    def generate_hashtags(self, content_keywords: List[str], audience: InstagramTarget,
                         style: CaptionStyle, strategy: HashtagStrategy, count: int = 20) -> List[str]:
        """Generate optimized hashtags."""
        
        hashtags = []
        
        # Add content-based hashtags
        hashtags.extend(content_keywords[:5])
        
        # Add audience-specific hashtags
        audience_tags = self.audience_hashtags.get(audience, [])
        hashtags.extend(audience_tags[:4])
        
        # Add style-specific hashtags
        style_tags = self.style_hashtags.get(style, [])
        hashtags.extend(style_tags[:3])
        
        # Add high-performance general hashtags
        general_tags = ["love", "instagood", "photooftheday", "beautiful"]
        hashtags.extend(general_tags[:3])
        
        # Add trending hashtags
        trending = ["trending", "viral", "explore", "fyp", "reels"]
        hashtags.extend(trending[:2])
        
        # Strategic mixing based on strategy
        if strategy == HashtagStrategy.TRENDING:
            trending_extra = ["viralcontent", "explorepage", "trendingaudio"]
            hashtags.extend(trending_extra[:3])
        elif strategy == HashtagStrategy.NICHE:
            # Focus on specific niche hashtags
            niche_tags = [tag for tag in self.hashtag_db.keys() if self.hashtag_db[tag].relevance > 0.8]
            hashtags.extend(niche_tags[:5])
        
        # Remove duplicates and format
        unique_hashtags = list(set(hashtags))
        formatted_hashtags = [f"#{tag.lower().replace(' ', '').replace('-', '')}" 
                             for tag in unique_hashtags]
        
        return formatted_hashtags[:count]

# ==================== CONTENT OPTIMIZER ====================

class ContentOptimizer:
    """Optimize content for maximum engagement."""
    
    def __init__(self) -> Any:
        self.quality_analyzer = QualityAnalyzer()
        self.hashtag_intelligence = HashtagIntelligence()
        
        self.style_enhancers = {
            CaptionStyle.CASUAL: {
                "tone": "friendly, conversational, relatable",
                "hooks": ["Quick question:", "Real talk:", "Honestly,"],
                "avoid": "formal language, corporate speak"
            },
            CaptionStyle.PROFESSIONAL: {
                "tone": "authoritative yet approachable",
                "hooks": ["Key insight:", "Here's the truth:", "Industry secret:"],
                "avoid": "overly casual slang"
            },
            CaptionStyle.INSPIRATIONAL: {
                "tone": "uplifting, empowering, motivational",
                "hooks": ["Truth bomb:", "Life lesson:", "Mindset shift:"],
                "avoid": "negative language"
            }
        }
        
        self.word_improvements = {
            'good': 'excellent', 'bad': 'disappointing', 'big': 'massive',
            'very': 'extremely', 'really': 'genuinely', 'nice': 'fantastic'
        }
        
        self.cta_templates = [
            "What are your thoughts? Drop them below! 👇",
            "Tag someone who needs to see this! 🏷️",
            "Save this for later reference! 📌",
            "Share your experience in the comments! 💬"
        ]
    
    def create_enhanced_prompt(self, content_desc: str, style: CaptionStyle, 
                             audience: InstagramTarget, brand_context: Optional[Dict] = None) -> str:
        """Create optimized prompt for AI generation."""
        
        style_guide = self.style_enhancers.get(style, {})
        
        prompt = f"""You are a world-class Instagram content strategist. Create a high-converting caption.

CONTENT: {content_desc}
TARGET: {audience.value} audience
STYLE: {style.value} - {style_guide.get('tone', 'engaging')}

QUALITY REQUIREMENTS:
🎯 HOOK: Start with attention-grabbing first line
💡 VALUE: Provide specific, actionable insight
❤️ EMOTION: Create genuine emotional connection  
📱 MOBILE: Optimize for mobile with line breaks
🔥 CTA: Include compelling call-to-action

STRUCTURE:
1. Hook (stop the scroll)
2. Value/Story (specific content)
3. Connection (relatable element)
4. Action (clear CTA)

OPTIMIZATION:
- Use "you/your" for direct connection
- Include questions to encourage responses
- Add specific examples/numbers
- Create curiosity gaps
- Mobile-friendly formatting

Generate a caption that drives real engagement, not just views."""

        if brand_context:
            prompt += f"\n\nBRAND: {brand_context.get('voice', '')} voice"
        
        return prompt
    
    async def optimize_caption(self, caption: str, style: CaptionStyle, 
                             audience: InstagramTarget) -> Tuple[str, QualityMetrics]:
        """Analyze and optimize caption."""
        
        # Analyze current quality
        metrics = self.quality_analyzer.analyze(caption)
        
        # Optimize if needed (grade B or below)
        if metrics.overall_score < 80:
            optimized_caption = await self._enhance_caption(caption, metrics, style)
            
            # Re-analyze optimized version
            optimized_metrics = self.quality_analyzer.analyze(optimized_caption)
            return optimized_caption, optimized_metrics
        
        return caption, metrics
    
    async def _enhance_caption(self, caption: str, metrics: QualityMetrics, 
                             style: CaptionStyle) -> str:
        """Enhance caption based on quality analysis."""
        enhanced = caption
        
        # Fix weak hook
        if metrics.hook_strength < 60:
            enhanced = self._enhance_hook(enhanced, style)
        
        # Add CTA if missing
        if metrics.cta_effectiveness < 40:
            enhanced = self._add_cta(enhanced)
        
        # Improve readability
        if metrics.readability < 60:
            enhanced = self._improve_formatting(enhanced)
        
        # Strengthen language
        enhanced = self._strengthen_language(enhanced)
        
        return enhanced
    
    def _enhance_hook(self, caption: str, style: CaptionStyle) -> str:
        """Enhance opening hook."""
        lines = caption.split('\n')
        first_line = lines[0] if lines else caption[:100]
        
        style_guide = self.style_enhancers.get(style, {})
        hooks = style_guide.get('hooks', ["Here's something interesting:"])
        
        # Check if hook needs enhancement
        if not any(pattern in first_line.lower() for pattern in ['what', 'how', 'why', 'here\'s']):
            enhanced_hook = f"{hooks[0]} {first_line}"
            if lines:
                lines[0] = enhanced_hook
                return '\n'.join(lines)
            else:
                return enhanced_hook
        
        return caption
    
    def _add_cta(self, caption: str) -> str:
        """Add effective call-to-action."""
        if not any(word in caption.lower() for word in ['comment', 'tag', 'share', 'save']):
            return f"{caption}\n\n{self.cta_templates[0]}"
        return caption
    
    def _improve_formatting(self, caption: str) -> str:
        """Improve mobile formatting."""
        if len(caption) > 200 and caption.count('\n') < 2:
            sentences = re.split(r'([.!?]\s+)', caption)
            formatted = ""
            char_count = 0
            
            for sentence in sentences:
                formatted += sentence
                char_count += len(sentence)
                
                if char_count > 150 and sentence.strip().endswith(('.', '!', '?')):
                    formatted += '\n'
                    char_count = 0
            
            return formatted.strip()
        return caption
    
    def _strengthen_language(self, caption: str) -> str:
        """Replace weak words with stronger alternatives."""
        words = caption.split()
        enhanced_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in self.word_improvements:
                replacement = self.word_improvements[clean_word]
                if word[0].isupper():
                    replacement = replacement.capitalize()
                punctuation = ''.join(c for c in word if not c.isalnum())
                enhanced_words.append(replacement + punctuation)
            else:
                enhanced_words.append(word)
        
        return ' '.join(enhanced_words)

# ==================== MAIN ENGINE ====================

class InstagramCaptionsEngine:
    """Main engine integrating all systems."""
    
    def __init__(self) -> Any:
        self.optimizer = ContentOptimizer()
        self.quality_analyzer = QualityAnalyzer()
        self.hashtag_intelligence = HashtagIntelligence()
    
    def create_optimized_prompt(self, content_desc: str, style: CaptionStyle, 
                              audience: InstagramTarget, brand_context: Optional[Dict] = None) -> str:
        """Create enhanced prompt for AI generation."""
        return self.optimizer.create_enhanced_prompt(content_desc, style, audience, brand_context)
    
    async def optimize_content(self, caption: str, style: CaptionStyle, 
                             audience: InstagramTarget) -> Tuple[str, QualityMetrics]:
        """Optimize caption content."""
        return await self.optimizer.optimize_caption(caption, style, audience)
    
    def generate_hashtags(self, content_keywords: List[str], audience: InstagramTarget,
                         style: CaptionStyle, strategy: HashtagStrategy, count: int = 20) -> List[str]:
        """Generate intelligent hashtags."""
        return self.hashtag_intelligence.generate_hashtags(
            content_keywords, audience, style, strategy, count
        )
    
    def analyze_quality(self, caption: str) -> QualityMetrics:
        """Analyze caption quality."""
        return self.quality_analyzer.analyze(caption)
    
    def get_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        return {
            "overall_score": round(metrics.overall_score, 1),
            "grade": metrics.grade.value,
            "metrics": {
                "hook_strength": round(metrics.hook_strength, 1),
                "engagement_potential": round(metrics.engagement_potential, 1),
                "readability": round(metrics.readability, 1),
                "cta_effectiveness": round(metrics.cta_effectiveness, 1),
                "specificity": round(metrics.specificity, 1)
            },
            "issues": metrics.issues,
            "suggestions": metrics.suggestions,
            "performance_expectation": self._get_performance_expectation(metrics.grade)
        }
    
    def _get_performance_expectation(self, grade: QualityGrade) -> str:
        """Get performance expectation based on grade."""
        expectations = {
            QualityGrade.EXCELLENT: "Outstanding engagement potential, viral possibility",
            QualityGrade.VERY_GOOD: "Excellent engagement expected, strong performance",
            QualityGrade.GOOD: "Good engagement likely, solid performance",
            QualityGrade.AVERAGE: "Moderate engagement, room for improvement",
            QualityGrade.POOR: "Below average performance, needs optimization",
            QualityGrade.FAILING: "Poor performance expected, requires significant enhancement"
        }
        return expectations.get(grade, "Performance unknown") 