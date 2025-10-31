#!/usr/bin/env python3
"""
Generative AI Agent v3.3
Revolutionary AI-powered content generation and optimization
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

@dataclass
class GenerativeAIConfig:
    """Configuration for Generative AI Agent"""
    # Generation parameters
    max_content_length: int = 500
    min_content_length: int = 50
    creativity_level: float = 0.8  # 0.0 to 1.0
    diversity_factor: float = 0.7
    quality_threshold: float = 0.75
    
    # Model parameters
    hidden_dim: int = 512
    num_layers: int = 6
    attention_heads: int = 8
    dropout_rate: float = 0.1
    
    # Audience targeting
    enable_audience_adaptation: bool = True
    enable_platform_optimization: bool = True
    enable_trend_integration: bool = True
    
    # A/B Testing
    enable_ab_testing: bool = True
    variant_count: int = 3
    test_duration_hours: int = 24

class ContentGenerator(nn.Module):
    """Neural network for content generation"""
    
    def __init__(self, config: GenerativeAIConfig):
        super().__init__()
        self.config = config
        self.logger = self._setup_logging()
        
        # Content generation network
        self.content_encoder = nn.Sequential(
            nn.Linear(256, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Multi-head attention for content understanding
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Content generation layers
        self.generation_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout_rate,
                activation='gelu',
                batch_first=True
            ) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Content type classifier
        self.content_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8),  # 8 content types
            nn.Softmax(dim=-1)
        )
        
        self.logger.info("🧠 Content Generator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the generator"""
        logger = logging.getLogger("ContentGenerator")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def forward(self, x, attention_mask=None):
        """Forward pass through the generation network"""
        # Encode input
        encoded = self.content_encoder(x)
        
        # Apply attention
        if attention_mask is not None:
            attended, _ = self.attention(encoded, encoded, encoded, attn_mask=attention_mask)
        else:
            attended, _ = self.attention(encoded, encoded, encoded)
        
        # Pass through generation layers
        generated = attended
        for layer in self.generation_layers:
            generated = layer(generated)
        
        # Project to output space
        output = self.output_projection(generated)
        
        # Classify content type
        content_type = self.content_classifier(output)
        
        return output, content_type

class GenerativeAIAgent:
    """Revolutionary AI agent for content generation"""
    
    def __init__(self, config: GenerativeAIConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize content generator
        self.content_generator = ContentGenerator(config)
        
        # Content templates and patterns
        self.content_templates = self._load_content_templates()
        self.engagement_patterns = self._load_engagement_patterns()
        
        # Generation history
        self.generation_history = []
        self.performance_metrics = []
        
        self.logger.info("🚀 Generative AI Agent initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the agent"""
        logger = logging.getLogger("GenerativeAIAgent")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_content_templates(self) -> Dict[str, List[str]]:
        """Load content generation templates"""
        return {
            'engagement': [
                "🚀 {topic} is absolutely {adjective}! What do you think?",
                "💡 Just discovered something amazing about {topic}!",
                "🔥 {topic} is trending and here's why you should care...",
                "✨ The future of {topic} is here and it's {adjective}!",
                "🎯 Want to know the secret behind {topic}?",
                "🌟 {topic} has changed my perspective completely!",
                "💪 Ready to revolutionize {topic}? Here's how...",
                "🎉 Celebrating the incredible impact of {topic}!"
            ],
            'viral': [
                "⚠️ SHOCKING: {topic} is not what you think!",
                "🔥 BREAKING: {topic} just went viral!",
                "🚨 ALERT: {topic} is about to explode!",
                "💥 EXPLOSIVE: {topic} revealed!",
                "⚡ FLASH: {topic} is trending worldwide!",
                "🌪️ VIRAL: {topic} is taking over!",
                "💣 BOMBSHELL: {topic} exposed!",
                "🎭 DRAMA: {topic} controversy revealed!"
            ],
            'educational': [
                "📚 Learn the truth about {topic} in 60 seconds",
                "🎓 The science behind {topic} explained simply",
                "🔬 Research shows {topic} is more complex than we thought",
                "📖 History lesson: How {topic} shaped our world",
                "🧠 Psychology behind {topic} - fascinating insights",
                "🌍 Global impact of {topic} - what you need to know",
                "⚡ Quick facts about {topic} that will surprise you",
                "🎯 Understanding {topic} - a beginner's guide"
            ],
            'inspirational': [
                "💫 {topic} reminds us that anything is possible",
                "✨ Let {topic} inspire your next breakthrough",
                "🌟 The beauty of {topic} lies in its simplicity",
                "💝 {topic} teaches us valuable life lessons",
                "🌈 Finding hope through {topic}",
                "🕊️ Peace comes from understanding {topic}",
                "🌺 The gentle power of {topic}",
                "🎭 Life is a journey, and {topic} is our guide"
            ]
        }
    
    def _load_engagement_patterns(self) -> Dict[str, Dict[str, float]]:
        """Load engagement patterns for different content types"""
        return {
            'engagement': {
                'question_marks': 1.2,
                'exclamation_marks': 1.1,
                'emojis': 1.15,
                'hashtags': 1.1,
                'call_to_action': 1.25
            },
            'viral': {
                'urgency_words': 1.3,
                'emotional_triggers': 1.4,
                'controversy': 1.35,
                'exclusivity': 1.2,
                'trending_topics': 1.25
            },
            'educational': {
                'numbers': 1.1,
                'facts': 1.15,
                'how_to': 1.2,
                'insights': 1.1,
                'expertise': 1.15
            },
            'inspirational': {
                'positive_words': 1.2,
                'metaphors': 1.1,
                'personal_stories': 1.25,
                'hope': 1.3,
                'transformation': 1.2
            }
        }
    
    def generate_content(self, topic: str, content_type: str, audience_profile: Dict[str, Any], 
                        platform: str = "facebook") -> Dict[str, Any]:
        """Generate optimized content for the given parameters"""
        try:
            self.logger.info(f"Generating {content_type} content for topic: {topic}")
            
            # Generate multiple variants for A/B testing
            variants = []
            for i in range(self.config.variant_count):
                variant = self._generate_single_variant(topic, content_type, audience_profile, platform)
                variants.append(variant)
            
            # Select best variant based on predicted performance
            best_variant = self._select_best_variant(variants, audience_profile, platform)
            
            # Generate A/B testing variants
            ab_variants = self._generate_ab_test_variants(best_variant, content_type)
            
            # Create comprehensive generation result
            result = {
                'topic': topic,
                'content_type': content_type,
                'platform': platform,
                'primary_content': best_variant,
                'ab_test_variants': ab_variants,
                'all_variants': variants,
                'predicted_performance': self._predict_performance(best_variant, audience_profile, platform),
                'generation_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'creativity_level': self.config.creativity_level,
                    'diversity_factor': self.config.diversity_factor,
                    'audience_adaptation': self.config.enable_audience_adaptation
                }
            }
            
            # Store in generation history
            self.generation_history.append(result)
            
            self.logger.info(f"Generated {len(variants)} content variants successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in content generation: {e}")
            return {'error': str(e)}
    
    def _generate_single_variant(self, topic: str, content_type: str, 
                                audience_profile: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Generate a single content variant"""
        # Get template for content type
        templates = self.content_templates.get(content_type, self.content_templates['engagement'])
        
        # Select template with creativity variation
        template = np.random.choice(templates)
        
        # Generate content using template
        content = self._fill_template(template, topic, content_type, audience_profile)
        
        # Optimize for platform
        if self.config.enable_platform_optimization:
            content = self._optimize_for_platform(content, platform)
        
        # Optimize for audience
        if self.config.enable_audience_adaptation:
            content = self._optimize_for_audience(content, audience_profile)
        
        # Add engagement elements
        content = self._add_engagement_elements(content, content_type)
        
        # Generate hashtags
        hashtags = self._generate_hashtags(topic, content_type, platform)
        
        return {
            'content': content,
            'hashtags': hashtags,
            'content_type': content_type,
            'platform': platform,
            'length': len(content),
            'engagement_score': self._calculate_engagement_score(content, content_type)
        }
    
    def _fill_template(self, template: str, topic: str, content_type: str, 
                      audience_profile: Dict[str, Any]) -> str:
        """Fill template with dynamic content"""
        # Extract audience characteristics
        age_group = audience_profile.get('age_group', 'general')
        interests = audience_profile.get('interests', [])
        engagement_style = audience_profile.get('engagement_style', 'balanced')
        
        # Generate dynamic adjectives based on audience
        adjectives = self._generate_audience_appropriate_adjectives(age_group, interests, content_type)
        
        # Fill template
        filled_content = template.format(
            topic=topic,
            adjective=np.random.choice(adjectives)
        )
        
        # Add creativity variations
        if self.config.creativity_level > 0.7:
            filled_content = self._add_creative_elements(filled_content, content_type)
        
        return filled_content
    
    def _generate_audience_appropriate_adjectives(self, age_group: str, interests: List[str], 
                                               content_type: str) -> List[str]:
        """Generate adjectives appropriate for the target audience"""
        base_adjectives = {
            'engagement': ['amazing', 'incredible', 'fantastic', 'awesome', 'brilliant'],
            'viral': ['shocking', 'explosive', 'controversial', 'trending', 'viral'],
            'educational': ['fascinating', 'intriguing', 'surprising', 'enlightening', 'revealing'],
            'inspirational': ['beautiful', 'powerful', 'transformative', 'uplifting', 'meaningful']
        }
        
        # Age-appropriate modifications
        age_modifiers = {
            'teen': ['lit', 'fire', 'sick', 'dope', 'rad'],
            'young_adult': ['epic', 'legendary', 'next_level', 'game_changing', 'revolutionary'],
            'adult': ['exceptional', 'remarkable', 'outstanding', 'extraordinary', 'notable'],
            'senior': ['valuable', 'meaningful', 'significant', 'important', 'essential']
        }
        
        # Get base adjectives for content type
        adjectives = base_adjectives.get(content_type, base_adjectives['engagement'])
        
        # Add age-appropriate modifiers
        if age_group in age_modifiers:
            adjectives.extend(age_modifiers[age_group])
        
        return adjectives
    
    def _add_creative_elements(self, content: str, content_type: str) -> str:
        """Add creative elements to content"""
        creative_elements = {
            'engagement': [
                " 🤔",
                " 💭",
                " 🎯",
                " ✨"
            ],
            'viral': [
                " 🔥",
                " ⚡",
                " 💥",
                " 🌪️"
            ],
            'educational': [
                " 📚",
                " 🧠",
                " 🔬",
                " 💡"
            ],
            'inspirational': [
                " 💫",
                " 🌟",
                " ✨",
                " 🕊️"
            ]
        }
        
        elements = creative_elements.get(content_type, creative_elements['engagement'])
        if elements and np.random.random() < 0.7:
            content += np.random.choice(elements)
        
        return content
    
    def _optimize_for_platform(self, content: str, platform: str) -> str:
        """Optimize content for specific platform"""
        platform_optimizations = {
            'facebook': {
                'max_length': 500,
                'preferred_style': 'conversational',
                'hashtag_count': 3
            },
            'instagram': {
                'max_length': 2200,
                'preferred_style': 'visual',
                'hashtag_count': 30
            },
            'twitter': {
                'max_length': 280,
                'preferred_style': 'concise',
                'hashtag_count': 2
            },
            'linkedin': {
                'max_length': 3000,
                'preferred_style': 'professional',
                'hashtag_count': 5
            }
        }
        
        optimization = platform_optimizations.get(platform, platform_optimizations['facebook'])
        
        # Adjust content length
        if len(content) > optimization['max_length']:
            content = content[:optimization['max_length'] - 3] + "..."
        
        # Adjust style based on platform
        if platform == 'linkedin' and '!' in content:
            content = content.replace('!', '.')
        
        return content
    
    def _optimize_for_audience(self, content: str, audience_profile: Dict[str, Any]) -> str:
        """Optimize content for specific audience"""
        age_group = audience_profile.get('age_group', 'general')
        interests = audience_profile.get('interests', [])
        engagement_style = audience_profile.get('engagement_style', 'balanced')
        
        # Adjust language complexity
        if age_group == 'teen':
            content = self._simplify_language(content)
        elif age_group == 'senior':
            content = self._formalize_language(content)
        
        # Add interest-specific references
        if interests and np.random.random() < 0.6:
            content = self._add_interest_references(content, interests)
        
        # Adjust engagement style
        if engagement_style == 'high':
            content = self._increase_engagement(content)
        elif engagement_style == 'low':
            content = self._decrease_engagement(content)
        
        return content
    
    def _simplify_language(self, content: str) -> str:
        """Simplify language for younger audiences"""
        # Replace complex words with simpler alternatives
        replacements = {
            'incredible': 'amazing',
            'fascinating': 'cool',
            'exceptional': 'great',
            'remarkable': 'awesome',
            'transformative': 'life-changing'
        }
        
        for complex_word, simple_word in replacements.items():
            content = content.replace(complex_word, simple_word)
        
        return content
    
    def _formalize_language(self, content: str) -> str:
        """Make language more formal for professional audiences"""
        # Replace casual words with formal alternatives
        replacements = {
            'awesome': 'exceptional',
            'cool': 'fascinating',
            'amazing': 'remarkable',
            'great': 'outstanding',
            'fire': 'excellent'
        }
        
        for casual_word, formal_word in replacements.items():
            content = content.replace(casual_word, formal_word)
        
        return content
    
    def _add_interest_references(self, content: str, interests: List[str]) -> str:
        """Add references to audience interests"""
        if not interests:
            return content
        
        interest = np.random.choice(interests)
        reference_templates = [
            f" Just like {interest}, this is game-changing!",
            f" If you love {interest}, you'll love this!",
            f" This reminds me of {interest} - absolutely incredible!",
            f" {interest} fans, this is for you!"
        ]
        
        if np.random.random() < 0.5:
            content += np.random.choice(reference_templates)
        
        return content
    
    def _increase_engagement(self, content: str) -> str:
        """Increase engagement elements in content"""
        engagement_additions = [
            " What's your take on this?",
            " Drop a comment below!",
            " Share if you agree!",
            " Tag someone who needs to see this!",
            " Double tap if you're excited!"
        ]
        
        if np.random.random() < 0.7:
            content += np.random.choice(engagement_additions)
        
        return content
    
    def _decrease_engagement(self, content: str) -> str:
        """Decrease engagement elements for more reserved audiences"""
        # Remove excessive exclamation marks
        content = content.replace('!!', '!')
        content = content.replace('!!!', '!')
        
        # Remove excessive emojis
        emoji_count = sum(1 for char in content if ord(char) > 127)
        if emoji_count > 3:
            # Keep only first few emojis
            content = content[:content.find('🚀') + 2] + content[content.find('🚀') + 2:].replace('🚀', '').replace('✨', '').replace('💫', '')
        
        return content
    
    def _add_engagement_elements(self, content: str, content_type: str) -> str:
        """Add engagement elements based on content type"""
        patterns = self.engagement_patterns.get(content_type, {})
        
        # Add question marks for engagement
        if patterns.get('question_marks', 0) > 1.0 and '?' not in content:
            content += " What do you think?"
        
        # Add exclamation marks for viral content
        if content_type == 'viral' and patterns.get('exclamation_marks', 0) > 1.0:
            content = content.replace('.', '!')
        
        # Add call to action
        if patterns.get('call_to_action', 0) > 1.0:
            cta_options = [
                " Share your thoughts below!",
                " What's your experience with this?",
                " Drop a comment if you agree!",
                " Tag someone who needs to see this!"
            ]
            content += np.random.choice(cta_options)
        
        return content
    
    def _generate_hashtags(self, topic: str, content_type: str, platform: str) -> List[str]:
        """Generate relevant hashtags for content"""
        # Base hashtags from topic
        topic_words = topic.lower().split()
        base_hashtags = [f"#{word}" for word in topic_words if len(word) > 3]
        
        # Content type specific hashtags
        type_hashtags = {
            'engagement': ['#Engagement', '#SocialMedia', '#Community'],
            'viral': ['#Viral', '#Trending', '#MustSee', '#Breaking'],
            'educational': ['#Learning', '#Education', '#Knowledge', '#Insights'],
            'inspirational': ['#Inspiration', '#Motivation', '#Positive', '#Growth']
        }
        
        # Platform specific hashtags
        platform_hashtags = {
            'facebook': ['#Facebook', '#SocialMedia'],
            'instagram': ['#Instagram', '#InstaGood', '#Photography'],
            'twitter': ['#Twitter', '#Tweeting'],
            'linkedin': ['#LinkedIn', '#Professional', '#Networking']
        }
        
        # Combine hashtags
        all_hashtags = base_hashtags + type_hashtags.get(content_type, []) + platform_hashtags.get(platform, [])
        
        # Limit hashtags based on platform
        max_hashtags = {
            'facebook': 5,
            'instagram': 30,
            'twitter': 3,
            'linkedin': 5
        }
        
        return all_hashtags[:max_hashtags.get(platform, 5)]
    
    def _generate_ab_test_variants(self, primary_content: Dict[str, Any], content_type: str) -> List[Dict[str, Any]]:
        """Generate A/B testing variants"""
        if not self.config.enable_ab_testing:
            return []
        
        variants = []
        base_content = primary_content['content']
        
        # Variant 1: Different tone
        tone_variant = self._create_tone_variant(base_content, content_type)
        variants.append({
            'variant_type': 'tone',
            'content': tone_variant,
            'hashtags': primary_content['hashtags'],
            'modification': 'Changed tone/style'
        })
        
        # Variant 2: Different length
        length_variant = self._create_length_variant(base_content, content_type)
        variants.append({
            'variant_type': 'length',
            'content': length_variant,
            'hashtags': primary_content['hashtags'],
            'modification': 'Modified length'
        })
        
        # Variant 3: Different engagement approach
        engagement_variant = self._create_engagement_variant(base_content, content_type)
        variants.append({
            'variant_type': 'engagement',
            'content': engagement_variant,
            'hashtags': primary_content['hashtags'],
            'modification': 'Changed engagement approach'
        })
        
        return variants
    
    def _create_tone_variant(self, content: str, content_type: str) -> str:
        """Create variant with different tone"""
        if content_type == 'viral':
            # Make it more subtle
            content = content.replace('SHOCKING', 'Interesting')
            content = content.replace('BREAKING', 'Update')
            content = content.replace('EXPLOSIVE', 'Important')
        elif content_type == 'inspirational':
            # Make it more casual
            content = content.replace('beautiful', 'awesome')
            content = content.replace('powerful', 'amazing')
            content = content.replace('transformative', 'life-changing')
        
        return content
    
    def _create_length_variant(self, content: str, content_type: str) -> str:
        """Create variant with different length"""
        if len(content) > 200:
            # Make it shorter
            sentences = content.split('.')
            if len(sentences) > 2:
                content = '. '.join(sentences[:2]) + '.'
        else:
            # Make it longer
            content += " This is something that really matters and deserves your attention."
        
        return content
    
    def _create_engagement_variant(self, content: str, content_type: str) -> str:
        """Create variant with different engagement approach"""
        if '?' in content:
            # Replace question with statement
            content = content.replace('?', '.')
            content = content.replace('What do you think?', 'This is worth considering.')
        else:
            # Add question
            content += " What's your take on this?"
        
        return content
    
    def _select_best_variant(self, variants: List[Dict[str, Any]], 
                            audience_profile: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Select the best variant based on predicted performance"""
        # Calculate performance score for each variant
        for variant in variants:
            variant['performance_score'] = self._predict_performance(variant, audience_profile, platform)
        
        # Sort by performance score
        variants.sort(key=lambda x: x['performance_score'], reverse=True)
        
        return variants[0]
    
    def _predict_performance(self, content: Dict[str, Any], audience_profile: Dict[str, Any], 
                           platform: str) -> float:
        """Predict content performance score"""
        base_score = 0.5
        
        # Content length optimization
        optimal_lengths = {
            'facebook': 150,
            'instagram': 200,
            'twitter': 200,
            'linkedin': 300
        }
        
        optimal_length = optimal_lengths.get(platform, 150)
        length_diff = abs(len(content['content']) - optimal_length)
        length_score = max(0, 1 - (length_diff / optimal_length))
        
        # Engagement elements score
        engagement_score = content.get('engagement_score', 0.5)
        
        # Audience match score
        audience_score = self._calculate_audience_match(content, audience_profile)
        
        # Platform optimization score
        platform_score = self._calculate_platform_optimization(content, platform)
        
        # Combine scores
        final_score = (base_score + length_score + engagement_score + audience_score + platform_score) / 5
        
        return min(1.0, max(0.0, final_score))
    
    def _calculate_audience_match(self, content: Dict[str, Any], audience_profile: Dict[str, Any]) -> float:
        """Calculate how well content matches audience profile"""
        score = 0.5
        
        age_group = audience_profile.get('age_group', 'general')
        interests = audience_profile.get('interests', [])
        
        # Age appropriateness
        if age_group == 'teen' and any(word in content['content'].lower() for word in ['lit', 'fire', 'sick']):
            score += 0.2
        elif age_group == 'senior' and any(word in content['content'].lower() for word in ['valuable', 'meaningful', 'important']):
            score += 0.2
        
        # Interest relevance
        if interests:
            interest_matches = sum(1 for interest in interests if interest.lower() in content['content'].lower())
            score += min(0.3, interest_matches * 0.1)
        
        return min(1.0, score)
    
    def _calculate_platform_optimization(self, content: Dict[str, Any], platform: str) -> float:
        """Calculate platform optimization score"""
        score = 0.5
        
        # Length optimization
        if platform == 'twitter' and len(content['content']) <= 280:
            score += 0.2
        elif platform == 'instagram' and len(content['content']) <= 2200:
            score += 0.2
        elif platform == 'facebook' and len(content['content']) <= 500:
            score += 0.2
        elif platform == 'linkedin' and len(content['content']) <= 3000:
            score += 0.2
        
        # Hashtag optimization
        optimal_hashtags = {
            'facebook': 3,
            'instagram': 30,
            'twitter': 2,
            'linkedin': 5
        }
        
        hashtag_count = len(content.get('hashtags', []))
        optimal_count = optimal_hashtags.get(platform, 3)
        
        if hashtag_count <= optimal_count:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_engagement_score(self, content: str, content_type: str) -> float:
        """Calculate engagement score for content"""
        score = 0.5
        
        # Question marks
        if '?' in content:
            score += 0.1
        
        # Exclamation marks
        exclamation_count = content.count('!')
        score += min(0.2, exclamation_count * 0.05)
        
        # Emojis
        emoji_count = sum(1 for char in content if ord(char) > 127)
        score += min(0.2, emoji_count * 0.05)
        
        # Call to action
        cta_words = ['share', 'comment', 'like', 'follow', 'tag']
        if any(word in content.lower() for word in cta_words):
            score += 0.2
        
        # Content type specific scoring
        if content_type == 'viral':
            viral_words = ['shocking', 'breaking', 'explosive', 'trending', 'viral']
            if any(word in content.lower() for word in viral_words):
                score += 0.1
        
        return min(1.0, score)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        if not self.generation_history:
            return {'total_generations': 0, 'average_performance': 0.0}
        
        total_generations = len(self.generation_history)
        performance_scores = [gen.get('predicted_performance', 0.0) for gen in self.generation_history]
        average_performance = np.mean(performance_scores) if performance_scores else 0.0
        
        return {
            'total_generations': total_generations,
            'average_performance': average_performance,
            'content_types_generated': list(set(gen.get('content_type', 'unknown') for gen in self.generation_history)),
            'platforms_targeted': list(set(gen.get('platform', 'unknown') for gen in self.generation_history)),
            'last_generation': self.generation_history[-1]['generation_metadata']['timestamp'] if self.generation_history else None
        }

# Example usage
if __name__ == "__main__":
    # Initialize Generative AI Agent
    config = GenerativeAIConfig(
        creativity_level=0.8,
        diversity_factor=0.7,
        enable_ab_testing=True,
        variant_count=3
    )
    
    agent = GenerativeAIAgent(config)
    
    # Example audience profile
    audience_profile = {
        'age_group': 'young_adult',
        'interests': ['technology', 'innovation', 'AI'],
        'engagement_style': 'high'
    }
    
    print("🚀 Generative AI Agent v3.3 initialized!")
    print("📊 Generation Stats:", agent.get_generation_stats())
    
    # Generate content
    result = agent.generate_content(
        topic="Artificial Intelligence",
        content_type="engagement",
        audience_profile=audience_profile,
        platform="facebook"
    )
    
    if 'error' not in result:
        print("✅ Content generated successfully!")
        print(f"📝 Primary Content: {result['primary_content']['content']}")
        print(f"🏷️ Hashtags: {result['primary_content']['hashtags']}")
        print(f"📊 Predicted Performance: {result['predicted_performance']:.3f}")
    else:
        print(f"❌ Error: {result['error']}")

