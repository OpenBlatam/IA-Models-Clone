"""
AI-powered content generation and analysis service
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import openai
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

from ..config.settings import get_settings
from ..core.exceptions import ExternalServiceError
from ..utils.text_processing import extract_keywords, calculate_reading_time


class AIService:
    """Service for AI-powered content generation and analysis."""
    
    def __init__(self):
        self.settings = get_settings()
        self.openai_client = None
        self.sentiment_analyzer = None
        self.text_classifier = None
        self.summarizer = None
        self.embeddings_model = None
        self.tokenizer = None
        
        # Initialize AI models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models and services."""
        try:
            # Initialize OpenAI client
            if self.settings.openai_api_key:
                openai.api_key = self.settings.openai_api_key
                self.openai_client = openai
            
            # Initialize Hugging Face models
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            self.text_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium"
            )
            
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            # Initialize embeddings model
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.model_name)
            self.embeddings_model = AutoModel.from_pretrained(self.settings.model_name)
            
        except Exception as e:
            print(f"Warning: Could not initialize some AI models: {e}")
    
    async def generate_blog_post(
        self,
        topic: str,
        style: str = "informative",
        length: str = "medium",
        tone: str = "professional"
    ) -> Dict[str, Any]:
        """Generate a blog post using AI."""
        try:
            if not self.openai_client:
                raise ExternalServiceError(
                    "OpenAI API not configured",
                    service_name="openai"
                )
            
            # Create enhanced prompt
            prompt = self._create_generation_prompt(topic, style, length, tone)
            
            # Generate content
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Process generated content
            processed_content = await self._process_generated_content(content, topic)
            
            return processed_content
            
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to generate blog post: {str(e)}",
                service_name="openai"
            )
    
    async def analyze_content_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze content sentiment."""
        try:
            # Truncate content if too long
            max_length = 512
            if len(content) > max_length:
                content = content[:max_length]
            
            # Analyze sentiment
            result = await asyncio.to_thread(
                self.sentiment_analyzer,
                content
            )
            
            sentiment_data = result[0]
            
            return {
                "label": sentiment_data["label"],
                "score": sentiment_data["score"],
                "confidence": abs(sentiment_data["score"] - 0.5) * 2
            }
            
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to analyze sentiment: {str(e)}",
                service_name="sentiment_analysis"
            )
    
    async def classify_content(self, content: str) -> Dict[str, Any]:
        """Classify content into categories."""
        try:
            # Truncate content if too long
            max_length = 512
            if len(content) > max_length:
                content = content[:max_length]
            
            # Classify content
            result = await asyncio.to_thread(
                self.text_classifier,
                content
            )
            
            return {
                "category": result[0]["label"],
                "confidence": result[0]["score"]
            }
            
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to classify content: {str(e)}",
                service_name="text_classification"
            )
    
    async def summarize_content(self, content: str, max_length: int = 150) -> str:
        """Generate a summary of the content."""
        try:
            # Truncate content if too long
            max_input_length = 1024
            if len(content) > max_input_length:
                content = content[:max_input_length]
            
            # Generate summary
            result = await asyncio.to_thread(
                self.summarizer,
                content,
                max_length=max_length,
                min_length=50,
                do_sample=False
            )
            
            return result[0]["summary_text"]
            
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to summarize content: {str(e)}",
                service_name="summarization"
            )
    
    async def generate_embeddings(self, text: str) -> List[float]:
        """Generate embeddings for text."""
        try:
            # Tokenize text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.settings.max_sequence_length,
                padding=True
            )
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.embeddings_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return embeddings.tolist()
            
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to generate embeddings: {str(e)}",
                service_name="embeddings"
            )
    
    async def find_similar_posts(
        self,
        content: str,
        existing_posts: List[Dict[str, Any]],
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find similar posts based on content similarity."""
        try:
            if not existing_posts:
                return []
            
            # Generate embeddings for input content
            input_embeddings = await self.generate_embeddings(content)
            
            # Calculate similarities
            similar_posts = []
            for post in existing_posts:
                if "embeddings" in post and post["embeddings"]:
                    # Calculate cosine similarity
                    similarity = cosine_similarity(
                        [input_embeddings],
                        [post["embeddings"]]
                    )[0][0]
                    
                    if similarity >= threshold:
                        similar_posts.append({
                            "post": post,
                            "similarity": float(similarity)
                        })
            
            # Sort by similarity
            similar_posts.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_posts[:5]  # Return top 5 similar posts
            
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to find similar posts: {str(e)}",
                service_name="similarity_search"
            )
    
    async def suggest_tags(self, content: str, existing_tags: List[str] = None) -> List[str]:
        """Suggest relevant tags for content."""
        try:
            # Extract keywords using traditional NLP
            keywords = extract_keywords(content, max_keywords=10)
            
            # Use AI to suggest additional tags
            if self.openai_client:
                prompt = f"""
                Based on the following content, suggest 5-10 relevant tags for a blog post.
                Focus on specific, actionable tags that would help with SEO and categorization.
                
                Content: {content[:500]}
                
                Existing tags: {existing_tags or []}
                
                Return only the tags, separated by commas.
                """
                
                response = await asyncio.to_thread(
                    self.openai_client.ChatCompletion.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                
                ai_tags = [tag.strip() for tag in response.choices[0].message.content.split(",")]
                keywords.extend(ai_tags)
            
            # Remove duplicates and return unique tags
            unique_tags = list(set(keywords))
            return unique_tags[:10]  # Return top 10 tags
            
        except Exception as e:
            # Fallback to traditional keyword extraction
            return extract_keywords(content, max_keywords=10)
    
    async def optimize_content_for_seo(
        self,
        title: str,
        content: str,
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """Optimize content for SEO."""
        try:
            if not self.openai_client:
                return self._basic_seo_optimization(title, content, target_keywords)
            
            prompt = f"""
            Optimize the following blog post for SEO:
            
            Title: {title}
            Content: {content[:1000]}
            Target Keywords: {', '.join(target_keywords)}
            
            Provide:
            1. An optimized title (max 60 characters)
            2. An optimized meta description (max 160 characters)
            3. 3-5 additional SEO keywords
            4. Suggestions for improving the content structure
            5. Internal linking suggestions
            
            Format as JSON.
            """
            
            response = await asyncio.to_thread(
                self.openai_client.ChatCompletion.create,
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3
            )
            
            # Parse JSON response
            import json
            try:
                seo_data = json.loads(response.choices[0].message.content)
                return seo_data
            except json.JSONDecodeError:
                return self._basic_seo_optimization(title, content, target_keywords)
            
        except Exception as e:
            return self._basic_seo_optimization(title, content, target_keywords)
    
    async def detect_plagiarism(
        self,
        content: str,
        existing_posts: List[Dict[str, Any]],
        threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Detect potential plagiarism in content."""
        try:
            similar_posts = await self.find_similar_posts(content, existing_posts, threshold)
            
            if similar_posts:
                return {
                    "is_plagiarized": True,
                    "similarity_score": similar_posts[0]["similarity"],
                    "similar_posts": [
                        {
                            "id": post["post"]["id"],
                            "title": post["post"]["title"],
                            "similarity": post["similarity"]
                        }
                        for post in similar_posts
                    ]
                }
            else:
                return {
                    "is_plagiarized": False,
                    "similarity_score": 0.0,
                    "similar_posts": []
                }
                
        except Exception as e:
            raise ExternalServiceError(
                f"Failed to detect plagiarism: {str(e)}",
                service_name="plagiarism_detection"
            )
    
    def _create_generation_prompt(
        self,
        topic: str,
        style: str,
        length: str,
        tone: str
    ) -> str:
        """Create a prompt for content generation."""
        length_map = {
            "short": "300-500 words",
            "medium": "800-1200 words",
            "long": "1500-2000 words"
        }
        
        return f"""
        Write a {style} blog post about {topic}.
        
        Requirements:
        - Length: {length_map.get(length, '800-1200 words')}
        - Tone: {tone}
        - Include a compelling title
        - Structure with clear headings
        - Include an engaging introduction
        - Provide valuable insights
        - End with a strong conclusion
        - Include a call-to-action
        
        Format the response as:
        Title: [Your title here]
        Content: [Your content here]
        Excerpt: [Brief excerpt]
        Tags: [Comma-separated tags]
        """
    
    async def _process_generated_content(self, content: str, topic: str) -> Dict[str, Any]:
        """Process AI-generated content."""
        lines = content.split('\n')
        
        # Extract components
        title = ""
        main_content = ""
        excerpt = ""
        tags = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
                current_section = 'title'
            elif line.startswith('Content:'):
                main_content = line.replace('Content:', '').strip()
                current_section = 'content'
            elif line.startswith('Excerpt:'):
                excerpt = line.replace('Excerpt:', '').strip()
                current_section = 'excerpt'
            elif line.startswith('Tags:'):
                tags = [tag.strip() for tag in line.replace('Tags:', '').split(',')]
                current_section = 'tags'
            elif line and current_section == 'content':
                main_content += '\n' + line
        
        # Generate additional metadata
        word_count = len(main_content.split())
        reading_time = calculate_reading_time(main_content)
        
        return {
            "title": title,
            "content": main_content,
            "excerpt": excerpt or await self.summarize_content(main_content, 150),
            "tags": tags,
            "word_count": word_count,
            "reading_time_minutes": reading_time,
            "category": await self.classify_content(main_content),
            "sentiment": await self.analyze_content_sentiment(main_content)
        }
    
    def _basic_seo_optimization(
        self,
        title: str,
        content: str,
        target_keywords: List[str]
    ) -> Dict[str, Any]:
        """Basic SEO optimization without AI."""
        # Extract keywords from content
        content_keywords = extract_keywords(content, max_keywords=10)
        
        # Combine with target keywords
        all_keywords = list(set(target_keywords + content_keywords))
        
        # Generate basic meta description
        meta_description = content[:150] + "..." if len(content) > 150 else content
        
        return {
            "optimized_title": title[:60] if len(title) <= 60 else title[:57] + "...",
            "meta_description": meta_description,
            "seo_keywords": all_keywords[:5],
            "suggestions": [
                "Add more internal links",
                "Include target keywords in headings",
                "Optimize images with alt text"
            ]
        }






























