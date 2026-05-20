"""
PDF Variantes AI Helpers
AI processing utilities and helpers
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import re
import json

import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.config import Settings

logger = logging.getLogger(__name__)

class AIProcessor:
    """AI processing utilities"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        
        # Initialize AI clients
        self.openai_client = None
        self.anthropic_client = None
        
        # Initialize models
        self.topic_extractor = None
        self.sentiment_analyzer = None
        self.text_generator = None
        self.sentence_transformer = None
        self.tfidf_vectorizer = None
        
        # Model configurations
        self.model_configs = {
            "openai": {
                "gpt-3.5-turbo": {"max_tokens": 4096, "temperature": 0.7},
                "gpt-4": {"max_tokens": 8192, "temperature": 0.7},
                "gpt-4-turbo": {"max_tokens": 128000, "temperature": 0.7}
            },
            "anthropic": {
                "claude-3-sonnet": {"max_tokens": 200000, "temperature": 0.7},
                "claude-3-opus": {"max_tokens": 200000, "temperature": 0.7}
            }
        }
    
    async def initialize(self):
        """Initialize AI models and clients"""
        try:
            # Initialize OpenAI client
            if self.settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.settings.OPENAI_API_KEY
                )
                logger.info("OpenAI client initialized")
            
            # Initialize Anthropic client
            if self.settings.ANTHROPIC_API_KEY:
                self.anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.settings.ANTHROPIC_API_KEY
                )
                logger.info("Anthropic client initialized")
            
            # Initialize Hugging Face models
            await self._initialize_huggingface_models()
            
            logger.info("AI Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Processor: {e}")
            raise
    
    async def _initialize_huggingface_models(self):
        """Initialize Hugging Face models"""
        try:
            # Topic extraction model
            self.topic_extractor = pipeline(
                "text-classification",
                model=self.settings.TOPIC_EXTRACTION_MODEL,
                return_all_scores=True
            )
            
            # Sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.settings.SENTIMENT_MODEL,
                return_all_scores=True
            )
            
            # Text generation model
            self.text_generator = pipeline(
                "text-generation",
                model=self.settings.TEXT_GENERATION_MODEL,
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            
            # Sentence transformer for embeddings
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # TF-IDF vectorizer
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("Hugging Face models initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face models: {e}")
            # Continue without HF models if they fail
    
    async def extract_topics(self, text: str, min_relevance: float = 0.5, max_topics: int = 50) -> List[Dict[str, Any]]:
        """Extract topics from text using AI"""
        try:
            topics = []
            
            # Split text into sentences
            sentences = self._split_into_sentences(text)
            
            # Extract keywords using TF-IDF
            keywords = await self._extract_keywords_tfidf(text, max_topics)
            
            # Extract named entities
            entities = await self._extract_named_entities(text)
            
            # Extract topics using sentence embeddings
            topic_candidates = await self._extract_topic_candidates(sentences, keywords)
            
            # Score and filter topics
            for topic_data in topic_candidates:
                if topic_data["relevance_score"] >= min_relevance:
                    topic = {
                        "topic": topic_data["topic"],
                        "category": topic_data["category"],
                        "relevance_score": topic_data["relevance_score"],
                        "mentions": topic_data["mentions"],
                        "context": topic_data["context"],
                        "related_topics": topic_data["related_topics"]
                    }
                    topics.append(topic)
            
            # Sort by relevance score
            topics.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return topics[:max_topics]
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
    
    async def generate_variant_content(self, original_text: str, config: Dict[str, Any]) -> str:
        """Generate variant content using AI"""
        try:
            # Get configuration
            similarity_level = config.get("similarity_level", 0.7)
            creativity_level = config.get("creativity_level", 0.6)
            preserve_structure = config.get("preserve_structure", True)
            preserve_meaning = config.get("preserve_meaning", True)
            
            # Choose AI provider
            provider = config.get("provider", "openai")
            model = config.get("model", self.settings.DEFAULT_AI_MODEL)
            
            # Create prompt
            prompt = self._create_variant_prompt(
                original_text, similarity_level, creativity_level,
                preserve_structure, preserve_meaning
            )
            
            # Generate variant
            if provider == "openai" and self.openai_client:
                variant_text = await self._generate_with_openai(prompt, model)
            elif provider == "anthropic" and self.anthropic_client:
                variant_text = await self._generate_with_anthropic(prompt, model)
            else:
                # Fallback to Hugging Face
                variant_text = await self._generate_with_huggingface(prompt)
            
            return variant_text
            
        except Exception as e:
            logger.error(f"Error generating variant content: {e}")
            return original_text  # Return original if generation fails
    
    async def generate_brainstorm_ideas(self, text: str, number_of_ideas: int = 20, 
                                      diversity_level: float = 0.7, creativity_level: float = 0.8) -> List[Dict[str, Any]]:
        """Generate brainstorm ideas from text"""
        try:
            ideas = []
            
            # Extract key concepts
            concepts = await self._extract_key_concepts(text)
            
            # Generate ideas for each concept
            for concept in concepts:
                concept_ideas = await self._generate_concept_ideas(
                    concept, text, number_of_ideas // len(concepts),
                    diversity_level, creativity_level
                )
                ideas.extend(concept_ideas)
            
            # Add diversity by generating cross-concept ideas
            cross_ideas = await self._generate_cross_concept_ideas(
                concepts, text, number_of_ideas // 4,
                diversity_level, creativity_level
            )
            ideas.extend(cross_ideas)
            
            # Score and filter ideas
            scored_ideas = []
            for idea in ideas:
                score = self._calculate_idea_score(idea, diversity_level, creativity_level)
                if score >= 0.5:  # Minimum score threshold
                    idea["priority_score"] = score
                    scored_ideas.append(idea)
            
            # Sort by priority score
            scored_ideas.sort(key=lambda x: x["priority_score"], reverse=True)
            
            return scored_ideas[:number_of_ideas]
            
        except Exception as e:
            logger.error(f"Error generating brainstorm ideas: {e}")
            return []
    
    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        try:
            # Use sentence transformer for semantic similarity
            if self.sentence_transformer:
                embeddings1 = self.sentence_transformer.encode([text1])
                embeddings2 = self.sentence_transformer.encode([text2])
                similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
                return float(similarity)
            
            # Fallback to TF-IDF similarity
            return await self._calculate_tfidf_similarity(text1, text2)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text"""
        try:
            if self.sentiment_analyzer:
                results = self.sentiment_analyzer(text)
                
                # Process results
                sentiment_scores = {}
                for result in results[0]:
                    sentiment_scores[result["label"]] = result["score"]
                
                # Determine overall sentiment
                overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
                
                return {
                    "overall_sentiment": overall_sentiment,
                    "confidence": sentiment_scores[overall_sentiment],
                    "scores": sentiment_scores
                }
            
            return {"overall_sentiment": "neutral", "confidence": 0.5, "scores": {}}
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"overall_sentiment": "neutral", "confidence": 0.5, "scores": {}}
    
    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Summarize text using AI"""
        try:
            # Use OpenAI for summarization if available
            if self.openai_client:
                prompt = f"Summarize the following text in {max_length} characters or less:\n\n{text}"
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_length // 4,  # Rough estimate
                    temperature=0.3
                )
                
                return response.choices[0].message.content.strip()
            
            # Fallback to extractive summarization
            return await self._extractive_summarization(text, max_length)
            
        except Exception as e:
            logger.error(f"Error summarizing text: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    async def _extract_keywords_tfidf(self, text: str, max_keywords: int) -> List[str]:
        """Extract keywords using TF-IDF"""
        try:
            if not self.tfidf_vectorizer:
                return []
            
            # Fit and transform text
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get top keywords
            scores = tfidf_matrix.toarray()[0]
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [keyword for keyword, score in keyword_scores[:max_keywords]]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    async def _extract_named_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        try:
            # Simple entity extraction using regex patterns
            entities = []
            
            # Extract dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
            dates = re.findall(date_pattern, text)
            for date in dates:
                entities.append({"text": date, "type": "DATE", "label": "date"})
            
            # Extract email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            emails = re.findall(email_pattern, text)
            for email in emails:
                entities.append({"text": email, "type": "EMAIL", "label": "email"})
            
            # Extract URLs
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            urls = re.findall(url_pattern, text)
            for url in urls:
                entities.append({"text": url, "type": "URL", "label": "url"})
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting named entities: {e}")
            return []
    
    async def _extract_topic_candidates(self, sentences: List[str], keywords: List[str]) -> List[Dict[str, Any]]:
        """Extract topic candidates from sentences and keywords"""
        try:
            candidates = []
            
            # Group sentences by keywords
            for keyword in keywords:
                related_sentences = [s for s in sentences if keyword.lower() in s.lower()]
                
                if related_sentences:
                    # Calculate relevance score
                    relevance_score = min(len(related_sentences) / len(sentences), 1.0)
                    
                    # Determine category
                    category = self._determine_topic_category(keyword, related_sentences)
                    
                    candidate = {
                        "topic": keyword,
                        "category": category,
                        "relevance_score": relevance_score,
                        "mentions": len(related_sentences),
                        "context": related_sentences[:3],  # First 3 sentences
                        "related_topics": []
                    }
                    
                    candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error extracting topic candidates: {e}")
            return []
    
    def _determine_topic_category(self, keyword: str, sentences: List[str]) -> str:
        """Determine topic category based on keyword and context"""
        # Simple categorization based on keywords
        text = " ".join(sentences).lower()
        
        if any(word in text for word in ["technology", "software", "computer", "digital"]):
            return "technology"
        elif any(word in text for word in ["business", "company", "market", "sales"]):
            return "business"
        elif any(word in text for word in ["education", "learning", "school", "university"]):
            return "education"
        elif any(word in text for word in ["health", "medical", "doctor", "patient"]):
            return "health"
        else:
            return "general"
    
    def _create_variant_prompt(self, original_text: str, similarity_level: float, 
                             creativity_level: float, preserve_structure: bool, 
                             preserve_meaning: bool) -> str:
        """Create prompt for variant generation"""
        prompt = f"""Rewrite the following text with the following requirements:
- Similarity level: {similarity_level} (0=completely different, 1=very similar)
- Creativity level: {creativity_level} (0=minimal changes, 1=maximum creativity)
- Preserve structure: {preserve_structure}
- Preserve meaning: {preserve_meaning}

Original text:
{original_text}

Rewritten text:"""
        
        return prompt
    
    async def _generate_with_openai(self, prompt: str, model: str) -> str:
        """Generate text using OpenAI"""
        try:
            config = self.model_configs["openai"].get(model, {})
            
            response = await self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=config.get("max_tokens", 2000),
                temperature=config.get("temperature", 0.7)
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating with OpenAI: {e}")
            raise
    
    async def _generate_with_anthropic(self, prompt: str, model: str) -> str:
        """Generate text using Anthropic"""
        try:
            config = self.model_configs["anthropic"].get(model, {})
            
            response = await self.anthropic_client.messages.create(
                model=model,
                max_tokens=config.get("max_tokens", 2000),
                temperature=config.get("temperature", 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Error generating with Anthropic: {e}")
            raise
    
    async def _generate_with_huggingface(self, prompt: str) -> str:
        """Generate text using Hugging Face"""
        try:
            if not self.text_generator:
                raise ValueError("Hugging Face text generator not initialized")
            
            result = self.text_generator(prompt, max_length=len(prompt.split()) + 100)
            generated_text = result[0]["generated_text"]
            
            # Remove the original prompt from generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating with Hugging Face: {e}")
            raise
    
    async def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        try:
            # Extract keywords using TF-IDF
            keywords = await self._extract_keywords_tfidf(text, 10)
            
            # Extract noun phrases (simple approach)
            noun_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            
            # Combine and deduplicate
            concepts = list(set(keywords + noun_phrases))
            
            return concepts[:10]  # Return top 10 concepts
            
        except Exception as e:
            logger.error(f"Error extracting key concepts: {e}")
            return []
    
    async def _generate_concept_ideas(self, concept: str, text: str, num_ideas: int,
                                    diversity_level: float, creativity_level: float) -> List[Dict[str, Any]]:
        """Generate ideas for a specific concept"""
        try:
            ideas = []
            
            # Create prompt for concept-based ideation
            prompt = f"""Generate {num_ideas} creative ideas related to "{concept}" based on this context:
{text[:500]}...

Requirements:
- Diversity level: {diversity_level}
- Creativity level: {creativity_level}
- Focus on practical applications and innovative approaches

Ideas:"""
            
            # Generate ideas using AI
            if self.openai_client:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=creativity_level
                )
                
                generated_text = response.choices[0].message.content.strip()
                
                # Parse ideas from generated text
                idea_lines = [line.strip() for line in generated_text.split('\n') if line.strip()]
                
                for i, idea_text in enumerate(idea_lines[:num_ideas]):
                    idea = {
                        "idea": idea_text,
                        "category": self._categorize_idea(idea_text),
                        "related_topics": [concept],
                        "potential_impact": self._assess_impact(idea_text),
                        "implementation_difficulty": self._assess_difficulty(idea_text),
                        "priority_score": 0.0  # Will be calculated later
                    }
                    ideas.append(idea)
            
            return ideas
            
        except Exception as e:
            logger.error(f"Error generating concept ideas: {e}")
            return []
    
    async def _generate_cross_concept_ideas(self, concepts: List[str], text: str, num_ideas: int,
                                         diversity_level: float, creativity_level: float) -> List[Dict[str, Any]]:
        """Generate ideas combining multiple concepts"""
        try:
            ideas = []
            
            # Create combinations of concepts
            concept_combinations = []
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    concept_combinations.append((concepts[i], concepts[j]))
            
            # Generate ideas for each combination
            for concept1, concept2 in concept_combinations[:num_ideas]:
                prompt = f"""Generate a creative idea that combines "{concept1}" and "{concept2}" based on this context:
{text[:500]}...

Requirements:
- Diversity level: {diversity_level}
- Creativity level: {creativity_level}
- Focus on innovative combinations and novel applications

Idea:"""
                
                if self.openai_client:
                    response = await self.openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=creativity_level
                    )
                    
                    idea_text = response.choices[0].message.content.strip()
                    
                    idea = {
                        "idea": idea_text,
                        "category": "cross_concept",
                        "related_topics": [concept1, concept2],
                        "potential_impact": self._assess_impact(idea_text),
                        "implementation_difficulty": self._assess_difficulty(idea_text),
                        "priority_score": 0.0
                    }
                    ideas.append(idea)
            
            return ideas
            
        except Exception as e:
            logger.error(f"Error generating cross-concept ideas: {e}")
            return []
    
    def _categorize_idea(self, idea_text: str) -> str:
        """Categorize an idea based on its content"""
        text = idea_text.lower()
        
        if any(word in text for word in ["technology", "software", "app", "digital"]):
            return "technology"
        elif any(word in text for word in ["business", "market", "sales", "revenue"]):
            return "business"
        elif any(word in text for word in ["education", "learning", "training"]):
            return "education"
        elif any(word in text for word in ["health", "medical", "wellness"]):
            return "health"
        elif any(word in text for word in ["environment", "sustainability", "green"]):
            return "environment"
        else:
            return "general"
    
    def _assess_impact(self, idea_text: str) -> str:
        """Assess the potential impact of an idea"""
        text = idea_text.lower()
        
        high_impact_words = ["revolutionary", "breakthrough", "transform", "disrupt", "innovative"]
        medium_impact_words = ["improve", "enhance", "optimize", "efficient", "effective"]
        
        if any(word in text for word in high_impact_words):
            return "high"
        elif any(word in text for word in medium_impact_words):
            return "medium"
        else:
            return "low"
    
    def _assess_difficulty(self, idea_text: str) -> str:
        """Assess the implementation difficulty of an idea"""
        text = idea_text.lower()
        
        hard_words = ["complex", "advanced", "sophisticated", "challenging", "difficult"]
        easy_words = ["simple", "easy", "straightforward", "basic", "quick"]
        
        if any(word in text for word in hard_words):
            return "hard"
        elif any(word in text for word in easy_words):
            return "easy"
        else:
            return "medium"
    
    def _calculate_idea_score(self, idea: Dict[str, Any], diversity_level: float, creativity_level: float) -> float:
        """Calculate priority score for an idea"""
        try:
            # Base score from impact and difficulty
            impact_scores = {"high": 0.8, "medium": 0.5, "low": 0.2}
            difficulty_scores = {"easy": 0.8, "medium": 0.5, "hard": 0.2}
            
            impact_score = impact_scores.get(idea["potential_impact"], 0.5)
            difficulty_score = difficulty_scores.get(idea["implementation_difficulty"], 0.5)
            
            # Weighted combination
            base_score = (impact_score * 0.6) + (difficulty_score * 0.4)
            
            # Apply diversity and creativity weights
            final_score = base_score * (diversity_level * 0.3 + creativity_level * 0.7)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating idea score: {e}")
            return 0.5
    
    async def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF similarity between texts"""
        try:
            if not self.tfidf_vectorizer:
                return 0.0
            
            # Fit and transform texts
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF similarity: {e}")
            return 0.0
    
    async def _extractive_summarization(self, text: str, max_length: int) -> str:
        """Perform extractive summarization"""
        try:
            sentences = self._split_into_sentences(text)
            
            if len(sentences) <= 3:
                return text
            
            # Score sentences based on word frequency
            word_freq = {}
            for sentence in sentences:
                words = sentence.lower().split()
                for word in words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Score sentences
            sentence_scores = []
            for sentence in sentences:
                words = sentence.lower().split()
                score = sum(word_freq.get(word, 0) for word in words)
                sentence_scores.append((sentence, score))
            
            # Sort by score and select top sentences
            sentence_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select sentences until max_length is reached
            summary_sentences = []
            current_length = 0
            
            for sentence, score in sentence_scores:
                if current_length + len(sentence) <= max_length:
                    summary_sentences.append(sentence)
                    current_length += len(sentence)
                else:
                    break
            
            return ". ".join(summary_sentences) + "."
            
        except Exception as e:
            logger.error(f"Error in extractive summarization: {e}")
            return text[:max_length] + "..." if len(text) > max_length else text
    
    async def cleanup(self):
        """Cleanup AI processor resources"""
        try:
            # Clear models
            self.topic_extractor = None
            self.sentiment_analyzer = None
            self.text_generator = None
            self.sentence_transformer = None
            self.tfidf_vectorizer = None
            
            logger.info("AI Processor cleaned up successfully")
            
        except Exception as e:
            logger.error(f"Error cleaning up AI Processor: {e}")

class ContentAnalyzer:
    """Content analysis utilities"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ai_processor = AIProcessor(settings)
    
    async def analyze_content(self, text: str) -> Dict[str, Any]:
        """Comprehensive content analysis"""
        try:
            analysis = {
                "word_count": len(text.split()),
                "character_count": len(text),
                "sentence_count": len(self.ai_processor._split_into_sentences(text)),
                "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                "readability_score": await self._calculate_readability(text),
                "sentiment_analysis": await self.ai_processor.analyze_sentiment(text),
                "topics": await self.ai_processor.extract_topics(text, max_topics=10),
                "keywords": await self.ai_processor._extract_keywords_tfidf(text, 20),
                "entities": await self.ai_processor._extract_named_entities(text),
                "summary": await self.ai_processor.summarize_text(text, 200)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing content: {e}")
            return {}
    
    async def _calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        try:
            sentences = self.ai_processor._split_into_sentences(text)
            words = text.split()
            
            if not sentences or not words:
                return 0.5
            
            # Simple readability calculation
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = sum(len(word) for word in words) / len(words)
            
            # Simplified Flesch Reading Ease approximation
            readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length)
            
            # Normalize to 0-1 scale
            normalized_score = max(0, min(1, readability / 100))
            
            return normalized_score
            
        except Exception as e:
            logger.error(f"Error calculating readability: {e}")
            return 0.5
