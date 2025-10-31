"""
Gamma App - Advanced AI Chatbot Service
Intelligent chatbot with memory, context awareness, and multi-modal capabilities
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import numpy as np
from pathlib import Path
import sqlite3
import redis
from collections import defaultdict, deque
import openai
import anthropic
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
import cv2
import base64
from PIL import Image
import io
import speech_recognition as sr
import pyttsx3
import pydub
from pydub import AudioSegment
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy

logger = logging.getLogger(__name__)

class ChatbotMode(Enum):
    """Chatbot interaction modes"""
    TEXT = "text"
    VOICE = "voice"
    MULTIMODAL = "multimodal"
    ASSISTANT = "assistant"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"

class MemoryType(Enum):
    """Types of memory"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

class IntentType(Enum):
    """Types of user intents"""
    GREETING = "greeting"
    QUESTION = "question"
    REQUEST = "request"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    GOODBYE = "goodbye"
    UNKNOWN = "unknown"

@dataclass
class ChatMessage:
    """Chat message structure"""
    message_id: str
    user_id: str
    content: str
    message_type: str = "text"  # text, image, audio, video
    timestamp: datetime = None
    metadata: Dict[str, Any] = None

@dataclass
class ChatbotResponse:
    """Chatbot response structure"""
    response_id: str
    content: str
    response_type: str = "text"
    confidence: float = 0.0
    intent: IntentType = IntentType.UNKNOWN
    entities: List[Dict[str, Any]] = None
    suggestions: List[str] = None
    actions: List[Dict[str, Any]] = None
    memory_updates: List[Dict[str, Any]] = None
    timestamp: datetime = None

@dataclass
class UserProfile:
    """User profile for personalization"""
    user_id: str
    name: str
    preferences: Dict[str, Any]
    conversation_history: List[str]
    personality_traits: Dict[str, float]
    interests: List[str]
    created_at: datetime
    last_updated: datetime

@dataclass
class ConversationContext:
    """Conversation context"""
    session_id: str
    user_id: str
    current_topic: str
    conversation_flow: List[str]
    active_memories: List[Dict[str, Any]]
    emotional_state: str
    user_satisfaction: float
    context_window: int = 10

class AdvancedAIChatbot:
    """Advanced AI Chatbot with memory and context awareness"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.db_path = config.get("database_path", "chatbot.db")
        self.redis_client = None
        self.conversations = {}
        self.user_profiles = {}
        self.memory_graph = nx.DiGraph()
        self.intent_classifier = None
        self.sentiment_analyzer = None
        self.nlp_model = None
        self.voice_recognizer = None
        self.tts_engine = None
        
        # Initialize components
        self._init_database()
        self._init_redis()
        self._init_nlp_models()
        self._init_voice_components()
        self._init_ai_models()
    
    def _init_database(self):
        """Initialize chatbot database"""
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create conversations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    messages TEXT NOT NULL,
                    context TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create user profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    preferences TEXT,
                    conversation_history TEXT,
                    personality_traits TEXT,
                    interests TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance_score REAL DEFAULT 0.5,
                    access_count INTEGER DEFAULT 0,
                    last_accessed DATETIME,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create intents table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS intents (
                    intent_id TEXT PRIMARY KEY,
                    intent_type TEXT NOT NULL,
                    patterns TEXT NOT NULL,
                    responses TEXT NOT NULL,
                    confidence_threshold REAL DEFAULT 0.7,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
        
        logger.info("Chatbot database initialized")
    
    def _init_redis(self):
        """Initialize Redis for caching"""
        try:
            redis_url = self.config.get("redis_url", "redis://localhost:6379")
            self.redis_client = redis.from_url(redis_url)
            self.redis_client.ping()
            logger.info("Redis client initialized for chatbot")
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}")
    
    def _init_nlp_models(self):
        """Initialize NLP models"""
        try:
            # Initialize sentiment analyzer
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize spaCy model
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found, using basic tokenization")
                self.nlp_model = None
            
            # Initialize TF-IDF vectorizer for similarity
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            logger.info("NLP models initialized")
        except Exception as e:
            logger.error(f"NLP models initialization failed: {e}")
    
    def _init_voice_components(self):
        """Initialize voice recognition and synthesis"""
        try:
            # Initialize speech recognition
            self.voice_recognizer = sr.Recognizer()
            
            # Initialize text-to-speech
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_engine.setProperty('volume', 0.8)
            
            logger.info("Voice components initialized")
        except Exception as e:
            logger.warning(f"Voice components initialization failed: {e}")
    
    def _init_ai_models(self):
        """Initialize AI models"""
        try:
            # Initialize OpenAI
            if self.config.get("openai_api_key"):
                openai.api_key = self.config["openai_api_key"]
            
            # Initialize Anthropic
            if self.config.get("anthropic_api_key"):
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.config["anthropic_api_key"]
                )
            
            # Initialize transformers pipeline for intent classification
            self.intent_classifier = pipeline(
                "text-classification",
                model="microsoft/DialoGPT-medium",
                return_all_scores=True
            )
            
            logger.info("AI models initialized")
        except Exception as e:
            logger.error(f"AI models initialization failed: {e}")
    
    async def process_message(
        self,
        user_id: str,
        message: str,
        message_type: str = "text",
        session_id: Optional[str] = None,
        mode: ChatbotMode = ChatbotMode.TEXT
    ) -> ChatbotResponse:
        """Process user message and generate response"""
        
        # Generate session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Create message object
        chat_message = ChatMessage(
            message_id=str(uuid.uuid4()),
            user_id=user_id,
            content=message,
            message_type=message_type,
            timestamp=datetime.now()
        )
        
        # Get or create conversation context
        context = await self._get_conversation_context(user_id, session_id)
        
        # Update context with new message
        context.conversation_flow.append(message)
        if len(context.conversation_flow) > context.context_window:
            context.conversation_flow = context.conversation_flow[-context.context_window:]
        
        # Analyze message
        intent = await self._classify_intent(message)
        entities = await self._extract_entities(message)
        sentiment = await self._analyze_sentiment(message)
        
        # Update user profile
        await self._update_user_profile(user_id, message, intent, sentiment)
        
        # Retrieve relevant memories
        relevant_memories = await self._retrieve_memories(user_id, message, context)
        
        # Generate response based on mode
        if mode == ChatbotMode.TEXT:
            response = await self._generate_text_response(
                message, intent, entities, context, relevant_memories
            )
        elif mode == ChatbotMode.VOICE:
            response = await self._generate_voice_response(
                message, intent, entities, context, relevant_memories
            )
        elif mode == ChatbotMode.MULTIMODAL:
            response = await self._generate_multimodal_response(
                message, intent, entities, context, relevant_memories
            )
        elif mode == ChatbotMode.ASSISTANT:
            response = await self._generate_assistant_response(
                message, intent, entities, context, relevant_memories
            )
        elif mode == ChatbotMode.CREATIVE:
            response = await self._generate_creative_response(
                message, intent, entities, context, relevant_memories
            )
        elif mode == ChatbotMode.ANALYTICAL:
            response = await self._generate_analytical_response(
                message, intent, entities, context, relevant_memories
            )
        else:
            response = await self._generate_text_response(
                message, intent, entities, context, relevant_memories
            )
        
        # Update memories
        memory_updates = await self._update_memories(
            user_id, message, response, intent, sentiment
        )
        response.memory_updates = memory_updates
        
        # Store conversation
        await self._store_conversation(session_id, user_id, chat_message, response)
        
        # Update context
        context.active_memories = relevant_memories
        context.current_topic = await self._extract_topic(message)
        context.emotional_state = sentiment
        self.conversations[session_id] = context
        
        return response
    
    async def _classify_intent(self, message: str) -> IntentType:
        """Classify user intent"""
        
        try:
            # Simple rule-based classification
            message_lower = message.lower()
            
            if any(word in message_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
                return IntentType.GREETING
            elif any(word in message_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
                return IntentType.QUESTION
            elif any(word in message_lower for word in ['please', 'can you', 'could you', 'help me']):
                return IntentType.REQUEST
            elif any(word in message_lower for word in ['bad', 'terrible', 'awful', 'hate', 'disappointed']):
                return IntentType.COMPLAINT
            elif any(word in message_lower for word in ['good', 'great', 'excellent', 'love', 'amazing']):
                return IntentType.COMPLIMENT
            elif any(word in message_lower for word in ['bye', 'goodbye', 'see you', 'farewell']):
                return IntentType.GOODBYE
            else:
                return IntentType.UNKNOWN
                
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return IntentType.UNKNOWN
    
    async def _extract_entities(self, message: str) -> List[Dict[str, Any]]:
        """Extract entities from message"""
        
        entities = []
        
        try:
            if self.nlp_model:
                doc = self.nlp_model(message)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.8
                    })
            else:
                # Basic entity extraction
                words = message.split()
                for i, word in enumerate(words):
                    if word.istitle() and len(word) > 2:
                        entities.append({
                            "text": word,
                            "label": "PERSON",
                            "start": message.find(word),
                            "end": message.find(word) + len(word),
                            "confidence": 0.6
                        })
        
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    async def _analyze_sentiment(self, message: str) -> Dict[str, float]:
        """Analyze sentiment of message"""
        
        try:
            if self.sentiment_analyzer:
                scores = self.sentiment_analyzer.polarity_scores(message)
                return {
                    "positive": scores["pos"],
                    "negative": scores["neg"],
                    "neutral": scores["neu"],
                    "compound": scores["compound"]
                }
            else:
                return {"positive": 0.5, "negative": 0.0, "neutral": 0.5, "compound": 0.0}
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"positive": 0.5, "negative": 0.0, "neutral": 0.5, "compound": 0.0}
    
    async def _generate_text_response(
        self,
        message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext,
        memories: List[Dict[str, Any]]
    ) -> ChatbotResponse:
        """Generate text response"""
        
        try:
            # Prepare context for AI model
            conversation_history = "\n".join(context.conversation_flow[-5:])
            memory_context = "\n".join([m["content"] for m in memories[:3]])
            
            # Create prompt
            prompt = f"""
            You are Gamma, an advanced AI assistant for content generation.
            
            Conversation History:
            {conversation_history}
            
            Relevant Memories:
            {memory_context}
            
            User Message: {message}
            Intent: {intent.value}
            
            Generate a helpful, engaging response that:
            1. Addresses the user's intent
            2. Uses relevant context from memories
            3. Maintains conversation flow
            4. Offers assistance with content generation when appropriate
            
            Response:
            """
            
            # Generate response using OpenAI
            if self.config.get("openai_api_key"):
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are Gamma, an AI assistant for content generation."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.7
                )
                content = response.choices[0].message.content
            else:
                # Fallback response
                content = await self._generate_fallback_response(intent, message)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(intent, entities, context)
            
            # Generate actions
            actions = await self._generate_actions(intent, entities, context)
            
            return ChatbotResponse(
                response_id=str(uuid.uuid4()),
                content=content,
                response_type="text",
                confidence=0.8,
                intent=intent,
                entities=entities,
                suggestions=suggestions,
                actions=actions,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Text response generation failed: {e}")
            return ChatbotResponse(
                response_id=str(uuid.uuid4()),
                content="I apologize, but I'm having trouble processing your message right now. Please try again.",
                response_type="text",
                confidence=0.3,
                intent=intent,
                timestamp=datetime.now()
            )
    
    async def _generate_voice_response(
        self,
        message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext,
        memories: List[Dict[str, Any]]
    ) -> ChatbotResponse:
        """Generate voice response"""
        
        # Generate text response first
        text_response = await self._generate_text_response(
            message, intent, entities, context, memories
        )
        
        # Convert to speech
        try:
            if self.tts_engine:
                # Generate audio file
                audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
                self.tts_engine.save_to_file(text_response.content, audio_path)
                self.tts_engine.runAndWait()
                
                # Read audio file
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up
                Path(audio_path).unlink()
                
                text_response.response_type = "voice"
                text_response.metadata = {"audio_data": base64.b64encode(audio_data).decode()}
            
        except Exception as e:
            logger.error(f"Voice synthesis failed: {e}")
        
        return text_response
    
    async def _generate_multimodal_response(
        self,
        message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext,
        memories: List[Dict[str, Any]]
    ) -> ChatbotResponse:
        """Generate multimodal response"""
        
        # Generate text response
        text_response = await self._generate_text_response(
            message, intent, entities, context, memories
        )
        
        # Add visual elements if appropriate
        if intent == IntentType.QUESTION and any(word in message.lower() for word in ['chart', 'graph', 'visual', 'image']):
            # Generate chart or image
            visual_data = await self._generate_visual_content(message, entities)
            if visual_data:
                text_response.metadata = text_response.metadata or {}
                text_response.metadata["visual_data"] = visual_data
                text_response.response_type = "multimodal"
        
        return text_response
    
    async def _generate_assistant_response(
        self,
        message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext,
        memories: List[Dict[str, Any]]
    ) -> ChatbotResponse:
        """Generate assistant response focused on productivity"""
        
        # Check if user needs help with content generation
        if any(word in message.lower() for word in ['create', 'generate', 'make', 'write', 'presentation', 'document']):
            return await self._generate_content_assistance_response(message, intent, entities, context, memories)
        
        # Default to text response
        return await self._generate_text_response(message, intent, entities, context, memories)
    
    async def _generate_creative_response(
        self,
        message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext,
        memories: List[Dict[str, Any]]
    ) -> ChatbotResponse:
        """Generate creative response"""
        
        try:
            # Create creative prompt
            prompt = f"""
            You are Gamma, a creative AI assistant. Be imaginative, inspiring, and creative in your response.
            
            User Message: {message}
            
            Generate a creative, engaging response that:
            1. Shows creativity and imagination
            2. Inspires the user
            3. Offers creative solutions
            4. Uses metaphors and storytelling when appropriate
            
            Response:
            """
            
            if self.config.get("openai_api_key"):
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a creative AI assistant. Be imaginative and inspiring."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.9
                )
                content = response.choices[0].message.content
            else:
                content = f"🌟 That's a fascinating idea! Let me help you explore the creative possibilities of '{message}'. What if we approached this from a completely different angle?"
            
            return ChatbotResponse(
                response_id=str(uuid.uuid4()),
                content=content,
                response_type="text",
                confidence=0.8,
                intent=intent,
                suggestions=await self._generate_creative_suggestions(message),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Creative response generation failed: {e}")
            return await self._generate_text_response(message, intent, entities, context, memories)
    
    async def _generate_analytical_response(
        self,
        message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext,
        memories: List[Dict[str, Any]]
    ) -> ChatbotResponse:
        """Generate analytical response"""
        
        try:
            # Analyze the message for analytical insights
            analysis = await self._analyze_message_analytically(message, entities, context)
            
            # Create analytical prompt
            prompt = f"""
            You are Gamma, an analytical AI assistant. Provide data-driven insights and analysis.
            
            User Message: {message}
            Analysis: {analysis}
            
            Generate an analytical response that:
            1. Provides data-driven insights
            2. Breaks down complex topics
            3. Offers evidence-based recommendations
            4. Uses logical reasoning
            
            Response:
            """
            
            if self.config.get("openai_api_key"):
                response = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are an analytical AI assistant. Provide data-driven insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=300,
                    temperature=0.3
                )
                content = response.choices[0].message.content
            else:
                content = f"📊 Let me analyze '{message}' from a data-driven perspective. Based on the available information, here are the key insights..."
            
            return ChatbotResponse(
                response_id=str(uuid.uuid4()),
                content=content,
                response_type="text",
                confidence=0.9,
                intent=intent,
                metadata={"analysis": analysis},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Analytical response generation failed: {e}")
            return await self._generate_text_response(message, intent, entities, context, memories)
    
    async def _generate_content_assistance_response(
        self,
        message: str,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext,
        memories: List[Dict[str, Any]]
    ) -> ChatbotResponse:
        """Generate content assistance response"""
        
        # Identify content type
        content_type = "document"
        if "presentation" in message.lower():
            content_type = "presentation"
        elif "web" in message.lower() or "page" in message.lower():
            content_type = "webpage"
        elif "blog" in message.lower():
            content_type = "blog"
        
        # Generate assistance response
        content = f"""
        🎯 I'd be happy to help you create {content_type} content!
        
        Based on your request, I can assist you with:
        • Generating {content_type} structure and content
        • Suggesting topics and themes
        • Creating outlines and templates
        • Optimizing for your target audience
        
        What specific aspect of {content_type} creation would you like help with?
        """
        
        suggestions = [
            f"Create a {content_type} outline",
            f"Generate {content_type} content",
            f"Suggest {content_type} topics",
            f"Optimize {content_type} for SEO"
        ]
        
        actions = [
            {
                "type": "generate_content",
                "content_type": content_type,
                "parameters": {"topic": message}
            }
        ]
        
        return ChatbotResponse(
            response_id=str(uuid.uuid4()),
            content=content,
            response_type="text",
            confidence=0.9,
            intent=intent,
            suggestions=suggestions,
            actions=actions,
            timestamp=datetime.now()
        )
    
    async def _generate_fallback_response(self, intent: IntentType, message: str) -> str:
        """Generate fallback response when AI models are unavailable"""
        
        responses = {
            IntentType.GREETING: "Hello! I'm Gamma, your AI assistant for content generation. How can I help you today?",
            IntentType.QUESTION: f"That's an interesting question about '{message}'. Let me help you find the information you need.",
            IntentType.REQUEST: "I'd be happy to help you with that request. Could you provide more details?",
            IntentType.COMPLAINT: "I understand your concern. Let me help you resolve this issue.",
            IntentType.COMPLIMENT: "Thank you for the kind words! I'm here to help you create amazing content.",
            IntentType.GOODBYE: "Goodbye! Feel free to come back anytime you need help with content creation.",
            IntentType.UNKNOWN: "I'm not sure I understand. Could you rephrase that or ask me something else?"
        }
        
        return responses.get(intent, responses[IntentType.UNKNOWN])
    
    async def _generate_suggestions(
        self,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext
    ) -> List[str]:
        """Generate response suggestions"""
        
        suggestions = []
        
        if intent == IntentType.GREETING:
            suggestions = [
                "Help me create a presentation",
                "Generate content for my blog",
                "Create a document template",
                "Show me your capabilities"
            ]
        elif intent == IntentType.QUESTION:
            suggestions = [
                "Tell me more about that",
                "Can you explain in detail?",
                "What are the benefits?",
                "How does that work?"
            ]
        elif intent == IntentType.REQUEST:
            suggestions = [
                "Create content for me",
                "Generate a template",
                "Help me organize ideas",
                "Suggest improvements"
            ]
        
        return suggestions[:4]  # Limit to 4 suggestions
    
    async def _generate_actions(
        self,
        intent: IntentType,
        entities: List[Dict[str, Any]],
        context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """Generate actionable items"""
        
        actions = []
        
        if intent == IntentType.REQUEST:
            actions.append({
                "type": "assist",
                "description": "Provide assistance with the request",
                "priority": "high"
            })
        
        if any(entity["label"] == "PERSON" for entity in entities):
            actions.append({
                "type": "remember_person",
                "description": "Remember the person mentioned",
                "priority": "medium"
            })
        
        return actions
    
    async def _retrieve_memories(
        self,
        user_id: str,
        message: str,
        context: ConversationContext
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories"""
        
        try:
            # Get user memories from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT memory_id, content, importance_score, memory_type
                    FROM memories 
                    WHERE user_id = ? 
                    ORDER BY importance_score DESC, last_accessed DESC
                    LIMIT 10
                """, (user_id,))
                memories = cursor.fetchall()
            
            # Filter relevant memories using similarity
            relevant_memories = []
            message_words = set(message.lower().split())
            
            for memory in memories:
                memory_id, content, importance_score, memory_type = memory
                memory_words = set(content.lower().split())
                
                # Calculate similarity
                similarity = len(message_words.intersection(memory_words)) / len(message_words.union(memory_words))
                
                if similarity > 0.1:  # Threshold for relevance
                    relevant_memories.append({
                        "memory_id": memory_id,
                        "content": content,
                        "importance_score": importance_score,
                        "memory_type": memory_type,
                        "similarity": similarity
                    })
            
            # Sort by relevance
            relevant_memories.sort(key=lambda x: x["similarity"], reverse=True)
            
            return relevant_memories[:5]  # Return top 5 relevant memories
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []
    
    async def _update_memories(
        self,
        user_id: str,
        message: str,
        response: ChatbotResponse,
        intent: IntentType,
        sentiment: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Update user memories"""
        
        memory_updates = []
        
        try:
            # Determine if this conversation should be remembered
            importance_score = 0.5
            
            # Increase importance for certain intents
            if intent in [IntentType.REQUEST, IntentType.COMPLAINT]:
                importance_score += 0.2
            
            # Increase importance for strong emotions
            if abs(sentiment["compound"]) > 0.5:
                importance_score += 0.2
            
            # Store important memories
            if importance_score > 0.6:
                memory_id = str(uuid.uuid4())
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO memories
                        (memory_id, user_id, memory_type, content, importance_score, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        memory_id,
                        user_id,
                        MemoryType.EPISODIC.value,
                        f"User: {message}\nAssistant: {response.content}",
                        importance_score,
                        datetime.now().isoformat()
                    ))
                    conn.commit()
                
                memory_updates.append({
                    "memory_id": memory_id,
                    "action": "created",
                    "importance_score": importance_score
                })
        
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
        
        return memory_updates
    
    async def _get_conversation_context(
        self,
        user_id: str,
        session_id: str
    ) -> ConversationContext:
        """Get or create conversation context"""
        
        if session_id in self.conversations:
            return self.conversations[session_id]
        
        # Create new context
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            current_topic="",
            conversation_flow=[],
            active_memories=[],
            emotional_state="neutral",
            user_satisfaction=0.5
        )
        
        self.conversations[session_id] = context
        return context
    
    async def _update_user_profile(
        self,
        user_id: str,
        message: str,
        intent: IntentType,
        sentiment: Dict[str, float]
    ):
        """Update user profile based on interaction"""
        
        try:
            # Get existing profile
            profile = self.user_profiles.get(user_id)
            
            if not profile:
                # Create new profile
                profile = UserProfile(
                    user_id=user_id,
                    name="User",
                    preferences={},
                    conversation_history=[],
                    personality_traits={},
                    interests=[],
                    created_at=datetime.now(),
                    last_updated=datetime.now()
                )
            
            # Update conversation history
            profile.conversation_history.append(message)
            if len(profile.conversation_history) > 100:
                profile.conversation_history = profile.conversation_history[-100:]
            
            # Update personality traits based on sentiment
            if "emotional_stability" not in profile.personality_traits:
                profile.personality_traits["emotional_stability"] = 0.5
            
            # Adjust based on sentiment
            if sentiment["compound"] > 0.3:
                profile.personality_traits["emotional_stability"] += 0.01
            elif sentiment["compound"] < -0.3:
                profile.personality_traits["emotional_stability"] -= 0.01
            
            # Clamp values
            profile.personality_traits["emotional_stability"] = max(0, min(1, profile.personality_traits["emotional_stability"]))
            
            # Update interests based on message content
            words = message.lower().split()
            for word in words:
                if len(word) > 4 and word not in profile.interests:
                    if len(profile.interests) < 20:  # Limit interests
                        profile.interests.append(word)
            
            profile.last_updated = datetime.now()
            self.user_profiles[user_id] = profile
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO user_profiles
                    (user_id, name, preferences, conversation_history, personality_traits, interests, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    profile.name,
                    json.dumps(profile.preferences),
                    json.dumps(profile.conversation_history),
                    json.dumps(profile.personality_traits),
                    json.dumps(profile.interests),
                    profile.last_updated.isoformat()
                ))
                conn.commit()
        
        except Exception as e:
            logger.error(f"User profile update failed: {e}")
    
    async def _extract_topic(self, message: str) -> str:
        """Extract main topic from message"""
        
        try:
            if self.nlp_model:
                doc = self.nlp_model(message)
                # Get noun phrases
                topics = [chunk.text for chunk in doc.noun_chunks]
                if topics:
                    return topics[0]
            
            # Fallback: use first few words
            words = message.split()[:3]
            return " ".join(words)
            
        except Exception as e:
            logger.error(f"Topic extraction failed: {e}")
            return "general"
    
    async def _analyze_message_analytically(
        self,
        message: str,
        entities: List[Dict[str, Any]],
        context: ConversationContext
    ) -> Dict[str, Any]:
        """Analyze message for analytical insights"""
        
        analysis = {
            "word_count": len(message.split()),
            "sentence_count": len(sent_tokenize(message)),
            "entity_count": len(entities),
            "complexity_score": 0.5,
            "topics": [],
            "sentiment": await self._analyze_sentiment(message)
        }
        
        # Calculate complexity score
        if analysis["word_count"] > 20:
            analysis["complexity_score"] += 0.2
        if analysis["sentence_count"] > 3:
            analysis["complexity_score"] += 0.2
        if analysis["entity_count"] > 2:
            analysis["complexity_score"] += 0.1
        
        # Extract topics
        if self.nlp_model:
            doc = self.nlp_model(message)
            analysis["topics"] = [chunk.text for chunk in doc.noun_chunks]
        
        return analysis
    
    async def _generate_visual_content(
        self,
        message: str,
        entities: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Generate visual content for multimodal response"""
        
        # This would integrate with image generation services
        # For now, return placeholder
        return {
            "type": "chart",
            "description": f"Visual representation of: {message}",
            "data": {"placeholder": True}
        }
    
    async def _generate_creative_suggestions(self, message: str) -> List[str]:
        """Generate creative suggestions"""
        
        return [
            "Explore this idea further",
            "Try a different approach",
            "Combine with other concepts",
            "Create a story around this"
        ]
    
    async def _store_conversation(
        self,
        session_id: str,
        user_id: str,
        message: ChatMessage,
        response: ChatbotResponse
    ):
        """Store conversation in database"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get existing conversation
                cursor.execute("""
                    SELECT messages FROM conversations 
                    WHERE conversation_id = ?
                """, (session_id,))
                row = cursor.fetchone()
                
                if row:
                    # Update existing conversation
                    messages = json.loads(row[0])
                    messages.append({
                        "message": asdict(message),
                        "response": asdict(response)
                    })
                    
                    cursor.execute("""
                        UPDATE conversations 
                        SET messages = ?, updated_at = ?
                        WHERE conversation_id = ?
                    """, (json.dumps(messages), datetime.now().isoformat(), session_id))
                else:
                    # Create new conversation
                    messages = [{
                        "message": asdict(message),
                        "response": asdict(response)
                    }]
                    
                    cursor.execute("""
                        INSERT INTO conversations
                        (conversation_id, user_id, session_id, messages, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        session_id,
                        user_id,
                        session_id,
                        json.dumps(messages),
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))
                
                conn.commit()
        
        except Exception as e:
            logger.error(f"Conversation storage failed: {e}")
    
    async def get_conversation_history(
        self,
        user_id: str,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get conversation history for user"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT messages, created_at, updated_at
                    FROM conversations 
                    WHERE user_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (user_id, limit))
                rows = cursor.fetchall()
            
            conversations = []
            for row in rows:
                messages, created_at, updated_at = row
                conversations.append({
                    "messages": json.loads(messages),
                    "created_at": created_at,
                    "updated_at": updated_at
                })
            
            return conversations
        
        except Exception as e:
            logger.error(f"Conversation history retrieval failed: {e}")
            return []
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile"""
        
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM user_profiles WHERE user_id = ?
                """, (user_id,))
                row = cursor.fetchone()
                
                if row:
                    profile = UserProfile(
                        user_id=row[0],
                        name=row[1],
                        preferences=json.loads(row[2]) if row[2] else {},
                        conversation_history=json.loads(row[3]) if row[3] else [],
                        personality_traits=json.loads(row[4]) if row[4] else {},
                        interests=json.loads(row[5]) if row[5] else [],
                        created_at=datetime.fromisoformat(row[6]),
                        last_updated=datetime.fromisoformat(row[7])
                    )
                    
                    self.user_profiles[user_id] = profile
                    return profile
        
        except Exception as e:
            logger.error(f"User profile retrieval failed: {e}")
        
        return None
    
    async def cleanup(self):
        """Cleanup resources"""
        
        if self.redis_client:
            self.redis_client.close()
        
        if self.tts_engine:
            self.tts_engine.stop()
        
        logger.info("AI Chatbot service cleanup completed")

# Global instance
ai_chatbot = None

async def get_ai_chatbot() -> AdvancedAIChatbot:
    """Get global AI chatbot instance"""
    global ai_chatbot
    if not ai_chatbot:
        config = {
            "database_path": "data/chatbot.db",
            "redis_url": "redis://localhost:6379",
            "openai_api_key": "your-openai-key",
            "anthropic_api_key": "your-anthropic-key"
        }
        ai_chatbot = AdvancedAIChatbot(config)
    return ai_chatbot



