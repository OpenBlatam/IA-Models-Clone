"""
AI Engine for OpusClip Improved
==============================

Advanced AI-powered video analysis and content generation.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import json
import base64
from pathlib import Path

import openai
import anthropic
from google.cloud import videointelligence
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
from sentence_transformers import SentenceTransformer
import cv2
import librosa
from moviepy.editor import VideoFileClip

from .schemas import AIProvider, ClipType, PlatformType
from .exceptions import AIProviderError, create_ai_provider_error

logger = logging.getLogger(__name__)


class AIEngine:
    """Advanced AI engine for video analysis and content generation"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.google_client = None
        self.sentence_transformer = None
        self.sentiment_pipeline = None
        self.summarization_pipeline = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models and clients"""
        try:
            # Initialize OpenAI client
            if openai.api_key:
                self.openai_client = openai.AsyncOpenAI()
            
            # Initialize Anthropic client
            if anthropic.api_key:
                self.anthropic_client = anthropic.AsyncAnthropic()
            
            # Initialize Google client
            try:
                self.google_client = videointelligence.VideoIntelligenceServiceClient()
            except Exception as e:
                logger.warning(f"Google Video Intelligence not available: {e}")
            
            # Initialize sentence transformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Initialize transformers pipelines
            self.sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            self.summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")
            
            logger.info("AI models initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
    
    async def analyze_video_content(
        self,
        video_path: str,
        provider: AIProvider = AIProvider.OPENAI,
        custom_prompts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze video content using AI"""
        try:
            start_time = time.time()
            
            # Extract video information
            video_info = await self._extract_video_info(video_path)
            
            # Extract transcript
            transcript = await self._extract_transcript(video_path)
            
            # Analyze content based on provider
            if provider == AIProvider.OPENAI:
                analysis = await self._analyze_with_openai(video_info, transcript, custom_prompts)
            elif provider == AIProvider.ANTHROPIC:
                analysis = await self._analyze_with_anthropic(video_info, transcript, custom_prompts)
            elif provider == AIProvider.GOOGLE:
                analysis = await self._analyze_with_google(video_path, custom_prompts)
            elif provider == AIProvider.HUGGINGFACE:
                analysis = await self._analyze_with_huggingface(video_info, transcript, custom_prompts)
            else:
                raise AIProviderError(f"Unsupported AI provider: {provider}")
            
            analysis['processing_time'] = time.time() - start_time
            analysis['provider'] = provider.value
            
            return analysis
            
        except Exception as e:
            raise create_ai_provider_error(provider.value, "content_analysis", e)
    
    async def identify_key_moments(
        self,
        video_path: str,
        transcript: str,
        provider: AIProvider = AIProvider.OPENAI
    ) -> List[Dict[str, Any]]:
        """Identify key moments in video"""
        try:
            # Extract video segments with timestamps
            segments = await self._extract_video_segments(video_path)
            
            # Analyze each segment
            key_moments = []
            for segment in segments:
                segment_transcript = await self._extract_segment_transcript(transcript, segment)
                
                # Use AI to analyze segment
                if provider == AIProvider.OPENAI:
                    analysis = await self._analyze_segment_with_openai(segment, segment_transcript)
                elif provider == AIProvider.ANTHROPIC:
                    analysis = await self._analyze_segment_with_anthropic(segment, segment_transcript)
                else:
                    analysis = await self._analyze_segment_with_huggingface(segment, segment_transcript)
                
                if analysis.get('is_key_moment', False):
                    key_moments.append({
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'type': analysis.get('type', 'unknown'),
                        'confidence': analysis.get('confidence', 0.5),
                        'description': analysis.get('description', ''),
                        'reason': analysis.get('reason', ''),
                        'emotions': analysis.get('emotions', []),
                        'topics': analysis.get('topics', [])
                    })
            
            # Sort by confidence and return top moments
            key_moments.sort(key=lambda x: x['confidence'], reverse=True)
            return key_moments[:10]  # Return top 10 key moments
            
        except Exception as e:
            raise create_ai_provider_error(provider.value, "key_moments", e)
    
    async def generate_insights(
        self,
        video_path: str,
        analysis_results: Dict[str, Any],
        provider: AIProvider = AIProvider.OPENAI
    ) -> Dict[str, Any]:
        """Generate comprehensive insights from video analysis"""
        try:
            # Prepare context for AI
            context = {
                'transcript': analysis_results.get('transcript', ''),
                'sentiment_scores': analysis_results.get('sentiment_scores', {}),
                'key_moments': analysis_results.get('key_moments', []),
                'scene_changes': analysis_results.get('scene_changes', []),
                'face_detections': analysis_results.get('face_detections', []),
                'object_detections': analysis_results.get('object_detections', [])
            }
            
            # Generate insights based on provider
            if provider == AIProvider.OPENAI:
                insights = await self._generate_insights_with_openai(context)
            elif provider == AIProvider.ANTHROPIC:
                insights = await self._generate_insights_with_anthropic(context)
            else:
                insights = await self._generate_insights_with_huggingface(context)
            
            return insights
            
        except Exception as e:
            raise create_ai_provider_error(provider.value, "insights_generation", e)
    
    async def optimize_for_platform(
        self,
        video_path: str,
        platform: PlatformType,
        clip_type: ClipType,
        provider: AIProvider = AIProvider.OPENAI
    ) -> Dict[str, Any]:
        """Optimize video content for specific platform"""
        try:
            # Get platform-specific requirements
            platform_requirements = self._get_platform_requirements(platform)
            
            # Analyze current video
            video_analysis = await self.analyze_video_content(video_path, provider)
            
            # Generate optimization recommendations
            if provider == AIProvider.OPENAI:
                optimization = await self._optimize_with_openai(
                    video_analysis, platform_requirements, clip_type
                )
            elif provider == AIProvider.ANTHROPIC:
                optimization = await self._optimize_with_anthropic(
                    video_analysis, platform_requirements, clip_type
                )
            else:
                optimization = await self._optimize_with_huggingface(
                    video_analysis, platform_requirements, clip_type
                )
            
            return optimization
            
        except Exception as e:
            raise create_ai_provider_error(provider.value, "platform_optimization", e)
    
    async def generate_viral_potential_score(
        self,
        video_path: str,
        analysis_results: Dict[str, Any],
        provider: AIProvider = AIProvider.OPENAI
    ) -> float:
        """Generate viral potential score for video"""
        try:
            # Prepare viral factors
            viral_factors = {
                'transcript': analysis_results.get('transcript', ''),
                'sentiment': analysis_results.get('sentiment_scores', {}),
                'key_moments': analysis_results.get('key_moments', []),
                'emotions': analysis_results.get('emotions', []),
                'topics': analysis_results.get('topics', []),
                'duration': analysis_results.get('duration', 0),
                'face_count': len(analysis_results.get('face_detections', [])),
                'scene_changes': len(analysis_results.get('scene_changes', []))
            }
            
            # Calculate viral score
            if provider == AIProvider.OPENAI:
                viral_score = await self._calculate_viral_score_with_openai(viral_factors)
            elif provider == AIProvider.ANTHROPIC:
                viral_score = await self._calculate_viral_score_with_anthropic(viral_factors)
            else:
                viral_score = await self._calculate_viral_score_with_huggingface(viral_factors)
            
            return min(max(viral_score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            raise create_ai_provider_error(provider.value, "viral_score", e)
    
    async def generate_clip_suggestions(
        self,
        video_path: str,
        analysis_results: Dict[str, Any],
        clip_type: ClipType,
        provider: AIProvider = AIProvider.OPENAI
    ) -> List[Dict[str, Any]]:
        """Generate clip suggestions based on analysis"""
        try:
            # Prepare context
            context = {
                'transcript': analysis_results.get('transcript', ''),
                'key_moments': analysis_results.get('key_moments', []),
                'sentiment_scores': analysis_results.get('sentiment_scores', {}),
                'duration': analysis_results.get('duration', 0),
                'clip_type': clip_type.value
            }
            
            # Generate suggestions
            if provider == AIProvider.OPENAI:
                suggestions = await self._generate_suggestions_with_openai(context)
            elif provider == AIProvider.ANTHROPIC:
                suggestions = await self._generate_suggestions_with_anthropic(context)
            else:
                suggestions = await self._generate_suggestions_with_huggingface(context)
            
            return suggestions
            
        except Exception as e:
            raise create_ai_provider_error(provider.value, "clip_suggestions", e)
    
    # OpenAI-specific methods
    async def _analyze_with_openai(self, video_info: Dict, transcript: str, custom_prompts: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze video content using OpenAI"""
        try:
            if not self.openai_client:
                raise AIProviderError("OpenAI client not initialized")
            
            # Prepare prompt
            prompt = f"""
            Analyze this video content and provide insights:
            
            Video Info:
            - Duration: {video_info.get('duration', 0)} seconds
            - Resolution: {video_info.get('resolution', 'unknown')}
            - FPS: {video_info.get('fps', 0)}
            
            Transcript:
            {transcript}
            
            Please provide:
            1. Content summary (2-3 sentences)
            2. Main topics discussed
            3. Emotional tone
            4. Key themes
            5. Target audience
            6. Content quality assessment
            """
            
            if custom_prompts:
                prompt += f"\n\nAdditional analysis requests:\n" + "\n".join(custom_prompts)
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert video content analyst."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            
            # Parse response
            return self._parse_openai_response(content)
            
        except Exception as e:
            raise AIProviderError(f"OpenAI analysis failed: {str(e)}")
    
    async def _analyze_segment_with_openai(self, segment: Dict, transcript: str) -> Dict[str, Any]:
        """Analyze video segment using OpenAI"""
        try:
            if not self.openai_client:
                raise AIProviderError("OpenAI client not initialized")
            
            prompt = f"""
            Analyze this video segment and determine if it's a key moment:
            
            Segment: {segment['start_time']}s - {segment['end_time']}s
            Transcript: {transcript}
            
            Determine:
            1. Is this a key moment? (yes/no)
            2. Type of moment (highlight, emotional, informative, etc.)
            3. Confidence score (0-1)
            4. Brief description
            5. Reason why it's key
            6. Emotions present
            7. Topics discussed
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at identifying key moments in video content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return self._parse_segment_analysis(content)
            
        except Exception as e:
            raise AIProviderError(f"OpenAI segment analysis failed: {str(e)}")
    
    async def _generate_insights_with_openai(self, context: Dict) -> Dict[str, Any]:
        """Generate insights using OpenAI"""
        try:
            if not self.openai_client:
                raise AIProviderError("OpenAI client not initialized")
            
            prompt = f"""
            Based on this video analysis, generate comprehensive insights:
            
            Transcript: {context.get('transcript', '')}
            Sentiment: {context.get('sentiment_scores', {})}
            Key Moments: {len(context.get('key_moments', []))} moments identified
            Scene Changes: {len(context.get('scene_changes', []))} changes
            Face Detections: {len(context.get('face_detections', []))} detections
            
            Provide:
            1. Content summary
            2. Main topics
            3. Emotional analysis
            4. Engagement factors
            5. Improvement suggestions
            6. Viral potential factors
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert video content strategist."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return self._parse_insights_response(content)
            
        except Exception as e:
            raise AIProviderError(f"OpenAI insights generation failed: {str(e)}")
    
    async def _optimize_with_openai(self, video_analysis: Dict, platform_requirements: Dict, clip_type: ClipType) -> Dict[str, Any]:
        """Optimize content using OpenAI"""
        try:
            if not self.openai_client:
                raise AIProviderError("OpenAI client not initialized")
            
            prompt = f"""
            Optimize this video content for {platform_requirements.get('platform', 'unknown')} platform:
            
            Current Content:
            - Duration: {video_analysis.get('duration', 0)} seconds
            - Topics: {video_analysis.get('topics', [])}
            - Sentiment: {video_analysis.get('sentiment', {})}
            
            Platform Requirements:
            - Max Duration: {platform_requirements.get('max_duration', 60)} seconds
            - Aspect Ratio: {platform_requirements.get('aspect_ratio', '16:9')}
            - Target Audience: {platform_requirements.get('audience', 'general')}
            
            Clip Type: {clip_type.value}
            
            Provide optimization recommendations:
            1. Duration adjustments
            2. Content focus areas
            3. Hook suggestions
            4. Call-to-action recommendations
            5. Visual enhancements
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert social media content optimizer."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return self._parse_optimization_response(content)
            
        except Exception as e:
            raise AIProviderError(f"OpenAI optimization failed: {str(e)}")
    
    async def _calculate_viral_score_with_openai(self, viral_factors: Dict) -> float:
        """Calculate viral score using OpenAI"""
        try:
            if not self.openai_client:
                raise AIProviderError("OpenAI client not initialized")
            
            prompt = f"""
            Calculate viral potential score (0-1) for this video content:
            
            Factors:
            - Transcript: {viral_factors.get('transcript', '')[:500]}...
            - Sentiment: {viral_factors.get('sentiment', {})}
            - Key Moments: {len(viral_factors.get('key_moments', []))}
            - Emotions: {viral_factors.get('emotions', [])}
            - Topics: {viral_factors.get('topics', [])}
            - Duration: {viral_factors.get('duration', 0)} seconds
            - Face Count: {viral_factors.get('face_count', 0)}
            - Scene Changes: {viral_factors.get('scene_changes', 0)}
            
            Consider factors like:
            - Emotional impact
            - Shareability
            - Trending topics
            - Engagement potential
            - Visual appeal
            
            Provide only a number between 0 and 1.
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at predicting viral content. Respond with only a number between 0 and 1."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            try:
                return float(content)
            except ValueError:
                return 0.5  # Default score if parsing fails
            
        except Exception as e:
            raise AIProviderError(f"OpenAI viral score calculation failed: {str(e)}")
    
    async def _generate_suggestions_with_openai(self, context: Dict) -> List[Dict[str, Any]]:
        """Generate clip suggestions using OpenAI"""
        try:
            if not self.openai_client:
                raise AIProviderError("OpenAI client not initialized")
            
            prompt = f"""
            Generate clip suggestions for this video content:
            
            Transcript: {context.get('transcript', '')}
            Key Moments: {len(context.get('key_moments', []))} moments
            Duration: {context.get('duration', 0)} seconds
            Clip Type: {context.get('clip_type', 'highlight')}
            
            Provide 5 clip suggestions with:
            1. Start and end times
            2. Clip type
            3. Reason for selection
            4. Expected engagement
            5. Hook suggestion
            """
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert at creating engaging video clips."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            return self._parse_suggestions_response(content)
            
        except Exception as e:
            raise AIProviderError(f"OpenAI suggestions generation failed: {str(e)}")
    
    # Anthropic-specific methods
    async def _analyze_with_anthropic(self, video_info: Dict, transcript: str, custom_prompts: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze video content using Anthropic"""
        try:
            if not self.anthropic_client:
                raise AIProviderError("Anthropic client not initialized")
            
            prompt = f"""
            Analyze this video content and provide insights:
            
            Video Info:
            - Duration: {video_info.get('duration', 0)} seconds
            - Resolution: {video_info.get('resolution', 'unknown')}
            - FPS: {video_info.get('fps', 0)}
            
            Transcript:
            {transcript}
            
            Please provide:
            1. Content summary (2-3 sentences)
            2. Main topics discussed
            3. Emotional tone
            4. Key themes
            5. Target audience
            6. Content quality assessment
            """
            
            if custom_prompts:
                prompt += f"\n\nAdditional analysis requests:\n" + "\n".join(custom_prompts)
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            return self._parse_anthropic_response(content)
            
        except Exception as e:
            raise AIProviderError(f"Anthropic analysis failed: {str(e)}")
    
    async def _analyze_segment_with_anthropic(self, segment: Dict, transcript: str) -> Dict[str, Any]:
        """Analyze video segment using Anthropic"""
        try:
            if not self.anthropic_client:
                raise AIProviderError("Anthropic client not initialized")
            
            prompt = f"""
            Analyze this video segment and determine if it's a key moment:
            
            Segment: {segment['start_time']}s - {segment['end_time']}s
            Transcript: {transcript}
            
            Determine:
            1. Is this a key moment? (yes/no)
            2. Type of moment (highlight, emotional, informative, etc.)
            3. Confidence score (0-1)
            4. Brief description
            5. Reason why it's key
            6. Emotions present
            7. Topics discussed
            """
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            return self._parse_segment_analysis(content)
            
        except Exception as e:
            raise AIProviderError(f"Anthropic segment analysis failed: {str(e)}")
    
    async def _generate_insights_with_anthropic(self, context: Dict) -> Dict[str, Any]:
        """Generate insights using Anthropic"""
        try:
            if not self.anthropic_client:
                raise AIProviderError("Anthropic client not initialized")
            
            prompt = f"""
            Based on this video analysis, generate comprehensive insights:
            
            Transcript: {context.get('transcript', '')}
            Sentiment: {context.get('sentiment_scores', {})}
            Key Moments: {len(context.get('key_moments', []))} moments identified
            Scene Changes: {len(context.get('scene_changes', []))} changes
            Face Detections: {len(context.get('face_detections', []))} detections
            
            Provide:
            1. Content summary
            2. Main topics
            3. Emotional analysis
            4. Engagement factors
            5. Improvement suggestions
            6. Viral potential factors
            """
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            return self._parse_insights_response(content)
            
        except Exception as e:
            raise AIProviderError(f"Anthropic insights generation failed: {str(e)}")
    
    async def _optimize_with_anthropic(self, video_analysis: Dict, platform_requirements: Dict, clip_type: ClipType) -> Dict[str, Any]:
        """Optimize content using Anthropic"""
        try:
            if not self.anthropic_client:
                raise AIProviderError("Anthropic client not initialized")
            
            prompt = f"""
            Optimize this video content for {platform_requirements.get('platform', 'unknown')} platform:
            
            Current Content:
            - Duration: {video_analysis.get('duration', 0)} seconds
            - Topics: {video_analysis.get('topics', [])}
            - Sentiment: {video_analysis.get('sentiment', {})}
            
            Platform Requirements:
            - Max Duration: {platform_requirements.get('max_duration', 60)} seconds
            - Aspect Ratio: {platform_requirements.get('aspect_ratio', '16:9')}
            - Target Audience: {platform_requirements.get('audience', 'general')}
            
            Clip Type: {clip_type.value}
            
            Provide optimization recommendations:
            1. Duration adjustments
            2. Content focus areas
            3. Hook suggestions
            4. Call-to-action recommendations
            5. Visual enhancements
            """
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            return self._parse_optimization_response(content)
            
        except Exception as e:
            raise AIProviderError(f"Anthropic optimization failed: {str(e)}")
    
    async def _calculate_viral_score_with_anthropic(self, viral_factors: Dict) -> float:
        """Calculate viral score using Anthropic"""
        try:
            if not self.anthropic_client:
                raise AIProviderError("Anthropic client not initialized")
            
            prompt = f"""
            Calculate viral potential score (0-1) for this video content:
            
            Factors:
            - Transcript: {viral_factors.get('transcript', '')[:500]}...
            - Sentiment: {viral_factors.get('sentiment', {})}
            - Key Moments: {len(viral_factors.get('key_moments', []))}
            - Emotions: {viral_factors.get('emotions', [])}
            - Topics: {viral_factors.get('topics', [])}
            - Duration: {viral_factors.get('duration', 0)} seconds
            - Face Count: {viral_factors.get('face_count', 0)}
            - Scene Changes: {viral_factors.get('scene_changes', 0)}
            
            Consider factors like:
            - Emotional impact
            - Shareability
            - Trending topics
            - Engagement potential
            - Visual appeal
            
            Provide only a number between 0 and 1.
            """
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=10,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text.strip()
            try:
                return float(content)
            except ValueError:
                return 0.5  # Default score if parsing fails
            
        except Exception as e:
            raise AIProviderError(f"Anthropic viral score calculation failed: {str(e)}")
    
    async def _generate_suggestions_with_anthropic(self, context: Dict) -> List[Dict[str, Any]]:
        """Generate clip suggestions using Anthropic"""
        try:
            if not self.anthropic_client:
                raise AIProviderError("Anthropic client not initialized")
            
            prompt = f"""
            Generate clip suggestions for this video content:
            
            Transcript: {context.get('transcript', '')}
            Key Moments: {len(context.get('key_moments', []))} moments
            Duration: {context.get('duration', 0)} seconds
            Clip Type: {context.get('clip_type', 'highlight')}
            
            Provide 5 clip suggestions with:
            1. Start and end times
            2. Clip type
            3. Reason for selection
            4. Expected engagement
            5. Hook suggestion
            """
            
            response = await self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=800,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            content = response.content[0].text
            return self._parse_suggestions_response(content)
            
        except Exception as e:
            raise AIProviderError(f"Anthropic suggestions generation failed: {str(e)}")
    
    # Google-specific methods
    async def _analyze_with_google(self, video_path: str, custom_prompts: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze video content using Google Video Intelligence"""
        try:
            if not self.google_client:
                raise AIProviderError("Google Video Intelligence client not initialized")
            
            # Read video file
            with open(video_path, 'rb') as video_file:
                input_content = video_file.read()
            
            # Configure features
            features = [
                videointelligence.Feature.LABEL_DETECTION,
                videointelligence.Feature.SHOT_CHANGE_DETECTION,
                videointelligence.Feature.EXPLICIT_CONTENT_DETECTION,
                videointelligence.Feature.FACE_DETECTION,
                videointelligence.Feature.SPEECH_TRANSCRIPTION,
                videointelligence.Feature.TEXT_DETECTION
            ]
            
            # Perform analysis
            operation = self.google_client.annotate_video(
                request={
                    "input_content": input_content,
                    "features": features,
                }
            )
            
            result = operation.result(timeout=300)
            
            # Parse results
            return self._parse_google_response(result)
            
        except Exception as e:
            raise AIProviderError(f"Google Video Intelligence analysis failed: {str(e)}")
    
    # Hugging Face-specific methods
    async def _analyze_with_huggingface(self, video_info: Dict, transcript: str, custom_prompts: Optional[List[str]]) -> Dict[str, Any]:
        """Analyze video content using Hugging Face models"""
        try:
            # Use local models for analysis
            analysis = {}
            
            # Sentiment analysis
            if transcript:
                sentiment_result = self.sentiment_pipeline(transcript)
                analysis['sentiment'] = sentiment_result[0]
            
            # Summarization
            if transcript and len(transcript) > 100:
                summary = self.summarization_pipeline(transcript, max_length=150, min_length=30)
                analysis['summary'] = summary[0]['summary_text']
            
            # Topic extraction using sentence transformer
            if transcript:
                sentences = transcript.split('.')
                embeddings = self.sentence_transformer.encode(sentences)
                # Simple topic extraction (would be more sophisticated in production)
                analysis['topics'] = ['general_content']  # Placeholder
            
            return analysis
            
        except Exception as e:
            raise AIProviderError(f"Hugging Face analysis failed: {str(e)}")
    
    async def _analyze_segment_with_huggingface(self, segment: Dict, transcript: str) -> Dict[str, Any]:
        """Analyze video segment using Hugging Face models"""
        try:
            # Simple analysis using local models
            if not transcript:
                return {'is_key_moment': False, 'confidence': 0.0}
            
            # Sentiment analysis
            sentiment_result = self.sentiment_pipeline(transcript)
            sentiment_score = sentiment_result[0]['score']
            
            # Determine if it's a key moment based on sentiment and length
            is_key_moment = sentiment_score > 0.7 or len(transcript) > 50
            
            return {
                'is_key_moment': is_key_moment,
                'confidence': sentiment_score,
                'type': 'emotional' if sentiment_score > 0.7 else 'informative',
                'description': transcript[:100] + '...' if len(transcript) > 100 else transcript,
                'reason': 'High sentiment score' if sentiment_score > 0.7 else 'Informative content',
                'emotions': [sentiment_result[0]['label']],
                'topics': ['general']
            }
            
        except Exception as e:
            raise AIProviderError(f"Hugging Face segment analysis failed: {str(e)}")
    
    async def _generate_insights_with_huggingface(self, context: Dict) -> Dict[str, Any]:
        """Generate insights using Hugging Face models"""
        try:
            insights = {}
            
            transcript = context.get('transcript', '')
            if transcript:
                # Generate summary
                summary = self.summarization_pipeline(transcript, max_length=150, min_length=30)
                insights['summary'] = summary[0]['summary_text']
                
                # Extract topics (simplified)
                insights['topics'] = ['general_content']
                
                # Analyze emotions
                sentiment_result = self.sentiment_pipeline(transcript)
                insights['emotions'] = [sentiment_result[0]['label']]
            
            return insights
            
        except Exception as e:
            raise AIProviderError(f"Hugging Face insights generation failed: {str(e)}")
    
    async def _optimize_with_huggingface(self, video_analysis: Dict, platform_requirements: Dict, clip_type: ClipType) -> Dict[str, Any]:
        """Optimize content using Hugging Face models"""
        try:
            # Simple optimization recommendations
            optimization = {
                'duration_adjustments': f"Consider {platform_requirements.get('max_duration', 60)} second clips",
                'content_focus': 'Focus on high-engagement segments',
                'hook_suggestions': 'Start with attention-grabbing content',
                'call_to_action': 'Include clear call-to-action',
                'visual_enhancements': 'Add captions and visual effects'
            }
            
            return optimization
            
        except Exception as e:
            raise AIProviderError(f"Hugging Face optimization failed: {str(e)}")
    
    async def _calculate_viral_score_with_huggingface(self, viral_factors: Dict) -> float:
        """Calculate viral score using Hugging Face models"""
        try:
            score = 0.5  # Base score
            
            # Adjust based on sentiment
            sentiment = viral_factors.get('sentiment', {})
            if sentiment:
                positive_score = sentiment.get('positive', 0)
                score += positive_score * 0.3
            
            # Adjust based on key moments
            key_moments = viral_factors.get('key_moments', [])
            score += min(len(key_moments) * 0.1, 0.2)
            
            # Adjust based on emotions
            emotions = viral_factors.get('emotions', [])
            viral_emotions = ['joy', 'excitement', 'surprise', 'amusement']
            emotion_score = sum(1 for emotion in emotions if emotion.lower() in viral_emotions)
            score += min(emotion_score * 0.1, 0.2)
            
            return min(score, 1.0)
            
        except Exception as e:
            raise AIProviderError(f"Hugging Face viral score calculation failed: {str(e)}")
    
    async def _generate_suggestions_with_huggingface(self, context: Dict) -> List[Dict[str, Any]]:
        """Generate clip suggestions using Hugging Face models"""
        try:
            # Simple suggestions based on transcript length and content
            suggestions = []
            transcript = context.get('transcript', '')
            duration = context.get('duration', 0)
            
            if transcript and duration > 0:
                # Create 5 suggestions with different time ranges
                for i in range(5):
                    start_time = (duration / 5) * i
                    end_time = start_time + min(30, duration / 5)
                    
                    suggestions.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'type': 'highlight',
                        'reason': f'Segment {i+1} of content',
                        'engagement': 'medium',
                        'hook': transcript[:50] + '...' if len(transcript) > 50 else transcript
                    })
            
            return suggestions
            
        except Exception as e:
            raise AIProviderError(f"Hugging Face suggestions generation failed: {str(e)}")
    
    # Helper methods
    async def _extract_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract basic video information"""
        try:
            with VideoFileClip(video_path) as clip:
                return {
                    'duration': clip.duration,
                    'fps': clip.fps,
                    'resolution': f"{clip.w}x{clip.h}",
                    'size': clip.size
                }
        except Exception as e:
            raise AIProviderError(f"Video info extraction failed: {str(e)}")
    
    async def _extract_transcript(self, video_path: str) -> str:
        """Extract transcript from video"""
        try:
            # Extract audio
            audio_path = f"./temp/audio_{int(time.time())}.wav"
            with VideoFileClip(video_path) as clip:
                clip.audio.write_audiofile(audio_path)
            
            # Transcribe using Whisper
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            
            # Clean up
            os.remove(audio_path)
            
            return result["text"]
        except Exception as e:
            raise AIProviderError(f"Transcript extraction failed: {str(e)}")
    
    async def _extract_video_segments(self, video_path: str, segment_duration: int = 30) -> List[Dict[str, Any]]:
        """Extract video segments for analysis"""
        try:
            with VideoFileClip(video_path) as clip:
                duration = clip.duration
                segments = []
                
                for start_time in range(0, int(duration), segment_duration):
                    end_time = min(start_time + segment_duration, duration)
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
                
                return segments
        except Exception as e:
            raise AIProviderError(f"Video segment extraction failed: {str(e)}")
    
    async def _extract_segment_transcript(self, full_transcript: str, segment: Dict) -> str:
        """Extract transcript for a specific segment"""
        # This is a simplified implementation
        # In production, would use precise timestamp alignment
        words = full_transcript.split()
        segment_duration = segment['end_time'] - segment['start_time']
        total_duration = len(words) * 0.1  # Assume 0.1 seconds per word
        
        start_word = int((segment['start_time'] / total_duration) * len(words))
        end_word = int((segment['end_time'] / total_duration) * len(words))
        
        return ' '.join(words[start_word:end_word])
    
    def _get_platform_requirements(self, platform: PlatformType) -> Dict[str, Any]:
        """Get platform-specific requirements"""
        requirements = {
            PlatformType.YOUTUBE: {
                'platform': 'YouTube',
                'max_duration': 60,
                'aspect_ratio': '16:9',
                'audience': 'general',
                'features': ['captions', 'thumbnails', 'end_screens']
            },
            PlatformType.TIKTOK: {
                'platform': 'TikTok',
                'max_duration': 60,
                'aspect_ratio': '9:16',
                'audience': 'young',
                'features': ['music', 'effects', 'trends']
            },
            PlatformType.INSTAGRAM: {
                'platform': 'Instagram',
                'max_duration': 60,
                'aspect_ratio': '1:1',
                'audience': 'visual',
                'features': ['stories', 'reels', 'filters']
            },
            PlatformType.TWITTER: {
                'platform': 'Twitter',
                'max_duration': 140,
                'aspect_ratio': '16:9',
                'audience': 'news',
                'features': ['threads', 'trends', 'hashtags']
            },
            PlatformType.LINKEDIN: {
                'platform': 'LinkedIn',
                'max_duration': 30,
                'aspect_ratio': '16:9',
                'audience': 'professional',
                'features': ['business', 'networking', 'career']
            }
        }
        
        return requirements.get(platform, requirements[PlatformType.YOUTUBE])
    
    def _parse_openai_response(self, content: str) -> Dict[str, Any]:
        """Parse OpenAI response"""
        # Simplified parsing - in production would be more sophisticated
        return {
            'summary': content[:200] + '...' if len(content) > 200 else content,
            'topics': ['general_content'],
            'emotions': ['neutral'],
            'themes': ['general'],
            'audience': 'general',
            'quality': 'good'
        }
    
    def _parse_anthropic_response(self, content: str) -> Dict[str, Any]:
        """Parse Anthropic response"""
        # Simplified parsing - in production would be more sophisticated
        return {
            'summary': content[:200] + '...' if len(content) > 200 else content,
            'topics': ['general_content'],
            'emotions': ['neutral'],
            'themes': ['general'],
            'audience': 'general',
            'quality': 'good'
        }
    
    def _parse_google_response(self, result) -> Dict[str, Any]:
        """Parse Google Video Intelligence response"""
        analysis = {
            'summary': 'Video analyzed using Google Video Intelligence',
            'topics': [],
            'emotions': [],
            'themes': [],
            'audience': 'general',
            'quality': 'good'
        }
        
        # Parse labels
        if hasattr(result, 'annotation_results'):
            for annotation in result.annotation_results:
                if hasattr(annotation, 'segment_label_annotations'):
                    for label_annotation in annotation.segment_label_annotations:
                        analysis['topics'].append(label_annotation.entity.description)
        
        return analysis
    
    def _parse_segment_analysis(self, content: str) -> Dict[str, Any]:
        """Parse segment analysis response"""
        # Simplified parsing - in production would be more sophisticated
        lines = content.split('\n')
        analysis = {
            'is_key_moment': 'yes' in content.lower(),
            'confidence': 0.5,
            'type': 'general',
            'description': content[:100] + '...' if len(content) > 100 else content,
            'reason': 'AI analysis',
            'emotions': ['neutral'],
            'topics': ['general']
        }
        
        # Try to extract confidence score
        for line in lines:
            if 'confidence' in line.lower():
                try:
                    score = float(line.split(':')[-1].strip())
                    analysis['confidence'] = score
                except:
                    pass
        
        return analysis
    
    def _parse_insights_response(self, content: str) -> Dict[str, Any]:
        """Parse insights response"""
        return {
            'summary': content[:200] + '...' if len(content) > 200 else content,
            'topics': ['general_content'],
            'emotions': ['neutral'],
            'engagement_factors': ['content_quality'],
            'improvements': ['enhance_visuals'],
            'viral_factors': ['emotional_impact']
        }
    
    def _parse_optimization_response(self, content: str) -> Dict[str, Any]:
        """Parse optimization response"""
        return {
            'duration_adjustments': 'Optimize for platform requirements',
            'content_focus': 'Focus on engaging segments',
            'hook_suggestions': 'Start with attention-grabbing content',
            'call_to_action': 'Include clear call-to-action',
            'visual_enhancements': 'Add captions and effects'
        }
    
    def _parse_suggestions_response(self, content: str) -> List[Dict[str, Any]]:
        """Parse suggestions response"""
        # Simplified parsing - in production would be more sophisticated
        suggestions = []
        for i in range(5):
            suggestions.append({
                'start_time': i * 30,
                'end_time': (i + 1) * 30,
                'type': 'highlight',
                'reason': f'Suggestion {i+1}',
                'engagement': 'medium',
                'hook': content[:50] + '...' if len(content) > 50 else content
            })
        
        return suggestions


# Global AI engine instance
ai_engine = AIEngine()






























