"""
🎤 ADVANCED VOICE COACHING AI EXAMPLE USAGE
============================================

This example demonstrates all the advanced capabilities of the Voice Coaching AI system,
including emotion detection, multi-language support, vocal health monitoring, and voice synthesis.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from .core import (
    VoiceCoachingConfig, VoiceToneType, ConfidenceLevel, CoachingFocus,
    EmotionType, LanguageType, VoiceSynthesisType, AdvancedMetrics
)
from .factories.voice_coaching_factory import create_voice_coaching_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVoiceCoachingDemo:
    """Advanced demonstration of Voice Coaching AI capabilities"""
    
    def __init__(self):
        self.config = VoiceCoachingConfig(
            openrouter_api_key="your-api-key-here",
            model_name="anthropic/claude-3.5-sonnet",
            max_tokens=4000,
            temperature=0.7,
            max_retries=3,
            timeout=30,
            cache_ttl=3600,
            session_timeout=1800
        )
        self.service = None
        self.user_id = "demo_user_advanced"
        
    async def initialize(self):
        """Initialize the advanced voice coaching service"""
        logger.info("🚀 Initializing Advanced Voice Coaching AI...")
        
        try:
            # Create and initialize service
            self.service = create_voice_coaching_service(self.config)
            success = await self.service.initialize()
            
            if success:
                logger.info("✅ Advanced Voice Coaching AI initialized successfully")
                return True
            else:
                logger.error("❌ Failed to initialize Advanced Voice Coaching AI")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error during initialization: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.service:
            await self.service.cleanup()
            logger.info("🧹 Advanced Voice Coaching AI cleaned up")
    
    async def demonstrate_comprehensive_analysis(self, audio_data: bytes):
        """Demonstrate comprehensive voice analysis with all advanced features"""
        logger.info("🎯 Demonstrating Comprehensive Voice Analysis...")
        
        try:
            # Perform comprehensive analysis
            comprehensive_result = await self.service.comprehensive_voice_analysis(
                self.user_id, audio_data
            )
            
            if "error" in comprehensive_result:
                logger.error(f"Comprehensive analysis failed: {comprehensive_result['error']}")
                return
            
            # Display results
            logger.info("📊 COMPREHENSIVE ANALYSIS RESULTS:")
            logger.info(f"Comprehensive Score: {comprehensive_result['comprehensive_score']:.2f}")
            
            # Basic analysis
            basic = comprehensive_result['basic_analysis']
            logger.info(f"Confidence Score: {basic.confidence_score:.2f}")
            logger.info(f"Tone Detected: {basic.tone_detected.value}")
            logger.info(f"Leadership Presence: {basic.leadership_presence:.2f}")
            
            # Emotion analysis
            emotion = comprehensive_result['emotion_analysis']
            if 'emotion_analysis' in emotion:
                emotion_data = emotion['emotion_analysis']
                logger.info(f"Primary Emotion: {emotion_data.get('primary_emotion', 'unknown')}")
                logger.info(f"Emotion Confidence: {emotion_data.get('emotion_confidence', 0.0):.2f}")
                logger.info(f"Emotion Intensity: {emotion_data.get('emotion_intensity', 0.0):.2f}")
            
            # Language analysis
            language = comprehensive_result['language_analysis']
            if 'language_analysis' in language:
                language_data = language['language_analysis']
                logger.info(f"Primary Language: {language_data.get('primary_language', 'unknown')}")
                logger.info(f"Language Confidence: {language_data.get('language_confidence', 0.0):.2f}")
            
            # Health analysis
            health = comprehensive_result['health_analysis']
            if 'health_analysis' in health:
                health_data = health['health_analysis']
                logger.info(f"Vocal Health Score: {health_data.get('vocal_health_score', 0.0):.2f}")
                logger.info(f"Vocal Fatigue: {health_data.get('vocal_fatigue', 0.0):.2f}")
                logger.info(f"Breathing Rhythm: {health_data.get('breathing_rhythm', 0.0):.2f}")
            
            logger.info("✅ Comprehensive analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Comprehensive analysis demonstration failed: {e}")
    
    async def demonstrate_emotion_coaching(self, audio_data: bytes):
        """Demonstrate emotion detection and coaching"""
        logger.info("😊 Demonstrating Emotion Detection and Coaching...")
        
        try:
            # Analyze emotion and get coaching
            emotion_result = await self.service.analyze_emotion_and_coach(
                self.user_id, audio_data
            )
            
            if "error" in emotion_result:
                logger.error(f"Emotion coaching failed: {emotion_result['error']}")
                return
            
            # Display emotion analysis
            emotion_analysis = emotion_result['emotion_analysis']
            logger.info("📊 EMOTION ANALYSIS:")
            logger.info(f"Primary Emotion: {emotion_analysis.get('primary_emotion', 'unknown')}")
            logger.info(f"Emotion Confidence: {emotion_analysis.get('emotion_confidence', 0.0):.2f}")
            logger.info(f"Secondary Emotions: {emotion_analysis.get('secondary_emotions', [])}")
            logger.info(f"Emotion Intensity: {emotion_analysis.get('emotion_intensity', 0.0):.2f}")
            
            # Display coaching recommendations
            coaching_recommendations = emotion_result['coaching_recommendations']
            logger.info("💡 EMOTION COACHING RECOMMENDATIONS:")
            for i, recommendation in enumerate(coaching_recommendations, 1):
                logger.info(f"{i}. {recommendation}")
            
            logger.info("✅ Emotion coaching completed successfully")
            
        except Exception as e:
            logger.error(f"Emotion coaching demonstration failed: {e}")
    
    async def demonstrate_language_adaptation(self, audio_data: bytes):
        """Demonstrate multi-language detection and adaptation"""
        logger.info("🌍 Demonstrating Multi-Language Detection and Adaptation...")
        
        try:
            # Detect language and adapt coaching
            language_result = await self.service.detect_language_and_adapt(
                self.user_id, audio_data
            )
            
            if "error" in language_result:
                logger.error(f"Language adaptation failed: {language_result['error']}")
                return
            
            # Display language analysis
            language_analysis = language_result['language_analysis']
            logger.info("📊 LANGUAGE ANALYSIS:")
            logger.info(f"Primary Language: {language_analysis.get('primary_language', 'unknown')}")
            logger.info(f"Language Confidence: {language_analysis.get('language_confidence', 0.0):.2f}")
            logger.info(f"Detected Languages: {language_analysis.get('detected_languages', [])}")
            
            accent_analysis = language_analysis.get('accent_analysis', {})
            logger.info(f"Accent Type: {accent_analysis.get('accent_type', 'unknown')}")
            logger.info(f"Accent Strength: {accent_analysis.get('accent_strength', 0.0):.2f}")
            
            # Display adapted coaching
            adapted_coaching = language_result['adapted_coaching']
            logger.info("🌍 ADAPTED COACHING:")
            logger.info(f"Coaching Language: {adapted_coaching.get('coaching_language', 'en')}")
            
            accent_tips = adapted_coaching.get('accent_specific_tips', [])
            if accent_tips:
                logger.info("Accent-Specific Tips:")
                for tip in accent_tips:
                    logger.info(f"  • {tip}")
            
            language_exercises = adapted_coaching.get('language_specific_exercises', [])
            if language_exercises:
                logger.info("Language-Specific Exercises:")
                for exercise in language_exercises:
                    logger.info(f"  • {exercise}")
            
            logger.info("✅ Language adaptation completed successfully")
            
        except Exception as e:
            logger.error(f"Language adaptation demonstration failed: {e}")
    
    async def demonstrate_vocal_health_monitoring(self, audio_data: bytes):
        """Demonstrate vocal health monitoring"""
        logger.info("🏥 Demonstrating Vocal Health Monitoring...")
        
        try:
            # Monitor vocal health
            health_result = await self.service.monitor_vocal_health(
                self.user_id, audio_data
            )
            
            if "error" in health_result:
                logger.error(f"Vocal health monitoring failed: {health_result['error']}")
                return
            
            # Display health analysis
            health_analysis = health_result['health_analysis']
            logger.info("📊 VOCAL HEALTH ANALYSIS:")
            logger.info(f"Vocal Health Score: {health_analysis.get('vocal_health_score', 0.0):.2f}")
            logger.info(f"Breathing Rhythm: {health_analysis.get('breathing_rhythm', 0.0):.2f}")
            logger.info(f"Vocal Fatigue: {health_analysis.get('vocal_fatigue', 0.0):.2f}")
            logger.info(f"Articulation Precision: {health_analysis.get('articulation_precision', 0.0):.2f}")
            logger.info(f"Phonation Efficiency: {health_analysis.get('phonation_efficiency', 0.0):.2f}")
            logger.info(f"Resonance Quality: {health_analysis.get('resonance_quality', 0.0):.2f}")
            
            # Display health indicators
            health_indicators = health_analysis.get('health_indicators', [])
            if health_indicators:
                logger.info("Health Indicators:")
                for indicator in health_indicators:
                    logger.info(f"  ✅ {indicator}")
            
            # Display health recommendations
            health_recommendations = health_result['health_recommendations']
            logger.info("💡 VOCAL HEALTH RECOMMENDATIONS:")
            for i, recommendation in enumerate(health_recommendations, 1):
                logger.info(f"{i}. {recommendation}")
            
            logger.info("✅ Vocal health monitoring completed successfully")
            
        except Exception as e:
            logger.error(f"Vocal health monitoring demonstration failed: {e}")
    
    async def demonstrate_voice_synthesis(self, text: str, synthesis_type: VoiceSynthesisType):
        """Demonstrate voice synthesis capabilities"""
        logger.info(f"🎭 Demonstrating Voice Synthesis ({synthesis_type.value})...")
        
        try:
            # Generate voice synthesis
            synthesis_result = await self.service.synthesize_voice_example(
                self.user_id, text, synthesis_type
            )
            
            if "error" in synthesis_result:
                logger.error(f"Voice synthesis failed: {synthesis_result['error']}")
                return
            
            # Display synthesis data
            synthesis_data = synthesis_result['synthesis_data']
            logger.info("📊 VOICE SYNTHESIS PARAMETERS:")
            logger.info(f"Synthesis Type: {synthesis_data.get('synthesis_type', 'unknown')}")
            
            voice_characteristics = synthesis_data.get('voice_characteristics', {})
            logger.info("Voice Characteristics:")
            for key, value in voice_characteristics.items():
                logger.info(f"  • {key}: {value:.2f}")
            
            # Display coaching example
            coaching_example = synthesis_result['coaching_example']
            logger.info("🎯 COACHING EXAMPLE:")
            logger.info(f"Text: {coaching_example.get('text', '')}")
            
            practice_instructions = coaching_example.get('practice_instructions', [])
            if practice_instructions:
                logger.info("Practice Instructions:")
                for instruction in practice_instructions:
                    logger.info(f"  • {instruction}")
            
            emphasis_points = coaching_example.get('emphasis_points', [])
            if emphasis_points:
                logger.info("Emphasis Points:")
                for point in emphasis_points:
                    logger.info(f"  • {point.get('word', '')}: {point.get('emphasis', '')}")
            
            practice_tips = coaching_example.get('practice_tips', [])
            if practice_tips:
                logger.info("Practice Tips:")
                for tip in practice_tips:
                    logger.info(f"  • {tip}")
            
            logger.info("✅ Voice synthesis completed successfully")
            
        except Exception as e:
            logger.error(f"Voice synthesis demonstration failed: {e}")
    
    async def demonstrate_real_time_coaching(self, audio_stream: Any):
        """Demonstrate real-time coaching capabilities"""
        logger.info("⏱️ Demonstrating Real-Time Coaching...")
        
        try:
            # Start real-time coaching
            coach_id = await self.service.start_real_time_coaching(
                self.user_id, CoachingFocus.LEADERSHIP_VOICE
            )
            logger.info(f"Started real-time coaching session: {coach_id}")
            
            # Simulate real-time analysis (in practice, this would be continuous)
            for i in range(3):
                logger.info(f"Real-time analysis iteration {i + 1}...")
                
                # In practice, this would analyze actual audio stream
                # For demo, we'll simulate the analysis
                await asyncio.sleep(1)
                
                # Simulate analysis result
                logger.info(f"  • Confidence: {0.7 + i * 0.1:.2f}")
                logger.info(f"  • Leadership Presence: {0.6 + i * 0.15:.2f}")
                logger.info(f"  • Emotional Expression: {0.5 + i * 0.2:.2f}")
            
            # Stop real-time coaching
            summary = await self.service.stop_real_time_coaching(coach_id)
            logger.info("📊 REAL-TIME COACHING SUMMARY:")
            logger.info(f"Session Duration: {summary.get('session_duration', 0.0):.2f} minutes")
            logger.info(f"Analysis Count: {summary.get('analysis_count', 0)}")
            logger.info(f"Feedback Given: {summary.get('feedback_given', 0)}")
            logger.info(f"Exercises Suggested: {summary.get('exercises_suggested', 0)}")
            
            logger.info("✅ Real-time coaching completed successfully")
            
        except Exception as e:
            logger.error(f"Real-time coaching demonstration failed: {e}")
    
    async def demonstrate_adaptive_difficulty(self):
        """Demonstrate adaptive difficulty system"""
        logger.info("🎯 Demonstrating Adaptive Difficulty System...")
        
        try:
            # Generate personalized exercises with different focus areas
            focus_areas = [
                CoachingFocus.LEADERSHIP_VOICE,
                CoachingFocus.CONFIDENCE_BUILDING,
                CoachingFocus.PRESENTATION_SKILLS,
                CoachingFocus.EMOTIONAL_INTELLIGENCE
            ]
            
            for focus_area in focus_areas:
                logger.info(f"Generating exercises for: {focus_area.value}")
                
                exercises = await self.service.generate_personalized_exercises(
                    self.user_id, focus_area
                )
                
                logger.info(f"Generated {len(exercises)} exercises:")
                for i, exercise in enumerate(exercises, 1):
                    logger.info(f"  {i}. {exercise.title} (Difficulty: {exercise.difficulty_level})")
                    logger.info(f"     Type: {exercise.exercise_type.value}")
                    logger.info(f"     Tips: {len(exercise.tips)} tips provided")
                
                logger.info("")
            
            logger.info("✅ Adaptive difficulty demonstration completed successfully")
            
        except Exception as e:
            logger.error(f"Adaptive difficulty demonstration failed: {e}")
    
    async def demonstrate_leadership_insights(self):
        """Demonstrate leadership voice insights"""
        logger.info("👑 Demonstrating Leadership Voice Insights...")
        
        try:
            # Get leadership insights
            leadership_insights = await self.service.get_leadership_voice_insights(
                self.user_id
            )
            
            if "error" in leadership_insights:
                logger.error(f"Leadership insights failed: {leadership_insights['error']}")
                return
            
            logger.info("📊 LEADERSHIP VOICE INSIGHTS:")
            logger.info(f"Leadership Level: {leadership_insights.get('leadership_level', 'Unknown')}")
            
            leadership_metrics = leadership_insights.get('leadership_metrics', {})
            logger.info("Leadership Metrics:")
            for metric, value in leadership_metrics.items():
                logger.info(f"  • {metric}: {value:.2f}")
            
            leadership_recommendations = leadership_insights.get('leadership_recommendations', [])
            if leadership_recommendations:
                logger.info("👑 LEADERSHIP RECOMMENDATIONS:")
                for i, recommendation in enumerate(leadership_recommendations, 1):
                    logger.info(f"{i}. {recommendation}")
            
            logger.info("✅ Leadership insights completed successfully")
            
        except Exception as e:
            logger.error(f"Leadership insights demonstration failed: {e}")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demonstration of all advanced features"""
        logger.info("🎤 STARTING COMPREHENSIVE ADVANCED VOICE COACHING DEMO")
        logger.info("=" * 60)
        
        try:
            # Initialize service
            if not await self.initialize():
                return
            
            # Mock audio data (in practice, this would be real audio)
            mock_audio_data = b"mock_audio_data_for_demo"
            mock_audio_stream = "mock_audio_stream"
            
            # Demonstrate all advanced features
            await self.demonstrate_comprehensive_analysis(mock_audio_data)
            logger.info("")
            
            await self.demonstrate_emotion_coaching(mock_audio_data)
            logger.info("")
            
            await self.demonstrate_language_adaptation(mock_audio_data)
            logger.info("")
            
            await self.demonstrate_vocal_health_monitoring(mock_audio_data)
            logger.info("")
            
            await self.demonstrate_voice_synthesis(
                "This is an example of confident leadership voice with clear articulation and strategic pauses.",
                VoiceSynthesisType.LEADERSHIP
            )
            logger.info("")
            
            await self.demonstrate_real_time_coaching(mock_audio_stream)
            logger.info("")
            
            await self.demonstrate_adaptive_difficulty()
            logger.info("")
            
            await self.demonstrate_leadership_insights()
            logger.info("")
            
            # Display final metrics
            metrics = self.service.get_enhanced_metrics()
            logger.info("📊 FINAL SYSTEM METRICS:")
            logger.info(f"Total Events: {metrics.get('analytics_summary', {}).get('total_events', 0)}")
            logger.info(f"Active Sessions: {metrics.get('session_metrics', {}).get('active_sessions', 0)}")
            logger.info(f"User Profiles: {metrics.get('session_metrics', {}).get('user_profiles_count', 0)}")
            
            logger.info("")
            logger.info("🎉 COMPREHENSIVE ADVANCED VOICE COACHING DEMO COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            logger.error(f"Comprehensive demo failed: {e}")
        
        finally:
            await self.cleanup()

async def main():
    """Main function to run the advanced voice coaching demo"""
    demo = AdvancedVoiceCoachingDemo()
    await demo.run_comprehensive_demo()

if __name__ == "__main__":
    asyncio.run(main()) 