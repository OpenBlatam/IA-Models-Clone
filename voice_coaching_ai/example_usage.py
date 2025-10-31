#!/usr/bin/env python3
"""
🎤 Voice Coaching AI - Example Usage
====================================

This script demonstrates how to use the Voice Coaching AI system
for voice analysis, leadership training, and progress tracking.
"""

import asyncio
import logging
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import voice coaching components
try:
    from voice_coaching_ai import (
        create_voice_coaching_ai, 
        quick_voice_analysis,
        quick_leadership_coaching,
        VoiceCoachingConfig,
        CoachingFocus,
        VoiceToneType,
        ConfidenceLevel
    )
    VOICE_COACHING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Voice coaching AI not available: {e}")
    VOICE_COACHING_AVAILABLE = False

def create_mock_audio_data() -> bytes:
    """Create mock audio data for demonstration"""
    # In a real application, this would be actual audio data
    # For demonstration purposes, we'll create a mock audio file
    mock_audio = b"mock_audio_data_for_demonstration"
    return mock_audio

async def demonstrate_voice_analysis():
    """Demonstrate basic voice analysis"""
    print("\n🎤 === VOICE ANALYSIS DEMONSTRATION ===")
    
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        return
    
    # Create mock audio data
    audio_data = create_mock_audio_data()
    
    try:
        # Quick voice analysis
        print("🔍 Performing quick voice analysis...")
        result = await quick_voice_analysis(
            api_key=api_key,
            user_id="demo_user_123",
            audio_data=audio_data
        )
        
        if "error" in result:
            print(f"❌ Analysis failed: {result['error']}")
            return
        
        print("✅ Voice analysis completed!")
        print(f"📊 Tone detected: {result.get('tone', 'unknown')}")
        print(f"📈 Confidence score: {result.get('confidence_score', 0.0):.2f}")
        print(f"💡 Suggestions: {result.get('suggestions', [])}")
        print(f"🎯 Improvement areas: {result.get('improvement_areas', [])}")
        
        # Display metrics
        metrics = result.get('metrics', {})
        if metrics:
            print("\n📊 Voice Metrics:")
            for metric, score in metrics.items():
                print(f"  • {metric}: {score:.2f}")
        
    except Exception as e:
        print(f"❌ Voice analysis demonstration failed: {e}")

async def demonstrate_leadership_coaching():
    """Demonstrate leadership coaching"""
    print("\n🏆 === LEADERSHIP COACHING DEMONSTRATION ===")
    
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        return
    
    try:
        # Quick leadership coaching
        print("🎯 Creating leadership training plan...")
        plan = await quick_leadership_coaching(
            api_key=api_key,
            user_id="executive_demo_456",
            target_context="Executive presentations to board members"
        )
        
        if "error" in plan:
            print(f"❌ Leadership coaching failed: {plan['error']}")
            return
        
        print("✅ Leadership training plan created!")
        print(f"📋 Template: {plan.get('template', {}).get('name', 'Unknown')}")
        print(f"🎯 Target confidence: {plan.get('target_confidence', 'unknown')}")
        print(f"⏱️ Timeline: {plan.get('timeline', 'Unknown')}")
        
        # Display milestones
        milestones = plan.get('milestones', [])
        if milestones:
            print("\n📅 Training Milestones:")
            for milestone in milestones:
                print(f"  • {milestone}")
        
    except Exception as e:
        print(f"❌ Leadership coaching demonstration failed: {e}")

async def demonstrate_full_system():
    """Demonstrate full voice coaching system"""
    print("\n🚀 === FULL SYSTEM DEMONSTRATION ===")
    
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Please set OPENROUTER_API_KEY environment variable")
        return
    
    try:
        # Create voice coaching AI system
        print("🔧 Initializing Voice Coaching AI system...")
        voice_ai = create_voice_coaching_ai(
            api_key=api_key,
            model="openai/gpt-4-turbo"
        )
        
        # Initialize the system
        if not await voice_ai.initialize():
            print("❌ Failed to initialize Voice Coaching AI system")
            return
        
        print("✅ Voice Coaching AI system initialized!")
        
        # Get system status
        status = voice_ai.get_system_status()
        print(f"📊 System status: {status['status']}")
        print(f"🔧 Available components: {status['components']}")
        
        # Create mock audio data
        audio_data = create_mock_audio_data()
        
        # Demonstrate voice analysis
        print("\n🎤 Performing voice analysis...")
        analysis = await voice_ai.analyze_voice("demo_user_789", audio_data)
        
        print(f"✅ Analysis completed!")
        print(f"🎯 Tone detected: {analysis.tone_detected.value}")
        print(f"📈 Confidence score: {analysis.confidence_score:.2f}")
        print(f"💡 Suggestions: {analysis.suggestions[:3]}...")  # Show first 3
        
        # Demonstrate coaching session
        print("\n🎯 Starting coaching session...")
        session = await voice_ai.start_coaching_session(
            user_id="demo_user_789",
            focus_area=CoachingFocus.LEADERSHIP_VOICE,
            audio_data=audio_data
        )
        
        print(f"✅ Coaching session started: {session.session_id}")
        
        # Get session progress
        progress = await voice_ai.get_session_progress(session.session_id)
        print(f"📊 Session progress: {progress['status']}")
        print(f"🎯 Focus area: {progress['focus_area']}")
        
        # Demonstrate exercise suggestions
        print("\n💪 Getting exercise suggestions...")
        exercises = await voice_ai.suggest_exercises(
            user_id="demo_user_789",
            focus_area=CoachingFocus.CONFIDENCE_BUILDING
        )
        
        print(f"✅ Found {len(exercises)} exercises")
        for i, exercise in enumerate(exercises[:3], 1):  # Show first 3
            print(f"  {i}. {exercise.name} ({exercise.duration_minutes} min)")
        
        # Demonstrate progress tracking
        print("\n📈 Getting user progress...")
        user_progress = await voice_ai.get_user_progress("demo_user_789")
        
        print(f"✅ Progress retrieved!")
        print(f"🎯 Current tone: {user_progress['current_tone']}")
        print(f"📊 Confidence level: {user_progress['confidence_level']}")
        print(f"📈 Total sessions: {user_progress['total_sessions']}")
        
        # Complete the session
        print("\n✅ Completing coaching session...")
        completed_session = await voice_ai.complete_coaching_session(session.session_id)
        print(f"✅ Session completed: {completed_session.session_id}")
        
        # Get system metrics
        metrics = voice_ai.get_system_metrics()
        print(f"\n📊 System metrics:")
        print(f"  • Total sessions: {metrics.get('service_metrics', {}).get('total_sessions', 0)}")
        print(f"  • Successful analyses: {metrics.get('service_metrics', {}).get('successful_analyses', 0)}")
        
        # Cleanup
        await voice_ai.cleanup()
        print("✅ System cleaned up successfully!")
        
    except Exception as e:
        print(f"❌ Full system demonstration failed: {e}")

async def demonstrate_enhanced_features():
    """Demonstrate enhanced voice coaching features"""
    print("\n🚀 === ENHANCED FEATURES DEMONSTRATION ===")
    
    try:
        # Initialize enhanced system
        voice_ai = create_voice_coaching_ai("your-openrouter-api-key-here")
        
        if not await voice_ai.initialize():
            print("❌ Failed to initialize enhanced system")
            return
        
        print("✅ Enhanced voice coaching system initialized!")
        
        # Mock audio data
        audio_data = create_mock_audio_data()
        user_id = "enhanced_demo_user"
        
        # 1. Enhanced Voice Analysis
        print("\n📊 Performing enhanced voice analysis...")
        analysis = await voice_ai.analyze_voice(user_id, audio_data)
        print(f"✅ Enhanced analysis completed:")
        print(f"   - Tone: {analysis.tone_detected.value}")
        print(f"   - Confidence: {analysis.confidence_score:.2f}")
        print(f"   - Suggestions: {len(analysis.suggestions)} items")
        print(f"   - Improvement areas: {len(analysis.improvement_areas)} items")
        
        # 2. Enhanced Analytics
        print("\n📊 Getting enhanced analytics...")
        analytics = await voice_ai.get_analytics(user_id)
        print(f"✅ Analytics retrieved:")
        print(f"   - Total sessions: {analytics.get('total_sessions', 0)}")
        print(f"   - Total analyses: {analytics.get('total_analyses', 0)}")
        print(f"   - Average confidence: {analytics.get('average_confidence', 0):.2f}")
        print(f"   - Most common tone: {analytics.get('most_common_tone', 'unknown')}")
        print(f"   - Improvement trend: {analytics.get('improvement_trend', {}).get('trend', 'unknown')}")
        
        # 3. Performance Report
        print("\n📈 Getting comprehensive performance report...")
        performance_report = await voice_ai.get_performance_report()
        print(f"✅ Performance report generated:")
        if "error" not in performance_report:
            system_status = performance_report.get("system_status", {})
            print(f"   - System status: {system_status.get('status', 'unknown')}")
            print(f"   - Components available: {len(system_status.get('components', {}))}")
            
            # Enhanced metrics if available
            enhanced_metrics = system_status.get("enhanced_metrics", {})
            if enhanced_metrics:
                request_stats = enhanced_metrics.get("request_stats", {})
                print(f"   - Total requests: {request_stats.get('total_requests', 0)}")
                success_rate = (request_stats.get('successful_requests', 0) / 
                              max(request_stats.get('total_requests', 1), 1)) * 100
                print(f"   - Success rate: {success_rate:.1f}%")
                print(f"   - Average response time: {request_stats.get('average_response_time', 0):.2f}s")
                
                # Cache stats
                cache_stats = enhanced_metrics.get("cache_stats", {})
                print(f"   - Cache utilization: {cache_stats.get('utilization', 0):.1f}%")
                print(f"   - Active cache entries: {cache_stats.get('active_entries', 0)}")
        else:
            print(f"   - Error: {performance_report.get('error')}")
        
        # 4. Enhanced System Status
        print("\n🔍 Getting enhanced system status...")
        status = voice_ai.get_system_status()
        print(f"✅ System status:")
        print(f"   - Status: {status.get('status', 'unknown')}")
        print(f"   - Initialized: {status.get('initialized', False)}")
        print(f"   - Components: {list(status.get('components', {}).keys())}")
        
        # 5. Factory Pattern Demonstration
        print("\n🏭 Demonstrating factory pattern...")
        try:
            from .factories import create_factory_manager
            factory_manager = create_factory_manager()
            system_result = await factory_manager.create_complete_system(
                "your-openrouter-api-key-here",
                "openai/gpt-4-turbo"
            )
            print(f"✅ Factory system creation: {system_result.get('status', 'unknown')}")
        except Exception as e:
            print(f"⚠️ Factory demonstration failed: {e}")
        
        # 6. Utils Demonstration
        print("\n🔧 Demonstrating utility functions...")
        try:
            from .utils import (
                create_audio_processor, create_analytics_tracker, 
                create_cache, create_validator
            )
            
            # Audio processor
            audio_processor = create_audio_processor()
            audio_features = audio_processor.extract_audio_features(audio_data)
            print(f"✅ Audio features extracted: {len(audio_features)} properties")
            
            # Analytics tracker
            analytics_tracker = create_analytics_tracker()
            analytics_tracker.track_event("demo_event", user_id, {"demo": True})
            print(f"✅ Analytics event tracked")
            
            # Cache
            cache = create_cache(max_size=100)
            cache.set("demo_key", "demo_value")
            cache_stats = cache.get_stats()
            print(f"✅ Cache created: {cache_stats['utilization']:.1f}% utilization")
            
            # Validator
            validator = create_validator()
            is_valid_user = validator.validate_user_id(user_id)
            is_valid_audio = validator.validate_audio_data(audio_data)
            print(f"✅ Validation results: user={is_valid_user}, audio={is_valid_audio}")
            
        except Exception as e:
            print(f"⚠️ Utils demonstration failed: {e}")
        
        # Cleanup
        await voice_ai.cleanup()
        print("\n✅ Enhanced features demonstration completed!")
        
    except Exception as e:
        print(f"❌ Enhanced features demonstration failed: {e}")

async def demonstrate_configuration():
    """Demonstrate system configuration"""
    print("\n⚙️ === CONFIGURATION DEMONSTRATION ===")
    
    # Create custom configuration
    config = VoiceCoachingConfig(
        openrouter_api_key="your_api_key_here",
        openrouter_model="openai/gpt-4-turbo",
        max_audio_duration=300,
        analysis_confidence_threshold=0.7,
        enable_real_time_feedback=True,
        enable_voice_enhancement=True,
        enable_progress_tracking=True
    )
    
    print("✅ Custom configuration created!")
    print(f"🤖 Model: {config.openrouter_model}")
    print(f"⏱️ Max audio duration: {config.max_audio_duration} seconds")
    print(f"📊 Confidence threshold: {config.analysis_confidence_threshold}")
    print(f"🔄 Real-time feedback: {config.enable_real_time_feedback}")
    print(f"🎵 Voice enhancement: {config.enable_voice_enhancement}")
    print(f"📈 Progress tracking: {config.enable_progress_tracking}")

def demonstrate_enums():
    """Demonstrate the available enums and types"""
    print("\n🎯 === ENUMS AND TYPES DEMONSTRATION ===")
    
    print("🎤 Voice Tone Types:")
    for tone in VoiceToneType:
        print(f"  • {tone.value}")
    
    print("\n📊 Confidence Levels:")
    for level in ConfidenceLevel:
        print(f"  • {level.value}")
    
    print("\n🎯 Coaching Focus Areas:")
    for focus in CoachingFocus:
        print(f"  • {focus.value}")

async def main():
    """Main demonstration function with enhanced features"""
    print("🎤 ENHANCED VOICE COACHING AI - Example Usage")
    print("=" * 60)
    
    if not VOICE_COACHING_AVAILABLE:
        print("❌ Voice Coaching AI system not available")
        return
    
    # Demonstrate enums and types
    demonstrate_enums()
    
    # Demonstrate configuration
    await demonstrate_configuration()
    
    # Demonstrate voice analysis
    await demonstrate_voice_analysis()
    
    # Demonstrate leadership coaching
    await demonstrate_leadership_coaching()
    
    # Demonstrate full system
    await demonstrate_full_system()
    
    # Demonstrate enhanced features
    await demonstrate_enhanced_features()
    
    print("\n🎉 All demonstrations completed!")
    print("\n💡 To use this enhanced system:")
    print("1. Set your OPENROUTER_API_KEY environment variable")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run this script: python example_usage.py")
    print("4. Explore enhanced features: analytics, caching, performance monitoring")

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main()) 