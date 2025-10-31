"""
🎤 VOICE COACHING AI - MAIN MODULE
==================================

Advanced AI-powered voice coaching system for leadership development,
confidence building, and professional voice training using OpenRouter.

FEATURES:
- 🎤 Real-time voice analysis and tone detection
- 🏆 Leadership voice training and coaching
- 📊 Confidence building and progress tracking
- 🎯 Personalized exercise recommendations
- 📈 Comprehensive progress analytics
- 🤖 OpenRouter AI integration for intelligent coaching
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# =============================================================================
# 🏗️ MODULAR IMPORTS
# =============================================================================

# Core architecture
try:
    from .core import (
        VoiceToneType, ConfidenceLevel, CoachingFocus, VoiceAnalysisMetrics,
        VoiceProfile, VoiceAnalysis, CoachingSession, VoiceExercise, LeadershipVoiceTemplate,
        VoiceCoachingConfig, PerformanceMetrics, VoiceCoachingComponent,
        create_default_voice_config, validate_voice_profile, calculate_progress_score
    )
    CORE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("🎤 Core voice coaching architecture loaded")
except ImportError as e:
    CORE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ Core voice coaching architecture not available: {e}")

# Engine management
try:
    from .engines.openrouter_voice_engine import OpenRouterVoiceEngine, create_voice_coaching_engine
    ENGINES_AVAILABLE = True
    logger.info("🚀 Voice coaching engines loaded")
except ImportError as e:
    ENGINES_AVAILABLE = False
    logger.warning(f"⚠️ Voice coaching engines not available: {e}")

# Service layer
try:
    from .services.voice_coaching_service import VoiceCoachingService, create_voice_coaching_service
    SERVICES_AVAILABLE = True
    logger.info("🔧 Voice coaching services loaded")
except ImportError as e:
    SERVICES_AVAILABLE = False
    logger.warning(f"⚠️ Voice coaching services not available: {e}")

# Factory management
try:
    from .factories import (
        VoiceCoachingFactory, EngineFactory, ServiceFactory, VoiceCoachingFactoryManager,
        create_voice_coaching_factory, create_engine_factory, create_service_factory,
        create_factory_manager, create_voice_coaching_system, create_voice_engine
    )
    FACTORIES_AVAILABLE = True
    logger.info("🏭 Voice coaching factories loaded")
except ImportError as e:
    FACTORIES_AVAILABLE = False
    logger.warning(f"⚠️ Voice coaching factories not available: {e}")

# Utils management
try:
    from .utils import (
        AudioProcessor, AnalyticsTracker, VoiceCoachingCache, VoiceCoachingValidator,
        ErrorHandler, DataTransformer, PerformanceMonitor,
        create_audio_processor, create_analytics_tracker, create_cache,
        create_validator, create_performance_monitor, create_data_transformer
    )
    UTILS_AVAILABLE = True
    logger.info("🔧 Voice coaching utilities loaded")
except ImportError as e:
    UTILS_AVAILABLE = False
    logger.warning(f"⚠️ Voice coaching utilities not available: {e}")

# =============================================================================
# 🎯 UNIFIED VOICE COACHING SYSTEM
# =============================================================================

class VoiceCoachingAI:
    """
    Unified Voice Coaching AI System
    
    Provides comprehensive voice coaching capabilities including:
    - Voice analysis and tone detection
    - Leadership voice training
    - Confidence building
    - Progress tracking
    - Personalized coaching sessions
    """
    
    def __init__(self, config: Optional[VoiceCoachingConfig] = None):
        self.config = config or create_default_voice_config()
        self.service: Optional[VoiceCoachingService] = None
        self.engine: Optional[OpenRouterVoiceEngine] = None
        self.logger = logging.getLogger(__name__)
        
        # System status
        self.initialized = False
        self.available_components = {
            "core": CORE_AVAILABLE,
            "engines": ENGINES_AVAILABLE,
            "services": SERVICES_AVAILABLE,
            "factories": FACTORIES_AVAILABLE,
            "utils": UTILS_AVAILABLE
        }
    
    async def initialize(self) -> bool:
        """Initialize the Voice Coaching AI system with enhanced features"""
        try:
            self.logger.info("🎤 Initializing Enhanced Voice Coaching AI System...")
            
            # Validate configuration
            if not self.config.openrouter_api_key:
                raise ValueError("OpenRouter API key is required for voice coaching")
            
            # Initialize with factory pattern if available
            if FACTORIES_AVAILABLE:
                try:
                    factory_manager = create_factory_manager()
                    system_result = await factory_manager.create_complete_system(
                        self.config.openrouter_api_key,
                        self.config.openrouter_model
                    )
                    
                    if system_result.get("status") == "initialized":
                        self.service = system_result["service"]
                        self.engine = system_result["engine"]
                        self.logger.info("✅ Voice Coaching AI System initialized with factory pattern")
                        self.initialized = True
                        return True
                    else:
                        self.logger.warning(f"Factory initialization failed: {system_result.get('error')}")
                except Exception as e:
                    self.logger.warning(f"Factory initialization failed, falling back to direct initialization: {e}")
            
            # Fallback to direct initialization
            if SERVICES_AVAILABLE:
                self.service = create_voice_coaching_service(self.config)
                if not await self.service.initialize():
                    raise RuntimeError("Failed to initialize voice coaching service")
                self.logger.info("✅ Voice coaching service initialized")
            
            # Initialize engine directly if needed
            if ENGINES_AVAILABLE and not self.service:
                self.engine = create_voice_coaching_engine(self.config)
                if not await self.engine.initialize():
                    raise RuntimeError("Failed to initialize voice coaching engine")
                self.logger.info("✅ Voice coaching engine initialized")
            
            self.initialized = True
            self.logger.info("🎤 Voice Coaching AI System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Voice Coaching AI System: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup system resources"""
        try:
            if self.service:
                await self.service.cleanup()
            elif self.engine:
                await self.engine.cleanup()
            
            self.logger.info("🎤 Voice Coaching AI System cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    # =============================================================================
    # 🎤 VOICE ANALYSIS METHODS
    # =============================================================================
    
    async def analyze_voice(self, user_id: str, audio_data: bytes) -> VoiceAnalysis:
        """Analyze user's voice and provide comprehensive feedback"""
        if not self.initialized:
            raise RuntimeError("Voice Coaching AI not initialized")
        
        if self.service:
            return await self.service.analyze_user_voice(user_id, audio_data)
        elif self.engine:
            return await self.engine.analyze_voice(audio_data, user_id)
        else:
            raise RuntimeError("No voice analysis engine available")
    
    async def detect_tone(self, audio_data: bytes) -> VoiceToneType:
        """Detect voice tone"""
        if not self.initialized or not self.engine:
            raise RuntimeError("Voice coaching engine not available")
        
        return await self.engine.detect_tone(audio_data)
    
    async def measure_confidence(self, audio_data: bytes) -> float:
        """Measure confidence level in voice"""
        if not self.initialized or not self.engine:
            raise RuntimeError("Voice coaching engine not available")
        
        return await self.engine.measure_confidence(audio_data)
    
    # =============================================================================
    # 🎯 COACHING SESSION METHODS
    # =============================================================================
    
    async def start_coaching_session(
        self, 
        user_id: str, 
        focus_area: CoachingFocus,
        audio_data: bytes
    ) -> CoachingSession:
        """Start a new voice coaching session"""
        if not self.initialized or not self.service:
            raise RuntimeError("Voice coaching service not available")
        
        return await self.service.start_coaching_session(user_id, focus_area, audio_data)
    
    async def complete_coaching_session(self, session_id: str) -> CoachingSession:
        """Complete a coaching session"""
        if not self.initialized or not self.service:
            raise RuntimeError("Voice coaching service not available")
        
        return await self.service.complete_coaching_session(session_id)
    
    async def get_session_progress(self, session_id: str) -> Dict[str, Any]:
        """Get progress for a coaching session"""
        if not self.initialized or not self.service:
            raise RuntimeError("Voice coaching service not available")
        
        return await self.service.get_session_progress(session_id)
    
    # =============================================================================
    # 🏆 LEADERSHIP TRAINING METHODS
    # =============================================================================
    
    async def create_leadership_training_plan(
        self, 
        user_id: str, 
        target_context: str
    ) -> Dict[str, Any]:
        """Create a personalized leadership training plan"""
        if not self.initialized or not self.service:
            raise RuntimeError("Voice coaching service not available")
        
        return await self.service.create_leadership_training_plan(user_id, target_context)
    
    async def suggest_exercises(
        self, 
        user_id: str, 
        focus_area: CoachingFocus
    ) -> List[VoiceExercise]:
        """Suggest personalized exercises for user"""
        if not self.initialized or not self.service:
            raise RuntimeError("Voice coaching service not available")
        
        return await self.service.suggest_exercises_for_user(user_id, focus_area)
    
    # =============================================================================
    # 📊 PROGRESS TRACKING METHODS
    # =============================================================================
    
    async def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user progress data"""
        if not self.initialized or not self.service:
            raise RuntimeError("Voice coaching service not available")
        
        return await self.service.get_user_progress(user_id)
    
    async def compare_sessions(
        self, 
        user_id: str, 
        session_id_1: str, 
        session_id_2: str
    ) -> Dict[str, Any]:
        """Compare two coaching sessions"""
        if not self.initialized or not self.service:
            raise RuntimeError("Voice coaching service not available")
        
        return await self.service.compare_sessions(user_id, session_id_1, session_id_2)
    
    # =============================================================================
    # 🔧 UTILITY METHODS
    # =============================================================================
    
    def get_user_profile(self, user_id: str) -> Optional[VoiceProfile]:
        """Get user voice profile"""
        if not self.initialized or not self.service:
            return None
        
        return self.service.get_user_profile(user_id)
    
    def get_active_sessions(self, user_id: str) -> List[CoachingSession]:
        """Get active sessions for user"""
        if not self.initialized or not self.service:
            return []
        
        return self.service.get_active_sessions(user_id)
    
    def get_coaching_history(self, user_id: str) -> List[CoachingSession]:
        """Get coaching history for user"""
        if not self.initialized or not self.service:
            return []
        
        return self.service.get_coaching_history(user_id)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        metrics = {
            "initialized": self.initialized,
            "available_components": self.available_components,
            "config": {
                "openrouter_model": self.config.openrouter_model,
                "max_audio_duration": self.config.max_audio_duration,
                "enable_real_time_feedback": self.config.enable_real_time_feedback
            }
        }
        
        if self.service:
            service_metrics = self.service.get_service_metrics()
            metrics["service_metrics"] = {
                "total_sessions": service_metrics.total_sessions,
                "successful_analyses": service_metrics.successful_analyses,
                "average_confidence_improvement": service_metrics.average_confidence_improvement,
                "error_rate": service_metrics.error_rate
            }
        
        return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get enhanced system status and health"""
        status = {
            "status": "healthy" if self.initialized else "not_initialized",
            "components": self.available_components,
            "initialized": self.initialized,
            "config_loaded": bool(self.config.openrouter_api_key)
        }
        
        # Add enhanced metrics if available
        if self.engine and hasattr(self.engine, 'get_enhanced_metrics'):
            try:
                status["enhanced_metrics"] = self.engine.get_enhanced_metrics()
            except Exception as e:
                status["enhanced_metrics_error"] = str(e)
        
        return status
    
    async def get_analytics(self, user_id: str = None) -> Dict[str, Any]:
        """Get analytics data"""
        if not self.engine:
            return {"error": "Engine not initialized"}
        
        try:
            if user_id:
                return await self.engine.get_user_analytics(user_id)
            else:
                return await self.engine.get_system_analytics()
        except Exception as e:
            return {"error": f"Analytics retrieval failed: {e}"}
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        if not self.engine:
            return {"error": "Engine not initialized"}
        
        try:
            return {
                "system_status": self.get_system_status(),
                "analytics": await self.get_analytics(),
                "performance_metrics": self.engine.get_enhanced_metrics() if hasattr(self.engine, 'get_enhanced_metrics') else {}
            }
        except Exception as e:
            return {"error": f"Performance report generation failed: {e}"}

# =============================================================================
# 🏭 FACTORY FUNCTIONS
# =============================================================================

def create_voice_coaching_ai(
    api_key: str,
    model: str = "openai/gpt-4-turbo",
    **kwargs
) -> VoiceCoachingAI:
    """Create Voice Coaching AI system with custom configuration"""
    
    config = VoiceCoachingConfig(
        openrouter_api_key=api_key,
        openrouter_model=model,
        **kwargs
    )
    
    return VoiceCoachingAI(config)

def create_default_voice_coaching_ai() -> VoiceCoachingAI:
    """Create Voice Coaching AI system with default configuration"""
    config = create_default_voice_config()
    return VoiceCoachingAI(config)

# =============================================================================
# 🚀 QUICK START FUNCTIONS
# =============================================================================

async def quick_voice_analysis(
    api_key: str,
    user_id: str,
    audio_data: bytes
) -> Dict[str, Any]:
    """Quick voice analysis without full system initialization"""
    try:
        # Create minimal engine for analysis
        config = VoiceCoachingConfig(openrouter_api_key=api_key)
        engine = OpenRouterVoiceEngine(config)
        
        if not await engine.initialize():
            raise RuntimeError("Failed to initialize voice engine")
        
        # Perform analysis
        analysis = await engine.analyze_voice(audio_data, user_id)
        
        # Cleanup
        await engine.cleanup()
        
        return {
            "tone": analysis.tone_detected.value,
            "confidence_score": analysis.confidence_score,
            "suggestions": analysis.suggestions,
            "improvement_areas": analysis.improvement_areas,
            "metrics": {k.value: v for k, v in analysis.metrics.items()}
        }
        
    except Exception as e:
        return {"error": str(e)}

async def quick_leadership_coaching(
    api_key: str,
    user_id: str,
    target_context: str
) -> Dict[str, Any]:
    """Quick leadership coaching session"""
    try:
        # Create service for coaching
        config = VoiceCoachingConfig(openrouter_api_key=api_key)
        service = VoiceCoachingService(config)
        
        if not await service.initialize():
            raise RuntimeError("Failed to initialize voice coaching service")
        
        # Create training plan
        training_plan = await service.create_leadership_training_plan(user_id, target_context)
        
        # Cleanup
        await service.cleanup()
        
        return training_plan
        
    except Exception as e:
        return {"error": str(e)}

# =============================================================================
# 📦 MODULE EXPORTS
# =============================================================================

__version__ = "1.0.0"
__author__ = "Blatam Academy"
__description__ = "Advanced AI-powered voice coaching system for leadership development"

__all__ = [
    # Main system
    'VoiceCoachingAI',
    'create_voice_coaching_ai',
    'create_default_voice_coaching_ai',
    
    # Quick start functions
    'quick_voice_analysis',
    'quick_leadership_coaching',
    
    # Core components
    'VoiceToneType',
    'ConfidenceLevel',
    'CoachingFocus',
    'VoiceAnalysisMetrics',
    'VoiceProfile',
    'VoiceAnalysis',
    'CoachingSession',
    'VoiceExercise',
    'LeadershipVoiceTemplate',
    'VoiceCoachingConfig',
    'PerformanceMetrics',
    
    # Utilities
    'create_default_voice_config',
    'validate_voice_profile',
    'calculate_progress_score',
    
    # System status
    'CORE_AVAILABLE',
    'ENGINES_AVAILABLE',
    'SERVICES_AVAILABLE',
] 