#!/usr/bin/env python3
"""
Enhanced Integration Manager
============================

Integrates all advanced HeyGen AI features into a unified system:
- Advanced body animation
- Real-time expression control
- Advanced accent and dialect system
- Enhanced performance optimization
- Unified API interface
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import json

# Import the new advanced services
from .advanced_body_animation import AdvancedBodyAnimationService, GestureConfig, AnimationSequence
from .advanced_expression_controller import AdvancedExpressionController, EmotionType, FacialPose
from .advanced_accent_system import AdvancedAccentSystem, AccentRegion, DialectType, AccentProfile
from .enhanced_performance_optimizer import EnhancedPerformanceOptimizer, CacheLevel, LoadBalancingStrategy

logger = logging.getLogger(__name__)

@dataclass
class EnhancedVideoRequest:
    """Enhanced video generation request with all new features."""
    # Basic content
    script_text: str
    language: str = "en"
    
    # Avatar settings
    avatar_style: str = "realistic"
    avatar_customization: Optional[Dict[str, Any]] = None
    
    # Voice settings
    voice_id: Optional[str] = None
    accent_region: Optional[AccentRegion] = None
    dialect_type: DialectType = DialectType.FORMAL
    accent_intensity: float = 1.0
    
    # Animation settings
    enable_body_animation: bool = True
    enable_facial_expressions: bool = True
    animation_style: str = "natural"
    
    # Performance settings
    enable_caching: bool = True
    enable_load_balancing: bool = True
    background_processing: bool = True
    
    # Output settings
    video_quality: str = "high"
    resolution: str = "1080p"
    output_format: str = "mp4"

@dataclass
class EnhancedVideoResult:
    """Enhanced video generation result."""
    success: bool
    video_path: Optional[str] = None
    animation_data: Optional[Dict[str, Any]] = None
    expression_data: Optional[Dict[str, Any]] = None
    accent_data: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    processing_time: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedIntegrationManager:
    """
    Manages integration of all advanced HeyGen AI features.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize all advanced services
        self.body_animation_service = AdvancedBodyAnimationService()
        self.expression_controller = AdvancedExpressionController()
        self.accent_system = AdvancedAccentSystem()
        self.performance_optimizer = EnhancedPerformanceOptimizer(self.config.get("performance", {}))
        
        # Integration state
        self.integration_status = {
            "body_animation": "initialized",
            "expressions": "initialized",
            "accents": "initialized",
            "performance": "initialized"
        }
        
        # Performance monitoring
        self.request_count = 0
        self.success_count = 0
        self.average_processing_time = 0.0
        
        logger.info("Enhanced Integration Manager initialized")
    
    async def generate_enhanced_video(self, request: EnhancedVideoRequest) -> EnhancedVideoResult:
        """
        Generate enhanced video with all advanced features.
        
        Args:
            request: Enhanced video generation request
            
        Returns:
            Enhanced video generation result
        """
        start_time = time.time()
        self.request_count += 1
        
        try:
            logger.info(f"Starting enhanced video generation for request {self.request_count}")
            
            # Step 1: Analyze script for emotions and gestures
            emotions = await self.expression_controller.analyze_text_emotion(request.script_text)
            gestures = await self.body_animation_service.analyze_script_for_gestures(request.script_text)
            
            # Step 2: Generate accent profile if specified
            accent_profile = None
            if request.accent_region:
                accent_profile = await self.accent_system.generate_accent_profile(
                    request.accent_region,
                    request.dialect_type,
                    request.accent_intensity
                )
            
            # Step 3: Apply accent modifications to script
            modified_script = request.script_text
            if accent_profile:
                modified_script = await self.accent_system.apply_accent_to_text(
                    request.script_text,
                    accent_profile
                )
            
            # Step 4: Generate body animation sequence
            animation_sequence = None
            if request.enable_body_animation and gestures:
                animation_sequence = await self.body_animation_service.generate_body_animation_sequence(
                    gestures,
                    "neutral",  # Base emotion
                    request.avatar_style
                )
            
            # Step 5: Generate facial expression sequence
            expression_sequence = None
            if request.enable_facial_expressions and emotions:
                expression_sequence = await self.expression_controller.generate_expression_sequence(emotions)
            
            # Step 6: Cache results for performance
            if request.enable_caching:
                cache_key = f"enhanced_video_{hash(request.script_text)}"
                await self.performance_optimizer.set_cached_value(
                    cache_key,
                    {
                        "animation_sequence": animation_sequence,
                        "expression_sequence": expression_sequence,
                        "accent_profile": accent_profile,
                        "modified_script": modified_script
                    },
                    ttl=3600,  # 1 hour
                    level=CacheLevel.L1_MEMORY
                )
            
            # Step 7: Submit background processing if enabled
            if request.background_processing:
                task_id = await self.performance_optimizer.submit_background_task(
                    self._process_video_background,
                    request,
                    animation_sequence,
                    expression_sequence,
                    accent_profile
                )
                logger.info(f"Background video processing task submitted: {task_id}")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            self.average_processing_time = (
                (self.average_processing_time * (self.request_count - 1) + processing_time) / self.request_count
            )
            
            # Success
            self.success_count += 1
            
            result = EnhancedVideoResult(
                success=True,
                animation_data=animation_sequence.__dict__ if animation_sequence else None,
                expression_data=expression_sequence if expression_sequence else None,
                accent_data=accent_profile.__dict__ if accent_profile else None,
                performance_metrics={
                    "processing_time": processing_time,
                    "cache_enabled": request.enable_caching,
                    "background_processing": request.background_processing
                },
                processing_time=processing_time,
                metadata={
                    "request_id": self.request_count,
                    "features_used": self._get_used_features(request),
                    "generated_at": time.time()
                }
            )
            
            logger.info(f"Enhanced video generation completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error in enhanced video generation: {e}"
            logger.error(error_msg)
            
            result = EnhancedVideoResult(
                success=False,
                processing_time=processing_time,
                error_message=error_msg,
                metadata={
                    "request_id": self.request_count,
                    "error_type": type(e).__name__,
                    "generated_at": time.time()
                }
            )
            
            return result
    
    def _get_used_features(self, request: EnhancedVideoRequest) -> List[str]:
        """Get list of features used in the request."""
        features = []
        
        if request.enable_body_animation:
            features.append("body_animation")
        if request.enable_facial_expressions:
            features.append("facial_expressions")
        if request.accent_region:
            features.append("accent_system")
        if request.enable_caching:
            features.append("performance_caching")
        if request.enable_load_balancing:
            features.append("load_balancing")
        if request.background_processing:
            features.append("background_processing")
        
        return features
    
    async def _process_video_background(
        self, 
        request: EnhancedVideoRequest,
        animation_sequence: Optional[AnimationSequence],
        expression_sequence: Optional[Dict[str, Any]],
        accent_profile: Optional[AccentProfile]
    ):
        """Background video processing task."""
        try:
            logger.info("Starting background video processing")
            
            # Simulate video processing
            await asyncio.sleep(2)
            
            # Export animation data
            if animation_sequence:
                animation_json = await self.body_animation_service.export_animation_sequence(
                    animation_sequence, "json"
                )
                logger.info(f"Exported animation sequence: {len(animation_json)} characters")
            
            # Export accent profile
            if accent_profile:
                accent_json = await self.accent_system.export_accent_profile(accent_profile, "json")
                logger.info(f"Exported accent profile: {len(accent_json)} characters")
            
            logger.info("Background video processing completed")
            
        except Exception as e:
            logger.error(f"Background video processing error: {e}")
    
    async def get_integration_status(self) -> Dict[str, Any]:
        """Get status of all integrated services."""
        try:
            # Get statistics from all services
            body_anim_stats = await self.body_animation_service.get_animation_statistics()
            expression_stats = await self.expression_controller.get_expression_statistics()
            accent_stats = await self.accent_system.get_accent_statistics()
            performance_stats = await self.performance_optimizer.get_performance_statistics()
            
            return {
                "integration_status": self.integration_status,
                "service_statistics": {
                    "body_animation": body_anim_stats,
                    "expressions": expression_stats,
                    "accents": accent_stats,
                    "performance": performance_stats
                },
                "performance_metrics": {
                    "total_requests": self.request_count,
                    "successful_requests": self.success_count,
                    "success_rate": self.success_count / self.request_count if self.request_count > 0 else 0.0,
                    "average_processing_time": self.average_processing_time
                },
                "features_available": [
                    "advanced_body_animation",
                    "real_time_expressions",
                    "accent_dialect_system",
                    "multi_level_caching",
                    "load_balancing",
                    "background_processing",
                    "performance_monitoring"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {"error": str(e)}
    
    async def test_integration(self) -> Dict[str, Any]:
        """Test integration of all services."""
        try:
            test_results = {}
            
            # Test body animation service
            try:
                test_script = "Hello! This is a test script for body animation."
                gestures = await self.body_animation_service.analyze_script_for_gestures(test_script)
                animation_sequence = await self.body_animation_service.generate_body_animation_sequence(
                    gestures, "neutral", "realistic"
                )
                test_results["body_animation"] = {
                    "status": "success",
                    "gestures_found": len(gestures),
                    "animation_poses": len(animation_sequence.poses)
                }
            except Exception as e:
                test_results["body_animation"] = {"status": "failed", "error": str(e)}
            
            # Test expression controller
            try:
                emotions = await self.expression_controller.analyze_text_emotion(test_script)
                expression_sequence = await self.expression_controller.generate_expression_sequence(emotions)
                test_results["expressions"] = {
                    "status": "success",
                    "emotions_found": len(emotions),
                    "expression_poses": len(expression_sequence["poses"])
                }
            except Exception as e:
                test_results["expressions"] = {"status": "failed", "error": str(e)}
            
            # Test accent system
            try:
                accent_profile = await self.accent_system.generate_accent_profile(
                    AccentRegion.AMERICAN_GENERAL,
                    DialectType.FORMAL,
                    1.0
                )
                modified_text = await self.accent_system.apply_accent_to_text(
                    test_script, accent_profile
                )
                test_results["accents"] = {
                    "status": "success",
                    "accent_id": accent_profile.accent_id,
                    "text_modified": modified_text != test_script
                }
            except Exception as e:
                test_results["accents"] = {"status": "failed", "error": str(e)}
            
            # Test performance optimizer
            try:
                await self.performance_optimizer.set_cached_value("test_key", "test_value")
                cached_value = await self.performance_optimizer.get_cached_value("test_key")
                test_results["performance"] = {
                    "status": "success",
                    "cache_set": True,
                    "cache_get": cached_value == "test_value"
                }
            except Exception as e:
                test_results["performance"] = {"status": "failed", "error": str(e)}
            
            # Overall test result
            overall_success = all(
                result["status"] == "success" 
                for result in test_results.values()
            )
            
            test_results["overall"] = {
                "status": "success" if overall_success else "failed",
                "services_tested": len(test_results),
                "services_passed": sum(1 for r in test_results.values() if r["status"] == "success")
            }
            
            logger.info(f"Integration test completed: {test_results['overall']['services_passed']}/{test_results['overall']['services_tested']} services passed")
            return test_results
            
        except Exception as e:
            logger.error(f"Error during integration test: {e}")
            return {"error": str(e)}
    
    async def optimize_all_services(self):
        """Run optimization for all integrated services."""
        try:
            logger.info("Starting optimization of all integrated services")
            
            # Optimize performance
            await self.performance_optimizer.optimize_performance()
            
            # Clear caches
            await self.body_animation_service.clear_cache()
            await self.expression_controller.clear_cache()
            await self.accent_system.clear_cache()
            await self.performance_optimizer.clear_cache()
            
            logger.info("All services optimization completed")
            
        except Exception as e:
            logger.error(f"Error during services optimization: {e}")
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all services."""
        try:
            health_status = {}
            
            # Check each service
            services = [
                ("body_animation", self.body_animation_service),
                ("expressions", self.expression_controller),
                ("accents", self.accent_system),
                ("performance", self.performance_optimizer)
            ]
            
            for service_name, service in services:
                try:
                    # Try to get statistics from each service
                    if hasattr(service, 'get_animation_statistics'):
                        stats = await service.get_animation_statistics()
                        health_status[service_name] = {"status": "healthy", "stats": stats}
                    elif hasattr(service, 'get_expression_statistics'):
                        stats = await service.get_expression_statistics()
                        health_status[service_name] = {"status": "healthy", "stats": stats}
                    elif hasattr(service, 'get_accent_statistics'):
                        stats = await service.get_accent_statistics()
                        health_status[service_name] = {"status": "healthy", "stats": stats}
                    elif hasattr(service, 'get_performance_statistics'):
                        stats = await service.get_performance_statistics()
                        health_status[service_name] = {"status": "healthy", "stats": stats}
                    else:
                        health_status[service_name] = {"status": "unknown", "error": "No health check method"}
                        
                except Exception as e:
                    health_status[service_name] = {"status": "unhealthy", "error": str(e)}
            
            # Overall health
            healthy_services = sum(1 for status in health_status.values() if status["status"] == "healthy")
            total_services = len(health_status)
            
            health_status["overall"] = {
                "status": "healthy" if healthy_services == total_services else "degraded",
                "healthy_services": healthy_services,
                "total_services": total_services,
                "health_percentage": (healthy_services / total_services) * 100
            }
            
            return health_status
            
        except Exception as e:
            logger.error(f"Error getting service health: {e}")
            return {"error": str(e)}
    
    async def shutdown(self):
        """Shutdown all integrated services."""
        try:
            logger.info("Shutting down Enhanced Integration Manager")
            
            # Clear all caches
            await self.optimize_all_services()
            
            # Update integration status
            self.integration_status = {service: "shutdown" for service in self.integration_status}
            
            logger.info("Enhanced Integration Manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

