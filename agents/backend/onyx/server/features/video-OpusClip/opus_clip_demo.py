"""
Opus Clip Enhanced Demo

Demonstrates all the new features including:
- Content Curation Engine (ClipGenius™)
- Speaker Tracking System
- B-roll Integration System
- Platform-specific optimization
"""

import asyncio
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import structlog

# Import processors
from processors.content_curation_engine import ContentCurationEngine
from processors.speaker_tracking_system import SpeakerTrackingSystem, TrackingConfig
from processors.broll_integration_system import BrollIntegrationSystem, BrollConfig
from enhanced_api import OpusClipRequest

logger = structlog.get_logger("opus_clip_demo")

class OpusClipDemo:
    """Demo class for Opus Clip features."""
    
    def __init__(self):
        self.content_curation_engine = ContentCurationEngine()
        self.speaker_tracking_system = SpeakerTrackingSystem()
        self.broll_integration_system = BrollIntegrationSystem()
        self.demo_results = {}
    
    async def run_full_demo(self, video_path: str, content_text: str = None):
        """Run complete Opus Clip demo."""
        logger.info("🎬 Starting Opus Clip Enhanced Demo")
        
        try:
            # Demo 1: Content Curation
            await self.demo_content_curation(video_path)
            
            # Demo 2: Speaker Tracking
            await self.demo_speaker_tracking(video_path)
            
            # Demo 3: B-roll Integration
            if content_text:
                await self.demo_broll_integration(video_path, content_text)
            
            # Demo 4: Full Pipeline
            await self.demo_full_pipeline(video_path, content_text)
            
            # Generate demo report
            await self.generate_demo_report()
            
            logger.info("✅ Opus Clip Enhanced Demo completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
    
    async def demo_content_curation(self, video_path: str):
        """Demo content curation engine."""
        logger.info("🔍 Demo 1: Content Curation Engine (ClipGenius™)")
        
        try:
            start_time = time.time()
            
            # Analyze video for engaging moments
            analysis_result = await self.content_curation_engine.analyze_video(video_path)
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results["content_curation"] = {
                "status": "success",
                "processing_time": processing_time,
                "total_duration": analysis_result.get("total_duration", 0),
                "segments_found": analysis_result.get("segments_found", 0),
                "optimized_clips": analysis_result.get("optimized_clips", 0),
                "clips": analysis_result.get("clips", []),
                "summary": {
                    "best_clip_score": max([clip.get("score", 0) for clip in analysis_result.get("clips", [])], default=0),
                    "average_clip_duration": sum([clip.get("duration", 0) for clip in analysis_result.get("clips", [])]) / max(len(analysis_result.get("clips", [])), 1),
                    "viral_potential": "high" if analysis_result.get("optimized_clips", 0) > 3 else "medium" if analysis_result.get("optimized_clips", 0) > 1 else "low"
                }
            }
            
            logger.info(f"✅ Content curation completed in {processing_time:.2f}s")
            logger.info(f"   Found {analysis_result.get('segments_found', 0)} segments")
            logger.info(f"   Generated {analysis_result.get('optimized_clips', 0)} optimized clips")
            
        except Exception as e:
            logger.error(f"❌ Content curation demo failed: {e}")
            self.demo_results["content_curation"] = {"status": "failed", "error": str(e)}
    
    async def demo_speaker_tracking(self, video_path: str):
        """Demo speaker tracking system."""
        logger.info("👤 Demo 2: Speaker Tracking System")
        
        try:
            start_time = time.time()
            
            # Configure tracking for vertical video (TikTok format)
            tracking_config = TrackingConfig()
            tracking_config.confidence_threshold = 0.7
            tracking_config.tracking_threshold = 0.6
            
            # Process video with speaker tracking
            tracking_result = await self.speaker_tracking_system.process_video(
                video_path,
                f"/tmp/demo_tracked_{int(time.time())}.mp4",
                target_width=1080,
                target_height=1920
            )
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results["speaker_tracking"] = {
                "status": "success",
                "processing_time": processing_time,
                "frames_processed": tracking_result.get("frames_processed", 0),
                "tracking_report": tracking_result.get("tracking_report", {}),
                "target_resolution": tracking_result.get("target_resolution", "1080x1920"),
                "summary": {
                    "tracking_success_rate": tracking_result.get("tracking_report", {}).get("tracking_success_rate", 0),
                    "average_zoom": tracking_result.get("tracking_report", {}).get("average_zoom", 1.0),
                    "tracking_quality": tracking_result.get("tracking_report", {}).get("tracking_quality", "unknown")
                }
            }
            
            logger.info(f"✅ Speaker tracking completed in {processing_time:.2f}s")
            logger.info(f"   Processed {tracking_result.get('frames_processed', 0)} frames")
            logger.info(f"   Tracking success rate: {tracking_result.get('tracking_report', {}).get('tracking_success_rate', 0):.1%}")
            
        except Exception as e:
            logger.error(f"❌ Speaker tracking demo failed: {e}")
            self.demo_results["speaker_tracking"] = {"status": "failed", "error": str(e)}
    
    async def demo_broll_integration(self, video_path: str, content_text: str):
        """Demo B-roll integration system."""
        logger.info("🎥 Demo 3: B-roll Integration System")
        
        try:
            start_time = time.time()
            
            # Configure B-roll integration
            broll_config = BrollConfig()
            broll_config.max_broll_duration = 5.0
            broll_config.confidence_threshold = 0.7
            broll_config.enable_ai_generation = True
            broll_config.enable_stock_footage = True
            
            # Process video with B-roll integration
            broll_result = await self.broll_integration_system.process_video(
                video_path,
                content_text,
                f"/tmp/demo_broll_{int(time.time())}.mp4"
            )
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results["broll_integration"] = {
                "status": "success",
                "processing_time": processing_time,
                "opportunities_found": broll_result.get("opportunities_found", 0),
                "suggestions_generated": broll_result.get("suggestions_generated", 0),
                "integration_report": broll_result.get("integration_report", {}),
                "summary": {
                    "integration_quality": broll_result.get("integration_report", {}).get("integration_quality", "unknown"),
                    "average_confidence": broll_result.get("integration_report", {}).get("average_confidence", 0),
                    "suggestion_types": broll_result.get("integration_report", {}).get("suggestion_counts", {})
                }
            }
            
            logger.info(f"✅ B-roll integration completed in {processing_time:.2f}s")
            logger.info(f"   Found {broll_result.get('opportunities_found', 0)} opportunities")
            logger.info(f"   Generated {broll_result.get('suggestions_generated', 0)} suggestions")
            
        except Exception as e:
            logger.error(f"❌ B-roll integration demo failed: {e}")
            self.demo_results["broll_integration"] = {"status": "failed", "error": str(e)}
    
    async def demo_full_pipeline(self, video_path: str, content_text: str = None):
        """Demo full Opus Clip pipeline."""
        logger.info("🚀 Demo 4: Full Opus Clip Pipeline")
        
        try:
            start_time = time.time()
            
            # Create Opus Clip request
            request = OpusClipRequest(
                video_path=video_path,
                content_text=content_text or "This is a demo video about AI and technology.",
                target_platform="tiktok",
                enable_content_curation=True,
                enable_speaker_tracking=True,
                enable_broll_integration=bool(content_text),
                max_clips=5,
                clip_duration_range=(8, 15),
                output_resolution=(1080, 1920),
                quality="high"
            )
            
            # Simulate full pipeline processing
            pipeline_results = {
                "content_curation": self.demo_results.get("content_curation", {}),
                "speaker_tracking": self.demo_results.get("speaker_tracking", {}),
                "broll_integration": self.demo_results.get("broll_integration", {}),
                "platform_optimization": {
                    "target_platform": request.target_platform,
                    "output_resolution": request.output_resolution,
                    "quality": request.quality
                }
            }
            
            processing_time = time.time() - start_time
            
            # Store results
            self.demo_results["full_pipeline"] = {
                "status": "success",
                "processing_time": processing_time,
                "request": request.dict(),
                "pipeline_results": pipeline_results,
                "summary": {
                    "total_processing_time": sum([
                        result.get("processing_time", 0) 
                        for result in pipeline_results.values() 
                        if isinstance(result, dict) and "processing_time" in result
                    ]),
                    "features_enabled": {
                        "content_curation": request.enable_content_curation,
                        "speaker_tracking": request.enable_speaker_tracking,
                        "broll_integration": request.enable_broll_integration
                    },
                    "target_platform": request.target_platform,
                    "output_quality": request.quality
                }
            }
            
            logger.info(f"✅ Full pipeline completed in {processing_time:.2f}s")
            logger.info(f"   Target platform: {request.target_platform}")
            logger.info(f"   Output resolution: {request.output_resolution}")
            logger.info(f"   Quality: {request.quality}")
            
        except Exception as e:
            logger.error(f"❌ Full pipeline demo failed: {e}")
            self.demo_results["full_pipeline"] = {"status": "failed", "error": str(e)}
    
    async def generate_demo_report(self):
        """Generate comprehensive demo report."""
        logger.info("📊 Generating Demo Report")
        
        try:
            report = {
                "demo_info": {
                    "timestamp": time.time(),
                    "version": "2.0.0",
                    "features_demoed": [
                        "Content Curation Engine (ClipGenius™)",
                        "Speaker Tracking System",
                        "B-roll Integration System",
                        "Full Opus Clip Pipeline"
                    ]
                },
                "results": self.demo_results,
                "summary": {
                    "total_demos": len(self.demo_results),
                    "successful_demos": len([r for r in self.demo_results.values() if r.get("status") == "success"]),
                    "failed_demos": len([r for r in self.demo_results.values() if r.get("status") == "failed"]),
                    "overall_success_rate": len([r for r in self.demo_results.values() if r.get("status") == "success"]) / max(len(self.demo_results), 1)
                },
                "performance_metrics": self._calculate_performance_metrics(),
                "feature_analysis": self._analyze_features()
            }
            
            # Save report
            report_path = f"/tmp/opus_clip_demo_report_{int(time.time())}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Demo report saved to: {report_path}")
            
            # Print summary
            self._print_demo_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Demo report generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics from demo results."""
        try:
            metrics = {
                "content_curation": {
                    "processing_time": self.demo_results.get("content_curation", {}).get("processing_time", 0),
                    "clips_generated": self.demo_results.get("content_curation", {}).get("optimized_clips", 0),
                    "efficiency": "high" if self.demo_results.get("content_curation", {}).get("processing_time", 0) < 30 else "medium"
                },
                "speaker_tracking": {
                    "processing_time": self.demo_results.get("speaker_tracking", {}).get("processing_time", 0),
                    "tracking_accuracy": self.demo_results.get("speaker_tracking", {}).get("summary", {}).get("tracking_success_rate", 0),
                    "efficiency": "high" if self.demo_results.get("speaker_tracking", {}).get("processing_time", 0) < 60 else "medium"
                },
                "broll_integration": {
                    "processing_time": self.demo_results.get("broll_integration", {}).get("processing_time", 0),
                    "suggestions_generated": self.demo_results.get("broll_integration", {}).get("suggestions_generated", 0),
                    "efficiency": "high" if self.demo_results.get("broll_integration", {}).get("processing_time", 0) < 45 else "medium"
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    def _analyze_features(self) -> Dict[str, Any]:
        """Analyze feature performance and quality."""
        try:
            analysis = {
                "content_curation_quality": "high" if self.demo_results.get("content_curation", {}).get("summary", {}).get("viral_potential") == "high" else "medium",
                "speaker_tracking_quality": self.demo_results.get("speaker_tracking", {}).get("summary", {}).get("tracking_quality", "unknown"),
                "broll_integration_quality": self.demo_results.get("broll_integration", {}).get("summary", {}).get("integration_quality", "unknown"),
                "overall_readiness": "production_ready" if all(
                    result.get("status") == "success" 
                    for result in self.demo_results.values()
                ) else "needs_improvement"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            return {}
    
    def _print_demo_summary(self, report: Dict[str, Any]):
        """Print demo summary to console."""
        print("\n" + "="*60)
        print("🎬 OPUS CLIP ENHANCED DEMO SUMMARY")
        print("="*60)
        
        # Overall status
        summary = report.get("summary", {})
        print(f"✅ Successful demos: {summary.get('successful_demos', 0)}/{summary.get('total_demos', 0)}")
        print(f"📊 Success rate: {summary.get('overall_success_rate', 0):.1%}")
        
        # Feature status
        print("\n🔍 FEATURE STATUS:")
        for feature, result in self.demo_results.items():
            status = "✅" if result.get("status") == "success" else "❌"
            print(f"   {status} {feature.replace('_', ' ').title()}")
        
        # Performance metrics
        print("\n⚡ PERFORMANCE METRICS:")
        metrics = report.get("performance_metrics", {})
        for feature, metric in metrics.items():
            if isinstance(metric, dict) and "processing_time" in metric:
                print(f"   {feature.replace('_', ' ').title()}: {metric['processing_time']:.2f}s")
        
        # Quality analysis
        print("\n🎯 QUALITY ANALYSIS:")
        analysis = report.get("feature_analysis", {})
        for metric, value in analysis.items():
            print(f"   {metric.replace('_', ' ').title()}: {value}")
        
        print("\n" + "="*60)
        print("🚀 Opus Clip Enhanced is ready for viral content creation!")
        print("="*60 + "\n")

async def main():
    """Main demo function."""
    # Initialize demo
    demo = OpusClipDemo()
    
    # Demo video path (replace with actual video)
    video_path = "/path/to/your/demo/video.mp4"
    content_text = """
    Welcome to the future of AI-powered video editing! 
    Today we're exploring how artificial intelligence can transform 
    long-form content into engaging short-form videos that go viral.
    Our advanced algorithms analyze every frame, track speakers perfectly,
    and automatically insert relevant B-roll footage to create
    professional-quality content that captures attention.
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
        return
    
    try:
        # Run demo
        await demo.run_full_demo(video_path, content_text)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    # Configure logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Run demo
    asyncio.run(main())


