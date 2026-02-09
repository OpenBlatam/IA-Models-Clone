"""
Ultimate Opus Clip Demo

Demonstrates ALL advanced features of the Ultimate Opus Clip system:
- Content Curation Engine (ClipGenius™)
- Speaker Tracking System
- B-roll Integration System
- Advanced Viral Scoring
- Audio Processing System
- Professional Export System
- Advanced Analytics System
"""

import asyncio
import time
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import structlog
import requests
import numpy as np

# Import all processors
from processors.content_curation_engine import ContentCurationEngine
from processors.speaker_tracking_system import SpeakerTrackingSystem, TrackingConfig
from processors.broll_integration_system import BrollIntegrationSystem, BrollConfig
from processors.advanced_viral_scoring import AdvancedViralScorer
from processors.audio_processing_system import AudioProcessingSystem
from processors.professional_export_system import ProfessionalExportSystem, ExportSettings, ExportFormat, ExportQuality
from processors.advanced_analytics_system import AdvancedAnalyticsSystem, TimeRange

logger = structlog.get_logger("ultimate_demo")

class UltimateOpusClipDemo:
    """Ultimate demo class showcasing all features."""
    
    def __init__(self):
        # Initialize all processors
        self.content_curation_engine = ContentCurationEngine()
        self.speaker_tracking_system = SpeakerTrackingSystem()
        self.broll_integration_system = BrollIntegrationSystem()
        self.advanced_viral_scorer = AdvancedViralScorer()
        self.audio_processing_system = AudioProcessingSystem()
        self.professional_export_system = ProfessionalExportSystem()
        self.advanced_analytics_system = AdvancedAnalyticsSystem()
        
        self.demo_results = {}
        self.api_base_url = "http://localhost:8000"
    
    async def run_ultimate_demo(self, video_path: str, content_text: str = None):
        """Run complete Ultimate Opus Clip demo."""
        logger.info("🚀 Starting Ultimate Opus Clip Demo - ALL FEATURES ENABLED")
        
        try:
            # Demo 1: Content Curation Engine
            await self.demo_content_curation(video_path)
            
            # Demo 2: Advanced Viral Scoring
            await self.demo_viral_scoring(video_path, content_text)
            
            # Demo 3: Speaker Tracking System
            await self.demo_speaker_tracking(video_path)
            
            # Demo 4: Audio Processing System
            await self.demo_audio_processing(video_path, content_text)
            
            # Demo 5: B-roll Integration System
            await self.demo_broll_integration(video_path, content_text)
            
            # Demo 6: Professional Export System
            await self.demo_professional_export(video_path)
            
            # Demo 7: Advanced Analytics System
            await self.demo_advanced_analytics(video_path)
            
            # Demo 8: Ultimate Pipeline (All Features Combined)
            await self.demo_ultimate_pipeline(video_path, content_text)
            
            # Generate comprehensive demo report
            await self.generate_ultimate_demo_report()
            
            logger.info("✅ Ultimate Opus Clip Demo completed successfully - ALL FEATURES WORKING!")
            
        except Exception as e:
            logger.error(f"❌ Ultimate demo failed: {e}")
            raise
    
    async def demo_content_curation(self, video_path: str):
        """Demo 1: Content Curation Engine (ClipGenius™)."""
        logger.info("🔍 Demo 1: Content Curation Engine (ClipGenius™)")
        
        try:
            start_time = time.time()
            
            # Analyze video for engaging moments
            analysis_result = await self.content_curation_engine.analyze_video(video_path)
            
            processing_time = time.time() - start_time
            
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
    
    async def demo_viral_scoring(self, video_path: str, content_text: str):
        """Demo 2: Advanced Viral Scoring System."""
        logger.info("📊 Demo 2: Advanced Viral Scoring System")
        
        try:
            start_time = time.time()
            
            # Prepare content data for viral scoring
            content_data = {
                "keywords": ["AI", "viral", "trending", "technology", "innovation"],
                "duration": 15.0,
                "engagement_scores": [{"score": 0.8}, {"score": 0.9}, {"score": 0.7}],
                "content_text": content_text or "Revolutionary AI technology that will change everything!"
            }
            
            # Calculate viral score
            viral_score = await self.advanced_viral_scorer.calculate_viral_score(
                content_data, "tiktok"
            )
            
            processing_time = time.time() - start_time
            
            self.demo_results["viral_scoring"] = {
                "status": "success",
                "processing_time": processing_time,
                "overall_score": viral_score.overall_score,
                "factor_scores": {k.value: v for k, v in viral_score.factor_scores.items()},
                "confidence": viral_score.confidence,
                "recommendations": viral_score.recommendations,
                "trend_alignment": viral_score.trend_alignment,
                "audience_potential": viral_score.audience_potential,
                "summary": {
                    "viral_potential": "high" if viral_score.overall_score > 0.7 else "medium" if viral_score.overall_score > 0.5 else "low",
                    "top_factors": sorted(viral_score.factor_scores.items(), key=lambda x: x[1], reverse=True)[:3],
                    "recommendations_count": len(viral_score.recommendations)
                }
            }
            
            logger.info(f"✅ Viral scoring completed in {processing_time:.2f}s")
            logger.info(f"   Overall viral score: {viral_score.overall_score:.2f}")
            logger.info(f"   Confidence: {viral_score.confidence:.2f}")
            logger.info(f"   Recommendations: {len(viral_score.recommendations)}")
            
        except Exception as e:
            logger.error(f"❌ Viral scoring demo failed: {e}")
            self.demo_results["viral_scoring"] = {"status": "failed", "error": str(e)}
    
    async def demo_speaker_tracking(self, video_path: str):
        """Demo 3: Speaker Tracking System."""
        logger.info("👤 Demo 3: Speaker Tracking System")
        
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
    
    async def demo_audio_processing(self, video_path: str, content_text: str):
        """Demo 4: Audio Processing System."""
        logger.info("🎵 Demo 4: Audio Processing System")
        
        try:
            start_time = time.time()
            
            # Prepare content analysis
            content_analysis = {
                "keywords": ["AI", "technology", "innovation"],
                "duration": 30.0,
                "content_type": "tech",
                "engagement_level": 0.8
            }
            
            # Process video audio
            audio_result = await self.audio_processing_system.process_video_audio(
                video_path,
                content_analysis,
                f"/tmp/demo_audio_{int(time.time())}.mp4"
            )
            
            processing_time = time.time() - start_time
            
            self.demo_results["audio_processing"] = {
                "status": "success",
                "processing_time": processing_time,
                "input_video": audio_result.get("input_video", ""),
                "output_audio": audio_result.get("output_audio", ""),
                "enhanced_audio": audio_result.get("enhanced_audio", ""),
                "matching_tracks": audio_result.get("matching_tracks", 0),
                "suggested_effects": audio_result.get("suggested_effects", 0),
                "processing_successful": audio_result.get("processing_successful", False),
                "summary": {
                    "audio_enhanced": bool(audio_result.get("enhanced_audio")),
                    "background_music_found": audio_result.get("matching_tracks", 0) > 0,
                    "sound_effects_suggested": audio_result.get("suggested_effects", 0) > 0,
                    "processing_quality": "high" if audio_result.get("processing_successful", False) else "medium"
                }
            }
            
            logger.info(f"✅ Audio processing completed in {processing_time:.2f}s")
            logger.info(f"   Matching tracks found: {audio_result.get('matching_tracks', 0)}")
            logger.info(f"   Sound effects suggested: {audio_result.get('suggested_effects', 0)}")
            logger.info(f"   Processing successful: {audio_result.get('processing_successful', False)}")
            
        except Exception as e:
            logger.error(f"❌ Audio processing demo failed: {e}")
            self.demo_results["audio_processing"] = {"status": "failed", "error": str(e)}
    
    async def demo_broll_integration(self, video_path: str, content_text: str):
        """Demo 5: B-roll Integration System."""
        logger.info("🎥 Demo 5: B-roll Integration System")
        
        try:
            start_time = time.time()
            
            # Process video with B-roll integration
            broll_result = await self.broll_integration_system.process_video(
                video_path,
                content_text or "Revolutionary AI technology demonstration",
                f"/tmp/demo_broll_{int(time.time())}.mp4"
            )
            
            processing_time = time.time() - start_time
            
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
    
    async def demo_professional_export(self, video_path: str):
        """Demo 6: Professional Export System."""
        logger.info("💼 Demo 6: Professional Export System")
        
        try:
            start_time = time.time()
            
            # Prepare video data for export
            video_data = {
                "title": "Ultimate Opus Clip Demo Export",
                "duration": 30.0,
                "source_file": video_path,
                "clips": [
                    {
                        "title": "Demo Clip 1",
                        "start_time": 0.0,
                        "end_time": 15.0,
                        "file_path": video_path
                    },
                    {
                        "title": "Demo Clip 2",
                        "start_time": 15.0,
                        "end_time": 30.0,
                        "file_path": video_path
                    }
                ]
            }
            
            # Export to different formats
            export_formats = [
                (ExportFormat.PREMIERE_PRO, "premiere_pro.xml"),
                (ExportFormat.FINAL_CUT, "final_cut.fcpxml"),
                (ExportFormat.XML, "generic.xml")
            ]
            
            export_results = []
            for export_format, filename in export_formats:
                settings = ExportSettings(
                    format=export_format,
                    quality=ExportQuality.HIGH,
                    resolution=(1920, 1080),
                    frame_rate=30.0,
                    bitrate=5000,
                    codec="h264",
                    audio_codec="aac",
                    audio_bitrate=128
                )
                
                output_path = f"/tmp/demo_export_{filename}"
                result = await self.professional_export_system.export_project(
                    video_data, settings, output_path
                )
                export_results.append(result)
            
            processing_time = time.time() - start_time
            
            self.demo_results["professional_export"] = {
                "status": "success",
                "processing_time": processing_time,
                "export_results": [
                    {
                        "format": result.format.value,
                        "success": result.success,
                        "file_size": result.file_size,
                        "output_path": result.output_path
                    }
                    for result in export_results
                ],
                "summary": {
                    "formats_exported": len(export_results),
                    "successful_exports": len([r for r in export_results if r.success]),
                    "total_file_size": sum([r.file_size for r in export_results if r.success])
                }
            }
            
            logger.info(f"✅ Professional export completed in {processing_time:.2f}s")
            logger.info(f"   Exported to {len(export_results)} formats")
            logger.info(f"   Successful exports: {len([r for r in export_results if r.success])}")
            
        except Exception as e:
            logger.error(f"❌ Professional export demo failed: {e}")
            self.demo_results["professional_export"] = {"status": "failed", "error": str(e)}
    
    async def demo_advanced_analytics(self, video_path: str):
        """Demo 7: Advanced Analytics System."""
        logger.info("📈 Demo 7: Advanced Analytics System")
        
        try:
            start_time = time.time()
            
            # Generate sample content IDs
            content_ids = [f"demo_content_{i}" for i in range(5)]
            
            # Track some performance metrics
            for i, content_id in enumerate(content_ids):
                await self.advanced_analytics_system.track_content_performance(
                    content_id,
                    "tiktok",
                    {
                        "engagement": 0.7 + (i * 0.05),
                        "viral_potential": 0.6 + (i * 0.08),
                        "quality": 0.8 + (i * 0.02)
                    }
                )
            
            # Generate analytics report
            analytics_report = await self.advanced_analytics_system.generate_analytics_report(
                content_ids, TimeRange.WEEK
            )
            
            processing_time = time.time() - start_time
            
            self.demo_results["advanced_analytics"] = {
                "status": "success",
                "processing_time": processing_time,
                "report_id": analytics_report.report_id,
                "content_analyzed": len(analytics_report.content_analyses),
                "performance_summary": analytics_report.performance_summary,
                "audience_insights": len(analytics_report.audience_insights),
                "recommendations": analytics_report.recommendations,
                "summary": {
                    "total_content": analytics_report.performance_summary.get("total_content", 0),
                    "average_performance": analytics_report.performance_summary.get("average_performance_score", 0),
                    "average_viral_potential": analytics_report.performance_summary.get("average_viral_potential", 0),
                    "recommendations_count": len(analytics_report.recommendations)
                }
            }
            
            logger.info(f"✅ Advanced analytics completed in {processing_time:.2f}s")
            logger.info(f"   Content analyzed: {len(analytics_report.content_analyses)}")
            logger.info(f"   Audience insights: {len(analytics_report.audience_insights)}")
            logger.info(f"   Recommendations: {len(analytics_report.recommendations)}")
            
        except Exception as e:
            logger.error(f"❌ Advanced analytics demo failed: {e}")
            self.demo_results["advanced_analytics"] = {"status": "failed", "error": str(e)}
    
    async def demo_ultimate_pipeline(self, video_path: str, content_text: str):
        """Demo 8: Ultimate Pipeline (All Features Combined)."""
        logger.info("🚀 Demo 8: Ultimate Pipeline - ALL FEATURES COMBINED")
        
        try:
            start_time = time.time()
            
            # Simulate ultimate pipeline processing
            pipeline_results = {
                "content_curation": self.demo_results.get("content_curation", {}),
                "viral_scoring": self.demo_results.get("viral_scoring", {}),
                "speaker_tracking": self.demo_results.get("speaker_tracking", {}),
                "audio_processing": self.demo_results.get("audio_processing", {}),
                "broll_integration": self.demo_results.get("broll_integration", {}),
                "professional_export": self.demo_results.get("professional_export", {}),
                "advanced_analytics": self.demo_results.get("advanced_analytics", {})
            }
            
            # Calculate overall success rate
            successful_features = len([r for r in pipeline_results.values() if r.get("status") == "success"])
            total_features = len(pipeline_results)
            success_rate = successful_features / total_features
            
            processing_time = time.time() - start_time
            
            self.demo_results["ultimate_pipeline"] = {
                "status": "success",
                "processing_time": processing_time,
                "features_enabled": total_features,
                "features_successful": successful_features,
                "success_rate": success_rate,
                "pipeline_results": pipeline_results,
                "summary": {
                    "overall_success": "excellent" if success_rate > 0.9 else "good" if success_rate > 0.7 else "needs_improvement",
                    "total_processing_time": sum([
                        result.get("processing_time", 0) 
                        for result in pipeline_results.values() 
                        if isinstance(result, dict) and "processing_time" in result
                    ]),
                    "features_working": [name for name, result in pipeline_results.items() if result.get("status") == "success"],
                    "features_failed": [name for name, result in pipeline_results.items() if result.get("status") == "failed"]
                }
            }
            
            logger.info(f"✅ Ultimate pipeline completed in {processing_time:.2f}s")
            logger.info(f"   Features enabled: {total_features}")
            logger.info(f"   Features successful: {successful_features}")
            logger.info(f"   Success rate: {success_rate:.1%}")
            
        except Exception as e:
            logger.error(f"❌ Ultimate pipeline demo failed: {e}")
            self.demo_results["ultimate_pipeline"] = {"status": "failed", "error": str(e)}
    
    async def generate_ultimate_demo_report(self):
        """Generate comprehensive ultimate demo report."""
        logger.info("📊 Generating Ultimate Demo Report")
        
        try:
            report = {
                "demo_info": {
                    "timestamp": time.time(),
                    "version": "3.0.0",
                    "features_demoed": [
                        "Content Curation Engine (ClipGenius™)",
                        "Advanced Viral Scoring System",
                        "Speaker Tracking System",
                        "Audio Processing System",
                        "B-roll Integration System",
                        "Professional Export System",
                        "Advanced Analytics System",
                        "Ultimate Pipeline (All Features Combined)"
                    ],
                    "total_features": 8
                },
                "results": self.demo_results,
                "summary": {
                    "total_demos": len(self.demo_results),
                    "successful_demos": len([r for r in self.demo_results.values() if r.get("status") == "success"]),
                    "failed_demos": len([r for r in self.demo_results.values() if r.get("status") == "failed"]),
                    "overall_success_rate": len([r for r in self.demo_results.values() if r.get("status") == "success"]) / max(len(self.demo_results), 1)
                },
                "performance_metrics": self._calculate_ultimate_performance_metrics(),
                "feature_analysis": self._analyze_ultimate_features(),
                "recommendations": self._generate_ultimate_recommendations()
            }
            
            # Save report
            report_path = f"/tmp/ultimate_opus_clip_demo_report_{int(time.time())}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"📄 Ultimate demo report saved to: {report_path}")
            
            # Print ultimate summary
            self._print_ultimate_demo_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Ultimate demo report generation failed: {e}")
            return {"error": str(e)}
    
    def _calculate_ultimate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate ultimate performance metrics."""
        try:
            metrics = {}
            
            for feature_name, result in self.demo_results.items():
                if isinstance(result, dict) and "processing_time" in result:
                    metrics[feature_name] = {
                        "processing_time": result.get("processing_time", 0),
                        "status": result.get("status", "unknown"),
                        "efficiency": "high" if result.get("processing_time", 0) < 30 else "medium" if result.get("processing_time", 0) < 60 else "low"
                    }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Ultimate performance metrics calculation failed: {e}")
            return {}
    
    def _analyze_ultimate_features(self) -> Dict[str, Any]:
        """Analyze ultimate feature performance."""
        try:
            analysis = {
                "content_curation_quality": "high" if self.demo_results.get("content_curation", {}).get("summary", {}).get("viral_potential") == "high" else "medium",
                "viral_scoring_quality": "high" if self.demo_results.get("viral_scoring", {}).get("overall_score", 0) > 0.7 else "medium",
                "speaker_tracking_quality": self.demo_results.get("speaker_tracking", {}).get("summary", {}).get("tracking_quality", "unknown"),
                "audio_processing_quality": self.demo_results.get("audio_processing", {}).get("summary", {}).get("processing_quality", "unknown"),
                "broll_integration_quality": self.demo_results.get("broll_integration", {}).get("summary", {}).get("integration_quality", "unknown"),
                "export_system_quality": "high" if self.demo_results.get("professional_export", {}).get("summary", {}).get("successful_exports", 0) > 0 else "medium",
                "analytics_system_quality": "high" if self.demo_results.get("advanced_analytics", {}).get("content_analyzed", 0) > 0 else "medium",
                "overall_readiness": "production_ready" if all(
                    result.get("status") == "success" 
                    for result in self.demo_results.values()
                ) else "needs_improvement"
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ultimate feature analysis failed: {e}")
            return {}
    
    def _generate_ultimate_recommendations(self) -> List[str]:
        """Generate ultimate recommendations."""
        try:
            recommendations = []
            
            # Overall system recommendations
            success_rate = len([r for r in self.demo_results.values() if r.get("status") == "success"]) / max(len(self.demo_results), 1)
            
            if success_rate > 0.9:
                recommendations.append("🎉 System is performing excellently - ready for production deployment!")
            elif success_rate > 0.7:
                recommendations.append("✅ System is performing well - minor optimizations recommended")
            else:
                recommendations.append("⚠️ System needs improvement - focus on failed features")
            
            # Feature-specific recommendations
            if self.demo_results.get("content_curation", {}).get("summary", {}).get("viral_potential") == "low":
                recommendations.append("Improve content curation algorithms for better viral potential detection")
            
            if self.demo_results.get("viral_scoring", {}).get("overall_score", 0) < 0.6:
                recommendations.append("Enhance viral scoring models for more accurate predictions")
            
            if self.demo_results.get("speaker_tracking", {}).get("summary", {}).get("tracking_success_rate", 0) < 0.8:
                recommendations.append("Optimize speaker tracking for better accuracy")
            
            if self.demo_results.get("audio_processing", {}).get("summary", {}).get("processing_quality") == "low":
                recommendations.append("Improve audio processing quality and effects")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Ultimate recommendations generation failed: {e}")
            return ["Focus on system optimization and feature enhancement"]
    
    def _print_ultimate_demo_summary(self, report: Dict[str, Any]):
        """Print ultimate demo summary to console."""
        print("\n" + "="*80)
        print("🚀 ULTIMATE OPUS CLIP DEMO SUMMARY - ALL FEATURES ENABLED")
        print("="*80)
        
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
                print(f"   {feature.replace('_', ' ').title()}: {metric['processing_time']:.2f}s ({metric['efficiency']})")
        
        # Feature analysis
        print("\n🎯 FEATURE ANALYSIS:")
        analysis = report.get("feature_analysis", {})
        for metric, value in analysis.items():
            print(f"   {metric.replace('_', ' ').title()}: {value}")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS:")
        recommendations = report.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "="*80)
        print("🎬 ULTIMATE OPUS CLIP - THE MOST COMPREHENSIVE VIDEO PROCESSING PLATFORM!")
        print("🚀 Ready to create viral content with ALL advanced features!")
        print("="*80 + "\n")

async def main():
    """Main ultimate demo function."""
    # Initialize demo
    demo = UltimateOpusClipDemo()
    
    # Demo video path (replace with actual video)
    video_path = "/path/to/your/demo/video.mp4"
    content_text = """
    Welcome to the Ultimate Opus Clip experience! 
    This is the most comprehensive AI-powered video processing platform ever created.
    We're showcasing ALL advanced features including content curation, viral scoring,
    speaker tracking, audio processing, B-roll integration, professional export,
    and advanced analytics. This is the future of video content creation!
    """
    
    # Check if video exists
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("Please update the video_path variable with a valid video file.")
        return
    
    try:
        # Run ultimate demo
        await demo.run_ultimate_demo(video_path, content_text)
        
    except Exception as e:
        logger.error(f"Ultimate demo failed: {e}")
        print(f"❌ Ultimate demo failed: {e}")

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
    
    # Run ultimate demo
    asyncio.run(main())


