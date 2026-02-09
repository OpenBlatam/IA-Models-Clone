"""
Test Script for Ultimate Opus Clip Improvements

This script tests all the improvements and enhancements made to the system.
"""

import asyncio
import time
import sys
from pathlib import Path
import structlog
import requests
import json

logger = structlog.get_logger("test_improvements")

class ImprovementTester:
    """Test all system improvements."""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.test_results = {}
    
    async def test_api_health(self):
        """Test API health endpoints."""
        try:
            logger.info("Testing API health...")
            
            # Test basic health
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            if response.status_code == 200:
                self.test_results["api_health"] = "PASS"
                logger.info("API health test passed")
            else:
                self.test_results["api_health"] = "FAIL"
                logger.error("API health test failed")
            
            # Test detailed health
            response = requests.get(f"{self.api_base_url}/health/detailed", timeout=10)
            if response.status_code == 200:
                self.test_results["detailed_health"] = "PASS"
                logger.info("Detailed health test passed")
            else:
                self.test_results["detailed_health"] = "FAIL"
                logger.error("Detailed health test failed")
                
        except Exception as e:
            self.test_results["api_health"] = "ERROR"
            logger.error(f"API health test error: {e}")
    
    async def test_content_curation(self):
        """Test content curation engine."""
        try:
            logger.info("Testing content curation engine...")
            
            # Test with sample data
            test_data = {
                "video_path": "sample_video.mp4",
                "analysis_depth": "high",
                "target_duration": 12.0
            }
            
            response = requests.post(
                f"{self.api_base_url}/content-curation/analyze",
                json=test_data,
                timeout=30
            )
            
            if response.status_code in [200, 202]:
                self.test_results["content_curation"] = "PASS"
                logger.info("Content curation test passed")
            else:
                self.test_results["content_curation"] = "FAIL"
                logger.error(f"Content curation test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["content_curation"] = "ERROR"
            logger.error(f"Content curation test error: {e}")
    
    async def test_speaker_tracking(self):
        """Test speaker tracking system."""
        try:
            logger.info("Testing speaker tracking system...")
            
            test_data = {
                "video_path": "sample_video.mp4",
                "target_resolution": [1080, 1920],
                "tracking_quality": "high"
            }
            
            response = requests.post(
                f"{self.api_base_url}/speaker-tracking/track",
                json=test_data,
                timeout=30
            )
            
            if response.status_code in [200, 202]:
                self.test_results["speaker_tracking"] = "PASS"
                logger.info("Speaker tracking test passed")
            else:
                self.test_results["speaker_tracking"] = "FAIL"
                logger.error(f"Speaker tracking test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["speaker_tracking"] = "ERROR"
            logger.error(f"Speaker tracking test error: {e}")
    
    async def test_broll_integration(self):
        """Test B-roll integration system."""
        try:
            logger.info("Testing B-roll integration system...")
            
            test_data = {
                "video_path": "sample_video.mp4",
                "content_text": "Sample content for B-roll testing",
                "broll_types": ["stock_footage", "ai_generated"]
            }
            
            response = requests.post(
                f"{self.api_base_url}/broll-integration/integrate",
                json=test_data,
                timeout=30
            )
            
            if response.status_code in [200, 202]:
                self.test_results["broll_integration"] = "PASS"
                logger.info("B-roll integration test passed")
            else:
                self.test_results["broll_integration"] = "FAIL"
                logger.error(f"B-roll integration test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["broll_integration"] = "ERROR"
            logger.error(f"B-roll integration test error: {e}")
    
    async def test_viral_scoring(self):
        """Test viral scoring system."""
        try:
            logger.info("Testing viral scoring system...")
            
            test_data = {
                "content_text": "Sample viral content",
                "platform": "tiktok",
                "target_audience": "young_adults"
            }
            
            response = requests.post(
                f"{self.api_base_url}/viral-scoring/analyze",
                json=test_data,
                timeout=30
            )
            
            if response.status_code in [200, 202]:
                self.test_results["viral_scoring"] = "PASS"
                logger.info("Viral scoring test passed")
            else:
                self.test_results["viral_scoring"] = "FAIL"
                logger.error(f"Viral scoring test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["viral_scoring"] = "ERROR"
            logger.error(f"Viral scoring test error: {e}")
    
    async def test_audio_processing(self):
        """Test audio processing system."""
        try:
            logger.info("Testing audio processing system...")
            
            test_data = {
                "audio_path": "sample_audio.wav",
                "enhancement_level": "high",
                "background_music": True
            }
            
            response = requests.post(
                f"{self.api_base_url}/audio-processing/enhance",
                json=test_data,
                timeout=30
            )
            
            if response.status_code in [200, 202]:
                self.test_results["audio_processing"] = "PASS"
                logger.info("Audio processing test passed")
            else:
                self.test_results["audio_processing"] = "FAIL"
                logger.error(f"Audio processing test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["audio_processing"] = "ERROR"
            logger.error(f"Audio processing test error: {e}")
    
    async def test_professional_export(self):
        """Test professional export system."""
        try:
            logger.info("Testing professional export system...")
            
            test_data = {
                "video_path": "sample_video.mp4",
                "export_format": "premiere_pro",
                "quality": "high"
            }
            
            response = requests.post(
                f"{self.api_base_url}/professional-export/export",
                json=test_data,
                timeout=30
            )
            
            if response.status_code in [200, 202]:
                self.test_results["professional_export"] = "PASS"
                logger.info("Professional export test passed")
            else:
                self.test_results["professional_export"] = "FAIL"
                logger.error(f"Professional export test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["professional_export"] = "ERROR"
            logger.error(f"Professional export test error: {e}")
    
    async def test_analytics(self):
        """Test analytics system."""
        try:
            logger.info("Testing analytics system...")
            
            response = requests.get(
                f"{self.api_base_url}/analytics/performance",
                timeout=30
            )
            
            if response.status_code == 200:
                self.test_results["analytics"] = "PASS"
                logger.info("Analytics test passed")
            else:
                self.test_results["analytics"] = "FAIL"
                logger.error(f"Analytics test failed: {response.status_code}")
                
        except Exception as e:
            self.test_results["analytics"] = "ERROR"
            logger.error(f"Analytics test error: {e}")
    
    async def run_all_tests(self):
        """Run all improvement tests."""
        logger.info("Starting Ultimate Opus Clip improvement tests...")
        
        tests = [
            self.test_api_health,
            self.test_content_curation,
            self.test_speaker_tracking,
            self.test_broll_integration,
            self.test_viral_scoring,
            self.test_audio_processing,
            self.test_professional_export,
            self.test_analytics
        ]
        
        for test in tests:
            try:
                await test()
                await asyncio.sleep(1)  # Brief pause between tests
            except Exception as e:
                logger.error(f"Test {test.__name__} failed: {e}")
        
        self.print_test_results()
    
    def print_test_results(self):
        """Print test results summary."""
        logger.info("=" * 50)
        logger.info("ULTIMATE OPUS CLIP IMPROVEMENT TEST RESULTS")
        logger.info("=" * 50)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result == "PASS")
        failed_tests = sum(1 for result in self.test_results.values() if result == "FAIL")
        error_tests = sum(1 for result in self.test_results.values() if result == "ERROR")
        
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result == "PASS" else "❌" if result == "FAIL" else "⚠️"
            logger.info(f"{status_icon} {test_name}: {result}")
        
        logger.info("-" * 50)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {passed_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Errors: {error_tests}")
        logger.info(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        logger.info("=" * 50)
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED! System improvements are working correctly.")
        elif passed_tests > total_tests // 2:
            logger.info("⚠️ Most tests passed. Some improvements may need attention.")
        else:
            logger.error("❌ Many tests failed. System improvements need significant work.")

async def main():
    """Main test function."""
    tester = ImprovementTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())


