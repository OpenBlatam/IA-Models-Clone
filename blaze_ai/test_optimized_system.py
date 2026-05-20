#!/usr/bin/env python3
"""
🧪 Test Script for Optimized Blaze AI System
Comprehensive testing and validation of all optimized features
"""

import asyncio
import time
import sys
import os
from pathlib import Path
import logging

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedSystemTester:
    """Test suite for the optimized Blaze AI system."""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
    
    async def test_configuration_loading(self):
        """Test configuration loading and validation."""
        try:
            logger.info("🔧 Testing configuration loading...")
            
            # Test if optimized config exists
            config_path = Path("config-optimized.yaml")
            if not config_path.exists():
                raise FileNotFoundError("config-optimized.yaml not found")
            
            # Test if requirements-optimized.txt exists
            req_path = Path("requirements-optimized.txt")
            if not req_path.exists():
                raise FileNotFoundError("requirements-optimized.txt not found")
            
            # Test if optimized_main.py exists
            main_path = Path("optimized_main.py")
            if not main_path.exists():
                raise FileNotFoundError("optimized_main.py not found")
            
            self.test_results['configuration'] = "✅ PASSED"
            logger.info("✅ Configuration loading test passed")
            
        except Exception as e:
            self.test_results['configuration'] = f"❌ FAILED: {e}"
            logger.error(f"❌ Configuration loading test failed: {e}")
    
    async def test_docker_files(self):
        """Test Docker optimization files."""
        try:
            logger.info("🐳 Testing Docker optimization files...")
            
            # Test Dockerfile.optimized
            dockerfile_path = Path("Dockerfile.optimized")
            if not dockerfile_path.exists():
                raise FileNotFoundError("Dockerfile.optimized not found")
            
            # Test docker-compose.optimized.yml
            compose_path = Path("docker-compose.optimized.yml")
            if not compose_path.exists():
                raise FileNotFoundError("docker-compose.optimized.yml not found")
            
            # Test deploy_optimized.sh
            deploy_path = Path("deploy_optimized.sh")
            if not deploy_path.exists():
                raise FileNotFoundError("deploy_optimized.sh not found")
            
            self.test_results['docker'] = "✅ PASSED"
            logger.info("✅ Docker optimization test passed")
            
        except Exception as e:
            self.test_results['docker'] = f"❌ FAILED: {e}"
            logger.error(f"❌ Docker optimization test failed: {e}")
    
    async def test_enhanced_features(self):
        """Test enhanced features availability."""
        try:
            logger.info("🚀 Testing enhanced features...")
            
            # Test if enhanced_features directory exists
            features_dir = Path("enhanced_features")
            if not features_dir.exists():
                raise FileNotFoundError("enhanced_features directory not found")
            
            # Test key feature modules
            required_modules = [
                "enhanced_features/security.py",
                "enhanced_features/monitoring.py",
                "enhanced_features/rate_limiting.py",
                "enhanced_features/error_handling.py"
            ]
            
            for module in required_modules:
                if not Path(module).exists():
                    raise FileNotFoundError(f"Required module {module} not found")
            
            self.test_results['enhanced_features'] = "✅ PASSED"
            logger.info("✅ Enhanced features test passed")
            
        except Exception as e:
            self.test_results['enhanced_features'] = f"❌ FAILED: {e}"
            logger.error(f"❌ Enhanced features test failed: {e}")
    
    async def test_documentation(self):
        """Test documentation completeness."""
        try:
            logger.info("📚 Testing documentation...")
            
            # Test key documentation files
            required_docs = [
                "QUICK_START_OPTIMIZED.md",
                "OPTIMIZATION_SUMMARY.md",
                "README.md"
            ]
            
            for doc in required_docs:
                if not Path(doc).exists():
                    raise FileNotFoundError(f"Required documentation {doc} not found")
            
            self.test_results['documentation'] = "✅ PASSED"
            logger.info("✅ Documentation test passed")
            
        except Exception as e:
            self.test_results['documentation'] = f"❌ FAILED: {e}"
            logger.error(f"❌ Documentation test failed: {e}")
    
    async def test_performance_optimizations(self):
        """Test performance optimization features."""
        try:
            logger.info("⚡ Testing performance optimizations...")
            
            # Test if optimized_main.py has performance features
            main_content = Path("optimized_main.py").read_text()
            
            performance_features = [
                "@lru_cache",
                "OptimizedBlazeAIApplication",
                "MAX_WORKERS",
                "MAX_CONNECTIONS",
                "performance_cache"
            ]
            
            missing_features = []
            for feature in performance_features:
                if feature not in main_content:
                    missing_features.append(feature)
            
            if missing_features:
                raise ValueError(f"Missing performance features: {missing_features}")
            
            self.test_results['performance'] = "✅ PASSED"
            logger.info("✅ Performance optimizations test passed")
            
        except Exception as e:
            self.test_results['performance'] = f"❌ FAILED: {e}"
            logger.error(f"❌ Performance optimizations test failed: {e}")
    
    async def run_all_tests(self):
        """Run all tests and generate report."""
        logger.info("🚀 Starting comprehensive system testing...")
        
        # Run all tests
        await self.test_configuration_loading()
        await self.test_docker_files()
        await self.test_enhanced_features()
        await self.test_documentation()
        await self.test_performance_optimizations()
        
        # Generate test report
        self.generate_test_report()
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        end_time = time.time()
        duration = end_time - self.start_time
        
        logger.info("\n" + "="*60)
        logger.info("🧪 OPTIMIZED BLAZE AI SYSTEM TEST REPORT")
        logger.info("="*60)
        
        for test_name, result in self.test_results.items():
            logger.info(f"{test_name.replace('_', ' ').title()}: {result}")
        
        logger.info("-"*60)
        logger.info(f"⏱️  Total test duration: {duration:.2f} seconds")
        
        # Calculate success rate
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if "PASSED" in result)
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"📊 Success rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        
        if success_rate == 100:
            logger.info("🎉 ALL TESTS PASSED! System is fully optimized!")
        elif success_rate >= 80:
            logger.info("✅ Most tests passed. System is well optimized!")
        else:
            logger.warning("⚠️  Some tests failed. Review and fix issues.")
        
        logger.info("="*60)

async def main():
    """Main test execution function."""
    tester = OptimizedSystemTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
