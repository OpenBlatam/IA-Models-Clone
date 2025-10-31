#!/usr/bin/env python3
"""
Gamma App - Comprehensive Test Script
Test script to verify all components are working correctly
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    """Setup test logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

class GammaAppTester:
    """Comprehensive tester for Gamma App"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
    
    def test_imports(self) -> bool:
        """Test if all modules can be imported"""
        self.logger.info("🧪 Testing imports...")
        
        try:
            # Test core imports
            from core.content_generator import ContentGenerator
            from core.design_engine import DesignEngine
            from core.collaboration_engine import CollaborationEngine
            
            # Test engine imports
            from engines.presentation_engine import PresentationEngine
            from engines.document_engine import DocumentEngine
            from engines.web_page_engine import WebPageEngine
            from engines.ai_models_engine import AIModelsEngine
            
            # Test service imports
            from services.cache_service import AdvancedCacheService
            from services.security_service import AdvancedSecurityService
            from services.analytics_service import AnalyticsService
            from services.collaboration_service import CollaborationService
            
            # Test utility imports
            from utils.config import get_settings
            from utils.auth import create_access_token, verify_token
            
            self.logger.info("✅ All imports successful")
            return True
            
        except ImportError as e:
            self.logger.error(f"❌ Import failed: {e}")
            return False
    
    def test_configuration(self) -> bool:
        """Test configuration system"""
        self.logger.info("🧪 Testing configuration...")
        
        try:
            from utils.config import get_settings, validate_config
            
            settings = get_settings()
            self.logger.info(f"  📋 App Name: {settings.app_name}")
            self.logger.info(f"  📋 Version: {settings.app_version}")
            self.logger.info(f"  📋 Environment: {settings.environment}")
            
            # Test configuration validation
            is_valid = validate_config()
            if is_valid:
                self.logger.info("✅ Configuration valid")
                return True
            else:
                self.logger.warning("⚠️  Configuration validation failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Configuration test failed: {e}")
            return False
    
    def test_content_generator(self) -> bool:
        """Test content generator"""
        self.logger.info("🧪 Testing content generator...")
        
        try:
            from core.content_generator import ContentGenerator
            
            # Initialize with minimal config
            config = {
                'openai_api_key': 'test-key',
                'anthropic_api_key': 'test-key'
            }
            
            generator = ContentGenerator(config)
            self.logger.info("✅ Content generator initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Content generator test failed: {e}")
            return False
    
    def test_engines(self) -> bool:
        """Test all engines"""
        self.logger.info("🧪 Testing engines...")
        
        engines = [
            ('PresentationEngine', 'engines.presentation_engine', 'PresentationEngine'),
            ('DocumentEngine', 'engines.document_engine', 'DocumentEngine'),
            ('WebPageEngine', 'engines.web_page_engine', 'WebPageEngine'),
            ('AIModelsEngine', 'engines.ai_models_engine', 'AIModelsEngine')
        ]
        
        all_passed = True
        for engine_name, module_path, class_name in engines:
            try:
                module = __import__(module_path, fromlist=[class_name])
                engine_class = getattr(module, class_name)
                engine = engine_class({})
                self.logger.info(f"  ✅ {engine_name} initialized")
            except Exception as e:
                self.logger.error(f"  ❌ {engine_name} failed: {e}")
                all_passed = False
        
        return all_passed
    
    def test_services(self) -> bool:
        """Test all services"""
        self.logger.info("🧪 Testing services...")
        
        services = [
            ('CacheService', 'services.cache_service', 'AdvancedCacheService'),
            ('SecurityService', 'services.security_service', 'AdvancedSecurityService'),
            ('AnalyticsService', 'services.analytics_service', 'AnalyticsService'),
            ('CollaborationService', 'services.collaboration_service', 'CollaborationService'),
            ('FileService', 'services.file_service', 'FileService'),
            ('SearchService', 'services.search_service', 'SearchService')
        ]
        
        all_passed = True
        for service_name, module_path, class_name in services:
            try:
                module = __import__(module_path, fromlist=[class_name])
                service_class = getattr(module, class_name)
                service = service_class({})
                self.logger.info(f"  ✅ {service_name} initialized")
            except Exception as e:
                self.logger.error(f"  ❌ {service_name} failed: {e}")
                all_passed = False
        
        return all_passed
    
    def test_api_structure(self) -> bool:
        """Test API structure"""
        self.logger.info("🧪 Testing API structure...")
        
        try:
            from api.main import app
            from api.routes import content_router, collaboration_router, export_router, analytics_router
            
            # Check if FastAPI app is properly configured
            if hasattr(app, 'routes') and len(app.routes) > 0:
                self.logger.info("✅ API structure valid")
                return True
            else:
                self.logger.warning("⚠️  API routes not properly configured")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ API structure test failed: {e}")
            return False
    
    def test_database_models(self) -> bool:
        """Test database models"""
        self.logger.info("🧪 Testing database models...")
        
        try:
            from models.database import Base
            from models.user import User
            from models.content import Content
            from models.collaboration import CollaborationSession
            
            self.logger.info("✅ Database models imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Database models test failed: {e}")
            return False
    
    def test_utilities(self) -> bool:
        """Test utility functions"""
        self.logger.info("🧪 Testing utilities...")
        
        try:
            from utils.auth import create_access_token, verify_token, hash_password
            from utils.encryption_utils import encrypt_data, decrypt_data
            from utils.validation_utils import validate_email, validate_password
            
            # Test basic utility functions
            test_password = "test_password_123"
            hashed = hash_password(test_password)
            
            test_email = "test@example.com"
            is_valid_email = validate_email(test_email)
            
            if hashed and is_valid_email:
                self.logger.info("✅ Utilities working correctly")
                return True
            else:
                self.logger.warning("⚠️  Some utilities failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Utilities test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests"""
        self.logger.info("🚀 Starting Gamma App comprehensive tests...")
        print("=" * 60)
        
        tests = [
            ("Imports", self.test_imports),
            ("Configuration", self.test_configuration),
            ("Content Generator", self.test_content_generator),
            ("Engines", self.test_engines),
            ("Services", self.test_services),
            ("API Structure", self.test_api_structure),
            ("Database Models", self.test_database_models),
            ("Utilities", self.test_utilities)
        ]
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1
            except Exception as e:
                self.logger.error(f"❌ Test {test_name} crashed: {e}")
                self.test_results[test_name] = False
                self.failed_tests += 1
        
        return self.test_results
    
    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📊 GAMMA APP TEST SUMMARY")
        print("=" * 60)
        
        print(f"\n📈 Test Results:")
        print(f"  ✅ Passed: {self.passed_tests}")
        print(f"  ❌ Failed: {self.failed_tests}")
        print(f"  📊 Total: {self.passed_tests + self.failed_tests}")
        
        print(f"\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        success_rate = (self.passed_tests / (self.passed_tests + self.failed_tests)) * 100
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if self.failed_tests == 0:
            print("\n🎉 All tests passed! Gamma App is ready to run!")
        else:
            print(f"\n⚠️  {self.failed_tests} tests failed. Please check the issues above.")
        
        print("=" * 60)

def main():
    """Main test function"""
    logger = setup_logging()
    
    try:
        tester = GammaAppTester()
        results = tester.run_all_tests()
        tester.print_summary()
        
        # Exit with appropriate code
        if tester.failed_tests == 0:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Test suite crashed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



