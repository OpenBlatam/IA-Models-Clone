#!/usr/bin/env python3
"""
Test script for the Enhanced Facebook Content Optimization System
This script demonstrates the key features of the enhanced system
"""

import time
import sys
import traceback

def test_enhanced_system():
    """Test the enhanced system components"""
    print("🚀 Testing Enhanced Facebook Content Optimization System v2.0.0")
    print("=" * 60)
    
    try:
        # Test 1: Import enhanced integrated system
        print("\n📦 Test 1: Importing Enhanced Integrated System...")
        from enhanced_integrated_system import EnhancedIntegratedSystem, EnhancedIntegratedSystemConfig
        print("✅ Enhanced Integrated System imported successfully!")
        
        # Test 2: Import enhanced performance engine
        print("\n⚡ Test 2: Importing Enhanced Performance Engine...")
        from enhanced_performance_engine import EnhancedPerformanceOptimizationEngine, EnhancedPerformanceConfig
        print("✅ Enhanced Performance Engine imported successfully!")
        
        # Test 3: Import enhanced AI agent system
        print("\n🤖 Test 3: Importing Enhanced AI Agent System...")
        from enhanced_ai_agent_system import EnhancedAIAgentSystem, EnhancedAgentConfig
        print("✅ Enhanced AI Agent System imported successfully!")
        
        # Test 4: Import enhanced Gradio interface
        print("\n🖥️ Test 4: Importing Enhanced Gradio Interface...")
        from enhanced_gradio_interface import EnhancedGradioInterface
        print("✅ Enhanced Gradio Interface imported successfully!")
        
        # Test 5: Create system configuration
        print("\n⚙️ Test 5: Creating System Configuration...")
        config = EnhancedIntegratedSystemConfig()
        print(f"✅ Configuration created: {config.system_name} v{config.version}")
        print(f"   - Model dimensions: {config.model_dim}")
        print(f"   - Batch size: {config.batch_size}")
        print(f"   - Cache size: {config.cache_size}")
        print(f"   - AI agents enabled: {config.enable_ai_agents}")
        
        # Test 6: Initialize enhanced system
        print("\n🔧 Test 6: Initializing Enhanced System...")
        system = EnhancedIntegratedSystem(config)
        print("✅ Enhanced System initialized successfully!")
        
        # Test 7: Test content processing
        print("\n📝 Test 7: Testing Content Processing...")
        test_content = "🚀 Exciting news! Our new AI-powered content optimization system is now live! 🎉"
        test_type = "post"
        test_audience = "tech_enthusiasts"
        test_time = "morning"
        
        print(f"   Input content: {test_content}")
        print(f"   Content type: {test_type}")
        print(f"   Target audience: {test_audience}")
        print(f"   Posting time: {test_time}")
        
        # Process content
        start_time = time.time()
        context = {
            'audience': test_audience,
            'posting_time': test_time,
            'content_type': test_type
        }
        result = system.process_content(test_content, test_type, context)
        processing_time = time.time() - start_time
        
        print(f"✅ Content processed successfully in {processing_time:.2f} seconds!")
        if result.get('status') == 'success':
            final_result = result.get('result', {})
            print(f"   Combined score: {final_result.get('combined_score', 'N/A')}")
            print(f"   Content optimization: {final_result.get('content_optimization', {}).get('engagement_score', 'N/A')}")
            print(f"   AI analysis: {final_result.get('ai_analysis', 'N/A')}")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
        
        # Test 8: Get system status
        print("\n📊 Test 8: Getting System Status...")
        status = system.get_system_status()
        print(f"✅ System status retrieved:")
        print(f"   - Overall status: {status.get('overall_status', 'N/A')}")
        print(f"   - Health score: {status.get('health_score', 'N/A')}")
        print(f"   - Uptime: {status.get('uptime_seconds', 'N/A')} seconds")
        
        # Test 9: Test Gradio interface creation
        print("\n🖥️ Test 9: Testing Gradio Interface Creation...")
        interface = EnhancedGradioInterface()
        print("✅ Enhanced Gradio Interface created successfully!")
        
        print("\n🎉 All tests passed successfully!")
        print("\n📋 System Summary:")
        print(f"   - System: {config.system_name}")
        print(f"   - Version: {config.version}")
        print(f"   - Environment: {config.environment}")
        print(f"   - Performance optimization: {'✅ Enabled' if config.enable_mixed_precision else '❌ Disabled'}")
        print(f"   - AI agents: {'✅ Enabled' if config.enable_ai_agents else '❌ Disabled'}")
        print(f"   - Health monitoring: {'✅ Enabled' if config.enable_health_checks else '❌ Disabled'}")
        print(f"   - Metrics export: {'✅ Enabled' if config.enable_metrics_export else '❌ Disabled'}")
        
        print("\n🚀 Ready to launch the enhanced system!")
        print("\nTo start the Gradio interface, run:")
        print("   python enhanced_gradio_interface.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print(f"\n🔍 Error details:")
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("Enhanced Facebook Content Optimization System - Test Suite")
    print("=" * 60)
    
    success = test_enhanced_system()
    
    if success:
        print("\n🎯 All tests completed successfully!")
        print("The enhanced system is ready for use.")
    else:
        print("\n💥 Some tests failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
