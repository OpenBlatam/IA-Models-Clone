"""
BUL System Validation Script
============================

Validates that the optimized BUL system is working correctly.
"""

import sys
import importlib
from pathlib import Path

def validate_imports():
    """Validate that all modules can be imported correctly."""
    print("🔍 Validating module imports...")
    
    try:
        # Test core modules
        from modules import DocumentProcessor, QueryAnalyzer, BusinessAgentManager, APIHandler
        print("✅ Core modules imported successfully")
        
        # Test configuration
        from config_optimized import BULConfig, get_config
        print("✅ Configuration module imported successfully")
        
        # Test main application
        from bul_optimized import BULSystem
        print("✅ Main application imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def validate_configuration():
    """Validate configuration system."""
    print("\n🔧 Validating configuration...")
    
    try:
        from config_optimized import BULConfig
        
        # Test default configuration
        config = BULConfig()
        print(f"✅ Default config loaded - API Port: {config.api_port}")
        
        # Test configuration validation
        errors = config.validate_config()
        if errors:
            print(f"⚠️  Configuration warnings: {len(errors)} issues found")
            for error in errors[:3]:  # Show first 3 errors
                print(f"   - {error}")
        else:
            print("✅ Configuration validation passed")
        
        # Test business area configuration
        area_config = config.get_business_area_config('marketing')
        if area_config['enabled']:
            print("✅ Business area configuration working")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def validate_modules():
    """Validate individual modules."""
    print("\n🧩 Validating individual modules...")
    
    try:
        from modules.document_processor import DocumentProcessor
        from modules.query_analyzer import QueryAnalyzer
        from modules.business_agents import BusinessAgentManager
        
        # Test DocumentProcessor
        config = {'supported_formats': ['markdown'], 'output_directory': 'test'}
        processor = DocumentProcessor(config)
        print("✅ DocumentProcessor initialized")
        
        # Test QueryAnalyzer
        analyzer = QueryAnalyzer()
        analysis = analyzer.analyze("Create a marketing strategy")
        if analysis.primary_area:
            print("✅ QueryAnalyzer working")
        
        # Test BusinessAgentManager
        agent_manager = BusinessAgentManager(config)
        areas = agent_manager.get_available_areas()
        if areas:
            print(f"✅ BusinessAgentManager working - {len(areas)} areas available")
        
        return True
        
    except Exception as e:
        print(f"❌ Module validation error: {e}")
        return False

def validate_file_structure():
    """Validate file structure."""
    print("\n📁 Validating file structure...")
    
    required_files = [
        'bul_optimized.py',
        'config_optimized.py',
        'start_optimized.py',
        'test_optimized.py',
        'requirements_optimized.txt',
        'env_optimized.txt',
        'README_OPTIMIZED.md',
        'modules/__init__.py',
        'modules/document_processor.py',
        'modules/query_analyzer.py',
        'modules/business_agents.py',
        'modules/api_handler.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def validate_dependencies():
    """Validate that required dependencies are available."""
    print("\n📦 Validating dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'pydantic',
        'httpx',
        'aiofiles',
        'python-dotenv',
        'jinja2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {missing_packages}")
        print("   Run: pip install -r requirements_optimized.txt")
        return False
    else:
        print("✅ All required packages available")
        return True

def run_quick_test():
    """Run a quick functionality test."""
    print("\n🧪 Running quick functionality test...")
    
    try:
        import asyncio
        from modules.document_processor import DocumentProcessor
        from modules.query_analyzer import QueryAnalyzer
        from modules.business_agents import BusinessAgentManager
        
        async def test_workflow():
            # Setup
            config = {
                'supported_formats': ['markdown'],
                'output_directory': 'test_output'
            }
            
            processor = DocumentProcessor(config)
            analyzer = QueryAnalyzer()
            agent_manager = BusinessAgentManager(config)
            
            # Test query analysis
            query = "Create a marketing strategy for a new product"
            analysis = analyzer.analyze(query)
            
            # Test agent processing
            result = await agent_manager.process_with_agent(
                analysis.primary_area, query, analysis.document_types[0]
            )
            
            # Test document generation
            document = await processor.generate_document(
                query=query,
                business_area=analysis.primary_area,
                document_type=analysis.document_types[0]
            )
            
            return document is not None
        
        # Run async test
        result = asyncio.run(test_workflow())
        
        if result:
            print("✅ Quick functionality test passed")
            return True
        else:
            print("❌ Quick functionality test failed")
            return False
            
    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False

def main():
    """Main validation function."""
    print("🚀 BUL System Validation")
    print("=" * 50)
    
    tests = [
        ("File Structure", validate_file_structure),
        ("Dependencies", validate_dependencies),
        ("Module Imports", validate_imports),
        ("Configuration", validate_configuration),
        ("Individual Modules", validate_modules),
        ("Quick Functionality", run_quick_test)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validations passed! System is ready to use.")
        print("\n🚀 To start the system:")
        print("   python start_optimized.py")
        return 0
    else:
        print("⚠️  Some validations failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
