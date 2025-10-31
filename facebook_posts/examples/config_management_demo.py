from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import os
import tempfile
import json
from typing import Dict, Any
    from cybersecurity.config.config_manager import (
        import yaml
        import yaml
        import yaml
        import jsonschema
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo script for enhanced configuration management with PyYAML and jsonschema.
Showcases configuration loading, validation, and management capabilities.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
        ConfigManager, SecurityConfig, ConfigValidationError,
        load_security_config, create_default_security_config,
        validate_config_schema
    )
    print("✓ Configuration management modules loaded successfully!")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

async def demo_yaml_config_loading():
    """Demo YAML configuration loading and validation."""
    print("\n" + "="*60)
    print("📄 YAML CONFIGURATION DEMO")
    print("="*60)
    
    manager = ConfigManager()
    
    # Create a sample YAML config
    sample_config = {
        "timeout": 15.0,
        "max_workers": 100,
        "retry_count": 3,
        "verify_ssl": True,
        "user_agent": "Security Scanner Demo",
        "log_level": "INFO",
        "output_format": "json",
        "enable_colors": True
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_file = f.name
    
    try:
        print(f"📝 Created temporary config file: {temp_file}")
        
        # Load and validate config
        config = await manager.load_config_async(temp_file, "security")
        print(f"✅ Config loaded successfully!")
        print(f"📊 Config data: {json.dumps(config, indent=2)}")
        
        # Validate against schema
        is_valid = manager.validate_config(config, "security")
        print(f"✅ Config validation: {'PASSED' if is_valid else 'FAILED'}")
        
    except ConfigValidationError as e:
        print(f"❌ Config validation error: {e.message}")
        if e.field:
            print(f"   Field: {e.field}")
        if e.value:
            print(f"   Value: {e.value}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
    finally:
        # Clean up
        os.unlink(temp_file)

async def demo_json_config_loading():
    """Demo JSON configuration loading and validation."""
    print("\n" + "="*60)
    print("📋 JSON CONFIGURATION DEMO")
    print("="*60)
    
    manager = ConfigManager()
    
    # Create a sample JSON config
    sample_config = {
        "network_scanner": {
            "scan_type": "tcp",
            "port_range": "22,80,443,8080",
            "common_ports": True,
            "banner_grab": True,
            "ssl_check": True,
            "use_nmap": True,
            "nmap_arguments": "-sS -sV -O"
        },
        "performance": {
            "timeout": 10.0,
            "max_workers": 50,
            "retry_count": 2
        },
        "output": {
            "format": "json",
            "include_timestamps": True,
            "compress_output": False
        }
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f, indent=2)
        temp_file = f.name
    
    try:
        print(f"📝 Created temporary config file: {temp_file}")
        
        # Load config
        config = await manager.load_config_async(temp_file)
        print(f"✅ Config loaded successfully!")
        print(f"📊 Config data: {json.dumps(config, indent=2)}")
        
        # Validate against network schema
        is_valid = manager.validate_config(config, "network")
        print(f"✅ Config validation: {'PASSED' if is_valid else 'FAILED'}")
        
    except ConfigValidationError as e:
        print(f"❌ Config validation error: {e.message}")
        if e.field:
            print(f"   Field: {e.field}")
        if e.value:
            print(f"   Value: {e.value}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
    finally:
        # Clean up
        os.unlink(temp_file)

async def demo_config_validation():
    """Demo configuration validation with various scenarios."""
    print("\n" + "="*60)
    print("🔍 CONFIGURATION VALIDATION DEMO")
    print("="*60)
    
    manager = ConfigManager()
    
    # Test cases
    test_cases = [
        {
            "name": "Valid Security Config",
            "config": {
                "timeout": 10.0,
                "max_workers": 50,
                "verify_ssl": True
            },
            "schema": "security",
            "should_pass": True
        },
        {
            "name": "Invalid Timeout",
            "config": {
                "timeout": -5.0,
                "max_workers": 50,
                "verify_ssl": True
            },
            "schema": "security",
            "should_pass": False
        },
        {
            "name": "Missing Required Field",
            "config": {
                "timeout": 10.0,
                "verify_ssl": True
                # Missing max_workers
            },
            "schema": "security",
            "should_pass": False
        },
        {
            "name": "Valid Network Config",
            "config": {
                "scan_type": "tcp"
            },
            "schema": "network",
            "should_pass": True
        },
        {
            "name": "Invalid Scan Type",
            "config": {
                "scan_type": "invalid_type"
            },
            "schema": "network",
            "should_pass": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            is_valid = manager.validate_config(test_case['config'], test_case['schema'])
            expected = test_case['should_pass']
            
            match is_valid:
    case expected:
                print(f"✅ PASS: Validation {'passed' if is_valid else 'failed'} as expected")
            else:
                print(f"❌ FAIL: Expected {'pass' if expected else 'fail'}, got {'pass' if is_valid else 'fail'}")
                
        except ConfigValidationError as e:
            if not test_case['should_pass']:
                print(f"✅ PASS: Validation correctly failed with error: {e.message}")
            else:
                print(f"❌ FAIL: Unexpected validation error: {e.message}")
        except Exception as e:
            print(f"❌ ERROR: {e}")

async def demo_config_templates():
    """Demo configuration template creation."""
    print("\n" + "="*60)
    print("📋 CONFIGURATION TEMPLATE DEMO")
    print("="*60)
    
    manager = ConfigManager()
    
    # Create templates for different config types
    template_types = ["security", "network"]
    
    for config_type in template_types:
        print(f"\n📝 Creating {config_type} config template...")
        
        try:
            # Create default config
            default_config = manager.create_default_config(config_type)
            print(f"✅ Default {config_type} config created:")
            print(f"   Required fields: {manager.schemas[config_type].required_fields}")
            print(f"   Optional fields: {manager.schemas[config_type].optional_fields}")
            print(f"   Config data: {json.dumps(default_config, indent=2)}")
            
            # Validate the default config
            is_valid = manager.validate_config(default_config, config_type)
            print(f"✅ Default config validation: {'PASSED' if is_valid else 'FAILED'}")
            
        except Exception as e:
            print(f"❌ Error creating {config_type} template: {e}")

async def demo_config_merging():
    """Demo configuration merging capabilities."""
    print("\n" + "="*60)
    print("🔄 CONFIGURATION MERGING DEMO")
    print("="*60)
    
    manager = ConfigManager()
    
    # Base configuration
    base_config = {
        "timeout": 10.0,
        "max_workers": 50,
        "verify_ssl": True,
        "log_level": "INFO",
        "output_format": "json"
    }
    
    # Override configuration
    override_config = {
        "timeout": 20.0,
        "max_workers": 100,
        "enable_colors": True,
        "user_agent": "Custom Scanner"
    }
    
    print("📋 Base configuration:")
    print(json.dumps(base_config, indent=2))
    
    print("\n📋 Override configuration:")
    print(json.dumps(override_config, indent=2))
    
    # Merge configurations
    merged_config = manager.merge_configs(base_config, override_config)
    
    print("\n🔄 Merged configuration:")
    print(json.dumps(merged_config, indent=2))
    
    # Validate merged config
    try:
        is_valid = manager.validate_config(merged_config, "security")
        print(f"\n✅ Merged config validation: {'PASSED' if is_valid else 'FAILED'}")
    except ConfigValidationError as e:
        print(f"\n❌ Merged config validation error: {e.message}")

async def demo_async_config_operations():
    """Demo async configuration operations."""
    print("\n" + "="*60)
    print("⚡ ASYNC CONFIGURATION OPERATIONS DEMO")
    print("="*60)
    
    manager = ConfigManager()
    
    # Sample configuration
    sample_config = {
        "timeout": 15.0,
        "max_workers": 75,
        "retry_count": 3,
        "verify_ssl": True,
        "user_agent": "Async Security Scanner",
        "log_level": "DEBUG",
        "output_format": "yaml",
        "enable_colors": True
    }
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f)
        temp_file = f.name
    
    try:
        print(f"📝 Created temporary config file: {temp_file}")
        
        # Async load
        print("🔄 Loading config asynchronously...")
        loaded_config = await manager.load_config_async(temp_file, "security")
        print(f"✅ Config loaded: {len(loaded_config)} fields")
        
        # Async save with modifications
        modified_config = loaded_config.copy()
        modified_config["timeout"] = 25.0
        modified_config["max_workers"] = 150
        
        save_file = temp_file.replace('.yaml', '_modified.yaml')
        print(f"🔄 Saving modified config asynchronously...")
        success = await manager.save_config_async(modified_config, save_file)
        print(f"✅ Config saved: {'SUCCESS' if success else 'FAILED'}")
        
        # Verify saved config
        if success:
            print("🔄 Loading saved config to verify...")
            saved_config = await manager.load_config_async(save_file, "security")
            print(f"✅ Saved config verified: {len(saved_config)} fields")
            
            # Compare
            if saved_config == modified_config:
                print("✅ Config integrity verified!")
            else:
                print("❌ Config integrity check failed!")
        
    except Exception as e:
        print(f"❌ Error in async operations: {e}")
    finally:
        # Clean up
        os.unlink(temp_file)
        if 'save_file' in locals():
            try:
                os.unlink(save_file)
            except:
                pass

async def demo_library_availability():
    """Check and display configuration library availability."""
    print("\n" + "="*60)
    print("📚 CONFIGURATION LIBRARY AVAILABILITY CHECK")
    print("="*60)
    
    # Check PyYAML
    try:
        print("✅ PyYAML: Available")
        print(f"   Version: {yaml.__version__}")
    except ImportError:
        print("❌ PyYAML: Not available")
        print("   Install with: pip install PyYAML")
    
    # Check jsonschema
    try:
        print("✅ jsonschema: Available")
        print(f"   Version: {jsonschema.__version__}")
    except ImportError:
        print("❌ jsonschema: Not available")
        print("   Install with: pip install jsonschema")
    
    # Check other dependencies
    dependencies = [
        ("json", "JSON support"),
        ("pathlib", "Path operations"),
        ("logging", "Logging support"),
        ("asyncio", "Async I/O support")
    ]
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            print(f"✅ {dep_name}: Available ({description})")
        except ImportError:
            print(f"❌ {dep_name}: Not available ({description})")

async def main():
    """Main demo function."""
    print("🚀 ENHANCED CONFIGURATION MANAGEMENT DEMO")
    print("="*60)
    print("This demo showcases configuration management with PyYAML and jsonschema.")
    print("Features: YAML/JSON loading, schema validation, async operations, merging.")
    
    # Check library availability
    await demo_library_availability()
    
    # Run demos
    await demo_yaml_config_loading()
    await demo_json_config_loading()
    await demo_config_validation()
    await demo_config_templates()
    await demo_config_merging()
    await demo_async_config_operations()
    
    print("\n" + "="*60)
    print("✅ CONFIGURATION MANAGEMENT DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key features demonstrated:")
    print("• YAML and JSON configuration loading")
    print("• Schema-based configuration validation")
    print("• Configuration template creation")
    print("• Configuration merging and override")
    print("• Async configuration operations")
    print("• Comprehensive error handling")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 