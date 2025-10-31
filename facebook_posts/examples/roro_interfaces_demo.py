from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import os
import json
import time
import subprocess
from typing import Dict, Any, List
    from cybersecurity.cli_interface import CybersecurityCLI, CLIRequest, CLIResponse
    from cybersecurity.api_interface import CybersecurityAPI, APIRequest, APIResponse
        import fastapi
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo script for CLI and RESTful API interfaces.
Showcases RORO (Receive an Object, Return an Object) pattern for tool control.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("✓ RORO interface modules loaded successfully!")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

async def demo_cli_interface():
    """Demo CLI interface using RORO pattern."""
    print("\n" + "="*60)
    print("💻 CLI INTERFACE DEMO (RORO Pattern)")
    print("="*60)
    
    cli = CybersecurityCLI()
    
    # Test different CLI commands
    test_requests = [
        CLIRequest(
            command="help",
            output_format="text"
        ),
        CLIRequest(
            command="scan",
            target="127.0.0.1",
            scan_type="port_scan",
            user="demo_user",
            output_format="json"
        ),
        CLIRequest(
            command="rate-limit",
            target="example.com",
            output_format="json"
        ),
        CLIRequest(
            command="secrets",
            target="api_key",
            options={"source": "env"},
            output_format="json"
        ),
        CLIRequest(
            command="config",
            output_format="json"
        )
    ]
    
    print("🧪 Testing CLI commands with RORO pattern:")
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n📋 Test {i}: {request.command}")
        print("-" * 40)
        
        try:
            response = await cli.execute_command(request)
            
            if response.success:
                print(f"✅ Success: {response.message}")
                if response.data:
                    print(f"📊 Data keys: {list(response.data.keys())}")
                if response.execution_time:
                    print(f"⏱️  Execution time: {response.execution_time:.3f}s")
            else:
                print(f"❌ Error: {response.error}")
                if response.error_code:
                    print(f"   Code: {response.error_code}")
            
        except Exception as e:
            print(f"❌ Exception: {e}")

async def demo_api_interface():
    """Demo API interface using RORO pattern."""
    print("\n" + "="*60)
    print("🌐 API INTERFACE DEMO (RORO Pattern)")
    print("="*60)
    
    api = CybersecurityAPI()
    
    # Test different API endpoints
    test_requests = [
        APIRequest(
            endpoint="/health",
            method="GET",
            api_key="demo_api_key_12345"
        ),
        APIRequest(
            endpoint="/scan",
            method="POST",
            data={
                "target": "127.0.0.1",
                "scan_type": "port_scan",
                "user": "api_user"
            },
            api_key="demo_api_key_12345"
        ),
        APIRequest(
            endpoint="/rate-limit",
            method="POST",
            data={"target": "example.com"},
            api_key="demo_api_key_12345"
        ),
        APIRequest(
            endpoint="/secrets",
            method="POST",
            data={
                "secret_name": "api_key",
                "source": "env",
                "required": False
            },
            api_key="demo_api_key_12345"
        ),
        APIRequest(
            endpoint="/config",
            method="GET",
            data={"include_secrets": False},
            api_key="demo_api_key_12345"
        )
    ]
    
    print("🧪 Testing API endpoints with RORO pattern:")
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n📋 Test {i}: {request.method} {request.endpoint}")
        print("-" * 40)
        
        try:
            response = await api.execute_api_request(request)
            
            if response.success:
                print(f"✅ Success: {response.message}")
                print(f"📊 Status code: {response.status_code}")
                if response.data:
                    print(f"📊 Data keys: {list(response.data.keys())}")
                if response.execution_time:
                    print(f"⏱️  Execution time: {response.execution_time:.3f}s")
            else:
                print(f"❌ Error: {response.error}")
                print(f"📊 Status code: {response.status_code}")
                if response.error_code:
                    print(f"   Code: {response.error_code}")
            
        except Exception as e:
            print(f"❌ Exception: {e}")

async def demo_roro_pattern():
    """Demo RORO pattern principles."""
    print("\n" + "="*60)
    print("🔄 RORO PATTERN DEMO")
    print("="*60)
    
    print("📋 RORO Pattern Principles:")
    print("   ✅ Receive an Object - All inputs are structured objects")
    print("   ✅ Return an Object - All outputs are structured objects")
    print("   ✅ Consistent interface - Same pattern across CLI and API")
    print("   ✅ Type safety - Strong typing with dataclasses")
    print("   ✅ Error handling - Structured error responses")
    print("   ✅ Extensibility - Easy to add new commands/endpoints")
    
    print(f"\n🔧 Implementation Benefits:")
    print("   ✅ Unified interface for CLI and API")
    print("   ✅ Consistent error handling")
    print("   ✅ Easy testing and mocking")
    print("   ✅ Clear input/output contracts")
    print("   ✅ Type safety and validation")
    print("   ✅ Extensible architecture")
    
    print(f"\n📊 Request/Response Structure:")
    
    # Show CLI request structure
    cli_request = CLIRequest(
        command="scan",
        target="127.0.0.1",
        scan_type="port_scan",
        user="demo_user",
        output_format="json"
    )
    
    print(f"   CLI Request:")
    print(f"     • command: {cli_request.command}")
    print(f"     • target: {cli_request.target}")
    print(f"     • scan_type: {cli_request.scan_type}")
    print(f"     • user: {cli_request.user}")
    print(f"     • output_format: {cli_request.output_format}")
    
    # Show API request structure
    api_request = APIRequest(
        endpoint="/scan",
        method="POST",
        data={"target": "127.0.0.1", "scan_type": "port_scan"},
        api_key="demo_api_key_12345"
    )
    
    print(f"   API Request:")
    print(f"     • endpoint: {api_request.endpoint}")
    print(f"     • method: {api_request.method}")
    print(f"     • data: {api_request.data}")
    print(f"     • api_key: {api_request.api_key[:10]}...")

async def demo_error_handling():
    """Demo error handling in RORO pattern."""
    print("\n" + "="*60)
    print("⚠️ ERROR HANDLING DEMO")
    print("="*60)
    
    cli = CybersecurityCLI()
    api = CybersecurityAPI()
    
    # Test error scenarios
    error_scenarios = [
        {
            "name": "Missing target for scan",
            "cli_request": CLIRequest(command="scan"),
            "api_request": APIRequest(
                endpoint="/scan",
                method="POST",
                data={},
                api_key="demo_api_key_12345"
            )
        },
        {
            "name": "Invalid API key",
            "cli_request": CLIRequest(command="config"),
            "api_request": APIRequest(
                endpoint="/config",
                method="GET",
                api_key="invalid_key"
            )
        },
        {
            "name": "Unknown command/endpoint",
            "cli_request": CLIRequest(command="unknown_command"),
            "api_request": APIRequest(
                endpoint="/unknown",
                method="GET",
                api_key="demo_api_key_12345"
            )
        }
    ]
    
    print("🧪 Testing error handling:")
    
    for i, scenario in enumerate(error_scenarios, 1):
        print(f"\n📋 Error Test {i}: {scenario['name']}")
        print("-" * 40)
        
        # Test CLI error
        try:
            cli_response = await cli.execute_command(scenario['cli_request'])
            print(f"   CLI: {'✅ Success' if cli_response.success else '❌ Error'}")
            if not cli_response.success:
                print(f"      Error: {cli_response.error}")
                print(f"      Code: {cli_response.error_code}")
        except Exception as e:
            print(f"   CLI Exception: {e}")
        
        # Test API error
        try:
            api_response = await api.execute_api_request(scenario['api_request'])
            print(f"   API: {'✅ Success' if api_response.success else '❌ Error'}")
            if not api_response.success:
                print(f"      Error: {api_response.error}")
                print(f"      Code: {api_response.error_code}")
                print(f"      Status: {api_response.status_code}")
        except Exception as e:
            print(f"   API Exception: {e}")

async def demo_cli_execution():
    """Demo actual CLI execution."""
    print("\n" + "="*60)
    print("🖥️ CLI EXECUTION DEMO")
    print("="*60)
    
    print("🧪 Testing CLI command execution:")
    
    # Test CLI commands
    cli_commands = [
        ["python", "-c", "import sys; sys.path.append('.'); from cybersecurity.cli_interface import main; import asyncio; asyncio.run(main())", "help"],
        ["python", "-c", "import sys; sys.path.append('.'); from cybersecurity.cli_interface import main; import asyncio; asyncio.run(main())", "config", "--output-format", "text"],
        ["python", "-c", "import sys; sys.path.append('.'); from cybersecurity.cli_interface import main; import asyncio; asyncio.run(main())", "scan", "--target", "127.0.0.1", "--scan-type", "port_scan"]
    ]
    
    for i, cmd in enumerate(cli_commands, 1):
        print(f"\n📋 CLI Test {i}: {' '.join(cmd[3:])}")
        print("-" * 40)
        
        try:
            # Note: This is a simplified demo - actual CLI execution would require proper setup
            print("   ⚠️  CLI execution demo (simulated)")
            print("   📝 Command would execute: " + " ".join(cmd))
            print("   ✅ RORO pattern ensures consistent interface")
            
        except Exception as e:
            print(f"   ❌ CLI execution error: {e}")

async def demo_api_server():
    """Demo API server functionality."""
    print("\n" + "="*60)
    print("🚀 API SERVER DEMO")
    print("="*60)
    
    print("🧪 Testing API server capabilities:")
    
    # Check if FastAPI is available
    try:
        print("   ✅ FastAPI is available")
        
        # Show API endpoints
        endpoints = [
            {"method": "GET", "path": "/health", "description": "Health check"},
            {"method": "POST", "path": "/scan", "description": "Perform network scan"},
            {"method": "POST", "path": "/rate-limit", "description": "Check rate limits"},
            {"method": "POST", "path": "/secrets", "description": "Get secrets"},
            {"method": "GET", "path": "/config", "description": "Get configuration"},
            {"method": "GET", "path": "/docs", "description": "API documentation"}
        ]
        
        print("   📋 Available endpoints:")
        for endpoint in endpoints:
            print(f"      {endpoint['method']} {endpoint['path']} - {endpoint['description']}")
        
        print("   🚀 To start API server:")
        print("      python cybersecurity/api_interface.py --host 0.0.0.0 --port 8000")
        print("   📚 API Documentation: http://localhost:8000/docs")
        print("   🔍 Health Check: http://localhost:8000/health")
        
    except ImportError:
        print("   ❌ FastAPI not available")
        print("   💡 Install with: pip install fastapi uvicorn")

async def demo_integration():
    """Demo integration between CLI and API."""
    print("\n" + "="*60)
    print("🔗 INTEGRATION DEMO")
    print("="*60)
    
    print("🧪 Testing CLI and API integration:")
    
    # Create instances
    cli = CybersecurityCLI()
    api = CybersecurityAPI()
    
    # Test same operation via both interfaces
    target = "127.0.0.1"
    scan_type = "port_scan"
    
    print(f"📋 Testing scan operation on {target}:")
    print("-" * 40)
    
    # CLI scan
    cli_request = CLIRequest(
        command="scan",
        target=target,
        scan_type=scan_type,
        user="integration_user"
    )
    
    try:
        cli_response = await cli.execute_command(cli_request)
        print(f"   CLI Result: {'✅ Success' if cli_response.success else '❌ Error'}")
        if cli_response.success and cli_response.data:
            print(f"      Scan type: {cli_response.data.get('scan_type', 'unknown')}")
            print(f"      Target: {cli_response.data.get('target', 'unknown')}")
    except Exception as e:
        print(f"   CLI Error: {e}")
    
    # API scan
    api_request = APIRequest(
        endpoint="/scan",
        method="POST",
        data={
            "target": target,
            "scan_type": scan_type,
            "user": "integration_user"
        },
        api_key="demo_api_key_12345"
    )
    
    try:
        api_response = await api.execute_api_request(api_request)
        print(f"   API Result: {'✅ Success' if api_response.success else '❌ Error'}")
        if api_response.success and api_response.data:
            print(f"      Scan type: {api_response.data.get('scan_type', 'unknown')}")
            print(f"      Target: {api_response.data.get('target', 'unknown')}")
    except Exception as e:
        print(f"   API Error: {e}")
    
    print(f"\n✅ RORO pattern ensures consistent behavior across interfaces")

async def main():
    """Main demo function."""
    print("🔄 RORO INTERFACES DEMO")
    print("="*60)
    print("This demo showcases CLI and RESTful API interfaces")
    print("using the RORO (Receive an Object, Return an Object) pattern.")
    
    # Run demos
    await demo_roro_pattern()
    await demo_cli_interface()
    await demo_api_interface()
    await demo_error_handling()
    await demo_cli_execution()
    await demo_api_server()
    await demo_integration()
    
    print("\n" + "="*60)
    print("✅ RORO INTERFACES DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key features demonstrated:")
    print("• RORO pattern implementation")
    print("• CLI interface with structured requests/responses")
    print("• RESTful API interface with FastAPI")
    print("• Consistent error handling")
    print("• Type safety with dataclasses")
    print("• Integration between CLI and API")
    print("• Extensible architecture")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 