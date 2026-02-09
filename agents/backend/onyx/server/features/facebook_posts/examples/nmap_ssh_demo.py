from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import os
import time
from typing import Dict, Any
    from cybersecurity.scanners.port_scanner import PortScanConfig, PortScanner
    from cybersecurity.scanners.ssh_scanner import SSHScanConfig, SSHScanner
        import nmap
        import asyncssh
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo script for python-nmap and asyncssh integration.
Showcases enhanced port scanning and SSH interaction capabilities.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("✓ Cybersecurity modules loaded successfully!")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

async def demo_port_scanning():
    """Demo enhanced port scanning with python-nmap."""
    print("\n" + "="*60)
    print("🔍 ENHANCED PORT SCANNING DEMO (python-nmap)")
    print("="*60)
    
    config = PortScanConfig(
        timeout=2.0,
        max_workers=20,
        use_nmap=True,
        nmap_arguments="-sS -sV -O --version-intensity 3"
    )
    
    scanner = PortScanner(config)
    
    # Test targets (replace with actual targets)
    test_targets = ["127.0.0.1", "localhost"]
    
    for target in test_targets:
        print(f"\n📡 Scanning target: {target}")
        print("-" * 40)
        
        try:
            # Common ports to scan
            common_ports = [22, 80, 443, 8080, 3306, 5432]
            
            results = await scanner.comprehensive_scan(target, common_ports)
            
            print(f"✅ Scan completed for {target}")
            print(f"📊 Methods used: {list(results['methods'].keys())}")
            
            if 'async_basic' in results['methods']:
                basic_results = results['methods']['async_basic']
                analysis = basic_results.get('analysis', {})
                print(f"🔢 Basic scan: {analysis.get('open_ports', 0)}/{analysis.get('total_ports', 0)} ports open")
            
            if 'nmap' in results['methods']:
                nmap_results = results['methods']['nmap']
                if 'results' in nmap_results:
                    print(f"🔍 Nmap found {len(nmap_results['results'])} services")
                    for service in nmap_results['results'][:3]:  # Show first 3
                        print(f"   • {service.get('service', 'unknown')} on port {service.get('port', 'unknown')}")
            
        except Exception as e:
            print(f"❌ Error scanning {target}: {e}")

async def demo_ssh_scanning():
    """Demo SSH scanning with asyncssh."""
    print("\n" + "="*60)
    print("🔐 SSH SCANNING DEMO (asyncssh)")
    print("="*60)
    
    config = SSHScanConfig(
        timeout=5.0,
        max_workers=5,
        banner_grab=True,
        version_detection=True,
        auth_testing=False  # Disabled for demo safety
    )
    
    scanner = SSHScanner(config)
    
    # Test SSH targets (replace with actual targets)
    test_targets = ["127.0.0.1:22", "localhost:22"]
    
    for target in test_targets:
        print(f"\n🔐 Scanning SSH target: {target}")
        print("-" * 40)
        
        try:
            results = await scanner.comprehensive_scan(target)
            
            print(f"✅ SSH scan completed for {target}")
            
            port_check = results.get('port_check', {})
            if port_check.get('is_open'):
                print(f"🟢 SSH port is open")
                print(f"⏱️  Response time: {port_check.get('response_time', 0):.3f}s")
                
                if port_check.get('banner'):
                    print(f"📋 Banner: {port_check['banner'][:100]}...")
            else:
                print(f"🔴 SSH port is closed")
                if port_check.get('error_message'):
                    print(f"❌ Error: {port_check['error_message']}")
            
            ssh_info = results.get('ssh_info')
            if ssh_info and ssh_info.get('is_open'):
                print(f"🔍 SSH Version: {ssh_info.get('ssh_version', 'Unknown')}")
                print(f"🔐 Encryption algorithms: {len(ssh_info.get('encryption_algorithms', []))}")
                print(f"🔑 Key exchange algorithms: {len(ssh_info.get('key_exchange_algorithms', []))}")
            
        except Exception as e:
            print(f"❌ Error scanning SSH {target}: {e}")

async def demo_multiple_ssh_targets():
    """Demo scanning multiple SSH targets concurrently."""
    print("\n" + "="*60)
    print("🚀 CONCURRENT SSH SCANNING DEMO")
    print("="*60)
    
    config = SSHScanConfig(
        timeout=3.0,
        max_workers=10,
        banner_grab=True
    )
    
    scanner = SSHScanner(config)
    
    # Multiple test targets
    targets = [
        "127.0.0.1:22",
        "localhost:22",
        "192.168.1.1:22",  # Example router
        "10.0.0.1:22"      # Example network
    ]
    
    print(f"🎯 Scanning {len(targets)} SSH targets concurrently...")
    
    try:
        results = await scanner.scan_multiple_targets(targets)
        
        print(f"✅ Concurrent scan completed!")
        print(f"📊 Summary:")
        print(f"   • Total targets: {results['summary']['total']}")
        print(f"   • Open SSH ports: {results['summary']['open']}")
        print(f"   • Closed ports: {results['summary']['closed']}")
        
        for result in results['results']:
            status = "🟢 OPEN" if result['is_open'] else "🔴 CLOSED"
            print(f"   • {result['target']}: {status}")
            
    except Exception as e:
        print(f"❌ Error in concurrent scan: {e}")

async def demo_library_availability():
    """Check and display library availability."""
    print("\n" + "="*60)
    print("📚 LIBRARY AVAILABILITY CHECK")
    print("="*60)
    
    # Check python-nmap
    try:
        print("✅ python-nmap: Available")
        print(f"   Version: {nmap.__version__}")
    except ImportError:
        print("❌ python-nmap: Not available")
        print("   Install with: pip install python-nmap")
    
    # Check asyncssh
    try:
        print("✅ asyncssh: Available")
        print(f"   Version: {asyncssh.__version__}")
    except ImportError:
        print("❌ asyncssh: Not available")
        print("   Install with: pip install asyncssh")
    
    # Check other dependencies
    dependencies = [
        ("aiohttp", "Async HTTP client"),
        ("cryptography", "Cryptographic operations"),
        ("pydantic", "Data validation"),
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
    print("🚀 PYTHON-NMAP & ASYNCSSH INTEGRATION DEMO")
    print("="*60)
    print("This demo showcases enhanced port scanning and SSH interaction capabilities.")
    print("⚠️  Note: This is a demonstration. Use responsibly and only on authorized targets.")
    
    # Check library availability
    await demo_library_availability()
    
    # Run demos
    await demo_port_scanning()
    await demo_ssh_scanning()
    await demo_multiple_ssh_targets()
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key features demonstrated:")
    print("• Enhanced port scanning with python-nmap integration")
    print("• SSH banner grabbing and version detection")
    print("• Concurrent SSH target scanning")
    print("• Comprehensive error handling and validation")
    print("• Async/await patterns for network operations")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 