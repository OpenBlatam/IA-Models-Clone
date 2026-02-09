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
    from cybersecurity.scanners.http_scanner import HTTPScanConfig, HTTPScanner
        import aiohttp
        import httpx
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo script for HTTP scanning with aiohttp and httpx.
Showcases web security assessment and HTTP-based tools.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("✓ HTTP scanning modules loaded successfully!")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

async def demo_http_scanning_aiohttp():
    """Demo HTTP scanning with aiohttp."""
    print("\n" + "="*60)
    print("🌐 HTTP SCANNING DEMO (aiohttp)")
    print("="*60)
    
    config = HTTPScanConfig(
        timeout=10.0,
        max_workers=10,
        follow_redirects=True,
        verify_ssl=True,
        check_security_headers=True,
        user_agent="Mozilla/5.0 (Security Scanner Demo) Chrome/91.0.4472.124"
    )
    
    scanner = HTTPScanner(config)
    
    # Test targets (replace with actual targets)
    test_targets = [
        "https://httpbin.org",
        "https://example.com",
        "https://google.com"
    ]
    
    for target in test_targets:
        print(f"\n🔍 Scanning HTTP target: {target}")
        print("-" * 40)
        
        try:
            results = await scanner.comprehensive_scan(target, use_httpx=False)
            
            print(f"✅ HTTP scan completed for {target}")
            
            scan_results = results.get('scan_results', {})
            if scan_results.get('success'):
                print(f"🟢 Status: {scan_results.get('status_code', 'Unknown')}")
                print(f"⏱️  Response time: {scan_results.get('response_time', 0):.3f}s")
                print(f"📏 Content length: {scan_results.get('content_length', 0)} bytes")
                
                # Security headers analysis
                security_headers = scan_results.get('security_headers', {})
                print(f"🔒 Security headers found: {len([h for h in security_headers.values() if h.get('present')])}")
                
                for header, info in security_headers.items():
                    if info.get('present'):
                        print(f"   ✅ {header}: {info.get('value', 'Present')}")
                    else:
                        print(f"   ❌ {header}: Not found")
                
                # SSL information
                ssl_info = scan_results.get('ssl_info')
                if ssl_info:
                    print(f"🔐 SSL/TLS: Enabled")
                    print(f"   Cipher: {ssl_info.get('cipher', 'Unknown')}")
                else:
                    print(f"🔓 SSL/TLS: Not enabled")
                
            else:
                print(f"🔴 Scan failed: {scan_results.get('error_message', 'Unknown error')}")
            
            # Analysis
            analysis = results.get('analysis')
            if analysis:
                print(f"📊 Security score: {analysis.get('security_score', 0)}/100")
                recommendations = analysis.get('recommendations', [])
                if recommendations:
                    print(f"💡 Recommendations:")
                    for rec in recommendations[:3]:  # Show first 3
                        print(f"   • {rec}")
            
        except Exception as e:
            print(f"❌ Error scanning {target}: {e}")

async def demo_http_scanning_httpx():
    """Demo HTTP scanning with httpx."""
    print("\n" + "="*60)
    print("🚀 HTTP SCANNING DEMO (httpx)")
    print("="*60)
    
    config = HTTPScanConfig(
        timeout=8.0,
        max_workers=15,
        follow_redirects=True,
        verify_ssl=True,
        check_security_headers=True,
        user_agent="Mozilla/5.0 (Security Scanner Demo) Chrome/91.0.4472.124"
    )
    
    scanner = HTTPScanner(config)
    
    # Test targets
    test_targets = [
        "https://httpbin.org/headers",
        "https://httpbin.org/status/200",
        "https://httpbin.org/redirect/1"
    ]
    
    for target in test_targets:
        print(f"\n🔍 Scanning HTTP target: {target}")
        print("-" * 40)
        
        try:
            results = await scanner.comprehensive_scan(target, use_httpx=True)
            
            print(f"✅ HTTP scan completed for {target}")
            
            scan_results = results.get('scan_results', {})
            if scan_results.get('success'):
                print(f"🟢 Status: {scan_results.get('status_code', 'Unknown')}")
                print(f"⏱️  Response time: {scan_results.get('response_time', 0):.3f}s")
                
                # Headers analysis
                headers = scan_results.get('headers', {})
                print(f"📋 Response headers: {len(headers)}")
                
                # Show some interesting headers
                interesting_headers = ['server', 'date', 'content-type', 'cache-control']
                for header in interesting_headers:
                    if header in headers:
                        print(f"   • {header}: {headers[header]}")
                
                # Security analysis
                security_headers = scan_results.get('security_headers', {})
                security_count = len([h for h in security_headers.values() if h.get('present')])
                print(f"🔒 Security headers: {security_count} found")
                
            else:
                print(f"🔴 Scan failed: {scan_results.get('error_message', 'Unknown error')}")
            
        except Exception as e:
            print(f"❌ Error scanning {target}: {e}")

async def demo_concurrent_http_scanning():
    """Demo concurrent HTTP scanning of multiple targets."""
    print("\n" + "="*60)
    print("⚡ CONCURRENT HTTP SCANNING DEMO")
    print("="*60)
    
    config = HTTPScanConfig(
        timeout=5.0,
        max_workers=20,
        follow_redirects=True,
        verify_ssl=True,
        check_security_headers=True
    )
    
    scanner = HTTPScanner(config)
    
    # Multiple test targets
    targets = [
        "https://httpbin.org",
        "https://example.com",
        "https://httpbin.org/status/404",
        "https://httpbin.org/status/500",
        "https://httpbin.org/delay/1"
    ]
    
    print(f"🎯 Scanning {len(targets)} HTTP targets concurrently...")
    
    try:
        # Test with aiohttp
        print("\n📡 Using aiohttp:")
        results_aiohttp = await scanner.scan_multiple_targets(targets, use_httpx=False)
        
        print(f"✅ Concurrent scan completed!")
        print(f"📊 Summary:")
        print(f"   • Total targets: {results_aiohttp['summary']['total']}")
        print(f"   • Successful scans: {results_aiohttp['summary']['successful']}")
        print(f"   • Failed scans: {results_aiohttp['summary']['failed']}")
        
        analysis = results_aiohttp.get('analysis', {})
        if analysis:
            print(f"   • Average response time: {analysis.get('average_response_time', 0):.3f}s")
            print(f"   • SSL enabled: {analysis.get('ssl_enabled', 0)}")
            print(f"   • Redirects found: {analysis.get('redirects_found', 0)}")
        
        # Test with httpx
        print("\n📡 Using httpx:")
        results_httpx = await scanner.scan_multiple_targets(targets, use_httpx=True)
        
        print(f"✅ Concurrent scan completed!")
        print(f"📊 Summary:")
        print(f"   • Total targets: {results_httpx['summary']['total']}")
        print(f"   • Successful scans: {results_httpx['summary']['successful']}")
        print(f"   • Failed scans: {results_httpx['summary']['failed']}")
        
        analysis = results_httpx.get('analysis', {})
        if analysis:
            print(f"   • Average response time: {analysis.get('average_response_time', 0):.3f}s")
            print(f"   • SSL enabled: {analysis.get('ssl_enabled', 0)}")
            print(f"   • Redirects found: {analysis.get('redirects_found', 0)}")
            
    except Exception as e:
        print(f"❌ Error in concurrent scan: {e}")

async def demo_security_assessment():
    """Demo comprehensive security assessment."""
    print("\n" + "="*60)
    print("🛡️ SECURITY ASSESSMENT DEMO")
    print("="*60)
    
    config = HTTPScanConfig(
        timeout=10.0,
        max_workers=5,
        check_security_headers=True,
        check_cors=True,
        check_csp=True,
        check_hsts=True,
        check_xss_protection=True
    )
    
    scanner = HTTPScanner(config)
    
    # Security-focused targets
    security_targets = [
        "https://securityheaders.com",  # Good security headers
        "https://httpbin.org",         # Basic headers
        "https://example.com"          # Minimal headers
    ]
    
    for target in security_targets:
        print(f"\n🔒 Security assessment: {target}")
        print("-" * 40)
        
        try:
            results = await scanner.comprehensive_scan(target)
            
            scan_results = results.get('scan_results', {})
            if scan_results.get('success'):
                analysis = results.get('analysis', {})
                security_score = analysis.get('security_score', 0)
                
                print(f"📊 Security Score: {security_score}/100")
                
                # Score interpretation
                if security_score >= 80:
                    print(f"🟢 Excellent security posture")
                elif security_score >= 60:
                    print(f"🟡 Good security posture")
                elif security_score >= 40:
                    print(f"🟠 Moderate security posture")
                else:
                    print(f"🔴 Poor security posture")
                
                # Recommendations
                recommendations = analysis.get('recommendations', [])
                if recommendations:
                    print(f"💡 Key recommendations:")
                    for i, rec in enumerate(recommendations[:5], 1):
                        print(f"   {i}. {rec}")
                else:
                    print(f"✅ No immediate security recommendations")
                
                # Detailed security headers
                security_headers = scan_results.get('security_headers', {})
                print(f"🔍 Security headers analysis:")
                for header, info in security_headers.items():
                    status = "✅" if info.get('present') else "❌"
                    print(f"   {status} {header}")
                
            else:
                print(f"❌ Assessment failed: {scan_results.get('error_message')}")
                
        except Exception as e:
            print(f"❌ Error in security assessment: {e}")

async def demo_library_availability():
    """Check and display HTTP library availability."""
    print("\n" + "="*60)
    print("📚 HTTP LIBRARY AVAILABILITY CHECK")
    print("="*60)
    
    # Check aiohttp
    try:
        print("✅ aiohttp: Available")
        print(f"   Version: {aiohttp.__version__}")
    except ImportError:
        print("❌ aiohttp: Not available")
        print("   Install with: pip install aiohttp")
    
    # Check httpx
    try:
        print("✅ httpx: Available")
        print(f"   Version: {httpx.__version__}")
    except ImportError:
        print("❌ httpx: Not available")
        print("   Install with: pip install httpx")
    
    # Check other dependencies
    dependencies = [
        ("urllib", "URL parsing"),
        ("ssl", "SSL/TLS support"),
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
    print("🚀 HTTP SCANNING & SECURITY ASSESSMENT DEMO")
    print("="*60)
    print("This demo showcases HTTP-based security tools using aiohttp and httpx.")
    print("⚠️  Note: This is a demonstration. Use responsibly and only on authorized targets.")
    
    # Check library availability
    await demo_library_availability()
    
    # Run demos
    await demo_http_scanning_aiohttp()
    await demo_http_scanning_httpx()
    await demo_concurrent_http_scanning()
    await demo_security_assessment()
    
    print("\n" + "="*60)
    print("✅ HTTP SCANNING DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key features demonstrated:")
    print("• HTTP security header analysis")
    print("• SSL/TLS certificate assessment")
    print("• Concurrent HTTP scanning")
    print("• Security scoring and recommendations")
    print("• Multiple HTTP client support (aiohttp/httpx)")
    print("• Comprehensive error handling and validation")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 