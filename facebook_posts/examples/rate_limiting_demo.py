from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import os
import time
from typing import Dict, Any, List
    from cybersecurity.security_implementation import (
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo script for rate limiting and back-off mechanisms.
Showcases advanced rate limiting, adaptive limits, and back-off strategies for network scans.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
        SecurityConfig, RateLimiter, AdaptiveRateLimiter, NetworkScanRateLimiter,
        SecureNetworkScanner, SecurityError, create_secure_config
    )
    print("✓ Rate limiting modules loaded successfully!")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

async def demo_basic_rate_limiting():
    """Demo basic rate limiting functionality."""
    print("\n" + "="*60)
    print("🚦 BASIC RATE LIMITING DEMO")
    print("="*60)
    
    # Create rate limiter with conservative limits for demo
    rate_limiter = RateLimiter(max_requests=5, window_seconds=10, backoff_multiplier=2.0)
    
    print("📊 Rate limiter configuration:")
    print(f"   • Max requests: {rate_limiter.max_requests}")
    print(f"   • Window: {rate_limiter.window_seconds} seconds")
    print(f"   • Back-off multiplier: {rate_limiter.backoff_multiplier}")
    print(f"   • Max back-off: {rate_limiter.max_backoff} seconds")
    
    # Test rapid requests
    print(f"\n🧪 Testing rapid requests (should trigger back-off):")
    user_id = "test_user"
    
    for i in range(8):
        try:
            is_allowed = await rate_limiter.check_rate_limit(user_id)
            remaining = rate_limiter.get_remaining_requests(user_id)
            backoff_status = rate_limiter.get_backoff_status(user_id)
            
            print(f"   Request {i+1}: {'✅ ALLOWED' if is_allowed else '❌ BLOCKED'}")
            print(f"      Remaining: {remaining}")
            
            if backoff_status['in_backoff']:
                print(f"      ⏰ Back-off: {backoff_status['remaining_backoff']:.1f}s remaining")
                print(f"      🔢 Failure count: {backoff_status['failure_count']}")
            
        except SecurityError as e:
            print(f"   Request {i+1}: ❌ {e.code} - {e.message}")
        
        # Small delay between requests
        await asyncio.sleep(0.1)
    
    # Wait for back-off to expire
    print(f"\n⏳ Waiting for back-off to expire...")
    await asyncio.sleep(5)
    
    # Test after back-off
    print(f"\n🧪 Testing after back-off expiration:")
    try:
        is_allowed = await rate_limiter.check_rate_limit(user_id)
        remaining = rate_limiter.get_remaining_requests(user_id)
        print(f"   ✅ Request allowed: {is_allowed}")
        print(f"   📊 Remaining requests: {remaining}")
    except SecurityError as e:
        print(f"   ❌ Still blocked: {e.message}")

async def demo_adaptive_rate_limiting():
    """Demo adaptive rate limiting based on target responses."""
    print("\n" + "="*60)
    print("🔄 ADAPTIVE RATE LIMITING DEMO")
    print("="*60)
    
    adaptive_limiter = AdaptiveRateLimiter(base_max_requests=10, base_window_seconds=30)
    
    # Test different response scenarios
    test_scenarios = [
        {
            "name": "Normal Response",
            "response": {"status_code": 200, "response_time": 1.0, "success": True}
        },
        {
            "name": "Rate Limited (429)",
            "response": {"status_code": 429, "response_time": 0.5, "success": False}
        },
        {
            "name": "Server Error (500)",
            "response": {"status_code": 500, "response_time": 2.0, "success": False}
        },
        {
            "name": "Slow Response",
            "response": {"status_code": 200, "response_time": 8.0, "success": True}
        }
    ]
    
    target = "example.com"
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing: {scenario['name']}")
        print("-" * 40)
        
        try:
            # Check rate limit with response
            is_allowed = await adaptive_limiter.check_rate_limit(target, scenario['response'])
            print(f"   ✅ Rate limit check: {'ALLOWED' if is_allowed else 'BLOCKED'}")
            
            # Get adaptive stats
            stats = adaptive_limiter.get_adaptive_stats(target)
            print(f"   📊 Adaptive config: {stats.get('adaptive_config', {})}")
            
        except SecurityError as e:
            print(f"   ❌ Rate limit error: {e.message}")
    
    # Show final adaptive stats
    print(f"\n📈 Final adaptive statistics:")
    final_stats = adaptive_limiter.get_adaptive_stats(target)
    print(f"   Target: {final_stats.get('target', 'Unknown')}")
    print(f"   Recent responses: {final_stats.get('recent_responses', 0)}")
    print(f"   Adaptive config: {final_stats.get('adaptive_config', {})}")

async def demo_network_scan_rate_limiting():
    """Demo network scan-specific rate limiting."""
    print("\n" + "="*60)
    print("🔍 NETWORK SCAN RATE LIMITING DEMO")
    print("="*60)
    
    scan_limiter = NetworkScanRateLimiter()
    
    # Show scan capabilities
    print("📋 Available scan types and their rate limits:")
    for scan_type, limits in scan_limiter.scan_limits.items():
        print(f"   • {scan_type}: {limits['max_requests']} requests per {limits['window_seconds']}s")
    
    # Test different scan types
    test_targets = ["192.168.1.1", "10.0.0.1", "172.16.0.1"]
    scan_types = ["port_scan", "vulnerability_scan", "web_scan", "network_discovery"]
    
    print(f"\n🧪 Testing scan rate limits:")
    
    for scan_type in scan_types:
        print(f"\n📡 Testing {scan_type}:")
        for i, target in enumerate(test_targets):
            try:
                is_allowed = await scan_limiter.check_scan_rate_limit(scan_type, target)
                print(f"   • {target}: {'✅ ALLOWED' if is_allowed else '❌ BLOCKED'}")
                
                # Record a sample result
                sample_result = {
                    "target": target,
                    "scan_type": scan_type,
                    "success": True,
                    "timestamp": time.time()
                }
                scan_limiter.record_scan_result(scan_type, target, sample_result)
                
            except SecurityError as e:
                print(f"   • {target}: ❌ {e.code} - {e.message}")
    
    # Show scan statistics
    print(f"\n📊 Scan statistics:")
    for scan_type in scan_types:
        stats = scan_limiter.get_scan_stats(scan_type)
        print(f"   • {scan_type}: {stats}")

async def demo_secure_network_scanner():
    """Demo secure network scanner with rate limiting."""
    print("\n" + "="*60)
    print("🛡️ SECURE NETWORK SCANNER DEMO")
    print("="*60)
    
    # Create secure configuration
    config = create_secure_config()
    config.rate_limit = 20  # Conservative limit for demo
    
    # Create scanner
    scanner = SecureNetworkScanner(config)
    
    # Setup authorization and consent
    scanner.auth_checker.add_authorized_target("192.168.1.1", "demo_user", 
                                             int(time.time()) + 3600, ["scan"])
    scanner.auth_checker.record_consent("demo_user", True, "network_scanning")
    
    print("✅ Scanner initialized with rate limiting")
    print(f"📊 Configuration: {config.rate_limit} requests per minute")
    
    # Test different scan types
    test_targets = ["192.168.1.1", "10.0.0.1"]
    scan_types = ["port_scan", "web_scan", "vulnerability_scan"]
    
    print(f"\n🧪 Testing secure scans with rate limiting:")
    
    for scan_type in scan_types:
        print(f"\n📡 Testing {scan_type}:")
        for target in test_targets:
            try:
                result = await scanner.secure_scan(target, "demo_user", "session_123", scan_type)
                
                if result['success']:
                    print(f"   ✅ {target}: Scan completed successfully")
                    print(f"      📊 Scan type: {result['scan_type']}")
                    print(f"      ⏱️  Timestamp: {result['timestamp']}")
                    
                    # Show rate limit info
                    rate_info = result.get('rate_limit_info', {})
                    if rate_info:
                        print(f"      📈 Rate limit info available")
                else:
                    print(f"   ❌ {target}: Scan failed - {result.get('error', 'Unknown error')}")
                    
            except SecurityError as e:
                print(f"   ❌ {target}: Security error - {e.code}: {e.message}")
    
    # Show scan capabilities
    print(f"\n📋 Scanner capabilities:")
    capabilities = scanner.get_scan_capabilities()
    for scan_type, info in capabilities['scan_types'].items():
        print(f"   • {scan_type}: {info['description']}")
        print(f"     Rate limit: {info['rate_limit']['max_requests']} per {info['rate_limit']['window_seconds']}s")

async def demo_back_off_strategies():
    """Demo different back-off strategies."""
    print("\n" + "="*60)
    print("⏰ BACK-OFF STRATEGIES DEMO")
    print("="*60)
    
    # Test different back-off configurations
    backoff_configs = [
        {"name": "Conservative", "multiplier": 1.5, "max_backoff": 300},
        {"name": "Standard", "multiplier": 2.0, "max_backoff": 3600},
        {"name": "Aggressive", "multiplier": 3.0, "max_backoff": 7200}
    ]
    
    for config in backoff_configs:
        print(f"\n🧪 Testing {config['name']} back-off strategy:")
        print("-" * 40)
        
        rate_limiter = RateLimiter(
            max_requests=3,
            window_seconds=10,
            backoff_multiplier=config['multiplier'],
            max_backoff=config['max_backoff']
        )
        
        user_id = f"user_{config['name'].lower()}"
        
        # Trigger rate limit
        for i in range(5):
            try:
                await rate_limiter.check_rate_limit(user_id)
                print(f"   Request {i+1}: ✅ ALLOWED")
            except SecurityError as e:
                print(f"   Request {i+1}: ❌ {e.code}")
                break
        
        # Show back-off status
        backoff_status = rate_limiter.get_backoff_status(user_id)
        if backoff_status['in_backoff']:
            print(f"   ⏰ Back-off active: {backoff_status['remaining_backoff']:.1f}s")
            print(f"   🔢 Failure count: {backoff_status['failure_count']}")

async def demo_rate_limit_monitoring():
    """Demo rate limit monitoring and statistics."""
    print("\n" + "="*60)
    print("📊 RATE LIMIT MONITORING DEMO")
    print("="*60)
    
    # Create rate limiter
    rate_limiter = RateLimiter(max_requests=10, window_seconds=30)
    
    # Simulate some activity
    user_id = "monitor_user"
    
    print("📈 Simulating rate limit activity:")
    for i in range(12):
        try:
            await rate_limiter.check_rate_limit(user_id)
            print(f"   Request {i+1}: ✅ ALLOWED")
        except SecurityError as e:
            print(f"   Request {i+1}: ❌ {e.code}")
            break
        
        await asyncio.sleep(0.1)
    
    # Show comprehensive statistics
    print(f"\n📊 Rate limit statistics:")
    stats = rate_limiter.get_rate_limit_stats(user_id)
    
    for key, value in stats.items():
        if key == "backoff_status":
            print(f"   📈 {key}:")
            for sub_key, sub_value in value.items():
                print(f"      • {sub_key}: {sub_value}")
        else:
            print(f"   📈 {key}: {value}")

async def demo_ethical_considerations():
    """Demo ethical considerations for rate limiting."""
    print("\n" + "="*60)
    print("🤝 ETHICAL CONSIDERATIONS DEMO")
    print("="*60)
    
    print("🔒 Rate limiting ethical principles:")
    print("   ✅ Respect target systems and their resources")
    print("   ✅ Avoid overwhelming networks or services")
    print("   ✅ Implement progressive back-off strategies")
    print("   ✅ Monitor and adjust based on target responses")
    print("   ✅ Provide clear feedback on rate limit status")
    print("   ✅ Allow manual back-off reset for legitimate use")
    
    print(f"\n🛡️ Security benefits:")
    print("   ✅ Prevents detection by IDS/IPS systems")
    print("   ✅ Reduces risk of being blocked or blacklisted")
    print("   ✅ Maintains operational stealth")
    print("   ✅ Respects network infrastructure")
    print("   ✅ Enables sustainable scanning operations")
    
    print(f"\n📋 Best practices implemented:")
    print("   ✅ Adaptive rate limiting based on responses")
    print("   ✅ Exponential back-off with maximum limits")
    print("   ✅ Scan-type specific rate limits")
    print("   ✅ Comprehensive monitoring and statistics")
    print("   ✅ Secure logging with sensitive data redaction")

async def main():
    """Main demo function."""
    print("🚦 RATE LIMITING & BACK-OFF DEMO")
    print("="*60)
    print("This demo showcases advanced rate limiting, adaptive limits,")
    print("and back-off strategies for ethical network scanning.")
    
    # Run demos
    await demo_basic_rate_limiting()
    await demo_adaptive_rate_limiting()
    await demo_network_scan_rate_limiting()
    await demo_secure_network_scanner()
    await demo_back_off_strategies()
    await demo_rate_limit_monitoring()
    await demo_ethical_considerations()
    
    print("\n" + "="*60)
    print("✅ RATE LIMITING & BACK-OFF DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key features demonstrated:")
    print("• Advanced rate limiting with exponential back-off")
    print("• Adaptive rate limiting based on target responses")
    print("• Scan-type specific rate limits")
    print("• Comprehensive monitoring and statistics")
    print("• Ethical scanning practices")
    print("• Detection avoidance strategies")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 