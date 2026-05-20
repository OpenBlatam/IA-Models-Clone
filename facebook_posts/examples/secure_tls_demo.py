from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import os
import ssl
import socket
import time
from typing import Dict, Any, List
    from cybersecurity.security_implementation import (
        import ssl
        import socket
        import cryptography
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo script for secure TLS defaults and cipher suite configuration.
Showcases TLSv1.2+ security, strong cipher suites, and secure defaults.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
        SecurityConfig, SecureTLSConfig, create_secure_ssl_context,
        validate_cipher_suite, SecurityError
    )
    print("✓ Secure TLS implementation modules loaded successfully!")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

def demo_tls_configuration():
    """Demo secure TLS configuration."""
    print("\n" + "="*60)
    print("🔐 SECURE TLS CONFIGURATION DEMO")
    print("="*60)
    
    tls_config = SecureTLSConfig()
    
    # Demo strong cipher suites
    print("🔒 Strong Cipher Suites (TLS 1.2+):")
    for i, cipher in enumerate(tls_config.strong_cipher_suites, 1):
        is_valid = tls_config.validate_cipher_suite(cipher)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"   {i:2d}. {cipher:<35} {status}")
    
    print(f"\n🔒 TLS 1.3 Cipher Suites:")
    for i, cipher in enumerate(tls_config.tls13_cipher_suites, 1):
        is_valid = tls_config.validate_cipher_suite(cipher)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"   {i:2d}. {cipher:<35} {status}")
    
    # Demo weak cipher detection
    print(f"\n❌ Weak Cipher Detection:")
    weak_ciphers = [
        "NULL-SHA",
        "aNULL",
        "DES-CBC-SHA",
        "RC4-SHA",
        "MD5",
        "EXPORT-RC4-MD5"
    ]
    
    for cipher in weak_ciphers:
        is_valid = tls_config.validate_cipher_suite(cipher)
        status = "❌ BLOCKED" if not is_valid else "⚠️  SHOULD BE BLOCKED"
        print(f"   • {cipher:<20} {status}")

def demo_ssl_context_creation():
    """Demo secure SSL context creation."""
    print("\n" + "="*60)
    print("🔧 SECURE SSL CONTEXT CREATION DEMO")
    print("="*60)
    
    tls_config = SecureTLSConfig()
    
    # Test TLS 1.2 context
    print("🔐 Creating TLS 1.2 SSL Context:")
    try:
        context_tls12 = tls_config.create_secure_ssl_context("TLSv1.2")
        print("   ✅ TLS 1.2 context created successfully")
        print(f"   📊 Minimum version: {context_tls12.minimum_version}")
        print(f"   📊 Maximum version: {context_tls12.maximum_version}")
        print(f"   🔒 Verify mode: {context_tls12.verify_mode}")
        print(f"   🔍 Check hostname: {context_tls12.check_hostname}")
        
        # Get cipher list
        cipher_list = context_tls12.get_ciphers()
        print(f"   🔐 Cipher count: {len(cipher_list)}")
        print("   📋 First 5 ciphers:")
        for i, cipher in enumerate(cipher_list[:5], 1):
            print(f"      {i}. {cipher['name']}")
            
    except Exception as e:
        print(f"   ❌ Error creating TLS 1.2 context: {e}")
    
    # Test TLS 1.3 context
    print(f"\n🔐 Creating TLS 1.3 SSL Context:")
    try:
        context_tls13 = tls_config.create_secure_ssl_context("TLSv1.3")
        print("   ✅ TLS 1.3 context created successfully")
        print(f"   📊 Minimum version: {context_tls13.minimum_version}")
        print(f"   📊 Maximum version: {context_tls13.maximum_version}")
        print(f"   🔒 Verify mode: {context_tls13.verify_mode}")
        print(f"   🔍 Check hostname: {context_tls13.check_hostname}")
        
        # Get cipher list
        cipher_list = context_tls13.get_ciphers()
        print(f"   🔐 Cipher count: {len(cipher_list)}")
        print("   📋 Available ciphers:")
        for i, cipher in enumerate(cipher_list, 1):
            print(f"      {i}. {cipher['name']}")
            
    except Exception as e:
        print(f"   ❌ Error creating TLS 1.3 context: {e}")

def demo_cipher_validation():
    """Demo cipher suite validation."""
    print("\n" + "="*60)
    print("🔍 CIPHER SUITE VALIDATION DEMO")
    print("="*60)
    
    tls_config = SecureTLSConfig()
    
    # Test various cipher suites
    test_ciphers = [
        # Strong ciphers (should pass)
        ("ECDHE-RSA-AES256-GCM-SHA384", True),
        ("ECDHE-RSA-AES128-GCM-SHA256", True),
        ("TLS_AES_256_GCM_SHA384", True),
        ("TLS_CHACHA20_POLY1305_SHA256", True),
        
        # Weak ciphers (should fail)
        ("NULL-SHA", False),
        ("DES-CBC-SHA", False),
        ("RC4-SHA", False),
        ("EXPORT-RC4-MD5", False),
        ("aNULL", False),
        ("MD5", False)
    ]
    
    print("🧪 Testing cipher suite validation:")
    for cipher, expected in test_ciphers:
        is_valid = tls_config.validate_cipher_suite(cipher)
        status = "✅ PASS" if is_valid == expected else "❌ FAIL"
        expected_text = "VALID" if expected else "INVALID"
        actual_text = "VALID" if is_valid else "INVALID"
        print(f"   • {cipher:<35} Expected: {expected_text:<8} Actual: {actual_text:<8} {status}")

def demo_secure_config():
    """Demo secure configuration with TLS defaults."""
    print("\n" + "="*60)
    print("⚙️ SECURE CONFIGURATION DEMO")
    print("="*60)
    
    # Create secure config
    config = SecurityConfig()
    print("✅ Secure configuration created")
    print(f"   🔐 TLS Version: {config.tls_version}")
    print(f"   🔐 Min TLS Version: {config.min_tls_version}")
    print(f"   🔒 Verify SSL: {config.verify_ssl}")
    print(f"   🔒 Cert Verify Mode: {config.cert_verify_mode}")
    print(f"   ⏱️  Timeout: {config.timeout}s")
    print(f"   🔄 Max Retries: {config.max_retries}")
    print(f"   🚦 Rate Limit: {config.rate_limit}")
    print(f"   ⏰ Session Timeout: {config.session_timeout}s")
    
    # Validate config
    try:
        config.validate()
        print("   ✅ Configuration validation: PASSED")
    except SecurityError as e:
        print(f"   ❌ Configuration validation: FAILED - {e.message}")
    
    # Show cipher suites
    print(f"\n🔐 Configured Cipher Suites ({len(config.cipher_suites)}):")
    for i, cipher in enumerate(config.cipher_suites, 1):
        is_valid = validate_cipher_suite(cipher)
        status = "✅" if is_valid else "❌"
        print(f"   {i:2d}. {status} {cipher}")

def demo_network_security():
    """Demo network security with secure TLS."""
    print("\n" + "="*60)
    print("🌐 NETWORK SECURITY DEMO")
    print("="*60)
    
    # Test targets (replace with actual targets for real testing)
    test_targets = [
        ("https://httpbin.org", 443),
        ("https://example.com", 443),
        ("https://google.com", 443)
    ]
    
    print("🔍 Testing secure TLS connections:")
    
    for hostname, port in test_targets:
        print(f"\n📡 Testing: {hostname}:{port}")
        
        try:
            # Create secure SSL context
            context = create_secure_ssl_context("TLSv1.2")
            
            # Create socket with timeout
            sock = socket.create_connection((hostname, port), timeout=10)
            
            # Wrap with SSL
            ssl_sock = context.wrap_socket(sock, server_hostname=hostname)
            
            # Get connection info
            cipher = ssl_sock.cipher()
            version = ssl_sock.version()
            cert = ssl_sock.getpeercert()
            
            print(f"   ✅ Connection successful")
            print(f"   🔐 TLS Version: {version}")
            print(f"   🔒 Cipher: {cipher[0]}")
            print(f"   🔑 Key Exchange: {cipher[1]}")
            print(f"   🔐 Authentication: {cipher[2]}")
            
            if cert:
                subject = dict(x[0] for x in cert['subject'])
                issuer = dict(x[0] for x in cert['issuer'])
                print(f"   📜 Subject: {subject.get('commonName', 'Unknown')}")
                print(f"   🏢 Issuer: {issuer.get('commonName', 'Unknown')}")
            
            ssl_sock.close()
            
        except ssl.SSLError as e:
            print(f"   ❌ SSL Error: {e}")
        except socket.timeout:
            print(f"   ⏰ Connection timeout")
        except Exception as e:
            print(f"   ❌ Connection failed: {e}")

def demo_utility_functions():
    """Demo utility functions for TLS security."""
    print("\n" + "="*60)
    print("🛠️ TLS UTILITY FUNCTIONS DEMO")
    print("="*60)
    
    # Test secure SSL context creation
    print("🔧 Testing secure SSL context creation:")
    try:
        context_tls12 = create_secure_ssl_context("TLSv1.2")
        print("   ✅ TLS 1.2 context created")
        
        context_tls13 = create_secure_ssl_context("TLSv1.3")
        print("   ✅ TLS 1.3 context created")
        
    except Exception as e:
        print(f"   ❌ Error creating contexts: {e}")
    
    # Test cipher validation
    print(f"\n🔍 Testing cipher validation:")
    test_ciphers = [
        "ECDHE-RSA-AES256-GCM-SHA384",  # Strong
        "TLS_AES_256_GCM_SHA384",       # TLS 1.3
        "NULL-SHA",                      # Weak
        "DES-CBC-SHA",                   # Weak
        "RC4-SHA"                        # Weak
    ]
    
    for cipher in test_ciphers:
        is_valid = validate_cipher_suite(cipher)
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"   • {cipher:<35} {status}")

def demo_security_comparison():
    """Demo security comparison between secure and insecure defaults."""
    print("\n" + "="*60)
    print("⚖️ SECURITY COMPARISON DEMO")
    print("="*60)
    
    print("🔒 SECURE DEFAULTS (Recommended):")
    print("   ✅ TLS 1.2+ only")
    print("   ✅ Strong cipher suites (AES-GCM, ChaCha20)")
    print("   ✅ Certificate verification required")
    print("   ✅ Hostname verification enabled")
    print("   ✅ Weak ciphers disabled")
    print("   ✅ Compression disabled")
    print("   ✅ Renegotiation disabled")
    
    print(f"\n❌ INSECURE DEFAULTS (Avoid):")
    print("   ❌ TLS 1.0/1.1 allowed")
    print("   ❌ Weak ciphers (RC4, DES, NULL)")
    print("   ❌ No certificate verification")
    print("   ❌ No hostname verification")
    print("   ❌ Compression enabled")
    print("   ❌ Renegotiation allowed")
    
    print(f"\n📊 Security Benefits:")
    print("   🛡️  Protection against downgrade attacks")
    print("   🛡️  Protection against cipher suite attacks")
    print("   🛡️  Protection against man-in-the-middle attacks")
    print("   🛡️  Protection against compression attacks")
    print("   🛡️  Protection against renegotiation attacks")

def demo_library_availability():
    """Check and display TLS library availability."""
    print("\n" + "="*60)
    print("📚 TLS LIBRARY AVAILABILITY CHECK")
    print("="*60)
    
    # Check SSL library
    try:
        print("✅ ssl: Available")
        print(f"   OpenSSL version: {ssl.OPENSSL_VERSION}")
        print(f"   SSL version: {ssl.SSLContext().minimum_version}")
    except ImportError:
        print("❌ ssl: Not available")
    
    # Check socket library
    try:
        print("✅ socket: Available")
    except ImportError:
        print("❌ socket: Not available")
    
    # Check cryptography library
    try:
        print("✅ cryptography: Available")
        print(f"   Version: {cryptography.__version__}")
    except ImportError:
        print("❌ cryptography: Not available")
        print("   Install with: pip install cryptography")

def main():
    """Main demo function."""
    print("🔐 SECURE TLS DEFAULTS DEMO")
    print("="*60)
    print("This demo showcases secure TLS configuration, strong cipher suites,")
    print("and secure defaults for network communications.")
    
    # Check library availability
    demo_library_availability()
    
    # Run demos
    demo_tls_configuration()
    demo_ssl_context_creation()
    demo_cipher_validation()
    demo_secure_config()
    demo_network_security()
    demo_utility_functions()
    demo_security_comparison()
    
    print("\n" + "="*60)
    print("✅ SECURE TLS DEFAULTS DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key security features demonstrated:")
    print("• TLS 1.2+ enforcement with secure defaults")
    print("• Strong cipher suite validation and filtering")
    print("• Secure SSL context creation")
    print("• Certificate and hostname verification")
    print("• Protection against common TLS attacks")
    print("• Network security with proper validation")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 