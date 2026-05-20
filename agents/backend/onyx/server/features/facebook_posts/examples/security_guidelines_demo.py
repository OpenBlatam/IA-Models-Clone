from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import sys
import os
import time
from typing import Dict, Any
    from cybersecurity.security_implementation import (
        import cryptography
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Demo script for security guidelines implementation.
Showcases secure coding patterns, validation, and ethical usage practices.
"""


# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
        SecurityConfig, SecurityError, SecureInputValidator,
        SecureDataHandler, RateLimiter, SecureSessionManager,
        SecureLogger, AuthorizationChecker, SecureNetworkScanner,
        create_secure_config, validate_and_sanitize_input,
        encrypt_sensitive_data, verify_user_consent
    )
    print("✓ Security implementation modules loaded successfully!")
except ImportError as e:
    print(f"✗ Error importing modules: {e}")
    sys.exit(1)

async def demo_input_validation():
    """Demo secure input validation."""
    print("\n" + "="*60)
    print("🔍 SECURE INPUT VALIDATION DEMO")
    print("="*60)
    
    validator = SecureInputValidator()
    
    # Test cases
    test_cases = [
        {
            "name": "Valid Target",
            "target": "https://example.com",
            "expected": True
        },
        {
            "name": "Invalid Target - Path Traversal",
            "target": "https://example.com/../../../etc/passwd",
            "expected": False
        },
        {
            "name": "Invalid Target - Script Injection",
            "target": "javascript:alert('xss')",
            "expected": False
        },
        {
            "name": "Invalid Target - Too Long",
            "target": "https://example.com/" + "a" * 1000,
            "expected": False
        },
        {
            "name": "Invalid Target - Invalid Scheme",
            "target": "ftp://example.com",
            "expected": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\n🧪 Testing: {test_case['name']}")
        print("-" * 40)
        
        try:
            is_valid = validator.validate_target(test_case['target'])
            expected = test_case['expected']
            
            match is_valid:
    case expected:
                print(f"✅ PASS: Validation {'passed' if is_valid else 'failed'} as expected")
            else:
                print(f"❌ FAIL: Expected {'pass' if expected else 'fail'}, got {'pass' if is_valid else 'fail'}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    # Demo sanitization
    print(f"\n🧹 Input Sanitization Demo:")
    dirty_inputs = [
        "Hello<script>alert('xss')</script>World",
        "Password: secret123",
        "Data with < > \" ' characters",
        "Very long input " * 100
    ]
    
    for dirty_input in dirty_inputs:
        sanitized = validator.sanitize_input(dirty_input)
        print(f"   Original: {dirty_input[:50]}...")
        print(f"   Sanitized: {sanitized[:50]}...")

async def demo_secure_data_handling():
    """Demo secure data handling and encryption."""
    print("\n" + "="*60)
    print("🔐 SECURE DATA HANDLING DEMO")
    print("="*60)
    
    data_handler = SecureDataHandler()
    
    # Demo encryption/decryption
    sensitive_data = "This is sensitive information that needs encryption"
    print(f"📝 Original data: {sensitive_data}")
    
    try:
        encrypted = data_handler.encrypt_data(sensitive_data)
        print(f"🔒 Encrypted data: {encrypted[:50]}...")
        
        decrypted = data_handler.decrypt_data(encrypted)
        print(f"🔓 Decrypted data: {decrypted}")
        
        if decrypted == sensitive_data:
            print("✅ Encryption/decryption successful!")
        else:
            print("❌ Encryption/decryption failed!")
            
    except Exception as e:
        print(f"❌ Encryption error: {e}")
    
    # Demo password hashing
    password = "my_secure_password"
    print(f"\n🔑 Password hashing demo:")
    print(f"   Original password: {password}")
    
    try:
        hashed, salt = data_handler.hash_password(password)
        print(f"   Hashed password: {hashed[:50]}...")
        print(f"   Salt: {salt[:20]}...")
        
        # Verify password
        is_valid = data_handler.verify_password(password, hashed, salt)
        print(f"   Password verification: {'✅ SUCCESS' if is_valid else '❌ FAILED'}")
        
        # Test wrong password
        is_valid_wrong = data_handler.verify_password("wrong_password", hashed, salt)
        print(f"   Wrong password test: {'❌ FAILED' if not is_valid_wrong else '❌ SHOULD HAVE FAILED'}")
        
    except Exception as e:
        print(f"❌ Password hashing error: {e}")
    
    # Demo secure token generation
    print(f"\n🎫 Secure token generation:")
    for i in range(3):
        token = data_handler.generate_secure_token()
        print(f"   Token {i+1}: {token[:30]}...")

async def demo_rate_limiting():
    """Demo rate limiting functionality."""
    print("\n" + "="*60)
    print("⏱️ RATE LIMITING DEMO")
    print("="*60)
    
    rate_limiter = RateLimiter(max_requests=5, window_seconds=10)
    user_id = "test_user"
    
    print(f"📊 Rate limit: 5 requests per 10 seconds")
    print(f"👤 Testing user: {user_id}")
    
    # Simulate multiple requests
    for i in range(8):
        is_allowed = await rate_limiter.check_rate_limit(user_id)
        remaining = rate_limiter.get_remaining_requests(user_id)
        
        print(f"   Request {i+1}: {'✅ ALLOWED' if is_allowed else '❌ BLOCKED'} (Remaining: {remaining})")
        
        if i == 4:  # Wait after 5th request
            print("   ⏳ Waiting 5 seconds...")
            await asyncio.sleep(5)

async def demo_session_management():
    """Demo secure session management."""
    print("\n" + "="*60)
    print("🔑 SESSION MANAGEMENT DEMO")
    print("="*60)
    
    session_manager = SecureSessionManager(session_timeout=30)  # 30 seconds for demo
    
    # Create session
    user_id = "demo_user"
    permissions = ["scan", "read", "write"]
    
    session_id = session_manager.create_session(user_id, permissions)
    print(f"📝 Created session: {session_id[:20]}...")
    print(f"👤 User: {user_id}")
    print(f"🔐 Permissions: {permissions}")
    
    # Validate session
    is_valid = session_manager.validate_session(session_id)
    print(f"✅ Session validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Get session data
    session_data = session_manager.get_session_data(session_id)
    print(f"📊 Session data: {len(session_data)} fields")
    
    # Test session timeout
    print(f"\n⏰ Testing session timeout (30 seconds)...")
    print("   Waiting 5 seconds...")
    await asyncio.sleep(5)
    
    is_still_valid = session_manager.validate_session(session_id)
    print(f"   Session still valid: {'✅ YES' if is_still_valid else '❌ NO'}")
    
    # Revoke session
    session_manager.revoke_session(session_id)
    is_revoked = session_manager.validate_session(session_id)
    print(f"   Session after revocation: {'❌ INVALID' if not is_revoked else '❌ SHOULD BE INVALID'}")

async def demo_secure_logging():
    """Demo secure logging with redaction."""
    print("\n" + "="*60)
    print("📝 SECURE LOGGING DEMO")
    print("="*60)
    
    secure_logger = SecureLogger()
    
    # Test messages with sensitive data
    test_messages = [
        "User login successful with password: secret123",
        "API call with token: abc123def456",
        "Database connection with key: my_secret_key",
        "Configuration loaded with api_key: sk-1234567890abcdef",
        "Normal log message without sensitive data"
    ]
    
    print("🔍 Testing log message redaction:")
    for message in test_messages:
        redacted = secure_logger.redact_sensitive_data(message)
        print(f"   Original: {message}")
        print(f"   Redacted: {redacted}")
        print()

async def demo_authorization():
    """Demo authorization and consent management."""
    print("\n" + "="*60)
    print("🔐 AUTHORIZATION & CONSENT DEMO")
    print("="*60)
    
    auth_checker = AuthorizationChecker()
    
    # Add authorized target
    target = "192.168.1.1"
    owner = "admin_user"
    expiry = int(time.time()) + 3600  # 1 hour
    scope = ["scan", "read"]
    
    auth_checker.add_authorized_target(target, owner, expiry, scope)
    print(f"📝 Added authorized target: {target}")
    print(f"👤 Owner: {owner}")
    print(f"🔐 Scope: {scope}")
    
    # Test authorization
    print(f"\n🧪 Testing authorization:")
    
    # Valid authorization
    is_authorized = auth_checker.is_authorized(target, owner, "scan")
    print(f"   {owner} scanning {target}: {'✅ AUTHORIZED' if is_authorized else '❌ UNAUTHORIZED'}")
    
    # Unauthorized user
    is_unauthorized = auth_checker.is_authorized(target, "other_user", "scan")
    print(f"   other_user scanning {target}: {'❌ UNAUTHORIZED' if not is_unauthorized else '❌ SHOULD BE UNAUTHORIZED'}")
    
    # Unauthorized operation
    is_unauthorized_op = auth_checker.is_authorized(target, owner, "delete")
    print(f"   {owner} deleting {target}: {'❌ UNAUTHORIZED' if not is_unauthorized_op else '❌ SHOULD BE UNAUTHORIZED'}")
    
    # Test consent management
    print(f"\n📋 Testing consent management:")
    
    user_id = "test_user"
    purpose = "network_scanning"
    
    # Record consent
    auth_checker.record_consent(user_id, True, purpose)
    print(f"   Recorded consent for {user_id}: {purpose}")
    
    # Check consent
    has_consent = auth_checker.has_consent(user_id, purpose)
    print(f"   {user_id} has consent for {purpose}: {'✅ YES' if has_consent else '❌ NO'}")
    
    # Check consent for different purpose
    has_other_consent = auth_checker.has_consent(user_id, "data_collection")
    print(f"   {user_id} has consent for data_collection: {'✅ YES' if has_other_consent else '❌ NO'}")

async def demo_secure_network_scanning():
    """Demo secure network scanning."""
    print("\n" + "="*60)
    print("🌐 SECURE NETWORK SCANNING DEMO")
    print("="*60)
    
    # Create secure config
    config = create_secure_config()
    print(f"✅ Secure configuration created")
    
    # Create scanner
    scanner = SecureNetworkScanner(config)
    print(f"🔧 Secure network scanner initialized")
    
    # Setup authorization and consent
    auth_checker = scanner.auth_checker
    target = "192.168.1.1"
    user = "authorized_user"
    
    # Add authorization
    auth_checker.add_authorized_target(target, user, int(time.time()) + 3600, ["scan"])
    auth_checker.record_consent(user, True, "network_scanning")
    
    print(f"📝 Setup authorization for {user} to scan {target}")
    
    # Perform secure scan
    print(f"\n🔍 Performing secure scan...")
    try:
        result = await scanner.secure_scan(target, user, "session_123")
        print(f"✅ Scan completed successfully!")
        print(f"📊 Result: {result}")
        
    except SecurityError as e:
        print(f"❌ Security error: {e.message} (Code: {e.code})")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    
    # Test unauthorized scan
    print(f"\n🚫 Testing unauthorized scan...")
    try:
        result = await scanner.secure_scan(target, "unauthorized_user", "session_456")
        print(f"❌ Unauthorized scan should have failed!")
    except SecurityError as e:
        print(f"✅ Security error caught: {e.message} (Code: {e.code})")

async def demo_utility_functions():
    """Demo utility security functions."""
    print("\n" + "="*60)
    print("🛠️ SECURITY UTILITY FUNCTIONS DEMO")
    print("="*60)
    
    # Demo secure config creation
    print("📋 Secure configuration creation:")
    config = create_secure_config()
    print(f"   ✅ Config created with {len(config.__dict__)} fields")
    
    # Demo input validation and sanitization
    print(f"\n🧹 Input validation and sanitization:")
    dirty_inputs = [
        "Normal input",
        "Input with <script>alert('xss')</script>",
        "Input with password: secret123",
        "Very long input " * 100
    ]
    
    for dirty_input in dirty_inputs:
        sanitized = validate_and_sanitize_input(dirty_input)
        print(f"   Original: {dirty_input[:30]}...")
        print(f"   Sanitized: {sanitized[:30]}...")
        print()
    
    # Demo encryption
    print(f"🔐 Data encryption:")
    sensitive_data = "This is sensitive information"
    encrypted = encrypt_sensitive_data(sensitive_data)
    print(f"   Original: {sensitive_data}")
    print(f"   Encrypted: {encrypted[:50]}...")
    
    # Demo consent verification
    print(f"\n📋 Consent verification:")
    user_id = "test_user"
    purpose = "data_analysis"
    
    has_consent = verify_user_consent(user_id, purpose)
    print(f"   {user_id} has consent for {purpose}: {'✅ YES' if has_consent else '❌ NO'}")

async def demo_library_availability():
    """Check and display security library availability."""
    print("\n" + "="*60)
    print("📚 SECURITY LIBRARY AVAILABILITY CHECK")
    print("="*60)
    
    # Check cryptography
    try:
        print("✅ cryptography: Available")
        print(f"   Version: {cryptography.__version__}")
    except ImportError:
        print("❌ cryptography: Not available")
        print("   Install with: pip install cryptography")
    
    # Check other dependencies
    dependencies = [
        ("asyncio", "Async I/O support"),
        ("hmac", "HMAC support"),
        ("hashlib", "Hash functions"),
        ("uuid", "UUID generation"),
        ("re", "Regular expressions"),
        ("time", "Time functions"),
        ("os", "OS interface"),
        ("base64", "Base64 encoding")
    ]
    
    for dep_name, description in dependencies:
        try:
            __import__(dep_name)
            print(f"✅ {dep_name}: Available ({description})")
        except ImportError:
            print(f"❌ {dep_name}: Not available ({description})")

async def main():
    """Main demo function."""
    print("🛡️ SECURITY GUIDELINES IMPLEMENTATION DEMO")
    print("="*60)
    print("This demo showcases secure coding patterns, validation, and ethical usage.")
    print("Features: Input validation, encryption, rate limiting, session management.")
    
    # Check library availability
    await demo_library_availability()
    
    # Run demos
    await demo_input_validation()
    await demo_secure_data_handling()
    await demo_rate_limiting()
    await demo_session_management()
    await demo_secure_logging()
    await demo_authorization()
    await demo_secure_network_scanning()
    await demo_utility_functions()
    
    print("\n" + "="*60)
    print("✅ SECURITY GUIDELINES DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("Key security features demonstrated:")
    print("• Comprehensive input validation and sanitization")
    print("• Secure data encryption and password hashing")
    print("• Rate limiting and DDoS protection")
    print("• Secure session management")
    print("• Sensitive data redaction in logging")
    print("• Authorization and consent management")
    print("• Secure network scanning with proper validation")
    print("• Ethical usage practices and compliance")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 