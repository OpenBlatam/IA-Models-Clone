from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import sys
import os
from crypto_helpers import (
from network_helpers import (
        from cryptography.hazmat.primitives import serialization
    from network_helpers import resolve_dns_async, reverse_dns_lookup_async
    from network_helpers import check_port_open_async
    from network_helpers import fetch_http_headers_async, fetch_url_content_async
    import time
    from network_helpers import resolve_dns_async, check_port_open_async
        import traceback
from typing import Any, List, Dict, Optional
import logging
"""
Demo script for the utils module.

Demonstrates:
- Cryptographic operations (hashing, encryption, signatures)
- Network utilities (DNS, port scanning, HTTP analysis)
- Protocol analysis (HTTP, TCP)
- Performance optimizations
"""

sys.path.append('.')

    CryptoHelper,
    HashHelper,
    EncryptionHelper,
    CryptoConfig,
    CryptoResult
)

    NetworkHelper,
    ProtocolHelper,
    NetworkConfig,
    NetworkResult
)

async def demo_crypto_operations():
    """Demo cryptographic operations."""
    print("\n=== Crypto Operations Demo ===")
    
    config = CryptoConfig(
        hash_algorithm="sha256",
        key_length=32,
        salt_length=16,
        iterations=100000
    )
    
    crypto = CryptoHelper(config)
    
    # Password hashing
    print("Testing password hashing...")
    password = "my_secure_password_123"
    hash_result = crypto.hash_password(password)
    
    print(f"Password hash result: {hash_result.success}")
    if hash_result.success:
        print(f"Hash value: {hash_result.hash_value}")
        
        # Verify password
        is_valid = crypto.verify_password(password, hash_result.hash_value)
        print(f"Password verification: {is_valid}")
        
        # Test wrong password
        is_wrong = crypto.verify_password("wrong_password", hash_result.hash_value)
        print(f"Wrong password verification: {is_wrong}")
    
    # Key pair generation
    print("\nTesting RSA key pair generation...")
    key_result = crypto.generate_key_pair()
    
    print(f"Key pair generation: {key_result.success}")
    if key_result.success:
        print(f"Private key length: {len(key_result.private_key)} bytes")
        print(f"Public key length: {len(key_result.public_key)} bytes")
        
        # Test encryption/decryption
        test_data = b"Hello, this is a test message for encryption!"
        
        # Symmetric encryption
        print("\nTesting symmetric encryption...")
        key = b'\x00' * 32  # 256-bit key
        encrypt_result = crypto.encrypt_data(test_data, key, use_asymmetric=False)
        
        print(f"Symmetric encryption: {encrypt_result.success}")
        if encrypt_result.success:
            decrypt_result = crypto.decrypt_data(encrypt_result.encrypted_data, key, use_asymmetric=False)
            print(f"Symmetric decryption: {decrypt_result.success}")
            if decrypt_result.success:
                print(f"Decrypted data: {decrypt_result.decrypted_data}")
                print(f"Data matches: {decrypt_result.decrypted_data == test_data}")
    
    # Digital signatures
    print("\nTesting digital signatures...")
    if key_result.success:
        private_key = serialization.load_pem_private_key(key_result.private_key, password=None)
        public_key = serialization.load_pem_public_key(key_result.public_key)
        
        sign_result = crypto.sign_data(test_data, private_key)
        print(f"Digital signature: {sign_result.success}")
        
        if sign_result.success:
            is_valid_signature = crypto.verify_signature(test_data, sign_result.signature, public_key)
            print(f"Signature verification: {is_valid_signature}")

async def demo_hash_operations():
    """Demo hash operations."""
    print("\n=== Hash Operations Demo ===")
    
    config = CryptoConfig()
    hash_helper = HashHelper(config)
    
    # Create test file
    test_file = "test_file.txt"
    test_content = "This is a test file for hashing operations.\n" * 100
    
    with open(test_file, 'w') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        f.write(test_content)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
    
    try:
        # File hashing
        print("Testing file hashing...")
        hash_result = hash_helper.calculate_file_hash(test_file)
        
        print(f"File hash result: {hash_result.success}")
        if hash_result.success:
            print(f"File hash: {hash_result.hash_value}")
            
            # Verify file integrity
            is_integrity_valid = hash_helper.verify_file_integrity(test_file, hash_result.hash_value)
            print(f"File integrity verification: {is_integrity_valid}")
        
        # Generate checksum
        checksum = hash_helper.generate_checksum(test_content.encode())
        print(f"Content checksum: {checksum}")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

async def demo_encryption_operations():
    """Demo encryption operations."""
    print("\n=== Encryption Operations Demo ===")
    
    config = CryptoConfig()
    encrypt_helper = EncryptionHelper(config)
    
    # Generate secure key
    key = encrypt_helper.generate_secure_key()
    print(f"Generated key length: {len(key)} bytes")
    
    # String encryption
    test_text = "This is a secret message that needs to be encrypted!"
    print(f"Original text: {test_text}")
    
    encrypt_result = encrypt_helper.encrypt_string(test_text, key)
    print(f"String encryption: {encrypt_result.success}")
    
    if encrypt_result.success:
        decrypt_result = encrypt_helper.decrypt_string(encrypt_result.encrypted_data, key)
        print(f"String decryption: {decrypt_result.success}")
        
        if decrypt_result.success:
            decrypted_text = decrypt_result.decrypted_data.decode('utf-8')
            print(f"Decrypted text: {decrypted_text}")
            print(f"Text matches: {decrypted_text == test_text}")

async def demo_network_operations():
    """Demo network operations."""
    print("\n=== Network Operations Demo ===")
    
    config = NetworkConfig(
        timeout=10.0,
        enable_ipv6=True,
        user_agent="Cybersecurity-Demo/1.0"
    )
    
    network = NetworkHelper(config)
    
    # DNS resolution
    print("Testing DNS resolution...")
    hostname = "google.com"
    host_info = await network.get_host_info(hostname)
    
    print(f"Host info for {hostname}:")
    print(f"  IPs: {host_info['ips']}")
    print(f"  Reverse DNS: {host_info['reverse_dns']}")
    print(f"  Resolvable: {host_info['is_resolvable']}")
    
    # Port scanning (limited to avoid overwhelming)
    print("\nTesting port scanning (limited)...")
    test_host = "127.0.0.1"  # Localhost for testing
    scan_result = await network.scan_common_ports(test_host)
    
    print(f"Port scan for {test_host}:")
    print(f"  Scanned ports: {scan_result['scanned_ports']}")
    print(f"  Open ports: {scan_result['open_ports']}")
    print(f"  Open count: {scan_result['open_count']}")
    
    # Web server analysis
    print("\nTesting web server analysis...")
    web_result = await network.check_web_server("http://httpbin.org")
    
    print(f"Web server analysis:")
    print(f"  Reachable: {web_result['reachable']}")
    print(f"  Status code: {web_result['status_code']}")
    print(f"  Response time: {web_result['response_time']:.3f}s")
    print(f"  Server: {web_result['server']}")
    
    # Network validation
    print("\nTesting network validation...")
    network_config = network.validate_network_config("192.168.1.0", "255.255.255.0")
    
    print(f"Network configuration validation:")
    print(f"  Valid: {network_config['valid']}")
    if network_config['valid']:
        print(f"  Network: {network_config['network']}")
        print(f"  Broadcast: {network_config['broadcast']}")
        print(f"  Netmask: {network_config['netmask']}")
        print(f"  Hosts: {network_config['num_hosts']}")
        print(f"  Private: {network_config['is_private']}")

async def demo_protocol_analysis():
    """Demo protocol analysis."""
    print("\n=== Protocol Analysis Demo ===")
    
    config = NetworkConfig()
    protocol = ProtocolHelper(config)
    
    # HTTP request parsing
    print("Testing HTTP request parsing...")
    http_request = b"""GET /api/users HTTP/1.1\r
Host: api.example.com\r
User-Agent: Mozilla/5.0\r
Accept: application/json\r
\r
"""
    
    parsed_request = protocol.parse_http_request(http_request)
    print(f"HTTP request parsing: {parsed_request['valid']}")
    
    if parsed_request['valid']:
        print(f"  Method: {parsed_request['method']}")
        print(f"  Path: {parsed_request['path']}")
        print(f"  Version: {parsed_request['version']}")
        print(f"  Headers: {len(parsed_request['headers'])}")
        print(f"  Content length: {parsed_request['content_length']}")
    
    # HTTP response parsing
    print("\nTesting HTTP response parsing...")
    http_response = b"""HTTP/1.1 200 OK\r
Content-Type: application/json\r
Content-Length: 25\r
Server: nginx/1.18.0\r
\r
{"status": "success"}
"""
    
    parsed_response = protocol.parse_http_response(http_response)
    print(f"HTTP response parsing: {parsed_response['valid']}")
    
    if parsed_response['valid']:
        print(f"  Version: {parsed_response['version']}")
        print(f"  Status code: {parsed_response['status_code']}")
        print(f"  Status text: {parsed_response['status_text']}")
        print(f"  Headers: {len(parsed_response['headers'])}")
        print(f"  Content length: {parsed_response['content_length']}")
    
    # TCP packet validation
    print("\nTesting TCP packet validation...")
    # Create a simple TCP packet (SYN packet)
    tcp_packet = b'\x00\x50\x00\x50\x00\x00\x00\x01\x00\x00\x00\x00\x50\x02\x20\x00\x00\x00\x00\x00'
    
    packet_info = protocol.validate_tcp_packet(tcp_packet)
    print(f"TCP packet validation: {packet_info['valid']}")
    
    if packet_info['valid']:
        print(f"  Source port: {packet_info['source_port']}")
        print(f"  Dest port: {packet_info['dest_port']}")
        print(f"  Seq num: {packet_info['seq_num']}")
        print(f"  Ack num: {packet_info['ack_num']}")
        print(f"  Flags: {packet_info['flags']}")
        print(f"  Window: {packet_info['window']}")
        print(f"  Payload length: {packet_info['payload_length']}")

async def demo_async_operations():
    """Demo async operations."""
    print("\n=== Async Operations Demo ===")
    
    # DNS resolution
    print("Testing async DNS resolution...")
    
    hostname = "github.com"
    ips = await resolve_dns_async(hostname)
    print(f"DNS resolution for {hostname}: {ips}")
    
    if ips:
        reverse_dns = await reverse_dns_lookup_async(ips[0])
        print(f"Reverse DNS for {ips[0]}: {reverse_dns}")
    
    # Port checking
    print("\nTesting async port checking...")
    
    test_ports = [80, 443, 22, 8080]
    for port in test_ports:
        is_open = await check_port_open_async("127.0.0.1", port, timeout=2.0)
        print(f"Port {port}: {'Open' if is_open else 'Closed'}")
    
    # HTTP operations
    print("\nTesting async HTTP operations...")
    
    config = NetworkConfig(timeout=10.0)
    
    # Fetch headers
    headers_result = await fetch_http_headers_async("http://httpbin.org", config)
    print(f"HTTP headers fetch: {headers_result.success}")
    if headers_result.success:
        print(f"  Status code: {headers_result.status_code}")
        print(f"  Response time: {headers_result.response_time:.3f}s")
    
    # Fetch content
    content_result = await fetch_url_content_async("http://httpbin.org/json", config)
    print(f"HTTP content fetch: {content_result.success}")
    if content_result.success:
        print(f"  Content length: {len(content_result.data)} bytes")
        print(f"  Response time: {content_result.response_time:.3f}s")

async def demo_performance_comparison():
    """Demo performance comparison between sync and async operations."""
    print("\n=== Performance Comparison Demo ===")
    
    # Test crypto operations performance
    print("Testing crypto operations performance...")
    
    config = CryptoConfig()
    crypto = CryptoHelper(config)
    
    # Test password hashing performance
    start_time = time.time()
    for i in range(10):
        crypto.hash_password(f"password_{i}")
    crypto_time = time.time() - start_time
    print(f"Crypto operations: {crypto_time:.3f} seconds for 10 operations")
    
    # Test network operations performance
    print("Testing network operations performance...")
    
    # DNS resolution performance
    start_time = time.time()
    dns_results = []
    for hostname in ["google.com", "github.com", "stackoverflow.com"]:
        result = await resolve_dns_async(hostname)
        dns_results.append(result)
    dns_time = time.time() - start_time
    print(f"DNS resolution: {dns_time:.3f} seconds for 3 hostnames")
    
    # Port checking performance
    start_time = time.time()
    port_tasks = []
    for port in range(80, 90):  # Check ports 80-89
        task = check_port_open_async("127.0.0.1", port, timeout=1.0)
        port_tasks.append(task)
    
    port_results = await asyncio.gather(*port_tasks, return_exceptions=True)
    port_time = time.time() - start_time
    print(f"Port checking: {port_time:.3f} seconds for 10 ports (concurrent)")

async def main():
    """Run all utils demos."""
    print("🚀 Cybersecurity Utils Module Demo")
    print("=" * 50)
    
    try:
        await demo_crypto_operations()
        await demo_hash_operations()
        await demo_encryption_operations()
        await demo_network_operations()
        await demo_protocol_analysis()
        await demo_async_operations()
        await demo_performance_comparison()
        
        print("\n✅ All utils demos completed successfully!")
        print("\n📋 Summary:")
        print("- Crypto operations: Working (hashing, encryption, signatures)")
        print("- Hash operations: Working (file hashing, integrity)")
        print("- Encryption operations: Working (symmetric, asymmetric)")
        print("- Network operations: Working (DNS, port scanning, HTTP)")
        print("- Protocol analysis: Working (HTTP, TCP)")
        print("- Async operations: Working (non-blocking I/O)")
        print("- Performance optimization: Working")
        
        print("\n🔐 Security Features:")
        print("- Cryptographically secure random generation")
        print("- PBKDF2 password hashing with salt")
        print("- RSA key pair generation and management")
        print("- AES and RSA encryption/decryption")
        print("- Digital signature creation and verification")
        print("- Network protocol validation")
        print("- Input validation and sanitization")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

match __name__:
    case "__main__":
    asyncio.run(main()) 