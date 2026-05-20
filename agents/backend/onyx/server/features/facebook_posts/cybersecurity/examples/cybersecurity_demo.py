from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from typing import Dict, Any
from dataclasses import dataclass
from ..scanners import (
from ..crypto import (
from ..network import (
from ..validators import (
from ..monitors import (
from typing import Any, List, Dict, Optional
import logging
"""
Comprehensive demo of cybersecurity tools with proper async/def distinction.
Demonstrates CPU-bound operations with def and I/O operations with async.
"""


# Import our cybersecurity modules
    scan_single_port, scan_port_range, enrich_scan_results, 
    format_scan_report, ScanResult, ScanConfig
)
    hash_password, verify_password, generate_secure_key,
    encrypt_data, decrypt_data, SecurityConfig
)
    check_connection, validate_url, test_ssl_certificate,
    monitor_bandwidth, NetworkConfig
)
    validate_input, sanitize_data, check_file_integrity,
    validate_credentials, ValidationConfig
)
    monitor_system_resources, detect_anomalies,
    log_security_events, track_user_activity, MonitoringConfig
)

@dataclass
class DemoConfig:
    """Configuration for cybersecurity demo."""
    target_host: str = "localhost"
    target_url: str = "https://httpbin.org"
    test_password: str = "SecurePassword123!"
    test_username: str = "testuser"
    scan_ports: list = None
    
    def __post_init__(self) -> Any:
        if self.scan_ports is None:
            self.scan_ports = [80, 443, 22, 21, 25, 53]

class CybersecurityDemo:
    """Comprehensive cybersecurity tools demonstration."""
    
    def __init__(self, config: DemoConfig):
        
    """__init__ function."""
self.config = config
        self.results = {}
    
    async def run_comprehensive_demo(self) -> Dict[str, Any]:
        """Run all cybersecurity demos."""
        print("🔒 Starting Comprehensive Cybersecurity Demo")
        print("=" * 50)
        
        # Run all demos
        await self.demo_port_scanning()
        await self.demo_cryptographic_operations()
        await self.demo_network_security()
        await self.demo_input_validation()
        await self.demo_system_monitoring()
        
        return self.results
    
    async def demo_port_scanning(self) -> Any:
        """Demonstrate port scanning capabilities."""
        print("\n🔍 Port Scanning Demo")
        print("-" * 30)
        
        scan_config = ScanConfig(timeout=2.0, max_workers=10)
        
        # Scan single port (async - network operation)
        print(f"Scanning single port {self.config.scan_ports[0]} on {self.config.target_host}...")
        single_result = await scan_single_port(
            self.config.target_host, 
            self.config.scan_ports[0], 
            scan_config
        )
        print(f"Result: {single_result}")
        
        # Scan port range (async - network operation)
        print(f"\nScanning ports {self.config.scan_ports} on {self.config.target_host}...")
        range_results = await scan_port_range(
            self.config.target_host,
            min(self.config.scan_ports),
            max(self.config.scan_ports),
            scan_config
        )
        
        # Enrich results (def - CPU-bound operation)
        enriched_results = enrich_scan_results(range_results)
        
        # Format report (def - CPU-bound operation)
        report = format_scan_report(enriched_results)
        print(f"\nScan Report:\n{report}")
        
        self.results['port_scanning'] = {
            'single_result': single_result,
            'range_results': enriched_results,
            'report': report
        }
    
    async def demo_cryptographic_operations(self) -> Any:
        """Demonstrate cryptographic operations."""
        print("\n🔐 Cryptographic Operations Demo")
        print("-" * 35)
        
        security_config = SecurityConfig(
            key_length=32,
            iterations=100000,
            hash_algorithm="sha256"
        )
        
        # Hash password (def - CPU-bound operation)
        print(f"Hashing password: {self.config.test_password}")
        hashed_password = hash_password(self.config.test_password, security_config)
        print(f"Hashed: {hashed_password[:50]}...")
        
        # Verify password (def - CPU-bound operation)
        is_valid = verify_password(self.config.test_password, hashed_password, security_config)
        print(f"Password verification: {'✅ Valid' if is_valid else '❌ Invalid'}")
        
        # Generate secure key (def - CPU-bound operation)
        secure_key = generate_secure_key(security_config)
        print(f"Generated secure key: {secure_key.hex()[:32]}...")
        
        # Encrypt/decrypt data (def - CPU-bound operation)
        test_data = b"Sensitive data that needs encryption"
        encrypted_data = encrypt_data(test_data, secure_key)
        decrypted_data = decrypt_data(encrypted_data, secure_key)
        
        print(f"Original data: {test_data}")
        print(f"Encrypted: {encrypted_data[:50]}...")
        print(f"Decrypted: {decrypted_data}")
        print(f"Encryption successful: {'✅ Yes' if test_data == decrypted_data else '❌ No'}")
        
        self.results['crypto'] = {
            'hashed_password': hashed_password,
            'password_valid': is_valid,
            'secure_key': secure_key.hex(),
            'encryption_successful': test_data == decrypted_data
        }
    
    async def demo_network_security(self) -> Any:
        """Demonstrate network security tools."""
        print("\n🌐 Network Security Demo")
        print("-" * 25)
        
        network_config = NetworkConfig(timeout=5.0, verify_ssl=True)
        
        # Check connection (async - network operation)
        print(f"Checking connection to {self.config.target_host}:80...")
        connection_result = await check_connection(
            self.config.target_host, 80, network_config
        )
        print(f"Connection result: {connection_result}")
        
        # Validate URL (async - network operation)
        print(f"\nValidating URL: {self.config.target_url}")
        url_validation = await validate_url(self.config.target_url, network_config)
        print(f"URL validation: {url_validation}")
        
        # Test SSL certificate (async - network operation)
        print(f"\nTesting SSL certificate for {self.config.target_url}")
        ssl_test = await test_ssl_certificate(self.config.target_url)
        print(f"SSL test: {ssl_test}")
        
        # Monitor bandwidth (async - network operation)
        print(f"\nMonitoring bandwidth to {self.config.target_host}:80...")
        bandwidth_result = await monitor_bandwidth(
            self.config.target_host, 80, duration=3
        )
        print(f"Bandwidth result: {bandwidth_result}")
        
        self.results['network_security'] = {
            'connection_result': connection_result,
            'url_validation': url_validation,
            'ssl_test': ssl_test,
            'bandwidth_result': bandwidth_result
        }
    
    async def demo_input_validation(self) -> Any:
        """Demonstrate input validation."""
        print("\n✅ Input Validation Demo")
        print("-" * 25)
        
        validation_config = ValidationConfig(
            max_length=100,
            min_length=1,
            strict_mode=True,
            block_sql_injection=True,
            block_xss=True
        )
        
        # Test various inputs
        test_inputs = [
            "normal_input",
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "a" * 200,  # Too long
            ""  # Too short
        ]
        
        validation_results = {}
        
        for test_input in test_inputs:
            print(f"\nTesting input: {test_input[:50]}...")
            
            # Validate input (def - CPU-bound operation)
            validation_result = validate_input(test_input, validation_config)
            print(f"Validation: {validation_result}")
            
            # Sanitize input (def - CPU-bound operation)
            sanitized = sanitize_data(test_input, validation_config)
            print(f"Sanitized: {sanitized[:50]}...")
            
            validation_results[test_input] = {
                'validation': validation_result,
                'sanitized': sanitized
            }
        
        # Validate credentials (def - CPU-bound operation)
        print(f"\nValidating credentials for user: {self.config.test_username}")
        credentials_result = validate_credentials(
            self.config.test_username,
            self.config.test_password,
            validation_config
        )
        print(f"Credentials validation: {credentials_result}")
        
        self.results['input_validation'] = {
            'test_inputs': validation_results,
            'credentials_validation': credentials_result
        }
    
    async def demo_system_monitoring(self) -> Any:
        """Demonstrate system monitoring."""
        print("\n📊 System Monitoring Demo")
        print("-" * 25)
        
        monitoring_config = MonitoringConfig(
            interval=1.0,
            log_file="demo_security_events.log",
            alert_thresholds={
                'cpu_percent': 50.0,
                'memory_percent': 70.0,
                'disk_usage_percent': 80.0
            }
        )
        
        # Monitor system resources (async - I/O operation)
        print("Monitoring system resources...")
        metrics = []
        for i in range(3):  # Monitor for 3 intervals
            metric = await monitor_system_resources(monitoring_config)
            metrics.append(metric)
            print(f"Metrics {i+1}: CPU={metric.cpu_percent:.1f}%, "
                  f"Memory={metric.memory_percent:.1f}%, "
                  f"Disk={metric.disk_usage_percent:.1f}%")
            await asyncio.sleep(1)
        
        # Detect anomalies (def - CPU-bound operation)
        print("\nDetecting anomalies...")
        anomalies = detect_anomalies(metrics, monitoring_config)
        print(f"Anomalies detected: {len(anomalies)}")
        for anomaly in anomalies:
            print(f"  - {anomaly['description']}")
        
        # Track user activity (async - I/O operation)
        print("\nTracking user activity...")
        user_event = await track_user_activity(
            "demo_user",
            "login_attempt",
            {"ip": "192.168.1.100", "user_agent": "Demo Browser"},
            monitoring_config
        )
        print(f"User activity event: {user_event}")
        
        self.results['system_monitoring'] = {
            'metrics': [vars(m) for m in metrics],
            'anomalies': anomalies,
            'user_event': vars(user_event)
        }

async def run_individual_demos():
    """Run individual cybersecurity demos."""
    config = DemoConfig()
    demo = CybersecurityDemo(config)
    
    # Run individual demos
    print("🔒 Individual Cybersecurity Demos")
    print("=" * 40)
    
    # Port scanning demo
    await demo.demo_port_scanning()
    
    # Cryptographic demo
    await demo.demo_cryptographic_operations()
    
    # Network security demo
    await demo.demo_network_security()
    
    # Input validation demo
    await demo.demo_input_validation()
    
    # System monitoring demo
    await demo.demo_system_monitoring()

async def run_performance_comparison():
    """Compare performance of async vs def operations."""
    print("\n⚡ Performance Comparison: Async vs Def")
    print("=" * 45)
    
    # CPU-bound operation (def)
    start_time = time.time()
    for i in range(1000):
        hash_password(f"password{i}", SecurityConfig())
    def_time = time.time() - start_time
    print(f"CPU-bound operations (def): {def_time:.3f}s")
    
    # I/O-bound operation (async)
    start_time = time.time()
    tasks = []
    for i in range(10):
        task = check_connection("localhost", 80, NetworkConfig())
        tasks.append(task)
    await asyncio.gather(*tasks)
    async_time = time.time() - start_time
    print(f"I/O-bound operations (async): {async_time:.3f}s")
    
    print(f"Performance ratio: {def_time/async_time:.2f}x")

def main():
    """Main demo function."""
    print("🚀 Cybersecurity Tools Demo")
    print("Following functional, declarative programming principles")
    print("Using async for I/O operations, def for CPU-bound operations")
    print("=" * 60)
    
    # Run comprehensive demo
    asyncio.run(run_comprehensive_demos())
    
    # Run individual demos
    asyncio.run(run_individual_demos())
    
    # Run performance comparison
    asyncio.run(run_performance_comparison())
    
    print("\n✅ All cybersecurity demos completed successfully!")

match __name__:
    case "__main__":
    main() 