"""
Contract Testing Framework for HeyGen AI Microservices.
Advanced contract testing including API contracts, message contracts,
and service integration validation.
"""

import json
import time
import asyncio
import aiohttp
import requests
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import logging
import jsonschema
from jsonschema import validate, ValidationError
import yaml
import re
from collections import defaultdict
import threading
import concurrent.futures

@dataclass
class APIEndpoint:
    """Represents an API endpoint contract."""
    path: str
    method: str
    request_schema: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    path_params: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class MessageContract:
    """Represents a message contract for event-driven systems."""
    topic: str
    message_type: str
    schema: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    producer: str = ""
    consumer: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class ServiceContract:
    """Represents a service contract."""
    service_name: str
    version: str
    base_url: str
    endpoints: List[APIEndpoint] = field(default_factory=list)
    message_contracts: List[MessageContract] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    health_check: str = "/health"
    documentation: str = ""

@dataclass
class ContractTestResult:
    """Result of a contract test."""
    contract_id: str
    test_type: str  # api, message, integration
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    response_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class SchemaValidator:
    """Validates JSON schemas and data."""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.load_common_schemas()
    
    def load_common_schemas(self):
        """Load common JSON schemas."""
        # Common response schemas
        self.schemas["error_response"] = {
            "type": "object",
            "properties": {
                "error": {"type": "string"},
                "message": {"type": "string"},
                "code": {"type": "integer"},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["error", "message"]
        }
        
        self.schemas["success_response"] = {
            "type": "object",
            "properties": {
                "success": {"type": "boolean"},
                "data": {"type": "object"},
                "message": {"type": "string"},
                "timestamp": {"type": "string", "format": "date-time"}
            },
            "required": ["success"]
        }
        
        self.schemas["pagination"] = {
            "type": "object",
            "properties": {
                "page": {"type": "integer", "minimum": 1},
                "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                "total": {"type": "integer", "minimum": 0},
                "pages": {"type": "integer", "minimum": 0}
            },
            "required": ["page", "limit", "total"]
        }
    
    def validate_data(self, data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate data against a schema."""
        try:
            validate(instance=data, schema=schema)
            return True, []
        except ValidationError as e:
            return False, [str(e)]
        except Exception as e:
            return False, [f"Schema validation error: {str(e)}"]
    
    def validate_api_response(self, response_data: Dict[str, Any], 
                            response_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate API response against schema."""
        return self.validate_data(response_data, response_schema)
    
    def validate_message(self, message_data: Dict[str, Any], 
                        message_schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate message against schema."""
        return self.validate_data(message_data, message_schema)

class APIContractTester:
    """Tests API contracts."""
    
    def __init__(self, validator: SchemaValidator):
        self.validator = validator
        self.session = requests.Session()
        self.timeout = 30
    
    def test_endpoint_contract(self, endpoint: APIEndpoint, base_url: str) -> ContractTestResult:
        """Test an API endpoint contract."""
        start_time = time.time()
        
        try:
            # Build URL
            url = f"{base_url.rstrip('/')}/{endpoint.path.lstrip('/')}"
            
            # Prepare headers
            headers = endpoint.headers.copy()
            headers.setdefault('Content-Type', 'application/json')
            headers.setdefault('Accept', 'application/json')
            
            # Prepare request data
            request_data = None
            if endpoint.method in ['POST', 'PUT', 'PATCH']:
                # Use first example or default data
                if endpoint.examples:
                    request_data = endpoint.examples[0].get('request', {})
                else:
                    request_data = self._generate_sample_data(endpoint.request_schema)
            
            # Make request
            response = self.session.request(
                method=endpoint.method,
                url=url,
                headers=headers,
                json=request_data,
                params=endpoint.query_params,
                timeout=self.timeout
            )
            
            execution_time = time.time() - start_time
            
            # Validate response
            success, errors = self._validate_response(response, endpoint)
            
            return ContractTestResult(
                contract_id=f"{endpoint.method}_{endpoint.path}",
                test_type="api",
                success=success,
                errors=errors,
                execution_time=execution_time,
                response_data={
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ContractTestResult(
                contract_id=f"{endpoint.method}_{endpoint.path}",
                test_type="api",
                success=False,
                errors=[f"Request failed: {str(e)}"],
                execution_time=execution_time
            )
    
    def _validate_response(self, response: requests.Response, endpoint: APIEndpoint) -> Tuple[bool, List[str]]:
        """Validate API response."""
        errors = []
        
        # Check status code
        if response.status_code >= 400:
            errors.append(f"Unexpected status code: {response.status_code}")
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('application/json'):
            errors.append(f"Unexpected content type: {content_type}")
            return len(errors) == 0, errors
        
        # Parse JSON response
        try:
            response_data = response.json()
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON response: {str(e)}")
            return len(errors) == 0, errors
        
        # Validate against schema
        if endpoint.response_schema:
            valid, schema_errors = self.validator.validate_api_response(
                response_data, endpoint.response_schema
            )
            if not valid:
                errors.extend(schema_errors)
        
        return len(errors) == 0, errors
    
    def _generate_sample_data(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate sample data from schema."""
        if not schema:
            return {}
        
        sample_data = {}
        
        if 'properties' in schema:
            for prop_name, prop_schema in schema['properties'].items():
                sample_data[prop_name] = self._generate_sample_value(prop_schema)
        
        return sample_data
    
    def _generate_sample_value(self, prop_schema: Dict[str, Any]) -> Any:
        """Generate sample value from property schema."""
        prop_type = prop_schema.get('type', 'string')
        
        if prop_type == 'string':
            return prop_schema.get('example', 'sample_string')
        elif prop_type == 'integer':
            return prop_schema.get('example', 42)
        elif prop_type == 'number':
            return prop_schema.get('example', 3.14)
        elif prop_type == 'boolean':
            return prop_schema.get('example', True)
        elif prop_type == 'array':
            return [self._generate_sample_value(prop_schema.get('items', {}))]
        elif prop_type == 'object':
            return self._generate_sample_data(prop_schema)
        else:
            return None

class MessageContractTester:
    """Tests message contracts for event-driven systems."""
    
    def __init__(self, validator: SchemaValidator):
        self.validator = validator
    
    def test_message_contract(self, message_contract: MessageContract, 
                            message_data: Dict[str, Any]) -> ContractTestResult:
        """Test a message contract."""
        start_time = time.time()
        
        try:
            # Validate message against schema
            valid, errors = self.validator.validate_message(
                message_data, message_contract.schema
            )
            
            execution_time = time.time() - start_time
            
            # Additional validations
            warnings = []
            
            # Check message type
            if 'message_type' in message_data:
                if message_data['message_type'] != message_contract.message_type:
                    warnings.append(f"Message type mismatch: expected {message_contract.message_type}, got {message_data['message_type']}")
            
            # Check version
            if 'version' in message_data:
                if message_data['version'] != message_contract.version:
                    warnings.append(f"Version mismatch: expected {message_contract.version}, got {message_data['version']}")
            
            return ContractTestResult(
                contract_id=f"{message_contract.topic}_{message_contract.message_type}",
                test_type="message",
                success=valid,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time,
                response_data=message_data
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ContractTestResult(
                contract_id=f"{message_contract.topic}_{message_contract.message_type}",
                test_type="message",
                success=False,
                errors=[f"Message validation failed: {str(e)}"],
                execution_time=execution_time
            )

class ServiceIntegrationTester:
    """Tests service integration contracts."""
    
    def __init__(self, validator: SchemaValidator):
        self.validator = validator
        self.session = requests.Session()
    
    def test_service_integration(self, service_contract: ServiceContract, 
                               dependent_services: List[ServiceContract]) -> ContractTestResult:
        """Test service integration."""
        start_time = time.time()
        
        try:
            errors = []
            warnings = []
            
            # Test health check
            health_url = f"{service_contract.base_url}{service_contract.health_check}"
            try:
                health_response = self.session.get(health_url, timeout=10)
                if health_response.status_code != 200:
                    errors.append(f"Health check failed: {health_response.status_code}")
            except Exception as e:
                errors.append(f"Health check error: {str(e)}")
            
            # Test dependencies
            for dep_service in dependent_services:
                dep_health_url = f"{dep_service.base_url}{dep_service.health_check}"
                try:
                    dep_response = self.session.get(dep_health_url, timeout=10)
                    if dep_response.status_code != 200:
                        warnings.append(f"Dependency {dep_service.service_name} health check failed")
                except Exception as e:
                    warnings.append(f"Dependency {dep_service.service_name} is unreachable")
            
            # Test API endpoints
            for endpoint in service_contract.endpoints:
                endpoint_tester = APIContractTester(self.validator)
                endpoint_result = endpoint_tester.test_endpoint_contract(
                    endpoint, service_contract.base_url
                )
                
                if not endpoint_result.success:
                    errors.extend([f"Endpoint {endpoint.method} {endpoint.path}: {error}" 
                                 for error in endpoint_result.errors])
            
            execution_time = time.time() - start_time
            
            return ContractTestResult(
                contract_id=f"integration_{service_contract.service_name}",
                test_type="integration",
                success=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return ContractTestResult(
                contract_id=f"integration_{service_contract.service_name}",
                test_type="integration",
                success=False,
                errors=[f"Integration test failed: {str(e)}"],
                execution_time=execution_time
            )

class ContractTestingFramework:
    """Main contract testing framework."""
    
    def __init__(self):
        self.validator = SchemaValidator()
        self.api_tester = APIContractTester(self.validator)
        self.message_tester = MessageContractTester(self.validator)
        self.integration_tester = ServiceIntegrationTester(self.validator)
        self.contracts: Dict[str, ServiceContract] = {}
        self.results: List[ContractTestResult] = []
    
    def load_contract_from_file(self, contract_file: str) -> ServiceContract:
        """Load service contract from file."""
        contract_path = Path(contract_file)
        
        if not contract_path.exists():
            raise FileNotFoundError(f"Contract file not found: {contract_file}")
        
        with open(contract_path, 'r') as f:
            if contract_path.suffix == '.yaml' or contract_path.suffix == '.yml':
                contract_data = yaml.safe_load(f)
            else:
                contract_data = json.load(f)
        
        # Parse contract data
        service_contract = ServiceContract(
            service_name=contract_data['service_name'],
            version=contract_data['version'],
            base_url=contract_data['base_url'],
            health_check=contract_data.get('health_check', '/health'),
            documentation=contract_data.get('documentation', '')
        )
        
        # Parse endpoints
        for endpoint_data in contract_data.get('endpoints', []):
            endpoint = APIEndpoint(
                path=endpoint_data['path'],
                method=endpoint_data['method'],
                request_schema=endpoint_data.get('request_schema', {}),
                response_schema=endpoint_data.get('response_schema', {}),
                headers=endpoint_data.get('headers', {}),
                query_params=endpoint_data.get('query_params', {}),
                path_params=endpoint_data.get('path_params', {}),
                examples=endpoint_data.get('examples', [])
            )
            service_contract.endpoints.append(endpoint)
        
        # Parse message contracts
        for message_data in contract_data.get('message_contracts', []):
            message_contract = MessageContract(
                topic=message_data['topic'],
                message_type=message_data['message_type'],
                schema=message_data.get('schema', {}),
                version=message_data.get('version', '1.0.0'),
                producer=message_data.get('producer', ''),
                consumer=message_data.get('consumer', ''),
                examples=message_data.get('examples', [])
            )
            service_contract.message_contracts.append(message_contract)
        
        # Store contract
        self.contracts[service_contract.service_name] = service_contract
        
        return service_contract
    
    def test_api_contracts(self, service_name: str) -> List[ContractTestResult]:
        """Test API contracts for a service."""
        if service_name not in self.contracts:
            raise ValueError(f"Service contract not found: {service_name}")
        
        service_contract = self.contracts[service_name]
        results = []
        
        print(f"🔗 Testing API Contracts for {service_name}")
        print("=" * 50)
        
        for endpoint in service_contract.endpoints:
            print(f"Testing {endpoint.method} {endpoint.path}")
            result = self.api_tester.test_endpoint_contract(endpoint, service_contract.base_url)
            results.append(result)
            
            status_icon = "✅" if result.success else "❌"
            print(f"  {status_icon} {result.contract_id}: {result.execution_time:.2f}s")
            
            if result.errors:
                for error in result.errors:
                    print(f"    ❌ {error}")
        
        self.results.extend(results)
        return results
    
    def test_message_contracts(self, service_name: str, 
                              message_data: Dict[str, Dict[str, Any]]) -> List[ContractTestResult]:
        """Test message contracts for a service."""
        if service_name not in self.contracts:
            raise ValueError(f"Service contract not found: {service_name}")
        
        service_contract = self.contracts[service_name]
        results = []
        
        print(f"📨 Testing Message Contracts for {service_name}")
        print("=" * 50)
        
        for message_contract in service_contract.message_contracts:
            contract_key = f"{message_contract.topic}_{message_contract.message_type}"
            
            if contract_key in message_data:
                print(f"Testing {message_contract.topic}/{message_contract.message_type}")
                result = self.message_tester.test_message_contract(
                    message_contract, message_data[contract_key]
                )
                results.append(result)
                
                status_icon = "✅" if result.success else "❌"
                print(f"  {status_icon} {result.contract_id}: {result.execution_time:.2f}s")
                
                if result.errors:
                    for error in result.errors:
                        print(f"    ❌ {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        print(f"    ⚠️  {warning}")
            else:
                print(f"⚠️  No test data for {contract_key}")
        
        self.results.extend(results)
        return results
    
    def test_service_integration(self, service_name: str, 
                               dependent_services: List[str] = None) -> ContractTestResult:
        """Test service integration."""
        if service_name not in self.contracts:
            raise ValueError(f"Service contract not found: {service_name}")
        
        service_contract = self.contracts[service_name]
        
        # Get dependent services
        dep_services = []
        if dependent_services:
            for dep_name in dependent_services:
                if dep_name in self.contracts:
                    dep_services.append(self.contracts[dep_name])
        
        print(f"🔗 Testing Service Integration for {service_name}")
        print("=" * 50)
        
        result = self.integration_tester.test_service_integration(
            service_contract, dep_services
        )
        
        status_icon = "✅" if result.success else "❌"
        print(f"{status_icon} Integration test: {result.execution_time:.2f}s")
        
        if result.errors:
            for error in result.errors:
                print(f"  ❌ {error}")
        
        if result.warnings:
            for warning in result.warnings:
                print(f"  ⚠️  {warning}")
        
        self.results.append(result)
        return result
    
    def run_contract_test_suite(self, services: List[str]) -> Dict[str, Any]:
        """Run contract testing suite for multiple services."""
        print("🧪 Running Contract Testing Suite")
        print("=" * 50)
        
        suite_results = {}
        
        for service_name in services:
            if service_name not in self.contracts:
                print(f"⚠️  Skipping {service_name} - contract not found")
                continue
            
            print(f"\n🔍 Testing {service_name}")
            print("-" * 30)
            
            # Test API contracts
            api_results = self.test_api_contracts(service_name)
            
            # Test message contracts (if any)
            message_results = self.test_message_contracts(service_name, {})
            
            # Test service integration
            integration_result = self.test_service_integration(service_name)
            
            # Calculate summary
            total_tests = len(api_results) + len(message_results) + 1
            successful_tests = sum(1 for r in api_results + message_results + [integration_result] if r.success)
            
            suite_results[service_name] = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                "api_tests": len(api_results),
                "message_tests": len(message_results),
                "integration_test": integration_result.success
            }
        
        # Print suite summary
        self._print_suite_summary(suite_results)
        
        return suite_results
    
    def _print_suite_summary(self, suite_results: Dict[str, Any]):
        """Print contract testing suite summary."""
        print("\n" + "=" * 60)
        print("📊 CONTRACT TESTING SUITE SUMMARY")
        print("=" * 60)
        
        total_services = len(suite_results)
        total_tests = sum(r['total_tests'] for r in suite_results.values())
        total_successful = sum(r['successful_tests'] for r in suite_results.values())
        overall_success_rate = (total_successful / total_tests * 100) if total_tests > 0 else 0
        
        print(f"📈 Overall Summary:")
        print(f"   Services Tested: {total_services}")
        print(f"   Total Tests: {total_tests}")
        print(f"   Successful: {total_successful}")
        print(f"   Success Rate: {overall_success_rate:.1f}%")
        
        print(f"\n📋 Service Details:")
        for service_name, results in suite_results.items():
            status_icon = "✅" if results['success_rate'] >= 80 else "⚠️" if results['success_rate'] >= 60 else "❌"
            print(f"   {status_icon} {service_name}: {results['success_rate']:.1f}% ({results['successful_tests']}/{results['total_tests']})")
        
        # Overall assessment
        if overall_success_rate >= 90:
            print(f"\n🎉 Contract Compliance: EXCELLENT")
        elif overall_success_rate >= 80:
            print(f"\n👍 Contract Compliance: GOOD")
        elif overall_success_rate >= 60:
            print(f"\n⚠️  Contract Compliance: FAIR")
        else:
            print(f"\n❌ Contract Compliance: POOR")
        
        print("=" * 60)
    
    def generate_contract_report(self, output_file: str = "contract_test_report.json"):
        """Generate contract testing report."""
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": sum(1 for r in self.results if r.success),
                "failed_tests": sum(1 for r in self.results if not r.success),
                "success_rate": (sum(1 for r in self.results if r.success) / len(self.results) * 100) if self.results else 0
            },
            "results": [
                {
                    "contract_id": r.contract_id,
                    "test_type": r.test_type,
                    "success": r.success,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "execution_time": r.execution_time,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Contract test report saved to: {output_file}")

# Example usage and demo
def demo_contract_testing():
    """Demonstrate contract testing capabilities."""
    print("🔗 Contract Testing Framework Demo")
    print("=" * 40)
    
    # Create contract testing framework
    framework = ContractTestingFramework()
    
    # Create sample service contract
    sample_contract = {
        "service_name": "user-service",
        "version": "1.0.0",
        "base_url": "http://localhost:8000",
        "health_check": "/health",
        "endpoints": [
            {
                "path": "/api/users",
                "method": "GET",
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "users": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "integer"},
                                    "name": {"type": "string"},
                                    "email": {"type": "string"}
                                }
                            }
                        }
                    }
                }
            },
            {
                "path": "/api/users",
                "method": "POST",
                "request_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    },
                    "required": ["name", "email"]
                },
                "response_schema": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                }
            }
        ],
        "message_contracts": [
            {
                "topic": "user.created",
                "message_type": "UserCreated",
                "schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "integer"},
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "timestamp": {"type": "string", "format": "date-time"}
                    },
                    "required": ["user_id", "name", "email", "timestamp"]
                }
            }
        ]
    }
    
    # Save sample contract
    with open("sample_contract.json", "w") as f:
        json.dump(sample_contract, f, indent=2)
    
    try:
        # Load contract
        framework.load_contract_from_file("sample_contract.json")
        
        # Test API contracts
        api_results = framework.test_api_contracts("user-service")
        
        # Test message contracts
        message_data = {
            "user.created_UserCreated": {
                "user_id": 123,
                "name": "John Doe",
                "email": "john@example.com",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
        message_results = framework.test_message_contracts("user-service", message_data)
        
        # Test service integration
        integration_result = framework.test_service_integration("user-service")
        
        # Generate report
        framework.generate_contract_report()
        
    finally:
        # Cleanup
        Path("sample_contract.json").unlink()

if __name__ == "__main__":
    # Run demo
    demo_contract_testing()
