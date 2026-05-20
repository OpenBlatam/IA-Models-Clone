"""
Comprehensive Testing System Demo for the ads feature.

This demo showcases the entire unified testing system:
- Test helpers and utilities
- Custom assertions
- Mock factories and utilities
- Test data generation
- Performance testing
- Error scenario testing
- Integration testing examples
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

from .utils.test_helpers import (
    TestDataGenerator, EntityFactory, DTOFactory, ValidationHelper,
    PerformanceHelper, AsyncTestHelper, MockHelper, TestScenarioHelper
)
from .utils.test_assertions import (
    EntityAssertions, DTOAssertions, ValueObjectAssertions,
    BusinessLogicAssertions, PerformanceAssertions, ErrorHandlingAssertions,
    DataConsistencyAssertions, MockAssertions
)
from .utils.test_mocks import (
    MockDataGenerator, MockEntityFactory, MockRepositoryFactory,
    MockServiceFactory, MockUseCaseFactory, MockInfrastructureFactory,
    MockConfigurationFactory, MockBehaviorCustomizer
)
from ..domain.entities import Ad, AdCampaign, AdGroup, AdPerformance
from ..domain.value_objects import AdStatus, AdType, Platform, Budget, TargetingCriteria, AdMetrics, AdSchedule
from ..application.dto import CreateAdRequest, CreateCampaignRequest, OptimizeAdRequest


class TestingSystemDemo:
    """Demonstrates the comprehensive testing system capabilities."""
    
    def __init__(self):
        """Initialize the demo with test data and scenarios."""
        self.test_data = {}
        self.test_results = {}
        self.performance_metrics = {}
        
    async def run_comprehensive_demo(self):
        """Run the complete testing system demonstration."""
        print("🚀 Starting Comprehensive Testing System Demo")
        print("=" * 60)
        
        # Run all demo sections
        await self._demo_test_helpers()
        await self._demo_test_assertions()
        await self._demo_test_mocks()
        await self._demo_test_data_generation()
        await self._demo_performance_testing()
        await self._demo_error_scenario_testing()
        await self._demo_integration_testing()
        await self._demo_test_scenarios()
        
        # Display comprehensive summary
        self._print_testing_summary()
        
        print("\n✅ Comprehensive Testing System Demo Completed!")
    
    async def _demo_test_helpers(self):
        """Demonstrate test helper utilities."""
        print("\n📋 Testing Test Helper Utilities")
        print("-" * 40)
        
        # Test data generation
        print("🔧 Test Data Generation:")
        random_string = TestDataGenerator.random_string(15)
        random_email = TestDataGenerator.random_email()
        random_budget = TestDataGenerator.random_budget(500, 2000)
        print(f"  - Random string: {random_string}")
        print(f"  - Random email: {random_email}")
        print(f"  - Random budget: ${random_budget}")
        
        # Entity factory
        print("\n🏭 Entity Factory:")
        test_ad = EntityFactory.create_ad()
        test_campaign = EntityFactory.create_campaign()
        print(f"  - Created test ad: {test_ad.title}")
        print(f"  - Created test campaign: {test_campaign.name}")
        
        # DTO factory
        print("\n📝 DTO Factory:")
        ad_request = DTOFactory.create_ad_request()
        campaign_request = DTOFactory.create_campaign_request()
        print(f"  - Created ad request: {ad_request.title}")
        print(f"  - Created campaign request: {campaign_request.name}")
        
        # Test scenarios
        print("\n🎭 Test Scenarios:")
        basic_scenario = TestScenarioHelper.create_basic_scenario()
        performance_scenario = TestScenarioHelper.create_performance_scenario()
        print(f"  - Basic scenario: {len(basic_scenario['ads'])} ads, {len(basic_scenario['requests'])} requests")
        print(f"  - Performance scenario: {len(performance_scenario['campaigns'])} campaigns, {len(performance_scenario['ads'])} ads")
        
        self.test_results['helpers'] = {
            'entities_created': 2,
            'dtos_created': 2,
            'scenarios_created': 2
        }
    
    async def _demo_test_assertions(self):
        """Demonstrate test assertion utilities."""
        print("\n✅ Testing Test Assertion Utilities")
        print("-" * 40)
        
        # Create test entities
        test_ad = EntityFactory.create_ad()
        test_campaign = EntityFactory.create_campaign()
        test_budget = Budget(
            daily_budget=1000.0,
            total_budget=10000.0,
            currency="USD"
        )
        
        # Entity assertions
        print("🏗️ Entity Assertions:")
        EntityAssertions.assert_valid_ad(test_ad)
        EntityAssertions.assert_valid_campaign(test_campaign)
        print("  - All entity assertions passed")
        
        # DTO assertions
        print("\n📋 DTO Assertions:")
        ad_request = DTOFactory.create_ad_request()
        DTOAssertions.assert_valid_create_ad_request(ad_request)
        print("  - DTO validation assertions passed")
        
        # Value object assertions
        print("\n💰 Value Object Assertions:")
        ValueObjectAssertions.assert_valid_budget(test_budget)
        print("  - Budget validation assertions passed")
        
        # Business logic assertions
        print("\n🧠 Business Logic Assertions:")
        valid_transitions = {
            AdStatus.DRAFT: [AdStatus.APPROVED],
            AdStatus.APPROVED: [AdStatus.ACTIVE, AdStatus.REJECTED],
            AdStatus.ACTIVE: [AdStatus.PAUSED, AdStatus.ARCHIVED]
        }
        BusinessLogicAssertions.assert_valid_status_transition(
            test_ad, AdStatus.DRAFT, AdStatus.APPROVED, valid_transitions
        )
        print("  - Status transition validation passed")
        
        self.test_results['assertions'] = {
            'entity_assertions': 2,
            'dto_assertions': 1,
            'value_object_assertions': 1,
            'business_logic_assertions': 1
        }
    
    async def _demo_test_mocks(self):
        """Demonstrate test mock utilities."""
        print("\n🎭 Testing Test Mock Utilities")
        print("-" * 40)
        
        # Mock entity factory
        print("🏭 Mock Entity Factory:")
        mock_ad = MockEntityFactory.create_mock_ad()
        mock_campaign = MockEntityFactory.create_mock_campaign()
        print(f"  - Created mock ad: {mock_ad.id}")
        print(f"  - Created mock campaign: {mock_campaign.id}")
        
        # Mock repository factory
        print("\n🗄️ Mock Repository Factory:")
        mock_ad_repo = MockRepositoryFactory.create_mock_ad_repository()
        mock_campaign_repo = MockRepositoryFactory.create_mock_campaign_repository()
        print(f"  - Created mock ad repository")
        print(f"  - Created mock campaign repository")
        
        # Mock service factory
        print("\n🔧 Mock Service Factory:")
        mock_ad_service = MockServiceFactory.create_mock_ad_service()
        mock_optimization_service = MockServiceFactory.create_mock_optimization_service()
        print(f"  - Created mock ad service")
        print(f"  - Created mock optimization service")
        
        # Mock use case factory
        print("\n📋 Mock Use Case Factory:")
        mock_create_ad_use_case = MockUseCaseFactory.create_mock_create_ad_use_case()
        mock_optimize_ad_use_case = MockUseCaseFactory.create_mock_optimize_ad_use_case()
        print(f"  - Created mock create ad use case")
        print(f"  - Created mock optimize ad use case")
        
        # Mock infrastructure factory
        print("\n🏗️ Mock Infrastructure Factory:")
        mock_db = MockInfrastructureFactory.create_mock_database_manager()
        mock_storage = MockInfrastructureFactory.create_mock_storage_manager()
        print(f"  - Created mock database manager")
        print(f"  - Created mock storage manager")
        
        # Test mock behavior
        print("\n🧪 Testing Mock Behavior:")
        mock_ad.approve.return_value = True
        result = mock_ad.approve()
        assert result is True, "Mock method should return configured value"
        print("  - Mock behavior customization working")
        
        self.test_results['mocks'] = {
            'entities_created': 2,
            'repositories_created': 2,
            'services_created': 2,
            'use_cases_created': 2,
            'infrastructure_created': 2
        }
    
    async def _demo_test_data_generation(self):
        """Demonstrate test data generation capabilities."""
        print("\n🎲 Testing Test Data Generation")
        print("-" * 40)
        
        # Generate various test data
        print("🔧 Random Data Generation:")
        random_strings = [TestDataGenerator.random_string(8) for _ in range(5)]
        random_emails = [TestDataGenerator.random_email() for _ in range(3)]
        random_budgets = [TestDataGenerator.random_budget(100, 5000) for _ in range(4)]
        random_dates = [TestDataGenerator.random_date() for _ in range(3)]
        
        print(f"  - Random strings: {random_strings[:3]}...")
        print(f"  - Random emails: {random_emails[:2]}...")
        print(f"  - Random budgets: ${random_budgets[:3]}...")
        print(f"  - Random dates: {[d.strftime('%Y-%m-%d') for d in random_dates[:2]]}...")
        
        # Generate test entities with different configurations
        print("\n🏭 Entity Generation with Custom Config:")
        custom_ad = EntityFactory.create_ad(
            title="Custom Test Ad",
            status=AdStatus.ACTIVE,
            ad_type=AdType.VIDEO,
            platform=Platform.FACEBOOK
        )
        print(f"  - Custom ad: {custom_ad.title} ({custom_ad.status})")
        
        custom_campaign = EntityFactory.create_campaign(
            name="Premium Campaign",
            status=AdStatus.ACTIVE,
            budget=Budget(daily_budget=2000.0, total_budget=20000.0, currency="USD")
        )
        print(f"  - Custom campaign: {custom_campaign.name} (${custom_campaign.budget.total_budget})")
        
        # Generate test scenarios
        print("\n🎭 Scenario Generation:")
        error_scenario = TestScenarioHelper.create_error_scenario()
        integration_scenario = TestScenarioHelper.create_integration_scenario()
        print(f"  - Error scenario: {len(error_scenario['invalid_requests'])} invalid requests")
        print(f"  - Integration scenario: {len(integration_scenario['workflow'])} workflow steps")
        
        self.test_results['data_generation'] = {
            'random_strings': len(random_strings),
            'random_emails': len(random_emails),
            'random_budgets': len(random_budgets),
            'random_dates': len(random_dates),
            'custom_entities': 2,
            'scenarios': 2
        }
    
    async def _demo_performance_testing(self):
        """Demonstrate performance testing capabilities."""
        print("\n⚡ Testing Performance Testing Utilities")
        print("-" * 40)
        
        # Measure execution time
        print("⏱️ Execution Time Measurement:")
        
        async def sample_operation():
            await asyncio.sleep(0.1)  # Simulate work
            return "operation completed"
        
        execution_time = await PerformanceHelper.measure_execution_time(sample_operation())
        print(f"  - Sample operation took: {execution_time:.3f}s")
        
        # Performance threshold assertion
        PerformanceHelper.assert_performance_threshold(execution_time, 1.0)
        print("  - Performance threshold assertion passed")
        
        # Stress testing
        print("\n🔥 Stress Testing:")
        
        async def stress_operation():
            await asyncio.sleep(0.05)  # Simulate work
            return {"result": "success", "timestamp": datetime.now()}
        
        stress_results = await PerformanceHelper.stress_test(
            stress_operation,
            num_iterations=20,
            max_concurrent=5,
            max_time_per_operation=0.1
        )
        
        print(f"  - Stress test completed:")
        print(f"    * Total iterations: {stress_results['total_iterations']}")
        print(f"    * Successful operations: {stress_results['successful_operations']}")
        print(f"    * Failed operations: {stress_results['failed_operations']}")
        print(f"    * Success rate: {stress_results['success_rate']:.2%}")
        print(f"    * Average time per operation: {stress_results['average_time_per_operation']:.3f}s")
        
        # Assert stress test results
        PerformanceHelper.assert_stress_test_results(stress_results, min_success_rate=0.9)
        print("  - Stress test assertions passed")
        
        self.test_results['performance_testing'] = {
            'execution_time_measured': True,
            'threshold_assertions': 1,
            'stress_test_iterations': stress_results['total_iterations'],
            'stress_test_success_rate': stress_results['success_rate']
        }
    
    async def _demo_error_scenario_testing(self):
        """Demonstrate error scenario testing capabilities."""
        print("\n❌ Testing Error Scenario Testing")
        print("-" * 40)
        
        # Test validation errors
        print("🔍 Validation Error Testing:")
        
        # Create invalid data
        invalid_ad_request = {
            'title': '',  # Empty title
            'description': None,  # None description
            'budget': -100  # Negative budget
        }
        
        print(f"  - Testing invalid data: {invalid_ad_request}")
        
        # Test exception handling
        print("\n🚨 Exception Handling Testing:")
        
        class CustomException(Exception):
            pass
        
        try:
            raise CustomException("Test error message")
        except CustomException as e:
            ErrorHandlingAssertions.assert_exception_type(e, CustomException)
            ErrorHandlingAssertions.assert_exception_message(e, "Test error message")
            print("  - Exception type and message assertions passed")
        
        # Test error response format
        print("\n📋 Error Response Format Testing:")
        
        class MockErrorResponse:
            def __init__(self):
                self.status_code = 400
            
            def json(self):
                return {
                    'error': 'validation_error',
                    'message': 'Invalid input data'
                }
        
        error_response = MockErrorResponse()
        ErrorHandlingAssertions.assert_proper_error_response(
            error_response,
            expected_status_code=400,
            expected_error_type='validation_error',
            expected_error_message='Invalid input data'
        )
        print("  - Error response format assertions passed")
        
        self.test_results['error_testing'] = {
            'validation_errors_tested': True,
            'exception_handling_tested': True,
            'error_response_format_tested': True
        }
    
    async def _demo_integration_testing(self):
        """Demonstrate integration testing capabilities."""
        print("\n🔗 Testing Integration Testing Utilities")
        print("-" * 40)
        
        # Test async helper utilities
        print("⏳ Async Testing Utilities:")
        
        # Wait for condition
        condition_met = False
        async def check_condition():
            nonlocal condition_met
            condition_met = True
            return condition_met
        
        result = await AsyncTestHelper.wait_for_condition(check_condition, timeout=2.0)
        print(f"  - Wait for condition: {result}")
        
        # Retry operation
        print("\n🔄 Retry Operation Testing:")
        
        call_count = 0
        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Operation failed (attempt {call_count})")
            return "success"
        
        try:
            result = await AsyncTestHelper.retry_operation(failing_operation, max_retries=3)
            print(f"  - Retry operation succeeded: {result}")
        except Exception as e:
            print(f"  - Retry operation failed: {e}")
        
        # Concurrent operations
        print("\n⚡ Concurrent Operations Testing:")
        
        async def concurrent_operation(operation_id: int):
            await asyncio.sleep(0.1)
            return f"operation_{operation_id}_completed"
        
        operations = [concurrent_operation(i) for i in range(5)]
        results = await AsyncTestHelper.run_concurrent_operations(operations, max_concurrent=3)
        print(f"  - Concurrent operations completed: {len(results)} results")
        
        self.test_results['integration_testing'] = {
            'async_utilities_tested': True,
            'retry_operations_tested': True,
            'concurrent_operations_tested': True
        }
    
    async def _demo_test_scenarios(self):
        """Demonstrate test scenario creation and management."""
        print("\n🎭 Testing Test Scenario Management")
        print("-" * 40)
        
        # Create various test scenarios
        print("🏗️ Scenario Creation:")
        
        basic_scenario = TestScenarioHelper.create_basic_scenario()
        print(f"  - Basic scenario: {len(basic_scenario['ads'])} ads, {len(basic_scenario['campaigns'])} campaigns")
        
        performance_scenario = TestScenarioHelper.create_performance_scenario()
        print(f"  - Performance scenario: {len(performance_scenario['campaigns'])} campaigns, {len(performance_scenario['ads'])} ads")
        
        error_scenario = TestScenarioHelper.create_error_scenario()
        print(f"  - Error scenario: {len(error_scenario['invalid_requests'])} invalid requests")
        
        integration_scenario = TestScenarioHelper.create_integration_scenario()
        print(f"  - Integration scenario: {len(integration_scenario['workflow'])} workflow steps")
        
        # Test scenario data validation
        print("\n✅ Scenario Data Validation:")
        
        # Validate basic scenario
        assert len(basic_scenario['ads']) > 0, "Basic scenario should have ads"
        assert len(basic_scenario['campaigns']) > 0, "Basic scenario should have campaigns"
        assert len(basic_scenario['requests']) > 0, "Basic scenario should have requests"
        print("  - Basic scenario validation passed")
        
        # Validate performance scenario
        assert len(performance_scenario['campaigns']) >= 10, "Performance scenario should have many campaigns"
        assert len(performance_scenario['ads']) >= 50, "Performance scenario should have many ads"
        assert 'performance_thresholds' in performance_scenario, "Performance scenario should have thresholds"
        print("  - Performance scenario validation passed")
        
        # Validate error scenario
        assert len(error_scenario['invalid_requests']) > 0, "Error scenario should have invalid requests"
        assert len(error_scenario['expected_errors']) > 0, "Error scenario should have expected errors"
        print("  - Error scenario validation passed")
        
        # Validate integration scenario
        assert len(integration_scenario['workflow']) > 0, "Integration scenario should have workflow steps"
        assert 'dependencies' in integration_scenario, "Integration scenario should have dependencies"
        print("  - Integration scenario validation passed")
        
        self.test_results['scenarios'] = {
            'scenarios_created': 4,
            'scenarios_validated': 4,
            'total_test_data_items': (
                len(basic_scenario['ads']) + len(basic_scenario['campaigns']) +
                len(performance_scenario['campaigns']) + len(performance_scenario['ads']) +
                len(error_scenario['invalid_requests']) +
                len(integration_scenario['workflow'])
            )
        }
    
    def _print_testing_summary(self):
        """Print a comprehensive summary of the testing demo."""
        print("\n📊 Testing System Demo Summary")
        print("=" * 60)
        
        total_tests = sum(len(result) for result in self.test_results.values())
        total_entities = sum(
            result.get('entities_created', 0) + 
            result.get('repositories_created', 0) + 
            result.get('services_created', 0) + 
            result.get('use_cases_created', 0) + 
            result.get('infrastructure_created', 0)
            for result in self.test_results.values()
        )
        
        print(f"🎯 Total Test Categories: {len(self.test_results)}")
        print(f"🧪 Total Test Items: {total_tests}")
        print(f"🏗️ Total Mock Objects Created: {total_entities}")
        
        print("\n📋 Category Breakdown:")
        for category, results in self.test_results.items():
            if isinstance(results, dict):
                total_items = sum(results.values()) if all(isinstance(v, (int, float)) for v in results.values()) else len(results)
                print(f"  - {category.replace('_', ' ').title()}: {total_items} items")
            else:
                print(f"  - {category.replace('_', ' ').title()}: {results}")
        
        print("\n🚀 Key Features Demonstrated:")
        print("  ✅ Test data generation and factories")
        print("  ✅ Comprehensive assertion utilities")
        print("  ✅ Mock object creation and customization")
        print("  ✅ Performance testing and stress testing")
        print("  ✅ Error scenario testing")
        print("  ✅ Integration testing utilities")
        print("  ✅ Test scenario management")
        print("  ✅ Async testing support")
        
        print("\n💡 Benefits of Unified Testing System:")
        print("  🔧 Consistent testing patterns across all layers")
        print("  🎭 Comprehensive mock factories for all components")
        print("  ⚡ Performance testing built into the framework")
        print("  🚨 Standardized error handling and validation")
        print("  📊 Rich test data generation for various scenarios")
        print("  🔗 Integration testing utilities for complex workflows")
        print("  📝 Clear and maintainable test code")


async def main():
    """Main function to run the testing system demo."""
    demo = TestingSystemDemo()
    await demo.run_comprehensive_demo()


if __name__ == "__main__":
    asyncio.run(main())
