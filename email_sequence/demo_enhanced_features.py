#!/usr/bin/env python3
"""
Enhanced Features Demo for Email Sequence System

This demo showcases all the new enhanced optimizations and improvements:
- Unified Configuration Management
- Advanced Caching System
- Enhanced Error Handling and Resilience
- Machine Learning-based optimization
- Intelligent monitoring
- Advanced performance optimization
- Real-time analytics
"""

import asyncio
import logging
import time
import json
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced components
from core.unified_config import UnifiedConfig, Environment
from core.advanced_caching import AdvancedCache, CacheConfig, CacheStrategy, CacheManager
from core.enhanced_error_handling import (
    ResilienceManager, CircuitBreakerConfig, RetryConfig, 
    RetryStrategy, ErrorSeverity, get_resilience_manager
)
from core.performance_optimizer import OptimizedPerformanceOptimizer, OptimizationConfig
from core.advanced_optimizer import AdvancedOptimizer, AdvancedOptimizationConfig
from core.intelligent_monitor import IntelligentMonitor, MonitoringConfig
from core.email_sequence_engine import EmailSequenceEngine, ProcessingResult
from models.sequence import EmailSequence, SequenceStep, StepType
from models.subscriber import Subscriber
from models.template import EmailTemplate


class EnhancedFeaturesDemo:
    """Demo class showcasing all enhanced features"""

    def __init__(self):
        self.unified_config = None
        self.cache = None
        self.resilience_manager = None
        self.performance_optimizer = None
        self.advanced_optimizer = None
        self.intelligent_monitor = None
        self.engine = None
        self.demo_data = {}

    async def setup_demo_environment(self):
        """Setup demo environment with all enhanced components"""
        logger.info("Setting up enhanced demo environment...")

        try:
            # Initialize unified configuration
            self.unified_config = UnifiedConfig()
            logger.info(f"Unified config loaded for environment: {self.unified_config.environment.value}")

            # Initialize advanced caching
            cache_config = CacheConfig(
                l1_enabled=True,
                l2_enabled=False,  # Disable Redis for demo
                strategy=CacheStrategy.HYBRID,
                enable_predictive_caching=True,
                enable_compression=True,
                enable_metrics=True
            )
            self.cache = AdvancedCache(cache_config)
            await self.cache.start()
            logger.info("Advanced cache initialized")

            # Initialize resilience manager
            self.resilience_manager = get_resilience_manager()
            
            # Create circuit breaker for email service
            cb_config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=30,
                enable_monitoring=True
            )
            email_circuit_breaker = self.resilience_manager.create_circuit_breaker(
                "email_service", cb_config
            )

            # Create retry handler
            retry_config = RetryConfig(
                max_retries=3,
                strategy=RetryStrategy.EXPONENTIAL,
                backoff_factor=2.0,
                timeout=10.0
            )
            email_retry_handler = self.resilience_manager.create_retry_handler(
                "email_retry", retry_config
            )

            # Initialize performance optimizer
            perf_config = OptimizationConfig(
                max_memory_usage=0.8,
                cache_size=1000,
                batch_size=64,
                max_concurrent_tasks=10,
                enable_caching=True,
                enable_memory_optimization=True,
                enable_batch_processing=True
            )
            self.performance_optimizer = OptimizedPerformanceOptimizer(perf_config)

            # Initialize advanced optimizer with ML
            ml_config = AdvancedOptimizationConfig(
                enable_ml_optimization=True,
                enable_predictive_caching=True,
                enable_adaptive_batching=True,
                enable_intelligent_resource_management=True,
                enable_performance_prediction=True,
                ml_model_path="models/demo_optimization_model.pkl"
            )
            self.advanced_optimizer = AdvancedOptimizer(ml_config)

            # Initialize intelligent monitor
            monitor_config = MonitoringConfig(
                monitoring_interval=2,  # Faster for demo
                alert_threshold=0.8,
                auto_optimization_enabled=True,
                enable_real_time_alerts=True,
                enable_performance_tracking=True,
                enable_resource_monitoring=True,
                enable_ml_insights=True
            )
            self.intelligent_monitor = IntelligentMonitor(
                monitor_config,
                self.performance_optimizer,
                self.advanced_optimizer
            )

            # Add demo callbacks
            self.intelligent_monitor.add_alert_callback(self._demo_alert_handler)
            self.intelligent_monitor.add_optimization_callback(self._demo_optimization_handler)

            logger.info("Enhanced demo environment setup completed")
            return True

        except Exception as e:
            logger.error(f"Error setting up enhanced demo environment: {e}")
            return False

    async def _demo_alert_handler(self, alert):
        """Demo alert handler"""
        logger.info(f"🚨 ALERT: {alert.severity.value} - {alert.message}")
        if alert.recommendations:
            logger.info(f"Recommendations: {alert.recommendations}")

    async def _demo_optimization_handler(self, action):
        """Demo optimization action handler"""
        logger.info(f"⚡ OPTIMIZATION: {action.action_type} - {action.description}")
        if action.parameters:
            logger.info(f"Parameters: {action.parameters}")

    def create_demo_data(self):
        """Create demo data for testing"""
        logger.info("Creating demo data...")

        # Create demo sequences
        sequences = []
        for i in range(5):
            sequence = EmailSequence(
                name=f"Demo Sequence {i+1}",
                description=f"Demo sequence for testing enhanced features {i+1}",
                target_audience="demo_users",
                goals=["engagement", "conversion"],
                tone="professional"
            )
            sequences.append(sequence)

        # Create demo subscribers
        subscribers = []
        for i in range(10):
            subscriber = Subscriber(
                email=f"user{i+1}@demo.com",
                first_name=f"User{i+1}",
                last_name="Demo",
                status="active"
            )
            subscribers.append(subscriber)

        # Create demo templates
        templates = []
        for i in range(3):
            template = EmailTemplate(
                name=f"Demo Template {i+1}",
                subject=f"Demo Email {i+1}",
                html_content=f"<h1>Demo Email {i+1}</h1><p>This is a demo email template.</p>",
                text_content=f"Demo Email {i+1}\n\nThis is a demo email template."
            )
            templates.append(template)

        self.demo_data = {
            "sequences": sequences,
            "subscribers": subscribers,
            "templates": templates
        }

        logger.info(f"Created {len(sequences)} sequences, {len(subscribers)} subscribers, {len(templates)} templates")

    async def demo_unified_configuration(self):
        """Demo unified configuration management"""
        logger.info("=== Demo: Unified Configuration Management ===")

        # Show configuration sections
        logger.info(f"Environment: {self.unified_config.environment.value}")
        logger.info(f"Database: {self.unified_config.database.connection_string}")
        logger.info(f"Redis: {self.unified_config.redis.connection_string}")
        logger.info(f"Cache Strategy: {self.unified_config.performance.strategy}")
        logger.info(f"Security Level: {self.unified_config.security.security_level}")

        # Show environment-specific defaults
        if self.unified_config.environment == Environment.DEVELOPMENT:
            logger.info("Development environment detected - debug logging enabled")
        elif self.unified_config.environment == Environment.PRODUCTION:
            logger.info("Production environment detected - optimized settings applied")

        # Demo configuration updates
        logger.info("Updating configuration...")
        self.unified_config.update_section("performance", batch_size=128)
        logger.info(f"Updated batch size to: {self.unified_config.performance.batch_size}")

        # Show configuration export
        config_dict = self.unified_config.get_config_dict()
        logger.info(f"Configuration exported with {len(config_dict)} sections")

    async def demo_advanced_caching(self):
        """Demo advanced caching system"""
        logger.info("=== Demo: Advanced Caching System ===")

        # Test basic caching
        test_key = "demo:test_data"
        test_data = {"message": "Hello from enhanced caching!", "timestamp": datetime.utcnow().isoformat()}
        
        # Set data in cache
        await self.cache.set(test_key, test_data, ttl=300)
        logger.info(f"Cached data with key: {test_key}")

        # Get data from cache
        cached_data = await self.cache.get(test_key)
        if cached_data:
            logger.info(f"Retrieved from cache: {cached_data['message']}")
        else:
            logger.info("Cache miss - data not found")

        # Test predictive caching
        predictions = self.cache.get_predictions()
        logger.info(f"Cache predictions: {len(predictions)} items")

        # Show cache metrics
        metrics = self.cache.get_metrics()
        logger.info(f"Cache metrics - Hits: {metrics['hits']}, Misses: {metrics['misses']}, Hit Rate: {metrics['hit_rate']:.2%}")

        # Test cache events
        def cache_event_handler(event_type, key, **kwargs):
            logger.info(f"Cache event: {event_type} for key: {key}")

        self.cache.add_event_callback("hit", cache_event_handler)
        self.cache.add_event_callback("miss", cache_event_handler)

        # Trigger some cache operations
        await self.cache.get("non_existent_key")  # Should trigger miss event
        await self.cache.get(test_key)  # Should trigger hit event

    async def demo_enhanced_error_handling(self):
        """Demo enhanced error handling and resilience"""
        logger.info("=== Demo: Enhanced Error Handling and Resilience ===")

        # Demo circuit breaker
        async def failing_operation():
            """Simulate a failing operation"""
            raise Exception("Simulated failure for demo")

        async def successful_operation():
            """Simulate a successful operation"""
            return "Operation successful"

        # Test circuit breaker with failing operation
        logger.info("Testing circuit breaker with failing operation...")
        try:
            await self.resilience_manager.execute_with_resilience(
                "failing_operation",
                failing_operation,
                circuit_breaker_name="email_service"
            )
        except Exception as e:
            logger.info(f"Circuit breaker caught error: {e}")

        # Test circuit breaker with successful operation
        logger.info("Testing circuit breaker with successful operation...")
        try:
            result = await self.resilience_manager.execute_with_resilience(
                "successful_operation",
                successful_operation,
                circuit_breaker_name="email_service"
            )
            logger.info(f"Successful operation result: {result}")
        except Exception as e:
            logger.info(f"Unexpected error: {e}")

        # Show resilience metrics
        metrics = self.resilience_manager.get_metrics()
        logger.info(f"Resilience metrics - Total errors: {metrics['resilience_metrics']['total_errors']}")
        logger.info(f"Circuit breaker state: {metrics['circuit_breakers']['email_service']['state']}")

        # Demo error tracking
        error_tracker = self.resilience_manager.error_tracker
        error_id = error_tracker.track_error(
            error=Exception("Demo error"),
            operation="demo_operation",
            component="demo_component",
            severity=ErrorSeverity.MEDIUM
        )
        logger.info(f"Tracked error with ID: {error_id}")

        # Show error analytics
        analytics = error_tracker.get_error_analytics(hours=1)
        logger.info(f"Error analytics - Total errors: {analytics['total_errors']}")

    async def demo_performance_optimization(self):
        """Demo performance optimization features"""
        logger.info("=== Demo: Performance Optimization ===")

        # Test performance optimization
        sequences = self.demo_data["sequences"]
        subscribers = self.demo_data["subscribers"]
        templates = self.demo_data["templates"]

        logger.info(f"Optimizing processing for {len(sequences)} sequences, {len(subscribers)} subscribers")

        # Run optimization
        result = await self.performance_optimizer.optimize_sequence_processing(
            sequences, subscribers, templates
        )

        logger.info(f"Optimization completed: {result['optimization_applied']}")
        if 'metrics' in result:
            metrics = result['metrics']
            logger.info(f"Performance metrics - Processing time: {metrics.get('avg_processing_time', 0):.2f}s")

    async def demo_ml_optimization(self):
        """Demo ML-based optimization"""
        logger.info("=== Demo: ML-Based Optimization ===")

        # Test ML performance prediction
        sequences = self.demo_data["sequences"]
        subscribers = self.demo_data["subscribers"]
        templates = self.demo_data["templates"]

        current_metrics = {
            'memory_usage': 0.5,
            'cpu_usage': 0.3,
            'batch_size': 64
        }

        prediction = self.advanced_optimizer.predict_performance(
            sequences, subscribers, templates, current_metrics
        )

        logger.info(f"ML Performance Prediction:")
        logger.info(f"  Predicted throughput: {prediction.predicted_throughput:.2f}")
        logger.info(f"  Confidence: {prediction.confidence:.2%}")
        logger.info(f"  Recommendations: {prediction.recommendations}")

        # Test ML optimization
        optimization_result = await self.advanced_optimizer.optimize_with_ml(
            sequences, subscribers, templates, current_metrics
        )

        logger.info(f"ML Optimization completed: {optimization_result['success']}")
        if 'optimizations_applied' in optimization_result:
            logger.info(f"Optimizations applied: {optimization_result['optimizations_applied']}")

    async def demo_intelligent_monitoring(self):
        """Demo intelligent monitoring system"""
        logger.info("=== Demo: Intelligent Monitoring ===")

        # Start monitoring
        await self.intelligent_monitor.start_monitoring()
        logger.info("Intelligent monitoring started")

        # Let it run for a few seconds to collect data
        await asyncio.sleep(5)

        # Get monitoring insights
        insights = await self.intelligent_monitor.get_ml_insights()
        logger.info(f"ML Insights: {len(insights)} insights generated")

        # Get system health
        health = await self.intelligent_monitor.get_system_health()
        logger.info(f"System health: {health['status']}")

        # Stop monitoring
        await self.intelligent_monitor.stop_monitoring()
        logger.info("Intelligent monitoring stopped")

    async def demo_integrated_features(self):
        """Demo integrated features working together"""
        logger.info("=== Demo: Integrated Features ===")

        # Simulate a complex operation using all features
        async def complex_operation():
            """Simulate a complex operation that uses caching, resilience, and monitoring"""
            
            # Use cache for intermediate results
            cache_key = "complex_operation:result"
            cached_result = await self.cache.get(cache_key)
            
            if cached_result:
                logger.info("Using cached result for complex operation")
                return cached_result

            # Simulate processing time
            await asyncio.sleep(1)
            
            result = {
                "operation": "complex_demo",
                "timestamp": datetime.utcnow().isoformat(),
                "processed_items": len(self.demo_data["sequences"]),
                "status": "completed"
            }

            # Cache the result
            await self.cache.set(cache_key, result, ttl=600)
            
            return result

        # Execute with full resilience
        try:
            result = await self.resilience_manager.execute_with_resilience(
                "complex_operation",
                complex_operation,
                circuit_breaker_name="email_service",
                retry_handler_name="email_retry"
            )
            logger.info(f"Complex operation completed: {result}")
        except Exception as e:
            logger.error(f"Complex operation failed: {e}")

    async def demo_advanced_analytics(self):
        """Demo advanced analytics and metrics"""
        logger.info("=== Demo: Advanced Analytics ===")

        # Get cache analytics
        cache_metrics = self.cache.get_metrics()
        logger.info("Cache Analytics:")
        logger.info(f"  Hit Rate: {cache_metrics['hit_rate']:.2%}")
        logger.info(f"  Avg Response Time: {cache_metrics['avg_response_time']:.3f}s")
        logger.info(f"  Memory Usage: {cache_metrics['memory_usage']:.2%}")

        # Get resilience analytics
        resilience_metrics = self.resilience_manager.get_metrics()
        logger.info("Resilience Analytics:")
        logger.info(f"  Total Errors: {resilience_metrics['resilience_metrics']['total_errors']}")
        logger.info(f"  Error Rate: {resilience_metrics['resilience_metrics']['error_rate']:.2%}")
        logger.info(f"  Successful Retries: {resilience_metrics['resilience_metrics']['successful_retries']}")

        # Get error analytics
        error_tracker = self.resilience_manager.error_tracker
        error_analytics = error_tracker.get_error_analytics(hours=1)
        logger.info("Error Analytics:")
        logger.info(f"  Total Errors: {error_analytics['total_errors']}")
        logger.info(f"  Error Rate per Hour: {error_analytics['error_rate_per_hour']:.2f}")
        logger.info(f"  Most Common Errors: {len(error_analytics['most_common_errors'])}")

    async def run_complete_demo(self):
        """Run the complete enhanced features demo"""
        logger.info("🚀 Starting Enhanced Features Demo")
        logger.info("=" * 50)

        try:
            # Setup environment
            if not await self.setup_demo_environment():
                logger.error("Failed to setup demo environment")
                return

            # Create demo data
            self.create_demo_data()

            # Run individual demos
            await self.demo_unified_configuration()
            await asyncio.sleep(1)

            await self.demo_advanced_caching()
            await asyncio.sleep(1)

            await self.demo_enhanced_error_handling()
            await asyncio.sleep(1)

            await self.demo_performance_optimization()
            await asyncio.sleep(1)

            await self.demo_ml_optimization()
            await asyncio.sleep(1)

            await self.demo_intelligent_monitoring()
            await asyncio.sleep(1)

            await self.demo_integrated_features()
            await asyncio.sleep(1)

            await self.demo_advanced_analytics()

            logger.info("=" * 50)
            logger.info("✅ Enhanced Features Demo Completed Successfully!")
            logger.info("All enhanced features demonstrated:")
            logger.info("  ✓ Unified Configuration Management")
            logger.info("  ✓ Advanced Caching System")
            logger.info("  ✓ Enhanced Error Handling and Resilience")
            logger.info("  ✓ Performance Optimization")
            logger.info("  ✓ ML-Based Optimization")
            logger.info("  ✓ Intelligent Monitoring")
            logger.info("  ✓ Integrated Features")
            logger.info("  ✓ Advanced Analytics")

        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
        finally:
            # Cleanup
            if self.cache:
                await self.cache.stop()
            logger.info("Demo cleanup completed")


async def main():
    """Main demo function"""
    demo = EnhancedFeaturesDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main()) 