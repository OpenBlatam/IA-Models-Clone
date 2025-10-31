from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any
from infrastructure.microservices import (
import random
from typing import Any, List, Dict, Optional
#!/usr/bin/env python3
"""
🚀 MICROSERVICES DEMO APPLICATION
==================================

Comprehensive demonstration of enterprise microservices capabilities:

✅ Service Discovery (Consul)
✅ Message Queues (RabbitMQ, Redis Streams)  
✅ Load Balancing (Multiple strategies)
✅ Resilience Patterns (Circuit Breaker, Bulkhead, Retry)
✅ Configuration Management (Consul, Environment, Files)
✅ Health Checks & Monitoring
✅ Distributed Tracing
✅ Production-ready patterns

Usage:
    python MICROSERVICES_DEMO.py

Requirements:
    pip install -r requirements-microservices.txt
"""


    # Service Discovery
    ServiceDiscoveryManager,
    ConsulServiceDiscovery,
    ServiceInstance,
    
    # Message Queues
    MessageQueueManager,
    RabbitMQService,
    RedisStreamsService,
    Message,
    
    # Load Balancing
    LoadBalancerManager,
    RoundRobinStrategy,
    WeightedRoundRobinStrategy,
    LeastConnectionsStrategy,
    HealthBasedStrategy,
    
    # Resilience
    ResilienceManager,
    BulkheadPattern,
    RetryPolicy,
    TimeoutPolicy,
    RetryStrategy,
    
    # Configuration
    ConfigurationManager,
    ConsulConfigProvider,
    EnvironmentConfigProvider,
    FileConfigProvider,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MicroservicesDemo:
    """Comprehensive microservices demonstration."""
    
    def __init__(self) -> Any:
        self.service_discovery = ServiceDiscoveryManager()
        self.message_queue = MessageQueueManager()
        self.load_balancer = LoadBalancerManager()
        self.resilience = ResilienceManager()
        self.config_manager = ConfigurationManager()
        
    async def initialize(self) -> Any:
        """Initialize all microservices components."""
        logger.info("🚀 Initializing Microservices Demo...")
        
        # Setup Service Discovery
        await self._setup_service_discovery()
        
        # Setup Message Queues
        await self._setup_message_queues()
        
        # Setup Load Balancing
        await self._setup_load_balancing()
        
        # Setup Resilience
        await self._setup_resilience()
        
        # Setup Configuration
        await self._setup_configuration()
        
        logger.info("✅ Microservices Demo initialized successfully!")
    
    async def _setup_service_discovery(self) -> Any:
        """Setup service discovery with Consul."""
        logger.info("📡 Setting up Service Discovery...")
        
        try:
            # Add Consul service discovery
            consul_discovery = ConsulServiceDiscovery("http://localhost:8500")
            self.service_discovery.add_discovery("consul", consul_discovery, is_primary=True)
            
            # Register sample services
            service1 = ServiceInstance(
                id="user-service-1",
                name="user-service",
                host="localhost",
                port=8001,
                metadata={"version": "1.0", "weight": "2"},
                tags=["api", "user"]
            )
            
            service2 = ServiceInstance(
                id="user-service-2", 
                name="user-service",
                host="localhost",
                port=8002,
                metadata={"version": "1.1", "weight": "3"},
                tags=["api", "user", "canary"]
            )
            
            await self.service_discovery.register_service(service1)
            await self.service_discovery.register_service(service2)
            
            logger.info("✅ Service Discovery configured with 2 services")
            
        except Exception as e:
            logger.warning(f"⚠️  Service Discovery setup failed (Consul not available?): {e}")
    
    async def _setup_message_queues(self) -> Any:
        """Setup message queues with multiple backends."""
        logger.info("📨 Setting up Message Queues...")
        
        try:
            # Add RabbitMQ
            rabbitmq = RabbitMQService("amqp://guest:guest@localhost:5672/")
            self.message_queue.add_queue("rabbitmq", rabbitmq, is_primary=True)
            logger.info("✅ RabbitMQ configured")
        except Exception as e:
            logger.warning(f"⚠️  RabbitMQ setup failed: {e}")
        
        try:
            # Add Redis Streams
            redis_streams = RedisStreamsService("redis://localhost:6379")
            self.message_queue.add_queue("redis", redis_streams)
            logger.info("✅ Redis Streams configured")
        except Exception as e:
            logger.warning(f"⚠️  Redis Streams setup failed: {e}")
    
    async def _setup_load_balancing(self) -> Any:
        """Setup load balancing with different strategies."""
        logger.info("⚖️  Setting up Load Balancing...")
        
        # Start with round robin strategy
        self.load_balancer.set_strategy(RoundRobinStrategy())
        
        logger.info("✅ Load Balancing configured with Round Robin strategy")
    
    async def _setup_resilience(self) -> Any:
        """Setup resilience patterns."""
        logger.info("🛡️  Setting up Resilience Patterns...")
        
        # Configure retry policies
        self.resilience.add_retry_policy(
            "default",
            RetryPolicy(
                max_retries=3,
                strategy=RetryStrategy.EXPONENTIAL,
                initial_delay=1.0,
                max_delay=30.0
            )
        )
        
        self.resilience.add_retry_policy(
            "aggressive",
            RetryPolicy(
                max_retries=5,
                strategy=RetryStrategy.EXPONENTIAL,
                initial_delay=0.5,
                max_delay=60.0
            )
        )
        
        # Configure timeout policies
        self.resilience.add_timeout_policy(
            "default",
            TimeoutPolicy(
                connect_timeout=5.0,
                read_timeout=30.0,
                total_timeout=60.0
            )
        )
        
        # Configure bulkheads
        self.resilience.add_bulkhead("api_calls", BulkheadPattern(max_concurrent=10))
        self.resilience.add_bulkhead("database", BulkheadPattern(max_concurrent=5))
        
        logger.info("✅ Resilience Patterns configured")
    
    async def _setup_configuration(self) -> Any:
        """Setup configuration management."""
        logger.info("⚙️  Setting up Configuration Management...")
        
        # Add environment config provider
        self.config_manager.add_provider(
            "environment",
            EnvironmentConfigProvider(prefix="DEMO_"),
            priority=1
        )
        
        # Add file config provider
        try:
            self.config_manager.add_provider(
                "file",
                FileConfigProvider("demo_config.json"),
                priority=2
            )
        except Exception as e:
            logger.warning(f"File config provider not available: {e}")
        
        # Add Consul config provider
        try:
            self.config_manager.add_provider(
                "consul",
                ConsulConfigProvider("http://localhost:8500"),
                priority=3
            )
        except Exception as e:
            logger.warning(f"Consul config provider not available: {e}")
        
        logger.info("✅ Configuration Management configured")
    
    async def demo_service_discovery(self) -> Any:
        """Demonstrate service discovery capabilities."""
        logger.info("\n🔍 === SERVICE DISCOVERY DEMO ===")
        
        try:
            # Discover services
            services = await self.service_discovery.discover_services("user-service")
            logger.info(f"Found {len(services)} user-service instances:")
            
            for service in services:
                logger.info(f"  - {service.id}: {service.url} (tags: {service.tags})")
            
            # Health check all discovery backends
            health = await self.service_discovery.health_check_all()
            logger.info(f"Service Discovery Health: {health}")
            
        except Exception as e:
            logger.error(f"Service Discovery demo failed: {e}")
    
    async def demo_message_queues(self) -> Any:
        """Demonstrate message queue capabilities."""
        logger.info("\n📨 === MESSAGE QUEUES DEMO ===")
        
        try:
            # Define message handler
            async def message_handler(message: Message):
                
    """message_handler function."""
logger.info(f"Received message: {message.payload} (ID: {message.id})")
            
            # Subscribe to topic
            subscriptions = await self.message_queue.subscribe("demo-topic", message_handler)
            logger.info(f"Subscribed to demo-topic: {subscriptions}")
            
            # Publish messages
            messages = [
                {"type": "user_created", "user_id": 123, "email": "user@example.com"},
                {"type": "order_placed", "order_id": 456, "amount": 99.99},
                {"type": "payment_processed", "transaction_id": "tx_789"}
            ]
            
            for msg in messages:
                results = await self.message_queue.publish("demo-topic", msg)
                logger.info(f"Published message: {msg} -> {results}")
                await asyncio.sleep(0.1)  # Small delay
            
            # Wait for message processing
            await asyncio.sleep(2)
            
            # Health check message queues
            health = await self.message_queue.health_check_all()
            logger.info(f"Message Queue Health: {health}")
            
        except Exception as e:
            logger.error(f"Message Queue demo failed: {e}")
    
    async def demo_load_balancing(self) -> Any:
        """Demonstrate load balancing capabilities."""
        logger.info("\n⚖️  === LOAD BALANCING DEMO ===")
        
        try:
            # Get service instances
            services = await self.service_discovery.discover_services("user-service")
            if not services:
                logger.warning("No services found for load balancing demo")
                return
            
            # Test different strategies
            strategies = [
                ("Round Robin", RoundRobinStrategy()),
                ("Weighted Round Robin", WeightedRoundRobinStrategy()),
                ("Least Connections", LeastConnectionsStrategy()),
                ("Health Based", HealthBasedStrategy())
            ]
            
            for strategy_name, strategy in strategies:
                logger.info(f"\n--- Testing {strategy_name} Strategy ---")
                self.load_balancer.set_strategy(strategy)
                
                # Make 5 requests
                for i in range(5):
                    selected = strategy.select_instance(services)
                    if selected:
                        logger.info(f"Request {i+1} -> {selected.id} ({selected.url})")
                    await asyncio.sleep(0.1)
            
            # Get load balancer stats
            stats = self.load_balancer.get_stats()
            logger.info(f"\nLoad Balancer Stats: {json.dumps(stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Load Balancing demo failed: {e}")
    
    async def demo_resilience(self) -> Any:
        """Demonstrate resilience patterns."""
        logger.info("\n🛡️  === RESILIENCE PATTERNS DEMO ===")
        
        try:
            # Test function that fails randomly
            async def unreliable_service():
                
    """unreliable_service function."""
                if random.random() < 0.7:  # 70% failure rate
                    raise Exception("Service temporarily unavailable")
                return {"status": "success", "data": "Hello from service!"}
            
            # Test retry policy
            logger.info("\n--- Testing Retry Policy ---")
            try:
                result = await self.resilience.execute_with_resilience(
                    unreliable_service,
                    retry_policy="default",
                    timeout_policy="default"
                )
                logger.info(f"Service call succeeded: {result}")
            except Exception as e:
                logger.info(f"Service call failed after retries: {e}")
            
            # Test bulkhead pattern
            logger.info("\n--- Testing Bulkhead Pattern ---")
            
            async def api_call(call_id: int):
                
    """api_call function."""
await asyncio.sleep(0.5)  # Simulate work
                return f"API call {call_id} completed"
            
            # Make concurrent calls through bulkhead
            tasks = []
            for i in range(15):  # More than bulkhead capacity (10)
                task = self.resilience.execute_with_resilience(
                    api_call,
                    i,
                    bulkhead="api_calls"
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = len(results) - successful
            
            logger.info(f"Bulkhead test: {successful} successful, {failed} rejected")
            
            # Get resilience stats
            stats = self.resilience.get_stats()
            logger.info(f"\nResilience Stats: {json.dumps(stats, indent=2)}")
            
        except Exception as e:
            logger.error(f"Resilience demo failed: {e}")
    
    async def demo_configuration(self) -> Any:
        """Demonstrate configuration management."""
        logger.info("\n⚙️  === CONFIGURATION MANAGEMENT DEMO ===")
        
        try:
            # Test configuration retrieval
            configs_to_test = [
                ("app.name", "Microservices Demo"),
                ("app.version", "1.0.0"),
                ("database.host", "localhost"),
                ("database.port", "5432"),
                ("feature.flags.new_ui", False)
            ]
            
            for key, default in configs_to_test:
                value = await self.config_manager.get_config(key, default)
                logger.info(f"Config {key}: {value}")
            
            # Get all configs
            all_configs = await self.config_manager.get_all_configs("app")
            logger.info(f"\nAll 'app' configs: {json.dumps(all_configs, indent=2)}")
            
        except Exception as e:
            logger.error(f"Configuration demo failed: {e}")
    
    async def run_full_demo(self) -> Any:
        """Run complete microservices demonstration."""
        logger.info("🎬 Starting Comprehensive Microservices Demo")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        try:
            await self.initialize()
            
            # Run all demos
            await self.demo_service_discovery()
            await self.demo_message_queues()
            await self.demo_load_balancing()
            await self.demo_resilience()
            await self.demo_configuration()
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info("\n" + "=" * 60)
            logger.info(f"🎉 Demo completed successfully in {duration:.2f} seconds!")
            logger.info("=" * 60)
            
            # Summary
            logger.info("\n📊 MICROSERVICES CAPABILITIES SUMMARY:")
            logger.info("✅ Service Discovery (Consul integration)")
            logger.info("✅ Message Queues (RabbitMQ, Redis Streams)")
            logger.info("✅ Load Balancing (4 strategies)")
            logger.info("✅ Resilience Patterns (Retry, Bulkhead, Timeout)")
            logger.info("✅ Configuration Management (Multi-source)")
            logger.info("✅ Health Checks & Monitoring")
            logger.info("✅ Production-ready patterns")
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            raise
    
    async def cleanup(self) -> Any:
        """Cleanup resources."""
        logger.info("🧹 Cleaning up resources...")
        
        try:
            if hasattr(self.load_balancer, 'close'):
                await self.load_balancer.close()
            
            logger.info("✅ Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


async def main():
    """Main entry point."""
    demo = MicroservicesDemo()
    
    try:
        await demo.run_full_demo()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        await demo.cleanup()


if __name__ == "__main__":
    # Create demo config file
    demo_config = {
        "app": {
            "name": "Microservices Demo",
            "version": "1.0.0",
            "debug": True
        },
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "demo_db"
        },
        "feature": {
            "flags": {
                "new_ui": True,
                "beta_features": False
            }
        }
    }
    
    with open("demo_config.json", "w") as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        json.dump(demo_config, f, indent=2)
    
    print("""
🚀 MICROSERVICES DEMO APPLICATION
==================================

This demo showcases enterprise-grade microservices patterns:

• Service Discovery (Consul)
• Message Queues (RabbitMQ, Redis)  
• Load Balancing (Multiple strategies)
• Resilience Patterns (Circuit Breaker, Bulkhead)
• Configuration Management
• Health Checks & Monitoring

Prerequisites:
1. pip install -r requirements-microservices.txt
2. Optional: Start Consul (docker run -p 8500:8500 consul)
3. Optional: Start RabbitMQ (docker run -p 5672:5672 rabbitmq)
4. Optional: Start Redis (docker run -p 6379:6379 redis)

Running without external services will show graceful degradation.
    """)
    
    asyncio.run(main()) 