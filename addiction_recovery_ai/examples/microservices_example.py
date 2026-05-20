"""
Microservices Architecture Example
Demonstrates service discovery, API Gateway, and event bus
"""

import asyncio
from microservices.service_discovery import get_service_registry, ServiceStatus
from microservices.service_client import get_service_client
from microservices.api_gateway import get_api_gateway, RateLimitConfig
from microservices.event_bus import get_event_bus, EventType


async def example_service_discovery():
    """Example: Service discovery"""
    print("=== Service Discovery Example ===\n")
    
    registry = get_service_registry()
    
    # Register services
    print("Registering services...")
    registry.register(
        service_name="user-service",
        instance_id="user-1",
        host="localhost",
        port=8001,
        metadata={"version": "1.0.0", "region": "us-east-1"}
    )
    
    registry.register(
        service_name="user-service",
        instance_id="user-2",
        host="localhost",
        port=8002,
        metadata={"version": "1.0.0", "region": "us-east-1"}
    )
    
    registry.register(
        service_name="notification-service",
        instance_id="notif-1",
        host="localhost",
        port=8003
    )
    
    # Discover services
    print("\nDiscovering services...")
    user_instances = registry.discover("user-service", healthy_only=True)
    print(f"User service instances: {len(user_instances)}")
    for instance in user_instances:
        print(f"  - {instance.instance_id}: {instance.url}")
    
    # Get instance with load balancing
    instance = registry.get_instance("user-service", strategy="round_robin")
    print(f"\nSelected instance: {instance.url}")
    
    # Service info
    info = registry.get_service_info("user-service")
    print(f"\nService info: {info}\n")


async def example_service_client():
    """Example: Inter-service communication"""
    print("=== Service Client Example ===\n")
    
    # Register service first
    registry = get_service_registry()
    registry.register(
        service_name="user-service",
        instance_id="user-1",
        host="localhost",
        port=8001
    )
    
    # Get service client
    client = get_service_client("user-service")
    
    print("Making requests to user-service...")
    
    try:
        # GET request
        print("GET /users/123")
        # user = await client.get("/users/123")
        # print(f"Response: {user}")
        
        # POST request
        print("POST /users")
        # result = await client.post("/users", json={"name": "John"})
        # print(f"Response: {result}")
        
        print("(Note: Actual requests would require running service)\n")
        
    except Exception as e:
        print(f"Error: {str(e)}\n")
    
    # Close client
    await client.close()


async def example_api_gateway():
    """Example: API Gateway"""
    print("=== API Gateway Example ===\n")
    
    gateway = get_api_gateway()
    
    # Register routes
    print("Registering routes...")
    gateway.register_route(
        path_prefix="/api/users",
        service_name="user-service",
        rate_limit=RateLimitConfig(requests_per_minute=60)
    )
    
    gateway.register_route(
        path_prefix="/api/recovery",
        service_name="recovery-service"
    )
    
    print("Routes registered:")
    for path, config in gateway.route_config.items():
        print(f"  {path} -> {config['service']}")
    
    print("\nAPI Gateway configured!\n")


async def example_event_bus():
    """Example: Event bus"""
    print("=== Event Bus Example ===\n")
    
    event_bus = get_event_bus()
    
    # Subscribe to events
    class NotificationHandler:
        async def handle(self, data):
            print(f"  📧 Notification: {data}")
    
    class AnalyticsHandler:
        async def handle(self, data):
            print(f"  📊 Analytics: {data}")
    
    event_bus.subscribe(EventType.MILESTONE_ACHIEVED.value, NotificationHandler())
    event_bus.subscribe(EventType.MILESTONE_ACHIEVED.value, AnalyticsHandler())
    
    # Publish events
    print("Publishing events...")
    
    await event_bus.publish_event(
        event_type=EventType.MILESTONE_ACHIEVED.value,
        source="recovery-service",
        data={
            "user_id": "123",
            "milestone": "30_days_sober",
            "date": "2024-01-15"
        }
    )
    
    await event_bus.publish_event(
        event_type=EventType.USER_CREATED.value,
        source="user-service",
        data={
            "user_id": "456",
            "email": "user@example.com"
        }
    )
    
    # List subscribers
    subscribers = event_bus.list_subscribers(EventType.MILESTONE_ACHIEVED.value)
    print(f"\nSubscribers to milestone.achieved: {subscribers}\n")


async def example_complete_flow():
    """Example: Complete microservices flow"""
    print("=== Complete Flow Example ===\n")
    
    # 1. Register services
    registry = get_service_registry()
    registry.register("user-service", "user-1", "localhost", 8001)
    registry.register("recovery-service", "recovery-1", "localhost", 8002)
    
    # 2. Configure API Gateway
    gateway = get_api_gateway()
    gateway.register_route("/api/users", "user-service")
    gateway.register_route("/api/recovery", "recovery-service")
    
    # 3. Setup event handlers
    event_bus = get_event_bus()
    
    class RecoveryEventHandler:
        async def handle(self, data):
            print(f"  ✅ Recovery event processed: {data.get('user_id')}")
    
    event_bus.subscribe("user.created", RecoveryEventHandler())
    
    # 4. Simulate flow
    print("Simulating microservices flow:")
    print("1. User created in user-service")
    print("2. Event published to event bus")
    print("3. Recovery service handles event")
    
    await event_bus.publish_event(
        event_type="user.created",
        source="user-service",
        data={"user_id": "789", "name": "Alice"}
    )
    
    print("\nFlow completed!\n")


async def main():
    """Run all examples"""
    print("=" * 60)
    print("Microservices Architecture Examples")
    print("=" * 60)
    print()
    
    await example_service_discovery()
    await example_service_client()
    await example_api_gateway()
    await example_event_bus()
    await example_complete_flow()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())















