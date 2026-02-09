"""
Example usage of the API endpoints.
This demonstrates how to interact with the Ultimate Quantum AI API.
"""
import httpx
import asyncio
from typing import Dict, Any


API_BASE_URL = "http://localhost:8000"
API_KEY = None  # Set if ENFORCE_AUTH=true


def get_headers() -> Dict[str, str]:
    """Get request headers."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


async def example_health_check():
    """Example: Check API health."""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE_URL}/health", headers=get_headers())
        print(f"Health Status: {response.status_code}")
        print(f"Response: {response.json()}")


async def example_create_task():
    """Example: Create a background task."""
    async with httpx.AsyncClient() as client:
        task_data = {
            "func_name": "process_data",
            "args": ["arg1", "arg2"],
            "kwargs": {"param1": "value1"}
        }
        response = await client.post(
            f"{API_BASE_URL}/api/v1/tasks",
            json=task_data,
            headers=get_headers()
        )
        if response.status_code == 200:
            task_info = response.json()
            task_id = task_info.get("task_id")
            print(f"Task created: {task_id}")
            return task_id
        else:
            print(f"Error creating task: {response.text}")
            return None


async def example_get_task_status(task_id: str):
    """Example: Get task status."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/api/v1/tasks/{task_id}",
            headers=get_headers()
        )
        if response.status_code == 200:
            task_status = response.json()
            print(f"Task Status: {task_status.get('status')}")
            print(f"Result: {task_status.get('result')}")
            return task_status
        else:
            print(f"Error getting task: {response.text}")
            return None


async def example_list_tasks():
    """Example: List all tasks."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/api/v1/tasks",
            headers=get_headers()
        )
        if response.status_code == 200:
            tasks = response.json()
            print(f"Total tasks: {len(tasks)}")
            for task in tasks:
                print(f"  - {task.get('task_id')}: {task.get('status')}")
            return tasks
        else:
            print(f"Error listing tasks: {response.text}")
            return []


async def example_get_capabilities():
    """Example: Get system capabilities."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/capabilities",
            headers=get_headers()
        )
        if response.status_code == 200:
            capabilities = response.json()
            print("System Capabilities:")
            for system_name, system_caps in capabilities.get("systems", {}).items():
                print(f"  - {system_name}: {len(system_caps)} capabilities")
            return capabilities
        else:
            print(f"Error getting capabilities: {response.text}")
            return None


async def example_get_metrics():
    """Example: Get system metrics."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_BASE_URL}/metrics",
            headers=get_headers()
        )
        if response.status_code == 200:
            metrics = response.json()
            overall = metrics.get("overall_metrics", {})
            print(f"Total Systems: {overall.get('total_systems')}")
            print(f"Active Systems: {overall.get('active_systems')}")
            return metrics
        else:
            print(f"Error getting metrics: {response.text}")
            return None


async def main():
    """Run all examples."""
    print("=" * 60)
    print("Ultimate Quantum AI API - Usage Examples")
    print("=" * 60)
    
    # Health check
    print("\n1. Health Check")
    await example_health_check()
    
    # Capabilities
    print("\n2. Get Capabilities")
    await example_get_capabilities()
    
    # Metrics
    print("\n3. Get Metrics")
    await example_get_metrics()
    
    # Create task
    print("\n4. Create Task")
    task_id = await example_create_task()
    
    if task_id:
        # Wait a bit
        await asyncio.sleep(1)
        
        # Get task status
        print("\n5. Get Task Status")
        await example_get_task_status(task_id)
        
        # List all tasks
        print("\n6. List All Tasks")
        await example_list_tasks()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())


