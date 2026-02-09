#!/usr/bin/env python3
"""Health check script for the service."""

import sys
import asyncio
import httpx
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def check_health(base_url: str = "http://localhost:8025"):
    """Check service health."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            # Check liveness
            response = await client.get(f"{base_url}/health/live")
            if response.status_code == 200:
                print("✓ Liveness check passed")
            else:
                print(f"✗ Liveness check failed: {response.status_code}")
                return False
            
            # Check health
            response = await client.get(f"{base_url}/health/")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Health check passed: {data.get('status')}")
                print(f"  Dependencies: {data.get('dependencies', {})}")
            else:
                print(f"✗ Health check failed: {response.status_code}")
                return False
            
            # Check readiness
            response = await client.get(f"{base_url}/health/ready")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Readiness check passed: {data.get('status')}")
                return True
            else:
                print(f"✗ Readiness check failed: {response.status_code}")
                if response.status_code == 503:
                    try:
                        data = response.json()
                        print(f"  Detail: {data.get('detail')}")
                    except:
                        print(f"  Response: {response.text}")
                return False
                
        except httpx.ConnectError:
            print(f"✗ Cannot connect to {base_url}")
            print("  Make sure the service is running")
            return False
        except Exception as e:
            print(f"✗ Error checking health: {e}")
            return False


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8025"
    success = asyncio.run(check_health(base_url))
    sys.exit(0 if success else 1)

