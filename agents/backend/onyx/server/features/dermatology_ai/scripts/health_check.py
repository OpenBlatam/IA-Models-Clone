#!/usr/bin/env python3
"""
Health Check Script for Dermatology AI Service
Can be used by orchestrators (Kubernetes, Docker, etc.)
"""

import sys
import asyncio
import aiohttp
import argparse
from typing import Optional


async def check_health(
    url: str = "http://localhost:8006",
    timeout: int = 5,
    detailed: bool = False
) -> bool:
    """
    Check service health
    
    Args:
        url: Service URL
        timeout: Request timeout in seconds
        detailed: Return detailed health information
        
    Returns:
        True if healthy, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            endpoint = f"{url}/health" if not detailed else f"{url}/dermatology/health/detailed"
            
            async with session.get(endpoint, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    
                    if detailed:
                        print(f"Health Status: {data.get('status', 'unknown')}")
                        print(f"Version: {data.get('version', 'unknown')}")
                        print(f"Uptime: {data.get('uptime', 'unknown')}")
                        
                        # Check dependencies
                        dependencies = data.get('dependencies', {})
                        print("\nDependencies:")
                        for dep, status in dependencies.items():
                            status_icon = "✅" if status.get('status') == 'healthy' else "❌"
                            print(f"  {status_icon} {dep}: {status.get('status', 'unknown')}")
                    else:
                        print(f"✅ Service is healthy: {data.get('status', 'ok')}")
                    
                    return True
                else:
                    print(f"❌ Service returned status {resp.status}")
                    return False
                    
    except asyncio.TimeoutError:
        print(f"❌ Health check timeout after {timeout}s")
        return False
    except aiohttp.ClientError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


async def check_readiness(url: str = "http://localhost:8006", timeout: int = 5) -> bool:
    """Check service readiness"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{url}/health",
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                return resp.status == 200
    except Exception:
        return False


async def check_liveness(url: str = "http://localhost:8006", timeout: int = 5) -> bool:
    """Check service liveness"""
    return await check_readiness(url, timeout)


def main():
    parser = argparse.ArgumentParser(description="Health check for Dermatology AI Service")
    parser.add_argument(
        "--url",
        default="http://localhost:8006",
        help="Service URL (default: http://localhost:8006)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Timeout in seconds (default: 5)"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed health information"
    )
    parser.add_argument(
        "--readiness",
        action="store_true",
        help="Check readiness (exit code 0 if ready)"
    )
    parser.add_argument(
        "--liveness",
        action="store_true",
        help="Check liveness (exit code 0 if alive)"
    )
    
    args = parser.parse_args()
    
    if args.readiness:
        healthy = asyncio.run(check_readiness(args.url, args.timeout))
    elif args.liveness:
        healthy = asyncio.run(check_liveness(args.url, args.timeout))
    else:
        healthy = asyncio.run(check_health(args.url, args.timeout, args.detailed))
    
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()















