#!/usr/bin/env python3
"""
Script de monitoreo de salud para Robot Movement AI v2.0
Monitorea el sistema y envía alertas
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


async def check_health(url: str = "http://localhost:8010") -> Dict[str, Any]:
    """Verificar salud del sistema"""
    if not HTTPX_AVAILABLE:
        return {"status": "error", "message": "httpx not available"}
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            response.raise_for_status()
            return {"status": "healthy", "data": response.json()}
    except httpx.TimeoutException:
        return {"status": "timeout", "message": "Health check timed out"}
    except httpx.HTTPStatusError as e:
        return {"status": "unhealthy", "message": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def check_metrics(url: str = "http://localhost:8010") -> Dict[str, Any]:
    """Verificar métricas"""
    if not HTTPX_AVAILABLE:
        return {"status": "error"}
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health/metrics")
            return {"status": "ok", "metrics": response.text[:500]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def monitor_loop(
    url: str = "http://localhost:8010",
    interval: int = 30,
    alert_threshold: int = 3
):
    """Loop de monitoreo continuo"""
    consecutive_failures = 0
    
    print(f"Starting health monitor for {url}")
    print(f"Check interval: {interval} seconds")
    print("=" * 60)
    
    while True:
        try:
            # Health check
            health_result = await check_health(url)
            
            if health_result["status"] == "healthy":
                consecutive_failures = 0
                data = health_result.get("data", {})
                status = data.get("status", "unknown")
                uptime = data.get("uptime_seconds", 0)
                
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✓ Healthy - Status: {status}, Uptime: {uptime:.0f}s")
            else:
                consecutive_failures += 1
                message = health_result.get("message", "Unknown error")
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ✗ Unhealthy - {message} ({consecutive_failures}/{alert_threshold})")
                
                if consecutive_failures >= alert_threshold:
                    print(f"⚠️  ALERT: {consecutive_failures} consecutive failures!")
                    # Aquí se podría enviar alerta real
                    from core.architecture.alerts import send_alert, AlertLevel
                    await send_alert(
                        title="System Health Alert",
                        message=f"System has been unhealthy for {consecutive_failures} consecutive checks",
                        level=AlertLevel.CRITICAL
                    )
            
            await asyncio.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            break
        except Exception as e:
            print(f"Error in monitor loop: {e}")
            await asyncio.sleep(interval)


async def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Health monitor for Robot Movement AI")
    parser.add_argument("--url", default="http://localhost:8010", help="API URL")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--threshold", type=int, default=3, help="Alert threshold")
    parser.add_argument("--once", action="store_true", help="Run once instead of continuous")
    
    args = parser.parse_args()
    
    if args.once:
        # Ejecutar una vez
        health = await check_health(args.url)
        print(f"Health Status: {health['status']}")
        if health.get("data"):
            print(f"Details: {health['data']}")
    else:
        # Monitoreo continuo
        await monitor_loop(args.url, args.interval, args.threshold)


if __name__ == "__main__":
    asyncio.run(main())




