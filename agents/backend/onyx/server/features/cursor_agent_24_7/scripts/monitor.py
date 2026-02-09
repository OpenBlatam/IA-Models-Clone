"""
Monitor - Monitor en tiempo real
=================================

Script para monitorear el agente en tiempo real.
"""

import asyncio
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    print("❌ httpx required: pip install httpx")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich import box
    USE_RICH = True
except ImportError:
    USE_RICH = False
    print("⚠️  rich not available, using basic output")


API_URL = "http://localhost:8024"


def create_status_table(status_data):
    """Crear tabla de estado"""
    if not USE_RICH:
        return None
    
    table = Table(title="Agent Status", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Status", status_data.get("status", "unknown").upper())
    table.add_row("Running", "✅ Yes" if status_data.get("running") else "❌ No")
    table.add_row("Total Tasks", str(status_data.get("tasks_total", 0)))
    table.add_row("Pending", str(status_data.get("tasks_pending", 0)))
    table.add_row("Running", str(status_data.get("tasks_running", 0)))
    table.add_row("Completed", str(status_data.get("tasks_completed", 0)))
    table.add_row("Failed", str(status_data.get("tasks_failed", 0)))
    
    return table


def create_metrics_panel(metrics_data):
    """Crear panel de métricas"""
    if not USE_RICH:
        return None
    
    if not metrics_data or "uptime_human" not in metrics_data:
        return None
    
    content = f"""
Uptime: {metrics_data.get('uptime_human', 'N/A')}
Total Metrics: {metrics_data.get('total_metrics_recorded', 0)}
"""
    
    return Panel(content, title="Metrics", border_style="blue")


async def fetch_status():
    """Obtener estado del agente"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_URL}/api/status")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        return {"error": str(e)}
    return None


async def monitor_loop(interval: float = 2.0):
    """Loop de monitoreo"""
    console = Console() if USE_RICH else None
    
    while True:
        try:
            status = await fetch_status()
            
            if "error" in status:
                if console:
                    console.print(f"[red]Error: {status['error']}[/red]")
                else:
                    print(f"Error: {status['error']}")
            else:
                if console:
                    table = create_status_table(status)
                    metrics_panel = create_metrics_panel(status.get("metrics"))
                    
                    console.clear()
                    console.print(table)
                    if metrics_panel:
                        console.print(metrics_panel)
                else:
                    # Output básico
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status: {status.get('status')} | "
                          f"Tasks: {status.get('tasks_total', 0)} | "
                          f"Completed: {status.get('tasks_completed', 0)}")
            
            await asyncio.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
            break
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(5)


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Cursor Agent 24/7")
    parser.add_argument("--interval", type=float, default=2.0, help="Update interval in seconds")
    parser.add_argument("--url", default=API_URL, help="API URL")
    
    args = parser.parse_args()
    global API_URL
    API_URL = args.url
    
    print("📊 Monitoring Cursor Agent 24/7...")
    print(f"API URL: {API_URL}")
    print("Press Ctrl+C to stop\n")
    
    asyncio.run(monitor_loop(args.interval))


if __name__ == "__main__":
    main()



