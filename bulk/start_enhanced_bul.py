"""
BUL - Business Universal Language (Enhanced Startup Script)
===========================================================

Enhanced startup script that integrates all improvements.
"""

import subprocess
import sys
import time
import os
import signal
import threading
from pathlib import Path
import json
import requests
from datetime import datetime

class BULSystemManager:
    """Enhanced BUL system manager with all improvements."""
    
    def __init__(self):
        self.processes = {}
        self.services = {
            "main_api": {
                "script": "bul_divine_ai.py",
                "port": 8000,
                "description": "Main BUL Divine AI API"
            },
            "performance_optimizer": {
                "script": "bul_performance_optimizer.py",
                "port": 8001,
                "description": "Performance monitoring and optimization"
            },
            "advanced_dashboard": {
                "script": "bul_advanced_dashboard.py",
                "port": 8050,
                "description": "Advanced real-time dashboard"
            }
        }
        self.system_status = {
            "started_at": None,
            "services_running": 0,
            "total_services": len(self.services),
            "health_status": "unknown"
        }
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        if service_name not in self.services:
            print(f"❌ Unknown service: {service_name}")
            return False
        
        service = self.services[service_name]
        script_path = Path(service["script"])
        
        if not script_path.exists():
            print(f"❌ Script not found: {script_path}")
            return False
        
        try:
            print(f"🚀 Starting {service['description']}...")
            
            # Start the service
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = process
            
            # Wait a moment and check if it's still running
            time.sleep(2)
            if process.poll() is None:
                print(f"✅ {service['description']} started successfully (PID: {process.pid})")
                return True
            else:
                stdout, stderr = process.communicate()
                print(f"❌ Failed to start {service['description']}")
                print(f"Error: {stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error starting {service['description']}: {e}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.processes:
            print(f"⚠️ Service {service_name} is not running")
            return False
        
        try:
            process = self.processes[service_name]
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            
            del self.processes[service_name]
            print(f"🛑 {service_name} stopped")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping {service_name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all services."""
        print("=" * 60)
        print("🚀 BUL Enhanced System Startup")
        print("=" * 60)
        
        self.system_status["started_at"] = datetime.now()
        success_count = 0
        
        for service_name in self.services:
            if self.start_service(service_name):
                success_count += 1
                self.system_status["services_running"] = success_count
        
        print(f"\n📊 Started {success_count}/{len(self.services)} services")
        
        if success_count == len(self.services):
            self.system_status["health_status"] = "healthy"
            print("✅ All services started successfully!")
            return True
        else:
            self.system_status["health_status"] = "degraded"
            print("⚠️ Some services failed to start")
            return False
    
    def stop_all_services(self):
        """Stop all services."""
        print("\n🛑 Stopping all services...")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        print("✅ All services stopped")
    
    def check_service_health(self, service_name: str) -> bool:
        """Check if a service is healthy."""
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        try:
            response = requests.get(f"http://localhost:{service['port']}/", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def monitor_services(self):
        """Monitor services and restart if needed."""
        def monitor_loop():
            while True:
                try:
                    for service_name in list(self.processes.keys()):
                        process = self.processes[service_name]
                        
                        # Check if process is still running
                        if process.poll() is not None:
                            print(f"⚠️ Service {service_name} stopped unexpectedly")
                            
                            # Try to restart
                            if self.start_service(service_name):
                                print(f"✅ Service {service_name} restarted")
                            else:
                                print(f"❌ Failed to restart {service_name}")
                    
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("🔍 Service monitoring started")
    
    def get_system_status(self) -> dict:
        """Get current system status."""
        status = self.system_status.copy()
        status["services"] = {}
        
        for service_name, service in self.services.items():
            is_running = service_name in self.processes
            is_healthy = self.check_service_health(service_name) if is_running else False
            
            status["services"][service_name] = {
                "running": is_running,
                "healthy": is_healthy,
                "port": service["port"],
                "description": service["description"]
            }
        
        return status
    
    def print_status(self):
        """Print current system status."""
        status = self.get_system_status()
        
        print("\n" + "=" * 60)
        print("📊 BUL System Status")
        print("=" * 60)
        
        print(f"🕐 Started at: {status['started_at']}")
        print(f"🏥 Health status: {status['health_status']}")
        print(f"🔧 Services running: {status['services_running']}/{status['total_services']}")
        
        print("\n📋 Service Details:")
        for service_name, service_status in status["services"].items():
            status_icon = "✅" if service_status["healthy"] else "❌" if service_status["running"] else "⏹️"
            print(f"  {status_icon} {service_name}: {service_status['description']}")
            print(f"     Port: {service_status['port']}")
            print(f"     Status: {'Healthy' if service_status['healthy'] else 'Running' if service_status['running'] else 'Stopped'}")
        
        print("\n🌐 Access URLs:")
        print(f"  📡 Main API: http://localhost:8000")
        print(f"  📊 Dashboard: http://localhost:8050")
        print(f"  ⚡ Performance: http://localhost:8001")
        print(f"  📚 API Docs: http://localhost:8000/docs")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    if 'manager' in globals():
        manager.stop_all_services()
    sys.exit(0)

def main():
    """Main function."""
    global manager
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create system manager
    manager = BULSystemManager()
    
    try:
        # Start all services
        if manager.start_all_services():
            # Start monitoring
            manager.monitor_services()
            
            # Print initial status
            manager.print_status()
            
            print("\n🎉 BUL Enhanced System is running!")
            print("Press Ctrl+C to stop all services")
            
            # Keep running
            while True:
                time.sleep(60)
                manager.print_status()
        
        else:
            print("❌ Failed to start all services")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
        manager.stop_all_services()
    
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        manager.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()
