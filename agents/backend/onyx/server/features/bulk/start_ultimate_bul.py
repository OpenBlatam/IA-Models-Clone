"""
BUL - Business Universal Language (Ultimate Enterprise Startup)
==============================================================

Ultimate startup script for enterprise BUL system with all advanced features.
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
import psutil

class BULUltimateManager:
    """Ultimate BUL enterprise system manager with all advanced features."""
    
    def __init__(self):
        self.processes = {}
        self.services = {
            "main_api": {
                "script": "bul_divine_ai.py",
                "port": 8000,
                "description": "Main BUL Divine AI API",
                "priority": "high"
            },
            "enterprise_system": {
                "script": "bul_enterprise.py",
                "port": 8002,
                "description": "Enterprise Management System",
                "priority": "high"
            },
            "external_integrations": {
                "script": "bul_integrations.py",
                "port": 8003,
                "description": "External API Integrations",
                "priority": "medium"
            },
            "advanced_security": {
                "script": "bul_security.py",
                "port": 8004,
                "description": "Advanced Security System",
                "priority": "high"
            },
            "auto_backup": {
                "script": "bul_backup.py",
                "port": 8005,
                "description": "Automatic Backup System",
                "priority": "medium"
            },
            "notifications": {
                "script": "bul_notifications.py",
                "port": 8006,
                "description": "Advanced Notifications System",
                "priority": "medium"
            },
            "performance_optimizer": {
                "script": "bul_performance_optimizer.py",
                "port": 8001,
                "description": "Performance Monitoring & Optimization",
                "priority": "low"
            },
            "advanced_dashboard": {
                "script": "bul_advanced_dashboard.py",
                "port": 8050,
                "description": "Advanced Real-time Dashboard",
                "priority": "low"
            }
        }
        self.system_status = {
            "started_at": None,
            "services_running": 0,
            "total_services": len(self.services),
            "health_status": "unknown",
            "enterprise_features": True,
            "security_enabled": True,
            "backup_enabled": True,
            "notifications_enabled": True,
            "system_resources": {}
        }
        self.startup_order = [
            "advanced_security",
            "main_api",
            "enterprise_system",
            "external_integrations",
            "auto_backup",
            "notifications",
            "performance_optimizer",
            "advanced_dashboard"
        ]
    
    def check_system_resources(self):
        """Check system resources."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.system_status["system_resources"] = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
            # Check if resources are sufficient
            if cpu_percent > 90:
                print("⚠️ High CPU usage detected")
            if memory.percent > 90:
                print("⚠️ High memory usage detected")
            if disk.percent > 90:
                print("⚠️ Low disk space detected")
                
        except Exception as e:
            print(f"❌ Error checking system resources: {e}")
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service with enhanced error handling."""
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
            
            # Check if port is already in use
            if self.is_port_in_use(service["port"]):
                print(f"⚠️ Port {service['port']} is already in use, trying to kill existing process...")
                self.kill_process_on_port(service["port"])
                time.sleep(2)
            
            # Start the service
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.processes[service_name] = process
            
            # Wait and check if it's still running
            time.sleep(5)
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
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if port is in use."""
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                return s.connect_ex(('localhost', port)) == 0
        except:
            return False
    
    def kill_process_on_port(self, port: int):
        """Kill process running on specific port."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    for conn in proc.info['connections'] or []:
                        if conn.laddr.port == port:
                            print(f"Killing process {proc.info['pid']} on port {port}")
                            proc.kill()
                            break
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except Exception as e:
            print(f"Error killing process on port {port}: {e}")
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service with graceful shutdown."""
        if service_name not in self.processes:
            print(f"⚠️ Service {service_name} is not running")
            return False
        
        try:
            process = self.processes[service_name]
            
            # Try graceful shutdown first
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(f"⚠️ Force killing {service_name}")
                process.kill()
                process.wait()
            
            del self.processes[service_name]
            print(f"🛑 {service_name} stopped")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping {service_name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all enterprise services in priority order."""
        print("=" * 80)
        print("🚀 BUL Ultimate Enterprise System Startup")
        print("=" * 80)
        
        self.system_status["started_at"] = datetime.now()
        success_count = 0
        
        # Check system resources first
        self.check_system_resources()
        
        # Start services in priority order
        for service_name in self.startup_order:
            if service_name in self.services:
                if self.start_service(service_name):
                    success_count += 1
                    self.system_status["services_running"] = success_count
                    
                    # Wait between high priority services
                    if self.services[service_name]["priority"] == "high":
                        time.sleep(3)
                else:
                    print(f"⚠️ Failed to start {service_name}, continuing with other services...")
        
        print(f"\n📊 Started {success_count}/{len(self.services)} services")
        
        if success_count == len(self.services):
            self.system_status["health_status"] = "healthy"
            print("✅ All enterprise services started successfully!")
            return True
        elif success_count >= len(self.services) * 0.8:  # 80% success rate
            self.system_status["health_status"] = "degraded"
            print("⚠️ Most services started successfully (degraded mode)")
            return True
        else:
            self.system_status["health_status"] = "unhealthy"
            print("❌ Critical services failed to start")
            return False
    
    def stop_all_services(self):
        """Stop all services gracefully."""
        print("\n🛑 Stopping all enterprise services...")
        
        # Stop in reverse order
        for service_name in reversed(self.startup_order):
            if service_name in self.processes:
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
                    
                    # Check system resources periodically
                    self.check_system_resources()
                    
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    print(f"❌ Monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("🔍 Enterprise service monitoring started")
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        status = self.system_status.copy()
        status["services"] = {}
        
        for service_name, service in self.services.items():
            is_running = service_name in self.processes
            is_healthy = self.check_service_health(service_name) if is_running else False
            
            status["services"][service_name] = {
                "running": is_running,
                "healthy": is_healthy,
                "port": service["port"],
                "description": service["description"],
                "priority": service["priority"]
            }
        
        return status
    
    def print_status(self):
        """Print comprehensive system status."""
        status = self.get_system_status()
        
        print("\n" + "=" * 80)
        print("📊 BUL Ultimate Enterprise System Status")
        print("=" * 80)
        
        print(f"🕐 Started at: {status['started_at']}")
        print(f"🏥 Health status: {status['health_status']}")
        print(f"🔧 Services running: {status['services_running']}/{status['total_services']}")
        print(f"🏢 Enterprise features: {'Enabled' if status['enterprise_features'] else 'Disabled'}")
        print(f"🔒 Security system: {'Enabled' if status['security_enabled'] else 'Disabled'}")
        print(f"💾 Backup system: {'Enabled' if status['backup_enabled'] else 'Disabled'}")
        print(f"📢 Notifications: {'Enabled' if status['notifications_enabled'] else 'Disabled'}")
        
        # System resources
        resources = status.get("system_resources", {})
        if resources:
            print(f"\n💻 System Resources:")
            print(f"  CPU Usage: {resources.get('cpu_percent', 0):.1f}%")
            print(f"  Memory Usage: {resources.get('memory_percent', 0):.1f}%")
            print(f"  Memory Available: {resources.get('memory_available_gb', 0):.1f} GB")
            print(f"  Disk Usage: {resources.get('disk_percent', 0):.1f}%")
            print(f"  Disk Free: {resources.get('disk_free_gb', 0):.1f} GB")
        
        print("\n📋 Service Details:")
        for service_name, service_status in status["services"].items():
            status_icon = "✅" if service_status["healthy"] else "❌" if service_status["running"] else "⏹️"
            priority_icon = "🔴" if service_status["priority"] == "high" else "🟡" if service_status["priority"] == "medium" else "🟢"
            print(f"  {status_icon} {priority_icon} {service_name}: {service_status['description']}")
            print(f"     Port: {service_status['port']}")
            print(f"     Status: {'Healthy' if service_status['healthy'] else 'Running' if service_status['running'] else 'Stopped'}")
        
        print("\n🌐 Access URLs:")
        print(f"  📡 Main API: http://localhost:8000")
        print(f"  🏢 Enterprise: http://localhost:8002")
        print(f"  🔗 Integrations: http://localhost:8003")
        print(f"  🔒 Security: http://localhost:8004")
        print(f"  💾 Backup: http://localhost:8005")
        print(f"  📢 Notifications: http://localhost:8006")
        print(f"  📊 Dashboard: http://localhost:8050")
        print(f"  ⚡ Performance: http://localhost:8001")
        print(f"  📚 API Docs: http://localhost:8000/docs")
        
        print("\n🏢 Enterprise Features:")
        print(f"  👥 User Management: http://localhost:8002/users")
        print(f"  📋 Project Management: http://localhost:8002/projects")
        print(f"  ✅ Task Management: http://localhost:8002/tasks")
        print(f"  📊 Analytics Dashboard: http://localhost:8002/analytics/dashboard")
        print(f"  📈 Performance Reports: http://localhost:8002/reports/project-performance")
        
        print("\n🔒 Security Features:")
        print(f"  🔐 Authentication: http://localhost:8004/auth/login")
        print(f"  👤 User Registration: http://localhost:8004/auth/register")
        print(f"  📋 Audit Logs: http://localhost:8004/security/audit-logs")
        print(f"  🚨 Security Dashboard: http://localhost:8004/security/dashboard")
        print(f"  ⚠️ Threat Detection: http://localhost:8004/security/threats")
        
        print("\n💾 Backup Features:")
        print(f"  📋 Backup Configs: http://localhost:8005/backup/configs")
        print(f"  🚀 Run Backup: http://localhost:8005/backup/run/{{config_name}}")
        print(f"  📊 Backup Status: http://localhost:8005/backup/status/{{config_name}}")
        print(f"  📈 Backup Dashboard: http://localhost:8005/backup/dashboard")
        print(f"  🔄 Restore Backup: http://localhost:8005/backup/restore/{{config_name}}")
        
        print("\n📢 Notification Features:")
        print(f"  📧 Send Notification: http://localhost:8006/notifications/send")
        print(f"  📢 Broadcast: http://localhost:8006/notifications/broadcast")
        print(f"  📋 Templates: http://localhost:8006/templates")
        print(f"  👥 Subscriptions: http://localhost:8006/subscriptions")
        print(f"  📊 Notifications Dashboard: http://localhost:8006/dashboard")
        print(f"  🔌 WebSocket: ws://localhost:8006/ws")
        
        print("\n🔗 External Integrations:")
        print(f"  🧪 Test Integrations: http://localhost:8003/integrations/test")
        print(f"  🔄 Sync Data: http://localhost:8003/integrations/sync")
        print(f"  ⚙️ Configure: http://localhost:8003/integrations/configure")
        print(f"  🏥 Health Check: http://localhost:8003/integrations/health")
        
        print("\n🎯 Quick Actions:")
        print(f"  🔍 Check all services: python start_ultimate_bul.py --status")
        print(f"  🛑 Stop all services: python start_ultimate_bul.py --stop")
        print(f"  🔄 Restart all services: python start_ultimate_bul.py --restart")
        print(f"  📊 Show logs: python start_ultimate_bul.py --logs")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down ultimate enterprise system...")
    if 'manager' in globals():
        manager.stop_all_services()
    sys.exit(0)

def main():
    """Main function with command line arguments."""
    global manager
    
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Ultimate Enterprise System")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--restart", action="store_true", help="Restart all services")
    parser.add_argument("--logs", action="store_true", help="Show service logs")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create ultimate enterprise system manager
    manager = BULUltimateManager()
    
    try:
        if args.status:
            manager.print_status()
            return
        
        if args.stop:
            manager.stop_all_services()
            return
        
        if args.restart:
            manager.stop_all_services()
            time.sleep(5)
            manager.start_all_services()
            return
        
        if args.logs:
            print("📋 Service Logs:")
            for service_name, process in manager.processes.items():
                print(f"\n🔍 {service_name} logs:")
                try:
                    stdout, stderr = process.communicate(timeout=1)
                    if stdout:
                        print(stdout)
                    if stderr:
                        print(stderr)
                except:
                    print("No recent logs available")
            return
        
        # Start all services
        if manager.start_all_services():
            # Start monitoring
            manager.monitor_services()
            
            # Print initial status
            manager.print_status()
            
            print("\n🎉 BUL Ultimate Enterprise System is running!")
            print("Press Ctrl+C to stop all services")
            print("Use --status to check system status")
            print("Use --stop to stop all services")
            print("Use --restart to restart all services")
            print("Use --logs to view service logs")
            
            # Keep running
            while True:
                time.sleep(60)
                manager.print_status()
        
        else:
            print("❌ Failed to start critical enterprise services")
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
