"""
BUL - Business Universal Language (Ultimate Advanced Startup)
============================================================

Ultimate startup script for advanced BUL system with all cutting-edge features.
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

class BULUltimateAdvancedManager:
    """Ultimate Advanced BUL system manager with all cutting-edge features."""
    
    def __init__(self):
        self.processes = {}
        self.services = {
            "main_api": {
                "script": "bul_divine_ai.py",
                "port": 8000,
                "description": "Main BUL Divine AI API",
                "priority": "critical",
                "category": "core"
            },
            "enterprise_system": {
                "script": "bul_enterprise.py",
                "port": 8002,
                "description": "Enterprise Management System",
                "priority": "high",
                "category": "enterprise"
            },
            "external_integrations": {
                "script": "bul_integrations.py",
                "port": 8003,
                "description": "External API Integrations",
                "priority": "high",
                "category": "enterprise"
            },
            "advanced_security": {
                "script": "bul_security.py",
                "port": 8004,
                "description": "Advanced Security System",
                "priority": "critical",
                "category": "security"
            },
            "auto_backup": {
                "script": "bul_backup.py",
                "port": 8005,
                "description": "Automatic Backup System",
                "priority": "medium",
                "category": "infrastructure"
            },
            "notifications": {
                "script": "bul_notifications.py",
                "port": 8006,
                "description": "Advanced Notifications System",
                "priority": "medium",
                "category": "communication"
            },
            "ml_system": {
                "script": "bul_ml_system.py",
                "port": 8007,
                "description": "Advanced Machine Learning System",
                "priority": "high",
                "category": "ai"
            },
            "blockchain_system": {
                "script": "bul_blockchain.py",
                "port": 8008,
                "description": "Blockchain & DeFi System",
                "priority": "high",
                "category": "blockchain"
            },
            "iot_system": {
                "script": "bul_iot_system.py",
                "port": 8009,
                "description": "IoT & Smart Devices System",
                "priority": "medium",
                "category": "iot"
            },
            "vr_system": {
                "script": "bul_vr_system.py",
                "port": 8010,
                "description": "Virtual Reality & Spatial Computing System",
                "priority": "medium",
                "category": "vr"
            },
            "quantum_system": {
                "script": "bul_quantum_system.py",
                "port": 8011,
                "description": "Quantum Computing & Algorithms System",
                "priority": "high",
                "category": "quantum"
            },
            "performance_optimizer": {
                "script": "bul_performance_optimizer.py",
                "port": 8001,
                "description": "Performance Monitoring & Optimization",
                "priority": "low",
                "category": "monitoring"
            },
            "advanced_dashboard": {
                "script": "bul_advanced_dashboard.py",
                "port": 8050,
                "description": "Advanced Real-time Dashboard",
                "priority": "low",
                "category": "monitoring"
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
            "ml_enabled": True,
            "blockchain_enabled": True,
            "iot_enabled": True,
            "system_resources": {},
            "categories": {
                "core": 0,
                "enterprise": 0,
                "security": 0,
                "infrastructure": 0,
                "communication": 0,
                "ai": 0,
                "blockchain": 0,
                "iot": 0,
                "vr": 0,
                "quantum": 0,
                "monitoring": 0
            }
        }
        self.startup_order = [
            "advanced_security",
            "main_api",
            "enterprise_system",
            "external_integrations",
            "ml_system",
            "blockchain_system",
            "quantum_system",
            "iot_system",
            "vr_system",
            "auto_backup",
            "notifications",
            "performance_optimizer",
            "advanced_dashboard"
        ]
    
    def check_system_resources(self):
        """Check comprehensive system resources."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Get additional system info
            boot_time = psutil.boot_time()
            uptime = time.time() - boot_time
            
            self.system_status["system_resources"] = {
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "memory_percent": memory.percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "disk_percent": disk.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "network_packets_sent": network.packets_sent,
                "network_packets_recv": network.packets_recv,
                "system_uptime_hours": uptime / 3600,
                "boot_time": datetime.fromtimestamp(boot_time).isoformat()
            }
            
            # Check resource thresholds
            warnings = []
            if cpu_percent > 90:
                warnings.append("⚠️ High CPU usage detected")
            if memory.percent > 90:
                warnings.append("⚠️ High memory usage detected")
            if disk.percent > 90:
                warnings.append("⚠️ Low disk space detected")
            if uptime < 3600:  # Less than 1 hour
                warnings.append("ℹ️ System recently restarted")
            
            if warnings:
                for warning in warnings:
                    print(warning)
                    
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
                
                # Update category count
                category = service["category"]
                self.system_status["categories"][category] += 1
                
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
            
            # Update category count
            service = self.services[service_name]
            category = service["category"]
            self.system_status["categories"][category] -= 1
            
            print(f"🛑 {service_name} stopped")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping {service_name}: {e}")
            return False
    
    def start_all_services(self) -> bool:
        """Start all advanced services in priority order."""
        print("=" * 90)
        print("🚀 BUL Ultimate Advanced System Startup")
        print("=" * 90)
        
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
                    
                    # Wait between critical services
                    if self.services[service_name]["priority"] == "critical":
                        time.sleep(5)
                    elif self.services[service_name]["priority"] == "high":
                        time.sleep(3)
                else:
                    print(f"⚠️ Failed to start {service_name}, continuing with other services...")
        
        print(f"\n📊 Started {success_count}/{len(self.services)} services")
        
        if success_count == len(self.services):
            self.system_status["health_status"] = "healthy"
            print("✅ All advanced services started successfully!")
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
        print("\n🛑 Stopping all advanced services...")
        
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
        print("🔍 Advanced service monitoring started")
    
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
                "priority": service["priority"],
                "category": service["category"]
            }
        
        return status
    
    def print_status(self):
        """Print comprehensive system status."""
        status = self.get_system_status()
        
        print("\n" + "=" * 90)
        print("📊 BUL Ultimate Advanced System Status")
        print("=" * 90)
        
        print(f"🕐 Started at: {status['started_at']}")
        print(f"🏥 Health status: {status['health_status']}")
        print(f"🔧 Services running: {status['services_running']}/{status['total_services']}")
        print(f"🏢 Enterprise features: {'Enabled' if status['enterprise_features'] else 'Disabled'}")
        print(f"🔒 Security system: {'Enabled' if status['security_enabled'] else 'Disabled'}")
        print(f"💾 Backup system: {'Enabled' if status['backup_enabled'] else 'Disabled'}")
        print(f"📢 Notifications: {'Enabled' if status['notifications_enabled'] else 'Disabled'}")
        print(f"🤖 ML system: {'Enabled' if status['ml_enabled'] else 'Disabled'}")
        print(f"⛓️ Blockchain: {'Enabled' if status['blockchain_enabled'] else 'Disabled'}")
        print(f"🌐 IoT system: {'Enabled' if status['iot_enabled'] else 'Disabled'}")
        print(f"🥽 VR system: {'Enabled' if status.get('vr_enabled', True) else 'Disabled'}")
        print(f"⚛️ Quantum system: {'Enabled' if status.get('quantum_enabled', True) else 'Disabled'}")
        
        # System resources
        resources = status.get("system_resources", {})
        if resources:
            print(f"\n💻 System Resources:")
            print(f"  CPU Usage: {resources.get('cpu_percent', 0):.1f}% ({resources.get('cpu_count', 0)} cores)")
            print(f"  Memory Usage: {resources.get('memory_percent', 0):.1f}% ({resources.get('memory_used_gb', 0):.1f}/{resources.get('memory_total_gb', 0):.1f} GB)")
            print(f"  Disk Usage: {resources.get('disk_percent', 0):.1f}% ({resources.get('disk_used_gb', 0):.1f}/{resources.get('disk_total_gb', 0):.1f} GB)")
            print(f"  Network: {resources.get('network_bytes_sent', 0)/1024/1024:.1f} MB sent, {resources.get('network_bytes_recv', 0)/1024/1024:.1f} MB received")
            print(f"  Uptime: {resources.get('system_uptime_hours', 0):.1f} hours")
        
        # Service categories
        print(f"\n📋 Service Categories:")
        for category, count in status["categories"].items():
            if count > 0:
                print(f"  {category.upper()}: {count} services")
        
        print("\n📋 Service Details:")
        for service_name, service_status in status["services"].items():
            status_icon = "✅" if service_status["healthy"] else "❌" if service_status["running"] else "⏹️"
            priority_icon = "🔴" if service_status["priority"] == "critical" else "🟡" if service_status["priority"] == "high" else "🟢"
            category_icon = {
                "core": "🎯",
                "enterprise": "🏢",
                "security": "🔒",
                "infrastructure": "🏗️",
                "communication": "📢",
                "ai": "🤖",
                "blockchain": "⛓️",
                "iot": "🌐",
                "vr": "🥽",
                "quantum": "⚛️",
                "monitoring": "📊"
            }.get(service_status["category"], "📦")
            
            print(f"  {status_icon} {priority_icon} {category_icon} {service_name}: {service_status['description']}")
            print(f"     Port: {service_status['port']}")
            print(f"     Status: {'Healthy' if service_status['healthy'] else 'Running' if service_status['running'] else 'Stopped'}")
        
        print("\n🌐 Access URLs:")
        print(f"  📡 Main API: http://localhost:8000")
        print(f"  🏢 Enterprise: http://localhost:8002")
        print(f"  🔗 Integrations: http://localhost:8003")
        print(f"  🔒 Security: http://localhost:8004")
        print(f"  💾 Backup: http://localhost:8005")
        print(f"  📢 Notifications: http://localhost:8006")
        print(f"  🤖 ML System: http://localhost:8007")
        print(f"  ⛓️ Blockchain: http://localhost:8008")
        print(f"  🌐 IoT System: http://localhost:8009")
        print(f"  🥽 VR System: http://localhost:8010")
        print(f"  ⚛️ Quantum System: http://localhost:8011")
        print(f"  📊 Dashboard: http://localhost:8050")
        print(f"  ⚡ Performance: http://localhost:8001")
        print(f"  📚 API Docs: http://localhost:8000/docs")
        
        print("\n🏢 Enterprise Features:")
        print(f"  👥 User Management: http://localhost:8002/users")
        print(f"  📋 Project Management: http://localhost:8002/projects")
        print(f"  ✅ Task Management: http://localhost:8002/tasks")
        print(f"  📊 Analytics Dashboard: http://localhost:8002/analytics/dashboard")
        
        print("\n🔒 Security Features:")
        print(f"  🔐 Authentication: http://localhost:8004/auth/login")
        print(f"  👤 User Registration: http://localhost:8004/auth/register")
        print(f"  📋 Audit Logs: http://localhost:8004/security/audit-logs")
        print(f"  🚨 Security Dashboard: http://localhost:8004/security/dashboard")
        
        print("\n🤖 AI & ML Features:")
        print(f"  📊 ML Dashboard: http://localhost:8007/dashboard")
        print(f"  🧠 Model Training: http://localhost:8007/models/train")
        print(f"  🔮 Predictions: http://localhost:8007/models/{{model_id}}/predict")
        print(f"  📈 Data Upload: http://localhost:8007/datasets/upload")
        
        print("\n⛓️ Blockchain Features:")
        print(f"  💼 Wallet Management: http://localhost:8008/wallets")
        print(f"  💸 Transactions: http://localhost:8008/transactions/send")
        print(f"  📜 Smart Contracts: http://localhost:8008/contracts")
        print(f"  🎨 NFTs: http://localhost:8008/nfts")
        print(f"  💰 DeFi Pools: http://localhost:8008/defi/pools")
        
        print("\n🌐 IoT Features:")
        print(f"  📱 Device Management: http://localhost:8009/devices")
        print(f"  📊 Data Collection: http://localhost:8009/devices/{{device_id}}/data")
        print(f"  🤖 Automations: http://localhost:8009/automations")
        print(f"  🚨 Alerts: http://localhost:8009/alerts")
        print(f"  📈 IoT Dashboard: http://localhost:8009/dashboard")
        print(f"  🔌 WebSocket: ws://localhost:8009/ws")
        
        print("\n🥽 VR Features:")
        print(f"  🎮 VR Sessions: http://localhost:8010/sessions")
        print(f"  🌍 Environments: http://localhost:8010/environments")
        print(f"  📦 VR Objects: http://localhost:8010/objects")
        print(f"  🤝 Interactions: http://localhost:8010/interactions")
        print(f"  📊 VR Dashboard: http://localhost:8010/dashboard")
        print(f"  🔌 WebSocket: ws://localhost:8010/ws")
        
        print("\n⚛️ Quantum Features:")
        print(f"  🔬 Quantum Circuits: http://localhost:8011/circuits")
        print(f"  ⚡ Job Execution: http://localhost:8011/jobs")
        print(f"  🧮 Algorithms: http://localhost:8011/algorithms")
        print(f"  📏 Measurements: http://localhost:8011/measurements")
        print(f"  📊 Quantum Dashboard: http://localhost:8011/dashboard")
        
        print("\n💾 Backup Features:")
        print(f"  📋 Backup Configs: http://localhost:8005/backup/configs")
        print(f"  🚀 Run Backup: http://localhost:8005/backup/run/{{config_name}}")
        print(f"  📊 Backup Status: http://localhost:8005/backup/status/{{config_name}}")
        print(f"  🔄 Restore Backup: http://localhost:8005/backup/restore/{{config_name}}")
        
        print("\n📢 Notification Features:")
        print(f"  📧 Send Notification: http://localhost:8006/notifications/send")
        print(f"  📢 Broadcast: http://localhost:8006/notifications/broadcast")
        print(f"  📋 Templates: http://localhost:8006/templates")
        print(f"  👥 Subscriptions: http://localhost:8006/subscriptions")
        print(f"  🔌 WebSocket: ws://localhost:8006/ws")
        
        print("\n🎯 Quick Actions:")
        print(f"  🔍 Check all services: python start_ultimate_advanced_bul.py --status")
        print(f"  🛑 Stop all services: python start_ultimate_advanced_bul.py --stop")
        print(f"  🔄 Restart all services: python start_ultimate_advanced_bul.py --restart")
        print(f"  📊 Show logs: python start_ultimate_advanced_bul.py --logs")
        print(f"  🔧 System info: python start_ultimate_advanced_bul.py --system")

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n🛑 Received signal {signum}, shutting down ultimate advanced system...")
    if 'manager' in globals():
        manager.stop_all_services()
    sys.exit(0)

def main():
    """Main function with command line arguments."""
    global manager
    
    import argparse
    
    parser = argparse.ArgumentParser(description="BUL Ultimate Advanced System")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    parser.add_argument("--restart", action="store_true", help="Restart all services")
    parser.add_argument("--logs", action="store_true", help="Show service logs")
    parser.add_argument("--system", action="store_true", help="Show system information")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create ultimate advanced system manager
    manager = BULUltimateAdvancedManager()
    
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
        
        if args.system:
            print("💻 System Information:")
            manager.check_system_resources()
            resources = manager.system_status["system_resources"]
            print(f"  OS: {os.name}")
            print(f"  Python: {sys.version}")
            print(f"  CPU: {resources.get('cpu_count', 0)} cores")
            print(f"  Memory: {resources.get('memory_total_gb', 0):.1f} GB")
            print(f"  Disk: {resources.get('disk_total_gb', 0):.1f} GB")
            print(f"  Uptime: {resources.get('system_uptime_hours', 0):.1f} hours")
            return
        
        # Start all services
        if manager.start_all_services():
            # Start monitoring
            manager.monitor_services()
            
            # Print initial status
            manager.print_status()
            
            print("\n🎉 BUL Ultimate Advanced System is running!")
            print("Press Ctrl+C to stop all services")
            print("Use --status to check system status")
            print("Use --stop to stop all services")
            print("Use --restart to restart all services")
            print("Use --logs to view service logs")
            print("Use --system to view system information")
            
            # Keep running
            while True:
                time.sleep(60)
                manager.print_status()
        
        else:
            print("❌ Failed to start critical advanced services")
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
