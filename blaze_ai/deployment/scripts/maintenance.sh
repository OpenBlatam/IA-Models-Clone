#!/bin/bash

# Production Maintenance Script for Blaze AI
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
LOG_DIR="/var/log/blaze-ai"
MAINTENANCE_DIR="/var/log/blaze_ai/maintenance"
PYTHON_SCRIPT_DIR="deployment/scripts/python"
NAMESPACE="blaze-ai"

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

check_prerequisites() {
    log_info "Checking maintenance prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 is not installed"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed"
        exit 1
    fi
    
    # Create directories
    mkdir -p $MAINTENANCE_DIR
    mkdir -p $PYTHON_SCRIPT_DIR
    
    log_info "Prerequisites check passed"
}

create_python_maintenance_script() {
    log_info "Creating Python maintenance script..."
    
    cat > $PYTHON_SCRIPT_DIR/maintenance_utils.py << 'EOF'
#!/usr/bin/env python3
"""
Blaze AI Maintenance Utilities
Production maintenance and optimization tools
"""

import os
import sys
import json
import psutil
import subprocess
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sqlite3
import redis
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/blaze-ai/maintenance/maintenance.log'),
        logging.StreamHandler()
    ]
)

class SystemMaintenance:
    """System maintenance and optimization utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.maintenance_db = '/var/log/blaze-ai/maintenance/maintenance.db'
        self._init_database()
    
    def _init_database(self):
        """Initialize maintenance database"""
        try:
            conn = sqlite3.connect(self.maintenance_db)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS maintenance_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details TEXT,
                    duration REAL
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def log_maintenance(self, operation: str, status: str, details: str = "", duration: float = 0.0):
        """Log maintenance operation"""
        try:
            conn = sqlite3.connect(self.maintenance_db)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO maintenance_log (timestamp, operation, status, details, duration)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), operation, status, details, duration))
            conn.commit()
            conn.close()
        except Exception as e:
            self.logger.error(f"Failed to log maintenance: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics"""
        try:
            stats = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg(),
                'uptime': self._get_uptime(),
                'processes': len(psutil.pids()),
                'network_connections': len(psutil.net_connections())
            }
            
            # GPU stats if available
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                stats['gpu_stats'] = [
                    {
                        'name': gpu.name,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'utilization': gpu.load * 100
                    } for gpu in gpus
                ]
            except ImportError:
                stats['gpu_stats'] = []
            
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        try:
            with open('/proc/uptime', 'r') as f:
                uptime_seconds = float(f.readline().split()[0])
            return str(timedelta(seconds=uptime_seconds))
        except:
            return "Unknown"
    
    def optimize_memory(self) -> Dict:
        """Optimize system memory"""
        start_time = datetime.now()
        try:
            # Clear page cache
            subprocess.run(['sync'], check=True)
            subprocess.run(['echo', '3', '>', '/proc/sys/vm/drop_caches'], check=True)
            
            # Clear swap if usage is low
            swap = psutil.swap_memory()
            if swap.percent < 10:
                subprocess.run(['swapoff', '-a'], check=True)
                subprocess.run(['swapon', '-a'], check=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('memory_optimization', 'success', 'Memory optimized', duration)
            
            return {
                'status': 'success',
                'memory_freed': 'Cache cleared',
                'swap_optimized': swap.percent < 10,
                'duration': duration
            }
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('memory_optimization', 'failed', str(e), duration)
            return {'status': 'failed', 'error': str(e)}
    
    def cleanup_logs(self, days: int = 30) -> Dict:
        """Clean up old log files"""
        start_time = datetime.now()
        try:
            log_dirs = ['/var/log', '/var/log/blaze-ai', '/tmp']
            total_size_freed = 0
            files_removed = 0
            
            for log_dir in log_dirs:
                if os.path.exists(log_dir):
                    for root, dirs, files in os.walk(log_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                if os.path.isfile(file_path):
                                    file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(file_path))
                                    if file_age.days > days:
                                        file_size = os.path.getsize(file_path)
                                        os.remove(file_path)
                                        total_size_freed += file_size
                                        files_removed += 1
                            except (OSError, PermissionError):
                                continue
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('log_cleanup', 'success', f'Removed {files_removed} files, freed {total_size_freed} bytes', duration)
            
            return {
                'status': 'success',
                'files_removed': files_removed,
                'bytes_freed': total_size_freed,
                'duration': duration
            }
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('log_cleanup', 'failed', str(e), duration)
            return {'status': 'failed', 'error': str(e)}
    
    def optimize_database(self) -> Dict:
        """Optimize database performance"""
        start_time = datetime.now()
        try:
            # PostgreSQL optimization
            if self._check_postgresql():
                subprocess.run(['sudo', '-u', 'postgres', 'psql', '-c', 'VACUUM ANALYZE;'], check=True)
                subprocess.run(['sudo', '-u', 'postgres', 'psql', '-c', 'REINDEX DATABASE blazeai;'], check=True)
            
            # Redis optimization
            if self._check_redis():
                redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                redis_client.bgrewriteaof()
                redis_client.bgsave()
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('database_optimization', 'success', 'Database optimized', duration)
            
            return {'status': 'success', 'duration': duration}
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('database_optimization', 'failed', str(e), duration)
            return {'status': 'failed', 'error': str(e)}
    
    def _check_postgresql(self) -> bool:
        """Check if PostgreSQL is running"""
        try:
            result = subprocess.run(['pg_isready'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_redis(self) -> bool:
        """Check if Redis is running"""
        try:
            result = subprocess.run(['redis-cli', 'ping'], capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def docker_maintenance(self) -> Dict:
        """Perform Docker maintenance tasks"""
        start_time = datetime.now()
        try:
            # Remove unused containers
            subprocess.run(['docker', 'container', 'prune', '-f'], check=True)
            
            # Remove unused images
            subprocess.run(['docker', 'image', 'prune', '-f'], check=True)
            
            # Remove unused volumes
            subprocess.run(['docker', 'volume', 'prune', '-f'], check=True)
            
            # Remove unused networks
            subprocess.run(['docker', 'network', 'prune', '-f'], check=True)
            
            # System prune
            subprocess.run(['docker', 'system', 'prune', '-f'], check=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('docker_maintenance', 'success', 'Docker cleaned up', duration)
            
            return {'status': 'success', 'duration': duration}
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('docker_maintenance', 'failed', str(e), duration)
            return {'status': 'failed', 'error': str(e)}
    
    def kubernetes_maintenance(self) -> Dict:
        """Perform Kubernetes maintenance tasks"""
        start_time = datetime.now()
        try:
            # Clean up completed jobs
            subprocess.run(['kubectl', 'delete', 'jobs', '--field-selector=status.successful=1', '--all-namespaces'], check=True)
            
            # Clean up failed pods
            subprocess.run(['kubectl', 'delete', 'pods', '--field-selector=status.phase=Failed', '--all-namespaces'], check=True)
            
            # Clean up evicted pods
            subprocess.run(['kubectl', 'delete', 'pods', '--field-selector=status.phase=Evicted', '--all-namespaces'], check=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('kubernetes_maintenance', 'success', 'Kubernetes cleaned up', duration)
            
            return {'status': 'success', 'duration': duration}
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_maintenance('kubernetes_maintenance', 'failed', str(e), duration)
            return {'status': 'failed', 'error': str(e)}
    
    def generate_maintenance_report(self) -> str:
        """Generate comprehensive maintenance report"""
        try:
            report_file = f'/var/log/blaze-ai/maintenance/report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            
            # Get recent maintenance history
            conn = sqlite3.connect(self.maintenance_db)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT operation, status, timestamp, details, duration 
                FROM maintenance_log 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            history = cursor.fetchall()
            conn.close()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'hostname': os.uname().nodename,
                'system_stats': self.get_system_stats(),
                'maintenance_history': [
                    {
                        'operation': row[0],
                        'status': row[1],
                        'timestamp': row[2],
                        'details': row[3],
                        'duration': row[4]
                    } for row in history
                ],
                'summary': {
                    'total_operations': len(history),
                    'successful_operations': len([h for h in history if h[1] == 'success']),
                    'failed_operations': len([h for h in history if h[1] == 'failed']),
                    'average_duration': sum(h[4] for h in history if h[4]) / len([h for h in history if h[4]]) if history else 0
                }
            }
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            return report_file
        except Exception as e:
            self.logger.error(f"Failed to generate report: {e}")
            return ""

def main():
    """Main maintenance execution"""
    maintenance = SystemMaintenance()
    
    print("Starting Blaze AI maintenance...")
    
    # Run maintenance tasks
    tasks = [
        ('Memory Optimization', maintenance.optimize_memory),
        ('Log Cleanup', lambda: maintenance.cleanup_logs(30)),
        ('Database Optimization', maintenance.optimize_database),
        ('Docker Maintenance', maintenance.docker_maintenance),
        ('Kubernetes Maintenance', maintenance.kubernetes_maintenance)
    ]
    
    results = {}
    for task_name, task_func in tasks:
        print(f"Running {task_name}...")
        try:
            result = task_func()
            results[task_name] = result
            print(f"{task_name}: {result['status']}")
        except Exception as e:
            print(f"{task_name}: Failed - {e}")
            results[task_name] = {'status': 'failed', 'error': str(e)}
    
    # Generate report
    report_file = maintenance.generate_maintenance_report()
    if report_file:
        print(f"Maintenance report generated: {report_file}")
    
    print("Maintenance completed!")

if __name__ == "__main__":
    main()
EOF
    
    log_info "Python maintenance script created"
}

run_python_maintenance() {
    log_info "Running Python maintenance script..."
    
    if [ -f "$PYTHON_SCRIPT_DIR/maintenance_utils.py" ]; then
        # Install required Python packages
        pip3 install psutil redis pyyaml
        
        # Run maintenance script
        python3 $PYTHON_SCRIPT_DIR/maintenance_utils.py
        
        log_info "Python maintenance completed"
    else
        log_error "Python maintenance script not found"
    fi
}

cleanup_old_backups() {
    log_info "Cleaning up old backups..."
    
    # Remove backups older than 30 days
    find /backups/blaze-ai -name "*.sql.gz" -mtime +30 -delete 2>/dev/null || true
    find /backups/blaze-ai -name "*.rdb.gz" -mtime +30 -delete 2>/dev/null || true
    find /backups/blaze-ai -name "*.tar" -mtime +30 -delete 2>/dev/null || true
    find /backups/blaze-ai -name "manifest_*.txt" -mtime +30 -delete 2>/dev/null || true
    
    log_info "Old backups cleanup completed"
}

cleanup_docker_resources() {
    log_info "Cleaning up Docker resources..."
    
    # Remove unused containers
    docker container prune -f
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes
    docker volume prune -f
    
    # Remove unused networks
    docker network prune -f
    
    # System prune
    docker system prune -f
    
    log_info "Docker cleanup completed"
}

cleanup_kubernetes_resources() {
    log_info "Cleaning up Kubernetes resources..."
    
    # Clean up completed jobs
    kubectl delete jobs --field-selector=status.successful=1 --all-namespaces 2>/dev/null || true
    
    # Clean up failed pods
    kubectl delete pods --field-selector=status.phase=Failed --all-namespaces 2>/dev/null || true
    
    # Clean up evicted pods
    kubectl delete pods --field-selector=status.phase=Evicted --all-namespaces 2>/dev/null || true
    
    log_info "Kubernetes cleanup completed"
}

optimize_system() {
    log_info "Optimizing system performance..."
    
    # Clear page cache
    sync
    echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
    
    # Clear swap if usage is low
    SWAP_USAGE=$(free | grep Swap | awk '{print $3/$2 * 100.0}')
    if (( $(echo "$SWAP_USAGE < 10" | bc -l) )); then
        swapoff -a && swapon -a 2>/dev/null || true
    fi
    
    log_info "System optimization completed"
}

check_disk_space() {
    log_info "Checking disk space..."
    
    # Check root filesystem
    ROOT_USAGE=$(df / | tail -1 | awk '{print $5}' | cut -d'%' -f1)
    
    if [ $ROOT_USAGE -gt 90 ]; then
        log_error "Critical disk usage: ${ROOT_USAGE}%"
        
        # Find large files
        log_info "Finding large files..."
        find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null | head -20
        
        # Find large directories
        log_info "Finding large directories..."
        du -h / 2>/dev/null | sort -hr | head -20
    elif [ $ROOT_USAGE -gt 80 ]; then
        log_warn "High disk usage: ${ROOT_USAGE}%"
    else
        log_info "Disk usage: ${ROOT_USAGE}% (OK)"
    fi
}

check_service_health() {
    log_info "Checking service health..."
    
    # Check Docker services
    if command -v docker &> /dev/null; then
        DOCKER_STATUS=$(docker info 2>/dev/null | grep -c "Running" || echo "0")
        if [ $DOCKER_STATUS -gt 0 ]; then
            log_info "Docker: Running"
        else
            log_warn "Docker: Not running properly"
        fi
    fi
    
    # Check Kubernetes
    if command -v kubectl &> /dev/null; then
        if kubectl cluster-info &> /dev/null; then
            log_info "Kubernetes: Running"
        else
            log_warn "Kubernetes: Not accessible"
        fi
    fi
    
    # Check application services
    if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "API Service: Healthy"
    else
        log_warn "API Service: Unhealthy"
    fi
}

generate_maintenance_report() {
    log_info "Generating maintenance report..."
    
    REPORT_FILE="$MAINTENANCE_DIR/maintenance_report_$(date +%Y%m%d_%H%M%S).txt"
    
    cat > $REPORT_FILE << EOF
Blaze AI Maintenance Report
===========================
Generated: $(date)
Hostname: $(hostname)
Maintainer: $(whoami)

System Information:
- Uptime: $(uptime)
- Load Average: $(cat /proc/loadavg 2>/dev/null || echo "Unknown")
- Memory Usage: $(free -h | grep Mem | awk '{print $3"/"$2}')
- Disk Usage: $(df -h / | tail -1 | awk '{print $5}')

Maintenance Tasks Completed:
- Python Maintenance: $(if [ -f "$PYTHON_SCRIPT_DIR/maintenance_utils.py" ]; then echo "Yes"; else echo "No"; fi)
- Docker Cleanup: $(if command -v docker &> /dev/null; then echo "Yes"; else echo "No"; fi)
- Kubernetes Cleanup: $(if command -v kubectl &> /dev/null; then echo "Yes"; else echo "No"; fi)
- System Optimization: Yes
- Backup Cleanup: Yes

Disk Space Analysis:
$(df -h | grep -E "^/dev/")

Large Files (Top 10):
$(find / -type f -size +100M -exec ls -lh {} \; 2>/dev/null | head -10)

Large Directories (Top 10):
$(du -h / 2>/dev/null | sort -hr | head -10)

Service Status:
- Docker: $(if command -v docker &> /dev/null && docker info &> /dev/null; then echo "Running"; else echo "Not Running"; fi)
- Kubernetes: $(if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then echo "Running"; else echo "Not Running"; fi)
- API Service: $(if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then echo "Healthy"; else echo "Unhealthy"; fi)

Recommendations:
$(if [ $ROOT_USAGE -gt 80 ]; then echo "- Consider cleaning up disk space"; fi)
$(if ! command -v docker &> /dev/null; then echo "- Install Docker for container management"; fi)
$(if ! command -v kubectl &> /dev/null; then echo "- Install kubectl for Kubernetes management"; fi)

Next Maintenance: $(date -d "+7 days" +%Y-%m-%d)
EOF
    
    log_info "Maintenance report generated: $REPORT_FILE"
}

# Main maintenance logic
main() {
    log_info "Starting Blaze AI production maintenance..."
    
    check_prerequisites
    create_python_maintenance_script
    
    # Run maintenance tasks
    run_python_maintenance
    cleanup_old_backups
    cleanup_docker_resources
    cleanup_kubernetes_resources
    optimize_system
    check_disk_space
    check_service_health
    
    # Generate report
    generate_maintenance_report
    
    log_info "Production maintenance completed successfully!"
}

# Run main function
main "$@"
