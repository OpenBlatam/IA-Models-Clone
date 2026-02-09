#!/bin/bash

# Ultra-Optimized SEO Service v14 - MAXIMUM PERFORMANCE
# Advanced Deployment Script with Monitoring, Testing, and Performance Optimization

set -euo pipefail

# Configuration
SERVICE_NAME="ultra-seo-service-v14"
VERSION="14.0.0"
DOCKER_COMPOSE_FILE="docker-compose.production_v14.yml"
DOCKERFILE="Dockerfile.production_v14"
REQUIREMENTS_FILE="requirements.ultra_optimized_v14.txt"
TEST_FILE="test_production_v14.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${PURPLE}================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}================================${NC}"
}

# Check prerequisites
check_prerequisites() {
    log_header "Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    log_success "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    log_success "Docker Compose found: $(docker-compose --version)"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    log_success "Python found: $(python3 --version)"
    
    # Check required files
    local required_files=("$DOCKER_COMPOSE_FILE" "$DOCKERFILE" "$REQUIREMENTS_FILE")
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log_error "Required file not found: $file"
            exit 1
        fi
    done
    log_success "All required files found"
    
    # Check system resources
    local total_mem=$(free -m | awk 'NR==2{printf "%.0f", $2/1024}')
    if [[ $total_mem -lt 8 ]]; then
        log_warning "System has less than 8GB RAM. Performance may be limited."
    else
        log_success "System memory: ${total_mem}GB"
    fi
    
    local cpu_cores=$(nproc)
    if [[ $cpu_cores -lt 4 ]]; then
        log_warning "System has less than 4 CPU cores. Performance may be limited."
    else
        log_success "CPU cores: $cpu_cores"
    fi
}

# Create configuration files
create_config_files() {
    log_header "Creating Configuration Files"
    
    # Create Redis configuration
    cat > redis.optimized.conf << 'EOF'
# Ultra-Optimized Redis Configuration for SEO Service v14
bind 0.0.0.0
port 6379
timeout 0
tcp-keepalive 300
daemonize no
supervised no
pidfile /var/run/redis_6379.pid
loglevel notice
logfile ""
databases 16
save 900 1
save 300 10
save 60 10000
stop-writes-on-bgsave-error yes
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data
maxmemory 1gb
maxmemory-policy allkeys-lru
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
aof-load-truncated yes
aof-use-rdb-preamble yes
lua-time-limit 5000
slowlog-log-slower-than 10000
slowlog-max-len 128
latency-monitor-threshold 100
notify-keyspace-events ""
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
list-compress-depth 0
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64
hll-sparse-max-bytes 3000
stream-node-max-bytes 4096
stream-node-max-entries 100
activerehashing yes
client-output-buffer-limit normal 0 0 0
client-output-buffer-limit replica 256mb 64mb 60
client-output-buffer-limit pubsub 32mb 8mb 60
hz 10
dynamic-hz yes
aof-rewrite-incremental-fsync yes
rdb-save-incremental-fsync yes
EOF
    log_success "Redis configuration created"
    
    # Create Nginx configuration
    cat > nginx.optimized.conf << 'EOF'
# Ultra-Optimized Nginx Configuration for SEO Service v14
user nginx;
worker_processes auto;
worker_cpu_affinity auto;
worker_rlimit_nofile 65535;
error_log /var/log/nginx/error.log warn;
pid /var/run/nginx.pid;

events {
    worker_connections 65535;
    use epoll;
    multi_accept on;
    accept_mutex off;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';
    access_log /var/log/nginx/access.log main;
    
    # Performance optimizations
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    keepalive_requests 1000;
    types_hash_max_size 2048;
    server_tokens off;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_comp_level 6;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml
        image/svg+xml;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=100r/s;
    limit_req_zone $binary_remote_addr zone=login:10m rate=10r/m;
    
    # Upstream configuration
    upstream seo_backend {
        least_conn;
        server seo-service:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    # Main server
    server {
        listen 80;
        server_name _;
        
        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";
        add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
        
        # Rate limiting
        limit_req zone=api burst=20 nodelay;
        
        # Proxy settings
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
        
        # Health check
        location /health {
            proxy_pass http://seo_backend;
            access_log off;
        }
        
        # API endpoints
        location / {
            proxy_pass http://seo_backend;
        }
        
        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
EOF
    log_success "Nginx configuration created"
    
    # Create Prometheus configuration
    cat > prometheus.yml << 'EOF'
# Ultra-Optimized Prometheus Configuration for SEO Service v14
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'seo-service'
    static_configs:
      - targets: ['seo-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    metrics_path: '/nginx_status'
EOF
    log_success "Prometheus configuration created"
    
    # Create load test script
    cat > load_test.py << 'EOF'
#!/usr/bin/env python3
"""
Ultra-Optimized Load Test for SEO Service v14
"""
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import json

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict] = []
    
    async def test_single_request(self, session: aiohttp.ClientSession, url: str) -> Dict:
        """Test a single request"""
        start_time = time.time()
        try:
            async with session.post(f"{self.base_url}/analyze", json={
                "url": url,
                "depth": 1,
                "include_metrics": True,
                "use_http3": True
            }) as response:
                end_time = time.time()
                duration = end_time - start_time
                
                return {
                    "status": response.status,
                    "duration": duration,
                    "success": response.status == 200
                }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            return {
                "status": 0,
                "duration": duration,
                "success": False,
                "error": str(e)
            }
    
    async def run_load_test(self, concurrent_users: int = 100, total_requests: int = 1000):
        """Run load test"""
        test_urls = [
            "https://www.google.com",
            "https://www.github.com",
            "https://www.stackoverflow.com",
            "https://www.wikipedia.org",
            "https://www.reddit.com"
        ]
        
        print(f"Starting load test: {concurrent_users} concurrent users, {total_requests} total requests")
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrent_users)
            
            async def make_request():
                async with semaphore:
                    url = test_urls[len(self.results) % len(test_urls)]
                    result = await self.test_single_request(session, url)
                    self.results.append(result)
            
            tasks = [make_request() for _ in range(total_requests)]
            await asyncio.gather(*tasks)
        
        self.print_results()
    
    def print_results(self):
        """Print test results"""
        successful_requests = [r for r in self.results if r["success"]]
        failed_requests = [r for r in self.results if not r["success"]]
        
        if successful_requests:
            durations = [r["duration"] for r in successful_requests]
            
            print("\n=== Load Test Results ===")
            print(f"Total Requests: {len(self.results)}")
            print(f"Successful: {len(successful_requests)}")
            print(f"Failed: {len(failed_requests)}")
            print(f"Success Rate: {len(successful_requests)/len(self.results)*100:.2f}%")
            print(f"Average Response Time: {statistics.mean(durations):.3f}s")
            print(f"Median Response Time: {statistics.median(durations):.3f}s")
            print(f"Min Response Time: {min(durations):.3f}s")
            print(f"Max Response Time: {max(durations):.3f}s")
            print(f"95th Percentile: {statistics.quantiles(durations, n=20)[18]:.3f}s")
            print(f"99th Percentile: {statistics.quantiles(durations, n=100)[98]:.3f}s")
        
        if failed_requests:
            print(f"\nFailed Requests: {len(failed_requests)}")
            for req in failed_requests[:5]:  # Show first 5 failures
                print(f"  - Error: {req.get('error', 'Unknown')}")

if __name__ == "__main__":
    import sys
    
    concurrent_users = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    total_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    tester = LoadTester()
    asyncio.run(tester.run_load_test(concurrent_users, total_requests))
EOF
    chmod +x load_test.py
    log_success "Load test script created"
}

# Deploy the service
deploy() {
    log_header "Deploying Ultra-Optimized SEO Service v14"
    
    # Stop existing containers
    log_info "Stopping existing containers..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans || true
    
    # Build images
    log_info "Building Docker images..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" build --no-cache --parallel
    
    # Start services
    log_info "Starting services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    check_health
    
    log_success "Deployment completed successfully!"
}

# Check service health
check_health() {
    log_header "Checking Service Health"
    
    local services=("seo-service" "redis" "nginx" "prometheus" "grafana")
    local ports=(8000 6379 80 9090 3000)
    
    for i in "${!services[@]}"; do
        local service="${services[$i]}"
        local port="${ports[$i]}"
        
        log_info "Checking $service..."
        if curl -f "http://localhost:$port" > /dev/null 2>&1 || \
           curl -f "http://localhost:$port/health" > /dev/null 2>&1 || \
           curl -f "http://localhost:$port/metrics" > /dev/null 2>&1; then
            log_success "$service is healthy"
        else
            log_warning "$service health check failed"
        fi
    done
}

# Run tests
run_tests() {
    log_header "Running Tests"
    
    # Run unit tests
    log_info "Running unit tests..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" run --rm testing
    
    # Run integration tests
    log_info "Running integration tests..."
    if command -v python3 &> /dev/null; then
        python3 -m pytest "$TEST_FILE" -v --tb=short
    else
        log_warning "Python3 not available for local testing"
    fi
    
    log_success "Tests completed!"
}

# Run load testing
run_load_test() {
    log_header "Running Load Test"
    
    local concurrent_users=${1:-100}
    local total_requests=${2:-1000}
    
    log_info "Starting load test with $concurrent_users concurrent users and $total_requests total requests..."
    
    if command -v python3 &> /dev/null; then
        python3 load_test.py "$concurrent_users" "$total_requests"
    else
        log_warning "Python3 not available for load testing"
    fi
    
    log_success "Load test completed!"
}

# Run security scan
run_security_scan() {
    log_header "Running Security Scan"
    
    log_info "Running Bandit security scan..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" run --rm security
    
    log_info "Running Safety check..."
    if command -v safety &> /dev/null; then
        safety check
    else
        log_warning "Safety not installed"
    fi
    
    log_success "Security scan completed!"
}

# Show service status
show_status() {
    log_header "Service Status"
    
    echo "=== Docker Containers ==="
    docker-compose -f "$DOCKER_COMPOSE_FILE" ps
    
    echo -e "\n=== Service URLs ==="
    echo "SEO Service: http://localhost:8000"
    echo "Nginx: http://localhost:80"
    echo "Prometheus: http://localhost:9090"
    echo "Grafana: http://localhost:3000 (admin/admin123)"
    echo "Jaeger: http://localhost:16686"
    echo "Flower: http://localhost:5555"
    echo "Load Testing: http://localhost:8089"
    
    echo -e "\n=== Health Checks ==="
    check_health
}

# Show performance metrics
show_metrics() {
    log_header "Performance Metrics"
    
    # Get container stats
    echo "=== Container Resource Usage ==="
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
    
    # Get service metrics
    echo -e "\n=== Service Metrics ==="
    if curl -f "http://localhost:8000/metrics" > /dev/null 2>&1; then
        curl -s "http://localhost:8000/metrics" | jq '.' 2>/dev/null || curl -s "http://localhost:8000/metrics"
    else
        log_warning "Service metrics endpoint not available"
    fi
}

# Optimize performance
optimize_performance() {
    log_header "Performance Optimization"
    
    # Optimize cache
    log_info "Optimizing cache..."
    curl -X POST "http://localhost:8000/cache/optimize" || log_warning "Cache optimization failed"
    
    # Run garbage collection
    log_info "Running garbage collection..."
    docker system prune -f
    
    # Restart services for fresh state
    log_info "Restarting services for optimization..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" restart seo-service
    
    log_success "Performance optimization completed!"
}

# Cleanup
cleanup() {
    log_header "Cleanup"
    
    log_info "Stopping all services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --remove-orphans
    
    log_info "Removing unused containers..."
    docker container prune -f
    
    log_info "Removing unused images..."
    docker image prune -f
    
    log_info "Removing unused volumes..."
    docker volume prune -f
    
    log_info "Removing unused networks..."
    docker network prune -f
    
    log_success "Cleanup completed!"
}

# Show help
show_help() {
    cat << EOF
Ultra-Optimized SEO Service v14 - Deployment Script

Usage: $0 [COMMAND]

Commands:
    deploy              Deploy the complete service stack
    test                Run all tests
    load-test [users] [requests]  Run load test (default: 100 users, 1000 requests)
    security            Run security scan
    status              Show service status
    metrics             Show performance metrics
    optimize            Optimize performance
    cleanup             Clean up all containers and images
    help                Show this help message

Examples:
    $0 deploy
    $0 test
    $0 load-test 200 2000
    $0 status
    $0 metrics

Environment Variables:
    DOCKER_COMPOSE_FILE  Docker Compose file (default: docker-compose.production_v14.yml)
    DOCKERFILE          Dockerfile (default: Dockerfile.production_v14)
    REQUIREMENTS_FILE   Requirements file (default: requirements.ultra_optimized_v14.txt)

EOF
}

# Main function
main() {
    local command=${1:-help}
    
    case $command in
        deploy)
            check_prerequisites
            create_config_files
            deploy
            ;;
        test)
            run_tests
            ;;
        load-test)
            run_load_test "${2:-100}" "${3:-1000}"
            ;;
        security)
            run_security_scan
            ;;
        status)
            show_status
            ;;
        metrics)
            show_metrics
            ;;
        optimize)
            optimize_performance
            ;;
        cleanup)
            cleanup
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 