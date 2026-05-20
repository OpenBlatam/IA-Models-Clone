#!/bin/bash
# Chaos Engineering Script
# Tests system resilience through controlled failures

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHAOS_DIR="/opt/chaos-engineering"
APP_PORT=8020

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# CPU stress test
cpu_chaos() {
    local duration="${1:-60}"
    local load="${2:-50}"
    
    log_warn "🔥 Injecting CPU chaos: ${load}% load for ${duration}s"
    
    # Create CPU load
    stress-ng --cpu 1 --cpu-load $load --timeout ${duration}s > /dev/null 2>&1 &
    local pid=$!
    
    # Monitor application
    monitor_during_chaos "$duration" "CPU"
    
    wait $pid
    log_info "✅ CPU chaos completed"
}

# Memory stress test
memory_chaos() {
    local duration="${1:-60}"
    local memory="${2:-512}"
    
    log_warn "🔥 Injecting memory chaos: ${memory}MB for ${duration}s"
    
    stress-ng --vm 1 --vm-bytes ${memory}M --timeout ${duration}s > /dev/null 2>&1 &
    local pid=$!
    
    monitor_during_chaos "$duration" "Memory"
    
    wait $pid
    log_info "✅ Memory chaos completed"
}

# Network chaos (latency, packet loss)
network_chaos() {
    local duration="${1:-60}"
    local latency="${2:-100}"
    local loss="${3:-5}"
    
    log_warn "🔥 Injecting network chaos: ${latency}ms latency, ${loss}% packet loss for ${duration}s"
    
    # Install tc if not available
    if ! command -v tc > /dev/null 2>&1; then
        sudo apt-get install -y iproute2
    fi
    
    # Add network delay and packet loss
    sudo tc qdisc add dev eth0 root netem delay ${latency}ms loss ${loss}% 2>/dev/null || \
    sudo tc qdisc change dev eth0 root netem delay ${latency}ms loss ${loss}%
    
    monitor_during_chaos "$duration" "Network"
    
    # Remove network chaos
    sudo tc qdisc del dev eth0 root 2>/dev/null || true
    
    log_info "✅ Network chaos completed"
}

# Disk I/O chaos
disk_chaos() {
    local duration="${1:-60}"
    
    log_warn "🔥 Injecting disk I/O chaos for ${duration}s"
    
    stress-ng --io 4 --timeout ${duration}s > /dev/null 2>&1 &
    local pid=$!
    
    monitor_during_chaos "$duration" "Disk"
    
    wait $pid
    log_info "✅ Disk chaos completed"
}

# Service failure
service_chaos() {
    local service="${1:-ai-project-generator}"
    local duration="${2:-30}"
    
    log_warn "🔥 Injecting service chaos: stopping $service for ${duration}s"
    
    # Stop service
    sudo systemctl stop "$service" 2>/dev/null || true
    
    # Monitor
    sleep "$duration"
    
    # Restart service
    sudo systemctl start "$service" 2>/dev/null || true
    
    # Wait for recovery
    sleep 10
    
    # Health check
    if curl -f -s "http://localhost:$APP_PORT/health" > /dev/null 2>&1; then
        log_info "✅ Service recovered successfully"
    else
        log_error "❌ Service failed to recover"
        return 1
    fi
}

# Container chaos (Docker)
container_chaos() {
    local container="${1:-ai-project-generator}"
    local duration="${2:-30}"
    
    log_warn "🔥 Injecting container chaos: stopping $container for ${duration}s"
    
    # Stop container
    sudo docker stop "$container" 2>/dev/null || true
    
    # Monitor
    sleep "$duration"
    
    # Start container
    sudo docker start "$container" 2>/dev/null || true
    
    # Wait for recovery
    sleep 10
    
    # Health check
    if curl -f -s "http://localhost:$APP_PORT/health" > /dev/null 2>&1; then
        log_info "✅ Container recovered successfully"
    else
        log_error "❌ Container failed to recover"
        return 1
    fi
}

# Monitor during chaos
monitor_during_chaos() {
    local duration="$1"
    local chaos_type="$2"
    
    log_info "Monitoring application during $chaos_type chaos..."
    
    local start_time=$(date +%s)
    local end_time=$((start_time + duration))
    local success_count=0
    local failure_count=0
    
    while [ $(date +%s) -lt $end_time ]; do
        if curl -f -s -m 2 "http://localhost:$APP_PORT/health" > /dev/null 2>&1; then
            success_count=$((success_count + 1))
        else
            failure_count=$((failure_count + 1))
        fi
        sleep 2
    done
    
    local total=$((success_count + failure_count))
    local success_rate=$((success_count * 100 / total))
    
    log_info "Chaos Test Results ($chaos_type):"
    log_info "  Success: $success_count/$total ($success_rate%)"
    log_info "  Failures: $failure_count/$total"
    
    if [ $success_rate -lt 80 ]; then
        log_error "❌ Application resilience below threshold (80%)"
        return 1
    else
        log_info "✅ Application handled chaos well"
        return 0
    fi
}

# Install chaos tools
install_chaos_tools() {
    log_info "Installing chaos engineering tools..."
    
    sudo apt-get update
    sudo apt-get install -y stress-ng iproute2
    
    log_info "✅ Chaos tools installed"
}

# Run full chaos test suite
run_chaos_suite() {
    log_info "=== Running Chaos Engineering Test Suite ==="
    
    local results_file="/tmp/chaos_results_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "=== Chaos Engineering Test Results ==="
        echo "Date: $(date)"
        echo ""
        
        echo "=== CPU Chaos Test ==="
        cpu_chaos 60 50
        echo ""
        
        echo "=== Memory Chaos Test ==="
        memory_chaos 60 512
        echo ""
        
        echo "=== Network Chaos Test ==="
        network_chaos 60 100 5
        echo ""
        
        echo "=== Disk I/O Chaos Test ==="
        disk_chaos 60
        echo ""
        
        echo "=== Service Chaos Test ==="
        service_chaos "ai-project-generator" 30
        echo ""
        
    } | tee "$results_file"
    
    log_info "✅ Chaos test suite completed: $results_file"
}

# Main function
main() {
    case "${1:-help}" in
        cpu)
            install_chaos_tools
            cpu_chaos "${2:-60}" "${3:-50}"
            ;;
        memory)
            install_chaos_tools
            memory_chaos "${2:-60}" "${3:-512}"
            ;;
        network)
            install_chaos_tools
            network_chaos "${2:-60}" "${3:-100}" "${4:-5}"
            ;;
        disk)
            install_chaos_tools
            disk_chaos "${2:-60}"
            ;;
        service)
            service_chaos "${2:-ai-project-generator}" "${3:-30}"
            ;;
        container)
            container_chaos "${2:-ai-project-generator}" "${3:-30}"
            ;;
        suite)
            install_chaos_tools
            run_chaos_suite
            ;;
        install)
            install_chaos_tools
            ;;
        *)
            echo "Usage: $0 {cpu|memory|network|disk|service|container|suite|install} [args...]"
            echo ""
            echo "Examples:"
            echo "  $0 cpu 60 50          # CPU chaos: 60s, 50% load"
            echo "  $0 memory 60 512       # Memory chaos: 60s, 512MB"
            echo "  $0 network 60 100 5    # Network chaos: 60s, 100ms latency, 5% loss"
            echo "  $0 suite               # Run full chaos test suite"
            exit 1
            ;;
    esac
}

main "$@"



