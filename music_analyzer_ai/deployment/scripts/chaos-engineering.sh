#!/bin/bash
# Chaos Engineering Script for Music Analyzer AI
# Tests system resilience through controlled failures

set -euo pipefail

# Configuration
readonly NAMESPACE="${NAMESPACE:-production}"
readonly DURATION="${CHAOS_DURATION:-300}"
readonly RECOVERY_TIME="${RECOVERY_TIME:-60}"

# Colors
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if kubectl is available
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed"
        exit 1
    fi
}

# Kill random pod
chaos_kill_pod() {
    local deployment="${1:-music-analyzer-ai-backend}"
    
    log_info "Killing random pod from deployment: ${deployment}"
    
    local pod=$(kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/name=music-analyzer-ai \
        -l app.kubernetes.io/component=backend \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    
    if [ -z "${pod}" ]; then
        log_error "No pods found for deployment ${deployment}"
        return 1
    fi
    
    log_warn "Deleting pod: ${pod}"
    kubectl delete pod "${pod}" -n "${NAMESPACE}" --grace-period=30
    
    log_info "Waiting for pod to be recreated..."
    kubectl wait --for=condition=Ready pod \
        -l app.kubernetes.io/name=music-analyzer-ai \
        -l app.kubernetes.io/component=backend \
        -n "${NAMESPACE}" \
        --timeout=300s
    
    log_success "Pod recovered successfully"
}

# Inject network latency
chaos_network_latency() {
    local latency="${1:-100ms}"
    
    log_info "Injecting network latency: ${latency}"
    
    # This would typically use a tool like Chaos Mesh or Litmus
    # For demonstration, we'll use tc (traffic control) if available
    if command -v tc &> /dev/null; then
        log_warn "Network latency injection requires root privileges"
        log_info "Example: tc qdisc add dev eth0 root netem delay ${latency}"
    else
        log_warn "tc not available, skipping network latency injection"
    fi
}

# Inject CPU stress
chaos_cpu_stress() {
    local cpu_percent="${1:-80}"
    
    log_info "Injecting CPU stress: ${cpu_percent}%"
    
    # Create a stress pod
    kubectl run cpu-stress-$(date +%s) \
        --image=containerstack/cpustress \
        --restart=Never \
        --limits=cpu=1000m \
        --requests=cpu=1000m \
        -- -cpu ${cpu_percent} \
        -duration ${DURATION}s \
        -n "${NAMESPACE}" || true
    
    log_info "CPU stress will run for ${DURATION}s"
    sleep ${DURATION}
    
    # Cleanup
    kubectl delete pod -l run=cpu-stress -n "${NAMESPACE}" || true
    log_success "CPU stress test completed"
}

# Inject memory stress
chaos_memory_stress() {
    local memory_mb="${1:-512}"
    
    log_info "Injecting memory stress: ${memory_mb}MB"
    
    # Create a memory stress pod
    kubectl run memory-stress-$(date +%s) \
        --image=polinux/stress \
        --restart=Never \
        --limits=memory=${memory_mb}Mi \
        --requests=memory=${memory_mb}Mi \
        -- stress --vm 1 --vm-bytes ${memory_mb}M --vm-hang 0 \
        -n "${NAMESPACE}" || true
    
    log_info "Memory stress will run for ${DURATION}s"
    sleep ${DURATION}
    
    # Cleanup
    kubectl delete pod -l run=memory-stress -n "${NAMESPACE}" || true
    log_success "Memory stress test completed"
}

# Simulate network partition
chaos_network_partition() {
    log_info "Simulating network partition..."
    
    # This would typically use network policies or service mesh
    log_warn "Network partition requires advanced tooling (Chaos Mesh, Litmus)"
    log_info "Example: Block traffic between pods using NetworkPolicy"
}

# Simulate disk I/O issues
chaos_disk_io() {
    log_info "Simulating disk I/O issues..."
    
    # Create I/O stress pod
    kubectl run io-stress-$(date +%s) \
        --image=polinux/stress \
        --restart=Never \
        -- stress --io 4 --timeout ${DURATION}s \
        -n "${NAMESPACE}" || true
    
    log_info "I/O stress will run for ${DURATION}s"
    sleep ${DURATION}
    
    # Cleanup
    kubectl delete pod -l run=io-stress -n "${NAMESPACE}" || true
    log_success "I/O stress test completed"
}

# Run chaos experiment
run_experiment() {
    local experiment_type="${1:-pod-kill}"
    
    log_info "Starting chaos experiment: ${experiment_type}"
    log_info "Duration: ${DURATION}s"
    log_info "Namespace: ${NAMESPACE}"
    
    case "${experiment_type}" in
        pod-kill)
            chaos_kill_pod
            ;;
        network-latency)
            chaos_network_latency "${2:-100ms}"
            ;;
        cpu-stress)
            chaos_cpu_stress "${2:-80}"
            ;;
        memory-stress)
            chaos_memory_stress "${2:-512}"
            ;;
        network-partition)
            chaos_network_partition
            ;;
        disk-io)
            chaos_disk_io
            ;;
        *)
            log_error "Unknown experiment type: ${experiment_type}"
            echo "Available experiments:"
            echo "  pod-kill          - Kill random pod"
            echo "  network-latency   - Inject network latency"
            echo "  cpu-stress        - Inject CPU stress"
            echo "  memory-stress     - Inject memory stress"
            echo "  network-partition - Simulate network partition"
            echo "  disk-io           - Simulate disk I/O issues"
            exit 1
            ;;
    esac
    
    log_info "Waiting for system recovery..."
    sleep ${RECOVERY_TIME}
    
    # Verify system health
    log_info "Verifying system health..."
    if kubectl get pods -n "${NAMESPACE}" \
        -l app.kubernetes.io/name=music-analyzer-ai \
        --field-selector=status.phase=Running | grep -q Running; then
        log_success "System recovered successfully"
    else
        log_error "System recovery failed"
        return 1
    fi
}

# Main function
main() {
    check_kubectl
    
    if [ $# -eq 0 ]; then
        echo "Usage: $0 <experiment-type> [parameters]"
        echo ""
        echo "Experiments:"
        echo "  pod-kill          - Kill random pod"
        echo "  network-latency   - Inject network latency (default: 100ms)"
        echo "  cpu-stress        - Inject CPU stress (default: 80%)"
        echo "  memory-stress     - Inject memory stress (default: 512MB)"
        echo "  network-partition - Simulate network partition"
        echo "  disk-io           - Simulate disk I/O issues"
        exit 1
    fi
    
    run_experiment "$@"
}

main "$@"




