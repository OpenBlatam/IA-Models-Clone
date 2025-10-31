#!/bin/bash

# Enhanced Blaze AI Quick Start Script
# Get up and running in minutes!

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="blaze-ai-enhanced"

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

# Print header
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  🚀 Enhanced Blaze AI Quick Start${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  Get up and running in minutes!${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

# Check if running as root
check_root() {
    if [ "$EUID" -eq 0 ]; then
        log_error "Please do not run this script as root"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    
    python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_success "Python $python_version detected"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is not installed. Please install pip3 first."
        exit 1
    fi
    
    log_success "pip3 is available"
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        log_success "Virtual environment already exists"
    else
        log_info "Virtual environment will be created"
    fi
}

# Setup Python environment
setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
        log_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    log_info "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    log_success "Python environment ready"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install requirements
    if [ -f "requirements.txt" ]; then
        log_info "Installing from requirements.txt..."
        pip install -r requirements.txt
        log_success "Dependencies installed"
    else
        log_error "requirements.txt not found"
        exit 1
    fi
}

# Quick validation
quick_validation() {
    log_info "Performing quick validation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test imports
    log_info "Testing critical imports..."
    python3 -c "
import fastapi
import uvicorn
import redis
import prometheus_client
print('✅ All critical packages imported successfully!')
"
    
    if [ $? -eq 0 ]; then
        log_success "Package validation passed"
    else
        log_error "Package validation failed"
        exit 1
    fi
    
    # Test configuration
    log_info "Testing configuration..."
    python3 -c "
import yaml
with open('config-enhanced.yaml', 'r') as f:
    config = yaml.safe_load(f)
print('✅ Configuration file is valid')
"
    
    if [ $? -eq 0 ]; then
        log_success "Configuration validation passed"
    else
        log_error "Configuration validation failed"
        exit 1
    fi
}

# Start development server
start_dev_server() {
    log_info "Starting development server..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Check if server is already running
    if curl -s http://localhost:8000/health &> /dev/null; then
        log_warning "Server is already running on port 8000"
        return
    fi
    
    # Start server in background
    log_info "Starting Blaze AI server in background..."
    nohup python3 main.py --dev > server.log 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    log_info "Waiting for server to start..."
    for i in {1..30}; do
        if curl -s http://localhost:8000/health &> /dev/null; then
            log_success "Server started successfully!"
            echo $SERVER_PID > server.pid
            break
        fi
        sleep 1
        if [ $i -eq 30 ]; then
            log_error "Server failed to start within 30 seconds"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    done
}

# Show quick start information
show_quick_start_info() {
    log_success "🎉 Enhanced Blaze AI is ready!"
    
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}  🚀 Quick Start Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    echo -e "\n${BLUE}🌐 Access Your System:${NC}"
    echo -e "  📊 API:          ${GREEN}http://localhost:8000${NC}"
    echo -e "  📖 Documentation: ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "  🏥 Health Check: ${GREEN}http://localhost:8000/health${NC}"
    
    echo -e "\n${BLUE}🧪 Test Your System:${NC}"
    echo -e "  🎯 Interactive Demo: ${GREEN}python3 demo_enhanced_features.py${NC}"
    echo -e "  🧪 Run Tests:        ${GREEN}python3 test_enhanced_features.py${NC}"
    echo -e "  🔍 Validate:         ${GREEN}python3 validate_system.py${NC}"
    
    echo -e "\n${BLUE}📋 Useful Commands:${NC}"
    echo -e "  📋 View logs:        ${GREEN}tail -f server.log${NC}"
    echo -e "  🛑 Stop server:      ${GREEN}kill \$(cat server.pid)${NC}"
    echo -e "  🔄 Restart:          ${GREEN}./quick_start.sh restart${NC}"
    
    echo -e "\n${BLUE}📚 Next Steps:${NC}"
    echo -e "  1. 🌐 Open http://localhost:8000/docs in your browser"
    echo -e "  2. 🎯 Run the demo: python3 demo_enhanced_features.py"
    echo -e "  3. 🚀 Deploy to production: ./deploy.sh"
    echo -e "  4. 📖 Read the full documentation: README_FINAL.md"
    
    echo -e "\n${GREEN}🎯 Your Enhanced Blaze AI system is running!${NC}"
    echo -e "${GREEN}Open http://localhost:8000/docs to explore the API${NC}\n"
}

# Stop server
stop_server() {
    log_info "Stopping server..."
    
    if [ -f "server.pid" ]; then
        SERVER_PID=$(cat server.pid)
        if kill -0 $SERVER_PID 2>/dev/null; then
            kill $SERVER_PID
            log_success "Server stopped"
        else
            log_warning "Server was not running"
        fi
        rm -f server.pid
    else
        log_warning "No server PID file found"
    fi
}

# Restart server
restart_server() {
    log_info "Restarting server..."
    stop_server
    sleep 2
    start_dev_server
    show_quick_start_info
}

# Show status
show_status() {
    log_info "Checking system status..."
    
    # Check if server is running
    if [ -f "server.pid" ]; then
        SERVER_PID=$(cat server.pid)
        if kill -0 $SERVER_PID 2>/dev/null; then
            log_success "Server is running (PID: $SERVER_PID)"
            
            # Check health
            if curl -s http://localhost:8000/health &> /dev/null; then
                log_success "Server is responding to health checks"
            else
                log_warning "Server is running but not responding to health checks"
            fi
        else
            log_warning "Server PID file exists but process is not running"
            rm -f server.pid
        fi
    else
        log_info "Server is not running"
    fi
    
    # Show logs
    if [ -f "server.log" ]; then
        echo -e "\n${BLUE}📋 Recent server logs:${NC}"
        tail -n 10 server.log
    fi
}

# Main function
main() {
    print_header
    
    # Check if running as root
    check_root
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup Python environment
    setup_python_env
    
    # Install dependencies
    install_dependencies
    
    # Quick validation
    quick_validation
    
    # Start development server
    start_dev_server
    
    # Show information
    show_quick_start_info
}

# Handle command line arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTION]"
        echo "Options:"
        echo "  --help, -h     Show this help message"
        echo "  start          Start the development server"
        echo "  stop           Stop the development server"
        echo "  restart        Restart the development server"
        echo "  status         Show system status"
        echo "  install        Install dependencies only"
        echo ""
        echo "Examples:"
        echo "  $0              # Full quick start (default)"
        echo "  $0 start        # Start server only"
        echo "  $0 stop         # Stop server only"
        echo "  $0 restart      # Restart server"
        echo "  $0 status       # Check status"
        ;;
    start)
        log_info "Starting development server..."
        start_dev_server
        show_quick_start_info
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        show_status
        ;;
    install)
        log_info "Installing dependencies only..."
        check_prerequisites
        setup_python_env
        install_dependencies
        quick_validation
        log_success "Dependencies installed successfully!"
        ;;
    "")
        main
        ;;
    *)
        log_error "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
esac
