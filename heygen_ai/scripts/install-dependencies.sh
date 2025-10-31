#!/bin/bash

# =============================================================================
# HeyGen AI Backend - Dependency Installation Script
# =============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        REQUIRED_VERSION="3.9"
        
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)"; then
            print_success "Python $PYTHON_VERSION found (>= $REQUIRED_VERSION required)"
            return 0
        else
            print_error "Python $PYTHON_VERSION found, but $REQUIRED_VERSION or higher is required"
            return 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9 or higher"
        return 1
    fi
}

# Function to create virtual environment
create_venv() {
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment not found. Run with --create-venv first"
        exit 1
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
    print_success "Pip upgraded"
}

# Function to install dependencies
install_dependencies() {
    local deps_file=$1
    local deps_name=$2
    
    if [ -f "$deps_file" ]; then
        print_status "Installing $deps_name dependencies from $deps_file..."
        pip install -r "$deps_file"
        print_success "$deps_name dependencies installed"
    else
        print_error "Dependencies file $deps_file not found"
        return 1
    fi
}

# Function to install package with optional dependencies
install_package() {
    local extras=$1
    local extras_name=$2
    
    print_status "Installing package with $extras_name dependencies..."
    pip install -e ".[$extras]"
    print_success "Package with $extras_name dependencies installed"
}

# Function to install development tools
install_dev_tools() {
    print_status "Installing development tools..."
    
    # Install pre-commit
    if command_exists pre-commit; then
        print_status "Installing pre-commit hooks..."
        pre-commit install
        print_success "Pre-commit hooks installed"
    fi
    
    # Install additional development tools
    pip install ipython rich typer
    print_success "Development tools installed"
}

# Function to check for security vulnerabilities
check_security() {
    print_status "Checking for security vulnerabilities..."
    
    if command_exists safety; then
        safety check -r requirements.txt
        print_success "Security check completed"
    else
        print_warning "Safety not installed. Install with: pip install safety"
    fi
}

# Function to validate installation
validate_installation() {
    print_status "Validating installation..."
    
    # Test imports
    python3 -c "
import fastapi
import pydantic
import sqlalchemy
import torch
import transformers
print('Core dependencies imported successfully')
"
    
    print_success "Installation validated"
}

# Function to show help
show_help() {
    echo "HeyGen AI Backend - Dependency Installation Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --create-venv          Create virtual environment"
    echo "  --install-prod         Install production dependencies"
    echo "  --install-dev          Install development dependencies"
    echo "  --install-test         Install testing dependencies"
    echo "  --install-minimal      Install minimal dependencies"
    echo "  --install-all          Install all dependencies (production + optional)"
    echo "  --install-package      Install as package with optional dependencies"
    echo "  --dev-tools            Install development tools"
    echo "  --security-check       Check for security vulnerabilities"
    echo "  --validate             Validate installation"
    echo "  --upgrade-pip          Upgrade pip"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --create-venv --install-prod"
    echo "  $0 --install-dev --dev-tools"
    echo "  $0 --install-all --security-check --validate"
    echo ""
}

# Main function
main() {
    local create_venv_flag=false
    local install_prod_flag=false
    local install_dev_flag=false
    local install_test_flag=false
    local install_minimal_flag=false
    local install_all_flag=false
    local install_package_flag=false
    local dev_tools_flag=false
    local security_check_flag=false
    local validate_flag=false
    local upgrade_pip_flag=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --create-venv)
                create_venv_flag=true
                shift
                ;;
            --install-prod)
                install_prod_flag=true
                shift
                ;;
            --install-dev)
                install_dev_flag=true
                shift
                ;;
            --install-test)
                install_test_flag=true
                shift
                ;;
            --install-minimal)
                install_minimal_flag=true
                shift
                ;;
            --install-all)
                install_all_flag=true
                shift
                ;;
            --install-package)
                install_package_flag=true
                shift
                ;;
            --dev-tools)
                dev_tools_flag=true
                shift
                ;;
            --security-check)
                security_check_flag=true
                shift
                ;;
            --validate)
                validate_flag=true
                shift
                ;;
            --upgrade-pip)
                upgrade_pip_flag=true
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # If no arguments provided, show help
    if [ $# -eq 0 ] && [ "$create_venv_flag" = false ] && [ "$install_prod_flag" = false ] && [ "$install_dev_flag" = false ] && [ "$install_test_flag" = false ] && [ "$install_minimal_flag" = false ] && [ "$install_all_flag" = false ] && [ "$install_package_flag" = false ] && [ "$dev_tools_flag" = false ] && [ "$security_check_flag" = false ] && [ "$validate_flag" = false ] && [ "$upgrade_pip_flag" = false ]; then
        show_help
        exit 0
    fi
    
    print_status "Starting HeyGen AI Backend dependency installation..."
    
    # Check Python version
    if ! check_python_version; then
        exit 1
    fi
    
    # Create virtual environment if requested
    if [ "$create_venv_flag" = true ]; then
        create_venv
    fi
    
    # Activate virtual environment if it exists
    if [ -d "venv" ]; then
        activate_venv
    fi
    
    # Upgrade pip if requested
    if [ "$upgrade_pip_flag" = true ]; then
        upgrade_pip
    fi
    
    # Install dependencies based on flags
    if [ "$install_minimal_flag" = true ]; then
        install_dependencies "requirements-minimal.txt" "minimal"
    fi
    
    if [ "$install_prod_flag" = true ]; then
        install_dependencies "requirements.txt" "production"
    fi
    
    if [ "$install_dev_flag" = true ]; then
        install_dependencies "requirements-dev.txt" "development"
    fi
    
    if [ "$install_test_flag" = true ]; then
        install_dependencies "requirements-test.txt" "testing"
    fi
    
    if [ "$install_all_flag" = true ]; then
        install_package "all" "all"
    fi
    
    if [ "$install_package_flag" = true ]; then
        install_package "dev,test,monitoring,ml,video,audio" "optional"
    fi
    
    # Install development tools if requested
    if [ "$dev_tools_flag" = true ]; then
        install_dev_tools
    fi
    
    # Check security if requested
    if [ "$security_check_flag" = true ]; then
        check_security
    fi
    
    # Validate installation if requested
    if [ "$validate_flag" = true ]; then
        validate_installation
    fi
    
    print_success "Dependency installation completed!"
    
    # Show next steps
    echo ""
    print_status "Next steps:"
    echo "  1. Activate virtual environment: source venv/bin/activate"
    echo "  2. Run the application: python -m heygen_ai.main"
    echo "  3. Run tests: pytest"
    echo "  4. Check documentation: mkdocs serve"
    echo ""
}

# Run main function with all arguments
main "$@" 