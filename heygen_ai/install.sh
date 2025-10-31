#!/bin/bash

# 🚀 HeyGen AI - Automated Installation Script
# ============================================
# This script automates the installation of all dependencies and optimizations

set -e  # Exit on any error

echo "🚀 Starting HeyGen AI Installation..."
echo "====================================="

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

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   print_error "This script should not be run as root"
   exit 1
fi

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.9+ is required. Found: $python_version"
    exit 1
fi

print_success "Python version: $python_version"

# Check if conda is available
if command -v conda &> /dev/null; then
    print_status "Conda found. Using conda for environment management..."
    USE_CONDA=true
else
    print_status "Conda not found. Using venv for environment management..."
    USE_CONDA=false
fi

# Create virtual environment
ENV_NAME="heygen-ai"
if [ "$USE_CONDA" = true ]; then
    print_status "Creating conda environment: $ENV_NAME"
    conda create -n $ENV_NAME python=3.11 -y
    print_success "Conda environment created successfully"
    
    print_status "Activating conda environment..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $ENV_NAME
else
    print_status "Creating virtual environment: $ENV_NAME"
    python3 -m venv $ENV_NAME
    print_success "Virtual environment created successfully"
    
    print_status "Activating virtual environment..."
    source $ENV_NAME/bin/activate
fi

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
print_status "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Flash Attention 2.0
print_status "Installing Flash Attention 2.0..."
if pip install flash-attn --no-build-isolation; then
    print_success "Flash Attention 2.0 installed successfully"
else
    print_warning "Flash Attention 2.0 installation failed. Continuing with standard attention..."
fi

# Install xFormers
print_status "Installing xFormers..."
if pip install xformers; then
    print_success "xFormers installed successfully"
else
    print_warning "xFormers installation failed. Continuing without xFormers..."
fi

# Install Triton
print_status "Installing Triton..."
if pip install triton; then
    print_success "Triton installed successfully"
else
    print_warning "Triton installation failed. Continuing without Triton..."
fi

# Install other dependencies
print_status "Installing remaining dependencies..."
pip install -r requirements.txt

# Verify installation
print_status "Verifying installation..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Test key imports
print_status "Testing key imports..."
python3 -c "
try:
    import transformers
    print('✓ Transformers imported successfully')
except ImportError as e:
    print(f'✗ Transformers import failed: {e}')

try:
    import diffusers
    print('✓ Diffusers imported successfully')
except ImportError as e:
    print(f'✗ Diffusers import failed: {e}')

try:
    import accelerate
    print('✓ Accelerate imported successfully')
except ImportError as e:
    print(f'✗ Accelerate import failed: {e}')

try:
    import xformers
    print('✓ xFormers imported successfully')
except ImportError as e:
    print(f'✗ xFormers import failed: {e}')

try:
    import flash_attn
    print('✓ Flash Attention imported successfully')
except ImportError as e:
    print(f'✗ Flash Attention import failed: {e}')
"

# Create activation script
print_status "Creating activation script..."
if [ "$USE_CONDA" = true ]; then
    cat > activate_heygen.sh << 'EOF'
#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate heygen-ai
echo "🚀 HeyGen AI environment activated!"
echo "Run 'conda deactivate' to exit the environment"
EOF
else
    cat > activate_heygen.sh << 'EOF'
#!/bin/bash
source heygen-ai/bin/activate
echo "🚀 HeyGen AI environment activated!"
echo "Run 'deactivate' to exit the environment"
EOF
fi

chmod +x activate_heygen.sh

# Create deactivation reminder
print_status "Creating deactivation reminder..."
if [ "$USE_CONDA" = true ]; then
    cat > deactivate_heygen.sh << 'EOF'
#!/bin/bash
conda deactivate
echo "👋 HeyGen AI environment deactivated!"
EOF
else
    cat > deactivate_heygen.sh << 'EOF'
#!/bin/bash
deactivate
echo "👋 HeyGen AI environment deactivated!"
EOF
fi

chmod +x deactivate_heygen.sh

# Final instructions
echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "To activate the HeyGen AI environment:"
echo "  source activate_heygen.sh"
echo ""
echo "To deactivate the environment:"
echo "  source deactivate_heygen.sh"
echo ""
echo "To run the demo:"
echo "  python run_refactored_demo.py"
echo ""
echo "To start training:"
echo "  python core/training_manager_refactored.py --config configs/training_config.yaml"
echo ""
echo "📚 Check setup_guide.md for detailed usage instructions"
echo ""

print_success "HeyGen AI is ready to use! 🚀✨"
