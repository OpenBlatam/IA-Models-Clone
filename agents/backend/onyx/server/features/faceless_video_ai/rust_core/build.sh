#!/bin/bash
# Build Script for Faceless Video Core

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RELEASE=false
INSTALL=false
TEST=false
BENCH=false
CLEAN=false

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --release) RELEASE=true ;;
        --install) INSTALL=true ;;
        --test) TEST=true ;;
        --bench) BENCH=true ;;
        --clean) CLEAN=true ;;
        -h|--help)
            echo "Usage: $0 [--release] [--install] [--test] [--bench] [--clean]"
            exit 0
            ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "🦀 Faceless Video Core - Rust Build Script"
echo "============================================"

# Check prerequisites
echo -e "\n📋 Checking prerequisites..."

if ! command -v rustc &> /dev/null; then
    echo "❌ Rust not found. Install from https://rustup.rs"
    exit 1
fi

if ! command -v maturin &> /dev/null; then
    echo "⚠️ Maturin not found. Installing..."
    pip install maturin
fi

echo "✅ Rust: $(rustc --version)"
echo "✅ Maturin: $(maturin --version)"

if $CLEAN; then
    echo -e "\n🧹 Cleaning build artifacts..."
    cargo clean
    rm -rf target/wheels
    echo "✅ Clean complete"
fi

if $TEST; then
    echo -e "\n🧪 Running tests..."
    cargo test --all-features
    echo "✅ Tests passed"
fi

if $BENCH; then
    echo -e "\n📊 Running benchmarks..."
    cargo bench
    echo "✅ Benchmarks complete"
fi

if $RELEASE; then
    echo -e "\n🔨 Building release..."
    maturin build --release
    WHEEL=$(ls target/wheels/*.whl 2>/dev/null | head -1)
    echo "✅ Built: $(basename $WHEEL)"
else
    echo -e "\n🔨 Building development version..."
    maturin develop
    echo "✅ Development build complete"
fi

if $INSTALL; then
    echo -e "\n📦 Installing wheel..."
    WHEEL=$(ls target/wheels/*.whl 2>/dev/null | head -1)
    if [ -n "$WHEEL" ]; then
        pip install "$WHEEL" --force-reinstall
        echo "✅ Installed: $(basename $WHEEL)"
    else
        echo "❌ No wheel found. Run with --release first"
    fi
fi

echo -e "\n✨ Build complete!"

# Print usage
echo -e "\n📖 Usage in Python:"
echo '
    from faceless_video_core import VideoProcessor, CryptoService, TextProcessor

    # Video processing
    processor = VideoProcessor()
    processor.optimize_video("input.mp4", "high")

    # Encryption
    crypto = CryptoService()
    encrypted = crypto.encrypt("secret data")

    # Text processing
    text = TextProcessor()
    segments = text.process_script("My script...", "en")
'




