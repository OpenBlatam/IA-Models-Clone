#!/bin/bash
# Docker build script with optimizations

set -e

ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
BUILD_ARGS=""

echo "Building Docker image for Music Analyzer AI"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"

# Build arguments based on environment
if [ "$ENVIRONMENT" == "production" ]; then
    BUILD_ARGS="--target production"
elif [ "$ENVIRONMENT" == "development" ]; then
    BUILD_ARGS="--target builder"
fi

# Build image
docker build \
    -f deployment/Dockerfile \
    -t music-analyzer-ai:$VERSION \
    -t music-analyzer-ai:latest \
    $BUILD_ARGS \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    --build-arg ENVIRONMENT=$ENVIRONMENT \
    ..

echo "Build complete!"
echo "Image tags: music-analyzer-ai:$VERSION, music-analyzer-ai:latest"

# Show image size
docker images music-analyzer-ai:$VERSION --format "Image size: {{.Size}}"




