#!/bin/bash

echo "Building GitHub Autonomous Agent AI for macOS..."
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Error installing dependencies"
        exit 1
    fi
fi

echo ""
echo "Building application..."
npm run build

if [ $? -ne 0 ]; then
    echo "Error building application"
    exit 1
fi

echo ""
echo "Creating macOS installer..."
npm run build:mac

if [ $? -ne 0 ]; then
    echo "Error creating macOS installer"
    exit 1
fi

echo ""
echo "Build completed successfully!"
echo "Installer location: release/GitHub Autonomous Agent AI-*.dmg"
echo ""


