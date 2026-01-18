#!/bin/bash
# Tajik Poetry Analyzer - Desktop Launcher
# Double-click this file to start the application

cd "$(dirname "$0")"

echo "Starting Tajik Poetry Analyzer..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Docker is not running. Starting Docker Desktop..."
    open -a Docker
    
    # Wait for Docker to start
    echo "Waiting for Docker to start..."
    while ! docker info > /dev/null 2>&1; do
        sleep 2
    done
    echo "Docker is ready."
fi

# Build and run with Docker
echo "Building and starting the application..."
./deploy.sh

# Keep terminal open on error
read -p "Press Enter to close..."
