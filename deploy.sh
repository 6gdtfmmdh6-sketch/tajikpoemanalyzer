#!/bin/bash
# Tajik Poetry Analyzer - Deploy Script

set -e

echo "Deploying Tajik Poetry Analyzer..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Create all required directories
mkdir -p data exports uploads tajik_corpus/contributions tajik_corpus/corpus tajik_corpus/exports tajik_poetry_library

# Build and start
echo -e "${YELLOW}Building Docker image...${NC}"
docker-compose build

echo -e "${YELLOW}Starting container...${NC}"
docker-compose up -d

# Wait for startup
echo -e "${YELLOW}Waiting for service...${NC}"
sleep 5

# Check health
if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
    echo -e "${GREEN}Tajik Poetry Analyzer running at http://localhost:8501${NC}"
    
    # Open browser automatically (macOS)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        open http://localhost:8501
    # Linux
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open http://localhost:8501 2>/dev/null || true
    fi
else
    echo -e "${YELLOW}Service starting... check http://localhost:8501 in a few seconds${NC}"
    
    # Try to open browser anyway
    if [[ "$OSTYPE" == "darwin"* ]]; then
        sleep 3
        open http://localhost:8501
    fi
fi

echo ""
echo "Commands:"
echo "  docker-compose logs -f    # View logs"
echo "  docker-compose down       # Stop"
echo "  docker-compose restart    # Restart"
