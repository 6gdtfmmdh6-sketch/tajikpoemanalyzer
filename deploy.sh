#!/bin/bash
# Tajik Poetry Analyzer - Deploy Script

set -e

echo "🚀 Deploying Tajik Poetry Analyzer..."

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}✗ Docker is not installed${NC}"
    exit 1
fi

# Create directories
mkdir -p data exports uploads

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
    echo -e "${GREEN}✓ Tajik Poetry Analyzer running at http://localhost:8501${NC}"
else
    echo -e "${YELLOW}Service starting... check http://localhost:8501 in a few seconds${NC}"
fi

echo ""
echo "Commands:"
echo "  docker-compose logs -f    # View logs"
echo "  docker-compose down       # Stop"
echo "  docker-compose restart    # Restart"
