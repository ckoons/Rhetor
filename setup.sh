#!/bin/bash
# Setup script for Rhetor component

# ANSI color codes for terminal output
BLUE="\033[94m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BLUE}${BOLD}Setting up Rhetor environment...${RESET}"

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${RESET}"
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create virtual environment.${RESET}"
        exit 1
    fi
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${RESET}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${RESET}"
pip install --upgrade pip
pip install -r requirements.txt

# Install Rhetor in development mode
echo -e "${YELLOW}Installing Rhetor in development mode...${RESET}"
pip install -e .

# Add parent directory to PYTHONPATH temporarily for imports
export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"

# Create directory structure if it doesn't exist
mkdir -p ~/.tekton/data/rhetor/templates
mkdir -p ~/.tekton/data/rhetor/conversations

echo -e "${GREEN}${BOLD}Rhetor environment setup complete!${RESET}"
echo -e "${BLUE}To activate the environment, run:${RESET}"
echo -e "source $SCRIPT_DIR/venv/bin/activate"
