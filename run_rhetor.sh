#!/bin/bash
# Rhetor LLM Management System - Launch Script

# ANSI color codes for terminal output
BLUE="\033[94m"
GREEN="\033[92m"
YELLOW="\033[93m"
RED="\033[91m"
BOLD="\033[1m"
RESET="\033[0m"

echo -e "${BLUE}${BOLD}Starting Rhetor LLM Management System...${RESET}"

# Find Tekton root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [[ "$SCRIPT_DIR" == *"/utils" ]]; then
    # Script is running from a symlink in utils
    TEKTON_ROOT=$(cd "$SCRIPT_DIR" && cd "$(readlink "${BASH_SOURCE[0]}" | xargs dirname | xargs dirname)" && pwd)
else
    # Script is running from Rhetor directory
    TEKTON_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

# Ensure we're in the correct directory
cd "$SCRIPT_DIR"

# Set up environment and Python path
source "$TEKTON_ROOT/shared/utils/setup_env.sh"
setup_tekton_env "$SCRIPT_DIR" "$TEKTON_ROOT"

# Create log directories
mkdir -p "$TEKTON_ROOT/.tekton/logs"

# Error handling function
handle_error() {
    echo -e "${RED}Error: $1${RESET}" >&2
    ${TEKTON_ROOT}/scripts/tekton-register unregister --component rhetor
    exit 1
}

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Register with Hermes using tekton-register
echo -e "${YELLOW}Registering Rhetor with Hermes...${RESET}"
${TEKTON_ROOT}/scripts/tekton-register register --component rhetor --config ${TEKTON_ROOT}/config/components/rhetor.yaml &
REGISTER_PID=$!

# Give registration a moment to complete
sleep 2

# Start the Rhetor service
echo -e "${YELLOW}Starting Rhetor API server...${RESET}"
python -m rhetor > "$TEKTON_ROOT/.tekton/logs/rhetor.log" 2>&1 &
RHETOR_PID=$!

# Trap signals for graceful shutdown
trap "${TEKTON_ROOT}/scripts/tekton-register unregister --component rhetor; kill $RHETOR_PID 2>/dev/null; exit" EXIT SIGINT SIGTERM

# Wait for the server to start
echo -e "${YELLOW}Waiting for Rhetor to start...${RESET}"
for i in {1..30}; do
    if curl -s http://localhost:$RHETOR_PORT/health >/dev/null; then
        echo -e "${GREEN}Rhetor started successfully on port $RHETOR_PORT${RESET}"
        echo -e "${GREEN}API available at: http://localhost:$RHETOR_PORT/api${RESET}"
        echo -e "${GREEN}WebSocket available at: ws://localhost:$RHETOR_PORT/ws${RESET}"
        break
    fi
    
    # Check if the process is still running
    if ! kill -0 $RHETOR_PID 2>/dev/null; then
        echo -e "${RED}Rhetor process terminated unexpectedly${RESET}"
        cat "$TEKTON_ROOT/.tekton/logs/rhetor.log"
        handle_error "Rhetor failed to start"
    fi
    
    echo -n "."
    sleep 1
done

# Keep the script running to maintain registration
echo -e "${BLUE}Rhetor is running. Press Ctrl+C to stop.${RESET}"
wait $RHETOR_PID
