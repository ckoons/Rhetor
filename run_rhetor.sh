#!/bin/bash
# Script to run the Rhetor LLM Manager

# Default port
PORT=${RHETOR_PORT:-8300}

# Log level
LOG_LEVEL=${RHETOR_LOG_LEVEL:-info}

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3."
    exit 1
fi

# Ensure running from the right directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# Create log directory if it doesn't exist
LOG_DIR="$HOME/.tekton/logs"
mkdir -p "$LOG_DIR"

# Run the server
echo "Starting Rhetor LLM Manager on port $PORT..."
echo "Logs will be written to $LOG_DIR/rhetor.log"

# Add the current directory to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run with logging to file
python3 -m rhetor --port "$PORT" --log-level "$LOG_LEVEL" > "$LOG_DIR/rhetor.log" 2>&1