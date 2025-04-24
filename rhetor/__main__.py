"""
Main entry point for running the Rhetor API server.
"""

import os
import sys
import logging
import argparse

from rhetor.api.app import run_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rhetor")

def main():
    """Run the Rhetor API server."""
    parser = argparse.ArgumentParser(description="Rhetor LLM Manager")
    parser.add_argument(
        "--port", "-p", 
        type=int, 
        default=int(os.environ.get("RHETOR_PORT", 8300)),
        help="Port to run the server on (default: 8300)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.environ.get("RHETOR_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--log-level", 
        type=str, 
        default=os.environ.get("RHETOR_LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level (default: info)"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting Rhetor API server on {args.host}:{args.port}")
    run_server(host=args.host, port=args.port, log_level=args.log_level)

if __name__ == "__main__":
    main()