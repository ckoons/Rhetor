"""
Main entry point for running the Rhetor API server.
"""

import os
import sys
import logging
import argparse

# Add Tekton root to path if not already present
tekton_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if tekton_root not in sys.path:
    sys.path.insert(0, tekton_root)

from rhetor.api.app import run_server
from shared.utils.env_config import get_component_config
from shared.utils.logging_setup import setup_component_logging

# Set up logging
logger = setup_component_logging("rhetor")

def main():
    """Run the Rhetor API server."""
    # Get port configuration
    config = get_component_config()
    default_port = config.rhetor.port if hasattr(config, 'rhetor') else int(os.environ.get("RHETOR_PORT", 8003))
    
    parser = argparse.ArgumentParser(description="Rhetor LLM Manager")
    parser.add_argument(
        "--port", "-p", 
        type=int, 
        default=default_port,
        help=f"Port to run the server on (default: {default_port})"
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