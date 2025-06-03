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

from shared.utils.socket_server import run_component_server
from shared.utils.env_config import get_component_config
from shared.utils.logging_setup import setup_component_logging

# Set up logging
logger = setup_component_logging("rhetor")

def main():
    """Run the Rhetor API server."""
    # Get port configuration
    config = get_component_config()
    default_port = config.rhetor.port if hasattr(config, 'rhetor') else int(os.environ.get("RHETOR_PORT"))
    
    parser = argparse.ArgumentParser(description="Rhetor LLM Manager")
    parser.add_argument(
        "--port", "-p", 
        type=int, 
        default=default_port,
        help=f"Port to run the server on (default: {default_port})"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Starting Rhetor API server on port {args.port}")
    run_component_server(
        component_name="rhetor",
        app_module="rhetor.api.app",
        default_port=args.port,
        reload=False
    )

if __name__ == "__main__":
    main()