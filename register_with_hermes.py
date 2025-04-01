#!/usr/bin/env python3
"""Register Rhetor with Hermes service registry.

This script registers all Rhetor services with the Hermes service registry.
"""

import os
import sys
import logging
import asyncio

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rhetor_registration")

# Add Rhetor to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Find the parent directory (Tekton root)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import registration helper
try:
    from rhetor.utils.hermes_helper import register_with_hermes
except ImportError as e:
    logger.error(f"Could not import registration helper: {e}")
    logger.error("Make sure to run setup.sh first")
    sys.exit(1)

async def register_rhetor_services():
    """Register all Rhetor services with Hermes."""
    
    # Register prompt service
    prompt_success = await register_with_hermes(
        service_id="rhetor-prompt",
        name="Rhetor Prompt Engineering",
        capabilities=["prompt_engineering", "template_management", "personality_management"],
        metadata={
            "component_type": "core",
            "description": "AI prompt engineering and management"
        }
    )
    
    # Register communication service
    comm_success = await register_with_hermes(
        service_id="rhetor-communication",
        name="Rhetor Communication Engine",
        capabilities=["messaging", "conversation_management"],
        metadata={
            "component_type": "core",
            "description": "AI component communication"
        }
    )
    
    # Display results
    if prompt_success and comm_success:
        logger.info("Successfully registered all Rhetor services with Hermes")
        return True
    elif prompt_success:
        logger.warning("Successfully registered prompt service, but failed to register communication service")
        return False
    elif comm_success:
        logger.warning("Successfully registered communication service, but failed to register prompt service")
        return False
    else:
        logger.error("Failed to register any Rhetor services with Hermes")
        return False

if __name__ == "__main__":
    # Run in the virtual environment if available
    venv_dir = os.path.join(script_dir, "venv")
    if os.path.exists(venv_dir):
        # Activate the virtual environment if not already activated
        if not os.environ.get("VIRTUAL_ENV"):
            print(f"Please run this script within the Rhetor virtual environment:")
            print(f"source {venv_dir}/bin/activate")
            print(f"python {os.path.basename(__file__)}")
            sys.exit(1)
    
    success = asyncio.run(register_rhetor_services())
    sys.exit(0 if success else 1)
