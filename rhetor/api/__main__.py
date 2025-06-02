"""Main entry point for running Rhetor API server."""
import uvicorn
import os
from rhetor.api.app import app

if __name__ == "__main__":
    port = int(os.environ.get("RHETOR_PORT", "8003"))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )