"""
Simplified Rhetor API - Minimal working version
"""
import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rhetor.api")

# Create FastAPI app
app = FastAPI(
    title="Rhetor LLM Management API",
    description="Simplified Rhetor API for LLM management",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock LLM providers for now
MOCK_PROVIDERS = {
    "ollama": {
        "id": "ollama",
        "name": "Ollama",
        "available": True,
        "models": ["llama3", "mistral", "codellama"]
    },
    "anthropic": {
        "id": "anthropic", 
        "name": "Anthropic",
        "available": True,
        "models": ["claude-3-sonnet", "claude-3-haiku"]
    },
    "openai": {
        "id": "openai",
        "name": "OpenAI", 
        "available": True,
        "models": ["gpt-4", "gpt-3.5-turbo"]
    }
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Rhetor LLM Management API",
        "version": "0.1.0",
        "status": "operational"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "providers": {k: v["available"] for k, v in MOCK_PROVIDERS.items()}
    }

@app.get("/providers")
async def get_providers():
    """Get available LLM providers"""
    return {
        "providers": MOCK_PROVIDERS
    }

@app.get("/providers/{provider_id}")
async def get_provider(provider_id: str):
    """Get specific provider info"""
    if provider_id not in MOCK_PROVIDERS:
        raise HTTPException(status_code=404, detail=f"Provider {provider_id} not found")
    return MOCK_PROVIDERS[provider_id]

@app.get("/models")
async def get_models():
    """Get all available models"""
    models = []
    for provider_id, provider in MOCK_PROVIDERS.items():
        for model in provider["models"]:
            models.append({
                "id": f"{provider_id}/{model}",
                "provider": provider_id,
                "model": model,
                "available": provider["available"]
            })
    return {"models": models}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("RHETOR_PORT", 8003))
    logger.info(f"Starting simplified Rhetor on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)