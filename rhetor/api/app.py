"""
FastAPI application for the Rhetor API that provides both HTTP and WebSocket interfaces.

This module provides a single-port API for LLM interactions, template management,
and prompt engineering capabilities.
"""

import os
import sys
import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from pydantic import Field
from tekton.models import TektonBaseModel

# Add Tekton root to path if not already present
tekton_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if tekton_root not in sys.path:
    sys.path.insert(0, tekton_root)

# Import shared utilities
from shared.utils.hermes_registration import HermesRegistration, heartbeat_loop
from shared.utils.logging_setup import setup_component_logging
from shared.utils.env_config import get_component_config
from shared.utils.errors import StartupError
from shared.utils.startup import component_startup, StartupMetrics
from shared.utils.shutdown import GracefulShutdown
from shared.utils.health_check import create_health_response
from shared.api import (
    create_standard_routers,
    mount_standard_routers,
    create_ready_endpoint,
    create_discovery_endpoint,
    get_openapi_configuration,
    EndpointInfo
)

from ..core.llm_client import LLMClient
from ..core.model_router import ModelRouter
from ..core.context_manager import ContextManager
from ..core.prompt_engine import PromptEngine
from ..core.template_manager import TemplateManager
from ..core.prompt_registry import PromptRegistry
from ..core.budget_manager import BudgetManager, BudgetPolicy, BudgetPeriod
from ..core.specialist_router import SpecialistRouter
from ..core.ai_specialist_manager import AISpecialistManager
from ..core.ai_messaging_integration import AIMessagingIntegration
from ..core.anthropic_max_config import AnthropicMaxConfig
from ..templates import system_prompts

# Set up logging
logger = setup_component_logging("rhetor")

# Component configuration
COMPONENT_NAME = "rhetor"
COMPONENT_VERSION = "0.1.0"
COMPONENT_DESCRIPTION = "LLM orchestration and management service"

# Initialize core components (will be set in lifespan)
llm_client = None
model_router = None
specialist_router = None
ai_specialist_manager = None
ai_messaging_integration = None
context_manager = None
prompt_engine = None
template_manager = None
prompt_registry = None
budget_manager = None
anthropic_max_config = None
start_time = None
is_registered_with_hermes = False

# Global variables for Hermes registration and heartbeat
hermes_registration = None
heartbeat_task = None
mcp_bridge = None
rhetor_port = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for Rhetor"""
    global llm_client, model_router, specialist_router, ai_specialist_manager, ai_messaging_integration
    global context_manager, prompt_engine, template_manager, prompt_registry, budget_manager
    global anthropic_max_config, start_time, is_registered_with_hermes
    global hermes_registration, heartbeat_task, mcp_bridge, rhetor_port
    
    # Startup
    logger.info("Starting Rhetor initialization...")
    
    async def rhetor_startup():
        """Initialize Rhetor components"""
        global llm_client, model_router, specialist_router, ai_specialist_manager, ai_messaging_integration
        global context_manager, prompt_engine, template_manager, prompt_registry, budget_manager
        global anthropic_max_config, start_time, is_registered_with_hermes
        global hermes_registration, heartbeat_task, mcp_bridge, rhetor_port
        
        try:
            # Track startup time
            import time
            start_time = time.time()
            
            # Get configuration
            config = get_component_config()
            port = config.rhetor.port if hasattr(config, 'rhetor') else int(os.environ.get("RHETOR_PORT"))
            rhetor_port = port  # Store globally for health check
            
            # Register with Hermes
            hermes_registration = HermesRegistration()
            hermes_url = os.environ.get("HERMES_URL", "http://localhost:8001")
            
            try:
                is_registered_with_hermes = await hermes_registration.register_component(
                    component_name="rhetor",
                    port=port,
                    version="0.1.0",
                    capabilities=["llm_orchestration", "template_management", "prompt_engineering"],
                    metadata={"description": "LLM orchestration and management service"}
                )
                
                if is_registered_with_hermes:
                    logger.info("Successfully registered with Hermes")
                    
                    # Start heartbeat task
                    heartbeat_task = asyncio.create_task(
                        heartbeat_loop(hermes_registration, "rhetor", interval=30)
                    )
                    logger.info("Started Hermes heartbeat task")
                    
                    # Store registration in app state
                    app.state.hermes_registration = hermes_registration
                else:
                    logger.warning("Failed to register with Hermes")
            except Exception as e:
                logger.warning(f"Error during Hermes registration: {e}")
            
            # Initialize core components
            llm_client = LLMClient()
            logger.info("LLM client initialized successfully")
            
            # Initialize template manager first
            template_data_dir = os.path.join(
                os.environ.get('TEKTON_DATA_DIR', 
                              os.path.join(os.environ.get('TEKTON_ROOT', os.path.expanduser('~')), '.tekton', 'data')),
                'rhetor', 'templates'
            )
            template_manager = TemplateManager(template_data_dir)
            logger.info("Template manager initialized")
            
            # Initialize prompt registry
            prompt_data_dir = os.path.join(
                os.environ.get('TEKTON_DATA_DIR',
                              os.path.join(os.environ.get('TEKTON_ROOT', os.path.expanduser('~')), '.tekton', 'data')),
                'rhetor', 'prompts'
            )
            prompt_registry = PromptRegistry(prompt_data_dir)
            logger.info("Prompt registry initialized")
            
            # Initialize enhanced context manager with token counting
            context_manager = ContextManager(llm_client=llm_client)
            logger.info("Initializing context manager...")
            await asyncio.wait_for(context_manager.initialize(), timeout=5.0)
            logger.info("Context manager initialized successfully")
            
            # Initialize Anthropic Max configuration
            anthropic_max_config = AnthropicMaxConfig()
            logger.info(f"Anthropic Max configuration initialized - enabled: {anthropic_max_config.enabled}")
            
            # Initialize budget manager for cost tracking and budget enforcement
            budget_manager = BudgetManager()
            
            # Apply Anthropic Max budget override if enabled
            if anthropic_max_config.enabled:
                max_budget = anthropic_max_config.get_budget_override()
                if max_budget:
                    logger.info("Applying Anthropic Max budget override - unlimited tokens")
                    # Budget manager will still track usage but not enforce limits
            
            logger.info("Budget manager initialized")
            
            # Initialize model router with budget manager
            model_router = ModelRouter(llm_client, budget_manager=budget_manager)
            logger.info("Model router initialized")
            
            # Initialize specialist router for AI specialist management
            specialist_router = SpecialistRouter(llm_client, budget_manager=budget_manager)
            logger.info("Specialist router initialized")
            
            # Initialize AI specialist manager
            ai_specialist_manager = AISpecialistManager(llm_client, specialist_router)
            specialist_router.set_specialist_manager(ai_specialist_manager)
            logger.info("AI specialist manager initialized")
            
            # Start core AI specialists
            try:
                core_results = await ai_specialist_manager.start_core_specialists()
                logger.info(f"Core AI specialists started: {core_results}")
            except Exception as e:
                logger.warning(f"Failed to start core AI specialists: {e}")
            
            # Initialize AI messaging integration with Hermes
            ai_messaging_integration = AIMessagingIntegration(ai_specialist_manager, hermes_url, specialist_router)
            try:
                await ai_messaging_integration.initialize()
                logger.info("AI messaging integration initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize AI messaging integration: {e}")
            
            # Initialize prompt engine with template manager integration
            prompt_engine = PromptEngine(template_manager)
            logger.info("Prompt engine initialized")
            
            # Initialize Hermes MCP Bridge
            try:
                from rhetor.core.mcp.hermes_bridge import RhetorMCPBridge
                mcp_bridge = RhetorMCPBridge(llm_client)
                await mcp_bridge.initialize()
                app.state.mcp_bridge = mcp_bridge
                logger.info("Initialized Hermes MCP Bridge for FastMCP tools")
            except Exception as e:
                logger.warning(f"Failed to initialize MCP Bridge: {e}")
            
            # Initialize MCP Tools Integration with live components
            try:
                from rhetor.core.mcp.init_integration import (
                    initialize_mcp_integration,
                    setup_hermes_subscriptions,
                    test_mcp_integration
                )
                
                # Create the integration
                mcp_integration = initialize_mcp_integration(
                    specialist_manager=ai_specialist_manager,
                    messaging_integration=ai_messaging_integration,
                    hermes_url=hermes_url
                )
                
                # Set up Hermes subscriptions for cross-component messaging
                await setup_hermes_subscriptions(mcp_integration)
                
                # Test the integration if in debug mode
                if logger.isEnabledFor(logging.DEBUG):
                    await test_mcp_integration(mcp_integration)
                
                app.state.mcp_integration = mcp_integration
                logger.info("MCP Tools Integration initialized with live components")
                
            except Exception as e:
                logger.warning(f"Failed to initialize MCP Tools Integration: {e}")
            
            logger.info(f"Rhetor API initialized successfully on port {port}")
            
        except Exception as e:
            logger.error(f"Error during Rhetor startup: {e}", exc_info=True)
            raise StartupError(str(e), "rhetor", "STARTUP_FAILED")
    
    # Execute startup with metrics
    try:
        metrics = await component_startup("rhetor", rhetor_startup, timeout=30)
        logger.info(f"Rhetor started successfully in {metrics.total_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to start Rhetor: {e}")
        raise
    
    # Create shutdown handler
    shutdown = GracefulShutdown("rhetor")
    
    # Register cleanup tasks
    async def cleanup_hermes():
        """Cleanup Hermes registration"""
        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if hermes_registration and hermes_registration.is_registered:
            await hermes_registration.deregister("rhetor")
            logger.info("Deregistered from Hermes")
    
    async def cleanup_components():
        """Cleanup Rhetor components"""
        try:
            if ai_messaging_integration:
                await ai_messaging_integration.cleanup()
                logger.info("AI messaging integration cleaned up")
            
            if context_manager:
                await context_manager.cleanup()
                logger.info("Context manager cleaned up")
            
            if llm_client:
                await llm_client.cleanup()
                logger.info("LLM client cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up Rhetor components: {e}")
    
    async def cleanup_mcp_bridge():
        """Cleanup MCP bridge"""
        global mcp_bridge
        if mcp_bridge:
            try:
                await mcp_bridge.shutdown()
                logger.info("MCP bridge cleaned up")
            except Exception as e:
                logger.warning(f"Error cleaning up MCP bridge: {e}")
    
    shutdown.register_cleanup(cleanup_hermes)
    shutdown.register_cleanup(cleanup_components)
    shutdown.register_cleanup(cleanup_mcp_bridge)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Rhetor LLM Orchestration API")
    await shutdown.shutdown_sequence(timeout=10)
    
    # Socket release delay for macOS
    await asyncio.sleep(0.5)


# Initialize FastAPI app with standard configuration
app = FastAPI(
    **get_openapi_configuration(
        component_name=COMPONENT_NAME,
        component_version=COMPONENT_VERSION,
        component_description=COMPONENT_DESCRIPTION
    ),
    lifespan=lifespan
)

# Add FastMCP endpoints
try:
    from .fastmcp_endpoints import mcp_router
    app.include_router(mcp_router)
    logger.info("FastMCP endpoints added to Rhetor API")
except ImportError as e:
    logger.warning(f"FastMCP endpoints not available: {e}")

# Add AI Specialist endpoints
try:
    from .ai_specialist_endpoints import router as ai_router
    app.include_router(ai_router)
    logger.info("AI Specialist endpoints added to Rhetor API")
except ImportError as e:
    logger.warning(f"AI Specialist endpoints not available: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create standard routers
routers = create_standard_routers(COMPONENT_NAME)


# Root endpoint
@routers.root.get("/")
async def root():
    """Root endpoint providing basic API information."""
    return {
        "name": f"{COMPONENT_NAME} LLM Orchestration API",
        "version": COMPONENT_VERSION,
        "status": "running",
        "description": COMPONENT_DESCRIPTION,
        "documentation": "/api/v1/docs"
    }


# Health check endpoint
@routers.root.get("/health")
async def health_check():
    """Check the health of the Rhetor service following Tekton standards."""
    try:
        return create_health_response(
            component_name=COMPONENT_NAME,
            port=rhetor_port,
            version=COMPONENT_VERSION,
            status="healthy" if llm_client and llm_client.is_initialized else "unhealthy",
            registered=is_registered_with_hermes,
            details={
                "llm_client": llm_client is not None and llm_client.is_initialized,
                "context_manager": context_manager is not None,
                "template_manager": template_manager is not None,
                "budget_manager": budget_manager is not None,
                "specialist_manager": ai_specialist_manager is not None,
                "prompt_engine": prompt_engine is not None,
                "uptime": time.time() - start_time if start_time else 0
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


# Ready endpoint
routers.root.add_api_route(
    "/ready",
    create_ready_endpoint(
        component_name=COMPONENT_NAME,
        component_version=COMPONENT_VERSION,
        start_time=start_time or 0,
        readiness_check=lambda: llm_client is not None and context_manager is not None
    ),
    methods=["GET"]
)

# API discovery endpoint
routers.root.add_api_route(
    "/api",
    create_discovery_endpoint(
        COMPONENT_NAME,
        COMPONENT_VERSION,
        COMPONENT_DESCRIPTION,
        endpoints=[
            EndpointInfo(
                path="/api/v1/llm",
                method="POST",
                description="Generate LLM response"
            ),
            EndpointInfo(
                path="/api/v1/models",
                method="GET",
                description="List available models"
            ),
            EndpointInfo(
                path="/api/v1/models",
                method="POST",
                description="Configure model settings"
            ),
            EndpointInfo(
                path="/api/v1/templates",
                method="GET",
                description="List templates"
            ),
            EndpointInfo(
                path="/api/v1/templates",
                method="POST",
                description="Create template"
            ),
            EndpointInfo(
                path="/api/v1/templates",
                method="PUT",
                description="Update template"
            ),
            EndpointInfo(
                path="/api/v1/templates",
                method="DELETE",
                description="Delete template"
            ),
            EndpointInfo(
                path="/api/v1/prompts",
                method="GET",
                description="List prompts"
            ),
            EndpointInfo(
                path="/api/v1/prompts",
                method="POST",
                description="Create prompt"
            ),
            EndpointInfo(
                path="/api/v1/context",
                method="GET",
                description="Get context"
            ),
            EndpointInfo(
                path="/api/v1/context",
                method="POST",
                description="Create context"
            ),
            EndpointInfo(
                path="/api/v1/context",
                method="PUT",
                description="Update context"
            ),
            EndpointInfo(
                path="/api/v1/context",
                method="DELETE",
                description="Delete context"
            ),
            EndpointInfo(
                path="/ws",
                method="GET",
                description="WebSocket for streaming"
            )
        ],
        capabilities=[
            "llm_orchestration",
            "multi_provider_support",
            "intelligent_routing",
            "context_management",
            "prompt_templates",
            "streaming",
            "model_management"
        ]
    ),
    methods=["GET"]
)


# Model for LLM generation requests
class LLMRequest(TektonBaseModel):
    """Request model for LLM generation."""
    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: Optional[str] = Field(None, description="Specific model to use")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the request")
    specialist: Optional[str] = Field(None, description="AI specialist to route the request to")


class LLMResponse(TektonBaseModel):
    """Response model for LLM generation."""
    content: str = Field(..., description="Generated content")
    model: str = Field(..., description="Model used for generation")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional response metadata")


# LLM generation endpoint
@routers.v1.post("/llm", response_model=LLMResponse)
async def generate_llm_response(request: LLMRequest) -> LLMResponse:
    """Generate a response from the LLM."""
    try:
        # Route through AI specialist if specified
        if request.specialist and ai_specialist_manager:
            specialist = ai_specialist_manager.get_specialist(request.specialist)
            if not specialist:
                raise HTTPException(status_code=404, detail=f"Specialist '{request.specialist}' not found")
            
            # Send message to specialist
            response = await ai_specialist_manager.send_message_to_specialist(
                specialist_id=request.specialist,
                message=request.prompt,
                context=request.context
            )
            
            return LLMResponse(
                content=response["response"],
                model=response.get("model", "unknown"),
                usage=response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
                metadata={
                    "specialist": request.specialist,
                    "conversation_id": response.get("conversation_id")
                }
            )
        
        # Regular LLM routing
        if request.model:
            # Use specific model
            response = await llm_client.generate(
                prompt=request.prompt,
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        else:
            # Use model router for intelligent routing
            response = await model_router.route_request(
                prompt=request.prompt,
                context=request.context,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
        
        # Track budget usage
        if budget_manager:
            await budget_manager.track_usage(
                model=response["model"],
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
                cost=response.get("cost", 0.0)
            )
        
        return LLMResponse(
            content=response["content"],
            model=response["model"],
            usage=response["usage"],
            metadata=response.get("metadata")
        )
        
    except Exception as e:
        logger.error(f"Error generating LLM response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Mount standard routers
mount_standard_routers(app, routers)


# WebSocket endpoint for streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming LLM responses."""
    await websocket.accept()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Validate request
            if "prompt" not in data:
                await websocket.send_json({
                    "error": "Missing required field: prompt"
                })
                continue
            
            # Generate streaming response
            try:
                model = data.get("model")
                specialist = data.get("specialist")
                
                # Route through specialist if specified
                if specialist and ai_specialist_manager:
                    # Stream from specialist
                    async for chunk in ai_specialist_manager.stream_to_specialist(
                        specialist_id=specialist,
                        message=data["prompt"],
                        context=data.get("context")
                    ):
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk.get("content", ""),
                            "metadata": chunk.get("metadata")
                        })
                    
                    await websocket.send_json({"type": "complete"})
                else:
                    # Regular streaming
                    async for chunk in llm_client.stream(
                        prompt=data["prompt"],
                        model=model,
                        temperature=data.get("temperature", 0.7),
                        max_tokens=data.get("max_tokens")
                    ):
                        await websocket.send_json({
                            "type": "chunk",
                            "content": chunk["content"]
                        })
                    
                    await websocket.send_json({"type": "complete"})
                    
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# Template management endpoints
@routers.v1.get("/templates")
async def list_templates():
    """List all available templates."""
    try:
        templates = template_manager.list_templates()
        return {"templates": templates}
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@routers.v1.get("/templates/{template_id}")
async def get_template(template_id: str = Path(..., description="Template ID")):
    """Get a specific template by ID."""
    try:
        template = template_manager.get_template(template_id)
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Model management endpoints
@routers.v1.get("/models")
async def list_models():
    """List all available models."""
    try:
        return {
            "models": llm_client.get_available_models(),
            "default": llm_client.default_model
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@routers.v1.post("/models/default")
async def set_default_model(model: str):
    """Set the default model."""
    try:
        llm_client.set_default_model(model)
        return {"message": f"Default model set to {model}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error setting default model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Main entry point
if __name__ == "__main__":
    # Get configuration
    config = get_component_config()
    port = config.rhetor.port if hasattr(config, 'rhetor') else int(os.environ.get("RHETOR_PORT", 8003))
    host = os.environ.get("RHETOR_HOST", "0.0.0.0")
    
    # Run the server
    uvicorn.run(
        "rhetor.api.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )