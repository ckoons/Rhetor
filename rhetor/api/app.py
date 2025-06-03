"""
FastAPI application for the Rhetor API that provides both HTTP and WebSocket interfaces.

This module provides a single-port API for LLM interactions, template management,
and prompt engineering capabilities.
"""

import os
import sys
import asyncio
import json
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

from ..core.llm_client import LLMClient
from ..core.model_router import ModelRouter
from ..core.context_manager import ContextManager
from ..core.prompt_engine import PromptEngine
from ..core.template_manager import TemplateManager
from ..core.prompt_registry import PromptRegistry
from ..core.budget_manager import BudgetManager, BudgetPolicy, BudgetPeriod
from ..templates import system_prompts

# Set up logging
logger = setup_component_logging("rhetor")

# Initialize core components (will be set in lifespan)
llm_client = None
model_router = None
context_manager = None
prompt_engine = None
template_manager = None
prompt_registry = None
budget_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for Rhetor"""
    global llm_client, model_router, context_manager, prompt_engine, template_manager, prompt_registry, budget_manager
    
    # Startup
    logger.info("Starting Rhetor initialization...")
    
    # Get configuration
    config = get_component_config()
    port = config.rhetor.port if hasattr(config, 'rhetor') else int(os.environ.get("RHETOR_PORT", 8003))
    
    try:
        # Initialize LLM client with timeout
        llm_client = LLMClient()
        logger.info("Initializing LLM client...")
        await asyncio.wait_for(llm_client.initialize(), timeout=10.0)
        logger.info("LLM client initialized successfully")
    
        # Initialize template manager
        template_manager = TemplateManager()
        logger.info("Template manager initialized")
        
        # Initialize prompt registry and connect to template manager
        prompt_registry = system_prompts.get_registry()
        logger.info("Prompt registry initialized")
        
        # Initialize enhanced context manager with token counting
        context_manager = ContextManager(llm_client=llm_client)
        logger.info("Initializing context manager...")
        await asyncio.wait_for(context_manager.initialize(), timeout=5.0)
        logger.info("Context manager initialized successfully")
        
        # Initialize budget manager for cost tracking and budget enforcement
        budget_manager = BudgetManager()
        logger.info("Budget manager initialized")
        
    except asyncio.TimeoutError:
        logger.error("Timeout during Rhetor initialization")
        raise StartupError("Timeout during Rhetor initialization")
    except Exception as e:
        logger.error(f"Error during Rhetor startup: {e}")
        raise StartupError(f"Error during Rhetor startup: {e}")
    
    # Initialize model router with budget manager
    model_router = ModelRouter(llm_client, budget_manager=budget_manager)
    logger.info("Model router initialized")
    
    # Initialize prompt engine with template manager integration
    prompt_engine = PromptEngine(template_manager)
    logger.info("Prompt engine initialized")
    
    # Register with Hermes
    hermes_registration = HermesRegistration()
    await hermes_registration.register_component(
        component_name="rhetor",
        port=port,
        version="1.0.0",
        capabilities=["llm_routing", "prompt_management", "context_management", "budget_tracking"],
        metadata={
            "description": "LLM orchestration and management",
            "category": "ai"
        }
    )
    app.state.hermes_registration = hermes_registration
    
    # Start heartbeat task
    if hermes_registration.is_registered:
        heartbeat_task = asyncio.create_task(heartbeat_loop(hermes_registration, "rhetor"))
    
    logger.info(f"Rhetor API initialized successfully on port {port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Rhetor...")
    
    # Cancel heartbeat task if running
    if hermes_registration.is_registered and 'heartbeat_task' in locals():
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
    
    # Cleanup components
    if context_manager:
        await context_manager.cleanup()
    
    if llm_client:
        await llm_client.cleanup()
    
    # Deregister from Hermes
    if hasattr(app.state, "hermes_registration") and app.state.hermes_registration:
        await app.state.hermes_registration.deregister("rhetor")
    
    # Give sockets time to close on macOS
    await asyncio.sleep(0.5)
    
    logger.info("Rhetor shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title="Rhetor LLM Manager", 
    version="1.0.0",
    lifespan=lifespan
)

# Add FastMCP endpoints
try:
    from .fastmcp_endpoints import mcp_router
    app.include_router(mcp_router)
    logger.info("FastMCP endpoints added to Rhetor API")
except ImportError as e:
    logger.warning(f"FastMCP endpoints not available: {e}")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request models
class MessageRequest(TektonBaseModel):
    message: str
    context_id: str = "default"
    task_type: str = "default"
    component: Optional[str] = None
    streaming: bool = False
    options: Optional[Dict[str, Any]] = None
    prompt_id: Optional[str] = None

class StreamRequest(TektonBaseModel):
    message: str
    context_id: str = "default"
    task_type: str = "default"
    component: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    prompt_id: Optional[str] = None

class ChatRequest(TektonBaseModel):
    messages: List[Dict[str, str]]
    context_id: str = "default"
    task_type: str = "default"
    component: Optional[str] = None
    streaming: bool = False
    options: Optional[Dict[str, Any]] = None
    prompt_id: Optional[str] = None

class ProviderModelRequest(TektonBaseModel):
    provider_id: str
    model_id: str

# Template and prompt management models
class TemplateCreateRequest(TektonBaseModel):
    name: str
    content: str
    category: str = "general"
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class TemplateUpdateRequest(TektonBaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class TemplateRenderRequest(TektonBaseModel):
    template_id: str
    variables: Dict[str, Any]
    version_id: Optional[str] = None

class PromptCreateRequest(TektonBaseModel):
    name: str
    component: str
    content: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    is_default: bool = False
    parent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class PromptUpdateRequest(TektonBaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

class PromptCompareRequest(TektonBaseModel):
    prompt_id1: str
    prompt_id2: str
    
# Budget management models
class BudgetLimitRequest(TektonBaseModel):
    period: str  # daily, weekly, monthly
    limit_amount: float
    provider: str = "all"
    enforcement: Optional[str] = None  # ignore, warn, enforce
    
class BudgetPolicyRequest(TektonBaseModel):
    period: str  # daily, weekly, monthly
    policy: str  # ignore, warn, enforce
    provider: str = "all"

@app.get("/")
async def root():
    """Root endpoint - provides basic information"""
    providers = {}
    if llm_client:
        providers = llm_client.get_all_providers()
    
    # Get template and prompt info if available
    template_info = {}
    prompt_info = {}
    if template_manager:
        template_categories = template_manager.get_categories()
        template_count = template_manager.count_templates()
        template_info = {
            "categories": template_categories,
            "count": template_count
        }
    
    if prompt_registry:
        prompt_components = prompt_registry.get_components()
        prompt_count = prompt_registry.count_prompts()
        prompt_info = {
            "components": prompt_components,
            "count": prompt_count
        }
    
    # Get budget info if available
    budget_info = {}
    if budget_manager:
        daily_usage = budget_manager.get_current_usage_total(BudgetPeriod.DAILY)
        daily_limit = budget_manager.get_budget_limit(BudgetPeriod.DAILY)
        
        budget_info = {
            "daily_usage": daily_usage,
            "daily_limit": daily_limit,
            "budget_enabled": True
        }
    
    return {
        "name": "Rhetor LLM Manager",
        "version": "0.1.0",
        "status": "running",
        "endpoints": [
            "/message", "/stream", "/chat", "/ws", "/providers", "/health", 
            "/templates", "/prompts", "/contexts", "/budget"
        ],
        "providers": providers,
        "templates": template_info,
        "prompts": prompt_info,
        "budget": budget_info
    }

@app.get("/health")
async def health():
    """Health check endpoint following Tekton standards"""
    provider_status = {}
    if llm_client:
        providers = llm_client.get_all_providers()
        for provider_id, provider_info in providers.items():
            provider_status[provider_id] = provider_info["available"]
    
    # Get port from config
    config = get_component_config()
    port = config.rhetor.port if hasattr(config, 'rhetor') else int(os.environ.get("RHETOR_PORT", 8003))
    
    return {
        "status": "healthy",
        "component": "rhetor",
        "version": "1.0.0",
        "port": port,
        "message": "Rhetor is running normally",
        "providers": provider_status
    }

@app.get("/providers")
async def providers():
    """Get available LLM providers and models"""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    all_providers = llm_client.get_all_providers()
    
    return {
        "providers": all_providers,
        "default_provider": llm_client.default_provider_id,
        "default_model": llm_client.default_model
    }

@app.post("/provider")
async def set_provider(request: ProviderModelRequest):
    """Set the active provider and model"""
    if not llm_client:
        raise HTTPException(status_code=503, detail="LLM client not initialized")
    
    try:
        provider = llm_client.get_provider(request.provider_id)
        
        # Check if model is available
        available_models = provider.get_available_models()
        model_ids = [model["id"] for model in available_models]
        
        if request.model_id not in model_ids:
            raise ValueError(f"Model {request.model_id} not available from provider {request.provider_id}")
        
        # Set as default
        llm_client.default_provider_id = request.provider_id
        llm_client.default_model = request.model_id
        
        return {
            "success": True,
            "provider": request.provider_id,
            "model": request.model_id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/message")
async def message(request: MessageRequest):
    """Send a message to the LLM and get a response"""
    if not model_router or not context_manager:
        raise HTTPException(status_code=503, detail="Rhetor components not initialized")
    
    try:
        # Get or create system prompt
        system_prompt = prompt_engine.get_system_prompt(
            component=request.component or request.context_id.split(":")[0] if ":" in request.context_id else None
        )
        
        # Add user message to context
        await context_manager.add_to_context(
            context_id=request.context_id,
            role="user",
            content=request.message,
            metadata={"task_type": request.task_type}
        )
        
        # Route to appropriate model
        response = await model_router.route_request(
            message=request.message,
            context_id=request.context_id,
            task_type=request.task_type,
            component=request.component,
            system_prompt=system_prompt,
            streaming=request.streaming,
            override_config=request.options
        )
        
        # Add response to context if not streaming
        if not request.streaming and "message" in response and not response.get("error"):
            await context_manager.add_to_context(
                context_id=request.context_id,
                role="assistant",
                content=response["message"],
                metadata={
                    "model": response.get("model"),
                    "provider": response.get("provider"),
                    "task_type": request.task_type
                }
            )
        
        return response
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream(request: StreamRequest):
    """Send a message to the LLM and get a streaming response"""
    if not model_router or not context_manager:
        raise HTTPException(status_code=503, detail="Rhetor components not initialized")
    
    try:
        # Get or create system prompt
        system_prompt = prompt_engine.get_system_prompt(
            component=request.component or request.context_id.split(":")[0] if ":" in request.context_id else None
        )
        
        # Add user message to context
        await context_manager.add_to_context(
            context_id=request.context_id,
            role="user",
            content=request.message,
            metadata={"task_type": request.task_type}
        )
        
        # Use server-sent events for streaming
        async def event_generator():
            full_response = ""
            
            async for chunk in model_router.route_request(
                message=request.message,
                context_id=request.context_id,
                task_type=request.task_type,
                component=request.component,
                system_prompt=system_prompt,
                streaming=True,
                override_config=request.options
            ):
                # Accumulate the full response
                if "chunk" in chunk and chunk["chunk"]:
                    full_response += chunk["chunk"]
                
                # Yield the chunk as an SSE event
                yield {
                    "event": "message",
                    "data": json.dumps(chunk)
                }
                
                # If this is the final chunk, save the full response to context
                if chunk.get("done") and not chunk.get("error"):
                    # Record in budget manager for streaming responses
                    if budget_manager and "model" in chunk and "provider" in chunk:
                        component_id = request.component or request.context_id.split(":")[0] if ":" in request.context_id else "unknown"
                        input_text = request.message
                        if system_prompt:
                            input_text = system_prompt + "\n\n" + input_text
                            
                        budget_manager.record_completion(
                            provider=chunk["provider"],
                            model=chunk["model"],
                            input_text=input_text,
                            output_text=full_response,
                            component=component_id,
                            task_type=request.task_type,
                            metadata={
                                "context_id": request.context_id,
                                "streaming": True,
                                "system_prompt_length": len(system_prompt) if system_prompt else 0,
                            }
                        )
                        
                    await context_manager.add_to_context(
                        context_id=request.context_id,
                        role="assistant",
                        content=full_response,
                        metadata={
                            "model": chunk.get("model"),
                            "provider": chunk.get("provider"),
                            "task_type": request.task_type
                        }
                    )
        
        return EventSourceResponse(event_generator())
    except Exception as e:
        logger.error(f"Error processing streaming request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: ChatRequest):
    """Send a chat conversation to the LLM and get a response"""
    if not model_router or not context_manager:
        raise HTTPException(status_code=503, detail="Rhetor components not initialized")
    
    try:
        # Get or create system prompt
        system_prompt = prompt_engine.get_system_prompt(
            component=request.component or request.context_id.split(":")[0] if ":" in request.context_id else None
        )
        
        # Add messages to context
        for message in request.messages:
            # Only add messages we haven't seen before
            existing_messages = await context_manager.get_context_history(request.context_id)
            existing_content = [msg["content"] for msg in existing_messages]
            
            if message["content"] not in existing_content:
                await context_manager.add_to_context(
                    context_id=request.context_id,
                    role=message["role"],
                    content=message["content"],
                    metadata={"task_type": request.task_type}
                )
        
        # Route to appropriate model
        if request.streaming:
            # For streaming, use SSE
            async def event_generator():
                full_response = ""
                
                async for chunk in model_router.route_chat_request(
                    messages=request.messages,
                    context_id=request.context_id,
                    task_type=request.task_type,
                    component=request.component,
                    system_prompt=system_prompt,
                    streaming=True,
                    override_config=request.options
                ):
                    # Accumulate the full response
                    if "chunk" in chunk and chunk["chunk"]:
                        full_response += chunk["chunk"]
                    
                    # Yield the chunk as an SSE event
                    yield {
                        "event": "message",
                        "data": json.dumps(chunk)
                    }
                    
                    # If this is the final chunk, save the full response to context
                    if chunk.get("done") and not chunk.get("error"):
                        # Record in budget manager for streaming responses
                        if budget_manager and "model" in chunk and "provider" in chunk:
                            component_id = request.component or request.context_id.split(":")[0] if ":" in request.context_id else "unknown"
                            
                            # Convert messages to a single string for budget tracking
                            combined_input = ""
                            if system_prompt:
                                combined_input += system_prompt + "\n\n"
                                
                            for message in request.messages:
                                role = message.get("role", "user")
                                content = message.get("content", "")
                                combined_input += f"{role}: {content}\n"
                                
                            budget_manager.record_completion(
                                provider=chunk["provider"],
                                model=chunk["model"],
                                input_text=combined_input,
                                output_text=full_response,
                                component=component_id,
                                task_type=request.task_type,
                                metadata={
                                    "context_id": request.context_id,
                                    "streaming": True,
                                    "message_count": len(request.messages),
                                    "system_prompt_length": len(system_prompt) if system_prompt else 0,
                                }
                            )
                        
                        await context_manager.add_to_context(
                            context_id=request.context_id,
                            role="assistant",
                            content=full_response,
                            metadata={
                                "model": chunk.get("model"),
                                "provider": chunk.get("provider"),
                                "task_type": request.task_type
                            }
                        )
            
            return EventSourceResponse(event_generator())
        else:
            # Non-streaming response
            response = await model_router.route_chat_request(
                messages=request.messages,
                context_id=request.context_id,
                task_type=request.task_type,
                component=request.component,
                system_prompt=system_prompt,
                streaming=False,
                override_config=request.options
            )
            
            # Add response to context
            if "message" in response and not response.get("error"):
                await context_manager.add_to_context(
                    context_id=request.context_id,
                    role="assistant",
                    content=response["message"],
                    metadata={
                        "model": response.get("model"),
                        "provider": response.get("provider"),
                        "task_type": request.task_type
                    }
                )
            
            return response
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket support
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    
    if not model_router or not context_manager:
        await websocket.send_json({
            "type": "ERROR",
            "payload": {"error": "Rhetor components not initialized"}
        })
        await websocket.close()
        return
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request = json.loads(data)
            
            # Process based on message type
            if request.get("type") == "LLM_REQUEST":
                payload = request.get("payload", {})
                
                # Extract parameters
                message = payload.get("message", "")
                context_id = payload.get("context", "default")
                task_type = payload.get("task_type", "default")
                component = payload.get("component")
                streaming = payload.get("streaming", True)
                options = payload.get("options", {})
                
                # Get or create system prompt
                system_prompt = prompt_engine.get_system_prompt(
                    component=component or context_id.split(":")[0] if ":" in context_id else None
                )
                
                # Add user message to context
                await context_manager.add_to_context(
                    context_id=context_id,
                    role="user",
                    content=message,
                    metadata={"task_type": task_type}
                )
                
                # Send typing indicator
                await websocket.send_json({
                    "type": "UPDATE",
                    "source": "LLM",
                    "target": request.get("source", "UI"),
                    "timestamp": datetime.now().isoformat(),
                    "payload": {
                        "status": "typing",
                        "isTyping": True,
                        "context": context_id
                    }
                })
                
                if streaming:
                    # Stream response
                    full_response = ""
                    async for chunk in model_router.route_request(
                        message=message,
                        context_id=context_id,
                        task_type=task_type,
                        component=component,
                        system_prompt=system_prompt,
                        streaming=True,
                        override_config=options
                    ):
                        # Accumulate the full response
                        if "chunk" in chunk and chunk["chunk"]:
                            full_response += chunk["chunk"]
                        
                        # Send the chunk
                        await websocket.send_json({
                            "type": "UPDATE",
                            "source": "LLM",
                            "target": request.get("source", "UI"),
                            "timestamp": datetime.now().isoformat(),
                            "payload": chunk
                        })
                        
                        # If this is the final chunk, save the full response to context
                        if chunk.get("done") and not chunk.get("error"):
                            # Record in budget manager for websocket streaming
                            if budget_manager and "model" in chunk and "provider" in chunk:
                                component_id = component or context_id.split(":")[0] if ":" in context_id else "unknown"
                                input_text = message
                                if system_prompt:
                                    input_text = system_prompt + "\n\n" + input_text
                                    
                                budget_manager.record_completion(
                                    provider=chunk["provider"],
                                    model=chunk["model"],
                                    input_text=input_text,
                                    output_text=full_response,
                                    component=component_id,
                                    task_type=task_type,
                                    metadata={
                                        "context_id": context_id,
                                        "streaming": True,
                                        "system_prompt_length": len(system_prompt) if system_prompt else 0,
                                        "interface": "websocket"
                                    }
                                )
                            
                            await context_manager.add_to_context(
                                context_id=context_id,
                                role="assistant",
                                content=full_response,
                                metadata={
                                    "model": chunk.get("model"),
                                    "provider": chunk.get("provider"),
                                    "task_type": task_type
                                }
                            )
                    
                    # End typing indicator
                    await websocket.send_json({
                        "type": "UPDATE",
                        "source": "LLM",
                        "target": request.get("source", "UI"),
                        "timestamp": datetime.now().isoformat(),
                        "payload": {
                            "status": "typing",
                            "isTyping": False,
                            "context": context_id
                        }
                    })
                else:
                    # Get complete response
                    response = await model_router.route_request(
                        message=message,
                        context_id=context_id,
                        task_type=task_type,
                        component=component,
                        system_prompt=system_prompt,
                        streaming=False,
                        override_config=options
                    )
                    
                    # Add response to context if not an error
                    if "message" in response and not response.get("error"):
                        await context_manager.add_to_context(
                            context_id=context_id,
                            role="assistant",
                            content=response["message"],
                            metadata={
                                "model": response.get("model"), 
                                "provider": response.get("provider"),
                                "task_type": task_type
                            }
                        )
                    
                    # Send the response
                    await websocket.send_json({
                        "type": "RESPONSE",
                        "source": "LLM",
                        "target": request.get("source", "UI"),
                        "timestamp": datetime.now().isoformat(),
                        "payload": response
                    })
                    
                    # End typing indicator
                    await websocket.send_json({
                        "type": "UPDATE",
                        "source": "LLM",
                        "target": request.get("source", "UI"),
                        "timestamp": datetime.now().isoformat(),
                        "payload": {
                            "status": "typing",
                            "isTyping": False,
                            "context": context_id
                        }
                    })
            
            elif request.get("type") == "CHAT_REQUEST":
                payload = request.get("payload", {})
                
                # Extract parameters
                messages = payload.get("messages", [])
                context_id = payload.get("context", "default")
                task_type = payload.get("task_type", "default")
                component = payload.get("component")
                streaming = payload.get("streaming", True)
                options = payload.get("options", {})
                
                # Get or create system prompt
                system_prompt = prompt_engine.get_system_prompt(
                    component=component or context_id.split(":")[0] if ":" in context_id else None
                )
                
                # Add messages to context
                for message in messages:
                    # Only add messages we haven't seen before
                    existing_messages = await context_manager.get_context_history(context_id)
                    existing_content = [msg["content"] for msg in existing_messages]
                    
                    if message["content"] not in existing_content:
                        await context_manager.add_to_context(
                            context_id=context_id,
                            role=message["role"],
                            content=message["content"],
                            metadata={"task_type": task_type}
                        )
                
                # Send typing indicator
                await websocket.send_json({
                    "type": "UPDATE",
                    "source": "LLM",
                    "target": request.get("source", "UI"),
                    "timestamp": datetime.now().isoformat(),
                    "payload": {
                        "status": "typing",
                        "isTyping": True,
                        "context": context_id
                    }
                })
                
                if streaming:
                    # Stream response
                    full_response = ""
                    async for chunk in model_router.route_chat_request(
                        messages=messages,
                        context_id=context_id,
                        task_type=task_type,
                        component=component,
                        system_prompt=system_prompt,
                        streaming=True,
                        override_config=options
                    ):
                        # Accumulate the full response
                        if "chunk" in chunk and chunk["chunk"]:
                            full_response += chunk["chunk"]
                        
                        # Send the chunk
                        await websocket.send_json({
                            "type": "UPDATE",
                            "source": "LLM",
                            "target": request.get("source", "UI"),
                            "timestamp": datetime.now().isoformat(),
                            "payload": chunk
                        })
                        
                        # If this is the final chunk, save the full response to context
                        if chunk.get("done") and not chunk.get("error"):
                            # Record in budget manager for websocket streaming
                            if budget_manager and "model" in chunk and "provider" in chunk:
                                component_id = component or context_id.split(":")[0] if ":" in context_id else "unknown"
                                
                                # Convert messages to string for budget tracking
                                combined_input = ""
                                if system_prompt:
                                    combined_input += system_prompt + "\n\n"
                                
                                for message in messages:
                                    role = message.get("role", "user")
                                    content = message.get("content", "")
                                    combined_input += f"{role}: {content}\n"
                                
                                budget_manager.record_completion(
                                    provider=chunk["provider"],
                                    model=chunk["model"],
                                    input_text=combined_input,
                                    output_text=full_response,
                                    component=component_id,
                                    task_type=task_type,
                                    metadata={
                                        "context_id": context_id,
                                        "streaming": True,
                                        "message_count": len(messages),
                                        "system_prompt_length": len(system_prompt) if system_prompt else 0,
                                        "interface": "websocket"
                                    }
                                )
                            
                            await context_manager.add_to_context(
                                context_id=context_id,
                                role="assistant",
                                content=full_response,
                                metadata={
                                    "model": chunk.get("model"),
                                    "provider": chunk.get("provider"),
                                    "task_type": task_type
                                }
                            )
                    
                    # End typing indicator
                    await websocket.send_json({
                        "type": "UPDATE",
                        "source": "LLM",
                        "target": request.get("source", "UI"),
                        "timestamp": datetime.now().isoformat(),
                        "payload": {
                            "status": "typing",
                            "isTyping": False,
                            "context": context_id
                        }
                    })
                else:
                    # Get complete response
                    response = await model_router.route_chat_request(
                        messages=messages,
                        context_id=context_id,
                        task_type=task_type,
                        component=component,
                        system_prompt=system_prompt,
                        streaming=False,
                        override_config=options
                    )
                    
                    # Add response to context if not an error
                    if "message" in response and not response.get("error"):
                        await context_manager.add_to_context(
                            context_id=context_id,
                            role="assistant",
                            content=response["message"],
                            metadata={
                                "model": response.get("model"), 
                                "provider": response.get("provider"),
                                "task_type": task_type
                            }
                        )
                    
                    # Send the response
                    await websocket.send_json({
                        "type": "RESPONSE",
                        "source": "LLM",
                        "target": request.get("source", "UI"),
                        "timestamp": datetime.now().isoformat(),
                        "payload": response
                    })
                    
                    # End typing indicator
                    await websocket.send_json({
                        "type": "UPDATE",
                        "source": "LLM",
                        "target": request.get("source", "UI"),
                        "timestamp": datetime.now().isoformat(),
                        "payload": {
                            "status": "typing",
                            "isTyping": False,
                            "context": context_id
                        }
                    })
            
            elif request.get("type") == "REGISTER":
                # Handle client registration
                await websocket.send_json({
                    "type": "RESPONSE",
                    "source": "SYSTEM",
                    "target": request.get("source", "UNKNOWN"),
                    "timestamp": datetime.now().isoformat(),
                    "payload": {
                        "status": "registered",
                        "message": "Client registered successfully with Rhetor LLM Manager"
                    }
                })
            
            elif request.get("type") == "STATUS":
                # Handle status request
                providers = llm_client.get_all_providers()
                available_providers = {
                    provider_id: info["available"] 
                    for provider_id, info in providers.items()
                }
                
                # Get budget status
                budget_status = {}
                if budget_manager:
                    daily_usage = budget_manager.get_current_usage_total(BudgetPeriod.DAILY)
                    weekly_usage = budget_manager.get_current_usage_total(BudgetPeriod.WEEKLY)
                    monthly_usage = budget_manager.get_current_usage_total(BudgetPeriod.MONTHLY)
                    
                    daily_limit = budget_manager.get_budget_limit(BudgetPeriod.DAILY)
                    weekly_limit = budget_manager.get_budget_limit(BudgetPeriod.WEEKLY)
                    monthly_limit = budget_manager.get_budget_limit(BudgetPeriod.MONTHLY)
                    
                    budget_status = {
                        "daily_usage": daily_usage,
                        "weekly_usage": weekly_usage,
                        "monthly_usage": monthly_usage,
                        "daily_limit": daily_limit,
                        "weekly_limit": weekly_limit,
                        "monthly_limit": monthly_limit,
                        "budget_enabled": True
                    }
                
                await websocket.send_json({
                    "type": "RESPONSE",
                    "source": "SYSTEM",
                    "target": request.get("source", "UI"),
                    "timestamp": datetime.now().isoformat(),
                    "payload": {
                        "status": "ok",
                        "service": "rhetor",
                        "version": "0.1.0",
                        "providers": available_providers,
                        "default_provider": llm_client.default_provider_id,
                        "default_model": llm_client.default_model,
                        "budget": budget_status,
                        "message": "Rhetor LLM Manager is running"
                    }
                })
            
            else:
                # Unsupported request type
                await websocket.send_json({
                    "type": "ERROR",
                    "payload": {"error": f"Unsupported request type: {request.get('type')}"}
                })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "ERROR",
                "payload": {"error": str(e)}
            })
        except:
            pass


# Template management endpoints
@app.get("/templates")
async def list_templates(
    category: Optional[str] = Query(None, description="Filter templates by category"),
    tag: Optional[str] = Query(None, description="Filter templates by tag")
):
    """List all templates with optional filtering by category or tag"""
    if not template_manager:
        raise HTTPException(status_code=503, detail="Template manager not initialized")
    
    try:
        templates = template_manager.list_templates(category=category, tag=tag)
        return {
            "count": len(templates),
            "templates": templates
        }
    except Exception as e:
        logger.error(f"Error listing templates: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/templates", status_code=201)
async def create_template(request: TemplateCreateRequest):
    """Create a new template"""
    if not template_manager:
        raise HTTPException(status_code=503, detail="Template manager not initialized")
    
    try:
        template = template_manager.create_template(
            name=request.name,
            content=request.content,
            category=request.category,
            description=request.description,
            tags=request.tags,
            metadata=request.metadata
        )
        return template
    except Exception as e:
        logger.error(f"Error creating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/templates/{template_id}")
async def get_template(
    template_id: str = Path(..., description="Template ID"),
    version_id: Optional[str] = Query(None, description="Version ID (defaults to latest)")
):
    """Get a template by ID with optional version"""
    if not template_manager:
        raise HTTPException(status_code=503, detail="Template manager not initialized")
    
    try:
        template = template_manager.get_template(template_id, version_id=version_id)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        return template
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/templates/{template_id}")
async def update_template(
    template_id: str = Path(..., description="Template ID"),
    request: TemplateUpdateRequest = None
):
    """Update a template by ID"""
    if not template_manager:
        raise HTTPException(status_code=503, detail="Template manager not initialized")
    
    try:
        template = template_manager.update_template(
            template_id=template_id,
            content=request.content,
            metadata=request.metadata
        )
        return template
    except Exception as e:
        logger.error(f"Error updating template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/templates/{template_id}")
async def delete_template(template_id: str = Path(..., description="Template ID")):
    """Delete a template by ID"""
    if not template_manager:
        raise HTTPException(status_code=503, detail="Template manager not initialized")
    
    try:
        success = template_manager.delete_template(template_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        return {"success": True, "template_id": template_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/templates/{template_id}/versions")
async def list_template_versions(template_id: str = Path(..., description="Template ID")):
    """List all versions of a template"""
    if not template_manager:
        raise HTTPException(status_code=503, detail="Template manager not initialized")
    
    try:
        versions = template_manager.get_template_versions(template_id)
        if not versions:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        return {
            "template_id": template_id,
            "count": len(versions),
            "versions": versions
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing template versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/templates/render")
async def render_template(request: TemplateRenderRequest):
    """Render a template with variables"""
    if not template_manager:
        raise HTTPException(status_code=503, detail="Template manager not initialized")
    
    try:
        rendered = template_manager.render_template(
            template_id=request.template_id,
            variables=request.variables,
            version_id=request.version_id
        )
        return {
            "template_id": request.template_id,
            "version_id": request.version_id or "latest",
            "rendered_content": rendered
        }
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prompt management endpoints
@app.get("/prompts")
async def list_prompts(
    component: Optional[str] = Query(None, description="Filter prompts by component"),
    tag: Optional[str] = Query(None, description="Filter prompts by tag")
):
    """List all prompts with optional filtering by component or tag"""
    if not prompt_registry:
        raise HTTPException(status_code=503, detail="Prompt registry not initialized")
    
    try:
        prompts = prompt_registry.list_prompts(component=component, tag=tag)
        return {
            "count": len(prompts),
            "prompts": prompts
        }
    except Exception as e:
        logger.error(f"Error listing prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prompts", status_code=201)
async def create_prompt(request: PromptCreateRequest):
    """Create a new prompt"""
    if not prompt_registry:
        raise HTTPException(status_code=503, detail="Prompt registry not initialized")
    
    try:
        prompt = prompt_registry.create_prompt(
            name=request.name,
            component=request.component,
            content=request.content,
            description=request.description,
            tags=request.tags,
            is_default=request.is_default,
            parent_id=request.parent_id,
            metadata=request.metadata
        )
        return prompt
    except Exception as e:
        logger.error(f"Error creating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/prompts/{prompt_id}")
async def get_prompt(prompt_id: str = Path(..., description="Prompt ID")):
    """Get a prompt by ID"""
    if not prompt_registry:
        raise HTTPException(status_code=503, detail="Prompt registry not initialized")
    
    try:
        prompt = prompt_registry.get_prompt(prompt_id)
        if not prompt:
            raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
        return prompt
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/prompts/{prompt_id}")
async def update_prompt(
    prompt_id: str = Path(..., description="Prompt ID"),
    request: PromptUpdateRequest = None
):
    """Update a prompt by ID"""
    if not prompt_registry:
        raise HTTPException(status_code=503, detail="Prompt registry not initialized")
    
    try:
        prompt = prompt_registry.update_prompt(
            prompt_id=prompt_id,
            content=request.content,
            metadata=request.metadata
        )
        return prompt
    except Exception as e:
        logger.error(f"Error updating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/prompts/{prompt_id}")
async def delete_prompt(prompt_id: str = Path(..., description="Prompt ID")):
    """Delete a prompt by ID"""
    if not prompt_registry:
        raise HTTPException(status_code=503, detail="Prompt registry not initialized")
    
    try:
        success = prompt_registry.delete_prompt(prompt_id)
        if not success:
            raise HTTPException(status_code=404, detail=f"Prompt {prompt_id} not found")
        return {"success": True, "prompt_id": prompt_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/prompts/compare")
async def compare_prompts(request: PromptCompareRequest):
    """Compare two prompts"""
    if not prompt_registry:
        raise HTTPException(status_code=503, detail="Prompt registry not initialized")
    
    try:
        comparison = prompt_registry.compare_prompts(
            prompt_id1=request.prompt_id1,
            prompt_id2=request.prompt_id2
        )
        return comparison
    except Exception as e:
        logger.error(f"Error comparing prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Context management endpoints
@app.get("/contexts")
async def list_contexts():
    """List all available contexts"""
    if not context_manager:
        raise HTTPException(status_code=503, detail="Context manager not initialized")
    
    try:
        contexts = await context_manager.list_contexts()
        return {
            "count": len(contexts),
            "contexts": contexts
        }
    except Exception as e:
        logger.error(f"Error listing contexts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/contexts/{context_id}")
async def get_context(
    context_id: str = Path(..., description="Context ID"),
    limit: int = Query(20, description="Maximum number of messages to return"),
    include_metadata: bool = Query(False, description="Include message metadata")
):
    """Get messages in a context"""
    if not context_manager:
        raise HTTPException(status_code=503, detail="Context manager not initialized")
    
    try:
        messages = await context_manager.get_context_history(
            context_id=context_id,
            limit=limit
        )
        
        # Remove metadata if not requested
        if not include_metadata:
            for message in messages:
                if "metadata" in message:
                    del message["metadata"]
        
        return {
            "context_id": context_id,
            "count": len(messages),
            "messages": messages
        }
    except Exception as e:
        logger.error(f"Error getting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/contexts/{context_id}")
async def delete_context(context_id: str = Path(..., description="Context ID")):
    """Delete a context and all its messages"""
    if not context_manager:
        raise HTTPException(status_code=503, detail="Context manager not initialized")
    
    try:
        success = await context_manager.delete_context(context_id)
        return {"success": success, "context_id": context_id}
    except Exception as e:
        logger.error(f"Error deleting context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contexts/{context_id}/search")
async def search_context(
    context_id: str = Path(..., description="Context ID"),
    query: str = Query(..., description="Search query"),
    limit: int = Query(5, description="Maximum number of results")
):
    """Search for messages in a context"""
    if not context_manager:
        raise HTTPException(status_code=503, detail="Context manager not initialized")
    
    try:
        results = await context_manager.search_context(
            context_id=context_id,
            query=query,
            limit=limit
        )
        return {
            "context_id": context_id,
            "query": query,
            "count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error searching context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/contexts/{context_id}/summarize")
async def summarize_context(
    context_id: str = Path(..., description="Context ID"),
    max_tokens: int = Query(150, description="Maximum tokens for summary")
):
    """Generate a summary of a context"""
    if not context_manager:
        raise HTTPException(status_code=503, detail="Context manager not initialized")
    
    try:
        summary = await context_manager.summarize_context(
            context_id=context_id,
            max_tokens=max_tokens
        )
        return {
            "context_id": context_id,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Error summarizing context: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Budget management endpoints
@app.get("/budget")
async def get_budget_status():
    """Get the current budget status"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        # Get current usage
        daily_usage = budget_manager.get_current_usage_total(BudgetPeriod.DAILY)
        weekly_usage = budget_manager.get_current_usage_total(BudgetPeriod.WEEKLY)
        monthly_usage = budget_manager.get_current_usage_total(BudgetPeriod.MONTHLY)
        
        # Get budget limits
        daily_limit = budget_manager.get_budget_limit(BudgetPeriod.DAILY)
        weekly_limit = budget_manager.get_budget_limit(BudgetPeriod.WEEKLY)
        monthly_limit = budget_manager.get_budget_limit(BudgetPeriod.MONTHLY)
        
        # Get enforcement policies
        daily_policy = budget_manager.get_enforcement_policy(BudgetPeriod.DAILY)
        weekly_policy = budget_manager.get_enforcement_policy(BudgetPeriod.WEEKLY)
        monthly_policy = budget_manager.get_enforcement_policy(BudgetPeriod.MONTHLY)
        
        return {
            "daily": {
                "usage": daily_usage,
                "limit": daily_limit,
                "remaining": max(0, daily_limit - daily_usage) if daily_limit > 0 else None,
                "policy": daily_policy,
                "percentage": (daily_usage / daily_limit) * 100 if daily_limit > 0 else None
            },
            "weekly": {
                "usage": weekly_usage,
                "limit": weekly_limit,
                "remaining": max(0, weekly_limit - weekly_usage) if weekly_limit > 0 else None, 
                "policy": weekly_policy,
                "percentage": (weekly_usage / weekly_limit) * 100 if weekly_limit > 0 else None
            },
            "monthly": {
                "usage": monthly_usage,
                "limit": monthly_limit,
                "remaining": max(0, monthly_limit - monthly_usage) if monthly_limit > 0 else None,
                "policy": monthly_policy,
                "percentage": (monthly_usage / monthly_limit) * 100 if monthly_limit > 0 else None
            }
        }
    except Exception as e:
        logger.error(f"Error getting budget status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/budget/settings")
async def get_budget_settings():
    """Get all budget settings"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        settings = budget_manager.get_budget_settings()
        return settings
    except Exception as e:
        logger.error(f"Error getting budget settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/budget/limit")
async def set_budget_limit(request: BudgetLimitRequest):
    """Set a budget limit for a period"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        # Validate period
        try:
            period = BudgetPeriod(request.period)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid period: {request.period}. Must be one of: daily, weekly, monthly")
        
        # Validate enforcement policy if provided
        if request.enforcement:
            try:
                policy = BudgetPolicy(request.enforcement)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid enforcement policy: {request.enforcement}. Must be one of: ignore, warn, enforce")
        
        # Set the budget limit
        success = budget_manager.set_budget_limit(
            period=period,
            limit_amount=request.limit_amount,
            provider=request.provider,
            enforcement=request.enforcement
        )
        
        return {
            "success": success,
            "period": request.period,
            "limit_amount": request.limit_amount,
            "provider": request.provider,
            "enforcement": request.enforcement
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting budget limit: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/budget/policy")
async def set_budget_policy(request: BudgetPolicyRequest):
    """Set a budget enforcement policy for a period"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        # Validate period
        try:
            period = BudgetPeriod(request.period)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid period: {request.period}. Must be one of: daily, weekly, monthly")
        
        # Validate policy
        try:
            policy = BudgetPolicy(request.policy)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid policy: {request.policy}. Must be one of: ignore, warn, enforce")
        
        # Set the policy
        success = budget_manager.set_enforcement_policy(
            period=period,
            policy=policy,
            provider=request.provider
        )
        
        return {
            "success": success,
            "period": request.period,
            "policy": request.policy,
            "provider": request.provider
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting budget policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/budget/usage")
async def get_budget_usage(
    period: str = Query("daily", description="Period (daily, weekly, monthly)"),
    provider: Optional[str] = Query(None, description="Filter by provider")
):
    """Get detailed usage data for a period"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        # Validate period
        try:
            period_enum = BudgetPeriod(period)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}. Must be one of: daily, weekly, monthly")
        
        # Get usage data
        usage = budget_manager.get_usage(period=period_enum, provider=provider)
        
        return {
            "period": period,
            "provider": provider,
            "count": len(usage),
            "usage": usage
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting budget usage: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/budget/summary")
async def get_budget_summary(
    period: str = Query("daily", description="Period (daily, weekly, monthly)"),
    group_by: str = Query("provider", description="Group by field (provider, model, component, task_type)")
):
    """Get a usage summary for a period, grouped by provider, model, or component"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        # Validate period
        try:
            period_enum = BudgetPeriod(period)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid period: {period}. Must be one of: daily, weekly, monthly")
        
        # Validate group_by
        valid_fields = ["provider", "model", "component", "task_type"]
        if group_by not in valid_fields:
            raise HTTPException(status_code=400, detail=f"Invalid group_by: {group_by}. Must be one of: {', '.join(valid_fields)}")
        
        # Get the summary
        summary = budget_manager.get_usage_summary(period=period_enum, group_by=group_by)
        
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting budget summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/budget/model-tiers")
async def get_model_tiers():
    """Get models categorized by price tier"""
    if not budget_manager:
        raise HTTPException(status_code=503, detail="Budget manager not initialized")
    
    try:
        tiers = budget_manager.get_model_tiers()
        return tiers
    except Exception as e:
        logger.error(f"Error getting model tiers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Add shutdown endpoint using shared utility




def run_server(host="0.0.0.0", port=None, log_level="info"):
    """
    Run the Rhetor API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        log_level: Logging level
    """
    # Get port configuration
    if port is None:
        config = get_component_config()
        port = config.rhetor.port if hasattr(config, 'rhetor') else int(os.environ.get("RHETOR_PORT", 8003))
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level
    )


if __name__ == "__main__":
    run_server()