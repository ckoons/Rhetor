# Rhetor Technical Documentation

## Overview

Rhetor is the LLM management system for the Tekton ecosystem, serving as the central interface for all AI interactions across components. It manages prompt engineering, LLM provider selection, context tracking, and intelligent model routing. Rhetor replaces the previous LLM Adapter component with a more comprehensive solution.

## Architecture

Rhetor follows a single-port architecture that simplifies deployment and integration:

```
┌─────────────────────────────────┐
│          Rhetor Server          │
│   (HTTP + WebSocket on Port 8300)  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│         Model Router            │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│          LLM Client             │
└─────────────┬───────────────────┘
              │
        ┌─────┴─────┐
        │           │
        ▼           ▼
┌─────────────┐  ┌─────────────┐
│  Provider 1 │  │  Provider 2 │ ...
└─────────────┘  └─────────────┘
```

### Key Components

1. **API Server**
   - FastAPI application serving both HTTP and WebSocket endpoints on a single port (8300)
   - RESTful API for standard HTTP requests
   - Server-Sent Events for streaming responses
   - WebSockets for real-time bidirectional communication
   - Health and status endpoints

2. **Model Router**
   - Selects appropriate model based on task requirements
   - Routes requests to the optimal provider based on task type and component
   - Implements budget awareness to control costs
   - Provides fallback mechanisms between models
   - Configurable via JSON configuration file

3. **LLM Client**
   - Central manager for all LLM providers
   - Handles multiple provider initialization and availability checking
   - Supports both streaming and non-streaming completions
   - Implements consistent interface across providers

4. **Context Manager**
   - Tracks conversation history across requests
   - Persists conversations to disk and Engram (if available)
   - Provides context windowing for token management
   - Formats messages appropriately for different LLM providers

5. **Budget Manager**
   - Tracks token usage and costs across providers
   - Enforces budget limits by period (daily/weekly/monthly)
   - Provides usage reporting and cost analysis
   - Routes to cheaper alternatives when budget limits are approached
   - Supports tiered model selection based on cost

6. **Template and Prompt Management**
   - Manages system prompts for all components
   - Provides template creation and rendering
   - Supports versioning of templates
   - Organizes prompts by component and task

## Provider Implementations

Rhetor supports multiple LLM providers through a consistent interface:

1. **AnthropicProvider**
   - Access to Claude API (Claude 3 Opus, Sonnet, Haiku)
   - Streaming and non-streaming support
   - Context formatting specific to Anthropic's requirements

2. **OpenAIProvider**
   - Access to GPT API (GPT-4, GPT-4o, GPT-3.5-Turbo)
   - Streaming and non-streaming support
   - OpenAI-specific request formatting

3. **OllamaProvider**
   - Local LLM support
   - Integration with locally hosted models
   - Cost-free alternative for appropriate tasks

4. **SimulatedProvider**
   - Fallback when no APIs are available
   - Testing and development environment
   - Zero-cost operation

## API Endpoints

### HTTP Endpoints

- `GET /`: Basic information about the server
- `GET /health`: Health check endpoint
- `GET /providers`: Get available LLM providers and models
- `POST /provider`: Set the active provider and model
- `POST /message`: Send a message to the LLM
- `POST /stream`: Send a message and get a streaming response
- `POST /chat`: Send a chat conversation to the LLM

### WebSocket Endpoint

- `WebSocket /ws`: Real-time bidirectional communication
  - Supports `LLM_REQUEST` message type for single messages
  - Supports `CHAT_REQUEST` message type for conversations
  - Provides typing indicators and streaming updates
  - Supports registration and status requests

### Template Management

- `GET /templates`: List available templates
- `POST /templates`: Create a new template
- `GET /templates/{template_id}`: Get a template by ID
- `PUT /templates/{template_id}`: Update a template
- `DELETE /templates/{template_id}`: Delete a template
- `POST /templates/render`: Render a template with variables

### Prompt Management

- `GET /prompts`: List available prompts
- `POST /prompts`: Create a new prompt
- `GET /prompts/{prompt_id}`: Get a prompt by ID
- `PUT /prompts/{prompt_id}`: Update a prompt
- `DELETE /prompts/{prompt_id}`: Delete a prompt
- `POST /prompts/compare`: Compare two prompts

### Context Management

- `GET /contexts`: List available contexts
- `GET /contexts/{context_id}`: Get messages in a context
- `DELETE /contexts/{context_id}`: Delete a context
- `POST /contexts/{context_id}/search`: Search for messages in a context
- `POST /contexts/{context_id}/summarize`: Generate a summary of a context

### Budget Management

- `GET /budget`: Get current budget status
- `GET /budget/settings`: Get all budget settings
- `POST /budget/limit`: Set a budget limit for a period
- `POST /budget/policy`: Set a budget enforcement policy
- `GET /budget/usage`: Get detailed usage data
- `GET /budget/summary`: Get a usage summary
- `GET /budget/model-tiers`: Get models categorized by price tier

## Integration with Tekton

### Hermes Registration

Rhetor registers itself with Hermes to enable discovery by other components:

```json
{
  "id": "rhetor",
  "name": "Rhetor",
  "description": "LLM Management System for Tekton",
  "version": "0.1.0",
  "url": "http://localhost:8300",
  "capabilities": [
    "llm_management",
    "prompt_engineering",
    "context_management",
    "model_selection"
  ],
  "endpoints": {
    "http": "http://localhost:8300",
    "ws": "ws://localhost:8300/ws"
  },
  "dependencies": ["engram"],
  "lifecycle": {
    "startup_script": "tekton-launch --components rhetor",
    "shutdown_script": "tekton-kill",
    "status_check": {
      "url": "http://localhost:8300/health",
      "success_code": 200
    }
  },
  "metadata": {
    "icon": "🗣️",
    "ui_color": "#7e57c2",
    "priority": 40,
    "managed_port": 8300,
    "core_component": true,
    "replaces": "llm_adapter"
  }
}
```

### Client Interface

Rhetor provides client libraries for other components to interact with it:

1. **RhetorPromptClient**
   - Template management and rendering
   - Personality creation and management
   - Prompt generation and optimization

2. **RhetorCommunicationClient**
   - Conversation creation and management
   - Message addition and retrieval
   - Conversation analysis

### Standardized Communication Format

All responses follow a consistent format:

```json
{
  "message": "Response content",
  "model": "claude-3-sonnet-20240229",
  "provider": "anthropic",
  "context": "context_id",
  "finished": true,
  "timestamp": "2025-04-24T12:34:56.789Z"
}
```

## Model Selection Rules

The model router selects models based on:

1. **Task Type**
   - Code: Claude 3 Opus → GPT-4 Turbo
   - Planning: Claude 3 Sonnet → GPT-4o
   - Reasoning: Claude 3 Sonnet → GPT-4o  
   - Chat: Claude 3 Haiku → GPT-3.5-Turbo

2. **Component-Specific Configurations**
   - Can override defaults for specific components
   - Supports component_task format (e.g., "ergon_code")

3. **Budget Constraints**
   - Switches to cheaper models when budget limits are approached
   - Falls back to free models when budget is exceeded
   - Provides warnings when approaching limits

## Budget Management

Rhetor includes a sophisticated budget management system:

1. **Cost Tracking**
   - Records token usage and costs per request
   - Tracks by provider, model, component, and task type
   - Persists to SQLite database

2. **Budget Periods**
   - Daily: Resets at midnight
   - Weekly: Resets on Monday
   - Monthly: Resets on the 1st of each month

3. **Enforcement Policies**
   - Ignore: Track costs without taking action
   - Warn: Show warnings when approaching limits
   - Enforce: Prevent usage when limits exceeded

4. **Price Tiers**
   - Free: $0.00 per token (Ollama, Simulated)
   - Low: <$0.01 per 1K tokens (GPT-3.5-Turbo)
   - Medium: $0.01-$0.05 per 1K tokens (Claude 3 Haiku, Sonnet)
   - High: >$0.05 per 1K tokens (Claude 3 Opus, GPT-4)

## File Structure

```
rhetor/
├── __init__.py
├── __main__.py              # Entry point
├── api/
│   ├── __init__.py
│   └── app.py               # FastAPI application
├── core/
│   ├── __init__.py
│   ├── budget_manager.py    # Budget management
│   ├── communication.py     # Connection management
│   ├── context_manager.py   # Context tracking
│   ├── llm_client.py        # LLM provider management
│   ├── model_router.py      # Model selection
│   ├── prompt_engine.py     # Prompt management
│   ├── prompt_registry.py   # Prompt storage
│   └── template_manager.py  # Template management
├── models/
│   ├── __init__.py
│   └── providers/
│       ├── __init__.py
│       ├── anthropic.py     # Claude provider
│       ├── base.py          # Base provider interface
│       ├── ollama.py        # Ollama provider
│       ├── openai.py        # OpenAI provider
│       └── simulated.py     # Simulated provider
├── client.py                # Client interface
├── config/                  # Configuration files
├── utils/                   # Utility modules
│   ├── __init__.py
│   ├── engram_helper.py     # Engram integration
│   └── hermes_helper.py     # Hermes integration
└── templates/               # Prompt templates
    └── system_prompts.py    # System prompts
```

## Configuration

### Environment Variables

- `RHETOR_PORT`: Port for the Rhetor server (default: 8300)
- `RHETOR_TASK_CONFIG`: Path to task configuration file
- `RHETOR_BUDGET_POLICY`: Default budget policy (ignore, warn, enforce)
- `RHETOR_BUDGET_DAILY_LIMIT`: Default daily budget limit
- `RHETOR_BUDGET_WEEKLY_LIMIT`: Default weekly budget limit
- `RHETOR_BUDGET_MONTHLY_LIMIT`: Default monthly budget limit
- `HERMES_API_URL`: URL of the Hermes API (default: http://localhost:8100)

### Task Configuration

Task configurations can be specified in a JSON file:

```json
{
  "code": {
    "provider": "anthropic",
    "model": "claude-3-opus-20240229",
    "options": {
      "temperature": 0.2,
      "max_tokens": 4000,
      "fallback_provider": "openai",
      "fallback_model": "gpt-4-turbo"
    }
  },
  "planning": {
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "options": {
      "temperature": 0.7,
      "max_tokens": 4000,
      "fallback_provider": "openai",
      "fallback_model": "gpt-4o"
    }
  }
}
```

## Usage Examples

### Standard HTTP Request

```bash
curl -X POST http://localhost:8300/message \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, world!",
    "context_id": "test",
    "task_type": "chat",
    "streaming": false
  }'
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8300/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(data);
};
ws.send(JSON.stringify({
  type: "LLM_REQUEST",
  source: "UI",
  payload: {
    message: "Hello, world!",
    context: "test",
    task_type: "chat",
    streaming: true
  }
}));
```

### Python Client

```python
from rhetor.client import get_rhetor_prompt_client

async def example():
    client = await get_rhetor_prompt_client()
    
    try:
        template = await client.create_prompt_template(
            name="Documentation Template",
            template="Write documentation for {component} that explains {feature}.",
            variables=["component", "feature"],
            description="Template for generating documentation"
        )
        
        rendered = await client.render_prompt(
            template_id=template["template_id"],
            variables={
                "component": "Rhetor",
                "feature": "model routing"
            }
        )
        
        print(rendered)
    
    finally:
        await client.close()
```

## Security Considerations

1. **CORS Configuration**
   - Currently allows all origins (development mode)
   - Should be restricted in production

2. **Budget Enforcement**
   - Prevents excessive LLM API usage
   - Controls costs for API-based models
   - Can be bypassed with explicit configuration

3. **Authentication**
   - Does not currently implement authentication
   - Should be integrated with Tekton's authentication system in the future

## Performance Optimization

1. **Connection Pooling**
   - Maintains persistent connections to providers
   - Reduces connection overhead

2. **Context Windowing**
   - Manages token limits for large conversations
   - Automatically prunes context to stay within limits

3. **Budget-Aware Routing**
   - Routes to cheaper models when appropriate
   - Avoids unnecessary high-tier model usage

## Future Enhancements

1. **Advanced Context Management**
   - Semantic retrieval of past conversations
   - Vector storage for efficient context recall
   - Improved context windowing strategies

2. **Enhanced Template System**
   - More sophisticated template inheritance
   - Template analytics and effectiveness tracking
   - A/B testing of different prompt approaches

3. **Additional Providers**
   - Expanded local model support (Llama, Falcon, etc.)
   - Azure OpenAI Service integration
   - Vertex AI integration

4. **Performance Metrics**
   - Response time tracking
   - Success rate monitoring
   - Cost-effectiveness analysis

## Troubleshooting

### Common Issues

1. **Provider Unavailable**
   - Check API keys are properly configured
   - Verify network connectivity
   - Ensure provider service is operational

2. **Budget Exceeded**
   - Increase budget limits or change enforcement policy
   - Use lower-tier models for non-critical tasks
   - Check for inefficient prompt patterns

3. **Context Too Large**
   - Reduce context size or use summarization
   - Split conversations into smaller contexts
   - Use more efficient prompting techniques

### Logging

Rhetor uses Python's standard logging module:

```python
# Configure logging level
RHETOR_LOG_LEVEL=debug ./run_rhetor.sh
```

Log levels:
- INFO: Standard operational information
- DEBUG: Detailed debugging information
- WARNING: Potential issues that don't prevent operation
- ERROR: Errors that prevent specific operations
- CRITICAL: Critical errors that prevent the entire system from functioning

## References

- [Tekton Architecture](../docs/SINGLE_PORT_ARCHITECTURE.md)
- [Component Integration](../docs/llm_integration_guide.md)
- [Port Assignments](../config/port_assignments.md)