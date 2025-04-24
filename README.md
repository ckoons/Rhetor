# Rhetor

## Overview

Rhetor is the LLM management system for the Tekton ecosystem. It handles prompt engineering, LLM selection, context management, and serves as the central interface for all AI interactions across components. Rhetor replaces the previous LLM Adapter component with a more comprehensive solution.

## Key Features

- Single API gateway for all LLM interactions (HTTP and WebSocket on a single port)
- Intelligent model selection and routing based on task requirements
- Support for multiple LLM providers (Anthropic, OpenAI, Ollama, etc.)
- Advanced context tracking and persistence
- System prompt management for all Tekton components
- Graceful fallback mechanisms between providers
- Standardized interface for all Tekton components

## Architecture

Rhetor is built around a unified architecture that combines multiple features:

1. **Single-Port API**: Exposes both HTTP endpoints and WebSockets on the same port 
2. **Provider Abstraction**: Supports multiple LLM providers through a consistent interface
3. **Context Management**: Tracks and persists conversation history
4. **Model Router**: Intelligently selects the best model for different task types
5. **System Prompts**: Manages component-specific system prompts

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Start the Rhetor server
./run_rhetor.sh

# Register with Hermes (if using Hermes)
python register_with_hermes.py

# Alternatively, start with Tekton
./scripts/tekton-launch --components rhetor
```

## API Endpoints

- `GET /`: Basic information about the server
- `GET /health`: Health check endpoint
- `GET /providers`: Get available LLM providers and models
- `POST /provider`: Set the active provider and model
- `POST /message`: Send a message to the LLM
- `POST /stream`: Send a message and get a streaming response
- `POST /chat`: Send a chat conversation to the LLM
- `WebSocket /ws`: Real-time bidirectional communication

## Documentation

For detailed documentation, see the following resources in the MetaData directory:

- [Component Summaries](../MetaData/ComponentSummaries.md) - Overview of all Tekton components
- [Tekton Architecture](../MetaData/TektonArchitecture.md) - Overall system architecture
- [Component Integration](../MetaData/ComponentIntegration.md) - How components interact
- [CLI Operations](../MetaData/CLI_Operations.md) - Command-line operations
- [Rhetor Implementation](../MetaData/Implementation/RhetorImplementation.md) - Detailed implementation plan