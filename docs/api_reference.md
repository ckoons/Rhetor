# Rhetor API Reference

This document provides detailed API reference for the Rhetor LLM Management System.

## HTTP API

### Root Endpoint

#### `GET /`

Returns basic information about the Rhetor server.

**Response:**
```json
{
  "name": "Rhetor LLM Manager",
  "version": "0.1.0",
  "status": "running",
  "endpoints": [
    "/message", "/stream", "/chat", "/ws", "/providers", "/health", 
    "/templates", "/prompts", "/contexts", "/budget"
  ],
  "providers": {
    "anthropic": {
      "available": true,
      "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"]
    },
    "openai": {
      "available": true,
      "models": ["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"]
    }
  },
  "templates": {
    "categories": ["general", "code", "chat"],
    "count": 12
  },
  "prompts": {
    "components": ["ergon", "terma", "telos"],
    "count": 8
  },
  "budget": {
    "daily_usage": 0.15,
    "daily_limit": 5.0,
    "budget_enabled": true
  }
}
```

### Health Check

#### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "providers": {
    "anthropic": true,
    "openai": true,
    "ollama": true,
    "simulated": true
  }
}
```

### Provider Management

#### `GET /providers`

Get available LLM providers and models.

**Response:**
```json
{
  "providers": {
    "anthropic": {
      "available": true,
      "models": [
        {
          "id": "claude-3-opus-20240229",
          "name": "Claude 3 Opus",
          "context_window": 200000,
          "capabilities": ["code", "reasoning", "planning", "chat"]
        },
        {
          "id": "claude-3-sonnet-20240229",
          "name": "Claude 3 Sonnet",
          "context_window": 200000,
          "capabilities": ["code", "reasoning", "planning", "chat"]
        },
        {
          "id": "claude-3-haiku-20240307",
          "name": "Claude 3 Haiku",
          "context_window": 200000,
          "capabilities": ["chat", "reasoning"]
        }
      ]
    },
    "openai": {
      "available": true,
      "models": [
        {
          "id": "gpt-4-turbo",
          "name": "GPT-4 Turbo",
          "context_window": 128000,
          "capabilities": ["code", "reasoning", "planning", "chat"]
        },
        {
          "id": "gpt-4o",
          "name": "GPT-4o",
          "context_window": 128000,
          "capabilities": ["code", "reasoning", "planning", "chat"]
        },
        {
          "id": "gpt-3.5-turbo",
          "name": "GPT-3.5 Turbo",
          "context_window": 16000,
          "capabilities": ["chat", "reasoning"]
        }
      ]
    }
  },
  "default_provider": "anthropic",
  "default_model": "claude-3-sonnet-20240229"
}
```

#### `POST /provider`

Set the active provider and model.

**Request Body:**
```json
{
  "provider_id": "anthropic",
  "model_id": "claude-3-sonnet-20240229"
}
```

**Response:**
```json
{
  "success": true,
  "provider": "anthropic",
  "model": "claude-3-sonnet-20240229"
}
```

### LLM Messaging

#### `POST /message`

Send a message to the LLM and get a response.

**Request Body:**
```json
{
  "message": "What is the Tekton architecture?",
  "context_id": "documentation",
  "task_type": "reasoning",
  "component": "terma",
  "streaming": false,
  "options": {
    "temperature": 0.7,
    "max_tokens": 1000
  },
  "prompt_id": "technical-explanation"
}
```

**Response:**
```json
{
  "message": "The Tekton architecture is built around a distributed component system where...",
  "model": "claude-3-sonnet-20240229",
  "provider": "anthropic",
  "context": "documentation",
  "finished": true,
  "timestamp": "2025-04-24T12:34:56.789Z"
}
```

#### `POST /stream`

Send a message to the LLM and get a streaming response using Server-Sent Events.

**Request Body:**
```json
{
  "message": "Explain quantum computing",
  "context_id": "science",
  "task_type": "reasoning",
  "component": "terma",
  "options": {
    "temperature": 0.7,
    "max_tokens": 2000
  }
}
```

**Response:**
Server-Sent Events stream with JSON data:
```json
{
  "event": "message",
  "data": {
    "chunk": "Quantum computing is a type of computation that",
    "model": "claude-3-sonnet-20240229",
    "provider": "anthropic",
    "context": "science",
    "done": false,
    "timestamp": "2025-04-24T12:34:56.789Z"
  }
}
// Additional events follow until completion
```

#### `POST /chat`

Send a chat conversation to the LLM and get a response.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What is the Tekton architecture?"},
    {"role": "assistant", "content": "Tekton is a distributed component system for AI orchestration..."},
    {"role": "user", "content": "How does Rhetor fit into this?"}
  ],
  "context_id": "documentation",
  "task_type": "reasoning",
  "component": "terma",
  "streaming": false,
  "options": {
    "temperature": 0.7,
    "max_tokens": 1000
  }
}
```

**Response:**
```json
{
  "message": "Rhetor serves as the LLM management component within Tekton, handling...",
  "model": "claude-3-sonnet-20240229",
  "provider": "anthropic",
  "context": "documentation",
  "finished": true,
  "timestamp": "2025-04-24T12:34:56.789Z"
}
```

### Template Management

#### `GET /templates`

List all templates with optional filtering.

**Query Parameters:**
- `category` (optional): Filter templates by category
- `tag` (optional): Filter templates by tag

**Response:**
```json
{
  "count": 3,
  "templates": [
    {
      "template_id": "tech-doc-1",
      "name": "Technical Documentation",
      "category": "documentation",
      "description": "Template for technical documentation",
      "tags": ["technical", "documentation"],
      "created_at": "2025-04-20T10:30:00.000Z",
      "updated_at": "2025-04-22T14:15:00.000Z"
    },
    {
      "template_id": "code-explanation",
      "name": "Code Explanation",
      "category": "code",
      "description": "Template for explaining code",
      "tags": ["code", "explanation"],
      "created_at": "2025-04-18T09:45:00.000Z",
      "updated_at": "2025-04-18T09:45:00.000Z"
    }
  ]
}
```

#### `POST /templates`

Create a new template.

**Request Body:**
```json
{
  "name": "Technical Documentation",
  "content": "You are creating documentation for {component}...",
  "category": "documentation",
  "description": "Template for technical documentation",
  "tags": ["technical", "documentation"],
  "metadata": {
    "author": "Tekton Team",
    "version": "1.0"
  }
}
```

**Response:**
```json
{
  "template_id": "tech-doc-1",
  "name": "Technical Documentation",
  "content": "You are creating documentation for {component}...",
  "category": "documentation",
  "description": "Template for technical documentation",
  "tags": ["technical", "documentation"],
  "metadata": {
    "author": "Tekton Team",
    "version": "1.0"
  },
  "created_at": "2025-04-24T12:34:56.789Z",
  "updated_at": "2025-04-24T12:34:56.789Z"
}
```

#### `GET /templates/{template_id}`

Get a template by ID with optional version.

**Query Parameters:**
- `version_id` (optional): Version ID (defaults to latest)

**Response:**
```json
{
  "template_id": "tech-doc-1",
  "name": "Technical Documentation",
  "content": "You are creating documentation for {component}...",
  "category": "documentation",
  "description": "Template for technical documentation",
  "tags": ["technical", "documentation"],
  "metadata": {
    "author": "Tekton Team",
    "version": "1.0"
  },
  "created_at": "2025-04-24T12:34:56.789Z",
  "updated_at": "2025-04-24T12:34:56.789Z",
  "version_id": "v1"
}
```

#### `PUT /templates/{template_id}`

Update a template by ID.

**Request Body:**
```json
{
  "content": "You are creating enhanced documentation for {component}...",
  "metadata": {
    "author": "Tekton Team",
    "version": "1.1"
  }
}
```

**Response:**
```json
{
  "template_id": "tech-doc-1",
  "name": "Technical Documentation",
  "content": "You are creating enhanced documentation for {component}...",
  "category": "documentation",
  "description": "Template for technical documentation",
  "tags": ["technical", "documentation"],
  "metadata": {
    "author": "Tekton Team",
    "version": "1.1"
  },
  "created_at": "2025-04-24T12:34:56.789Z",
  "updated_at": "2025-04-24T13:45:00.000Z",
  "version_id": "v2"
}
```

#### `DELETE /templates/{template_id}`

Delete a template by ID.

**Response:**
```json
{
  "success": true,
  "template_id": "tech-doc-1"
}
```

#### `GET /templates/{template_id}/versions`

List all versions of a template.

**Response:**
```json
{
  "template_id": "tech-doc-1",
  "count": 2,
  "versions": [
    {
      "version_id": "v1",
      "created_at": "2025-04-24T12:34:56.789Z",
      "metadata": {
        "author": "Tekton Team",
        "version": "1.0"
      }
    },
    {
      "version_id": "v2",
      "created_at": "2025-04-24T13:45:00.000Z",
      "metadata": {
        "author": "Tekton Team",
        "version": "1.1"
      }
    }
  ]
}
```

#### `POST /templates/render`

Render a template with variables.

**Request Body:**
```json
{
  "template_id": "tech-doc-1",
  "variables": {
    "component": "Rhetor",
    "version": "1.0",
    "author": "Tekton Team"
  },
  "version_id": "v2"
}
```

**Response:**
```json
{
  "template_id": "tech-doc-1",
  "version_id": "v2",
  "rendered_content": "You are creating enhanced documentation for Rhetor..."
}
```

### Prompt Management

#### `GET /prompts`

List all prompts with optional filtering.

**Query Parameters:**
- `component` (optional): Filter prompts by component
- `tag` (optional): Filter prompts by tag

**Response:**
```json
{
  "count": 2,
  "prompts": [
    {
      "prompt_id": "terma-system",
      "name": "Terma System Prompt",
      "component": "terma",
      "description": "System prompt for Terma component",
      "tags": ["system", "terma"],
      "is_default": true,
      "created_at": "2025-04-20T10:30:00.000Z",
      "updated_at": "2025-04-22T14:15:00.000Z"
    },
    {
      "prompt_id": "ergon-system",
      "name": "Ergon System Prompt",
      "component": "ergon",
      "description": "System prompt for Ergon component",
      "tags": ["system", "ergon"],
      "is_default": true,
      "created_at": "2025-04-18T09:45:00.000Z",
      "updated_at": "2025-04-18T09:45:00.000Z"
    }
  ]
}
```

#### `POST /prompts`

Create a new prompt.

**Request Body:**
```json
{
  "name": "Terma System Prompt",
  "component": "terma",
  "content": "You are Terma, a terminal interface for the Tekton system...",
  "description": "System prompt for Terma component",
  "tags": ["system", "terma"],
  "is_default": true,
  "metadata": {
    "author": "Tekton Team",
    "version": "1.0"
  }
}
```

**Response:**
```json
{
  "prompt_id": "terma-system",
  "name": "Terma System Prompt",
  "component": "terma",
  "content": "You are Terma, a terminal interface for the Tekton system...",
  "description": "System prompt for Terma component",
  "tags": ["system", "terma"],
  "is_default": true,
  "metadata": {
    "author": "Tekton Team",
    "version": "1.0"
  },
  "created_at": "2025-04-24T12:34:56.789Z",
  "updated_at": "2025-04-24T12:34:56.789Z"
}
```

#### `GET /prompts/{prompt_id}`

Get a prompt by ID.

**Response:**
```json
{
  "prompt_id": "terma-system",
  "name": "Terma System Prompt",
  "component": "terma",
  "content": "You are Terma, a terminal interface for the Tekton system...",
  "description": "System prompt for Terma component",
  "tags": ["system", "terma"],
  "is_default": true,
  "metadata": {
    "author": "Tekton Team",
    "version": "1.0"
  },
  "created_at": "2025-04-24T12:34:56.789Z",
  "updated_at": "2025-04-24T12:34:56.789Z"
}
```

#### `PUT /prompts/{prompt_id}`

Update a prompt by ID.

**Request Body:**
```json
{
  "content": "You are Terma v2, an enhanced terminal interface for the Tekton system...",
  "metadata": {
    "author": "Tekton Team",
    "version": "2.0"
  }
}
```

**Response:**
```json
{
  "prompt_id": "terma-system",
  "name": "Terma System Prompt",
  "component": "terma",
  "content": "You are Terma v2, an enhanced terminal interface for the Tekton system...",
  "description": "System prompt for Terma component",
  "tags": ["system", "terma"],
  "is_default": true,
  "metadata": {
    "author": "Tekton Team",
    "version": "2.0"
  },
  "created_at": "2025-04-24T12:34:56.789Z",
  "updated_at": "2025-04-24T13:45:00.000Z"
}
```

#### `DELETE /prompts/{prompt_id}`

Delete a prompt by ID.

**Response:**
```json
{
  "success": true,
  "prompt_id": "terma-system"
}
```

#### `POST /prompts/compare`

Compare two prompts.

**Request Body:**
```json
{
  "prompt_id1": "terma-system-v1",
  "prompt_id2": "terma-system-v2"
}
```

**Response:**
```json
{
  "prompt_id1": "terma-system-v1",
  "prompt_id2": "terma-system-v2",
  "differences": {
    "added": ["enhanced terminal interface", "new capabilities"],
    "removed": ["basic terminal interface"],
    "changed": ["version reference"],
    "similarity_score": 0.85
  }
}
```

### Context Management

#### `GET /contexts`

List all available contexts.

**Response:**
```json
{
  "count": 3,
  "contexts": [
    {
      "context_id": "documentation",
      "message_count": 12,
      "last_updated": "2025-04-24T12:34:56.789Z"
    },
    {
      "context_id": "code-examples",
      "message_count": 8,
      "last_updated": "2025-04-23T10:15:30.123Z"
    },
    {
      "context_id": "terma:session-1",
      "message_count": 25,
      "last_updated": "2025-04-24T11:22:33.456Z"
    }
  ]
}
```

#### `GET /contexts/{context_id}`

Get messages in a context.

**Query Parameters:**
- `limit` (optional): Maximum number of messages to return (default: 20)
- `include_metadata` (optional): Include message metadata (default: false)

**Response:**
```json
{
  "context_id": "documentation",
  "count": 3,
  "messages": [
    {
      "role": "user",
      "content": "What is the Tekton architecture?",
      "timestamp": "2025-04-24T12:30:00.000Z"
    },
    {
      "role": "assistant",
      "content": "Tekton is a distributed component system for AI orchestration...",
      "timestamp": "2025-04-24T12:30:30.000Z"
    },
    {
      "role": "user",
      "content": "How does Rhetor fit into this?",
      "timestamp": "2025-04-24T12:31:00.000Z"
    }
  ]
}
```

#### `DELETE /contexts/{context_id}`

Delete a context and all its messages.

**Response:**
```json
{
  "success": true,
  "context_id": "documentation"
}
```

#### `POST /contexts/{context_id}/search`

Search for messages in a context.

**Query Parameters:**
- `query`: Search query
- `limit` (optional): Maximum number of results (default: 5)

**Response:**
```json
{
  "context_id": "documentation",
  "query": "Rhetor",
  "count": 2,
  "results": [
    {
      "role": "user",
      "content": "How does Rhetor fit into this?",
      "timestamp": "2025-04-24T12:31:00.000Z",
      "score": 0.95
    },
    {
      "role": "assistant",
      "content": "Rhetor serves as the LLM management component within Tekton...",
      "timestamp": "2025-04-24T12:31:30.000Z",
      "score": 0.87
    }
  ]
}
```

#### `POST /contexts/{context_id}/summarize`

Generate a summary of a context.

**Query Parameters:**
- `max_tokens` (optional): Maximum tokens for summary (default: 150)

**Response:**
```json
{
  "context_id": "documentation",
  "summary": "This conversation covers the Tekton architecture and how Rhetor fits into it as the LLM management component..."
}
```

### Budget Management

#### `GET /budget`

Get the current budget status.

**Response:**
```json
{
  "daily": {
    "usage": 0.87,
    "limit": 5.0,
    "remaining": 4.13,
    "policy": "warn",
    "percentage": 17.4
  },
  "weekly": {
    "usage": 3.25,
    "limit": 20.0,
    "remaining": 16.75,
    "policy": "warn",
    "percentage": 16.25
  },
  "monthly": {
    "usage": 15.32,
    "limit": 50.0,
    "remaining": 34.68,
    "policy": "enforce",
    "percentage": 30.64
  }
}
```

#### `GET /budget/settings`

Get all budget settings.

**Response:**
```json
{
  "daily": [
    {
      "id": 1,
      "provider": "all",
      "limit_amount": 5.0,
      "enforcement": "warn",
      "start_date": "2025-04-01T00:00:00.000Z"
    },
    {
      "id": 2,
      "provider": "anthropic",
      "limit_amount": 3.0,
      "enforcement": "warn",
      "start_date": "2025-04-01T00:00:00.000Z"
    }
  ],
  "weekly": [
    {
      "id": 3,
      "provider": "all",
      "limit_amount": 20.0,
      "enforcement": "warn",
      "start_date": "2025-04-01T00:00:00.000Z"
    }
  ],
  "monthly": [
    {
      "id": 4,
      "provider": "all",
      "limit_amount": 50.0,
      "enforcement": "enforce",
      "start_date": "2025-04-01T00:00:00.000Z"
    }
  ]
}
```

#### `POST /budget/limit`

Set a budget limit for a period.

**Request Body:**
```json
{
  "period": "daily",
  "limit_amount": 10.0,
  "provider": "all",
  "enforcement": "warn"
}
```

**Response:**
```json
{
  "success": true,
  "period": "daily",
  "limit_amount": 10.0,
  "provider": "all",
  "enforcement": "warn"
}
```

#### `POST /budget/policy`

Set a budget enforcement policy for a period.

**Request Body:**
```json
{
  "period": "monthly",
  "policy": "enforce",
  "provider": "all"
}
```

**Response:**
```json
{
  "success": true,
  "period": "monthly",
  "policy": "enforce",
  "provider": "all"
}
```

#### `GET /budget/usage`

Get detailed usage data for a period.

**Query Parameters:**
- `period`: Period (daily, weekly, monthly)
- `provider` (optional): Filter by provider

**Response:**
```json
{
  "period": "daily",
  "provider": null,
  "count": 2,
  "usage": [
    {
      "id": 1,
      "timestamp": "2025-04-24T10:15:30.123Z",
      "provider": "anthropic",
      "model": "claude-3-sonnet-20240229",
      "component": "terma",
      "task_type": "chat",
      "input_tokens": 250,
      "output_tokens": 1200,
      "cost": 0.0225,
      "metadata": {
        "context_id": "terma:session-1",
        "streaming": true
      }
    },
    {
      "id": 2,
      "timestamp": "2025-04-24T11:22:33.456Z",
      "provider": "openai",
      "model": "gpt-4o",
      "component": "ergon",
      "task_type": "code",
      "input_tokens": 500,
      "output_tokens": 1500,
      "cost": 0.0650,
      "metadata": {
        "context_id": "ergon:code-generation",
        "streaming": false
      }
    }
  ]
}
```

#### `GET /budget/summary`

Get a usage summary for a period, grouped by provider, model, or component.

**Query Parameters:**
- `period`: Period (daily, weekly, monthly)
- `group_by`: Group by field (provider, model, component, task_type)

**Response:**
```json
{
  "period": "daily",
  "total_cost": 0.87,
  "total_input_tokens": 2500,
  "total_output_tokens": 12000,
  "total_tokens": 14500,
  "count": 15,
  "groups": {
    "anthropic": {
      "cost": 0.45,
      "input_tokens": 1200,
      "output_tokens": 7500,
      "count": 8
    },
    "openai": {
      "cost": 0.42,
      "input_tokens": 1300,
      "output_tokens": 4500,
      "count": 7
    }
  }
}
```

#### `GET /budget/model-tiers`

Get models categorized by price tier.

**Response:**
```json
{
  "free": [
    {
      "provider": "ollama",
      "model": "llama3",
      "cost_per_1k_tokens": 0.0
    }
  ],
  "low": [
    {
      "provider": "openai",
      "model": "gpt-3.5-turbo",
      "cost_per_1k_tokens": 0.002
    }
  ],
  "medium": [
    {
      "provider": "anthropic",
      "model": "claude-3-haiku-20240307",
      "cost_per_1k_tokens": 0.00175
    },
    {
      "provider": "anthropic",
      "model": "claude-3-sonnet-20240229",
      "cost_per_1k_tokens": 0.018
    }
  ],
  "high": [
    {
      "provider": "anthropic",
      "model": "claude-3-opus-20240229",
      "cost_per_1k_tokens": 0.09
    },
    {
      "provider": "openai",
      "model": "gpt-4",
      "cost_per_1k_tokens": 0.09
    }
  ]
}
```

## WebSocket API

### Connection

Connect to `ws://localhost:8300/ws`

### Message Types

#### Registration

```json
{
  "type": "REGISTER",
  "source": "MyComponent"
}
```

**Response:**
```json
{
  "type": "RESPONSE",
  "source": "SYSTEM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:34:56.789Z",
  "payload": {
    "status": "registered",
    "message": "Client registered successfully with Rhetor LLM Manager"
  }
}
```

#### Status Request

```json
{
  "type": "STATUS",
  "source": "MyComponent"
}
```

**Response:**
```json
{
  "type": "RESPONSE",
  "source": "SYSTEM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:34:56.789Z",
  "payload": {
    "status": "ok",
    "service": "rhetor",
    "version": "0.1.0",
    "providers": {
      "anthropic": true,
      "openai": true,
      "ollama": true,
      "simulated": true
    },
    "default_provider": "anthropic",
    "default_model": "claude-3-sonnet-20240229",
    "budget": {
      "daily_usage": 0.87,
      "weekly_usage": 3.25,
      "monthly_usage": 15.32,
      "daily_limit": 5.0,
      "weekly_limit": 20.0,
      "monthly_limit": 50.0,
      "budget_enabled": true
    },
    "message": "Rhetor LLM Manager is running"
  }
}
```

#### LLM Request

```json
{
  "type": "LLM_REQUEST",
  "source": "MyComponent",
  "payload": {
    "message": "What is the Tekton architecture?",
    "context": "documentation",
    "task_type": "reasoning",
    "streaming": true,
    "component": "terma",
    "options": {
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
}
```

**Response (Streaming):**
```json
// Typing indicator
{
  "type": "UPDATE",
  "source": "LLM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:34:56.789Z",
  "payload": {
    "status": "typing",
    "isTyping": true,
    "context": "documentation"
  }
}

// First chunk
{
  "type": "UPDATE",
  "source": "LLM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:34:57.000Z",
  "payload": {
    "chunk": "The Tekton architecture is",
    "model": "claude-3-sonnet-20240229",
    "provider": "anthropic",
    "context": "documentation",
    "done": false
  }
}

// Additional chunks...

// Final chunk
{
  "type": "UPDATE",
  "source": "LLM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:35:05.000Z",
  "payload": {
    "chunk": " orchestration system.",
    "model": "claude-3-sonnet-20240229",
    "provider": "anthropic",
    "context": "documentation",
    "done": true
  }
}

// Typing indicator off
{
  "type": "UPDATE",
  "source": "LLM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:35:05.100Z",
  "payload": {
    "status": "typing",
    "isTyping": false,
    "context": "documentation"
  }
}
```

**Response (Non-Streaming):**
```json
// Typing indicator
{
  "type": "UPDATE",
  "source": "LLM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:34:56.789Z",
  "payload": {
    "status": "typing",
    "isTyping": true,
    "context": "documentation"
  }
}

// Complete response
{
  "type": "RESPONSE",
  "source": "LLM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:35:05.000Z",
  "payload": {
    "message": "The Tekton architecture is built around a distributed component system where each component provides specific capabilities...",
    "model": "claude-3-sonnet-20240229",
    "provider": "anthropic",
    "context": "documentation",
    "finished": true
  }
}

// Typing indicator off
{
  "type": "UPDATE",
  "source": "LLM",
  "target": "MyComponent",
  "timestamp": "2025-04-24T12:35:05.100Z",
  "payload": {
    "status": "typing",
    "isTyping": false,
    "context": "documentation"
  }
}
```

#### Chat Request

```json
{
  "type": "CHAT_REQUEST",
  "source": "MyComponent",
  "payload": {
    "messages": [
      {"role": "user", "content": "What is the Tekton architecture?"},
      {"role": "assistant", "content": "Tekton is a distributed component system..."},
      {"role": "user", "content": "How does Rhetor fit into this?"}
    ],
    "context": "documentation",
    "task_type": "reasoning",
    "streaming": true,
    "component": "terma",
    "options": {
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }
}
```

**Response:**
Similar to LLM_REQUEST responses, with streaming or non-streaming formats.

## Python Client API

### RhetorPromptClient

```python
from rhetor.client import get_rhetor_prompt_client

# Create client
client = await get_rhetor_prompt_client()

# Create template
template = await client.create_prompt_template(
    name="Documentation Template",
    template="Write documentation for {component} that explains {feature}.",
    variables=["component", "feature"],
    description="Template for generating documentation"
)

# Render template
rendered = await client.render_prompt(
    template_id=template["template_id"],
    variables={
        "component": "Rhetor",
        "feature": "model routing"
    }
)

# Create personality
personality = await client.create_personality(
    name="Technical Expert",
    traits={
        "expertise": ["AI", "LLMs"],
        "formality": 0.8
    },
    description="Expert in technical domains",
    tone="professional"
)

# Generate prompt
prompt = await client.generate_prompt(
    task="Explain model routing in Rhetor",
    context={"audience": "developers"},
    personality_id=personality["personality_id"],
    format="instruction"
)

# Close client
await client.close()
```

### RhetorCommunicationClient

```python
from rhetor.client import get_rhetor_communication_client

# Create client
client = await get_rhetor_communication_client()

# Create conversation
conversation = await client.create_conversation(
    name="Technical Support Session",
    metadata={"priority": "high"}
)

# Add message
message = await client.add_message(
    conversation_id=conversation["conversation_id"],
    sender="user",
    message="How do I configure the budget limits?",
    metadata={"component": "terma"}
)

# Get conversation
retrieved = await client.get_conversation(conversation["conversation_id"])

# Analyze conversation
analysis = await client.analyze_conversation(
    conversation_id=conversation["conversation_id"],
    analysis_type="sentiment"
)

# Close client
await client.close()
```

## JavaScript Client

```javascript
import { RhetorClient } from 'tekton-llm-client';

// Create client
const client = new RhetorClient({
  url: 'http://localhost:8300'
});

// Send message (promise-based)
client.sendMessage({
  message: "What is the Tekton architecture?",
  contextId: "documentation",
  taskType: "reasoning",
  streaming: false
})
.then(response => {
  console.log(response);
})
.catch(error => {
  console.error(error);
});

// Send message (streaming)
client.streamMessage({
  message: "What is the Tekton architecture?",
  contextId: "documentation",
  taskType: "reasoning",
  onChunk: (chunk) => {
    console.log(chunk);
  },
  onComplete: (response) => {
    console.log("Complete:", response);
  },
  onError: (error) => {
    console.error(error);
  }
});

// Get available providers
client.getProviders()
.then(providers => {
  console.log(providers);
})
.catch(error => {
  console.error(error);
});

// Set provider and model
client.setProvider({
  providerId: "anthropic",
  modelId: "claude-3-sonnet-20240229"
})
.then(result => {
  console.log(result);
})
.catch(error => {
  console.error(error);
});

// Close client
client.close();
```

## WebSocket Client

```javascript
import { RhetorWebSocketClient } from 'tekton-llm-client';

// Create client
const client = new RhetorWebSocketClient({
  url: 'ws://localhost:8300/ws',
  source: 'MyComponent'
});

// Connect and register
client.connect()
.then(() => {
  console.log("Connected to Rhetor WebSocket");
  
  // Send LLM request
  client.sendRequest({
    message: "What is the Tekton architecture?",
    context: "documentation",
    taskType: "reasoning",
    streaming: true,
    component: "terma"
  }, {
    onTypingStart: () => {
      console.log("LLM is typing...");
    },
    onChunk: (chunk) => {
      console.log("Chunk:", chunk);
    },
    onComplete: (response) => {
      console.log("Complete response:", response);
    },
    onTypingEnd: () => {
      console.log("LLM stopped typing");
    },
    onError: (error) => {
      console.error("Error:", error);
    }
  });
  
  // Get status
  client.getStatus()
  .then(status => {
    console.log("Rhetor status:", status);
  });
})
.catch(error => {
  console.error("Connection error:", error);
});

// Close connection
client.close();
```