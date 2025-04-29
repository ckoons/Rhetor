# Rhetor Integration Guide

This guide explains how to integrate your Tekton component with Rhetor, the LLM management system.

## Overview

Rhetor provides centralized LLM access for all Tekton components through:
- HTTP API endpoint (`http://localhost:8300`)
- WebSocket API endpoint (`ws://localhost:8300/ws`)
- Python client library
- JavaScript client library

By integrating with Rhetor, your component can:
- Access multiple LLM providers through a single interface
- Automatically use the appropriate model for different tasks
- Benefit from budget management and cost optimization
- Maintain conversation context across requests
- Use customized system prompts

## Prerequisites

- Tekton environment with Rhetor installed
- Rhetor running (port 8300)
- Component registered with Hermes (recommended)

## Integration Methods

Choose the integration method that best fits your component:

1. **HTTP API**: Direct HTTP requests for simple integration
2. **WebSocket API**: Real-time bidirectional communication
3. **Python Client**: Easy integration for Python-based components
4. **JavaScript Client**: For frontend or Node.js components

## HTTP API Integration

### Basic Request Example

```python
import aiohttp
import json

async def send_llm_request(message, context_id="default", task_type="default", component=None):
    async with aiohttp.ClientSession() as session:
        payload = {
            "message": message,
            "context_id": context_id,
            "task_type": task_type,
            "component": component,
            "streaming": False
        }
        
        async with session.post(
            "http://localhost:8300/message",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"Error from Rhetor: {response.status} - {error_text}")

# Usage
response = await send_llm_request(
    message="What is Tekton?",
    context_id="my-component:session1",
    task_type="chat",
    component="my-component"
)
print(response["message"])
```

### Streaming Request Example

```python
import aiohttp
import json
import asyncio

async def stream_llm_request(message, context_id="default", task_type="default", component=None):
    async with aiohttp.ClientSession() as session:
        payload = {
            "message": message,
            "context_id": context_id,
            "task_type": task_type,
            "component": component
        }
        
        async with session.post(
            "http://localhost:8300/stream",
            json=payload,
            headers={"Content-Type": "application/json"}
        ) as response:
            if response.status == 200:
                # Server-Sent Events stream
                full_response = ""
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    
                    # SSE format: data: {...}
                    if line.startswith('data: '):
                        json_str = line[6:]  # Remove 'data: ' prefix
                        try:
                            chunk = json.loads(json_str)
                            if "chunk" in chunk and chunk["chunk"]:
                                full_response += chunk["chunk"]
                                yield chunk["chunk"]
                                
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            pass
                            
                return full_response
            else:
                error_text = await response.text()
                raise Exception(f"Error from Rhetor: {response.status} - {error_text}")

# Usage
async def process_stream():
    async for chunk in stream_llm_request(
        message="What is Tekton?",
        context_id="my-component:session1",
        task_type="chat",
        component="my-component"
    ):
        print(chunk, end="", flush=True)
    print()  # Final newline

asyncio.run(process_stream())
```

## WebSocket Integration

### Basic WebSocket Example

```python
import asyncio
import json
import websockets
import uuid

class RhetorWebSocketClient:
    def __init__(self, url="ws://localhost:8300/ws", source="MyComponent"):
        self.url = url
        self.source = source
        self.ws = None
        self.connected = False
        self.message_handlers = {}
        
    async def connect(self):
        self.ws = await websockets.connect(self.url)
        self.connected = True
        
        # Register with Rhetor
        await self.ws.send(json.dumps({
            "type": "REGISTER",
            "source": self.source
        }))
        
        # Start listening for messages
        asyncio.create_task(self._listen())
        
        return True
        
    async def _listen(self):
        while self.connected:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                
                # Process message
                message_id = data.get("id")
                if message_id and message_id in self.message_handlers:
                    handler = self.message_handlers[message_id]
                    handler(data)
                    
                # Handle typing indicators
                if data.get("type") == "UPDATE" and data.get("payload", {}).get("status") == "typing":
                    is_typing = data.get("payload", {}).get("isTyping", False)
                    context = data.get("payload", {}).get("context")
                    if context in self.message_handlers:
                        handler = self.message_handlers[context]
                        if is_typing:
                            handler({"type": "typing_start", "context": context})
                        else:
                            handler({"type": "typing_end", "context": context})
                
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                break
            except Exception as e:
                print(f"Error in WebSocket listener: {e}")
                
    async def close(self):
        if self.ws and self.connected:
            await self.ws.close()
            self.connected = False
            
    async def send_message(self, message, context_id="default", task_type="default", 
                           component=None, streaming=True, callback=None):
        if not self.connected:
            raise Exception("Not connected to Rhetor WebSocket")
            
        # Generate unique message ID
        message_id = str(uuid.uuid4())
        
        # Register callback
        if callback:
            self.message_handlers[message_id] = callback
            # Also register with context_id for typing indicators
            self.message_handlers[context_id] = callback
            
        # Send message
        await self.ws.send(json.dumps({
            "type": "LLM_REQUEST",
            "id": message_id,
            "source": self.source,
            "payload": {
                "message": message,
                "context": context_id,
                "task_type": task_type,
                "component": component,
                "streaming": streaming
            }
        }))
        
        return message_id
        
    async def get_status(self):
        if not self.connected:
            raise Exception("Not connected to Rhetor WebSocket")
            
        # Generate unique message ID
        message_id = str(uuid.uuid4())
        
        # Create a future to wait for the response
        response_future = asyncio.Future()
        
        # Register callback
        def status_callback(data):
            response_future.set_result(data)
            
        self.message_handlers[message_id] = status_callback
        
        # Send status request
        await self.ws.send(json.dumps({
            "type": "STATUS",
            "id": message_id,
            "source": self.source
        }))
        
        # Wait for response
        response = await response_future
        
        # Clean up
        del self.message_handlers[message_id]
        
        return response

# Usage
async def main():
    client = RhetorWebSocketClient()
    await client.connect()
    
    full_response = ""
    
    def message_handler(data):
        nonlocal full_response
        
        if data.get("type") == "typing_start":
            print("LLM is typing...")
        elif data.get("type") == "typing_end":
            print("LLM finished typing")
        elif data.get("type") == "UPDATE" and "chunk" in data.get("payload", {}):
            chunk = data["payload"]["chunk"]
            full_response += chunk
            print(chunk, end="", flush=True)
        elif data.get("type") == "RESPONSE":
            message = data.get("payload", {}).get("message", "")
            full_response = message
            print(f"\nFull response: {message}")
    
    await client.send_message(
        message="What is Tekton?",
        context_id="websocket-example",
        task_type="chat",
        component="my-component",
        streaming=True,
        callback=message_handler
    )
    
    # Wait a bit for the response to complete
    await asyncio.sleep(5)
    
    # Check status
    status = await client.get_status()
    print(f"Rhetor status: {status}")
    
    await client.close()

asyncio.run(main())
```

## Python Client Integration

Rhetor provides a Python client library for easy integration:

### Installation

The client is part of the Rhetor package. Make sure it's in your Python path:

```python
import sys
import os

# Add Rhetor to Python path if needed
rhetor_path = "/path/to/Tekton/Rhetor"
if rhetor_path not in sys.path:
    sys.path.append(rhetor_path)
```

### Basic Usage

```python
import asyncio
from rhetor.client import get_rhetor_prompt_client

async def llm_example():
    # Create the client
    client = await get_rhetor_prompt_client()
    
    try:
        # Use template system
        template = await client.create_prompt_template(
            name="Component Documentation",
            template="Write documentation for {component} in the Tekton ecosystem.",
            variables=["component"],
            description="Template for component documentation"
        )
        
        # Render the template
        prompt = await client.render_prompt(
            template_id=template["template_id"],
            variables={"component": "MyComponent"}
        )
        
        print(f"Generated prompt: {prompt}")
        
        # Generate a custom prompt
        custom_prompt = await client.generate_prompt(
            task="Explain MyComponent's integration with Rhetor",
            context={"audience": "developers"},
            format="instruction"
        )
        
        print(f"Custom prompt: {custom_prompt['prompt']}")
        
    finally:
        # Close the client
        await client.close()

# Run the example
asyncio.run(llm_example())
```

### Integration in Component Class

Here's a more complete example of integrating Rhetor into a Tekton component:

```python
import asyncio
import logging
from typing import Dict, Any, Optional

# Try to import from rhetor package
try:
    from rhetor.client import get_rhetor_prompt_client
except ImportError:
    # Add fallback path if needed
    import sys
    import os
    
    rhetor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Rhetor")
    if rhetor_path not in sys.path:
        sys.path.append(rhetor_path)
    
    from rhetor.client import get_rhetor_prompt_client

class MyComponent:
    def __init__(self, component_id="my-component"):
        self.component_id = component_id
        self.logger = logging.getLogger(f"tekton.{component_id}")
        self.rhetor_client = None
        
    async def initialize(self):
        """Initialize the component and connect to Rhetor"""
        self.logger.info("Initializing component and connecting to Rhetor")
        
        try:
            # Connect to Rhetor
            self.rhetor_client = await get_rhetor_prompt_client()
            self.logger.info("Successfully connected to Rhetor")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to Rhetor: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown the component and close connections"""
        if self.rhetor_client:
            await self.rhetor_client.close()
            self.logger.info("Closed connection to Rhetor")
    
    async def process_user_query(self, query: str, session_id: str, 
                                 task_type: str = "chat") -> Dict[str, Any]:
        """Process a user query using Rhetor"""
        if not self.rhetor_client:
            raise RuntimeError("Component not initialized or Rhetor not connected")
        
        context_id = f"{self.component_id}:{session_id}"
        
        try:
            # Generate an appropriate prompt
            prompt_data = await self.rhetor_client.generate_prompt(
                task=query,
                context={"session_id": session_id},
                format="chat"
            )
            
            enhanced_prompt = prompt_data["prompt"]
            
            # TODO: Implement actual LLM request with Rhetor
            # For now, just return a dummy response
            return {
                "query": query,
                "enhanced_prompt": enhanced_prompt,
                "response": f"This is a response to: {query}",
                "context_id": context_id,
                "task_type": task_type
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query with Rhetor: {e}")
            raise

# Example usage
async def main():
    component = MyComponent()
    
    try:
        await component.initialize()
        
        response = await component.process_user_query(
            query="How does Rhetor work?",
            session_id="user123",
            task_type="reasoning"
        )
        
        print(f"Enhanced prompt: {response['enhanced_prompt']}")
        print(f"Response: {response['response']}")
        
    finally:
        await component.shutdown()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
```

## JavaScript Client Integration

For frontend or Node.js components, use the JavaScript client:

### Installation

```bash
# From the Tekton root directory
npm install ./tekton-llm-client
```

### Basic Usage

```javascript
import { RhetorClient } from 'tekton-llm-client';

class MyComponent {
  constructor() {
    this.client = new RhetorClient({
      url: 'http://localhost:8300',
      component: 'my-component'
    });
  }
  
  async initialize() {
    try {
      // Check if Rhetor is available
      const providers = await this.client.getProviders();
      console.log('Connected to Rhetor with providers:', providers);
      return true;
    } catch (error) {
      console.error('Failed to connect to Rhetor:', error);
      return false;
    }
  }
  
  async processQuery(query, sessionId, taskType = 'chat') {
    const contextId = `my-component:${sessionId}`;
    
    // Non-streaming request
    try {
      const response = await this.client.sendMessage({
        message: query,
        contextId: contextId,
        taskType: taskType,
        component: 'my-component'
      });
      
      return {
        query,
        response: response.message,
        model: response.model,
        provider: response.provider
      };
    } catch (error) {
      console.error('Error processing query:', error);
      throw error;
    }
  }
  
  processStreamingQuery(query, sessionId, taskType = 'chat', callbacks = {}) {
    const contextId = `my-component:${sessionId}`;
    
    // Streaming request
    this.client.streamMessage({
      message: query,
      contextId: contextId,
      taskType: taskType,
      component: 'my-component',
      onChunk: (chunk) => {
        if (callbacks.onChunk) callbacks.onChunk(chunk);
      },
      onComplete: (response) => {
        if (callbacks.onComplete) callbacks.onComplete(response);
      },
      onError: (error) => {
        console.error('Streaming error:', error);
        if (callbacks.onError) callbacks.onError(error);
      }
    });
  }
  
  shutdown() {
    this.client.close();
    console.log('Closed connection to Rhetor');
  }
}

// Usage example
async function main() {
  const component = new MyComponent();
  
  try {
    await component.initialize();
    
    // Non-streaming example
    const response = await component.processQuery(
      'What is Tekton?',
      'user123',
      'reasoning'
    );
    
    console.log('Response:', response);
    
    // Streaming example
    component.processStreamingQuery(
      'Explain how Rhetor works',
      'user123',
      'reasoning',
      {
        onChunk: (chunk) => {
          process.stdout.write(chunk);
        },
        onComplete: (response) => {
          console.log('\nComplete!');
          console.log('Model:', response.model);
          console.log('Provider:', response.provider);
        },
        onError: (error) => {
          console.error('Error:', error);
        }
      }
    );
    
  } finally {
    // Wait a bit for streaming to complete
    setTimeout(() => {
      component.shutdown();
    }, 5000);
  }
}

main();
```

### React Integration

Here's an example of integrating Rhetor with a React component:

```jsx
import React, { useState, useEffect, useRef } from 'react';
import { RhetorClient } from 'tekton-llm-client';

const RhetorChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [error, setError] = useState(null);
  
  const clientRef = useRef(null);
  const sessionIdRef = useRef(`session_${Date.now()}`);
  
  // Initialize Rhetor client
  useEffect(() => {
    clientRef.current = new RhetorClient({
      url: 'http://localhost:8300',
      component: 'react-ui'
    });
    
    // Cleanup on component unmount
    return () => {
      if (clientRef.current) {
        clientRef.current.close();
      }
    };
  }, []);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!input.trim() || !clientRef.current) return;
    
    // Add user message
    const userMessage = {
      role: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setError(null);
    
    const contextId = `react-ui:${sessionIdRef.current}`;
    
    try {
      let fullResponse = '';
      
      // Stream the response
      clientRef.current.streamMessage({
        message: input,
        contextId: contextId,
        taskType: 'chat',
        component: 'react-ui',
        onChunk: (chunk) => {
          fullResponse += chunk;
          
          // Update the partial response
          setMessages(prev => {
            const newMessages = [...prev];
            
            // Find or create the assistant message
            const assistantIndex = newMessages.findIndex(
              msg => msg.role === 'assistant' && msg.isPartial
            );
            
            if (assistantIndex >= 0) {
              // Update existing partial message
              newMessages[assistantIndex] = {
                ...newMessages[assistantIndex],
                content: fullResponse
              };
            } else {
              // Create new partial message
              newMessages.push({
                role: 'assistant',
                content: fullResponse,
                isPartial: true,
                timestamp: new Date().toISOString()
              });
            }
            
            return newMessages;
          });
        },
        onComplete: (response) => {
          setIsTyping(false);
          
          // Update with final message
          setMessages(prev => {
            const newMessages = prev.filter(msg => !msg.isPartial);
            
            newMessages.push({
              role: 'assistant',
              content: response.message,
              model: response.model,
              provider: response.provider,
              timestamp: new Date().toISOString()
            });
            
            return newMessages;
          });
        },
        onError: (error) => {
          setIsTyping(false);
          setError(error.message || 'Error communicating with Rhetor');
          
          // Remove partial message if there was one
          setMessages(prev => prev.filter(msg => !msg.isPartial));
        }
      });
    } catch (error) {
      setIsTyping(false);
      setError(error.message || 'Error communicating with Rhetor');
    }
  };
  
  return (
    <div className="rhetor-chat">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="content">{msg.content}</div>
            {msg.model && (
              <div className="metadata">
                Model: {msg.model} ({msg.provider})
              </div>
            )}
          </div>
        ))}
        
        {isTyping && (
          <div className="typing-indicator">
            <span>Thinking</span>
            <span className="dot">.</span>
            <span className="dot">.</span>
            <span className="dot">.</span>
          </div>
        )}
        
        {error && (
          <div className="error">
            Error: {error}
          </div>
        )}
      </div>
      
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question..."
          disabled={isTyping}
        />
        <button type="submit" disabled={isTyping || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
};

export default RhetorChat;
```

## Best Practices

### Context Management

1. **Use Consistent Context IDs**:
   - Format: `{component-id}:{session-or-user-id}`
   - Example: `terma:user123` or `ergon:session456`

2. **Consider Context Lifecycle**:
   - Start new contexts for new sessions
   - Delete contexts when they're no longer needed
   - Summarize long contexts to reduce token usage

### Task Types

Choose the appropriate task type to get the best model for your needs:

| Task Type | Use Case | Example |
|-----------|----------|---------|
| `code` | Code generation, analysis | "Create a function that validates email addresses" |
| `planning` | Planning, design | "Design a database schema for user management" |
| `reasoning` | Logic, inference | "What would happen if we modify this class?" |
| `chat` | Simple conversations | "What is Tekton?" |
| `default` | General purpose | Fallback for other tasks |

### Budget Awareness

1. **Set Appropriate Task Types**:
   - Avoid using high-tier models for simple tasks
   - Use appropriate task types to get the best cost-performance ratio

2. **Handle Budget Warnings**:
   - Check response for budget warnings
   - Inform users when approaching limits
   - Implement fallback behavior when limits are exceeded

### Error Handling

```python
try:
    response = await send_llm_request(message, context_id, task_type)
except Exception as e:
    # Log the error
    logger.error(f"Error communicating with Rhetor: {e}")
    
    # Implement fallback behavior
    response = {
        "message": "I'm sorry, but I couldn't process your request at this time.",
        "error": str(e)
    }
```

### Security Considerations

1. **Validate User Input**:
   - Sanitize user input before sending to Rhetor
   - Implement input length limits
   - Check for malicious content

2. **Handle Sensitive Information**:
   - Don't include sensitive data in prompts
   - Be cautious with context persistence
   - Consider implementing context expiration

## Troubleshooting

### Connection Issues

```python
import aiohttp
import asyncio

async def check_rhetor_health():
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8300/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unhealthy", "error": await response.text()}
    except aiohttp.ClientError as e:
        return {"status": "unavailable", "error": str(e)}

# Usage
health = await check_rhetor_health()
if health["status"] != "healthy":
    print(f"Rhetor is not healthy: {health.get('error', 'Unknown error')}")
    # Implement fallback behavior
```

### Fallback Options

```python
async def send_message_with_fallback(message, context_id, task_type="default"):
    try:
        # Try Rhetor first
        return await send_llm_request(message, context_id, task_type)
    except Exception as rhetor_error:
        logger.warning(f"Rhetor unavailable: {rhetor_error}")
        
        try:
            # Fallback to direct API call
            # Implement your direct API call to Claude, GPT, etc.
            pass
        except Exception as api_error:
            logger.error(f"API fallback failed: {api_error}")
            
            # Final fallback: Return a static response
            return {
                "message": "I'm sorry, but I couldn't process your request at this time.",
                "error": "LLM services unavailable"
            }
```

## Example: Full Component Integration

Here's a complete example of integrating Rhetor into a Tekton component:

```python
import asyncio
import logging
import uuid
import os
from typing import Dict, Any, Optional, List

try:
    from rhetor.client import get_rhetor_prompt_client
    from tekton.utils.component_client import ComponentClient
except ImportError:
    import sys
    # Add paths if needed
    tekton_path = os.path.dirname(os.path.dirname(__file__))
    if tekton_path not in sys.path:
        sys.path.append(tekton_path)
    
    from rhetor.client import get_rhetor_prompt_client
    from tekton.utils.component_client import ComponentClient

class MyTektonComponent:
    def __init__(self, component_id="my-component"):
        self.component_id = component_id
        self.logger = logging.getLogger(f"tekton.{component_id}")
        self.rhetor_client = None
        self.hermes_client = None
        self.sessions = {}
        
    async def initialize(self):
        """Initialize the component and connect to required services"""
        self.logger.info(f"Initializing {self.component_id}")
        
        # Connect to Hermes for service discovery
        try:
            hermes_url = os.environ.get("HERMES_URL", "http://localhost:8100")
            self.hermes_client = ComponentClient(
                component_id="hermes",
                hermes_url=hermes_url
            )
            self.logger.info("Connected to Hermes")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Hermes: {e}")
        
        # Connect to Rhetor
        try:
            self.rhetor_client = await get_rhetor_prompt_client()
            self.logger.info("Connected to Rhetor")
        except Exception as e:
            self.logger.error(f"Failed to connect to Rhetor: {e}")
            return False
        
        # Set up system prompts
        try:
            await self._setup_system_prompts()
        except Exception as e:
            self.logger.warning(f"Failed to set up system prompts: {e}")
        
        return True
    
    async def _setup_system_prompts(self):
        """Set up system prompts for this component"""
        if not self.rhetor_client:
            return
        
        # Create a system prompt for this component
        try:
            system_prompt = f"""You are {self.component_id}, a Tekton component that helps users with tasks related to your domain.
            
Your responsibilities:
1. Provide accurate information about {self.component_id}
2. Help users understand how to use your features
3. Maintain a helpful and professional tone

When responding:
- Be concise and clear
- Provide examples when helpful
- Refer to other Tekton components when appropriate
"""
            
            await self.rhetor_client.create_prompt(
                name=f"{self.component_id} System Prompt",
                component=self.component_id,
                content=system_prompt,
                description=f"System prompt for {self.component_id}",
                tags=["system", self.component_id],
                is_default=True
            )
            
            self.logger.info(f"Created system prompt for {self.component_id}")
        except Exception as e:
            self.logger.warning(f"Failed to create system prompt: {e}")
    
    async def create_session(self, user_id: str) -> str:
        """Create a new session for a user"""
        session_id = str(uuid.uuid4())
        context_id = f"{self.component_id}:{session_id}"
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "context_id": context_id,
            "created_at": asyncio.get_event_loop().time(),
            "messages": []
        }
        
        self.logger.info(f"Created session {session_id} for user {user_id}")
        return session_id
    
    async def process_message(
        self, 
        session_id: str, 
        message: str, 
        task_type: str = "chat", 
        streaming: bool = False,
        callbacks: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a user message"""
        if session_id not in self.sessions:
            raise ValueError(f"Invalid session ID: {session_id}")
        
        session = self.sessions[session_id]
        context_id = session["context_id"]
        
        # Add message to session history
        session["messages"].append({
            "role": "user",
            "content": message,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        if not self.rhetor_client:
            return {
                "error": "Rhetor client not available",
                "message": "Sorry, I cannot process your request at this time."
            }
        
        try:
            if streaming and callbacks:
                # Streaming response
                full_response = ""
                
                async for chunk in self._stream_from_rhetor(
                    message=message,
                    context_id=context_id,
                    task_type=task_type
                ):
                    full_response += chunk
                    
                    # Call the chunk callback if provided
                    if "on_chunk" in callbacks:
                        callbacks["on_chunk"](chunk)
                
                # Add response to session history
                session["messages"].append({
                    "role": "assistant",
                    "content": full_response,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                response = {
                    "message": full_response,
                    "context_id": context_id,
                    "streaming": True
                }
                
                # Call the complete callback if provided
                if "on_complete" in callbacks:
                    callbacks["on_complete"](response)
                
                return response
            else:
                # Non-streaming response
                response = await self._get_from_rhetor(
                    message=message,
                    context_id=context_id,
                    task_type=task_type
                )
                
                # Add response to session history
                session["messages"].append({
                    "role": "assistant",
                    "content": response["message"],
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                return response
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            
            # Call the error callback if provided
            if callbacks and "on_error" in callbacks:
                callbacks["on_error"](str(e))
            
            return {
                "error": str(e),
                "message": "Sorry, I encountered an error processing your request."
            }
    
    async def _get_from_rhetor(
        self, 
        message: str, 
        context_id: str, 
        task_type: str = "chat"
    ) -> Dict[str, Any]:
        """Get a response from Rhetor"""
        try:
            # Generate prompt if needed
            generated_prompt = await self.rhetor_client.generate_prompt(
                task=message,
                context={"component": self.component_id},
                format="chat"
            )
            
            enhanced_message = generated_prompt["prompt"]
            
            # TODO: Replace with actual Rhetor client implementation 
            # This is a placeholder for the actual client logic
            
            return {
                "message": f"Response to: {enhanced_message}",
                "model": "claude-3-sonnet-20240229",
                "provider": "anthropic",
                "context": context_id,
                "finished": True
            }
        except Exception as e:
            self.logger.error(f"Error getting response from Rhetor: {e}")
            raise
    
    async def _stream_from_rhetor(
        self, 
        message: str, 
        context_id: str, 
        task_type: str = "chat"
    ):
        """Stream a response from Rhetor"""
        try:
            # Generate prompt if needed
            generated_prompt = await self.rhetor_client.generate_prompt(
                task=message,
                context={"component": self.component_id},
                format="chat"
            )
            
            enhanced_message = generated_prompt["prompt"]
            
            # TODO: Replace with actual Rhetor client implementation
            # This is a placeholder for the actual client streaming logic
            
            # Simulate streaming response
            chunks = ["This ", "is ", "a ", "simulated ", "streaming ", "response."]
            
            for chunk in chunks:
                await asyncio.sleep(0.2)  # Simulate network delay
                yield chunk
        except Exception as e:
            self.logger.error(f"Error streaming from Rhetor: {e}")
            raise
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get the message history for a session"""
        if session_id not in self.sessions:
            raise ValueError(f"Invalid session ID: {session_id}")
        
        return self.sessions[session_id]["messages"]
    
    async def shutdown(self):
        """Shutdown the component and close connections"""
        if self.rhetor_client:
            await self.rhetor_client.close()
            self.logger.info("Closed connection to Rhetor")
        
        if self.hermes_client:
            await self.hermes_client.close()
            self.logger.info("Closed connection to Hermes")

# Example usage
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    component = MyTektonComponent(component_id="example-component")
    
    try:
        await component.initialize()
        
        # Create a session
        session_id = await component.create_session(user_id="user123")
        
        # Process a message (non-streaming)
        response = await component.process_message(
            session_id=session_id,
            message="What can you help me with?",
            task_type="chat",
            streaming=False
        )
        
        print(f"Response: {response['message']}")
        
        # Process a message (streaming)
        await component.process_message(
            session_id=session_id,
            message="Tell me about Rhetor",
            task_type="reasoning",
            streaming=True,
            callbacks={
                "on_chunk": lambda chunk: print(chunk, end="", flush=True),
                "on_complete": lambda resp: print(f"\nComplete! Model: {resp.get('model')}"),
                "on_error": lambda err: print(f"\nError: {err}")
            }
        )
        
        # Get session history
        history = await component.get_session_history(session_id)
        print(f"\nSession history: {len(history)} messages")
        
    finally:
        await component.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
```

## Reference

- [Rhetor API Reference](./api_reference.md)
- [Rhetor Technical Documentation](./technical_documentation.md)
- [Rhetor Quick Reference](./quick_reference.md)
- [Tekton Single Port Architecture](../../docs/SINGLE_PORT_ARCHITECTURE.md)