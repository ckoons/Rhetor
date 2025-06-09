#!/usr/bin/env python3
"""Debug script to check MCP tool metadata."""

import sys
import os
sys.path.insert(0, '/Users/cskoons/projects/github/Tekton')
sys.path.insert(0, '/Users/cskoons/projects/github/Tekton/Rhetor')

from Rhetor.rhetor.core.mcp.tools import (
    get_available_models, set_default_model, get_model_capabilities,
    test_model_connection, get_model_performance, manage_model_rotation,
    create_prompt_template, optimize_prompt, validate_prompt_syntax,
    get_prompt_history, analyze_prompt_performance, manage_prompt_library,
    analyze_context_usage, optimize_context_window, track_context_history,
    compress_context, list_ai_specialists, activate_ai_specialist,
    send_message_to_specialist, orchestrate_team_chat,
    get_specialist_conversation_history, configure_ai_orchestration
)

# Check if tools have metadata
all_tools = [
    # LLM Management tools
    get_available_models, set_default_model, get_model_capabilities,
    test_model_connection, get_model_performance, manage_model_rotation,
    # Prompt Engineering tools
    create_prompt_template, optimize_prompt, validate_prompt_syntax,
    get_prompt_history, analyze_prompt_performance, manage_prompt_library,
    # Context Management tools
    analyze_context_usage, optimize_context_window, track_context_history,
    compress_context,
    # AI Orchestration tools
    list_ai_specialists, activate_ai_specialist, send_message_to_specialist,
    orchestrate_team_chat, get_specialist_conversation_history, configure_ai_orchestration
]

print("Checking MCP tool metadata...\n")

tools_with_meta = 0
tools_without_meta = 0

for tool_func in all_tools:
    if hasattr(tool_func, '_mcp_tool_meta'):
        tools_with_meta += 1
        meta = tool_func._mcp_tool_meta
        print(f"✓ {tool_func.__name__} - has metadata")
        if hasattr(meta, 'name'):
            print(f"  Name: {meta.name}")
        if hasattr(meta, 'to_dict'):
            # Try to convert to dict
            try:
                meta_dict = meta.to_dict()
                print(f"  Dict keys: {list(meta_dict.keys())}")
            except Exception as e:
                print(f"  Error converting to dict: {e}")
    else:
        tools_without_meta += 1
        print(f"✗ {tool_func.__name__} - NO metadata")

print(f"\nSummary:")
print(f"Tools with metadata: {tools_with_meta}")
print(f"Tools without metadata: {tools_without_meta}")

# Check if FastMCP is available
print("\nChecking FastMCP availability...")
try:
    from tekton.mcp.fastmcp.decorators import mcp_tool
    print("✓ FastMCP decorators available")
    print(f"  mcp_tool type: {type(mcp_tool)}")
except ImportError as e:
    print(f"✗ FastMCP decorators not available: {e}")

# Check if the decorator is being applied
print("\nChecking decorator application...")
import inspect
source = inspect.getsource(get_available_models)
if "@mcp_tool" in source:
    print("✓ @mcp_tool decorator is in source code")
else:
    print("✗ @mcp_tool decorator NOT found in source")