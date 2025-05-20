"""
Rhetor MCP (Model Context Protocol) integration.

This module provides MCP capabilities and tools for Rhetor's LLM management,
prompt engineering, and context management features.
"""

from .capabilities import (
    LLMManagementCapability,
    PromptEngineeringCapability,
    ContextManagementCapability
)

from .tools import (
    llm_management_tools,
    prompt_engineering_tools,
    context_management_tools
)


def get_all_capabilities():
    """Get all Rhetor MCP capabilities."""
    return [
        LLMManagementCapability,
        PromptEngineeringCapability,
        ContextManagementCapability
    ]


def get_all_tools():
    """Get all Rhetor MCP tools."""
    return llm_management_tools + prompt_engineering_tools + context_management_tools


__all__ = [
    "LLMManagementCapability",
    "PromptEngineeringCapability", 
    "ContextManagementCapability",
    "llm_management_tools",
    "prompt_engineering_tools",
    "context_management_tools",
    "get_all_capabilities",
    "get_all_tools"
]