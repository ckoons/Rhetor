"""Command-line argument parser for Rhetor.

This module provides the argument parser for the Rhetor CLI.
"""

import argparse
from typing import List, Optional


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the Rhetor CLI.
    
    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(description="Rhetor - Tekton's communication specialist")
    
    # Add global options
    parser.add_argument("--data-dir", help="Data directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    # Add subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Prompt commands
    prompt_parser = subparsers.add_parser("prompt", help="Prompt commands")
    prompt_subparsers = prompt_parser.add_subparsers(dest="subcommand", help="Prompt subcommand")
    
    prompt_create_parser = prompt_subparsers.add_parser("create", help="Create a prompt template")
    prompt_create_parser.add_argument("name", help="Template name")
    prompt_create_parser.add_argument("template", help="Template text")
    prompt_create_parser.add_argument("--variables", help="Variables (comma-separated)")
    prompt_create_parser.add_argument("--description", help="Template description")
    prompt_create_parser.add_argument("--model-type", help="Model type this template is optimized for")
    
    prompt_list_parser = prompt_subparsers.add_parser("list", help="List prompt templates")
    
    prompt_show_parser = prompt_subparsers.add_parser("show", help="Show a prompt template")
    prompt_show_parser.add_argument("name", help="Template name")
    
    prompt_delete_parser = prompt_subparsers.add_parser("delete", help="Delete a prompt template")
    prompt_delete_parser.add_argument("name", help="Template name")
    
    prompt_generate_parser = prompt_subparsers.add_parser("generate", help="Generate a prompt")
    prompt_generate_parser.add_argument("name", help="Template name")
    prompt_generate_parser.add_argument("--component", help="Component to adapt for")
    prompt_generate_parser.add_argument("--variables", help="Variable values (key=value,...)")
    
    # System prompt commands
    system_parser = subparsers.add_parser("system", help="System prompt commands")
    system_subparsers = system_parser.add_subparsers(dest="subcommand", help="System prompt subcommand")
    
    system_create_parser = system_subparsers.add_parser("create", help="Create a system prompt")
    system_create_parser.add_argument("component", help="Component name")
    system_create_parser.add_argument("--role", help="Override role description")
    system_create_parser.add_argument("--capabilities", help="Override capabilities (comma-separated)")
    system_create_parser.add_argument("--tone", help="Override tone")
    system_create_parser.add_argument("--focus", help="Override focus")
    system_create_parser.add_argument("--style", help="Override style")
    system_create_parser.add_argument("--personality", help="Override personality")
    system_create_parser.add_argument("--output", help="Output file")
    
    system_show_parser = system_subparsers.add_parser("show", help="Show a system prompt")
    system_show_parser.add_argument("component", help="Component name")
    
    system_list_components_parser = system_subparsers.add_parser("list-components", help="List available components")
    
    # Message commands
    message_parser = subparsers.add_parser("message", help="Message commands")
    message_subparsers = message_parser.add_subparsers(dest="subcommand", help="Message subcommand")
    
    message_send_parser = message_subparsers.add_parser("send", help="Send a message")
    message_send_parser.add_argument("content", help="Message content")
    message_send_parser.add_argument("--recipient", help="Recipient component")
    message_send_parser.add_argument("--conversation", help="Conversation ID")
    message_send_parser.add_argument("--type", dest="message_type", default="text", help="Message type")
    
    message_list_parser = message_subparsers.add_parser("list", help="List messages")
    message_list_parser.add_argument("conversation", help="Conversation ID")
    message_list_parser.add_argument("--limit", type=int, help="Maximum number of messages")
    
    message_show_parser = message_subparsers.add_parser("show", help="Show a message")
    message_show_parser.add_argument("conversation", help="Conversation ID")
    message_show_parser.add_argument("message", help="Message ID")
    
    # Conversation commands
    conversation_parser = subparsers.add_parser("conversation", help="Conversation commands")
    conversation_subparsers = conversation_parser.add_subparsers(dest="subcommand", help="Conversation subcommand")
    
    conversation_create_parser = conversation_subparsers.add_parser("create", help="Create a conversation")
    conversation_create_parser.add_argument("participants", help="Participants (comma-separated)")
    
    conversation_list_parser = conversation_subparsers.add_parser("list", help="List conversations")
    
    conversation_show_parser = conversation_subparsers.add_parser("show", help="Show a conversation")
    conversation_show_parser.add_argument("conversation", help="Conversation ID")
    
    # Hermes commands
    hermes_parser = subparsers.add_parser("hermes", help="Hermes commands")
    hermes_subparsers = hermes_parser.add_subparsers(dest="subcommand", help="Hermes subcommand")
    
    hermes_register_parser = hermes_subparsers.add_parser("register", help="Register with Hermes")
    
    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Parsed arguments
    """
    parser = create_parser()
    return parser.parse_args(args)