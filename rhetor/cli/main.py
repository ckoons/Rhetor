"""Command-line interface for Rhetor.

This module provides a command-line interface for interacting with Rhetor.
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable

from rhetor.core.prompt_engine import PromptEngine, PromptLibrary, PromptTemplate
from rhetor.core.communication import CommunicationEngine, Message, Conversation
from rhetor.cli.cli_parser import parse_args
from rhetor.cli.cli_commands import (
    # Prompt commands
    create_prompt, list_prompts, show_prompt, delete_prompt, generate_prompt,
    # System prompt commands
    create_system_prompt, show_system_prompt, list_components,
    # Message commands
    send_message, list_messages, show_message,
    # Conversation commands
    create_conversation, list_conversations, show_conversation,
    # Hermes commands
    register_with_hermes
)

logger = logging.getLogger(__name__)


class RhetorCLI:
    """Command-line interface for Rhetor."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize the CLI.
        
        Args:
            data_dir: Directory for storing data
        """
        # Set up data directory
        if data_dir:
            self.data_dir = data_dir
        else:
            home_dir = os.path.expanduser("~")
            self.data_dir = os.path.join(home_dir, ".tekton", "data", "rhetor")
        
        os.makedirs(self.data_dir, exist_ok=True)
        self.templates_dir = os.path.join(self.data_dir, "templates")
        os.makedirs(self.templates_dir, exist_ok=True)
        self.conversations_dir = os.path.join(self.data_dir, "conversations")
        os.makedirs(self.conversations_dir, exist_ok=True)
        
        # Initialize modules
        self.prompt_library = PromptLibrary(self.templates_dir)
        self.prompt_engine = PromptEngine(self.prompt_library)
        self.communication_engine = CommunicationEngine("rhetor")
        
        # Load conversations
        self.communication_engine.load_conversations(self.conversations_dir)
        
        # Set up command handlers
        self.commands = {
            "prompt": {
                "create": self.create_prompt,
                "list": self.list_prompts,
                "show": self.show_prompt,
                "delete": self.delete_prompt,
                "generate": self.generate_prompt,
            },
            "system": {
                "create": self.create_system_prompt,
                "show": self.show_system_prompt,
                "list-components": self.list_components,
            },
            "message": {
                "send": self.send_message,
                "list": self.list_messages,
                "show": self.show_message,
            },
            "conversation": {
                "create": self.create_conversation,
                "list": self.list_conversations,
                "show": self.show_conversation,
            },
            "hermes": {
                "register": self.register_with_hermes,
            }
        }
    
    def run(self, args: Optional[List[str]] = None) -> None:
        """Run the CLI with the given arguments.
        
        Args:
            args: Command-line arguments
        """
        # Parse arguments
        parsed_args = parse_args(args)
        
        # Set up logging
        log_level = logging.DEBUG if parsed_args.debug else logging.INFO
        logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Override data directory if specified
        if parsed_args.data_dir:
            self.data_dir = parsed_args.data_dir
            self.templates_dir = os.path.join(self.data_dir, "templates")
            os.makedirs(self.templates_dir, exist_ok=True)
            self.conversations_dir = os.path.join(self.data_dir, "conversations")
            os.makedirs(self.conversations_dir, exist_ok=True)
            
            # Reinitialize modules
            self.prompt_library = PromptLibrary(self.templates_dir)
            self.prompt_engine = PromptEngine(self.prompt_library)
            self.communication_engine = CommunicationEngine("rhetor")
            self.communication_engine.load_conversations(self.conversations_dir)
        
        # Execute command
        if not parsed_args.command:
            parser = parse_args()
            parser.print_help()
            return
        
        if not parsed_args.subcommand:
            from rhetor.cli.cli_parser import create_parser
            parser = create_parser()
            subparsers = parser._subparsers._group_actions[0]
            subparsers._name_parser_map[parsed_args.command].print_help()
            return
        
        # Look up and execute the appropriate command handler
        cmd_group = self.commands.get(parsed_args.command, {})
        cmd_handler = cmd_group.get(parsed_args.subcommand)
        
        if cmd_handler:
            # Convert args to dictionary and remove command and subcommand
            args_dict = vars(parsed_args)
            args_dict.pop("command")
            args_dict.pop("subcommand")
            args_dict.pop("debug", None)
            args_dict.pop("data_dir", None)
            
            # For commands that may need asyncio
            if parsed_args.command == "hermes":
                asyncio.run(cmd_handler(**args_dict))
            else:
                cmd_handler(**args_dict)
        else:
            print(f"Unknown command: {parsed_args.command} {parsed_args.subcommand}")
    
    # Command handler methods that delegate to module functions
    
    # Prompt commands
    def create_prompt(self, **kwargs) -> None:
        create_prompt(self.prompt_library, self.templates_dir, **kwargs)
    
    def list_prompts(self, **kwargs) -> None:
        list_prompts(self.prompt_library, **kwargs)
    
    def show_prompt(self, **kwargs) -> None:
        show_prompt(self.prompt_library, **kwargs)
    
    def delete_prompt(self, **kwargs) -> None:
        delete_prompt(self.prompt_library, self.templates_dir, **kwargs)
    
    def generate_prompt(self, **kwargs) -> None:
        generate_prompt(self.prompt_engine, **kwargs)
    
    # System prompt commands
    def create_system_prompt(self, **kwargs) -> None:
        create_system_prompt(**kwargs)
    
    def show_system_prompt(self, **kwargs) -> None:
        show_system_prompt(**kwargs)
    
    def list_components(self, **kwargs) -> None:
        list_components(**kwargs)
    
    # Message commands
    def send_message(self, **kwargs) -> None:
        send_message(self.communication_engine, self.conversations_dir, **kwargs)
    
    def list_messages(self, **kwargs) -> None:
        list_messages(self.communication_engine, **kwargs)
    
    def show_message(self, **kwargs) -> None:
        show_message(self.communication_engine, **kwargs)
    
    # Conversation commands
    def create_conversation(self, **kwargs) -> None:
        create_conversation(self.communication_engine, self.conversations_dir, **kwargs)
    
    def list_conversations(self, **kwargs) -> None:
        list_conversations(self.communication_engine, **kwargs)
    
    def show_conversation(self, **kwargs) -> None:
        show_conversation(self.communication_engine, **kwargs)
    
    # Hermes commands
    async def register_with_hermes(self, **kwargs) -> None:
        await register_with_hermes(self.prompt_engine, self.communication_engine, **kwargs)


def main() -> None:
    """Run the CLI."""
    cli = RhetorCLI()
    cli.run()


if __name__ == "__main__":
    main()