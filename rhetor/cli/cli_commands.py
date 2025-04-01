"""Command implementations for the Rhetor CLI.

This module provides the implementation of commands for the Rhetor CLI.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable

from rhetor.core.prompt_engine import PromptEngine, PromptLibrary, PromptTemplate
from rhetor.core.communication import CommunicationEngine, Message, Conversation
from rhetor.cli.cli_helpers import format_timestamp, parse_key_value_pairs

logger = logging.getLogger(__name__)


# Prompt commands
def create_prompt(
    prompt_library: PromptLibrary,
    templates_dir: str,
    name: str,
    template: str,
    variables: Optional[str] = None,
    description: Optional[str] = None,
    model_type: Optional[str] = None
) -> None:
    """Create a prompt template.
    
    Args:
        prompt_library: Prompt library
        templates_dir: Templates directory
        name: Template name
        template: Template text
        variables: Variables (comma-separated)
        description: Template description
        model_type: Model type this template is optimized for
    """
    # Parse variables
    variable_list = variables.split(",") if variables else []
    
    # Create the template
    prompt_template = PromptTemplate(
        template=template,
        variables=variable_list,
        name=name,
        description=description,
        model_type=model_type
    )
    
    # Add to library
    prompt_library.add_template(name, prompt_template)
    
    # Save to disk
    prompt_library.save_to_directory(templates_dir)
    
    print(f"Created prompt template '{name}'")


def list_prompts(prompt_library: PromptLibrary) -> None:
    """List all prompt templates.
    
    Args:
        prompt_library: Prompt library
    """
    templates = prompt_library.templates
    
    if not templates:
        print("No prompt templates found")
        return
    
    print("Prompt templates:")
    for name, template in templates.items():
        print(f"  {name}: {template.description or 'No description'}")
        if template.model_type:
            print(f"    Model type: {template.model_type}")
        print(f"    Variables: {', '.join(template.variables)}")


def show_prompt(prompt_library: PromptLibrary, name: str) -> None:
    """Show a prompt template.
    
    Args:
        prompt_library: Prompt library
        name: Template name
    """
    try:
        template = prompt_library.get_template(name)
        
        print(f"Prompt template: {name}")
        if template.description:
            print(f"Description: {template.description}")
        if template.model_type:
            print(f"Model type: {template.model_type}")
        print(f"Variables: {', '.join(template.variables)}")
        print("\nTemplate:")
        print(template.template)
    except KeyError:
        print(f"Prompt template '{name}' not found")


def delete_prompt(prompt_library: PromptLibrary, templates_dir: str, name: str) -> None:
    """Delete a prompt template.
    
    Args:
        prompt_library: Prompt library
        templates_dir: Templates directory
        name: Template name
    """
    if name in prompt_library.templates:
        del prompt_library.templates[name]
        
        # Save changes
        prompt_library.save_to_directory(templates_dir)
        
        print(f"Deleted prompt template '{name}'")
    else:
        print(f"Prompt template '{name}' not found")


def generate_prompt(
    prompt_engine: PromptEngine,
    name: str,
    component: Optional[str] = None,
    variables: Optional[str] = None
) -> None:
    """Generate a prompt from a template.
    
    Args:
        prompt_engine: Prompt engine
        name: Template name
        component: Component to adapt for
        variables: Variable values (key=value,...)
    """
    try:
        # Parse variables
        variable_dict = parse_key_value_pairs(variables) if variables else {}
        
        # Generate the prompt
        prompt = prompt_engine.generate_prompt(
            template_name=name,
            component_name=component,
            **variable_dict
        )
        
        print("Generated prompt:")
        print(prompt)
    except Exception as e:
        print(f"Error generating prompt: {e}")


# System prompt commands
def create_system_prompt(
    component: str,
    role: Optional[str] = None,
    capabilities: Optional[str] = None,
    tone: Optional[str] = None,
    focus: Optional[str] = None,
    style: Optional[str] = None,
    personality: Optional[str] = None,
    output: Optional[str] = None
) -> None:
    """Create a system prompt for a component.
    
    Args:
        component: Component name
        role: Override role description
        capabilities: Override capabilities (comma-separated)
        tone: Override tone
        focus: Override focus
        style: Override style
        personality: Override personality
        output: Output file
    """
    try:
        from rhetor.templates.system_prompts import get_system_prompt
        
        # Prepare custom fields
        custom_fields = {}
        if role:
            custom_fields["role_description"] = role
        if capabilities:
            custom_fields["capabilities"] = capabilities.replace(",", "\n- ")
        if tone:
            custom_fields["tone"] = tone
        if focus:
            custom_fields["focus"] = focus
        if style:
            custom_fields["style"] = style
        if personality:
            custom_fields["personality"] = personality
        
        # Generate the system prompt
        system_prompt = get_system_prompt(component, custom_fields)
        
        # Output
        if output:
            with open(output, "w") as f:
                f.write(system_prompt)
            print(f"Saved system prompt to {output}")
        else:
            print("System prompt:")
            print(system_prompt)
    except Exception as e:
        print(f"Error creating system prompt: {e}")


def show_system_prompt(component: str) -> None:
    """Show the system prompt for a component.
    
    Args:
        component: Component name
    """
    try:
        from rhetor.templates.system_prompts import get_system_prompt
        
        system_prompt = get_system_prompt(component)
        print(system_prompt)
    except Exception as e:
        print(f"Error showing system prompt: {e}")


def list_components() -> None:
    """List all available components."""
    try:
        from rhetor.templates.system_prompts import COMPONENT_PROMPTS
        
        print("Available components:")
        for component, data in COMPONENT_PROMPTS.items():
            print(f"  {component}: {data.get('role_description', '').split('.',1)[0]}.")
    except Exception as e:
        print(f"Error listing components: {e}")


# Message commands
def send_message(
    communication_engine: CommunicationEngine,
    conversations_dir: str,
    content: str,
    recipient: Optional[str] = None,
    conversation: Optional[str] = None,
    message_type: str = "text"
) -> None:
    """Send a message.
    
    Args:
        communication_engine: Communication engine
        conversations_dir: Conversations directory
        content: Message content
        recipient: Recipient component
        conversation: Conversation ID
        message_type: Message type
    """
    message = communication_engine.send_message(
        content=content,
        recipient=recipient,
        conversation_id=conversation,
        message_type=message_type
    )
    
    # Save conversations
    communication_engine.save_conversations(conversations_dir)
    
    print(f"Sent message {message.message_id}")
    if not conversation and message.conversation_id:
        print(f"Created new conversation {message.conversation_id}")


def list_messages(
    communication_engine: CommunicationEngine,
    conversation: str,
    limit: Optional[int] = None
) -> None:
    """List messages in a conversation.
    
    Args:
        communication_engine: Communication engine
        conversation: Conversation ID
        limit: Maximum number of messages
    """
    conversation_obj = communication_engine.get_conversation(conversation)
    
    if not conversation_obj:
        print(f"Conversation {conversation} not found")
        return
    
    messages = conversation_obj.get_messages(limit=limit)
    
    if not messages:
        print("No messages found")
        return
    
    print(f"Messages in conversation {conversation}:")
    for message in messages:
        sender_str = message.sender
        recipient_str = f" to {message.recipient}" if message.recipient else ""
        print(f"  {message.message_id}: [{message.message_type}] {sender_str}{recipient_str}")
        print(f"    Content: {message.content[:50]}{'...' if len(message.content) > 50 else ''}")
        print(f"    Time: {format_timestamp(message.timestamp)}")


def show_message(
    communication_engine: CommunicationEngine,
    conversation: str,
    message: str
) -> None:
    """Show a message.
    
    Args:
        communication_engine: Communication engine
        conversation: Conversation ID
        message: Message ID
    """
    conversation_obj = communication_engine.get_conversation(conversation)
    
    if not conversation_obj:
        print(f"Conversation {conversation} not found")
        return
    
    # Find the message
    message_obj = None
    for msg in conversation_obj.messages:
        if msg.message_id == message:
            message_obj = msg
            break
    
    if not message_obj:
        print(f"Message {message} not found in conversation {conversation}")
        return
    
    print(f"Message: {message_obj.message_id}")
    print(f"Conversation: {conversation}")
    print(f"From: {message_obj.sender}")
    if message_obj.recipient:
        print(f"To: {message_obj.recipient}")
    print(f"Type: {message_obj.message_type}")
    print(f"Time: {format_timestamp(message_obj.timestamp)}")
    print("\nContent:")
    print(message_obj.content)
    
    if message_obj.references:
        print("\nReferences:")
        for ref in message_obj.references:
            print(f"  {ref}")
    
    if message_obj.metadata:
        print("\nMetadata:")
        for key, value in message_obj.metadata.items():
            print(f"  {key}: {value}")


# Conversation commands
def create_conversation(
    communication_engine: CommunicationEngine,
    conversations_dir: str,
    participants: str
) -> None:
    """Create a conversation.
    
    Args:
        communication_engine: Communication engine
        conversations_dir: Conversations directory
        participants: Participants (comma-separated)
    """
    # Parse participants
    participant_list = [p.strip() for p in participants.split(",")]
    
    # Create the conversation
    conversation_id = communication_engine.create_conversation(participant_list)
    
    # Save conversations
    communication_engine.save_conversations(conversations_dir)
    
    print(f"Created conversation {conversation_id} with participants: {participants}")


def list_conversations(communication_engine: CommunicationEngine) -> None:
    """List all conversations.
    
    Args:
        communication_engine: Communication engine
    """
    conversations = communication_engine.conversations
    
    if not conversations:
        print("No conversations found")
        return
    
    print("Conversations:")
    for conversation_id, conversation in conversations.items():
        participants_str = ", ".join(conversation.participants)
        message_count = len(conversation.messages)
        print(f"  {conversation_id}: {participants_str} ({message_count} messages)")
        print(f"    Created: {format_timestamp(conversation.created_at)}")
        print(f"    Updated: {format_timestamp(conversation.updated_at)}")


def show_conversation(communication_engine: CommunicationEngine, conversation: str) -> None:
    """Show a conversation.
    
    Args:
        communication_engine: Communication engine
        conversation: Conversation ID
    """
    conversation_obj = communication_engine.get_conversation(conversation)
    
    if not conversation_obj:
        print(f"Conversation {conversation} not found")
        return
    
    print(f"Conversation: {conversation}")
    print(f"Participants: {', '.join(conversation_obj.participants)}")
    print(f"Created: {format_timestamp(conversation_obj.created_at)}")
    print(f"Updated: {format_timestamp(conversation_obj.updated_at)}")
    print(f"Messages: {len(conversation_obj.messages)}")
    
    if conversation_obj.metadata:
        print("\nMetadata:")
        for key, value in conversation_obj.metadata.items():
            print(f"  {key}: {value}")
    
    if conversation_obj.messages:
        print("\nMessages:")
        for message in conversation_obj.messages:
            sender_str = message.sender
            recipient_str = f" to {message.recipient}" if message.recipient else ""
            print(f"  {format_timestamp(message.timestamp)} - {sender_str}{recipient_str}:")
            print(f"    {message.content}")


# Hermes commands
async def register_with_hermes(
    prompt_engine: PromptEngine,
    communication_engine: CommunicationEngine
) -> None:
    """Register with the Hermes service registry.
    
    Args:
        prompt_engine: Prompt engine
        communication_engine: Communication engine
    """
    try:
        # Register prompt engine
        prompt_result = await prompt_engine.register_with_hermes()
        
        # Register communication engine
        comm_result = await communication_engine.register_with_hermes()
        
        if prompt_result and comm_result:
            print("Successfully registered all Rhetor services with Hermes")
        elif prompt_result:
            print("Successfully registered prompt engine with Hermes")
            print("Failed to register communication engine with Hermes")
        elif comm_result:
            print("Failed to register prompt engine with Hermes")
            print("Successfully registered communication engine with Hermes")
        else:
            print("Failed to register with Hermes service registry")
    except Exception as e:
        print(f"Error registering with Hermes: {e}")