"""
AI Specialist Manager for managing independent AI specialist processes.

This module manages dynamic AI specialist allocation and coordination through 
Hermes message bus with Rhetor orchestration.
"""

import os
import asyncio
import subprocess
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path

# Standard logging - removed debug_log import
from .llm_client import LLMClient
from .model_router import ModelRouter

logger = logging.getLogger(__name__)

@dataclass
class AISpecialistConfig:
    """Configuration for an AI specialist."""
    specialist_id: str
    specialist_type: str  # e.g., "rhetor-orchestrator", "apollo-coordinator", "engram-memory"
    component_id: str     # e.g., "rhetor", "apollo", "engram"
    model_config: Dict[str, Any]
    personality: Dict[str, Any]
    capabilities: List[str]
    process_id: Optional[int] = None
    status: str = "inactive"  # inactive, starting, active, error, stopping

@dataclass 
class AIMessage:
    """Message structure for AI-to-AI communication."""
    message_id: str
    sender_id: str
    receiver_id: str
    content: str
    message_type: str  # chat, coordination, task_assignment, status_update
    context: Dict[str, Any]
    timestamp: float

class AISpecialistManager:
    """
    Manages dynamic AI specialist processes and coordinates communication.
    
    This class handles:
    - Dynamic AI specialist allocation 
    - Process lifecycle management
    - AI-to-AI communication through Hermes
    - Rhetor orchestration and message filtering
    """
    
    def __init__(self, llm_client: LLMClient, model_router: ModelRouter):
        """
        Initialize the AI Specialist Manager.
        
        Args:
            llm_client: LLM client for AI communication
            model_router: Model router for specialist configuration
        """
        logger.info("Initializing AISpecialistManager")
        
        self.llm_client = llm_client
        self.model_router = model_router
        self.specialists: Dict[str, AISpecialistConfig] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.active_conversations: Dict[str, List[AIMessage]] = {}
        
        # Rhetor orchestration flags
        self.rhetor_filters_enabled = True
        self.auto_translation_enabled = True
        
        # Load specialist configurations
        self._load_specialist_configs()
        
    def _load_specialist_configs(self) -> None:
        """Load AI specialist configurations from file."""
        config_file = Path(__file__).parent.parent / "config" / "ai_specialists.json"
        
        try:
            if config_file.exists():
                with open(config_file, 'r') as f:
                    configs = json.load(f)
                    logger.info(f"Loaded {len(configs)} specialist configs")
            else:
                # Create default configurations
                configs = self._create_default_specialist_configs()
                # Save default configs
                config_file.parent.mkdir(exist_ok=True)
                with open(config_file, 'w') as f:
                    json.dump(configs, f, indent=2)
                logger.info("Created default specialist configurations")
                    
        except Exception as e:
            logger.error(f"Error loading specialist configs: {e}")
            configs = self._create_default_specialist_configs()
            
        # Convert to AISpecialistConfig objects
        for config_data in configs:
            config = AISpecialistConfig(**config_data)
            self.specialists[config.specialist_id] = config
            
    def _create_default_specialist_configs(self) -> List[Dict[str, Any]]:
        """Create default AI specialist configurations."""
        return [
            {
                "specialist_id": "rhetor-orchestrator",
                "specialist_type": "meta-orchestrator",
                "component_id": "rhetor",
                "model_config": {
                    "provider": "anthropic",
                    "model": "claude-3-opus-20240229",
                    "options": {
                        "temperature": 0.3,
                        "max_tokens": 4000
                    }
                },
                "personality": {
                    "role": "Meta-AI Orchestrator",
                    "traits": ["analytical", "coordinating", "strategic"],
                    "system_prompt": "You are Rhetor, the meta-AI orchestrator for the Tekton ecosystem. You coordinate communication between AI specialists, filter and translate messages as needed, and ensure optimal task allocation.",
                    "communication_style": "clear, directive, collaborative"
                },
                "capabilities": [
                    "ai_coordination", 
                    "message_filtering", 
                    "task_allocation", 
                    "conflict_resolution",
                    "specialist_management"
                ]
            },
            {
                "specialist_id": "engram-memory",
                "specialist_type": "memory-specialist", 
                "component_id": "engram",
                "model_config": {
                    "provider": "anthropic",
                    "model": "claude-3-haiku-20240307",
                    "options": {
                        "temperature": 0.2,
                        "max_tokens": 2000
                    }
                },
                "personality": {
                    "role": "Memory and Context Specialist",
                    "traits": ["precise", "organized", "contextual"],
                    "system_prompt": "You are Engram, the memory and context specialist. You manage conversation history, context windows, and knowledge persistence across the AI ecosystem.",
                    "communication_style": "precise, factual, contextual"
                },
                "capabilities": [
                    "memory_management", 
                    "context_tracking", 
                    "knowledge_storage",
                    "history_search"
                ]
            }
        ]
        
    async def create_specialist(self, specialist_id: str) -> bool:
        """
        Create and start an AI specialist process.
        
        Args:
            specialist_id: ID of the specialist to create
            
        Returns:
            True if specialist was created successfully
        """
        if specialist_id not in self.specialists:
            logger.error(f"Unknown specialist ID: {specialist_id}")
            return False
            
        config = self.specialists[specialist_id]
        
        if config.status == "active":
            logger.info(f"Specialist {specialist_id} already active")
            return True
            
        try:
            logger.info(f"Creating specialist: {specialist_id}")
            config.status = "starting"
            
            # For now, we'll simulate process creation
            # In full implementation, this would spawn actual subprocess
            await asyncio.sleep(0.1)  # Simulate startup time
            
            config.process_id = os.getpid()  # Placeholder
            config.status = "active"
            
            logger.info(f"Specialist {specialist_id} created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating specialist {specialist_id}: {e}")
            config.status = "error"
            return False
            
    async def send_message(self, sender_id: str, receiver_id: str, content: str, 
                          message_type: str = "chat", context: Optional[Dict[str, Any]] = None) -> str:
        """
        Send a message between AI specialists via Hermes message bus.
        
        Args:
            sender_id: ID of sending specialist
            receiver_id: ID of receiving specialist  
            content: Message content
            message_type: Type of message
            context: Optional context data
            
        Returns:
            Message ID
        """
        message_id = str(uuid.uuid4())
        message = AIMessage(
            message_id=message_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            message_type=message_type,
            context=context or {},
            timestamp=asyncio.get_event_loop().time()
        )
        
        logger.info(f"Message {sender_id} -> {receiver_id}: {message_type}")
        
        # Rhetor orchestration: filter/translate message if needed
        if self.rhetor_filters_enabled and sender_id != "rhetor-orchestrator":
            message = await self._rhetor_filter_message(message)
            
        # Queue message for delivery
        await self.message_queue.put(message)
        
        # Track conversation
        conversation_key = f"{sender_id}:{receiver_id}"
        if conversation_key not in self.active_conversations:
            self.active_conversations[conversation_key] = []
        self.active_conversations[conversation_key].append(message)
        
        return message_id
        
    async def _rhetor_filter_message(self, message: AIMessage) -> AIMessage:
        """
        Rhetor orchestration: filter and potentially translate messages.
        
        Args:
            message: Original message
            
        Returns:
            Filtered/translated message
        """
        logger.debug(f"Rhetor filtering message from {message.sender_id}")
        
        # For now, pass most messages unfiltered
        # In full implementation, this would use Rhetor AI to analyze and potentially modify
        
        # Example: translate technical terms between components
        if self.auto_translation_enabled:
            if "token budget" in message.content.lower() and message.receiver_id != "apollo-coordinator":
                # Don't translate budget terms when talking to Apollo (financial coordinator)
                pass
            elif "memory context" in message.content.lower() and message.receiver_id != "engram-memory":
                # Don't translate memory terms when talking to Engram
                pass
                
        return message
        
    async def get_specialist_status(self, specialist_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an AI specialist.
        
        Args:
            specialist_id: ID of the specialist
            
        Returns:
            Status dictionary or None if not found
        """
        if specialist_id not in self.specialists:
            return None
            
        config = self.specialists[specialist_id]
        return {
            "specialist_id": specialist_id,
            "status": config.status,
            "process_id": config.process_id,
            "specialist_type": config.specialist_type,
            "component_id": config.component_id,
            "capabilities": config.capabilities
        }
        
    async def list_active_specialists(self) -> List[Dict[str, Any]]:
        """
        List all active AI specialists.
        
        Returns:
            List of active specialist status dictionaries
        """
        active = []
        for specialist_id, config in self.specialists.items():
            if config.status == "active":
                active.append(await self.get_specialist_status(specialist_id))
        return active
        
    async def start_core_specialists(self) -> Dict[str, bool]:
        """
        Start the core AI specialists (Rhetor and Engram).
        
        Returns:
            Dictionary mapping specialist IDs to success status
        """
        logger.info("Starting core AI specialists")
        
        results = {}
        core_specialists = ["rhetor-orchestrator", "engram-memory"]
        
        for specialist_id in core_specialists:
            results[specialist_id] = await self.create_specialist(specialist_id)
            
        logger.info(f"Core specialists started: {results}")
        return results
        
    async def simulate_ai_conversation(self, specialist1_id: str, specialist2_id: str, 
                                     topic: str) -> List[AIMessage]:
        """
        Simulate a conversation between two AI specialists for testing.
        
        Args:
            specialist1_id: First specialist ID
            specialist2_id: Second specialist ID  
            topic: Conversation topic
            
        Returns:
            List of messages in the conversation
        """
        logger.info(f"Simulating conversation: {specialist1_id} <-> {specialist2_id}")
        
        conversation = []
        
        # Specialist 1 initiates
        msg1_id = await self.send_message(
            sender_id=specialist1_id,
            receiver_id=specialist2_id,
            content=f"Hello {specialist2_id}, I'd like to discuss {topic}. How can we coordinate on this?",
            message_type="coordination"
        )
        
        # Specialist 2 responds
        msg2_id = await self.send_message(
            sender_id=specialist2_id,
            receiver_id=specialist1_id,
            content=f"Hello {specialist1_id}, I'm ready to work on {topic}. What specific aspects should we focus on?",
            message_type="coordination"
        )
        
        # Specialist 1 follows up
        msg3_id = await self.send_message(
            sender_id=specialist1_id,
            receiver_id=specialist2_id,
            content=f"Let's start with the core requirements for {topic} and then move to implementation details.",
            message_type="chat"
        )
        
        # Return conversation history
        conversation_key = f"{specialist1_id}:{specialist2_id}"
        return self.active_conversations.get(conversation_key, [])