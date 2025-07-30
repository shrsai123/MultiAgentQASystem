from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import uuid
from dataclasses import dataclass,field
import time
from enum import Enum
import os
import json

OUTPUT_DIR = "outputs"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

class MessageType(Enum):
    # Planning messages
    PLAN_REQUEST = "plan_request"
    PLAN_RESPONSE = "plan_response"
    PLAN_ADAPTATION_REQUEST = "plan_adaptation_request"
    PLAN_ADAPTATION_RESPONSE = "plan_adaptation_response"
    
    # Execution messages
    EXECUTION_REQUEST = "execution_request"
    EXECUTION_RESPONSE = "execution_response"
    EXECUTION_FAILED = "execution_failed"
    
    # Verification messages
    VERIFICATION_REQUEST = "verification_request"
    VERIFICATION_RESPONSE = "verification_response"
    
    # Supervision messages
    SUPERVISION_REQUEST = "supervision_request"
    SUPERVISION_RESPONSE = "supervision_response"
    
    # System messages
    SYSTEM_ERROR = "system_error"
    SYSTEM_STATUS = "system_status"
    TEST_STARTED = "test_started"
    TEST_COMPLETED = "test_completed"
    
    # Coordination messages
    AGENT_READY = "agent_ready"
    AGENT_BUSY = "agent_busy"
    AGENT_ERROR = "agent_error"

@dataclass
class Message:
    """Message structure for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.SYSTEM_STATUS
    sender: str = "system"
    recipient: Optional[str] = None  # None means broadcast
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None 
    priority: int = 1 

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, message_bus):
        self.name = name
        self.message_bus = message_bus
        self.state = "idle"
        self._setup_message_handlers()
        
    def _setup_message_handlers(self):
        """Setup message handlers - to be overridden by subclasses"""
        pass
    
    async def send_message(self, message_type: MessageType, recipient: Optional[str], 
                          payload: Dict[str, Any], correlation_id: Optional[str] = None,
                          priority: int = 1):
        """Send a message through the message bus"""
        message = Message(
            type=message_type,
            sender=self.name,
            recipient=recipient,
            payload=payload,
            correlation_id=correlation_id,
            priority=priority
        )
        await self.message_bus.publish(message)
        
    async def broadcast_status(self, status: str, details: Optional[Dict] = None):
        """Broadcast agent status"""
        payload = {
            "status": status,
            "details": details or {}
        }
        await self.send_message(MessageType.AGENT_READY if status == "ready" else MessageType.AGENT_BUSY,
                               None, payload)
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        pass
    
    def log_decision(self, decision: Dict, log_file: str = "agent_decisions.json"):
        """Log agent decisions to a file"""
        log_path = os.path.join(LOG_DIR, log_file)
        try:
            with open(log_path, "a") as f:
                f.write(json.dumps({
                    "agent": self.name,
                    "timestamp": time.time(),
                    "decision": decision
                }) + "\n")
        except Exception as e:
            self.logger.error(f"Error logging decision: {e}")