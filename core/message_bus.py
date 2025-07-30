import asyncio
from typing import Dict, List, Callable, Optional
from collections import defaultdict
from core.agent_base import Message, MessageType
import logging

class MessageBus:
    """Event-driven message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history: List[Message] = []
        self.logger = logging.getLogger("MessageBus")
        self._running = False
        self._process_task = None
        
    async def start(self):
        """Start the message bus processing"""
        self._running = True
        self._process_task = asyncio.create_task(self._process_messages())
        self.logger.info("Message bus started")
        
    async def stop(self):
        """Stop the message bus processing"""
        self._running = False
        if self._process_task:
            await self._process_task
        self.logger.info("Message bus stopped")
        
    async def publish(self, message: Message):
        """Publish a message to the bus"""
        await self.message_queue.put(message)
        self.logger.debug(f"Published message: {message.type.value} from {message.sender}")
        
    def subscribe(self, message_type: MessageType, handler: Callable):
        """Subscribe to a specific message type"""
        self.subscribers[message_type].append(handler)
        self.logger.debug(f"Subscribed to {message_type.value}")
        
    def unsubscribe(self, message_type: MessageType, handler: Callable):
        """Unsubscribe from a message type"""
        if handler in self.subscribers[message_type]:
            self.subscribers[message_type].remove(handler)
            self.logger.debug(f"Unsubscribed from {message_type.value}")
            
    async def _process_messages(self):
        """Process messages from the queue"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                
                self.message_history.append(message)
                handlers = self.subscribers.get(message.type, [])
                
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(message)
                        else:
                            handler(message)
                    except Exception as e:
                        self.logger.error(f"Error in handler for {message.type.value}: {e}")
                        
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message: {e}")
                
    def get_message_history(self, message_type: Optional[MessageType] = None, 
                          sender: Optional[str] = None,
                          limit: int = 100) -> List[Message]:
        """Get message history with optional filters"""
        history = self.message_history
        
        if message_type:
            history = [m for m in history if m.type == message_type]
        if sender:
            history = [m for m in history if m.sender == sender]
            
        return history[-limit:]
    
    
